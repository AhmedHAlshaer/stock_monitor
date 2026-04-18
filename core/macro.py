"""
FRED macro data: async client, series fetch, and DB-backed change summaries.

Rate limits: FRED allows generous usage; we use a semaphore + pacing to avoid 429s.
"""

from __future__ import annotations

import asyncio
import os
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field, field_validator
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from core.persistence import MacroObservation, Store


FRED_API_ROOT = "https://api.stlouisfed.org/fred"


class SeriesKind(str, Enum):
    """How to format levels and moves for Discord / change strings."""

    YIELD_PCT = "yield_pct"  # DGS10, DGS2, DFF — values are %, 1pp = 100bp
    VIX = "vix"
    DXY = "dxy"
    CPI_INDEX = "cpi_index"
    UNRATE_PCT = "unrate_pct"
    PAYEMS_LEVEL = "payems_thousands"


# Exact series IDs from ENHANCEMENT_GUIDE §6; order preserved for display.
MACRO_SERIES_ORDER: tuple[str, ...] = (
    "DGS10",
    "DGS2",
    "DFF",
    "VIXCLS",
    "DTWEXBGS",
    "CPIAUCSL",
    "UNRATE",
    "PAYEMS",
)

SERIES_CONFIG: dict[str, tuple[str, SeriesKind, bool]] = {
    # series_id -> (display name, kind, is_monthly)
    "DGS10": ("10Y Treasury", SeriesKind.YIELD_PCT, False),
    "DGS2": ("2Y Treasury", SeriesKind.YIELD_PCT, False),
    "DFF": ("Fed funds (effective)", SeriesKind.YIELD_PCT, False),
    "VIXCLS": ("VIX", SeriesKind.VIX, False),
    "DTWEXBGS": ("Trade-weighted USD index", SeriesKind.DXY, False),
    "CPIAUCSL": ("CPI (all urban, SA)", SeriesKind.CPI_INDEX, True),
    "UNRATE": ("Unemployment rate", SeriesKind.UNRATE_PCT, True),
    "PAYEMS": ("Nonfarm payrolls", SeriesKind.PAYEMS_LEVEL, True),
}


# --- Pydantic: FRED API (series/observations) ---


class FredObservationApiRow(BaseModel):
    """Single observation row from FRED JSON."""

    realtime_start: str = ""
    realtime_end: str = ""
    date: str
    value: str

    @field_validator("value")
    @classmethod
    def strip_val(cls, v: str) -> str:
        return v.strip()


class FredObservationsResponse(BaseModel):
    """`fred/series/observations` response body."""

    realtime_start: str = ""
    realtime_end: str = ""
    observations: list[FredObservationApiRow] = Field(default_factory=list)


class FredSeriesInfoResponse(BaseModel):
    """Optional `fred/series` metadata (for validation)."""

    seriess: list[dict[str, Any]] = Field(default_factory=list)


class ParsedObservation(BaseModel):
    """One observation after parsing FRED strings."""

    observation_date: date
    value: Decimal


def _decimal_from_fred_value(raw: str) -> Optional[Decimal]:
    s = raw.strip()
    if not s or s == ".":
        return None
    return Decimal(s)


def _retryable_httpx(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


class FREDClient:
    """
    Async FRED API client using httpx + tenacity.

    Uses a semaphore and short sleeps between requests to stay under rate limits.
    Does not perform DB I/O.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        timeout: float = 60.0,
        max_concurrent: int = 2,
        min_interval_sec: float = 0.2,
    ) -> None:
        key = api_key or os.environ.get("FRED_API_KEY")
        if not key or not str(key).strip():
            raise ValueError("FRED_API_KEY is missing or empty")
        self._api_key = str(key).strip()
        self._sem = asyncio.Semaphore(max_concurrent)
        self._min_interval = min_interval_sec
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": "stock_monitor/1.0 (macro; +https://github.com/AhmedHAlshaer/stock_monitor)"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=1, max=45),
        retry=retry_if_exception(_retryable_httpx),
        reraise=True,
    )
    async def _get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        # Serial pacing under semaphore + sleep reduces burst traffic (avoids 429).
        async with self._sem:
            await asyncio.sleep(self._min_interval)
            url = f"{FRED_API_ROOT}/{path.lstrip('/')}"
            r = await self._client.get(url, params=params)
            r.raise_for_status()
            return r.json()

    async def fetch_observations(
        self,
        series_id: str,
        *,
        observation_start: Optional[date] = None,
        observation_end: Optional[date] = None,
        sort_order: str = "asc",
    ) -> list[ParsedObservation]:
        """
        Fetch observations for one series; validate with Pydantic; skip missing values.
        """
        params: dict[str, Any] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": sort_order,
        }
        if observation_start is not None:
            params["observation_start"] = observation_start.isoformat()
        if observation_end is not None:
            params["observation_end"] = observation_end.isoformat()

        raw = await self._get_json("series/observations", params)
        parsed = FredObservationsResponse.model_validate(raw)
        out: list[ParsedObservation] = []
        for row in parsed.observations:
            dv = _decimal_from_fred_value(row.value)
            if dv is None:
                continue
            try:
                od = date.fromisoformat(row.date[:10])
            except ValueError:
                continue
            out.append(ParsedObservation(observation_date=od, value=dv))
        return out

    async def fetch_series_metadata(self, series_id: str) -> FredSeriesInfoResponse:
        """Validate series exists (optional sanity check)."""
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
        }
        raw = await self._get_json("series", params)
        return FredSeriesInfoResponse.model_validate(raw)


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _direction_and_streak(vals: list[Decimal]) -> tuple[str, int]:
    """vals chronological; returns (direction, streak) where direction is up|down|flat.

    *Streak* counts how many consecutive periods match the latest move (e.g. three
    down moves → streak 3, i.e. "3rd day of decline"). Does not double-count the
    last diff: the loop counts every matching segment including the most recent one.
    """
    if len(vals) < 2:
        return ("flat", 1)
    last_diff = vals[-1] - vals[-2]
    if last_diff == 0:
        return ("flat", 1)
    direction = "up" if last_diff > 0 else "down"
    streak = 0
    for i in range(len(vals) - 1, 0, -1):
        d = vals[i] - vals[i - 1]
        if (direction == "up" and d > 0) or (direction == "down" and d < 0):
            streak += 1
        else:
            break
    return (direction, streak)


def _streak_phrase(direction: str, streak: int, period_word: str) -> str:
    if direction == "flat":
        return "unchanged vs prior"
    move = "decline" if direction == "down" else "rise"
    return f"{_ordinal(streak)} {period_word} of {move}"


def _format_change_str(
    _series_id: str,
    series_name: str,
    kind: SeriesKind,
    is_monthly: bool,
    latest: Decimal,
    prev: Optional[Decimal],
    change: Decimal,
    direction: str,
    streak: int,
) -> str:
    period_word = "month" if is_monthly else "day"
    if prev is None:
        return f"{series_name}: {latest} (first observation in window)"

    if kind == SeriesKind.YIELD_PCT:
        # Moves under 1 percentage point are shown in basis points (1pp = 100bp).
        if abs(change) < Decimal("1"):
            bp = change * Decimal("100")
            move = f"{float(bp):+.0f}bp"
        else:
            move = f"{float(change):+.2f}pp"
        sp = _streak_phrase(direction, streak, period_word)
        return f"{move} to {float(latest):.2f}%, {sp}"

    if kind == SeriesKind.VIX:
        sp = _streak_phrase(direction, streak, period_word)
        return f"{float(change):+.2f} to {float(latest):.1f}, {sp}"

    if kind == SeriesKind.DXY:
        sp = _streak_phrase(direction, streak, period_word)
        return f"{float(change):+.3f} on index to {float(latest):.3f}, {sp}"

    if kind == SeriesKind.CPI_INDEX:
        mom = (change / prev * Decimal("100")) if prev != 0 else Decimal("0")
        sp = _streak_phrase(direction, streak, period_word)
        return f"index {float(latest):.3f}, MoM {float(mom):+.2f}% ({sp})"

    if kind == SeriesKind.UNRATE_PCT:
        sp = _streak_phrase(direction, streak, period_word)
        return f"{float(change):+.2f}pp to {float(latest):.2f}%, {sp}"

    if kind == SeriesKind.PAYEMS_LEVEL:
        sp = _streak_phrase(direction, streak, period_word)
        # FRED: thousands of persons; show level in millions.
        lvl_m = float(latest) / 1000.0
        ch_k = float(change)
        return f"{ch_k:+.1f}k vs prior month, level {lvl_m:.2f}M jobs, {sp}"

    return f"{series_name}: {latest}"


class MacroAnalyzer:
    """
    Reads macro_series via :class:`Store` and summarizes recent moves.

    Does not call FRED; use :class:`FREDClient` + job to populate the database first.
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    async def get_changes(self, as_of_date: date) -> list[dict[str, Any]]:
        """
        For each configured series, return a dict:
        series_id, series_name, latest_value (float | None), change (float | None),
        change_str (str). ``change`` is ``None`` when only one observation exists in
        the window (no prior period to compare).

        Uses observations on or before ``as_of_date`` only.
        """
        results: list[dict[str, Any]] = []
        for series_id in MACRO_SERIES_ORDER:
            if series_id not in SERIES_CONFIG:
                continue
            name, kind, is_monthly = SERIES_CONFIG[series_id]
            obs: list[MacroObservation] = await self._store.get_macro_latest(
                series_id, n=120
            )
            filtered = [o for o in obs if o.date <= as_of_date and o.value is not None]
            if not filtered:
                results.append(
                    {
                        "series_id": series_id,
                        "series_name": name,
                        "latest_value": None,
                        "change": None,
                        "change_str": f"{name}: no data in database (run jobs.update_macro)",
                    }
                )
                continue

            vals_dec = [o.value for o in filtered]
            obs_dates = [o.date for o in filtered]

            latest_dec = vals_dec[-1]
            prev_dec = vals_dec[-2] if len(vals_dec) >= 2 else None
            ch: Optional[Decimal] = (
                (latest_dec - prev_dec) if prev_dec is not None else None
            )
            direction, streak = _direction_and_streak(vals_dec)

            change_str = _format_change_str(
                series_id,
                name,
                kind,
                is_monthly,
                latest_dec,
                prev_dec,
                ch if ch is not None else Decimal("0"),
                direction,
                streak,
            )

            results.append(
                {
                    "series_id": series_id,
                    "series_name": name,
                    "latest_value": float(latest_dec),
                    "change": float(ch) if ch is not None else None,
                    "change_str": change_str,
                    "observation_date": obs_dates[-1].isoformat(),
                }
            )
        return results


async def fetch_all_series_and_upsert(
    store: Store,
    client: FREDClient,
    *,
    lookback_days: int = 1100,
) -> dict[str, int]:
    """
    Fetch all macro series from FRED and upsert into ``macro_series``.

    Returns counts of observations written per series_id.
    """
    end_d = datetime.now(timezone.utc).date()
    start_d = end_d - timedelta(days=lookback_days)
    counts: dict[str, int] = {}
    for series_id in MACRO_SERIES_ORDER:
        rows = await client.fetch_observations(
            series_id, observation_start=start_d, observation_end=end_d
        )
        n = 0
        for row in rows:
            await store.upsert_macro(series_id, row.observation_date, row.value)
            n += 1
        counts[series_id] = n
    return counts
