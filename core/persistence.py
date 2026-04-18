"""
Supabase-backed persistence for stock_monitor.

All database access for the project should go through :class:`Store`.

Uses the **async** Supabase client so callers in asyncio (Discord.py, briefing
jobs) can ``await`` I/O without blocking the event loop. Numeric values are sent
to PostgREST as strings so PostgreSQL ``numeric`` columns preserve precision.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from supabase import AsyncClient, create_async_client

try:
    from postgrest.exceptions import APIError as PostgrestAPIError
except ImportError:  # pragma: no cover
    PostgrestAPIError = Exception  # type: ignore[misc, assignment]

T = TypeVar("T")


class StoreError(Exception):
    """Base class for persistence errors."""


class StoreConfigurationError(StoreError):
    """Missing or invalid Supabase configuration."""


class StoreConnectionError(StoreError):
    """Failed to reach Supabase or execute a query."""


class FilingNotFoundError(StoreError):
    """No filing row exists for the given accession number."""


class SupabaseConfig(BaseSettings):
    """Load Supabase credentials from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    supabase_url: str = Field(validation_alias="SUPABASE_URL")
    supabase_service_key: str = Field(validation_alias="SUPABASE_SERVICE_KEY")


def _parse_decimal(v: Any) -> Optional[Decimal]:
    if v is None:
        return None
    if isinstance(v, Decimal):
        return v
    return Decimal(str(v))


def _parse_uuid(v: Any) -> UUID:
    if isinstance(v, UUID):
        return v
    return UUID(str(v))


class Position(BaseModel):
    id: UUID
    symbol: str
    quantity: Decimal
    cost_basis: Decimal
    opened_at: datetime
    closed_at: Optional[datetime] = None
    notes: Optional[str] = None
    target_pct: Optional[Decimal] = None
    stop_pct: Optional[Decimal] = None

    @field_validator("quantity", "cost_basis", "target_pct", "stop_pct", mode="before")
    @classmethod
    def _dec(cls, v: Any) -> Any:
        if v is None:
            return v
        return _parse_decimal(v)

    @field_validator("opened_at", "closed_at", mode="before")
    @classmethod
    def _dt(cls, v: Any) -> Any:
        if v is None or isinstance(v, datetime):
            return v
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    @field_validator("id", mode="before")
    @classmethod
    def _uid(cls, v: Any) -> Any:
        return _parse_uuid(v) if v is not None else v


class FilingRecord(BaseModel):
    """Row from ``filings`` (SEC EDGAR cache)."""

    accession_number: str
    symbol: str
    cik: str
    form_type: str
    filed_at: datetime
    period_of_report: Optional[date] = None
    url: str
    raw_text: Optional[str] = None
    summary: Optional[str] = None
    summary_model: Optional[str] = None
    summary_at: Optional[datetime] = None
    alert_sent: bool = False

    @field_validator("filed_at", "summary_at", mode="before")
    @classmethod
    def _dt(cls, v: Any) -> Any:
        if v is None or isinstance(v, datetime):
            return v
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    @field_validator("period_of_report", mode="before")
    @classmethod
    def _d(cls, v: Any) -> Any:
        if v is None or isinstance(v, date):
            return v
        if isinstance(v, str):
            return date.fromisoformat(v[:10])
        return v


class MacroObservation(BaseModel):
    series_id: str
    date: date
    value: Optional[Decimal] = None
    fetched_at: Optional[datetime] = None

    @field_validator("value", mode="before")
    @classmethod
    def _dec(cls, v: Any) -> Any:
        if v is None:
            return None
        return _parse_decimal(v)

    @field_validator("date", mode="before")
    @classmethod
    def _d(cls, v: Any) -> Any:
        if v is None or isinstance(v, date):
            return v
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, str):
            return date.fromisoformat(v[:10])
        return v

    @field_validator("fetched_at", mode="before")
    @classmethod
    def _dt(cls, v: Any) -> Any:
        if v is None or isinstance(v, datetime):
            return v
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


class WatchlistMeta(BaseModel):
    symbol: str
    cik: Optional[str] = None
    company_name: Optional[str] = None
    bucket: Optional[str] = None
    alert_threshold_pct: Optional[Decimal] = None
    updated_at: Optional[datetime] = None

    @field_validator("alert_threshold_pct", mode="before")
    @classmethod
    def _atp(cls, v: Any) -> Any:
        if v is None:
            return v
        return _parse_decimal(v)

    @field_validator("updated_at", mode="before")
    @classmethod
    def _dt(cls, v: Any) -> Any:
        if v is None or isinstance(v, datetime):
            return v
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


class Store:
    """
    Async Supabase wrapper for positions, filings, macro series, alerts, and watchlist metadata.

    Construct with ``await Store.create(...)`` or ``await Store.from_env()``. For tests,
    pass ``client=`` with an ``AsyncClient`` or compatible async mock.
    """

    def __init__(self, *, client: AsyncClient) -> None:
        self._client = client

    @classmethod
    async def create(
        cls,
        supabase_url: Optional[str] = None,
        supabase_service_key: Optional[str] = None,
        *,
        client: Optional[AsyncClient] = None,
    ) -> "Store":
        """Build a :class:`Store`, creating an async Supabase client from env/args when needed."""
        if client is not None:
            return cls(client=client)

        url = supabase_url or os.environ.get("SUPABASE_URL")
        key = supabase_service_key or os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not str(url).strip():
            raise StoreConfigurationError(
                "SUPABASE_URL is missing. Set it in the environment or pass supabase_url=."
            )
        if not key or not str(key).strip():
            raise StoreConfigurationError(
                "SUPABASE_SERVICE_KEY is missing. Set it in the environment or pass "
                "supabase_service_key=."
            )

        try:
            ac = await create_async_client(str(url).strip(), str(key).strip())
        except Exception as exc:
            raise StoreConnectionError(
                f"Could not create Supabase client: {exc}"
            ) from exc
        return cls(client=ac)

    @classmethod
    async def from_env(cls) -> "Store":
        """Build using :class:`SupabaseConfig` (loads ``.env`` if present)."""
        try:
            cfg = SupabaseConfig()
        except ValidationError as exc:
            raise StoreConfigurationError(
                "Invalid or missing SUPABASE_URL / SUPABASE_SERVICE_KEY in environment."
            ) from exc
        return await cls.create(cfg.supabase_url, cfg.supabase_service_key)

    async def _run(
        self,
        description: str,
        fn: Callable[[], Awaitable[T]],
    ) -> T:
        try:
            return await fn()
        except StoreError:
            raise
        except PostgrestAPIError as exc:
            raise StoreConnectionError(f"{description}: {exc}") from exc
        except Exception as exc:
            raise StoreConnectionError(f"{description}: {exc}") from exc

    async def get_positions(self, *, include_closed: bool = False) -> list[Position]:
        """Return open positions by default (``closed_at`` IS NULL)."""

        async def _q() -> Any:
            q = self._client.table("positions").select("*")
            if not include_closed:
                q = q.is_("closed_at", "null")
            return await q.execute()

        resp = await self._run("get_positions", _q)
        rows = getattr(resp, "data", None) or []
        return [Position.model_validate(r) for r in rows]

    async def add_position(
        self,
        symbol: str,
        quantity: Decimal | float | str,
        cost_basis: Decimal | float | str,
        opened_at: datetime,
        notes: Optional[str] = None,
        *,
        target_pct: Optional[Decimal | float | str] = None,
        stop_pct: Optional[Decimal | float | str] = None,
    ) -> Position:
        """Insert a new position row; ``opened_at`` should be timezone-aware (stored as UTC).

        ``cost_basis`` is **per share** (average cost). Optional ``target_pct`` / ``stop_pct``
        are stored as positive numbers meaning percent move vs entry (Phase 6 alerts).
        """
        sym = symbol.strip().upper()
        qty = _parse_decimal(quantity)
        cost = _parse_decimal(cost_basis)
        if qty is None:
            raise StoreError("quantity is required")
        if cost is None:
            raise StoreError("cost_basis is required")

        opened = opened_at
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)

        # Strings preserve numeric precision over the wire (avoid float → PostgreSQL numeric).
        payload: dict[str, Any] = {
            "symbol": sym,
            "quantity": str(qty),
            "cost_basis": str(cost),
            "opened_at": opened.astimezone(timezone.utc).isoformat(),
        }
        if notes is not None:
            payload["notes"] = notes
        tp = _parse_decimal(target_pct) if target_pct is not None else None
        sp = _parse_decimal(stop_pct) if stop_pct is not None else None
        if tp is not None:
            payload["target_pct"] = str(tp)
        if sp is not None:
            payload["stop_pct"] = str(sp)

        async def _ins() -> Any:
            return await self._client.table("positions").insert(payload).execute()

        resp = await self._run("add_position", _ins)
        rows = getattr(resp, "data", None) or []
        if not rows:
            raise StoreConnectionError("add_position: empty response")
        return Position.model_validate(rows[0])

    async def close_open_positions_for_symbol(self, symbol: str) -> list[Position]:
        """Set ``closed_at`` for all rows with ``symbol`` and ``closed_at`` IS NULL."""
        sym = symbol.strip().upper()
        now = datetime.now(timezone.utc)

        async def _up() -> Any:
            return (
                await self._client.table("positions")
                .update({"closed_at": now.isoformat()})
                .eq("symbol", sym)
                .is_("closed_at", "null")
                .execute()
            )

        resp = await self._run("close_open_positions_for_symbol", _up)
        rows = getattr(resp, "data", None) or []
        return [Position.model_validate(r) for r in rows]

    async def close_position(self, position_id: UUID) -> Position:
        """Set ``closed_at`` to now (UTC) for the given position id."""
        now = datetime.now(timezone.utc)

        async def _up() -> Any:
            return (
                await self._client.table("positions")
                .update({"closed_at": now.isoformat()})
                .eq("id", str(position_id))
                .execute()
            )

        resp = await self._run("close_position", _up)
        rows = getattr(resp, "data", None) or []
        if not rows:
            raise StoreConnectionError(
                f"close_position: no row updated for id={position_id}"
            )
        return Position.model_validate(rows[0])

    async def get_filing(self, accession: str) -> Optional[FilingRecord]:
        """Return a full filing row, if present."""

        async def _sel() -> Any:
            return (
                await self._client.table("filings")
                .select("*")
                .eq("accession_number", accession.strip())
                .maybe_single()
                .execute()
            )

        resp = await self._run("get_filing", _sel)
        if resp is None:
            return None
        row = getattr(resp, "data", None)
        if not row or not isinstance(row, dict):
            return None
        return FilingRecord.model_validate(row)

    async def list_filings_for_symbol(self, symbol: str, limit: int = 20) -> list[FilingRecord]:
        """Most recent filings for a ticker (by ``filed_at`` desc)."""
        sym = symbol.strip().upper()
        lim = max(1, min(limit, 100))

        async def _sel() -> Any:
            return (
                await self._client.table("filings")
                .select("*")
                .eq("symbol", sym)
                .order("filed_at", desc=True)
                .limit(lim)
                .execute()
            )

        resp = await self._run("list_filings_for_symbol", _sel)
        rows = getattr(resp, "data", None) or []
        return [FilingRecord.model_validate(r) for r in rows]

    async def upsert_filing_row(
        self,
        *,
        accession_number: str,
        symbol: str,
        cik: str,
        form_type: str,
        filed_at: datetime,
        url: str,
        period_of_report: Optional[date] = None,
        raw_text: Optional[str] = None,
    ) -> None:
        """Insert or replace filing metadata (primary key ``accession_number``).

        Preserves an existing ``summary`` / ``summary_*`` row so re-syncs do not wipe LLM cache.
        """
        acc = accession_number.strip()
        sym = symbol.strip().upper()
        filed = filed_at if filed_at.tzinfo else filed_at.replace(tzinfo=timezone.utc)
        payload: dict[str, Any] = {
            "accession_number": acc,
            "symbol": sym,
            "cik": cik.strip(),
            "form_type": form_type.strip(),
            "filed_at": filed.astimezone(timezone.utc).isoformat(),
            "url": url,
        }
        if period_of_report is not None:
            payload["period_of_report"] = period_of_report.isoformat()
        if raw_text is not None:
            payload["raw_text"] = raw_text

        ex = await self.get_filing(acc)
        if ex and ex.summary is not None:
            payload["summary"] = ex.summary
            payload["summary_model"] = ex.summary_model
            if ex.summary_at is not None:
                payload["summary_at"] = ex.summary_at.astimezone(timezone.utc).isoformat()

        async def _up() -> Any:
            return (
                await self._client.table("filings")
                .upsert(payload, on_conflict="accession_number")
                .execute()
            )

        await self._run("upsert_filing_row", _up)

    async def set_filing_alert_sent(self, accession: str, *, alert_sent: bool = True) -> None:
        """Update ``filings.alert_sent`` for event dedup."""

        acc = accession.strip()

        async def _up() -> Any:
            return (
                await self._client.table("filings")
                .update({"alert_sent": alert_sent})
                .eq("accession_number", acc)
                .execute()
            )

        await self._run("set_filing_alert_sent", _up)

    async def list_filings_pending_event_alert(
        self,
        *,
        form_types: list[str],
        symbols: Optional[set[str]] = None,
        limit: int = 50,
    ) -> list[FilingRecord]:
        """Filings with ``alert_sent=false`` and optional symbol filter."""
        fts = [f.strip() for f in form_types if f.strip()]
        lim = max(1, min(limit, 200))

        async def _sel() -> Any:
            q = (
                self._client.table("filings")
                .select("*")
                .eq("alert_sent", False)
                .in_("form_type", fts)
            )
            if symbols:
                q = q.in_("symbol", [s.strip().upper() for s in symbols])
            return await q.order("filed_at", desc=True).limit(lim).execute()

        resp = await self._run("list_filings_pending_event_alert", _sel)
        rows = getattr(resp, "data", None) or []
        return [FilingRecord.model_validate(r) for r in rows]

    async def get_filing_summary(self, accession: str) -> Optional[str]:
        """Return cached LLM summary text, if any."""
        row = await self.get_filing(accession)
        if row is None:
            return None
        return row.summary

    async def cache_filing_summary(
        self,
        accession: str,
        summary: str,
        summary_model: Optional[str] = None,
    ) -> None:
        """Persist summary fields on an existing filing row."""
        now = datetime.now(timezone.utc).isoformat()
        acc = accession.strip()

        async def _up() -> Any:
            return (
                await self._client.table("filings")
                .update(
                    {
                        "summary": summary,
                        "summary_model": summary_model,
                        "summary_at": now,
                    }
                )
                .eq("accession_number", acc)
                .execute()
            )

        resp = await self._run("cache_filing_summary", _up)
        rows = getattr(resp, "data", None) or []
        if len(rows) == 0:
            raise FilingNotFoundError(
                f"No filing with accession_number={acc!r}; cannot cache summary."
            )

    async def upsert_macro(
        self,
        series_id: str,
        observation_date: date,
        value: Optional[Decimal | float | str],
    ) -> None:
        """Insert or update one observation for a FRED series."""
        sid = series_id.strip()
        d = observation_date.isoformat()
        val = _parse_decimal(value) if value is not None else None
        payload: dict[str, Any] = {
            "series_id": sid,
            "date": d,
            "value": str(val) if val is not None else None,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

        async def _up() -> Any:
            return (
                await self._client.table("macro_series")
                .upsert(payload, on_conflict="series_id,date")
                .execute()
            )

        await self._run("upsert_macro", _up)

    async def get_macro_latest(self, series_id: str, n: int = 10) -> list[MacroObservation]:
        """Return up to ``n`` most recent observations (by calendar date)."""
        if n < 1:
            return []
        sid = series_id.strip()

        async def _sel() -> Any:
            return (
                await self._client.table("macro_series")
                .select("*")
                .eq("series_id", sid)
                .order("date", desc=True)
                .limit(n)
                .execute()
            )

        resp = await self._run("get_macro_latest", _sel)
        rows = getattr(resp, "data", None) or []
        chronological = list(reversed(rows))
        return [MacroObservation.model_validate(r) for r in chronological]

    async def has_alert_been_sent(self, alert_type: str, reference_id: str) -> bool:
        async def _sel() -> Any:
            return (
                await self._client.table("alerts_sent")
                .select("id")
                .eq("alert_type", alert_type)
                .eq("reference_id", reference_id)
                .limit(1)
                .execute()
            )

        resp = await self._run("has_alert_been_sent", _sel)
        rows = getattr(resp, "data", None) or []
        return len(rows) > 0

    async def mark_alert_sent(
        self,
        alert_type: str,
        reference_id: str,
        channel_id: Optional[str] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        row: dict[str, Any] = {
            "alert_type": alert_type,
            "reference_id": reference_id,
        }
        if channel_id is not None:
            row["channel_id"] = channel_id
        if payload is not None:
            row["payload"] = payload

        async def _ins() -> Any:
            return await self._client.table("alerts_sent").insert(row).execute()

        await self._run("mark_alert_sent", _ins)

    async def list_alerts_sent_since(
        self,
        *,
        hours: int = 24,
        limit: int = 100,
        alert_type_prefix: Optional[str] = "event_",
    ) -> list[dict[str, Any]]:
        """Recent rows from ``alerts_sent`` for status / debugging (newest first)."""
        h = max(1, min(int(hours), 24 * 365))
        lim = max(1, min(int(limit), 500))
        cutoff = datetime.now(timezone.utc) - timedelta(hours=h)

        async def _sel() -> Any:
            q = (
                self._client.table("alerts_sent")
                .select("alert_type,reference_id,sent_at,channel_id,payload")
                .gte("sent_at", cutoff.isoformat())
            )
            if alert_type_prefix:
                q = q.like("alert_type", f"{alert_type_prefix}%")
            return await q.order("sent_at", desc=True).limit(lim).execute()

        resp = await self._run("list_alerts_sent_since", _sel)
        rows = getattr(resp, "data", None) or []
        return [r for r in rows if isinstance(r, dict)]

    async def ping_database(self) -> None:
        """Cheap connectivity check (health probe)."""

        async def _q() -> Any:
            return await self._client.table("alerts_sent").select("id").limit(1).execute()

        await self._run("ping_database", _q)

    async def latest_morning_briefing_sent_at(self) -> Optional[datetime]:
        """Most recent ``sent_at`` for ``morning_briefing`` alerts, if any."""

        async def _q() -> Any:
            return (
                await self._client.table("alerts_sent")
                .select("sent_at")
                .eq("alert_type", "morning_briefing")
                .order("sent_at", desc=True)
                .limit(1)
                .execute()
            )

        resp = await self._run("latest_morning_briefing_sent_at", _q)
        rows = getattr(resp, "data", None) or []
        if not rows or not isinstance(rows[0], dict):
            return None
        raw = rows[0].get("sent_at")
        if raw is None:
            return None
        if isinstance(raw, datetime):
            return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        if isinstance(raw, str):
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return None

    async def get_alerts_sent_payload(
        self, alert_type: str, reference_id: str
    ) -> Optional[dict[str, Any]]:
        """Return ``payload`` jsonb if a row exists for (alert_type, reference_id)."""

        async def _sel() -> Any:
            return (
                await self._client.table("alerts_sent")
                .select("payload")
                .eq("alert_type", alert_type)
                .eq("reference_id", reference_id)
                .maybe_single()
                .execute()
            )

        resp = await self._run("get_alerts_sent_payload", _sel)
        if resp is None:
            return None
        row = getattr(resp, "data", None)
        if not row or not isinstance(row, dict):
            return None
        p = row.get("payload")
        return p if isinstance(p, dict) else None

    async def record_llm_usage(
        self,
        *,
        usage_date: date,
        operation: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Append a row to ``llm_usage`` for daily budget accounting."""

        async def _ins() -> Any:
            return (
                await self._client.table("llm_usage")
                .insert(
                    {
                        "usage_date": usage_date.isoformat(),
                        "operation": operation[:200],
                        "input_tokens": max(0, int(input_tokens)),
                        "output_tokens": max(0, int(output_tokens)),
                    }
                )
                .execute()
            )

        await self._run("record_llm_usage", _ins)

    async def sum_llm_tokens_for_date(self, usage_date: date) -> int:
        """Sum input+output tokens recorded for ``usage_date``."""

        async def _sel() -> Any:
            return (
                await self._client.table("llm_usage")
                .select("input_tokens,output_tokens")
                .eq("usage_date", usage_date.isoformat())
                .execute()
            )

        resp = await self._run("sum_llm_tokens_for_date", _sel)
        rows = getattr(resp, "data", None) or []
        total = 0
        for r in rows:
            if isinstance(r, dict):
                total += int(r.get("input_tokens") or 0) + int(r.get("output_tokens") or 0)
        return total

    async def get_watchlist_meta(self, symbol: str) -> Optional[WatchlistMeta]:
        sym = symbol.strip().upper()

        async def _sel() -> Any:
            return (
                await self._client.table("watchlist_meta")
                .select("*")
                .eq("symbol", sym)
                .maybe_single()
                .execute()
            )

        resp = await self._run("get_watchlist_meta", _sel)
        if resp is None:
            return None
        row = getattr(resp, "data", None)
        if not row or not isinstance(row, dict):
            return None
        return WatchlistMeta.model_validate(row)

    async def set_watchlist_meta(
        self,
        symbol: str,
        *,
        cik: Optional[str] = None,
        company_name: Optional[str] = None,
        bucket: Optional[str] = None,
        alert_threshold_pct: Optional[Decimal | float | str] = None,
    ) -> None:
        """Upsert watchlist metadata; unspecified fields keep existing values when possible."""
        sym = symbol.strip().upper()
        existing = await self.get_watchlist_meta(sym)
        if alert_threshold_pct is not None:
            atp_merged = _parse_decimal(alert_threshold_pct)
        else:
            atp_merged = existing.alert_threshold_pct if existing else None
        row: dict[str, Any] = {
            "symbol": sym,
            "cik": cik if cik is not None else (existing.cik if existing else None),
            "company_name": company_name
            if company_name is not None
            else (existing.company_name if existing else None),
            "bucket": bucket if bucket is not None else (existing.bucket if existing else None),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if atp_merged is not None:
            row["alert_threshold_pct"] = str(atp_merged)

        async def _up() -> Any:
            return (
                await self._client.table("watchlist_meta")
                .upsert(row, on_conflict="symbol")
                .execute()
            )

        await self._run("set_watchlist_meta", _up)
