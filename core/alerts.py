"""
Event-driven alerts (Phase 6): SEC filings, price moves, macro releases, position targets.

Schedules flows by **NYSE calendar** and **Eastern Time** session. Dedup via ``alerts_sent``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Optional
from zoneinfo import ZoneInfo

import httpx
import pandas_market_calendars as mcal
import yfinance as yf

from core.persistence import Store
from core.sec_filings import parse_form4_xml
from core.stock_data import StockDataFetcher
from core.synthesis import filing_excerpt_summarize

log = logging.getLogger(__name__)

EASTERN_TZ = ZoneInfo("America/New_York")

# NYSE calendar (cached factory)
_nyse_calendar: Any = None


def _get_nyse() -> Any:
    global _nyse_calendar
    if _nyse_calendar is None:
        _nyse_calendar = mcal.get_calendar("NYSE")
    return _nyse_calendar


def nyse_is_trading_day(d: date) -> bool:
    """True if NYSE has a regular session on this **calendar** date (weekday holiday → False)."""
    cal = _get_nyse()
    sessions = cal.valid_days(start_date=d.isoformat(), end_date=d.isoformat())
    return len(sessions) > 0


def is_weekend(d: date) -> bool:
    return d.weekday() >= 5


DEFAULT_THRESHOLDS: dict[str, Decimal] = {
    "AAPL": Decimal("3.0"),
    "META": Decimal("3.0"),
    "NVDA": Decimal("3.0"),
    "AMZN": Decimal("3.0"),
    "GLD": Decimal("3.0"),
    "QQQ": Decimal("3.0"),
    "AMD": Decimal("4.0"),
    "WULF": Decimal("6.0"),
    "IONQ": Decimal("8.0"),
    "RGTI": Decimal("8.0"),
    "QBTS": Decimal("8.0"),
}

DEFAULT_UNKNOWN = Decimal("5.0")

ALERT_TYPE_8K = "event_8k"
ALERT_TYPE_FORM4 = "event_form4"
ALERT_TYPE_PREMARKET = "event_premarket"
ALERT_TYPE_INTRADAY = "event_intraday"
ALERT_TYPE_TARGET = "event_position_target"
ALERT_TYPE_STOP = "event_position_stop"
ALERT_TYPE_MACRO = "event_macro_release"
# --- EventAlert (Discord + REST) ---


@dataclass
class EventAlert:
    """Compact “check this now” alert for Discord (colors come from :mod:`bot.embeds.alerts`)."""

    alert_type: str  # "8k", "form4", "premarket", "intraday", "target", "stop", "macro_release", "batch"
    symbol: Optional[str]
    title: str
    body: str
    url: Optional[str] = None
    intraday_direction: Optional[Literal["up", "down"]] = None  # embed coloring only


@dataclass
class PendingDelivery:
    alert: EventAlert
    reference_id: str
    db_alert_type: str
    accession: Optional[str] = None
    payload: Optional[dict[str, Any]] = None


async def _post_event_alert_rest(
    alert: EventAlert,
    *,
    channel_id: int,
    bot_token: str,
) -> None:
    """POST via Discord REST using the same embed shape as :func:`bot.embeds.alerts.build_event_alert_embed`."""
    from bot.embeds.alerts import event_alert_to_payload_dict

    emb = event_alert_to_payload_dict(alert)
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    headers = {"Authorization": f"Bot {bot_token.strip()}"}
    async with httpx.AsyncClient(timeout=45.0) as client:
        r = await client.post(url, headers=headers, json={"embeds": [emb]})
        r.raise_for_status()


async def post_discord_event_embed(
    *,
    bot_token: str,
    channel_id: int,
    embed: dict[str, Any],
) -> None:
    """Low-level REST post (legacy dict embeds)."""
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    headers = {"Authorization": f"Bot {bot_token.strip()}"}
    async with httpx.AsyncClient(timeout=45.0) as client:
        r = await client.post(url, headers=headers, json={"embeds": [embed]})
        r.raise_for_status()


async def _mark_sent(
    store: Store,
    alert_type: str,
    reference_id: str,
    *,
    channel_id: Optional[int] = None,
    payload: Optional[dict[str, Any]] = None,
) -> None:
    await store.mark_alert_sent(
        alert_type,
        reference_id,
        channel_id=str(channel_id) if channel_id is not None else None,
        payload=payload,
    )


def default_threshold_for_symbol(symbol: str) -> Decimal:
    return DEFAULT_THRESHOLDS.get(symbol.strip().upper(), DEFAULT_UNKNOWN)


async def resolve_alert_threshold_pct(symbol: str, store: Optional[Store]) -> Decimal:
    sym = symbol.strip().upper()
    if store is not None:
        meta = await store.get_watchlist_meta(sym)
        if meta and meta.alert_threshold_pct is not None:
            return meta.alert_threshold_pct
    return default_threshold_for_symbol(sym)


async def ensure_default_watchlist_threshold(store: Store, symbol: str) -> None:
    sym = symbol.strip().upper()
    existing = await store.get_watchlist_meta(sym)
    if existing and existing.alert_threshold_pct is not None:
        return
    await store.set_watchlist_meta(sym, alert_threshold_pct=default_threshold_for_symbol(sym))


def _xml_local(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def form4_max_transaction_usd(xml_text: str) -> Optional[Decimal]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    max_usd: Optional[Decimal] = None

    for el in root.iter():
        ln = _xml_local(el.tag).lower()
        if ln != "nonderivativetransaction" and ln != "derivativetransaction":
            continue
        shares = Decimal("0")
        price = Decimal("0")
        value_direct: Optional[Decimal] = None
        for ch in el.iter():
            t = _xml_local(ch.tag).lower()
            if t == "transactionshares" and ch.text:
                try:
                    shares = max(shares, Decimal(ch.text.strip().replace(",", "")))
                except Exception:
                    pass
            if t == "transactionpricepershare" and ch.text:
                try:
                    price = Decimal(ch.text.strip().replace(",", ""))
                except Exception:
                    pass
            if t == "transactionvalue" and ch.text:
                try:
                    value_direct = Decimal(ch.text.strip().replace(",", ""))
                except Exception:
                    pass
        cand: Optional[Decimal] = None
        if value_direct is not None and value_direct > 0:
            cand = value_direct
        elif shares > 0 and price > 0:
            cand = shares * price
        if cand is not None:
            max_usd = cand if max_usd is None else max(max_usd, cand)

    return max_usd


def _prompt_path(name: str) -> Path:
    return Path(__file__).resolve().parent / "prompts" / name


# --- Session → flows (ET) ---

FLOW_PREMARKET = "premarket_gaps"
FLOW_INTRADAY = "intraday_moves"
FLOW_TARGET_STOP = "target_stop"
FLOW_8K = "eight_k"
FLOW_FORM4 = "form4_large"
FLOW_MACRO = "macro_release"


def compute_active_flows(now_et: datetime) -> set[str]:
    """
    Which alert pipelines may run at this instant (NYSE calendar + clock).

    * **Weekend** (Sat/Sun): **macro_release** only (FRED may publish).
    * **NYSE weekday holiday** (closed): empty — caller logs *market closed* and skips posting.
    * **Trading day**:
        * 07:30–09:30 — premarket + macro
        * 09:30–16:00 — intraday, target/stop, 8-K, Form 4, macro
        * otherwise — 8-K + macro (after-hours / pre-premarket)
    """
    if now_et.tzinfo is None:
        now_et = now_et.replace(tzinfo=EASTERN_TZ)
    else:
        now_et = now_et.astimezone(EASTERN_TZ)

    d = now_et.date()
    if not nyse_is_trading_day(d):
        if is_weekend(d):
            return {FLOW_MACRO}
        return set()

    hm = now_et.hour * 60 + now_et.minute
    if (7 * 60 + 30) <= hm < (9 * 60 + 30):
        return {FLOW_PREMARKET, FLOW_MACRO}
    if (9 * 60 + 30) <= hm < (16 * 60):
        return {
            FLOW_INTRADAY,
            FLOW_TARGET_STOP,
            FLOW_8K,
            FLOW_FORM4,
            FLOW_MACRO,
        }
    return {FLOW_8K, FLOW_MACRO}


def _ensure_et(now: Optional[datetime]) -> datetime:
    if now is None:
        return datetime.now(EASTERN_TZ)
    if now.tzinfo is None:
        return now.replace(tzinfo=EASTERN_TZ)
    return now.astimezone(EASTERN_TZ)


# --- Collectors (no post) ---


async def collect_8k_filings(
    store: Store,
    watchlist: set[str],
) -> list[PendingDelivery]:
    if not watchlist:
        return []
    rows = await store.list_filings_pending_event_alert(
        form_types=["8-K"],
        symbols={s.upper() for s in watchlist},
        limit=30,
    )
    out: list[PendingDelivery] = []
    system = _prompt_path("filing_8k.md").read_text(encoding="utf-8")
    for rec in rows:
        ref = f"8k:{rec.accession_number}"
        if await store.has_alert_been_sent(ALERT_TYPE_8K, ref):
            continue
        body = rec.summary
        if not body and rec.raw_text:
            try:
                body = await filing_excerpt_summarize(
                    store,
                    system,
                    f"Excerpt from 8-K:\n\n{rec.raw_text[:120_000]}",
                )
                await store.cache_filing_summary(
                    rec.accession_number, body, summary_model="event-alert-haiku"
                )
            except Exception as exc:
                body = f"(Could not summarize: {exc})"
        elif not body:
            body = "New 8-K filing (no text cached yet)."

        lines = body.strip().splitlines()[:4]
        short_body = "\n".join(lines)[:900]
        if len(body) > len(short_body):
            short_body += "…"

        out.append(
            PendingDelivery(
                alert=EventAlert(
                    alert_type="8k",
                    symbol=rec.symbol,
                    title=f"8-K — {rec.symbol}",
                    body=short_body,
                    url=rec.url,
                ),
                reference_id=ref,
                db_alert_type=ALERT_TYPE_8K,
                accession=rec.accession_number,
                payload={"symbol": rec.symbol, "accession": rec.accession_number},
            )
        )
    return out


async def collect_form4_large(
    store: Store,
    watchlist: set[str],
    *,
    min_value: Decimal = Decimal("1000000"),
) -> list[PendingDelivery]:
    if not watchlist:
        return []
    rows = await store.list_filings_pending_event_alert(
        form_types=["4"],
        symbols={s.upper() for s in watchlist},
        limit=40,
    )
    out: list[PendingDelivery] = []
    for rec in rows:
        ref = f"form4:{rec.accession_number}"
        if await store.has_alert_been_sent(ALERT_TYPE_FORM4, ref):
            continue
        raw = rec.raw_text
        if not raw:
            continue
        max_usd = form4_max_transaction_usd(raw)
        if max_usd is None or max_usd < min_value:
            continue

        parsed = parse_form4_xml(raw)
        body = (
            f"{parsed[:600]}\n\n"
            f"Max notional **${float(max_usd):,.0f}** (floor ${float(min_value):,.0f})."
        )
        out.append(
            PendingDelivery(
                alert=EventAlert(
                    alert_type="form4",
                    symbol=rec.symbol,
                    title=f"Form 4 — {rec.symbol}",
                    body=body[:3500],
                    url=rec.url,
                ),
                reference_id=ref,
                db_alert_type=ALERT_TYPE_FORM4,
                accession=rec.accession_number,
                payload={"max_usd": str(max_usd)},
            )
        )
    return out


def _yf_fast_info(symbol: str) -> dict[str, Any]:
    return getattr(yf.Ticker(symbol), "fast_info", {}) or {}


def _yf_regular_price_and_open(symbol: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        cur = info.get("regularMarketPrice") or info.get("currentPrice")
        op = info.get("regularMarketOpen")
        pc = info.get("previousClose") or info.get("regularMarketPreviousClose")
        fi = getattr(t, "fast_info", {}) or {}
        if cur is None:
            cur = fi.get("last_price")
        if pc is None:
            pc = fi.get("previous_close")
        return (
            float(cur) if cur is not None else None,
            float(op) if op is not None else None,
            float(pc) if pc is not None else None,
        )
    except Exception:
        return None, None, None


async def _finnhub_quote_async(symbol: str) -> Optional[dict[str, float]]:
    key = (os.environ.get("FINNHUB_API_KEY") or "").strip()
    if not key:
        return None
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                "https://finnhub.io/api/v1/quote",
                params={"symbol": symbol, "token": key},
            )
            r.raise_for_status()
            d = r.json()
        return {
            "c": float(d.get("c") or 0),
            "pc": float(d.get("pc") or 0),
            "o": float(d.get("o") or 0),
        }
    except Exception:
        return None


async def collect_premarket_gaps(
    store: Store,
    watchlist: set[str],
    *,
    as_of: Optional[datetime] = None,
) -> list[PendingDelivery]:
    now = _ensure_et(as_of)
    today = now.date()
    out: list[PendingDelivery] = []
    for sym in sorted(watchlist):
        sym = sym.upper()
        ref_base = f"premarket:{sym}:{today.isoformat()}"
        if await store.has_alert_been_sent(ALERT_TYPE_PREMARKET, ref_base):
            continue

        fh = await _finnhub_quote_async(sym)
        if fh and fh.get("pc"):
            cur = fh.get("c") or 0
            prev = fh["pc"]
        else:
            fi = await asyncio.to_thread(_yf_fast_info, sym)
            cur = fi.get("preMarketPrice") or fi.get("last_price")
            prev = fi.get("previous_close")
            if cur is None or prev is None:
                _, _, pc2 = await asyncio.to_thread(_yf_regular_price_and_open, sym)
                cur = cur or fi.get("last_price")
                prev = prev or pc2
        if not cur or not prev:
            continue
        gap_pct = abs((float(cur) - float(prev)) / float(prev) * 100.0)
        thresh = float(await resolve_alert_threshold_pct(sym, store))
        if gap_pct < thresh:
            continue

        body = (
            f"Gap vs prior close **{gap_pct:.2f}%** (threshold {thresh:.1f}%).\n"
            f"~ ${float(prev):.2f} → ${float(cur):.2f}"
        )
        out.append(
            PendingDelivery(
                alert=EventAlert(
                    alert_type="premarket",
                    symbol=sym,
                    title=f"Pre-market — {sym}",
                    body=body,
                    url=None,
                ),
                reference_id=ref_base,
                db_alert_type=ALERT_TYPE_PREMARKET,
                payload={"gap_pct": gap_pct},
            )
        )
    return out


async def collect_intraday_moves(
    store: Store,
    watchlist: set[str],
    *,
    as_of: Optional[datetime] = None,
) -> list[PendingDelivery]:
    now = _ensure_et(as_of)
    today = now.date()
    out: list[PendingDelivery] = []
    for sym in sorted(watchlist):
        sym = sym.upper()
        cur, op, _ = await asyncio.to_thread(_yf_regular_price_and_open, sym)
        if not cur or not op or op == 0:
            continue
        move_pct = (float(cur) - float(op)) / float(op) * 100.0
        thresh = float(await resolve_alert_threshold_pct(sym, store))
        if abs(move_pct) < thresh:
            continue
        direction: Literal["up", "down"] = "up" if move_pct >= 0 else "down"
        ref = f"intraday:{sym}:{today.isoformat()}"
        if await store.has_alert_been_sent(ALERT_TYPE_INTRADAY, ref):
            continue

        body = (
            f"Move vs **today’s open**: **{move_pct:+.2f}%** (±{thresh:.1f}% threshold).\n"
            f"Open ${float(op):.2f} → now ${float(cur):.2f}"
        )
        out.append(
            PendingDelivery(
                alert=EventAlert(
                    alert_type="intraday",
                    symbol=sym,
                    title=f"Intraday — {sym}",
                    body=body,
                    url=None,
                    intraday_direction=direction,
                ),
                reference_id=ref,
                db_alert_type=ALERT_TYPE_INTRADAY,
                payload={"move_pct": move_pct},
            )
        )
    return out


async def collect_target_stop(store: Store) -> list[PendingDelivery]:
    positions = await store.get_positions(include_closed=False)
    out: list[PendingDelivery] = []
    for p in positions:
        if p.target_pct is None and p.stop_pct is None:
            continue
        cur, _, _ = await asyncio.to_thread(_yf_regular_price_and_open, p.symbol)
        if cur is None:
            continue
        cost = float(p.cost_basis)
        cur_f = float(cur)
        if p.target_pct is not None:
            tgt_px = cost * (1 + float(p.target_pct) / 100.0)
            ref = f"target:{p.id}"
            if cur_f >= tgt_px and not await store.has_alert_been_sent(ALERT_TYPE_TARGET, ref):
                body = (
                    f"**${cur_f:.2f}** ≥ target **${tgt_px:.2f}** "
                    f"(+{p.target_pct}% vs cost ${cost:.2f})."
                )
                out.append(
                    PendingDelivery(
                        alert=EventAlert(
                            alert_type="target",
                            symbol=p.symbol,
                            title=f"Target — {p.symbol}",
                            body=body,
                            url=None,
                        ),
                        reference_id=ref,
                        db_alert_type=ALERT_TYPE_TARGET,
                        payload={"position_id": str(p.id)},
                    )
                )
        if p.stop_pct is not None:
            stop_px = cost * (1 - float(p.stop_pct) / 100.0)
            ref = f"stop:{p.id}"
            if cur_f <= stop_px and not await store.has_alert_been_sent(ALERT_TYPE_STOP, ref):
                body = (
                    f"**${cur_f:.2f}** ≤ stop **${stop_px:.2f}** "
                    f"(−{p.stop_pct}% vs cost ${cost:.2f})."
                )
                out.append(
                    PendingDelivery(
                        alert=EventAlert(
                            alert_type="stop",
                            symbol=p.symbol,
                            title=f"Stop — {p.symbol}",
                            body=body,
                            url=None,
                        ),
                        reference_id=ref,
                        db_alert_type=ALERT_TYPE_STOP,
                        payload={"position_id": str(p.id)},
                    )
                )
    return out


MACRO_WATCH_SERIES = ("CPIAUCSL", "UNRATE", "PAYEMS")


async def collect_macro_releases(
    store: Store,
    *,
    as_of: Optional[datetime] = None,
) -> list[PendingDelivery]:
    now = _ensure_et(as_of)
    today = now.date()
    out: list[PendingDelivery] = []
    for sid in MACRO_WATCH_SERIES:
        obs_list = await store.get_macro_latest(sid, n=3)
        if not obs_list:
            continue
        latest = obs_list[-1]
        if latest.date != today or latest.value is None:
            continue
        ref = f"macro_release:{sid}:{latest.date.isoformat()}"
        if await store.has_alert_been_sent(ALERT_TYPE_MACRO, ref):
            continue

        body = f"Observation **{latest.date}** — value **{latest.value}**."
        out.append(
            PendingDelivery(
                alert=EventAlert(
                    alert_type="macro_release",
                    symbol=None,
                    title=f"Macro — {sid}",
                    body=body,
                    url=None,
                ),
                reference_id=ref,
                db_alert_type=ALERT_TYPE_MACRO,
                payload={"series_id": sid},
            )
        )
    return out


BATCH_THRESHOLD = 20


async def flush_pending_deliveries(
    store: Store,
    pending: list[PendingDelivery],
    *,
    channel_id: int,
    bot_token: str,
) -> None:
    """Post Discord messages and persist ``alerts_sent`` (+ filing flags). Never double-fire."""
    if not pending:
        return

    batched = len(pending) > BATCH_THRESHOLD
    if batched:
        lines: list[str] = []
        for p in pending[:50]:
            sym = f" **{p.alert.symbol}** —" if p.alert.symbol else ""
            lines.append(f"•{sym} {p.alert.title}: {p.alert.body[:100]}…")
        if len(pending) > 50:
            lines.append(f"… +{len(pending) - 50} more (dedup keys stored).")
        batch_alert = EventAlert(
            alert_type="batch",
            symbol=None,
            title=f"{len(pending)} event alerts (batched)",
            body="\n".join(lines)[:3900],
            url=None,
        )
        await _post_event_alert_rest(batch_alert, channel_id=channel_id, bot_token=bot_token)
    else:
        for p in pending:
            await _post_event_alert_rest(p.alert, channel_id=channel_id, bot_token=bot_token)

    for p in pending:
        await _mark_sent(
            store,
            p.db_alert_type,
            p.reference_id,
            channel_id=channel_id,
            payload=p.payload,
        )
        if p.accession:
            await store.set_filing_alert_sent(p.accession, alert_sent=True)


async def run_event_alert_job(
    store: Store,
    fetcher: StockDataFetcher,
    watchlist: set[str],
    *,
    channel_id: int,
    bot_token: str,
    now: Optional[datetime] = None,
    active_flows: Optional[set[str]] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Run gated event alerts. Posts to **ALERT_CHANNEL_ID** only (never ``BRIEFING_CHANNEL_ID``).

    * ``active_flows`` — test override; default from :func:`compute_active_flows`.
    * ``now`` — frozen clock (ET).
    * ``dry_run`` — collect candidates but do not post or write ``alerts_sent`` / filing flags.
    """
    _ = fetcher  # reserved for future unified quotes
    wl = {w.strip().upper() for w in watchlist if w.strip()}
    out: dict[str, Any] = {
        "flows": [],
        "posted": 0,
        "market_closed_logged": False,
        "dry_run": dry_run,
    }

    if not dry_run and (not channel_id or not (bot_token or "").strip()):
        log.warning(
            "ALERT_CHANNEL_ID or DISCORD_TOKEN missing; skipping event alerts (no post, no DB marks)."
        )
        return out

    now_et = _ensure_et(now)
    d = now_et.date()
    flows = active_flows if active_flows is not None else compute_active_flows(now_et)
    out["flows"] = sorted(flows)

    if not flows:
        log.info("market closed")
        out["market_closed_logged"] = True
        return out

    pending: list[PendingDelivery] = []

    if FLOW_8K in flows:
        pending.extend(await collect_8k_filings(store, wl))
    if FLOW_FORM4 in flows:
        pending.extend(await collect_form4_large(store, wl))
    if FLOW_PREMARKET in flows:
        pending.extend(await collect_premarket_gaps(store, wl, as_of=now_et))
    if FLOW_INTRADAY in flows:
        pending.extend(await collect_intraday_moves(store, wl, as_of=now_et))
    if FLOW_TARGET_STOP in flows:
        pending.extend(await collect_target_stop(store))
    if FLOW_MACRO in flows:
        pending.extend(await collect_macro_releases(store, as_of=now_et))

    out["posted"] = len(pending)
    if dry_run:
        return out
    await flush_pending_deliveries(store, pending, channel_id=channel_id, bot_token=bot_token)
    return out


# Back-compat name
async def run_all_event_checks(
    store: Store,
    fetcher: StockDataFetcher,
    watchlist: set[str],
    *,
    channel_id: int,
    bot_token: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Deprecated: use :func:`run_event_alert_job` (NYSE + session aware)."""
    return await run_event_alert_job(
        store,
        fetcher,
        watchlist,
        channel_id=channel_id,
        bot_token=bot_token,
        dry_run=dry_run,
    )


def dummy_event_alert(alert_type: str) -> EventAlert:
    """For ``/alerts test`` routing checks."""
    t = alert_type.strip().lower()
    if t in ("8k", "8-k", "eight_k"):
        return EventAlert(
            "8k",
            "TEST",
            "Test 8-K routing",
            "Dummy material event line.\nSecond line.",
            "https://www.sec.gov/",
        )
    if t in ("form4", "form_4", "4"):
        return EventAlert(
            "form4",
            "TEST",
            "Test Form 4 routing",
            "Dummy insider activity.\n$1M+ notional.",
            None,
        )
    if t in ("premarket", "pre"):
        return EventAlert(
            "premarket",
            "TEST",
            "Test pre-market routing",
            "Gap vs prior **2.5%** (dummy).",
            None,
        )
    if t in ("intraday", "intra"):
        return EventAlert(
            "intraday",
            "TEST",
            "Test intraday routing",
            "Move vs open **+1.0%** (dummy).",
            None,
            intraday_direction="up",
        )
    if t == "target":
        return EventAlert(
            "target",
            "NVDA",
            "Test target routing",
            "Price crossed target band (dummy).",
            None,
        )
    if t == "stop":
        return EventAlert(
            "stop",
            "NVDA",
            "Test stop routing",
            "Price crossed stop band (dummy).",
            None,
        )
    if t in ("macro", "fred", "macro_release"):
        return EventAlert(
            "macro_release",
            None,
            "Test macro routing",
            "CPI observation **dummy**.",
            None,
        )
    raise ValueError(f"Unknown alert type: {alert_type!r}")
