"""
Morning briefing orchestrator: loads context, gathers data concurrently, persists,
and returns a structured :class:`BriefingReport` (no Discord / no buy-sell language).

Optional :class:`core.synthesis.Synthesizer` adds a cross-reference narrative (Phase 5).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from core.macro import FREDClient, MacroAnalyzer, fetch_all_series_and_upsert
from core.news_sentiment import MarketDataAnalyzer, NewsAnalyzer
from core.persistence import FilingRecord, Store
from core.positions import PortfolioSummary, PositionTracker
from core.sec_filings import SECClient, ensure_filing_summaries
from core.stock_data import StockDataFetcher


def price_maps_from_fetcher(
    fetcher: StockDataFetcher,
    symbols: set[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Current and previous close from yfinance history (via fetcher)."""
    prices: dict[str, float] = {}
    prev_close: dict[str, float] = {}
    for s in symbols:
        df = fetcher.fetch_52_week_data(s)
        if df is None or df.empty:
            continue
        su = s.strip().upper()
        prices[su] = float(df["Close"].iloc[-1])
        if len(df) >= 2:
            prev_close[su] = float(df["Close"].iloc[-2])
    return prices, prev_close


@dataclass
class BriefingReport:
    """Structured morning briefing for tests and embed formatting."""

    date: date
    macro_changes: list[dict[str, Any]] = field(default_factory=list)
    new_filings: list[dict[str, Any]] = field(default_factory=list)
    news_items: list[str] = field(default_factory=list)
    price_moves: list[str] = field(default_factory=list)
    position_pnl: Optional[PortfolioSummary] = None
    upcoming_earnings: list[str] = field(default_factory=list)
    section_errors: list[str] = field(default_factory=list)
    synthesis: Optional[Any] = None  # core.synthesis.SynthesisResult when set


class BriefingAssembler:
    """
    Steps (guide §6):
    1. Load watchlist + open positions.
    2. Concurrently: macro refresh + filings pipeline + news + price moves (+ earnings).
    3. Persist macro/filings via existing store/SEC paths.
    4. Position P&L via :class:`PositionTracker` + ``price_maps_from_fetcher``.
    5. Optional filing Haiku summaries (``ensure_filing_summaries``).
    6. Optional cross-reference synthesis via injectable :class:`core.synthesis.Synthesizer`.
    7. Return :class:`BriefingReport` (caller formats embed / posts).
    """

    def __init__(
        self,
        store: Store,
        fetcher: StockDataFetcher,
        watchlist: list[str],
        *,
        news_analyzer: Optional[NewsAnalyzer] = None,
        market_analyzer: Optional[MarketDataAnalyzer] = None,
        synthesizer: Optional[Any] = None,
    ) -> None:
        self._store = store
        self._fetcher = fetcher
        self._watchlist = [s.strip().upper() for s in watchlist]
        self._news = news_analyzer or NewsAnalyzer()
        self._market = market_analyzer or MarketDataAnalyzer()
        self._synthesizer = synthesizer

    async def build(
        self,
        as_of_date: date,
        *,
        summarize_filings: bool = True,
        max_filing_haiku_total: int = 8,
        macro_lookback_days: int = 1100,
    ) -> BriefingReport:
        section_errors: list[str] = []
        since_filings = datetime.now(timezone.utc) - timedelta(hours=36)

        open_positions = await self._store.get_positions(include_closed=False)
        symbols: set[str] = set(self._watchlist) | {p.symbol.upper() for p in open_positions}

        macro_task = self._run_macro(as_of_date, section_errors, macro_lookback_days)
        filings_task = self._run_filings(
            since_filings,
            section_errors,
            summarize_filings,
            max_filing_haiku_total,
        )
        news_task = self._run_news(section_errors)
        prices_task = self._run_price_moves(section_errors)
        earnings_task = self._run_earnings(as_of_date, section_errors)

        results = await asyncio.gather(
            macro_task,
            filings_task,
            news_task,
            prices_task,
            earnings_task,
            return_exceptions=True,
        )
        labels = ("macro", "filings", "news", "prices", "earnings")
        merged: list[Any] = [[], [], [], [], []]
        for i, res in enumerate(results):
            if isinstance(res, BaseException):
                section_errors.append(f"{labels[i]}: {res}")
            else:
                merged[i] = res
        macro_changes, filing_rows, news_items, price_moves, upcoming = merged

        position_pnl: Optional[PortfolioSummary] = None
        try:
            prices, prev = price_maps_from_fetcher(self._fetcher, symbols)
            tracker = PositionTracker(self._store)
            position_pnl = await tracker.portfolio_summary(prices, prev)
        except Exception as exc:
            section_errors.append(f"positions: {exc}")

        report = BriefingReport(
            date=as_of_date,
            macro_changes=macro_changes,
            new_filings=filing_rows,
            news_items=news_items,
            price_moves=price_moves,
            position_pnl=position_pnl,
            upcoming_earnings=upcoming,
            section_errors=section_errors,
        )
        if self._synthesizer is not None:
            try:
                report.synthesis = await self._synthesizer.synthesize(report)
            except Exception as exc:
                report.section_errors.append(f"synthesis: {exc}")
        return report

    async def _run_macro(
        self,
        as_of_date: date,
        errors: list[str],
        lookback_days: int,
    ) -> list[dict[str, Any]]:
        try:
            if os.environ.get("FRED_API_KEY", "").strip():
                client = FREDClient()
                try:
                    await fetch_all_series_and_upsert(
                        self._store, client, lookback_days=lookback_days
                    )
                finally:
                    await client.aclose()
            analyzer = MacroAnalyzer(self._store)
            return await analyzer.get_changes(as_of_date)
        except Exception as exc:
            errors.append(f"macro: {exc}")
            return []

    async def _run_filings(
        self,
        since: datetime,
        errors: list[str],
        summarize: bool,
        haiku_budget: int,
    ) -> list[dict[str, Any]]:
        acc: list[dict[str, Any]] = []
        haiku_left = haiku_budget
        try:
            sec = SECClient()
        except ValueError as exc:
            errors.append(f"filings: {exc}")
            return []

        try:
            for sym in self._watchlist:
                try:
                    meta = await self._store.get_watchlist_meta(sym)
                    if meta and meta.cik:
                        cik = meta.cik
                    else:
                        cik = await sec.resolve_cik(sym, self._store)
                    recent = await sec.fetch_recent_filings(cik, since)
                    cap = min(len(recent), 12)
                    batch = recent[:cap]
                    if summarize and batch:
                        use = min(haiku_left, 3)
                        await ensure_filing_summaries(
                            self._store,
                            sec,
                            sym,
                            batch,
                            max_haiku_summaries=max(1, use),
                        )
                        haiku_left -= use
                    rows = await self._store.list_filings_for_symbol(sym, limit=15)
                    for r in rows:
                        fa = r.filed_at
                        if fa.tzinfo is None:
                            fa = fa.replace(tzinfo=timezone.utc)
                        else:
                            fa = fa.astimezone(timezone.utc)
                        if fa >= since:
                            acc.append(_filing_record_to_dict(r))
                except Exception as exc:
                    errors.append(f"filings[{sym}]: {exc}")
        finally:
            await sec.aclose()

        # Dedupe by accession for display
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for d in sorted(acc, key=lambda x: x.get("filed_at", ""), reverse=True):
            aid = d.get("accession_number", "")
            if aid in seen:
                continue
            seen.add(aid)
            unique.append(d)
        return unique[:20]

    async def _run_news(self, errors: list[str]) -> list[str]:
        lines: list[str] = []
        try:
            for sym in self._watchlist[:12]:
                try:
                    articles = self._news.fetch_news(sym, days=3)
                    for a in articles[:3]:
                        title = (a.title or "")[:120]
                        lines.append(f"**{sym}** — {title}")
                except Exception as exc:
                    errors.append(f"news[{sym}]: {exc}")
        except Exception as exc:
            errors.append(f"news: {exc}")
        return lines[:18]

    async def _run_price_moves(self, errors: list[str]) -> list[str]:
        out: list[str] = []
        try:
            for sym in self._watchlist[:15]:
                try:
                    df = self._fetcher.fetch_52_week_data(sym)
                    if df is None or len(df) < 2:
                        continue
                    cur = float(df["Close"].iloc[-1])
                    prev = float(df["Close"].iloc[-2])
                    pct = (cur - prev) / prev * 100.0 if prev else 0.0
                    out.append(f"**{sym}** {pct:+.2f}% (last session vs prior close)")
                except Exception as exc:
                    errors.append(f"prices[{sym}]: {exc}")
        except Exception as exc:
            errors.append(f"prices: {exc}")
        return out

    async def _run_earnings(self, as_of_date: date, errors: list[str]) -> list[str]:
        lines: list[str] = []
        try:
            for sym in self._watchlist:
                try:
                    info = self._market.get_earnings_info(sym)
                    if info.earnings_date is None or info.days_until is None:
                        continue
                    if info.days_until < 0 or info.days_until > 45:
                        continue
                    d = info.earnings_date
                    if hasattr(d, "date"):
                        d_str = d.date().isoformat()
                    else:
                        d_str = str(d)[:10]
                    lines.append(
                        f"**{sym}** — earnings in **{info.days_until}**d ({d_str})"
                    )
                except Exception as exc:
                    errors.append(f"earnings[{sym}]: {exc}")
        except Exception as exc:
            errors.append(f"earnings: {exc}")
        lines.sort(key=lambda s: s)
        return lines[:12]


def _filing_record_to_dict(r: FilingRecord) -> dict[str, Any]:
    summ = (r.summary or "").strip().replace("\n", " ")
    if len(summ) > 240:
        summ = summ[:237] + "…"
    return {
        "accession_number": r.accession_number,
        "symbol": r.symbol,
        "form_type": r.form_type,
        "filed_at": r.filed_at.isoformat(),
        "url": r.url,
        "summary": summ or "—",
    }


# Re-export for callers that import fetch_recent_filings from a single place
__all__ = (
    "BriefingAssembler",
    "BriefingReport",
    "price_maps_from_fetcher",
)
