"""Tests for morning briefing assembly and embed formatting."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from bot.embeds.briefing import build_briefing_embed
from core.briefing import BriefingReport, BriefingAssembler
from core.positions import PortfolioSummary, PositionPnL


def test_build_briefing_embed_fields() -> None:
    pos_ln = PositionPnL(
        position_id=uuid4(),
        symbol="NVDA",
        quantity=Decimal("10"),
        cost_basis_per_share=Decimal("100"),
        current_price=110.0,
        market_value=Decimal("1100"),
        cost_total=Decimal("1000"),
        unrealized_dollar=Decimal("100"),
        unrealized_pct=Decimal("10"),
        day_change_dollar=Decimal("5"),
    )
    summary = PortfolioSummary(
        total_market_value=Decimal("1100"),
        total_cost_total=Decimal("1000"),
        total_unrealized_dollar=Decimal("100"),
        total_unrealized_pct=Decimal("10"),
        total_day_change_dollar=Decimal("5"),
        lines=[pos_ln],
    )
    report = BriefingReport(
        date=date(2026, 4, 17),
        macro_changes=[
            {"series_id": "DGS10", "change_str": "10Y: +1bp to 4.2% (test)"},
        ],
        new_filings=[
            {
                "symbol": "META",
                "form_type": "8-K",
                "summary": "Item 2.02 results.",
                "url": "https://www.sec.gov/example",
            }
        ],
        news_items=["**NVDA** — headline one"],
        price_moves=["**AAPL** +0.5% (last session vs prior close)"],
        position_pnl=summary,
        upcoming_earnings=["**AMD** — earnings in **5**d"],
        section_errors=["macro: forced partial failure"],
    )
    embed = build_briefing_embed(report)
    assert embed.title and "Briefing" in embed.title
    names = {f.name for f in embed.fields}
    assert "🌍 Macro" in names
    assert "📁 Filings (recent)" in names
    assert "💼 Positions" in names


@pytest.mark.asyncio
async def test_assembler_build_degrades_when_macro_fails() -> None:
    store = AsyncMock()
    store.get_positions = AsyncMock(return_value=[])

    fetcher = MagicMock()
    fetcher.fetch_52_week_data = MagicMock(return_value=None)

    asm = BriefingAssembler(store, fetcher, ["AAPL"])

    async def boom_macro(*_a, **_k):
        raise RuntimeError("fred down")

    with patch.object(BriefingAssembler, "_run_macro", boom_macro):
        with patch.object(BriefingAssembler, "_run_filings", AsyncMock(return_value=[])):
            with patch.object(BriefingAssembler, "_run_news", AsyncMock(return_value=[])):
                with patch.object(
                    BriefingAssembler, "_run_price_moves", AsyncMock(return_value=[])
                ):
                    with patch.object(
                        BriefingAssembler, "_run_earnings", AsyncMock(return_value=[])
                    ):
                        report = await asm.build(date.today())

    assert any("macro" in e.lower() for e in report.section_errors)
    assert report.macro_changes == []


@pytest.mark.asyncio
async def test_deliver_morning_briefing_skips_duplicate() -> None:
    import bot.discord_bot as db

    bot = MagicMock()
    bot.store = AsyncMock()
    bot.store.has_alert_been_sent = AsyncMock(return_value=True)
    bot.fetcher = MagicMock()
    bot.watchlist = ["AAPL"]

    rep, status = await db.deliver_morning_briefing(bot, skip_dedup=False)
    assert rep is None
    assert status == "already_sent"
