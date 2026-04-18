"""Tests for :mod:`core.positions` (mocked store; no yfinance)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from core.persistence import Position
from core.positions import PortfolioSummary, PositionTracker


def _pos(
    *,
    symbol: str = "NVDA",
    qty: str = "10",
    cost: str = "100",
) -> Position:
    return Position(
        id=uuid4(),
        symbol=symbol,
        quantity=Decimal(qty),
        cost_basis=Decimal(cost),
        opened_at=datetime(2025, 1, 15, tzinfo=timezone.utc),
        target_pct=None,
        stop_pct=None,
    )


@pytest.mark.asyncio
async def test_compute_pnl_multiple_positions() -> None:
    p1 = _pos(symbol="NVDA", qty="10", cost="100")
    p2 = _pos(symbol="AAPL", qty="5", cost="200")
    store = AsyncMock()
    store.get_positions = AsyncMock(return_value=[p1, p2])

    tracker = PositionTracker(store)
    prices = {"NVDA": 110.0, "AAPL": 190.0}
    prev = {"NVDA": 105.0, "AAPL": 195.0}
    lines = await tracker.compute_pnl_for_open(prices, prev)
    assert len(lines) == 2

    by_sym = {ln.symbol: ln for ln in lines}
    assert float(by_sym["NVDA"].unrealized_dollar) == pytest.approx(100.0)
    assert float(by_sym["NVDA"].day_change_dollar or 0) == pytest.approx(50.0)
    assert float(by_sym["AAPL"].unrealized_dollar) == pytest.approx(-50.0)


@pytest.mark.asyncio
async def test_compute_pnl_skips_missing_price() -> None:
    p = _pos()
    store = AsyncMock()
    store.get_positions = AsyncMock(return_value=[p])
    tracker = PositionTracker(store)
    lines = await tracker.compute_pnl_for_open({}, None)
    assert lines == []


@pytest.mark.asyncio
async def test_portfolio_summary_empty() -> None:
    store = AsyncMock()
    store.get_positions = AsyncMock(return_value=[])
    tracker = PositionTracker(store)
    s = await tracker.portfolio_summary({"NVDA": 1.0}, None)
    assert isinstance(s, PortfolioSummary)
    assert s.total_market_value == Decimal("0")
    assert s.lines == []


@pytest.mark.asyncio
async def test_close_delegates_to_store() -> None:
    closed = [_pos()]
    store = AsyncMock()
    store.close_open_positions_for_symbol = AsyncMock(return_value=closed)
    tracker = PositionTracker(store)
    out = await tracker.close("NVDA")
    assert out == closed
    store.close_open_positions_for_symbol.assert_awaited_once_with("NVDA")


@pytest.mark.asyncio
async def test_add_delegates_with_targets() -> None:
    inserted = _pos()
    store = AsyncMock()
    store.add_position = AsyncMock(return_value=inserted)
    tracker = PositionTracker(store)
    out = await tracker.add(
        "nvda",
        "10",
        "450",
        target_pct=Decimal("15"),
        stop_pct=Decimal("10"),
    )
    assert out.symbol == "NVDA"
    store.add_position.assert_awaited()
    call_kw = store.add_position.call_args
    assert call_kw.kwargs.get("target_pct") == Decimal("15")
    assert call_kw.kwargs.get("stop_pct") == Decimal("10")
