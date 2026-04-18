"""
Portfolio positions and P&L — uses :class:`core.persistence.Store` for data;
callers supply prices from :class:`core.stock_data.StockDataFetcher` (no price I/O here).
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field

from core.persistence import Position, Store


class PositionPnL(BaseModel):
    """Mark-to-market for one open lot."""

    position_id: UUID
    symbol: str
    quantity: Decimal
    cost_basis_per_share: Decimal
    current_price: float
    market_value: Decimal
    cost_total: Decimal
    unrealized_dollar: Decimal
    unrealized_pct: Decimal
    day_change_dollar: Optional[Decimal] = None


class PortfolioSummary(BaseModel):
    """Aggregates across open positions."""

    total_market_value: Decimal
    total_cost_total: Decimal
    total_unrealized_dollar: Decimal
    total_unrealized_pct: Decimal
    total_day_change_dollar: Optional[Decimal] = None
    lines: list[PositionPnL] = Field(default_factory=list)


class PositionTracker:
    """Domain layer for positions + P&L (prices supplied by caller)."""

    def __init__(self, store: Store) -> None:
        self._store = store

    async def add(
        self,
        symbol: str,
        quantity: Decimal | float | str,
        cost_basis: Decimal | float | str,
        notes: Optional[str] = None,
        *,
        opened_at: Optional[datetime] = None,
        target_pct: Optional[Decimal | float | str] = None,
        stop_pct: Optional[Decimal | float | str] = None,
    ) -> Position:
        """Add a lot. ``cost_basis`` is average cost **per share**."""
        opened = opened_at or datetime.now(timezone.utc)
        return await self._store.add_position(
            symbol,
            quantity,
            cost_basis,
            opened,
            notes=notes,
            target_pct=target_pct,
            stop_pct=stop_pct,
        )

    async def close(self, symbol: str) -> list[Position]:
        """Close all open lots for ``symbol``."""
        return await self._store.close_open_positions_for_symbol(symbol)

    async def list_open(self) -> list[Position]:
        return await self._store.get_positions(include_closed=False)

    def compute_pnl(
        self,
        positions: list[Position],
        current_prices: dict[str, float],
        prev_close: Optional[dict[str, float]] = None,
    ) -> list[PositionPnL]:
        """
        Build P&L lines. ``current_prices`` / ``prev_close`` are keyed by upper-case symbol.
        Missing price for a symbol skips that position (omitted from results).
        """
        prev_close = prev_close or {}
        out: list[PositionPnL] = []
        for p in positions:
            sym = p.symbol.upper()
            px = current_prices.get(sym)
            if px is None:
                continue
            qty = p.quantity
            cps = p.cost_basis
            cost_total = qty * cps
            mv = qty * Decimal(str(px))
            unreal = mv - cost_total
            unreal_pct = (
                (unreal / cost_total * Decimal("100"))
                if cost_total != 0
                else Decimal("0")
            )
            day_ch: Optional[Decimal] = None
            prev = prev_close.get(sym)
            if prev is not None:
                day_ch = qty * (Decimal(str(px)) - Decimal(str(prev)))

            out.append(
                PositionPnL(
                    position_id=p.id,
                    symbol=sym,
                    quantity=qty,
                    cost_basis_per_share=cps,
                    current_price=float(px),
                    market_value=mv,
                    cost_total=cost_total,
                    unrealized_dollar=unreal,
                    unrealized_pct=unreal_pct,
                    day_change_dollar=day_ch,
                )
            )
        return out

    async def compute_pnl_for_open(
        self,
        current_prices: dict[str, float],
        prev_close: Optional[dict[str, float]] = None,
    ) -> list[PositionPnL]:
        positions = await self.list_open()
        return self.compute_pnl(positions, current_prices, prev_close)

    async def portfolio_summary(
        self,
        current_prices: dict[str, float],
        prev_close: Optional[dict[str, float]] = None,
    ) -> PortfolioSummary:
        lines = await self.compute_pnl_for_open(current_prices, prev_close)
        if not lines:
            return PortfolioSummary(
                total_market_value=Decimal("0"),
                total_cost_total=Decimal("0"),
                total_unrealized_dollar=Decimal("0"),
                total_unrealized_pct=Decimal("0"),
                total_day_change_dollar=None,
                lines=[],
            )

        tmv = sum((ln.market_value for ln in lines), Decimal("0"))
        tct = sum((ln.cost_total for ln in lines), Decimal("0"))
        tu = sum((ln.unrealized_dollar for ln in lines), Decimal("0"))
        tup = (tu / tct * Decimal("100")) if tct != 0 else Decimal("0")
        tdc: Optional[Decimal] = None
        if any(ln.day_change_dollar is not None for ln in lines):
            tdc = sum(
                (ln.day_change_dollar or Decimal("0") for ln in lines),
                Decimal("0"),
            )

        return PortfolioSummary(
            total_market_value=tmv,
            total_cost_total=tct,
            total_unrealized_dollar=tu,
            total_unrealized_pct=tup,
            total_day_change_dollar=tdc,
            lines=lines,
        )
