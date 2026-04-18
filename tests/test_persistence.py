"""Unit tests for :mod:`core.persistence` with a mocked async Supabase client."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
import pytest_asyncio

from core.persistence import (
    FilingNotFoundError,
    Store,
    StoreConfigurationError,
    StoreConnectionError,
)


def _resp(data):
    r = MagicMock()
    r.data = data
    return r


@pytest.fixture
def mock_client() -> MagicMock:
    c = MagicMock()
    # Default chain for table().select()...execute() → awaitable returning empty list
    exec_default = AsyncMock(return_value=_resp([]))
    sel = c.table.return_value.select.return_value
    sel.is_.return_value.execute = exec_default
    sel.execute = exec_default
    sel.eq.return_value.maybe_single.return_value.execute = AsyncMock(return_value=None)
    sel.eq.return_value.eq.return_value.limit.return_value.execute = AsyncMock(
        return_value=_resp([])
    )
    sel.eq.return_value.order.return_value.limit.return_value.execute = AsyncMock(
        return_value=_resp([])
    )
    c.table.return_value.insert.return_value.execute = AsyncMock(return_value=_resp([]))
    c.table.return_value.update.return_value.eq.return_value.execute = AsyncMock(
        return_value=_resp([])
    )
    c.table.return_value.upsert.return_value.execute = AsyncMock(return_value=_resp([]))
    return c


@pytest_asyncio.fixture
async def store(mock_client: MagicMock) -> Store:
    return await Store.create(client=mock_client)


@pytest.mark.asyncio
async def test_store_requires_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    with pytest.raises(StoreConfigurationError):
        await Store.create()


@pytest.mark.asyncio
async def test_create_client_wraps_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("network down")

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "key")
    monkeypatch.setattr("core.persistence.create_async_client", boom)
    with pytest.raises(StoreConnectionError, match="Could not create Supabase"):
        await Store.create()


@pytest.mark.asyncio
async def test_get_positions_open_only(store: Store, mock_client: MagicMock) -> None:
    exec_mock = AsyncMock(return_value=_resp([]))
    mock_client.table.return_value.select.return_value.is_.return_value.execute = (
        exec_mock
    )

    assert await store.get_positions() == []
    mock_client.table.assert_called_with("positions")
    mock_client.table.return_value.select.assert_called_with("*")
    mock_client.table.return_value.select.return_value.is_.assert_called_once_with(
        "closed_at", "null"
    )


@pytest.mark.asyncio
async def test_get_positions_include_closed(store: Store, mock_client: MagicMock) -> None:
    mock_client.table.return_value.select.return_value.execute = AsyncMock(
        return_value=_resp([])
    )
    await store.get_positions(include_closed=True)
    mock_client.table.return_value.select.return_value.is_.assert_not_called()


@pytest.mark.asyncio
async def test_add_position_sends_decimal_strings(
    store: Store, mock_client: MagicMock
) -> None:
    pid = "550e8400-e29b-41d4-a716-446655440000"
    row = {
        "id": pid,
        "symbol": "NVDA",
        "quantity": "10",
        "cost_basis": "450",
        "opened_at": "2025-01-01T12:00:00+00:00",
        "closed_at": None,
        "notes": None,
        "target_pct": None,
        "stop_pct": None,
    }
    mock_client.table.return_value.insert.return_value.execute = AsyncMock(
        return_value=_resp([row])
    )

    opened = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    p = await store.add_position("nvda", Decimal("10"), "450", opened)

    assert p.symbol == "NVDA"
    assert p.quantity == Decimal("10")
    assert p.id == UUID(pid)
    mock_client.table.return_value.insert.assert_called_once()
    inserted = mock_client.table.return_value.insert.call_args[0][0]
    assert inserted["quantity"] == "10"
    assert inserted["cost_basis"] == "450"


@pytest.mark.asyncio
async def test_add_position_optional_targets(store: Store, mock_client: MagicMock) -> None:
    pid = "550e8400-e29b-41d4-a716-446655440000"
    row = {
        "id": pid,
        "symbol": "NVDA",
        "quantity": "10",
        "cost_basis": "450",
        "opened_at": "2025-01-01T12:00:00+00:00",
        "closed_at": None,
        "notes": None,
        "target_pct": "15",
        "stop_pct": "10",
    }
    mock_client.table.return_value.insert.return_value.execute = AsyncMock(
        return_value=_resp([row])
    )
    opened = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    p = await store.add_position(
        "NVDA",
        Decimal("10"),
        "450",
        opened,
        target_pct="15",
        stop_pct="10",
    )
    assert p.target_pct == Decimal("15")
    assert p.stop_pct == Decimal("10")
    ins = mock_client.table.return_value.insert.call_args[0][0]
    assert ins["target_pct"] == "15"
    assert ins["stop_pct"] == "10"


@pytest.mark.asyncio
async def test_close_open_positions_for_symbol(store: Store, mock_client: MagicMock) -> None:
    pid = UUID("550e8400-e29b-41d4-a716-446655440000")
    row = {
        "id": str(pid),
        "symbol": "NVDA",
        "quantity": "10",
        "cost_basis": "450",
        "opened_at": "2025-01-01T12:00:00+00:00",
        "closed_at": "2025-06-01T12:00:00+00:00",
        "notes": None,
        "target_pct": None,
        "stop_pct": None,
    }
    mock_client.table.return_value.update.return_value.eq.return_value.is_.return_value.execute = (
        AsyncMock(return_value=_resp([row]))
    )

    out = await store.close_open_positions_for_symbol("nvda")
    assert len(out) == 1
    assert out[0].symbol == "NVDA"
    mock_client.table.return_value.update.assert_called_once()


@pytest.mark.asyncio
async def test_close_position(store: Store, mock_client: MagicMock) -> None:
    pid = UUID("550e8400-e29b-41d4-a716-446655440000")
    row = {
        "id": str(pid),
        "symbol": "NVDA",
        "quantity": "10",
        "cost_basis": "450",
        "opened_at": "2025-01-01T12:00:00+00:00",
        "closed_at": "2025-06-01T12:00:00+00:00",
        "notes": None,
        "target_pct": None,
        "stop_pct": None,
    }
    mock_client.table.return_value.update.return_value.eq.return_value.execute = (
        AsyncMock(return_value=_resp([row]))
    )

    out = await store.close_position(pid)
    assert out.closed_at is not None
    mock_client.table.return_value.update.assert_called_once()


@pytest.mark.asyncio
async def test_close_position_no_row(store: Store, mock_client: MagicMock) -> None:
    mock_client.table.return_value.update.return_value.eq.return_value.execute = (
        AsyncMock(return_value=_resp([]))
    )
    with pytest.raises(StoreConnectionError, match="no row updated"):
        await store.close_position(UUID("550e8400-e29b-41d4-a716-446655440000"))


@pytest.mark.asyncio
async def test_get_filing_summary_found(store: Store, mock_client: MagicMock) -> None:
    ms = mock_client.table.return_value.select.return_value.eq.return_value.maybe_single
    ms.return_value.execute = AsyncMock(
        return_value=SimpleNamespace(
            data={
                "accession_number": "0001171843-24-000123",
                "symbol": "NVDA",
                "cik": "0001045810",
                "form_type": "8-K",
                "filed_at": "2024-06-01T12:00:00+00:00",
                "url": "https://www.sec.gov/example.htm",
                "summary": "hello",
            }
        )
    )

    assert await store.get_filing_summary(" 0001171843-24-000123 ") == "hello"


@pytest.mark.asyncio
async def test_get_filing_summary_none(store: Store, mock_client: MagicMock) -> None:
    ms = mock_client.table.return_value.select.return_value.eq.return_value.maybe_single
    ms.return_value.execute = AsyncMock(return_value=None)

    assert await store.get_filing_summary("x") is None


@pytest.mark.asyncio
async def test_cache_filing_summary_ok(store: Store, mock_client: MagicMock) -> None:
    mock_client.table.return_value.update.return_value.eq.return_value.execute = (
        AsyncMock(return_value=_resp([{"accession_number": "a1"}]))
    )
    await store.cache_filing_summary("a1", "sum", summary_model="haiku")
    mock_client.table.return_value.update.assert_called_once()


@pytest.mark.asyncio
async def test_cache_filing_summary_missing_row(
    store: Store, mock_client: MagicMock
) -> None:
    mock_client.table.return_value.update.return_value.eq.return_value.execute = (
        AsyncMock(return_value=_resp([]))
    )
    with pytest.raises(FilingNotFoundError):
        await store.cache_filing_summary("missing", "x")


@pytest.mark.asyncio
async def test_upsert_macro_sends_value_as_string(
    store: Store, mock_client: MagicMock
) -> None:
    mock_client.table.return_value.upsert.return_value.execute = AsyncMock(
        return_value=_resp([])
    )
    await store.upsert_macro("DGS10", date(2025, 1, 2), Decimal("4.25"))
    mock_client.table.return_value.upsert.assert_called_once()
    call_args = mock_client.table.return_value.upsert.call_args
    assert call_args[1].get("on_conflict") == "series_id,date"
    row = call_args[0][0]
    assert row["value"] == "4.25"


@pytest.mark.asyncio
async def test_get_macro_latest(store: Store, mock_client: MagicMock) -> None:
    rows = [
        {"series_id": "DGS10", "date": "2025-01-02", "value": 4.1, "fetched_at": None},
        {"series_id": "DGS10", "date": "2025-01-01", "value": 4.0, "fetched_at": None},
    ]
    mock_client.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute = AsyncMock(
        return_value=_resp(rows)
    )

    obs = await store.get_macro_latest("DGS10", n=10)
    assert len(obs) == 2
    assert obs[0].date == date(2025, 1, 1)
    assert obs[1].value == Decimal("4.1")


@pytest.mark.asyncio
async def test_has_alert_been_sent(store: Store, mock_client: MagicMock) -> None:
    mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute = AsyncMock(
        return_value=_resp([{"id": "1"}])
    )
    assert await store.has_alert_been_sent("morning_briefing", "2025-01-01") is True

    mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute = AsyncMock(
        return_value=_resp([])
    )
    assert await store.has_alert_been_sent("morning_briefing", "2025-01-02") is False


@pytest.mark.asyncio
async def test_mark_alert_sent(store: Store, mock_client: MagicMock) -> None:
    mock_client.table.return_value.insert.return_value.execute = AsyncMock(
        return_value=_resp([])
    )
    await store.mark_alert_sent(
        "filing",
        "0001171843-24-000001",
        channel_id="99",
        payload={"k": "v"},
    )
    mock_client.table.return_value.insert.assert_called_once()


@pytest.mark.asyncio
async def test_get_watchlist_meta(store: Store, mock_client: MagicMock) -> None:
    mock_client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute = AsyncMock(
        return_value=SimpleNamespace(
            data={
                "symbol": "NVDA",
                "cik": "1045810",
                "company_name": "NVIDIA",
                "bucket": "core",
                "updated_at": "2025-01-01T00:00:00+00:00",
            }
        )
    )
    m = await store.get_watchlist_meta("nvda")
    assert m is not None
    assert m.cik == "1045810"


@pytest.mark.asyncio
async def test_set_watchlist_meta_merge(store: Store, mock_client: MagicMock) -> None:
    mock_client.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute = AsyncMock(
        return_value=SimpleNamespace(
            data={
                "symbol": "NVDA",
                "cik": "1045810",
                "company_name": "NVIDIA Corp",
                "bucket": "core",
                "updated_at": "2025-01-01T00:00:00+00:00",
            }
        )
    )
    mock_client.table.return_value.upsert.return_value.execute = AsyncMock(
        return_value=_resp([])
    )

    await store.set_watchlist_meta("NVDA", company_name="NVIDIA")

    upsert_row = mock_client.table.return_value.upsert.call_args[0][0]
    assert upsert_row["symbol"] == "NVDA"
    assert upsert_row["cik"] == "1045810"
    assert upsert_row["company_name"] == "NVIDIA"


@pytest.mark.asyncio
async def test_execute_api_error_wrapped(store: Store, mock_client: MagicMock) -> None:
    try:
        from postgrest.exceptions import APIError
    except ImportError:
        pytest.skip("postgrest not available")

    mock_client.table.return_value.select.return_value.is_.return_value.execute = (
        AsyncMock(
            side_effect=APIError(
                {"message": "bad", "code": "42501", "hint": None, "details": None}
            )
        )
    )
    with pytest.raises(StoreConnectionError, match="get_positions"):
        await store.get_positions()


@pytest.mark.asyncio
async def test_get_macro_latest_n_zero(store: Store) -> None:
    assert await store.get_macro_latest("X", n=0) == []
