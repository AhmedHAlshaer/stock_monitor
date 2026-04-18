"""Event alerts: thresholds, NYSE gating, dedup, targets, batching (mocked I/O)."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

import core.alerts as alerts_mod
from core.alerts import (
    DEFAULT_THRESHOLDS,
    EventAlert,
    PendingDelivery,
    collect_target_stop,
    default_threshold_for_symbol,
    flush_pending_deliveries,
    form4_max_transaction_usd,
    resolve_alert_threshold_pct,
    run_event_alert_job,
    compute_active_flows,
    nyse_is_trading_day,
    dummy_event_alert,
    FLOW_MACRO,
    ALERT_TYPE_8K,
)
from core.persistence import Position, WatchlistMeta

ET = ZoneInfo("America/New_York")


def test_default_thresholds_known_symbols() -> None:
    assert DEFAULT_THRESHOLDS["NVDA"] == Decimal("3.0")
    assert DEFAULT_THRESHOLDS["WULF"] == Decimal("6.0")


def test_default_threshold_unknown() -> None:
    assert default_threshold_for_symbol("XYZUNKNOWN") == Decimal("5.0")


@pytest.mark.asyncio
async def test_resolve_uses_db_then_default() -> None:
    store = MagicMock()
    store.get_watchlist_meta = AsyncMock(
        return_value=WatchlistMeta(symbol="NVDA", alert_threshold_pct=Decimal("2.5"))
    )
    out = await resolve_alert_threshold_pct("NVDA", store)  # type: ignore[arg-type]
    assert out == Decimal("2.5")


def test_form4_max_transaction_usd() -> None:
    xml = """<?xml version="1.0"?>
    <root xmlns="http://www.sec.gov/edgar/document/thirteenf">
      <nonDerivativeTransaction>
        <transactionShares>1000</transactionShares>
        <transactionPricePerShare>50.00</transactionPricePerShare>
      </nonDerivativeTransaction>
      <nonDerivativeTransaction>
        <transactionValue>250000</transactionValue>
      </nonDerivativeTransaction>
    </root>
    """
    m = form4_max_transaction_usd(xml)
    assert m is not None
    assert m == Decimal("250000")


def test_compute_active_flows_premarket_vs_afternoon() -> None:
    wed = datetime(2026, 4, 15, 8, 0, tzinfo=ET)
    if not nyse_is_trading_day(wed.date()):
        pytest.skip("NYSE calendar closed")
    flows_am = compute_active_flows(wed)
    assert "premarket_gaps" in flows_am
    assert "intraday_moves" not in flows_am

    pm = datetime(2026, 4, 15, 14, 0, tzinfo=ET)
    if not nyse_is_trading_day(pm.date()):
        pytest.skip("NYSE calendar closed")
    flows_pm = compute_active_flows(pm)
    assert "premarket_gaps" not in flows_pm
    assert "intraday_moves" in flows_pm


def test_compute_3am_no_intraday() -> None:
    wed = datetime(2026, 4, 15, 3, 0, tzinfo=ET)
    if not nyse_is_trading_day(wed.date()):
        pytest.skip("NYSE calendar closed")
    f = compute_active_flows(wed)
    assert "intraday_moves" not in f


def test_weekend_only_macro() -> None:
    sat = datetime(2026, 4, 18, 12, 0, tzinfo=ET)
    assert compute_active_flows(sat) == {FLOW_MACRO}


def test_weekday_holiday_empty_flows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alerts_mod, "nyse_is_trading_day", lambda d: False)
    monkeypatch.setattr(alerts_mod, "is_weekend", lambda d: False)
    wed = datetime(2026, 4, 15, 14, 0, tzinfo=ET)
    assert compute_active_flows(wed) == set()


@pytest.mark.asyncio
async def test_run_job_skips_without_channel() -> None:
    store = MagicMock()
    fetcher = MagicMock()
    out = await run_event_alert_job(
        store,  # type: ignore[arg-type]
        fetcher,  # type: ignore[arg-type]
        {"AAPL"},
        channel_id=0,
        bot_token="x",
        dry_run=False,
    )
    assert out.get("posted") == 0


@pytest.mark.asyncio
async def test_dedup_8k_second_run_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = MagicMock()
    rec.accession_number = "0000320193-24-000123"
    rec.symbol = "AAPL"
    rec.summary = "Cached summary line."
    rec.raw_text = None
    rec.url = "https://sec.gov/"

    store = MagicMock()
    store.list_filings_pending_event_alert = AsyncMock(return_value=[rec])
    store.has_alert_been_sent = AsyncMock(return_value=False)
    store.cache_filing_summary = AsyncMock()
    store.set_filing_alert_sent = AsyncMock()
    store.mark_alert_sent = AsyncMock()

    post_rest = AsyncMock()
    monkeypatch.setattr(alerts_mod, "_post_event_alert_rest", post_rest)

    from core.alerts import collect_8k_filings

    p1 = await collect_8k_filings(store, {"AAPL"})  # type: ignore[arg-type]
    assert len(p1) == 1
    await flush_pending_deliveries(store, p1, channel_id=1, bot_token="t")
    assert post_rest.call_count == 1

    store.has_alert_been_sent = AsyncMock(return_value=True)
    p2 = await collect_8k_filings(store, {"AAPL"})  # type: ignore[arg-type]
    assert len(p2) == 0


@pytest.mark.asyncio
async def test_threshold_premarket_wulf_vs_nvda(monkeypatch: pytest.MonkeyPatch) -> None:
    async def resolve(sym: str, store: object) -> Decimal:
        if sym == "WULF":
            return Decimal("6.0")
        if sym == "NVDA":
            return Decimal("3.0")
        return Decimal("5.0")

    monkeypatch.setattr(alerts_mod, "resolve_alert_threshold_pct", resolve)
    monkeypatch.setattr(
        alerts_mod,
        "_finnhub_quote_async",
        AsyncMock(return_value={"c": 100.0, "pc": 96.0}),
    )

    store = MagicMock()
    store.has_alert_been_sent = AsyncMock(return_value=False)
    store.mark_alert_sent = AsyncMock()
    store.set_filing_alert_sent = AsyncMock()

    wulf = await alerts_mod.collect_premarket_gaps(
        store, {"WULF"}, as_of=datetime(2026, 4, 15, 8, 0, tzinfo=ET)
    )  # type: ignore[arg-type]
    assert len(wulf) == 0

    nv = await alerts_mod.collect_premarket_gaps(
        store, {"NVDA"}, as_of=datetime(2026, 4, 15, 8, 0, tzinfo=ET)
    )  # type: ignore[arg-type]
    assert len(nv) == 1


@pytest.mark.asyncio
async def test_target_nvda_450_15pct(monkeypatch: pytest.MonkeyPatch) -> None:
    pid = uuid4()
    pos = Position(
        id=pid,
        symbol="NVDA",
        quantity=Decimal("1"),
        cost_basis=Decimal("450"),
        opened_at=datetime(2026, 1, 1, tzinfo=ET),
        target_pct=Decimal("15"),
        stop_pct=None,
    )
    store = MagicMock()
    store.get_positions = AsyncMock(return_value=[pos])
    store.has_alert_been_sent = AsyncMock(return_value=False)
    monkeypatch.setattr(
        alerts_mod,
        "_yf_regular_price_and_open",
        lambda s: (518.0, 400.0, 440.0),
    )
    out = await collect_target_stop(store)  # type: ignore[arg-type]
    assert len(out) == 1
    assert "517.5" in out[0].alert.body or "517.50" in out[0].alert.body


@pytest.mark.asyncio
async def test_stop_nvda_450_10pct(monkeypatch: pytest.MonkeyPatch) -> None:
    pid = uuid4()
    pos = Position(
        id=pid,
        symbol="NVDA",
        quantity=Decimal("1"),
        cost_basis=Decimal("450"),
        opened_at=datetime(2026, 1, 1, tzinfo=ET),
        target_pct=None,
        stop_pct=Decimal("10"),
    )
    store = MagicMock()
    store.get_positions = AsyncMock(return_value=[pos])
    store.has_alert_been_sent = AsyncMock(return_value=False)
    monkeypatch.setattr(
        alerts_mod,
        "_yf_regular_price_and_open",
        lambda s: (400.0, 400.0, 440.0),
    )
    out = await collect_target_stop(store)  # type: ignore[arg-type]
    assert len(out) == 1
    assert "405" in out[0].alert.body


@pytest.mark.asyncio
async def test_target_wulf_10pct(monkeypatch: pytest.MonkeyPatch) -> None:
    pid = uuid4()
    pos = Position(
        id=pid,
        symbol="WULF",
        quantity=Decimal("1"),
        cost_basis=Decimal("100"),
        opened_at=datetime(2026, 1, 1, tzinfo=ET),
        target_pct=Decimal("10"),
        stop_pct=None,
    )
    store = MagicMock()
    store.get_positions = AsyncMock(return_value=[pos])
    store.has_alert_been_sent = AsyncMock(return_value=False)
    # +12% from cost → 112
    monkeypatch.setattr(
        alerts_mod,
        "_yf_regular_price_and_open",
        lambda s: (112.0, 100.0, 100.0),
    )
    out = await collect_target_stop(store)  # type: ignore[arg-type]
    assert len(out) == 1


@pytest.mark.asyncio
async def test_batch_single_discord_when_over_20(monkeypatch: pytest.MonkeyPatch) -> None:
    store = MagicMock()
    store.mark_alert_sent = AsyncMock()
    store.set_filing_alert_sent = AsyncMock()
    post = AsyncMock()
    monkeypatch.setattr(alerts_mod, "_post_event_alert_rest", post)

    pending = [
        PendingDelivery(
            EventAlert("8k", "S", f"x{i}", "body", None),
            f"8k:acc{i}",
            ALERT_TYPE_8K,
        )
        for i in range(21)
    ]
    await flush_pending_deliveries(store, pending, channel_id=1, bot_token="t")
    assert post.call_count == 1
    assert store.mark_alert_sent.await_count == 21  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_post_event_alert_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    with patch("core.alerts.httpx.AsyncClient") as m:
        inst = MagicMock()
        m.return_value.__aenter__.return_value = inst
        post = AsyncMock()
        post.return_value.raise_for_status = MagicMock()
        inst.post = post
        await alerts_mod._post_event_alert_rest(
            EventAlert("macro_release", None, "t", "b", None),
            channel_id=1,
            bot_token="tok",
        )
        post.assert_awaited_once()


def test_dummy_event_alert_types() -> None:
    a = dummy_event_alert("8k")
    assert a.alert_type == "8k"
    with pytest.raises(ValueError):
        dummy_event_alert("nope")


def test_build_event_alert_embed_bell_prefix() -> None:
    from bot.embeds.alerts import build_event_alert_embed

    e = build_event_alert_embed(
        EventAlert("8k", "X", "T", "B", "https://x.com"),
    )
    assert str(e.title).startswith("🔔")


@pytest.mark.asyncio
async def test_holiday_job_no_post(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    post = AsyncMock()
    monkeypatch.setattr(alerts_mod, "_post_event_alert_rest", post)
    store = MagicMock()
    fetcher = MagicMock()
    with caplog.at_level("INFO"):
        await run_event_alert_job(
            store,  # type: ignore[arg-type]
            fetcher,  # type: ignore[arg-type]
            {"AAPL"},
            channel_id=1,
            bot_token="tok",
            active_flows=set(),
        )
    assert "market closed" in caplog.text.lower()
    post.assert_not_called()


@pytest.mark.asyncio
async def test_dry_run_no_marks(monkeypatch: pytest.MonkeyPatch) -> None:
    post = AsyncMock()
    monkeypatch.setattr(alerts_mod, "_post_event_alert_rest", post)
    store = MagicMock()
    store.get_macro_latest = AsyncMock(return_value=[])
    fetcher = MagicMock()
    await run_event_alert_job(
        store,  # type: ignore[arg-type]
        fetcher,  # type: ignore[arg-type]
        {"AAPL"},
        channel_id=0,
        bot_token="",
        active_flows={FLOW_MACRO},
        dry_run=True,
    )
    post.assert_not_called()
