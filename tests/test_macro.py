"""Tests for FRED client and macro analyzer (respx + async mocks)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from core.macro import (
    FREDClient,
    FredObservationsResponse,
    MacroAnalyzer,
    ParsedObservation,
    _direction_and_streak,
    fetch_all_series_and_upsert,
)
from core.persistence import MacroObservation, Store


FRED_JSON_DGS10 = {
    "realtime_start": "2025-01-01",
    "realtime_end": "2025-01-02",
    "observations": [
        {
            "realtime_start": "2025-01-01",
            "realtime_end": "2025-01-02",
            "date": "2025-01-01",
            "value": "4.28",
        },
        {
            "realtime_start": "2025-01-01",
            "realtime_end": "2025-01-02",
            "date": "2025-01-02",
            "value": "4.21",
        },
        {
            "realtime_start": "2025-01-01",
            "realtime_end": "2025-01-02",
            "date": "2025-01-03",
            "value": ".",
        },
    ],
}


@pytest.mark.asyncio
@respx.mock
async def test_fred_client_fetch_observations_parses_and_skips_dot() -> None:
    route = respx.get("https://api.stlouisfed.org/fred/series/observations").mock(
        return_value=httpx.Response(200, json=FRED_JSON_DGS10)
    )
    client = FREDClient(api_key="test_key")
    try:
        rows = await client.fetch_observations("DGS10")
    finally:
        await client.aclose()

    assert len(rows) == 2
    assert rows[0] == ParsedObservation(
        observation_date=date(2025, 1, 1), value=Decimal("4.28")
    )
    assert rows[1].value == Decimal("4.21")
    assert route.called
    request = route.calls[0].request
    assert request.url.params["series_id"] == "DGS10"
    assert request.url.params["api_key"] == "test_key"
    assert request.url.params["file_type"] == "json"


def test_fred_observations_response_pydantic() -> None:
    m = FredObservationsResponse.model_validate(FRED_JSON_DGS10)
    assert len(m.observations) == 3


def test_direction_and_streak_one_move_up() -> None:
    assert _direction_and_streak([Decimal("100"), Decimal("105")]) == ("up", 1)


def test_direction_and_streak_one_move_down() -> None:
    assert _direction_and_streak([Decimal("105"), Decimal("100")]) == ("down", 1)


def test_direction_and_streak_two_consecutive_ups() -> None:
    assert _direction_and_streak(
        [Decimal("100"), Decimal("102"), Decimal("105")]
    ) == ("up", 2)


def test_direction_and_streak_three_consecutive_downs() -> None:
    assert _direction_and_streak(
        [Decimal("4.25"), Decimal("4.20"), Decimal("4.15"), Decimal("4.10")]
    ) == ("down", 3)


def test_direction_and_streak_two_consecutive_downs() -> None:
    assert _direction_and_streak(
        [Decimal("4.20"), Decimal("4.15"), Decimal("4.10")]
    ) == ("down", 2)


@pytest.mark.asyncio
async def test_macro_analyzer_get_changes_yield_streak() -> None:
    """Four levels with three consecutive down moves → '3rd day of decline'."""
    store = AsyncMock(spec=Store)

    async def fake_latest(sid: str, n: int = 10) -> list[MacroObservation]:
        if sid != "DGS10":
            return []
        return [
            MacroObservation(
                series_id=sid,
                date=date(2025, 1, 1),
                value=Decimal("4.42"),
                fetched_at=None,
            ),
            MacroObservation(
                series_id=sid,
                date=date(2025, 1, 2),
                value=Decimal("4.35"),
                fetched_at=None,
            ),
            MacroObservation(
                series_id=sid,
                date=date(2025, 1, 3),
                value=Decimal("4.28"),
                fetched_at=None,
            ),
            MacroObservation(
                series_id=sid,
                date=date(2025, 1, 4),
                value=Decimal("4.21"),
                fetched_at=None,
            ),
        ]

    store.get_macro_latest = AsyncMock(side_effect=fake_latest)

    analyzer = MacroAnalyzer(store)
    changes = await analyzer.get_changes(date(2025, 1, 6))

    dgs = [c for c in changes if c["series_id"] == "DGS10"][0]
    assert dgs["latest_value"] == pytest.approx(4.21)
    assert dgs["change"] == pytest.approx(-0.07)
    assert "3rd day of decline" in dgs["change_str"]
    assert "7bp" in dgs["change_str"] or "-7bp" in dgs["change_str"]


@pytest.mark.asyncio
@respx.mock
async def test_fetch_all_series_and_upsert_calls_store() -> None:
    respx.get("https://api.stlouisfed.org/fred/series/observations").mock(
        return_value=httpx.Response(
            200,
            json={
                "observations": [
                    {"date": "2025-01-01", "value": "1.0"},
                ]
            },
        )
    )

    store = AsyncMock(spec=Store)
    client = FREDClient(api_key="k")
    try:
        counts = await fetch_all_series_and_upsert(
            store, client, lookback_days=30
        )
    finally:
        await client.aclose()

    assert len(counts) == 8
    assert store.upsert_macro.await_count == 8
    assert all(counts[sid] == 1 for sid in counts)


@pytest.mark.asyncio
@respx.mock
async def test_fred_retries_on_429() -> None:
    calls = {"n": 0}

    def side_effect(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(429, text="Too Many Requests")
        return httpx.Response(
            200,
            json={"observations": [{"date": "2025-01-01", "value": "2.0"}]},
        )

    respx.get("https://api.stlouisfed.org/fred/series/observations").mock(
        side_effect=side_effect
    )

    client = FREDClient(api_key="k", min_interval_sec=0.0)
    try:
        rows = await client.fetch_observations("DGS2")
    finally:
        await client.aclose()

    assert len(rows) == 1
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_macro_analyzer_single_observation_change_is_none() -> None:
    store = AsyncMock(spec=Store)

    async def fake_latest(sid: str, n: int = 10) -> list[MacroObservation]:
        if sid != "DGS10":
            return []
        return [
            MacroObservation(
                series_id=sid,
                date=date(2025, 1, 3),
                value=Decimal("4.21"),
                fetched_at=None,
            ),
        ]

    store.get_macro_latest = AsyncMock(side_effect=fake_latest)
    analyzer = MacroAnalyzer(store)
    changes = await analyzer.get_changes(date(2025, 1, 5))
    dgs = [c for c in changes if c["series_id"] == "DGS10"][0]
    assert dgs["change"] is None
    assert dgs["latest_value"] == pytest.approx(4.21)
