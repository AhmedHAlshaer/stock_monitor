"""Tests for briefing synthesis (mocked LLM / store)."""

from __future__ import annotations

import re
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.briefing import BriefingReport
from core.synthesis import (
    LLMService,
    Synthesizer,
    SYNTHESIS_CACHE_ALERT,
    _sanitize_text,
    briefing_cache_reference_id,
    filing_excerpt_summarize,
)


def _minimal_briefing() -> BriefingReport:
    return BriefingReport(
        date=date(2026, 4, 17),
        macro_changes=[{"change_str": "VIX +0.2 to 18", "series_id": "VIXCLS"}],
        new_filings=[
            {
                "symbol": "NVDA",
                "form_type": "8-K",
                "summary": "Item 2.02 results",
                "url": "https://sec.gov/x",
            }
        ],
        news_items=[
            "**NVDA** — semiconductor demand headline",
            "**NVDA** — insider activity story",
        ],
        price_moves=["**NVDA** +1.0% (last session vs prior close)"],
        upcoming_earnings=[],
        section_errors=[],
    )


@pytest.mark.asyncio
async def test_briefing_skip_synthesis_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRIEFING_SKIP_SYNTHESIS", "1")
    store = AsyncMock()
    store.get_alerts_sent_payload = AsyncMock(
        return_value={"paragraph": "Would be cached", "key_points": []}
    )
    syn = Synthesizer(store)
    out = await syn.synthesize(_minimal_briefing())
    assert out.model_used == "disabled"
    assert "BRIEFING_SKIP_SYNTHESIS" in out.paragraph
    store.get_alerts_sent_payload.assert_not_awaited()


@pytest.mark.asyncio
async def test_synthesis_cache_hit_instant() -> None:
    store = AsyncMock()
    store.get_alerts_sent_payload = AsyncMock(
        return_value={
            "paragraph": "Cached paragraph.",
            "key_points": ["Point one"],
            "model_used": "claude-3-5-haiku-20241022",
        }
    )
    store.sum_llm_tokens_for_date = AsyncMock(return_value=0)

    syn = Synthesizer(store)
    b = _minimal_briefing()
    out = await syn.synthesize(b)

    assert out.cached is True
    assert out.model_used == "cached"
    assert out.paragraph == "Cached paragraph."
    assert out.key_points == ["Point one"]
    store.get_alerts_sent_payload.assert_awaited()
    ref = briefing_cache_reference_id(b)
    store.get_alerts_sent_payload.assert_awaited_with(SYNTHESIS_CACHE_ALERT, ref)


@pytest.mark.asyncio
async def test_synthesis_nvda_cross_reference_mock_llm() -> None:
    """NVDA 8-K + headlines — mock response links themes without buy/sell."""
    store = AsyncMock()
    store.get_alerts_sent_payload = AsyncMock(return_value=None)
    store.sum_llm_tokens_for_date = AsyncMock(return_value=0)
    store.mark_alert_sent = AsyncMock()

    mock_json = (
        '{"opener":"NVDA filed an 8-K per overnight data while headlines cited '
        'semiconductor demand; Form 4 activity appeared in parallel news items.",'
        '"bullets":["Per the 8-K, results were disclosed this cycle.",'
        '"Per overnight headlines, sector demand narratives overlap the ticker."],'
        '"nothing":false}'
    )

    syn = Synthesizer(store)
    syn._llm = MagicMock()
    syn._llm.complete = AsyncMock(
        return_value=(mock_json, 400, 120, "claude-3-5-haiku-20241022")
    )

    out = await syn.synthesize(_minimal_briefing())

    assert out.cached is False
    assert "NVDA" in out.paragraph or "8-K" in out.paragraph
    lowered = (out.paragraph + " ".join(out.key_points)).lower()
    assert "buy" not in lowered
    assert "sell" not in lowered
    assert "you should" not in lowered
    assert "recommend" not in lowered
    store.mark_alert_sent.assert_awaited()


@pytest.mark.asyncio
async def test_budget_exhausted_skips_llm() -> None:
    store = AsyncMock()
    store.get_alerts_sent_payload = AsyncMock(return_value=None)
    store.sum_llm_tokens_for_date = AsyncMock(return_value=100_000)

    syn = Synthesizer(store)
    out = await syn.synthesize(_minimal_briefing())

    assert "budget" in out.paragraph.lower()
    assert out.model_used == "none"


def test_sanitize_strips_trading_imperatives() -> None:
    t = _sanitize_text("You should buy this. I recommend selling. Hold tight.")
    assert not re.search(r"\b(buy|sell|hold|buying|selling)\b", t, re.I)


@pytest.mark.asyncio
async def test_filing_excerpt_summarize_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    store = AsyncMock()
    store.sum_llm_tokens_for_date = AsyncMock(return_value=0)
    store.record_llm_usage = AsyncMock()

    with patch.object(LLMService, "complete", new_callable=AsyncMock) as m:
        m.return_value = ("summary text", 10, 20, "claude-3-5-haiku-20241022")
        text = await filing_excerpt_summarize(store, "sys", "user")
    assert text == "summary text"
    m.assert_awaited()
    call_kw = m.call_args.kwargs
    assert call_kw.get("operation") == "filing_summary"
