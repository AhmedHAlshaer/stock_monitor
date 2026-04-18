"""SEC EDGAR client tests (respx mocks; no live SEC in CI)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from core.persistence import FilingRecord, Store
from core.sec_filings import (
    Filing,
    SECClient,
    SubmissionsResponse,
    ensure_filing_summaries,
    parse_form4_xml,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

COMPANY_TICKERS_NVDA = {
    "0": {
        "cik_str": 1045810,
        "ticker": "NVDA",
        "title": "NVIDIA CORP",
    }
}

SUBMISSIONS_NVDA = {
    "cik": "0001045810",
    "name": "NVIDIA CORP",
    "filings": {
        "recent": {
            "accessionNumber": ["0001045810-24-000065"],
            "filingDate": ["2024-02-21"],
            "form": ["8-K"],
            "primaryDocument": ["nvda-redacted-8k.htm"],
            "reportDate": [""],
        }
    },
}


@pytest.fixture
def sec_ua() -> str:
    return "Test User test@example.com"


def test_submissions_model_parses() -> None:
    m = SubmissionsResponse.model_validate(SUBMISSIONS_NVDA)
    assert m.filings.recent.form[0] == "8-K"


@pytest.mark.asyncio
@respx.mock
async def test_resolve_cik_nvda(sec_ua: str) -> None:
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=httpx.Response(200, json=COMPANY_TICKERS_NVDA)
    )
    store = AsyncMock(spec=Store)
    client = SECClient(user_agent=sec_ua)
    try:
        cik = await client.resolve_cik("NVDA", store)
    finally:
        await client.aclose()
    assert cik == "0001045810"
    store.set_watchlist_meta.assert_awaited()


@pytest.mark.asyncio
@respx.mock
async def test_fetch_recent_filings_filters_since(sec_ua: str) -> None:
    respx.get(
        "https://data.sec.gov/submissions/CIK0001045810.json",
    ).mock(return_value=httpx.Response(200, json=SUBMISSIONS_NVDA))

    client = SECClient(user_agent=sec_ua)
    try:
        filings = await client.fetch_recent_filings(
            "0001045810",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
    finally:
        await client.aclose()

    assert len(filings) == 1
    assert filings[0].form_type == "8-K"
    assert filings[0].accession_number == "0001045810-24-000065"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_filing_text_from_primary(sec_ua: str) -> None:
    html_body = "<html><body><p>Hello 8-K body</p></body></html>"
    respx.get(
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581024000065/nvda-redacted-8k.htm",
    ).mock(return_value=httpx.Response(200, text=html_body))

    client = SECClient(user_agent=sec_ua)
    try:
        text = await client.fetch_filing_text(
            "0001045810-24-000065",
            cik="0001045810",
            primary_document="nvda-redacted-8k.htm",
        )
    finally:
        await client.aclose()
    assert "Hello 8-K body" in text


def test_parse_form4_minimal_xml() -> None:
    xml = """<?xml version="1.0"?>
<ownershipDocument>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionCode>P</transactionCode>
      <transactionShares>100</transactionShares>
      <transactionPricePerShare>450.12</transactionPricePerShare>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>REDACTED OFFICER</rptOwnerName>
    </reportingOwnerId>
  </reportingOwner>
</ownershipDocument>"""
    s = parse_form4_xml(xml)
    assert "REDACTED OFFICER" in s
    assert "P" in s or "code" in s.lower()


def test_fixture_nvda_8k_file_exists() -> None:
    p = FIXTURE_DIR / "nvda_8k_redacted.htm"
    assert p.is_file()
    txt = p.read_text(encoding="utf-8")
    assert "8-K" in txt
    assert "REDACTED" in txt


@pytest.mark.asyncio
async def test_ensure_skips_when_summary_cached(sec_ua: str) -> None:
    """Second run does not call Haiku when summary already present."""
    f = Filing(
        accession_number="0001045810-24-000065",
        form_type="8-K",
        filed_at=datetime(2024, 2, 21, tzinfo=timezone.utc),
        cik="0001045810",
        primary_document="x.htm",
    )
    store = AsyncMock(spec=Store)
    store.get_watchlist_meta = AsyncMock(return_value=None)
    store.upsert_filing_row = AsyncMock()
    store.get_filing = AsyncMock(
        return_value=FilingRecord(
            accession_number=f.accession_number,
            symbol="NVDA",
            cik=f.cik,
            form_type=f.form_type,
            filed_at=f.filed_at,
            url="https://example.com/doc.htm",
            summary="Already summarized.",
            summary_model="cached",
            summary_at=datetime.now(timezone.utc),
            alert_sent=False,
        )
    )
    store.cache_filing_summary = AsyncMock()

    client = SECClient(user_agent=sec_ua)
    mock_fetch = AsyncMock(return_value="dummy")
    try:
        with patch.object(client, "fetch_filing_text", mock_fetch):
            await ensure_filing_summaries(store, client, "NVDA", [f], max_haiku_summaries=3)
    finally:
        await client.aclose()

    mock_fetch.assert_not_awaited()
    store.cache_filing_summary.assert_not_called()


def test_sec_client_requires_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(ValueError, match="SEC_USER_AGENT"):
        SECClient()
