"""
Direct SEC EDGAR access: CIK resolution, filing lists, document text, and summaries.

All HTTP uses ``SEC_USER_AGENT`` (required). All DB writes go through :class:`core.persistence.Store`.
"""

from __future__ import annotations

import asyncio
import html
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from core.persistence import FilingRecord, Store

SEC_DATA = "https://data.sec.gov"
SEC_WWW = "https://www.sec.gov"

TRACKED_FORMS = frozenset(
    {
        "10-K",
        "10-Q",
        "8-K",
        "4",
        "SC 13G",
        "SC 13D",
    }
)


def _cik_path_segment(cik: str) -> str:
    """Directory segment under /Archives/edgar/data/{segment}/ (no leading zeros)."""
    return str(int(cik))


def _accession_no_dashes(accession: str) -> str:
    return accession.replace("-", "")


def _retryable_httpx(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


class CompanyTickerEntry(BaseModel):
    cik_str: int
    ticker: str
    title: str


class CompanyTickersFile(BaseModel):
    """SEC ``company_tickers.json`` is a dict of string keys to ticker rows."""

    model_config = {"extra": "allow"}

    @classmethod
    def parse_rows(cls, raw: dict[str, Any]) -> list[CompanyTickerEntry]:
        rows: list[CompanyTickerEntry] = []
        for _k, v in raw.items():
            if isinstance(v, dict) and "ticker" in v:
                rows.append(CompanyTickerEntry.model_validate(v))
        return rows


class SubmissionsRecent(BaseModel):
    """Parallel arrays under ``filings.recent``."""

    accessionNumber: list[str] = Field(default_factory=list)
    filingDate: list[str] = Field(default_factory=list)
    form: list[str] = Field(default_factory=list)
    primaryDocument: list[str] = Field(default_factory=list)
    reportDate: list[str] = Field(default_factory=list)


class SubmissionsFilings(BaseModel):
    recent: SubmissionsRecent = Field(default_factory=SubmissionsRecent)


class SubmissionsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cik: str
    name: str
    filings: SubmissionsFilings = Field(default_factory=SubmissionsFilings)

    @field_validator("cik", mode="before")
    @classmethod
    def _cik_str(cls, v: Any) -> str:
        return str(v)


class Filing(BaseModel):
    """One filing parsed from SEC submissions (recent)."""

    accession_number: str
    form_type: str
    filed_at: datetime
    cik: str
    primary_document: Optional[str] = None
    report_date: Optional[str] = None

    @field_validator("filed_at", mode="before")
    @classmethod
    def _dt(cls, v: Any) -> Any:
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            # filingDate is YYYY-MM-DD
            return datetime.fromisoformat(v[:10]).replace(tzinfo=timezone.utc)
        return v


class SECClient:
    """
    Async EDGAR client: mandatory User-Agent, ≤10 req/sec effective pacing, retries on 429/5xx.

    Uses ``asyncio.Semaphore(10)`` to cap concurrent work and a global lock + 110ms spacing
    between outbound requests so bursts stay under SEC fair-access limits.
    """

    def __init__(
        self,
        *,
        user_agent: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        ua = (user_agent or os.environ.get("SEC_USER_AGENT") or "").strip()
        if not ua:
            raise ValueError(
                "SEC_USER_AGENT is required (e.g. 'Your Name you@domain.com'). "
                "Set it in the environment."
            )
        self._ua = ua
        self._sem = asyncio.Semaphore(10)
        self._pace_lock = asyncio.Lock()
        self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _pace(self) -> None:
        async with self._pace_lock:
            await asyncio.sleep(0.11)

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=1, max=45),
        retry=retry_if_exception(_retryable_httpx),
        reraise=True,
    )
    async def _get(
        self,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
    ) -> httpx.Response:
        async with self._sem:
            await self._pace()
            hdr = {"User-Agent": self._ua, "Accept-Encoding": "gzip, deflate"}
            if headers:
                hdr.update(headers)
            r = await self._client.get(url, headers=hdr)
            r.raise_for_status()
            return r

    async def resolve_cik(self, symbol: str, store: Store) -> str:
        """
        Resolve ticker → 10-digit CIK via ``company_tickers.json`` and cache in ``watchlist_meta``.
        """
        sym = symbol.strip().upper()
        r = await self._get(f"{SEC_WWW}/files/company_tickers.json")
        data = r.json()
        rows = CompanyTickersFile.parse_rows(data if isinstance(data, dict) else {})
        found: Optional[CompanyTickerEntry] = None
        for row in rows:
            if row.ticker.upper() == sym:
                found = row
                break
        if found is None:
            raise ValueError(f"No CIK found for ticker {sym!r}")

        cik10 = f"{found.cik_str:010d}"
        await store.set_watchlist_meta(
            sym, cik=cik10, company_name=found.title
        )
        return cik10

    async def fetch_recent_filings(
        self,
        cik: str,
        since: datetime,
    ) -> list[Filing]:
        """Recent filings from ``submissions`` API (forms in :data:`TRACKED_FORMS`)."""
        cik10 = f"{int(cik):010d}"
        url = f"{SEC_DATA}/submissions/CIK{cik10}.json"
        r = await self._get(url)
        body = SubmissionsResponse.model_validate(r.json())
        recent = body.filings.recent
        n = len(recent.accessionNumber)
        out: list[Filing] = []
        since_utc = since.astimezone(timezone.utc) if since.tzinfo else since.replace(
            tzinfo=timezone.utc
        )
        for i in range(n):
            form = recent.form[i] if i < len(recent.form) else ""
            if form not in TRACKED_FORMS:
                continue
            acc = recent.accessionNumber[i]
            fdate_s = recent.filingDate[i] if i < len(recent.filingDate) else ""
            filed = datetime.fromisoformat(fdate_s[:10]).replace(tzinfo=timezone.utc)
            if filed < since_utc:
                continue
            prim = (
                recent.primaryDocument[i]
                if i < len(recent.primaryDocument)
                else None
            )
            rep = (
                recent.reportDate[i] if i < len(recent.reportDate) else None
            )
            out.append(
                Filing(
                    accession_number=acc,
                    form_type=form,
                    filed_at=filed,
                    cik=cik10,
                    primary_document=prim,
                    report_date=rep,
                )
            )
        out.sort(key=lambda x: x.filed_at, reverse=True)
        return out

    def build_document_url(self, filing: Filing) -> str:
        """Primary document URL for a filing (requires ``primary_document``)."""
        if not filing.primary_document:
            raise ValueError("primary_document is required to build document URL")
        seg = _cik_path_segment(filing.cik)
        ad = _accession_no_dashes(filing.accession_number)
        return f"{SEC_WWW}/Archives/edgar/data/{seg}/{ad}/{filing.primary_document}"

    async def fetch_filing_text(
        self,
        accession_number: str,
        *,
        cik: str,
        primary_document: Optional[str] = None,
    ) -> str:
        """
        Fetch filing body text: HTML/XML stripped to plain text for LLM sizing.

        If ``primary_document`` is omitted, tries ``index.json`` to find an ``.htm`` file.
        """
        cik10 = f"{int(cik):010d}"
        seg = _cik_path_segment(cik10)
        ad = _accession_no_dashes(accession_number)
        base = f"{SEC_WWW}/Archives/edgar/data/{seg}/{ad}/"

        if primary_document:
            r = await self._get(base + primary_document)
        else:
            idx = await self._get(base + "index.json")
            doc = _pick_primary_from_index_json(idx.json(), accession_number)
            if not doc:
                raise ValueError(f"Could not resolve primary document for {accession_number}")
            r = await self._get(base + doc)

        raw = r.text
        return _html_to_text(raw)[:400_000]


def _pick_primary_from_index_json(data: Any, accession: str) -> Optional[str]:
    """Walk SEC index.json tree for a plausible main document."""

    def walk(node: Any) -> list[str]:
        names: list[str] = []
        if isinstance(node, dict):
            if "name" in node and isinstance(node["name"], str):
                names.append(node["name"])
            for v in node.values():
                names.extend(walk(v))
        elif isinstance(node, list):
            for it in node:
                names.extend(walk(it))
        return names

    names = walk(data)
    htm = [n for n in names if n.lower().endswith((".htm", ".html"))]
    if not htm:
        xml = [n for n in names if n.lower().endswith(".xml")]
        return xml[0] if xml else None
    # Prefer filename containing accession fragment
    ad_short = accession.replace("-", "")
    for n in htm:
        if ad_short[:8] in n.replace("-", ""):
            return n
    return htm[0]


def _html_to_text(s: str) -> str:
    s = re.sub(r"(?is)<script.*?>.*?</script>", " ", s)
    s = re.sub(r"(?is)<style.*?>.*?</style>", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _prompt_path(name: str) -> Path:
    return Path(__file__).resolve().parent / "prompts" / name


async def _summarize_with_llm(
    store: Store,
    *,
    system_prompt: str,
    user_content: str,
) -> str:
    """Delegate to :mod:`core.synthesis` for one LLM budget + provider surface."""
    from core.synthesis import filing_excerpt_summarize

    return await filing_excerpt_summarize(store, system_prompt, user_content)


def _xml_local(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def parse_form4_xml(xml_text: str) -> str:
    """
    Minimal Form 4 structured parse (no LLM). Best-effort insider / transaction lines.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return "Form 4: could not parse XML."

    names: list[str] = []
    for el in root.iter():
        ln = _xml_local(el.tag).lower()
        if ln == "rptownername" and el.text:
            names.append(el.text.strip())

    txs: list[str] = []
    for el in root.iter():
        ln = _xml_local(el.tag).lower()
        if "nonderivativetransaction" in ln or ln == "derivativetransaction":
            code = ""
            shares = ""
            price = ""
            for ch in el.iter():
                t = _xml_local(ch.tag).lower()
                if t == "transactioncode" and ch.text:
                    code = ch.text.strip()
                if t == "transactionshares" and ch.text:
                    shares = ch.text.strip()
                if t == "transactionpricepershare" and ch.text:
                    price = ch.text.strip()
            txs.append(
                f"code {code or '?'} | shares {shares or '?'} | price {price or 'n/a'}"
            )

    bullets: list[str] = []
    if names:
        bullets.append(f"Reporting owner(s): {', '.join(names[:3])}")
    if txs:
        bullets.append("Transactions (raw): " + "; ".join(txs[:3]))
    if not bullets:
        return "Form 4: XML present; no owner/transaction fields extracted (namespace may differ)."
    return "Form 4 (parsed):\n- " + "\n- ".join(bullets[:3])


async def ensure_filing_summaries(
    store: Store,
    client: SECClient,
    symbol: str,
    filings: list[Filing],
    *,
    max_haiku_summaries: int = 5,
) -> None:
    """
    For each filing, upsert metadata then summarize if ``filings.summary`` is empty.

    Skips re-summarization when summary is already cached. Form 4 uses XML parse only.
    Haiku calls are capped by ``max_haiku_summaries`` to avoid Discord timeouts.
    """
    sym = symbol.strip().upper()
    haiku_used = 0
    for filing in filings:
        try:
            doc_url = client.build_document_url(filing)
        except ValueError:
            doc_url = (
                f"{SEC_WWW}/cgi-bin/viewer?action=view&cik={filing.cik}"
                f"&accession_number={filing.accession_number}&xbrl_type=v"
            )

        period: Optional[Any] = None
        if filing.report_date:
            try:
                period = datetime.fromisoformat(filing.report_date[:10]).date()
            except ValueError:
                period = None

        await store.upsert_filing_row(
            accession_number=filing.accession_number,
            symbol=sym,
            cik=filing.cik,
            form_type=filing.form_type,
            filed_at=filing.filed_at,
            url=doc_url,
            period_of_report=period,
        )

        existing = await store.get_filing(filing.accession_number)
        if existing and existing.summary:
            continue

        try:
            text = await client.fetch_filing_text(
                filing.accession_number,
                cik=filing.cik,
                primary_document=filing.primary_document,
            )
        except Exception as exc:
            await store.cache_filing_summary(
                filing.accession_number,
                f"(Could not fetch filing text: {exc})",
                summary_model="error",
            )
            continue

        await store.upsert_filing_row(
            accession_number=filing.accession_number,
            symbol=sym,
            cik=filing.cik,
            form_type=filing.form_type,
            filed_at=filing.filed_at,
            url=doc_url,
            period_of_report=period,
            raw_text=text[:500_000],
        )

        if filing.form_type == "4":
            summary = parse_form4_xml(text)
            model_label = "form4-xml-parse"
        elif filing.form_type in ("10-K", "10-Q"):
            if haiku_used >= max_haiku_summaries:
                summary = "(Not summarized: Haiku budget exhausted for this run.)"
                model_label = "skipped-limit"
            else:
                haiku_used += 1
                body = _prompt_path("filing_10k_10q.md").read_text(encoding="utf-8")
                summary = await _summarize_with_llm(
                    store,
                    system_prompt=body,
                    user_content=f"Excerpt from {filing.form_type}:\n\n{text[:120_000]}",
                )
                model_label = "claude-haiku-10k10q"
        elif filing.form_type == "8-K":
            if haiku_used >= max_haiku_summaries:
                summary = "(Not summarized: Haiku budget exhausted for this run.)"
                model_label = "skipped-limit"
            else:
                haiku_used += 1
                body = _prompt_path("filing_8k.md").read_text(encoding="utf-8")
                summary = await _summarize_with_llm(
                    store,
                    system_prompt=body,
                    user_content=f"Excerpt from 8-K:\n\n{text[:120_000]}",
                )
                model_label = "claude-haiku-8k"
        else:
            summary = (
                f"{filing.form_type}: stored; automatic Haiku summary not enabled "
                "for this form in Phase 2."
            )
            model_label = "skipped"

        await store.cache_filing_summary(
            filing.accession_number,
            summary,
            summary_model=model_label,
        )
