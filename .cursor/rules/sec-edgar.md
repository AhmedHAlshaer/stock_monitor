```
SEC EDGAR client rules:
- User-Agent from SEC_USER_AGENT env var is MANDATORY on every request.
  Without it, SEC will rate-limit or ban.
- Max 10 req/sec. Enforce with asyncio.Semaphore(10).
- All filings write to the filings table via persistence.Store.
- Never re-summarize a filing if filings.summary is non-null for that accession.
- 10-K / 10-Q: summarize with Haiku, prompt in core/prompts/filing_summary.md.
- 8-K: summarize with Haiku, different prompt core/prompts/8k_summary.md.
- Form 4: parse structured XML, no LLM needed. Extract insider name, transaction
  type, shares, value.
```
