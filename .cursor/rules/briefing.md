```
Morning briefing rules:
- Briefing is assembled in core/briefing.py. Discord formatting in bot/ only.
- The synthesis paragraph is generated once per day, cached by input hash.
- If synthesis fails (LLM error), still post the structured sections. Degrade
  gracefully — never fail the whole briefing because of one subsystem.
- Every section (macro / filings / pre-market / positions) is independent.
  One failing does not block the others.
- Dedup: check alerts_sent before posting the briefing. reference_id is the
  date in YYYY-MM-DD format.
- The briefing never uses the words "buy" or "sell" outside of quoted
  analyst data ("Analysts rate META BUY").
```
