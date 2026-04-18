# Stock Monitor Bot — Enhancement Guide

> **Status:** Enhancing existing `stock_monitor` Discord bot, not starting from scratch.
> **Owner:** Ahmed
> **Goal:** Extend the current signal-generation bot with macro context, direct SEC filings, persistence, position tracking, and a dedicated morning briefing — without breaking what already works.

---

## 1. What We Already Have (Preserve)

The existing bot (`github.com/AhmedHAlshaer/stock_monitor`) already implements:

- **Discord bot** with slash commands: `/signal`, `/check`, `/analyze`, `/news`, `/scan`, `/summary`, `/watchlist`, `/setchannel`.
- **ML ensemble** (`core/ml_models.py`): LSTM + XGBoost + LightGBM with 70+ engineered features.
- **DeepSeek sentiment** (`core/news_sentiment.py`): LLM-powered news analysis with keyword fallback.
- **yfinance data layer** (`core/stock_data.py`): prices, 52-week ranges, analyst ratings, insider activity, earnings history.
- **Historical patterns** (`core/historical_patterns.py`): earnings beat/miss streaks, seasonality.
- **Ridge regression** (`core/regression.py`): Mon-Thu trend prediction.
- **Ultimate signal** (`core/ultimate_signal.py`): weighted 7-factor combined score.
- **Chart generation** (`core/visualizer.py`): dark-themed matplotlib charts.
- **Scheduled tasks**: 4 PM daily 52-week scan, Monday 9 AM weekly analysis.

**Do not rewrite any of this.** Everything below extends it.

---

## 2. Gap Analysis — What's Missing

| Gap | Impact | Priority |
|---|---|---|
| **No FRED macro data** | No regime awareness (yields, VIX, CPI, dollar) | High |
| **No direct SEC EDGAR** | No 8-K alerts, no 10-K/Q summaries, insider data is secondhand | High |
| **No persistence layer** | Can't dedupe alerts, no "since yesterday" diffing, no position tracking, LLM re-summarizes same filings | **Critical** |
| **No morning briefing** | Scheduled tasks are after-close, fragmented | High |
| **No position tracking** | Watchlist only, no P&L, no sizing context | Medium |
| **No event-driven alerts** | Everything is time-scheduled; 8-Ks wait until next scan | Medium |
| **No cross-reference synthesis** | Weighted scoring ≠ narrative "these three signals rhyme" | Medium |
| **No Finnhub** | Stuck with yfinance (fragile; breaks when yfinance changes) | Low |

---

## 3. Design Principle — Two Modes

The bot will operate in two complementary modes:

**Mode A — On-demand analysis (keep as-is):**
- `/signal TICKER` → full ML + sentiment + analyst deep-dive, outputs BUY/HOLD/SELL.
- User-invoked. User takes responsibility for acting on it.
- Unchanged from today.

**Mode B — Daily morning briefing (NEW):**
- Pushed automatically at 9:00 AM ET, Mon–Fri.
- **Informational**, not prescriptive. Surfaces what changed, doesn't say "buy X."
- Content: new filings, macro prints, overnight news summaries, position P&L, pre-market moves, upcoming earnings.
- Optionally includes a synthesis paragraph ("three items worth your attention today") via Claude/DeepSeek.

This separation is deliberate. Signal generation and information surfacing are different products. Keep them in different surfaces.

---

## 4. Target Architecture (after enhancements)

```
stock_monitor/
├── bot/
│   └── discord_bot.py              # extended with /briefing, /position, /pnl commands
├── core/                           # existing — preserved
│   ├── ml_models.py                # no change
│   ├── news_sentiment.py           # minor: cache summaries to DB
│   ├── historical_patterns.py      # no change
│   ├── regression.py               # no change
│   ├── stock_data.py               # no change
│   ├── ultimate_signal.py          # minor: read from DB where possible
│   ├── visualizer.py               # no change
│   │
│   │ ─── NEW modules ───
│   ├── persistence.py              # Supabase client wrapper
│   ├── macro.py                    # FRED client + macro diff logic
│   ├── sec_filings.py              # EDGAR client: 8-K, Form 4, 10-K/Q
│   ├── finnhub_data.py             # Finnhub client (earnings cal, news)
│   ├── positions.py                # Position tracking + P&L calc
│   ├── briefing.py                 # Morning briefing assembler
│   └── synthesis.py                # LLM cross-reference layer (Haiku/DeepSeek)
├── jobs/                           # NEW — long-running job entrypoints
│   ├── __init__.py
│   ├── morning_briefing.py         # 9 AM ET job
│   └── event_alerts.py             # 15-min polling during market hours
├── migrations/                     # NEW — Supabase SQL
│   ├── 001_init.sql
│   ├── 002_positions.sql
│   ├── 003_alerts_dedup.sql
│   └── 004_filings_cache.sql
├── tests/                          # NEW — currently no tests
│   └── fixtures/
├── .cursor/
│   └── rules/                      # NEW — project coding rules
└── requirements.txt                # add: supabase, httpx, tenacity, pandas-market-calendars
```

---

## 5. Data Model (Supabase — NEW)

Add these tables. Keep it minimal.

### `positions`
User holdings, manually entered.
```sql
CREATE TABLE positions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  symbol text NOT NULL,
  quantity numeric NOT NULL,
  cost_basis numeric NOT NULL,   -- avg cost per share
  opened_at timestamptz NOT NULL,
  closed_at timestamptz,
  notes text
);
CREATE INDEX ON positions(symbol) WHERE closed_at IS NULL;
```

### `filings`
Cache of SEC filings + LLM summaries (so we never re-summarize the same filing).
```sql
CREATE TABLE filings (
  accession_number text PRIMARY KEY,
  symbol text NOT NULL,
  cik text NOT NULL,
  form_type text NOT NULL,         -- '10-K', '10-Q', '8-K', '4', etc.
  filed_at timestamptz NOT NULL,
  period_of_report date,
  url text NOT NULL,
  raw_text text,                   -- optional, may be large
  summary text,                    -- LLM-generated
  summary_model text,
  summary_at timestamptz,
  alert_sent boolean DEFAULT false
);
CREATE INDEX ON filings(symbol, filed_at DESC);
CREATE INDEX ON filings(form_type, filed_at DESC) WHERE alert_sent = false;
```

### `macro_series`
FRED observations, append-only.
```sql
CREATE TABLE macro_series (
  series_id text NOT NULL,         -- 'DGS10', 'VIXCLS', 'CPIAUCSL', etc.
  date date NOT NULL,
  value numeric,
  fetched_at timestamptz DEFAULT now(),
  PRIMARY KEY (series_id, date)
);
```

### `alerts_sent`
Dedup for alerts — morning briefings, 8-K alerts, etc.
```sql
CREATE TABLE alerts_sent (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  alert_type text NOT NULL,        -- 'morning_briefing', 'filing', 'price_gap'
  reference_id text NOT NULL,      -- filing accession OR date OR symbol+date
  sent_at timestamptz DEFAULT now(),
  channel_id text,
  payload jsonb
);
CREATE UNIQUE INDEX ON alerts_sent(alert_type, reference_id);
```

### `watchlist_meta`
Optional — CIK mapping so we only look it up once per ticker.
```sql
CREATE TABLE watchlist_meta (
  symbol text PRIMARY KEY,
  cik text,
  company_name text,
  bucket text,                     -- 'core', 'quant', 'hedge', freeform
  updated_at timestamptz DEFAULT now()
);
```

---

## 6. New Modules — Detailed Specs

### `core/persistence.py`
Thin wrapper around `supabase-py`. Exposes a `Store` class with methods like:
- `get_positions()`, `add_position()`, `close_position()`
- `cache_filing_summary(accession, summary)`, `get_filing_summary(accession)`
- `upsert_macro(series_id, date, value)`, `get_macro_latest(series_id, n=10)`
- `has_alert_been_sent(alert_type, ref_id)`, `mark_alert_sent(...)`
- `get_watchlist_meta(symbol)`, `set_watchlist_meta(...)`

Implementation uses the **async** Supabase client (`create_async_client`); all of the above are `async` methods and must be awaited. Construct with `await Store.create(...)` or `await Store.from_env()`. Quantities, cost basis, and macro values are serialized as **strings** for PostgreSQL `numeric` columns (no `float` on write).

Environment: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`.

### `core/macro.py`
Fetches and analyzes FRED data.

- Auth: `FRED_API_KEY` env var, free at https://fred.stlouisfed.org/docs/api/api_key.html.
- Daily pull for these series:
  - `DGS10` — 10Y Treasury yield
  - `DGS2` — 2Y Treasury yield
  - `DFF` — Fed funds rate
  - `VIXCLS` — VIX
  - `DTWEXBGS` — Dollar index
  - `CPIAUCSL` — CPI (monthly, flag on release)
  - `UNRATE` — Unemployment (monthly)
  - `PAYEMS` — Nonfarm payrolls (monthly)
- Writes to `macro_series` table.
- `MacroAnalyzer.get_changes()` returns a list of notable changes (e.g., "10Y yield -7bp to 4.21%, 3rd day of decline").
- Uses `httpx.AsyncClient` with `tenacity` retry.

### `core/sec_filings.py`
Direct SEC EDGAR client. **Not** yfinance-based.

- No auth, but **MUST** send `User-Agent: "Ahmed Alshaer your@email.com"` (env: `SEC_USER_AGENT`).
- Rate limit: 10 req/sec. Enforce via `asyncio.Semaphore`.
- Entrypoint: `SECClient.fetch_recent_filings(cik, since=datetime)` → list of filings.
- CIK lookup: `SECClient.resolve_cik(symbol)` → cache result in `watchlist_meta`.
- Filing types to track: `10-K`, `10-Q`, `8-K`, `4`, `SC 13G/D`.
- For filings already in `filings` table with a non-null `summary`, return cached.
- For new filings, write raw metadata immediately; summarization happens in a separate step.

### `core/finnhub_data.py` (optional, phase 3)
Wraps Finnhub for what yfinance does poorly:
- Earnings calendar (`/calendar/earnings`) — more reliable than yfinance's calendar.
- Pre-market quote (`/quote`) — yfinance struggles with pre-market.
- Keep it optional; gate behind `FINNHUB_API_KEY` env var presence.

### `core/positions.py`
Position math. No external APIs here — reads from `positions` table and current prices (via existing `stock_data.py`).

- `compute_pnl(positions, current_prices) -> list[dict]` → symbol, qty, cost, current, unrealized $, unrealized %.
- `total_portfolio_value(positions, current_prices) -> float`.
- `day_change(positions, prev_close, current) -> float` for the briefing.

### `core/synthesis.py`
The cross-reference layer. Takes structured briefing data in, returns a short narrative paragraph.

- Uses Anthropic Claude Haiku 4.5 (cheap) or DeepSeek (cheaper).
- Prompt lives in `core/prompts/synthesis.md` — version controlled.
- Hard constraints in the prompt:
  - Max 3 bullet points.
  - No buy/sell recommendations.
  - Must reference sources ("per 8-K filed this morning").
  - If nothing is genuinely cross-referenceable, output "Nothing requires action today."
- Cache result keyed by hash(briefing_data) to avoid duplicate spend on replay.

### `core/briefing.py`
Orchestrator for the morning briefing. Runs in order:

1. Load watchlist + positions from DB.
2. Pull overnight data via async gather: macro, filings since last run, news, pre-market prices.
3. Persist new records.
4. Compute position P&L.
5. Summarize any new 10-K/10-Q/8-K via Haiku (writes back to `filings.summary`).
6. Call synthesis layer with structured output of the above.
7. Assemble Discord embed via existing embed patterns.
8. Post to channel, mark alert as sent.

Returns a structured `BriefingReport` dataclass for testability.

---

## 7. Modifications to Existing Modules

Keep changes minimal and backward-compatible.

### `bot/discord_bot.py`
- Refactor scheduled tasks into proper Cog instead of the bottom-of-file class-bolt pattern.
- Add new slash commands:
  - `/briefing` — manually trigger today's briefing.
  - `/position add SYMBOL QTY COST` — add holding.
  - `/position list` — show open positions with P&L.
  - `/position close SYMBOL` — close a position.
  - `/pnl` — portfolio summary.
- Wire up the new `core/briefing.py` to a 9 AM ET daily task.
- Add 15-min polling task for event alerts (during market hours only).

### `core/news_sentiment.py`
- Add DB caching: `get_cached_sentiment(ticker, date)` before calling DeepSeek.
- No other changes — the module is solid.

### `core/ultimate_signal.py`
- No behavior change required. Optionally pass a persisted `macro_context` into scoring later.

### `requirements.txt`
Add:
```
supabase>=2.0.0
httpx>=0.25.0
tenacity>=8.2.0
pandas-market-calendars>=4.3.0
structlog>=23.0.0
anthropic>=0.18.0          # optional if using Claude for synthesis
pydantic>=2.0.0
pydantic-settings>=2.0.0
```

---

## 8. Morning Briefing Format (Discord Embed)

Same visual style as existing bot embeds. Structure:

```
📊 Morning Briefing — Mon, Apr 20 2026

🌍 MACRO
• 10Y yield -7bp to 4.21% (3rd day of decline)
• VIX unchanged at 14.2
• No major releases today | CPI Wednesday 8:30 AM

📁 FILINGS (last 24h)
• META 8-K — new $50B buyback announced. [summary] [link]
• NVDA Form 4 — CFO sold $7.2M (10b5-1 pre-planned). [link]

📈 PRE-MARKET
• WULF +4.2% (hosting deal news)
• IONQ -2.1% (no catalyst)

💼 POSITIONS
NVDA: 10 sh @ $450 → $481 | +6.9% | +$310
META: 5 sh @ $520 → $538 | +3.5% | +$90
Portfolio: $7,242 | +$400 day

🧠 SYNTHESIS
• META buyback is ~3% of float — watch opening strength.
• NVDA insider sell is scheduled, not informative alone; worth
  monitoring with last week's soft Taiwan semi data.
• No other items require attention today.

⏰ Next earnings: AMD (Apr 23), AMZN (Apr 28)
```

Use the existing Discord embed patterns in `discord_bot.py` — this is just a new, longer embed type.

---

## 9. Event-Driven Alerts (separate job)

Runs every 15 minutes during market hours (9:30 AM – 4:00 PM ET).

Triggers:
- New 8-K on watchlist → summarize with Haiku, post embed within ~15 min of filing.
- New Form 4 with transaction > $1M → post alert.
- Pre-market gap >3% on a watchlist symbol → post alert (pre-market only, 7:30–9:30 AM).
- FRED release days (CPI, FOMC, NFP) → post the print as soon as the observation appears.

Every alert checks `alerts_sent` first for dedup. All alerts write to `alerts_sent` after successful post.

Separate GitHub Actions workflow (or Railway worker) runs this — do NOT bundle into the morning briefing job.

---

## 10. Implementation Phases

**Phase ordering matters. Do not skip.**

### Phase 0 — Foundations (Supabase + project hygiene) — 1–2 days
- [ ] Supabase project, migrations 001–004 applied.
- [ ] `core/persistence.py` with unit tests.
- [ ] Add new env vars to `.env.example`.
- [ ] `.cursor/rules/` seeded (see section 11).
- [ ] Test: async `Store` works end-to-end, e.g. `store = await Store.from_env(); await store.add_position(...)` (see `core/persistence.py`).

### Phase 1 — Macro integration — 1–2 days
- [ ] `core/macro.py` implemented with all 8 FRED series.
- [ ] Daily upsert job for FRED observations.
- [ ] `MacroAnalyzer.get_changes()` returns usable summaries.
- [ ] Add `/macro` slash command for manual testing.
- [ ] Test: run once, verify `macro_series` table populated.

### Phase 2 — SEC EDGAR — 2–3 days
- [ ] `core/sec_filings.py` with rate-limited httpx client.
- [ ] CIK resolution + caching into `watchlist_meta`.
- [ ] Fetch recent filings for each watchlist symbol.
- [ ] Haiku-based summarizer for 8-K/10-Q (prompt in `core/prompts/filing_summary.md`).
- [ ] Cache summaries in `filings.summary` — never re-summarize.
- [ ] Add `/filings TICKER` slash command.
- [ ] Test: pull NVDA filings, verify summaries written, verify cache hit on re-run.

### Phase 3 — Position tracking — 1 day
- [ ] `core/positions.py` implemented.
- [ ] `/position add/list/close` slash commands.
- [ ] `/pnl` command shows current portfolio state.
- [ ] Test: add a position, verify P&L calculation matches manually.

### Phase 4 — Morning briefing — 2–3 days
- [ ] `core/briefing.py` orchestrator.
- [ ] `jobs/morning_briefing.py` entry point.
- [ ] Refactor existing scheduled tasks into Cog-based structure.
- [ ] New 9 AM ET daily task fires the briefing.
- [ ] Dedup via `alerts_sent`.
- [ ] Test: run manually via `/briefing`, verify full embed posts.

### Phase 5 — LLM synthesis — 1–2 days
- [ ] `core/synthesis.py` with prompt in `core/prompts/synthesis.md`.
- [ ] Integrate into `briefing.py` as final step before posting.
- [ ] Cache results by input hash.
- [ ] Test: run same briefing twice, verify second run hits cache.

### Phase 6 — Event alerts — 2 days
- [ ] `jobs/event_alerts.py` runs every 15 min during market hours.
- [ ] 8-K alert flow.
- [ ] Form 4 alert flow (>$1M transactions).
- [ ] Pre-market gap detection using Finnhub quote endpoint.
- [ ] Test: simulate a filing (insert into DB with `alert_sent=false`), verify alert posts.

### Phase 7 — Polish (ongoing)
- [ ] Finnhub integration for better earnings calendar (optional).
- [ ] Error channel — all exceptions post to a separate Discord channel.
- [ ] Weekly summary (Sunday evening).
- [ ] Backtesting harness (stretch; not worth building until v1 proves valuable).

---

## 11. Environment Variables (additions)

Append to existing `.env.example`:

```bash
# NEW — Persistence
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_KEY=eyJ...

# NEW — Data APIs
FRED_API_KEY=...
FINNHUB_API_KEY=...                         # optional, phase 3+
SEC_USER_AGENT="Ahmed Alshaer your@email.com"   # REQUIRED by SEC

# NEW — Optional Claude for synthesis (else DeepSeek is used)
ANTHROPIC_API_KEY=sk-ant-...

# NEW — Error handling
ERROR_CHANNEL_ID=...

# NEW — Timezone
BRIEFING_TZ=America/New_York
```

---

## 12. Cursor Rules

Create `.cursor/rules/` with these files. Ahmed already uses cursor rules heavily from the Hipoink project — same style applies.

### `.cursor/rules/project.md`
```
Project: stock_monitor (enhanced)

## Core principles
- This is an enhancement of existing code, NOT a rewrite.
- Preserve all existing modules in core/ and bot/. Modify only when spec calls for it.
- The bot is an information tool. /signal is the user-invoked analysis. Morning
  briefing is informational only — never generate BUY/SELL in the briefing.
- No auto-trading, no brokerage integration, ever.
- All state goes through core/persistence.py. Do not touch Supabase directly
  from other modules.

## Coding conventions
- Python 3.10+ (existing), type hints on new code.
- Existing files: match their dataclass + docstring style. Do not impose new style.
- New files: Pydantic v2 for data boundaries, dataclasses fine for internal use.
- Async httpx + tenacity for all external HTTP. No sync `requests` anywhere.
- Every external call must respect rate limits documented in the source module.
- Store timestamps in UTC. Display in ET at the Discord layer only.

## Testing
- New modules get a tests/ file with fixture-based tests.
- Existing modules are not required to gain tests retroactively.
- Use respx to mock httpx calls.
```

### `.cursor/rules/new-modules.md`
```
When writing NEW modules in core/:
- Start with Pydantic models for all inbound external data.
- Define a clear class (e.g., MacroAnalyzer, SECClient, PositionTracker).
- Expose a narrow public API; make helpers private (_leading_underscore).
- Every class gets a docstring explaining what it does and what it doesn't.
- If the module calls an LLM, the prompt goes in core/prompts/<name>.md, not inline.
- If the module writes to Supabase, it goes through core.persistence.Store,
  never via the supabase client directly.
```

### `.cursor/rules/briefing.md`
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

### `.cursor/rules/sec-edgar.md`
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

---

## 13. Operational Notes

### Timezones
- Morning briefing fires at 9:00 AM ET (before market open at 9:30).
- If running on GitHub Actions, cron in UTC:
  - Winter: `0 14 * * 1-5`
  - Summer: `0 13 * * 1-5`
  - Simpler: run at both, skip if not 9:00 ET (cheap defensive check), or use `pandas_market_calendars` to determine.
- Skip NYSE holidays (`pandas_market_calendars`).

### Hosting
- Current setup: presumably runs locally or on a dev machine.
- Recommended: migrate to Railway ($5/mo hobby) or keep GitHub Actions + a small Railway instance for event alerts.
- Discord bot requires always-on process (websocket connection). GitHub Actions alone is not sufficient for the bot itself — only for periodic jobs. If the bot currently runs on your machine, Railway is the right next step.

### Cost controls
- Daily token budget cap in code. Fall back to DeepSeek or skip synthesis if exceeded.
- Filing summaries are cached forever in `filings.summary`.
- FRED and EDGAR are both free — no concern there.
- Finnhub free tier is 60 calls/min, plenty.

### ML model persistence (existing bug / improvement)
Current `ultimate_signal.py` retrains ML models per-ticker on each call. This is slow and wastes compute. Consider:
- Serialize trained models to disk (joblib for XGB/LGB, native `.keras` for LSTM).
- Key by (ticker, training_date). Retrain weekly, not per-request.
- Not in scope for the enhancement phases above, but worth noting for phase 7.

---

## 14. What NOT To Do

- **Do not rewrite `core/ml_models.py`.** It's working. Leave it.
- **Do not replace yfinance wholesale.** yfinance is fine for prices and analyst data. Only swap for something better when the upgrade is concrete (EDGAR for filings, Finnhub for earnings calendar).
- **Do not add buy/sell recommendations to the morning briefing.** That's what `/signal` is for.
- **Do not build a backtesting framework before the core is solid.** It's a rabbit hole.
- **Do not add more ML models.** 3 is already enough. Improvements come from better features and persistence, not more models.
- **Do not auto-execute trades.** This is not a trading bot. Information tool only.

---

## 15. Open Questions to Resolve Before Coding

1. **Hosting** — is the bot currently running locally? If so, Railway deployment is a Phase 0 task.
2. **QBTS confirmation** — added to watchlist? (Grouped with IONQ, RGTI per prior discussion.)
3. **Claude vs DeepSeek for synthesis** — both work. Claude Haiku ~10x cheaper than Sonnet and fine for this. DeepSeek is cheaper still.
4. **Position tracking UX** — slash commands are clean. Stick with that. Direct Supabase row edits also work for power-user use.
5. **Event alert channel** — same Discord channel as morning briefing, or separate? Recommend separate to keep the briefing channel clean.

---

*This guide extends what's already working. Preserve the good, fill the gaps, resist the urge to rebuild.*
