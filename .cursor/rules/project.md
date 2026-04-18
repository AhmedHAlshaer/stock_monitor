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
- `core.persistence.Store` is async (Supabase `create_async_client`); await all `Store` methods.
- Every external call must respect rate limits documented in the source module.
- Store timestamps in UTC. Display in ET at the Discord layer only.

## Testing
- New modules get a tests/ file with fixture-based tests.
- Existing modules are not required to gain tests retroactively.
- Use respx to mock httpx calls.
```
