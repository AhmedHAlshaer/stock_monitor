"""
Fetch all configured FRED series and upsert into ``macro_series``.

Usage::

    python -m jobs.update_macro

Requires ``FRED_API_KEY`` and Supabase env vars (see ``core.persistence.Store``).
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

from core.logging import configure_structlog, get_logger
from core.macro import FREDClient, fetch_all_series_and_upsert
from core.persistence import Store


async def _run() -> int:
    from bot.error_notifications import notify_job_failure_http

    load_dotenv()
    configure_structlog()
    log = get_logger("jobs.update_macro")
    if not os.environ.get("FRED_API_KEY", "").strip():
        log.error("fred_api_key_missing")
        return 2
    try:
        client = FREDClient()
        store = await Store.from_env()
        try:
            counts = await fetch_all_series_and_upsert(store, client)
        finally:
            await client.aclose()
        for sid, n in counts.items():
            log.info("macro_series_upserted", series_id=sid, rows=n)
        return 0
    except Exception as exc:
        log.error("update_macro_failed", error=str(exc), exc_info=True)
        await notify_job_failure_http("jobs.update_macro", exc)
        return 1


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
