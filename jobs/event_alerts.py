"""
Event-driven alerts (Phase 6) — cron-friendly entrypoint.

Runs :func:`core.alerts.run_event_alert_job` (NYSE calendar + ET session gating).
Posts to **ALERT_CHANNEL_ID** (same as bot automated alerts).

Usage::

    python -m jobs.event_alerts
    python -m jobs.event_alerts --dry-run

``--dry-run`` evaluates flows and prints candidate counts but does **not** post to Discord
or write ``alerts_sent``.

If ``ALERT_CHANNEL_ID`` is unset or 0, logs a warning and exits **0** (unless ``--dry-run``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from dotenv import load_dotenv

from core.alerts import run_event_alert_job
from core.logging import configure_structlog, get_logger
from core.persistence import Store, StoreConfigurationError
from core.stock_data import StockDataFetcher


async def _run(*, dry_run: bool) -> int:
    from bot.error_notifications import notify_job_failure_http

    load_dotenv()
    configure_structlog()
    log = get_logger("jobs.event_alerts")
    token = (os.environ.get("DISCORD_TOKEN") or "").strip()
    cid = int(os.environ.get("ALERT_CHANNEL_ID", "0") or "0")

    if dry_run:
        try:
            store = await Store.from_env()
        except StoreConfigurationError as exc:
            log.error("store_config_error", error=str(exc))
            return 1
        wl = os.environ.get("BRIEFING_WATCHLIST", "").strip()
        if wl:
            watchlist = [s.strip().upper() for s in wl.split(",") if s.strip()]
        else:
            watchlist = [
                "AAPL",
                "GOOGL",
                "MSFT",
                "TSLA",
                "NVDA",
                "AMD",
                "META",
                "AMZN",
            ]
        fetcher = StockDataFetcher(watchlist)
        try:
            result = await run_event_alert_job(
                store,
                fetcher,
                set(watchlist),
                channel_id=0,
                bot_token="",
                dry_run=True,
            )
            print(json.dumps(result, indent=2, default=str))
            return 0
        except Exception as exc:
            log.error("event_alerts_dry_run_failed", error=str(exc), exc_info=True)
            await notify_job_failure_http("jobs.event_alerts", exc)
            return 1

    if not cid:
        log.warning("alert_channel_missing_skipping")
        return 0
    if not token:
        log.error("discord_token_required")
        return 1

    try:
        store = await Store.from_env()
    except StoreConfigurationError as exc:
        log.error("store_config_error", error=str(exc))
        return 1

    wl = os.environ.get("BRIEFING_WATCHLIST", "").strip()
    if wl:
        watchlist = [s.strip().upper() for s in wl.split(",") if s.strip()]
    else:
        watchlist = [
            "AAPL",
            "GOOGL",
            "MSFT",
            "TSLA",
            "NVDA",
            "AMD",
            "META",
            "AMZN",
        ]

    fetcher = StockDataFetcher(watchlist)
    try:
        result = await run_event_alert_job(
            store,
            fetcher,
            set(watchlist),
            channel_id=cid,
            bot_token=token,
            dry_run=False,
        )
        log.info("event_alerts_ok", result=result)
        return 0
    except Exception as exc:
        log.error("event_alerts_failed", error=str(exc), exc_info=True)
        await notify_job_failure_http("jobs.event_alerts", exc)
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 6 event alert job.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute flows and candidates only; no Discord post or alerts_sent writes.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(dry_run=args.dry_run)))


if __name__ == "__main__":
    main()
