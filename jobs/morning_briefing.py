"""
Standalone morning briefing for cron / CI (posts via Discord REST API).

Usage::

    python -m jobs.morning_briefing
    python -m jobs.morning_briefing --dry-run

``--dry-run`` runs the full briefing pipeline against your DB (live Supabase data),
prints the assembled embed JSON to stdout, and does **not** post to Discord or
write ``alerts_sent``.

Requires ``DISCORD_TOKEN``, ``SUPABASE_*``, channel ``BRIEFING_CHANNEL_ID`` or
``ALERT_CHANNEL_ID``, optional ``BRIEFING_WATCHLIST`` (comma-separated tickers).

Dedup: skips if ``alerts_sent`` already has ``morning_briefing`` + today's date
(non-dry-run only).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date

import discord
import httpx
from dotenv import load_dotenv

from bot.embeds.briefing import build_briefing_embed
from core.briefing import BriefingAssembler
from core.logging import configure_structlog, get_logger
from core.persistence import Store
from core.stock_data import StockDataFetcher
from core.synthesis import Synthesizer


async def _post_embed_to_channel(embed: discord.Embed) -> None:
    token = (os.environ.get("DISCORD_TOKEN") or "").strip()
    cid = int(os.environ.get("BRIEFING_CHANNEL_ID") or os.environ.get("ALERT_CHANNEL_ID") or "0")
    if not token or not cid:
        raise RuntimeError("DISCORD_TOKEN and BRIEFING_CHANNEL_ID or ALERT_CHANNEL_ID are required")

    payload = {"embeds": [embed.to_dict()]}
    url = f"https://discord.com/api/v10/channels/{cid}/messages"
    headers = {"Authorization": f"Bot {token}"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()


async def _run(*, dry_run: bool = False) -> int:
    from bot.error_notifications import notify_job_failure_http

    load_dotenv()
    configure_structlog()
    log = get_logger("jobs.morning_briefing")
    ref = date.today().isoformat()

    try:
        store = await Store.from_env()
        if not dry_run:
            if await store.has_alert_been_sent("morning_briefing", ref):
                log.info("skip_already_sent", reference=ref)
                return 0

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
        assembler = BriefingAssembler(
            store, fetcher, watchlist, synthesizer=Synthesizer(store)
        )
        report = await assembler.build(date.today())
        embed = build_briefing_embed(report)

        if dry_run:
            payload = embed.to_dict()
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0

        await _post_embed_to_channel(embed)
        cid = os.environ.get("BRIEFING_CHANNEL_ID") or os.environ.get("ALERT_CHANNEL_ID") or ""
        await store.mark_alert_sent("morning_briefing", ref, channel_id=str(cid))
        log.info("morning_briefing_posted_ok", reference=ref)
        return 0
    except Exception as exc:
        log.error("morning_briefing_failed", error=str(exc), exc_info=True)
        await notify_job_failure_http("jobs.morning_briefing", exc)
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Post morning briefing to Discord (or dry-run).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build from DB, print embed JSON to stdout; do not post or write alerts_sent.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(dry_run=args.dry_run)))


if __name__ == "__main__":
    main()
