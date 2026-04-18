"""
Post job failure embeds to Discord (ERROR_CHANNEL_ID, then ALERT_CHANNEL_ID fallback).

Used by scheduled tasks (with ``bot``) and CLI jobs (HTTP + DISCORD_TOKEN).
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime, timezone
from typing import Any

import discord
import httpx
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def build_job_failure_embed(job_name: str, exc: BaseException) -> discord.Embed:
    err_short = f"{type(exc).__name__}: {exc!s}"[:200]
    utc = datetime.now(timezone.utc)
    et = utc.astimezone(ET)
    tb_text = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    lines = tb_text.strip().splitlines()
    tail_lines = lines[-20:] if len(lines) > 20 else lines
    tail = "\n".join(tail_lines)
    if len(tail) > 1800:
        tail = tail[-1800:]

    embed = discord.Embed(
        title=f"⚠️ {job_name} failed",
        color=discord.Color.red(),
    )
    embed.add_field(name="Error", value=err_short[:1024], inline=False)
    embed.add_field(
        name="When",
        value=f"UTC: `{utc.isoformat()}`\nET: `{et.isoformat()}`",
        inline=False,
    )
    tb_block = f"```\n{tail}\n```"
    if len(tb_block) > 1024:
        tb_block = f"```\n{tail[:900]}…\n```"
    embed.add_field(name="Traceback", value=tb_block[:1024], inline=False)
    return embed


def _resolve_error_channel_id(bot: Any = None) -> int:
    eid = int(os.getenv("ERROR_CHANNEL_ID", "0") or "0")
    if eid:
        return eid
    eid = int(os.getenv("ALERT_CHANNEL_ID", "0") or "0")
    if eid:
        return eid
    if bot is not None:
        return int(getattr(bot, "alert_channel_id", 0) or 0)
    return 0


async def notify_job_failure(bot: Any, job_name: str, exc: BaseException) -> None:
    """Post failure embed using bot connection (scheduled tasks)."""
    from core.logging import get_logger

    log = get_logger("job_errors")
    cid = _resolve_error_channel_id(bot)
    embed = build_job_failure_embed(job_name, exc)
    if not cid:
        log.error("job_failed_no_channel", job=job_name, error=str(exc))
        print(f"[job failure] {job_name}: {exc}", flush=True)
        return
    try:
        ch = bot.get_channel(cid)
        if ch is not None and isinstance(ch, discord.abc.Messageable):
            await ch.send(embed=embed)
        else:
            log.error("job_failed_channel_missing", job=job_name, channel_id=cid)
            print(f"[job failure] {job_name}: {exc}", flush=True)
    except Exception as send_exc:
        log.error("job_failed_send_error", job=job_name, error=str(send_exc))
        print(f"[job failure] {job_name}: {exc}", flush=True)


async def notify_job_failure_http(job_name: str, exc: BaseException) -> None:
    """Post failure embed via Discord REST (CLI jobs, no bot object)."""
    from core.logging import get_logger

    log = get_logger("job_errors")
    token = (os.environ.get("DISCORD_TOKEN") or "").strip()
    cid = _resolve_error_channel_id(None)
    if not token or not cid:
        log.error("job_failed_no_token_or_channel", job=job_name, error=str(exc))
        print(f"[job failure] {job_name}: {exc}", file=__import__("sys").stderr, flush=True)
        return
    embed = build_job_failure_embed(job_name, exc)
    payload = {"embeds": [embed.to_dict()]}
    url = f"https://discord.com/api/v10/channels/{cid}/messages"
    headers = {"Authorization": f"Bot {token}"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
    except Exception as send_exc:
        log.error("job_failed_http_error", job=job_name, error=str(send_exc))
        print(f"[job failure] {job_name}: {exc}", file=__import__("sys").stderr, flush=True)
