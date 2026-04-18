"""Compact Discord embeds for Phase 6 event alerts (shared ALERT_CHANNEL_ID with other automations)."""

from __future__ import annotations

import os
from typing import Any

import discord

from core.alerts import EventAlert


def _bell_title(title: str) -> str:
    t = title.strip()
    if t.startswith("🔔"):
        return t[:256]
    return f"🔔 {t}"[:256]


def _color_for_alert(alert: EventAlert) -> discord.Color:
    """Spec: 8-K dark teal, Form 4 gold, pre-market orange, intraday red/green, target green, stop red, macro blue."""
    at = alert.alert_type.lower()
    if at == "8k":
        return discord.Color(0x00796B)
    if at == "form4":
        return discord.Color.gold()
    if at == "premarket":
        return discord.Color.orange()
    if at == "intraday":
        d = (alert.intraday_direction or "").lower()
        if d == "down":
            return discord.Color.red()
        return discord.Color.green()
    if at == "target":
        return discord.Color.green()
    if at == "stop":
        return discord.Color.red()
    if at in ("macro", "macro_release"):
        return discord.Color.blue()
    if at == "batch":
        return discord.Color(0x546E7A)
    return discord.Color(0x607D8B)


def build_event_alert_embed(alert: EventAlert) -> discord.Embed:
    """Headline + short context; optional link. 🔔 prefix distinguishes from 📊 morning briefings."""
    embed = discord.Embed(
        title=_bell_title(alert.title),
        description=alert.body[:4096],
        url=alert.url,
        color=_color_for_alert(alert),
    )
    if alert.symbol:
        embed.set_footer(text=alert.symbol)
    return embed


def event_alert_to_payload_dict(alert: EventAlert) -> dict[str, Any]:
    """REST API embed JSON (for CLI / httpx), matching :func:`build_event_alert_embed`."""
    return build_event_alert_embed(alert).to_dict()


async def deliver_event_alert(bot: Any, alert: EventAlert) -> None:
    """
    Post one event alert using the bot connection — **ALERT_CHANNEL_ID** / ``bot.alert_channel_id`` only.
    """
    cid = int(getattr(bot, "alert_channel_id", 0) or 0) or int(
        os.getenv("ALERT_CHANNEL_ID", "0") or "0"
    )
    if not cid:
        raise RuntimeError("ALERT_CHANNEL_ID is not set or is 0")
    channel = bot.get_channel(cid)
    if channel is None:
        raise RuntimeError(f"Alert channel {cid} not found")
    embed = build_event_alert_embed(alert)
    await channel.send(embed=embed)
