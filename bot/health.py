"""
HTTP health probe for Railway: ``GET /health`` on ``$PORT`` (default 8080).

Runs in-process alongside the Discord bot (aiohttp, same event loop).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

from aiohttp import web
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


async def health_handler(request: web.Request) -> web.Response:
    bot = request.app["bot"]
    errors: list[str] = []
    db_ok = False
    last_briefing_at = None
    last_hours: float | None = None

    store = getattr(bot, "store", None)
    if store is None:
        errors.append("store_not_configured")
    else:
        try:
            await asyncio.wait_for(store.ping_database(), timeout=5.0)
            db_ok = True
        except Exception as exc:
            errors.append(f"db:{exc!s}")

    if store is not None and db_ok:
        try:
            last_briefing_at = await asyncio.wait_for(
                store.latest_morning_briefing_sent_at(), timeout=5.0
            )
        except Exception as exc:
            errors.append(f"briefing_query:{exc!s}")

    now = datetime.now(timezone.utc)
    et_now = now.astimezone(ET)
    weekend = et_now.weekday() >= 5
    max_hours = 72.0 if weekend else 36.0

    briefing_ok = False
    if last_briefing_at is not None:
        lb = last_briefing_at
        if lb.tzinfo is None:
            lb = lb.replace(tzinfo=timezone.utc)
        delta = now - lb.astimezone(timezone.utc)
        last_hours = delta.total_seconds() / 3600.0
        briefing_ok = last_hours <= max_hours
    elif db_ok:
        errors.append("no_morning_briefing_record")

    overall_ok = db_ok and briefing_ok
    status = "ok" if overall_ok else "degraded"
    code = 200 if overall_ok else 503

    payload: dict[str, Any] = {
        "status": status,
        "db": db_ok,
        "last_briefing_hours_ago": round(last_hours, 2) if last_hours is not None else None,
        "last_briefing": last_briefing_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        if last_briefing_at
        else None,
        "briefing_window_hours": max_hours,
        "errors": errors,
    }
    return web.json_response(payload, status=code)


async def start_health_server(bot: Any) -> web.AppRunner:
    """Bind aiohttp on ``0.0.0.0:$PORT`` and register ``GET /health``."""
    app = web.Application()
    app["bot"] = bot
    app.router.add_get("/health", health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", "8080"))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    return runner
