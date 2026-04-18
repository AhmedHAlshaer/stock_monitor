#!/usr/bin/env python3
"""
Check that public API tables exist by querying PostgREST (same path the bot uses).

Requires SUPABASE_URL and SUPABASE_SERVICE_KEY in the environment (e.g. from .env).

Usage (from repo root):
  python scripts/verify_supabase_schema.py

Exit code 0 if all tables respond with HTTP 200; non-zero otherwise.

Apply migrations first (Supabase SQL editor or psql against DATABASE_URL):
  migrations/001_positions.sql … 007_llm_usage.sql in order.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

TABLES = (
    "positions",
    "filings",
    "macro_series",
    "alerts_sent",
    "watchlist_meta",
    "llm_usage",
)


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    url = (os.environ.get("SUPABASE_URL") or "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print(
            "error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set",
            file=sys.stderr,
        )
        return 2

    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    failed = False
    with httpx.Client(timeout=30.0) as client:
        for name in TABLES:
            r = client.get(f"{url}/rest/v1/{name}?select=*&limit=1", headers=headers)
            ok = r.status_code == 200
            status = "ok" if ok else "missing_or_error"
            print(f"{name}: {r.status_code} {status}")
            if not ok:
                failed = True
                snippet = (r.text or "")[:300].replace("\n", " ")
                print(f"  response: {snippet}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
