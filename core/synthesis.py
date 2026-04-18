"""
Cross-reference briefing synthesis and shared async LLM completion (Haiku / DeepSeek).

Filing summaries in :mod:`core.sec_filings` delegate here for one place to track usage.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from core.persistence import Store

SYNTHESIS_CACHE_ALERT = "briefing_synthesis"
DEFAULT_DAILY_BUDGET = 50_000
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_HAIKU_MODEL = os.environ.get(
    "ANTHROPIC_HAIKU_MODEL", "claude-3-5-haiku-20241022"
)


class LLMBudgetExceeded(Exception):
    """Raised when the daily token budget would be exceeded."""


def _synthesis_prompt_path() -> Path:
    return Path(__file__).resolve().parent / "prompts" / "synthesis.md"


def _retryable_llm(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def briefing_to_fingerprint_dict(briefing: Any) -> dict[str, Any]:
    """Serialize briefing for hashing (excludes synthesis)."""
    pos = None
    if briefing.position_pnl is not None:
        pos = briefing.position_pnl.model_dump(mode="json")
    return {
        "date": briefing.date.isoformat(),
        "macro_changes": briefing.macro_changes,
        "new_filings": briefing.new_filings,
        "news_items": briefing.news_items,
        "price_moves": briefing.price_moves,
        "position_pnl": pos,
        "upcoming_earnings": briefing.upcoming_earnings,
        "section_errors": briefing.section_errors,
    }


def briefing_cache_reference_id(briefing: Any) -> str:
    """``YYYY-MM-DD:sha256`` for alerts_sent dedup + cache lookup."""
    canonical = json.dumps(
        briefing_to_fingerprint_dict(briefing),
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{briefing.date.isoformat()}:{h}"


def _daily_budget_tokens() -> int:
    raw = os.environ.get("DAILY_LLM_BUDGET_TOKENS", str(DEFAULT_DAILY_BUDGET)).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_DAILY_BUDGET


def _sanitize_text(text: str) -> str:
    """Remove disallowed trading-imperative phrasing (best-effort)."""
    t = text
    t = re.sub(
        r"\b(you should|i recommend|i would recommend|consider buying|consider selling)\b",
        "[—]",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"\b(buy|buying|bought|sell|selling|sold|hold|holding)\b",
        "[—]",
        t,
        flags=re.IGNORECASE,
    )
    return t


def _truncate_words(text: str, max_words: int = 150) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


@dataclass
class SynthesisResult:
    paragraph: str
    key_points: list[str]
    model_used: str
    cached: bool


class LLMService:
    """Async LLM calls with daily token budget (via Store) and usage logging."""

    def __init__(self, store: Optional[Store] = None) -> None:
        self._store = store

    async def _budget_remaining(self) -> int:
        budget = _daily_budget_tokens()
        if self._store is None or budget <= 0:
            return budget
        used = await self._store.sum_llm_tokens_for_date(date.today())
        return max(0, budget - used)

    async def _ensure_budget_headroom(self, estimated: int) -> None:
        rem = await self._budget_remaining()
        if rem <= 0:
            raise LLMBudgetExceeded("daily LLM token budget exhausted")
        if estimated > rem:
            raise LLMBudgetExceeded(
                f"request needs ~{estimated} tokens but only {rem} remain today"
            )

    async def _record_usage(
        self, operation: str, input_tokens: int, output_tokens: int
    ) -> None:
        if self._store is None:
            return
        await self._store.record_llm_usage(
            usage_date=date.today(),
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception(_retryable_llm),
        reraise=True,
    )
    async def complete_anthropic(
        self,
        *,
        system_prompt: str,
        user_content: str,
        max_tokens: int,
        operation: str,
        model: str,
    ) -> tuple[str, int, int]:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")

        import anthropic

        est = len(system_prompt) // 4 + len(user_content) // 4 + max_tokens
        await self._ensure_budget_headroom(est)

        client = anthropic.AsyncAnthropic(api_key=api_key)
        msg = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content[:200_000]}],
        )
        parts: list[str] = []
        for block in msg.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        text = "\n".join(parts).strip()
        usage = getattr(msg, "usage", None)
        inp = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        out = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        if inp + out == 0:
            inp = len(system_prompt + user_content) // 4
            out = len(text) // 4
        await self._record_usage(operation, inp, out)
        return text, inp, out

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception(_retryable_llm),
        reraise=True,
    )
    async def complete_deepseek(
        self,
        *,
        system_prompt: str,
        user_content: str,
        max_tokens: int,
        operation: str,
    ) -> tuple[str, int, int]:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set")

        est = len(system_prompt) // 4 + len(user_content) // 4 + max_tokens
        await self._ensure_budget_headroom(est)

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        resp = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content[:200_000]},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        inp = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        out = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        if inp + out == 0:
            inp = len(system_prompt + user_content) // 4
            out = len(text) // 4
        await self._record_usage(operation, inp, out)
        return text, inp, out

    async def complete(
        self,
        *,
        system_prompt: str,
        user_content: str,
        max_tokens: int,
        operation: str,
        provider: str,
        model: Optional[str] = None,
    ) -> tuple[str, int, int, str]:
        """Returns ``text, input_tokens, output_tokens, model_label``."""
        if provider == "anthropic":
            m = model or DEFAULT_HAIKU_MODEL
            t, i, o = await self.complete_anthropic(
                system_prompt=system_prompt,
                user_content=user_content,
                max_tokens=max_tokens,
                operation=operation,
                model=m,
            )
            return t, i, o, m
        if provider == "deepseek":
            t, i, o = await self.complete_deepseek(
                system_prompt=system_prompt,
                user_content=user_content,
                max_tokens=max_tokens,
                operation=operation,
            )
            return t, i, o, DEEPSEEK_MODEL
        raise ValueError(f"Unknown provider: {provider}")


def _default_synthesis_provider() -> str:
    p = os.environ.get("SYNTHESIS_PROVIDER", "").strip().lower()
    if p in ("anthropic", "deepseek"):
        return p
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return "anthropic"
    return "deepseek"


class Synthesizer:
    """Cross-reference narrative for :class:`BriefingReport` (cached by content hash)."""

    def __init__(
        self,
        store: Optional[Store] = None,
        *,
        llm: Optional[LLMService] = None,
    ) -> None:
        self._store = store
        self._llm = llm or LLMService(store)

    async def synthesize(self, briefing: Any) -> SynthesisResult:
        ref = briefing_cache_reference_id(briefing)
        if os.environ.get("BRIEFING_SKIP_SYNTHESIS", "").strip() == "1":
            return SynthesisResult(
                paragraph="Synthesis skipped (BRIEFING_SKIP_SYNTHESIS=1).",
                key_points=[],
                model_used="disabled",
                cached=False,
            )
        if self._store is not None:
            cached = await self._store.get_alerts_sent_payload(
                SYNTHESIS_CACHE_ALERT, ref
            )
            if cached and isinstance(cached.get("paragraph"), str):
                return SynthesisResult(
                    paragraph=cached["paragraph"],
                    key_points=list(cached.get("key_points") or []),
                    model_used="cached",
                    cached=True,
                )

        budget = _daily_budget_tokens()
        if self._store is not None and budget > 0:
            used = await self._store.sum_llm_tokens_for_date(date.today())
            if used >= budget:
                return SynthesisResult(
                    paragraph="Synthesis skipped: daily LLM token budget reached.",
                    key_points=[],
                    model_used="none",
                    cached=False,
                )

        system = _synthesis_prompt_path().read_text(encoding="utf-8")
        user_obj = briefing_to_fingerprint_dict(briefing)
        user_content = (
            "Analyze and cross-reference the following briefing JSON. "
            "Reply with JSON only (no markdown fences):\n"
            '{"opener":"2-4 sentences","bullets":["up to 3 strings"],"nothing":true|false}\n'
            "Set nothing=true only if nothing meaningfully cross-references; then opener must be "
            'exactly: Nothing from today\'s data requires specific attention.\n'
            f"\nBRIEFING_JSON:\n{json.dumps(user_obj, default=str, indent=2)}"
        )

        provider = _default_synthesis_provider()
        try:
            raw, _i, _o, model_used = await self._llm.complete(
                system_prompt=system,
                user_content=user_content,
                max_tokens=600,
                operation="briefing_synthesis",
                provider=provider,
                model=DEFAULT_HAIKU_MODEL if provider == "anthropic" else None,
            )
        except LLMBudgetExceeded:
            return SynthesisResult(
                paragraph="Synthesis skipped: daily LLM token budget reached.",
                key_points=[],
                model_used="none",
                cached=False,
            )

        paragraph, bullets, nothing = _parse_synthesis_json(raw)
        if nothing:
            paragraph = "Nothing from today's data requires specific attention."
            bullets = []

        paragraph = _sanitize_text(_truncate_words(paragraph))
        bullets = [_sanitize_text(_truncate_words(b, 40)) for b in bullets[:3]]

        result = SynthesisResult(
            paragraph=paragraph,
            key_points=bullets,
            model_used=model_used,
            cached=False,
        )

        if self._store is not None:
            await self._store.mark_alert_sent(
                SYNTHESIS_CACHE_ALERT,
                ref,
                payload={
                    "paragraph": result.paragraph,
                    "key_points": result.key_points,
                    "model_used": result.model_used,
                    "cached": False,
                    "ref": ref,
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                },
            )

        return result


def _parse_synthesis_json(raw: str) -> tuple[str, list[str], bool]:
    """Parse model output JSON; fallback to plain text."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        opener = str(data.get("opener") or "").strip()
        bullets = [str(x).strip() for x in (data.get("bullets") or []) if str(x).strip()]
        nothing = bool(data.get("nothing"))
        return opener, bullets[:3], nothing
    except (json.JSONDecodeError, TypeError):
        return raw, [], False


async def filing_excerpt_summarize(
    store: Optional[Store],
    system_prompt: str,
    user_content: str,
    *,
    model: Optional[str] = None,
) -> str:
    """
    Haiku filing summary (SEC). Uses Anthropic only; records ``llm_usage`` when ``store`` is set.
    """
    llm = LLMService(store)
    m = model or DEFAULT_HAIKU_MODEL
    try:
        text, _, _, _ = await llm.complete(
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=1024,
            operation="filing_summary",
            provider="anthropic",
            model=m,
        )
        return text
    except LLMBudgetExceeded as exc:
        raise ValueError(f"LLM budget: {exc}") from exc
