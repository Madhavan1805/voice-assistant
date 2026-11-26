"""Intent parsing: deterministic rules first, LLM routing second, fallback skill last.

Why this order: rules are instant, offline and predictable ("turn off the fan" must
never depend on a model's mood). The LLM is only consulted for phrasings the rules
miss, and *not at all* for plain questions (those go straight to the knowledge skill,
saving a whole LLM round-trip of latency).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re

from ..core.skill import SkillRegistry
from ..core.types import Intent
from ..errors import LLMError
from ..utils.text import is_question, light_clean
from .llm import LLMClient

log = logging.getLogger(__name__)

_ROUTER_SYSTEM = (
    "You route spoken commands to skills for a voice assistant. "
    "Reply with ONLY one compact JSON object and nothing else:\n"
    '{"skill": "<name>", "slots": {"<slot>": "<value>"}, "confidence": <0.0-1.0>}\n'
    'Use "knowledge" for general questions and chit-chat, or when no other skill clearly fits. '
    "The user may speak English, Tamil, or Tanglish (Tamil in English letters)."
)


def extract_json(raw: str) -> dict | None:
    """Pull the first JSON object out of an LLM reply (tolerates prose / code fences)."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


class IntentParser:
    def __init__(
        self,
        registry: SkillRegistry,
        llm: LLMClient | None = None,
        *,
        llm_routing: bool = True,
        min_confidence: float = 0.5,
        router_timeout_s: float = 8.0,
    ) -> None:
        self._registry = registry
        self._llm = llm
        self._routing = llm_routing
        self._min_conf = min_confidence
        self._timeout = router_timeout_s

    async def parse(self, text: str) -> Intent:
        clean = light_clean(text)

        # 1) deterministic rules
        for skill in self._registry.all():
            if skill.is_fallback:
                continue
            slots = skill.match(clean)
            if slots is not None:
                log.debug("rule match: %s %s", skill.name, slots)
                return Intent(skill.name, slots, 0.95, "rules", text)

        fallback = self._registry.fallback
        fallback_name = fallback.name if fallback else "knowledge"

        # 2) LLM routing — skipped for plain questions (they'd just be answered anyway)
        if self._llm is not None and self._routing and not is_question(clean):
            try:
                intent = await asyncio.wait_for(self._route(clean, text), self._timeout)
                if intent is not None:
                    return intent
            except (LLMError, asyncio.TimeoutError) as exc:
                log.info("LLM routing unavailable (%s); using fallback skill", exc or "timeout")
            except Exception:  # noqa: BLE001 - routing must never break the pipeline
                log.exception("unexpected error in LLM routing")

        # 3) fallback (general knowledge / chat)
        return Intent(fallback_name, {}, 0.3, "fallback", text)

    async def _route(self, clean: str, original: str) -> Intent | None:
        assert self._llm is not None
        prompt = f"Skills:\n{self._registry.catalog()}\n\nUser said: {clean}"
        raw = await self._llm.complete(
            [{"role": "user", "content": prompt}],
            system=_ROUTER_SYSTEM,
            max_tokens=120,
            temperature=0.0,
        )
        data = extract_json(raw)
        if data is None:
            log.debug("router returned non-JSON: %r", raw[:120])
            return None
        name = str(data.get("skill", ""))
        skill = self._registry.get(name)
        if skill is None or not skill.llm_routable or skill.is_fallback:
            return None
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < self._min_conf:
            return None
        raw_slots = data.get("slots") if isinstance(data.get("slots"), dict) else {}
        slots = {str(k): re.sub(r"\s+", " ", str(v)).strip() for k, v in raw_slots.items() if v is not None}
        return Intent(name, slots, confidence, "llm", original)
