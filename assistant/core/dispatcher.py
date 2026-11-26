"""The central dispatcher: transcript in, spoken result out. It never raises.

Responsibilities
  1. Resolve pending confirmations ("Send it?" → "yes"/"aama"/"venam").
  2. Parse the intent (rules → LLM → fallback) and route to a skill.
  3. Guard every skill call: timeout + translate any failure into a friendly,
     speakable message so one broken skill can't take the assistant down.
  4. Keep short-term conversation history for follow-up questions.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from ..errors import AssistantError, LLMError
from ..messages import msg
from ..nlp.intent import IntentParser
from ..utils.text import detect_language, normalize
from .context import ConversationContext
from .skill import SkillRegistry
from .types import Intent, SkillResult

log = logging.getLogger(__name__)

_YES = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "confirmed", "proceed",
    "aama", "aam", "sari", "seri", "ஆமா", "ஆம்", "சரி",
}
_NO = {
    "no", "nope", "nah", "cancel", "stop", "don't", "dont", "never",
    "venam", "vendam", "vena", "illa", "illai", "வேண்டாம்", "வேணாம்", "இல்லை",
}
_YES_PHRASES = {"go ahead", "do it", "send it"}


def interpret_confirmation(text: str) -> bool | None:
    """True = yes, False = no, None = something else entirely (a new command)."""
    norm = normalize(text)
    tokens = norm.split()
    if not tokens or len(tokens) > 4:
        return None
    if any(t in _NO for t in tokens):
        return False
    if norm in _YES_PHRASES or any(t in _YES for t in tokens):
        return True
    return None


class Dispatcher:
    def __init__(
        self,
        registry: SkillRegistry,
        parser: IntentParser,
        context: ConversationContext,
        *,
        skill_timeout_s: float = 20.0,
        confirm_ttl_s: float = 20.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self._parser = parser
        self._ctx = context
        self._timeout = skill_timeout_s
        self._ttl = confirm_ttl_s
        self._clock = clock

    @property
    def context(self) -> ConversationContext:
        return self._ctx

    async def handle(self, text: str, language: str | None = None) -> SkillResult:
        text = text.strip()
        lang = language or detect_language(text)
        if not text:
            return SkillResult("")
        self._ctx.add_user(text)
        try:
            result = await self._dispatch(text, lang)
        except Exception:  # noqa: BLE001 - last line of defence; CancelledError passes through
            log.exception("dispatcher crashed while handling %r", text)
            result = SkillResult(msg("skill_error", lang), ok=False)
        if not result.language:
            result.language = detect_language(result.speech)
        self._ctx.add_assistant(result.speech)
        return result

    async def run_intent(self, intent: Intent, lang: str = "en") -> SkillResult:
        """Execute one already-parsed intent (also used by routines for their steps)."""
        skill = self._registry.get(intent.skill)
        if skill is None:
            if intent.source == "routine":  # e.g. routine uses smart_home but it's disabled
                log.warning("routine step references unavailable skill %r; skipped", intent.skill)
                return SkillResult("", ok=False)
            skill = self._registry.fallback
        if skill is None:
            return SkillResult(msg("didnt_catch", lang), ok=False)
        log.info("→ %s %s [%s %.2f]", skill.name, intent.slots, intent.source, intent.confidence)
        return await self._guard(skill.handle(intent, self._ctx), skill.name, lang)

    # ------------------------------------------------------------------
    async def _dispatch(self, text: str, lang: str) -> SkillResult:
        pending, self._ctx.pending = self._ctx.pending, None
        if pending is not None and self._clock() <= pending.expires_at:
            answer = interpret_confirmation(text)
            if answer is True:
                return await self._guard(pending.execute(), pending.description or "confirmation", lang)
            if answer is False:
                return SkillResult(pending.cancel_speech or msg("cancelled", lang))
            # anything else: the user moved on → treat as a fresh command

        intent = await self._parser.parse(text)
        result = await self.run_intent(intent, lang)
        if result.pending is not None:
            result.pending.expires_at = self._clock() + self._ttl
            self._ctx.pending = result.pending
        return result

    async def _guard(self, awaitable: Awaitable[SkillResult], label: str, lang: str) -> SkillResult:
        try:
            return await asyncio.wait_for(awaitable, self._timeout)
        except asyncio.TimeoutError:
            log.warning("skill %r timed out after %.0fs", label, self._timeout)
            return SkillResult(msg("timeout", lang), ok=False)
        except LLMError as exc:
            log.warning("skill %r: LLM unavailable: %s", label, exc)
            return SkillResult(exc.user_message or msg("offline", lang), ok=False)
        except AssistantError as exc:
            log.warning("skill %r failed: %s", label, exc)
            return SkillResult(exc.user_message or msg("skill_error", lang), ok=False)
        except Exception:  # noqa: BLE001
            log.exception("skill %r crashed", label)
            return SkillResult(msg("skill_error", lang), ok=False)
