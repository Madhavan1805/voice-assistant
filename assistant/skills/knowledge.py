"""General knowledge / chit-chat via the LLM — the dispatcher's fallback skill."""
from __future__ import annotations

from ..core.context import ConversationContext
from ..core.skill import Skill
from ..core.types import Intent, SkillResult
from ..errors import LLMError
from ..nlp.llm import LLMClient
from ..utils.text import clean_for_speech

_SYSTEM = (
    "You are {name}, a friendly voice assistant. Your reply will be read aloud by a "
    "text-to-speech engine, so: answer in one to three short spoken sentences; no "
    "markdown, bullet points, emoji, or URLs. If the user speaks Tamil or Tanglish "
    "(Tamil written in English letters), reply the same way. If you are unsure or lack "
    "live data (news, weather, prices), say so briefly instead of guessing."
)


class KnowledgeSkill(Skill):
    name = "knowledge"
    description = "General questions, facts, explanations and chit-chat."
    priority = -100
    llm_routable = False  # the parser falls back to it; the router never needs to pick it
    is_fallback = True

    def __init__(self, llm: LLMClient | None, assistant_name: str = "Madhav", max_tokens: int = 250) -> None:
        super().__init__()
        self._llm = llm
        self._system = _SYSTEM.format(name=assistant_name)
        self._max_tokens = max_tokens

    async def handle(self, intent: Intent, ctx: ConversationContext) -> SkillResult:
        if self._llm is None:
            raise LLMError("no LLM backend configured")
        messages = ctx.messages() or [{"role": "user", "content": intent.text}]
        answer = await self._llm.complete(
            messages, system=self._system, max_tokens=self._max_tokens, temperature=0.5
        )
        return SkillResult(clean_for_speech(answer))
