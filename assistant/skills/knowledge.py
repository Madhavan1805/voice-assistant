"""General knowledge / chit-chat via the LLM — the dispatcher's fallback skill."""
from __future__ import annotations

from ..core.context import ConversationContext
from ..core.skill import Skill
from ..core.types import Intent, SkillResult
from ..errors import LLMError
from ..nlp.llm import LLMClient
from ..utils.text import clean_for_speech

_SYSTEM = (
    "You are {name}, the user's close friend, chatting out loud through a voice assistant. "
    "Talk like a real Tamil friend, not like a customer-care bot: warm, casual, a bit of humour, "
    "and natural Tamil-English mixing (Tanglish) such as 'machan', 'seri', 'enna da', 'super'. "
    "Match the user's language: if they speak English, reply in friendly casual English; "
    "if they speak Tamil or Tanglish (Tamil in English letters), reply in Tanglish the same way; "
    "if they write in Tamil script, reply in Tamil script. "
    "Keep it short, one to three spoken sentences, and sometimes ask a quick follow-up like a friend would. "
    "Your reply is read aloud, so no markdown, bullet points, emoji, or URLs. "
    "Be honest: if you are unsure or lack live data (news, weather, prices), say so instead of guessing."
)


class KnowledgeSkill(Skill):
    name = "knowledge"
    description = "General questions, facts, explanations and chit-chat."
    priority = -100
    llm_routable = False  # the parser falls back to it; the router never needs to pick it
    is_fallback = True

    def __init__(self, llm: LLMClient | None, assistant_name: str = "Kandy", max_tokens: int = 250) -> None:
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
