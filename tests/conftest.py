from __future__ import annotations

import asyncio
import inspect

import numpy as np
import pytest

from assistant.core.context import ConversationContext
from assistant.core.dispatcher import Dispatcher
from assistant.core.skill import SkillRegistry
from assistant.errors import LLMError
from assistant.nlp.intent import IntentParser
from assistant.nlp.llm import LLMClient
from assistant.skills.knowledge import KnowledgeSkill
from assistant.skills.routines import RoutineSkill


def pytest_pyfunc_call(pyfuncitem):
    """Tiny built-in async test runner (no pytest-asyncio dependency)."""
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        args = {a: pyfuncitem.funcargs[a] for a in pyfuncitem._fixtureinfo.argnames}
        asyncio.run(pyfuncitem.obj(**args))
        return True


class FakeLLM(LLMClient):
    name = "fake"

    def __init__(self, *replies: str | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def complete(self, messages, *, system="", max_tokens=300, temperature=0.3) -> str:
        self.calls.append({"messages": messages, "system": system})
        reply = self.replies.pop(0) if self.replies else "ok"
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def make_assistant():
    def _make(llm: LLMClient | None = None, *skills, routines=None, llm_routing=True):
        registry = SkillRegistry()
        routine_skill = RoutineSkill(routines or [])
        registry.register(routine_skill)
        for s in skills:
            registry.register(s)
        registry.register(KnowledgeSkill(llm))
        ctx = ConversationContext()
        dispatcher = Dispatcher(registry, IntentParser(registry, llm, llm_routing=llm_routing), ctx)
        routine_skill.bind(dispatcher.run_intent)
        return dispatcher, registry

    return _make


def tone(n: int, amp: float = 8000.0, rate: int = 16000, freq: float = 220.0) -> np.ndarray:
    t = np.arange(n) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.int16)


def noise(n: int, amp: float = 30.0, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(n) * amp).astype(np.int16)


__all__ = ["FakeLLM", "tone", "noise", "LLMError"]
