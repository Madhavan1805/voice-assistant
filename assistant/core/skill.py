"""Skill plug-in interface + registry.

To add a skill: subclass :class:`Skill`, give it ``name``/``description`` and either
regex ``patterns`` (fast, offline, deterministic) or rely on LLM routing via
``description``/``slot_hints``; implement ``handle``; register it in
``assistant/bootstrap.py``.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import ClassVar, Sequence

from ..errors import ConfigError
from .context import ConversationContext
from .types import Intent, SkillResult


class Skill(ABC):
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""          # shown to the LLM router
    slot_hints: ClassVar[str] = ""           # JSON-ish slot schema shown to the LLM router
    patterns: ClassVar[Sequence[str]] = ()   # regexes with (?P<slot>…) groups, case-insensitive
    priority: ClassVar[int] = 0              # higher wins when several skills match
    llm_routable: ClassVar[bool] = True
    is_fallback: ClassVar[bool] = False      # catch-all for anything nothing else claimed

    def __init__(self) -> None:
        self._rx = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def match(self, text: str) -> dict[str, str] | None:
        """Return slots if this skill claims ``text``, else ``None``."""
        for rx in self._rx:
            m = rx.search(text)
            if m:
                return {k: v.strip() for k, v in m.groupdict().items() if v}
        return None

    @abstractmethod
    async def handle(self, intent: Intent, ctx: ConversationContext) -> SkillResult: ...

    async def aclose(self) -> None:
        """Release resources (browser sessions, HTTP clients…)."""


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> Skill:
        if not skill.name:
            raise ConfigError(f"{type(skill).__name__} has no name")
        if skill.name in self._skills:
            raise ConfigError(f"duplicate skill name: {skill.name}")
        self._skills[skill.name] = skill
        return skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def all(self) -> list[Skill]:
        """Highest priority first; ties keep registration order."""
        return sorted(self._skills.values(), key=lambda s: -s.priority)

    @property
    def fallback(self) -> Skill | None:
        return next((s for s in self._skills.values() if s.is_fallback), None)

    def catalog(self) -> str:
        """Text description of routable skills for the LLM router prompt."""
        lines = []
        for s in self.all():
            if s.llm_routable:
                slots = f" slots: {s.slot_hints}" if s.slot_hints else ""
                lines.append(f'- "{s.name}": {s.description}{slots}')
        return "\n".join(lines)

    async def aclose(self) -> None:
        for skill in self._skills.values():
            await skill.aclose()
