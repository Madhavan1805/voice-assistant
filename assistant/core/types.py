"""Core data contracts between the NLP layer, the dispatcher and skills."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class Intent:
    skill: str
    slots: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    source: str = "rules"        # rules | llm | routine | fallback
    text: str = ""               # the original utterance


@dataclass
class PendingAction:
    """A risky action waiting for the user to say yes/no."""

    execute: "Callable[[], Awaitable[SkillResult]]"
    cancel_speech: str = ""
    description: str = ""
    expires_at: float = 0.0      # filled in by the dispatcher


@dataclass
class SkillResult:
    speech: str
    ok: bool = True
    language: str | None = None
    pending: PendingAction | None = None
    data: dict[str, Any] = field(default_factory=dict)
