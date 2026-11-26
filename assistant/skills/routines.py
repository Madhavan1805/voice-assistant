"""User-defined routines (``routines.yaml``): one phrase → several steps."""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..core.context import ConversationContext
from ..core.skill import Skill
from ..core.types import Intent, SkillResult
from ..errors import ConfigError
from ..utils.text import normalize

log = logging.getLogger(__name__)

_STEP_KEYS = {"say", "skill", "wait"}
MAX_STEPS = 20


@dataclass(frozen=True)
class Routine:
    name: str
    triggers: tuple[str, ...]
    steps: tuple[dict[str, Any], ...]


def _validate_step(step: Any) -> dict[str, Any]:
    if not isinstance(step, dict):
        raise ValueError(f"step must be a mapping, got {step!r}")
    kinds = _STEP_KEYS & set(step)
    if len(kinds) != 1:
        raise ValueError(f"step needs exactly one of {sorted(_STEP_KEYS)}: {step!r}")
    extra = set(step) - _STEP_KEYS - {"slots"}
    if extra:
        raise ValueError(f"unknown step key(s) {sorted(extra)}")
    if "wait" in step:
        float(step["wait"])
    return step


def load_routines(path: str | Path) -> list[Routine]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        import yaml

        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"cannot read {p}: {exc}") from exc
    routines: list[Routine] = []
    for i, raw in enumerate(data.get("routines") or [], start=1):
        try:
            routines.append(
                Routine(
                    name=str(raw["name"]),
                    triggers=tuple(t for t in (normalize(str(x)) for x in raw["triggers"]) if t),
                    steps=tuple(_validate_step(s) for s in raw["steps"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"{p}: routine #{i}: {exc}") from exc
    return routines


class RoutineSkill(Skill):
    name = "routine"
    description = "Runs a user-defined multi-step routine."
    priority = 100          # routines beat built-in patterns
    llm_routable = False

    def __init__(self, routines: list[Routine]) -> None:
        super().__init__()
        self._routines = routines
        self._runner: Callable[[Intent], Awaitable[SkillResult]] | None = None

    def bind(self, runner: Callable[[Intent], Awaitable[SkillResult]]) -> None:
        """Give the skill a way to run other skills (breaks the dispatcher↔skill cycle)."""
        self._runner = runner

    def match(self, text: str) -> dict[str, str] | None:
        norm = normalize(text)
        for routine in self._routines:
            for trigger in routine.triggers:
                if re.search(rf"(?:^|\s){re.escape(trigger)}(?:\s|$)", norm):
                    return {"routine": routine.name}
        return None

    async def handle(self, intent: Intent, ctx: ConversationContext) -> SkillResult:
        routine = next((r for r in self._routines if r.name == intent.slots.get("routine")), None)
        if routine is None or self._runner is None:
            return SkillResult("I don't know that routine.", ok=False)

        parts: list[str] = []
        failed = 0
        for step in routine.steps[:MAX_STEPS]:
            if "say" in step:
                parts.append(str(step["say"]))
            elif "wait" in step:
                await asyncio.sleep(min(max(float(step["wait"]), 0.0), 10.0))
            else:
                name = str(step["skill"])
                if name == self.name:
                    log.warning("routine %r: nested routines are not allowed; skipped", routine.name)
                    continue
                slots = {str(k): str(v) for k, v in (step.get("slots") or {}).items()}
                result = await self._runner(Intent(name, slots, 1.0, "routine", intent.text))
                if result.pending is not None:
                    log.warning("routine %r: step %r needs confirmation; skipped", routine.name, name)
                    continue
                failed += not result.ok
                if result.speech:
                    parts.append(result.speech)
        return SkillResult(" ".join(parts), ok=failed == 0)
