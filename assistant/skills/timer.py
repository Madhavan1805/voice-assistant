"""Timers and reminders ('set a timer for 5 minutes', 'remind me in 10 minutes to call amma').

They live in memory for as long as the assistant is running (they are lost on restart).
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import re
import time
from dataclasses import dataclass

from ..core.announce import Announcer
from ..core.context import ConversationContext
from ..core.skill import Skill
from ..core.types import Intent, SkillResult

log = logging.getLogger(__name__)

_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty five": 45, "forty": 40, "sixty": 60,
}
_UNITS = {"sec": 1, "second": 1, "min": 60, "minute": 60, "hr": 3600, "hour": 3600}
_NUM_ALT = "|".join(sorted(_NUMBERS, key=len, reverse=True))
_DURATION = re.compile(
    rf"(?P<n>\d+(?:\.\d+)?|half an?|half|{_NUM_ALT})\s*(?P<u>seconds?|secs?|minutes?|mins?|hours?|hrs?)\b"
)
MAX_SECONDS = 24 * 3600


def parse_duration(text: str) -> float | None:
    """'1 hour 30 minutes' / 'half an hour' / 'an hour and a half' → seconds."""
    t = text.lower()
    total, last_unit = 0.0, 0
    for m in _DURATION.finditer(t):
        raw = m.group("n")
        n = 0.5 if raw.startswith("half") else float(raw) if raw[0].isdigit() else _NUMBERS[raw]
        unit = _UNITS[m.group("u").rstrip("s")] if m.group("u").rstrip("s") in _UNITS else _UNITS[m.group("u")[:3]]
        total += n * unit
        last_unit = unit
    if last_unit and re.search(r"\band a half\b", t):
        total += 0.5 * last_unit
    return total if 0 < total <= MAX_SECONDS else None


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h} hour{'s' if h != 1 else ''}")
    if m:
        parts.append(f"{m} minute{'s' if m != 1 else ''}")
    if s and not h:
        parts.append(f"{s} second{'s' if s != 1 else ''}")
    return " ".join(parts) or "0 seconds"


@dataclass
class _Timer:
    id: int
    ends_at: float
    seconds: float
    what: str | None
    task: asyncio.Task | None = None


class TimerSkill(Skill):
    name = "timer"
    description = "Sets, lists and cancels timers and reminders."
    slot_hints = '{"when": "<duration like 5 minutes>", "what": "<reminder text, optional>"}'
    priority = 45
    patterns = (
        r"^(?:please\s+)?(?:set|start|create)\s+(?:a|an)\s+(?P<when>[\w\s.]+?)\s+timer$",
        r"^(?:please\s+)?(?:set|start|create)\s+(?:a\s+|an\s+)?(?:\w+\s+)?timer\s+(?:for\s+|of\s+)?(?P<when>.+)$",
        r"\bremind me\s+(?:in|after)\s+(?P<when>.+?)\s+(?:to|that)\s+(?P<what>.+)$",
        r"\bremind me\s+to\s+(?P<what>.+?)\s+(?:in|after)\s+(?P<when>.+)$",
        r"\b(?P<cancel>cancel|stop|clear|delete)\s+(?:the\s+|my\s+|all\s+)?(?:timers?|reminders?)\b",
        r"\b(?P<list>how much time (?:is )?left|what timers|list (?:my )?(?:timers|reminders)|any timers)\b",
    )

    def __init__(self, announcer: Announcer) -> None:
        super().__init__()
        self._announcer = announcer
        self._timers: dict[int, _Timer] = {}
        self._ids = itertools.count(1)

    def match(self, text: str) -> dict[str, str] | None:
        slots = super().match(text)
        if slots and "when" in slots and parse_duration(slots["when"]) is None:
            return None   # "set a timer" with no usable duration: let the handler ask? no — not ours
        return slots

    async def handle(self, intent: Intent, ctx: ConversationContext) -> SkillResult:
        s = intent.slots
        if "cancel" in s:
            n = self._cancel_all()
            return SkillResult("Okay, cancelled." if n else "You don't have any timers running.")
        if "list" in s:
            return SkillResult(self._describe())
        seconds = parse_duration(s.get("when") or intent.text)
        if seconds is None:
            return SkillResult("How long should I set it for?", ok=False)

        timer = _Timer(next(self._ids), time.monotonic() + seconds, seconds, (s.get("what") or "").strip() or None)
        timer.task = asyncio.create_task(self._run(timer), name=f"timer-{timer.id}")
        self._timers[timer.id] = timer
        dur = format_duration(seconds)
        if timer.what:
            return SkillResult(f"Okay, I'll remind you to {timer.what} in {dur}.")
        return SkillResult(f"Okay, timer set for {dur}.")

    async def _run(self, timer: _Timer) -> None:
        try:
            await asyncio.sleep(timer.seconds)
        except asyncio.CancelledError:
            return
        self._timers.pop(timer.id, None)
        if timer.what:
            self._announcer.say(f"Hey, reminder: {timer.what}.")
        else:
            self._announcer.say(f"Time's up! Your {format_duration(timer.seconds)} timer is done.")

    def _cancel_all(self) -> int:
        n = len(self._timers)
        for t in self._timers.values():
            if t.task:
                t.task.cancel()
        self._timers.clear()
        return n

    def _describe(self) -> str:
        if not self._timers:
            return "You don't have any timers running."
        now = time.monotonic()
        bits = []
        for t in sorted(self._timers.values(), key=lambda t: t.ends_at):
            left = format_duration(max(0, t.ends_at - now))
            bits.append(f"a reminder to {t.what} in {left}" if t.what else f"a timer with {left} left")
        return "You have " + ", and ".join(bits) + "."

    async def aclose(self) -> None:
        self._cancel_all()
