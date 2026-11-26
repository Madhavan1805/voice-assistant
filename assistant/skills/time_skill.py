"""Current time / date."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..core.context import ConversationContext
from ..core.skill import Skill
from ..core.types import Intent, SkillResult
from ..utils.text import ordinal

log = logging.getLogger(__name__)


class TimeSkill(Skill):
    name = "time"
    description = "Tells the current time or today's date."
    slot_hints = '{"kind": "time" | "date"}'
    priority = 20
    patterns = (
        r"\bwhat(?:'s| is)? the (?:current )?time\b",
        r"\bwhat time is it\b",
        r"\b(?:tell me|got|have you got) the time\b",
        r"\bcurrent time\b",
        r"\bwhat(?:'s| is)? (?:today'?s |the )?date\b",
        r"\bwhat day is (?:it|today)\b",
        r"\b(?:time|neram)\s+(?:enna|evlo|evvalavu)\b",
        r"\b(?:inniku|indru)\s+(?:enna\s+)?(?:date|thethi|naal|kizhamai)\b",
    )

    def __init__(self, tz: str = "Asia/Kolkata", now: Callable[[], datetime] | None = None) -> None:
        super().__init__()
        try:
            self._tz = ZoneInfo(tz)
        except ZoneInfoNotFoundError:
            log.warning("unknown timezone %r, falling back to UTC", tz)
            self._tz = timezone.utc
        self._now = now or (lambda: datetime.now(self._tz))

    async def handle(self, intent: Intent, ctx: ConversationContext) -> SkillResult:
        now = self._now()
        kind = intent.slots.get("kind") or (
            "date" if re.search(r"\b(?:date|day|thethi|naal|kizhamai)\b", intent.text, re.I) else "time"
        )
        if kind == "date":
            return SkillResult(f"Today is {now:%A}, {now:%B} {ordinal(now.day)}.")
        return SkillResult(f"It's {now.strftime('%I:%M %p').lstrip('0')}.")
