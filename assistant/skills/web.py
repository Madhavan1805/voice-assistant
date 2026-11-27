"""'Play <song>' (opens YouTube results) and 'search for <thing>' (opens a web search).

Uses the standard-library ``webbrowser`` with URL-encoded queries — no shell involved.
"""
from __future__ import annotations

import asyncio
import webbrowser
from typing import Callable
from urllib.parse import quote_plus

from ..core.context import ConversationContext
from ..core.skill import Skill
from ..core.types import Intent, SkillResult
from ..errors import SkillError

MAX_QUERY = 100


class WebSkill(Skill):
    name = "web"
    description = "Plays a song or video on YouTube, or searches the web for something."
    slot_hints = '{"play": "<song or video>"} or {"search": "<query>"}'
    priority = 28
    patterns = (
        r"^(?:please\s+)?play\s+(?P<play>.+?)(?:\s+(?:on|in)\s+(?:you\s?tube|spotify|music))?$",
        r"^(?:please\s+)?(?:search|google|look up)\s+(?:for\s+)?(?P<search>.+?)(?:\s+(?:on|in)\s+(?:google|the web|the internet))?$",
    )

    def __init__(self, opener: Callable[[str], bool] | None = None) -> None:
        super().__init__()
        self._open = opener or (lambda url: webbrowser.open(url, new=2))

    async def handle(self, intent: Intent, ctx: ConversationContext) -> SkillResult:
        s = intent.slots
        if "play" in s:
            query = s["play"][:MAX_QUERY]
            url = "https://www.youtube.com/results?search_query=" + quote_plus(query)
            done = f"Okay, I've opened YouTube for {query}."
        elif "search" in s:
            query = s["search"][:MAX_QUERY]
            url = "https://www.google.com/search?q=" + quote_plus(query)
            done = f"Okay, searching the web for {query}."
        else:
            return SkillResult("What should I look for?", ok=False)
        try:
            opened = await asyncio.to_thread(self._open, url)
        except Exception as exc:  # noqa: BLE001
            raise SkillError(str(exc), user_message="I couldn't open the browser.") from exc
        if not opened:
            raise SkillError("no browser", user_message="I couldn't open the browser.")
        return SkillResult(done)
