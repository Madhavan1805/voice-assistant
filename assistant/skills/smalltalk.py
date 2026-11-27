"""Greetings, jokes, 'what can you do' — works offline, no LLM needed."""
from __future__ import annotations

import random

from ..core.context import ConversationContext
from ..core.skill import Skill
from ..core.types import Intent, SkillResult

_JOKES = [
    "Why don't programmers like nature? Too many bugs.",
    "I told my computer I needed a break. Now it won't stop sending me vacation ads.",
    "Why did the scarecrow win an award? He was outstanding in his field.",
    "What do you call a fake noodle? An impasta.",
    "Why did the phone go to school? To improve its grades, machan.",
    "I would tell you a UDP joke, but you might not get it.",
    "Why was the math book sad? It had too many problems.",
    "What do you call cheese that isn't yours? Nacho cheese.",
    "Why do Java developers wear glasses? Because they don't C sharp.",
    "I asked the fridge for advice. It said: chill, machan.",
]

_CAPABILITIES = (
    "I can tell you the time and date, set timers and reminders, check the weather, "
    "do quick maths, play songs on YouTube, search the web, open apps, change the volume, "
    "tell jokes, and chat about anything once my AI brain is connected."
)


class SmallTalkSkill(Skill):
    name = "smalltalk"
    description = "Greetings, thanks, jokes, coin flips, and questions about what the assistant can do."
    priority = 10
    patterns = (
        r"^(?P<greet>(?:hey|hi|hello|hai|yo|vanakkam)\b[\s,!.]*(?:kandy|candy|there|machan|da)?[\s!.]*)$",
        r"(?P<how>\bhow are you\b|\bhow're you\b|\bhow do you do\b|\bepdi irukka\b|\bepdi irukeenga\b|\bnalama\b)",
        r"(?P<who>\bwho are you\b|\bwhat(?:'s| is) your name\b|\bunga peyar\b|\bun peyar\b|\bunnoda peyar\b)",
        r"^(?P<thanks>(?:thanks|thank you|thank u|thanks a lot|nandri)\b.*)$",
        r"(?P<capabilities>\bwhat (?:else )?(?:can|could) you (?:do|help)\b|\bwhat do you do\b|\bhow can you help\b)",
        r"(?P<joke>\bjoke\b|\bmake me laugh\b)",
        r"(?P<coin>\b(?:flip|toss)\s+a\s+coin\b)",
        r"(?P<dice>\broll\s+a\s+(?:die|dice)\b)",
    )

    def __init__(self, assistant_name: str = "Kandy") -> None:
        super().__init__()
        self._name = assistant_name

    async def handle(self, intent: Intent, ctx: ConversationContext) -> SkillResult:
        s = intent.slots
        if "joke" in s:
            return SkillResult(random.choice(_JOKES))
        if "capabilities" in s:
            return SkillResult(_CAPABILITIES)
        if "coin" in s:
            return SkillResult(f"It's {random.choice(['heads', 'tails'])}.")
        if "dice" in s:
            return SkillResult(f"You rolled a {random.randint(1, 6)}.")
        if "who" in s:
            return SkillResult(f"I'm {self._name}, your voice assistant. Think of me as a friend who's always around.")
        if "how" in s:
            return SkillResult(random.choice(
                ["I'm doing great, machan! How about you?", "All good here! What's up with you?"]
            ))
        if "thanks" in s:
            return SkillResult(random.choice(["Anytime, machan!", "No problem at all!", "Always happy to help!"]))
        return SkillResult(random.choice(
            ["Hey machan! What can I do for you?", "Hi! Sollunga, what do you need?", "Hello! I'm listening."]
        ))
