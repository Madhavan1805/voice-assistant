"""Short-term conversation memory shared by the dispatcher and skills."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .types import PendingAction


@dataclass(frozen=True)
class Turn:
    role: str   # "user" | "assistant"
    text: str


class ConversationContext:
    def __init__(self, max_turns: int = 6) -> None:
        self._turns: deque[Turn] = deque(maxlen=max_turns * 2)
        self.pending: PendingAction | None = None

    def add_user(self, text: str) -> None:
        self._turns.append(Turn("user", text))

    def add_assistant(self, text: str) -> None:
        if text:
            self._turns.append(Turn("assistant", text))

    def messages(self) -> list[dict[str, str]]:
        """History in chat-API format; always starts with a user turn."""
        turns = list(self._turns)
        while turns and turns[0].role != "user":
            turns.pop(0)
        return [{"role": t.role, "content": t.text} for t in turns]

    def clear(self) -> None:
        self._turns.clear()
        self.pending = None
