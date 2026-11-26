"""Wake-phrase gate.

Local-first and dependency-free: we match the wake phrase on the *transcript*
(fuzzy, because STT mangles names), then keep the conversation open for a
follow-up window so you don't repeat the wake word every turn.

Trade-off: every utterance gets transcribed (CPU), vs. a dedicated always-on
keyword spotter (openWakeWord / Porcupine) that gates *before* STT. The gate is a
single class, so swapping that in later only touches this module + the pipeline.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable

from ..utils.text import normalize


@dataclass(frozen=True)
class WakeResult:
    accepted: bool
    command: str = ""
    woke: bool = False   # True if the wake phrase itself was heard


class WakeGate:
    def __init__(
        self,
        phrases: list[str],
        *,
        required: bool = True,
        threshold: float = 0.8,
        followup_window_s: float = 8.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        cleaned = {normalize(p) for p in phrases if normalize(p)}
        self._phrases = sorted(cleaned, key=lambda p: (-len(p.split()), p))  # longest first
        self._required = required
        self._threshold = threshold
        self._window = followup_window_s
        self._clock = clock
        self._open_until = float("-inf")

    def extend(self) -> None:
        """Keep the conversation open for ``followup_window_s`` from now."""
        self._open_until = self._clock() + self._window

    def check(self, text: str) -> WakeResult:
        woke, command = self._strip_wake_phrase(text)
        if woke:
            return WakeResult(True, command, True)
        if not self._required or self._clock() < self._open_until:
            return WakeResult(True, text.strip(), False)
        return WakeResult(False)

    def _strip_wake_phrase(self, text: str) -> tuple[bool, str]:
        tokens = [(normalize(m.group()), m.end()) for m in re.finditer(r"\S+", text)]
        tokens = [(t, end) for t, end in tokens if t]
        for phrase in self._phrases:
            n = len(phrase.split())
            if len(tokens) < n:
                continue
            head = " ".join(t for t, _ in tokens[:n])
            if SequenceMatcher(None, head, phrase).ratio() >= self._threshold:
                rest = text[tokens[n - 1][1] :]
                return True, rest.lstrip(" ,.!?;:-–—").strip()
        return False, text.strip()
