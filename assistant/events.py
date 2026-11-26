"""Plain data objects that flow between pipeline stages."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, eq=False)
class Utterance:
    """One VAD-segmented chunk of speech (mono int16 PCM)."""

    audio: np.ndarray
    sample_rate: int = 16000
    captured_at: float = field(default_factory=time.time)

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str | None = None
    confidence: float = 1.0
    engine: str = ""


@dataclass(frozen=True)
class Command:
    """A transcript that passed the wake-word gate and is ready for dispatch."""

    text: str
    language: str | None = None
