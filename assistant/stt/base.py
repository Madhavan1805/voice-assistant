from __future__ import annotations

from abc import ABC, abstractmethod

from ..events import Transcript, Utterance


class STTEngine(ABC):
    """Contract for speech recognisers.

    Raise :class:`TranscriptionEmpty` when the engine worked but heard no speech,
    and :class:`STTError` when the engine itself failed (so a fallback can try).
    """

    name: str = "stt"

    @abstractmethod
    async def transcribe(self, utterance: Utterance) -> Transcript: ...

    async def warmup(self) -> None:
        """Load models ahead of the first utterance (optional)."""

    async def aclose(self) -> None:
        """Release resources (optional)."""
