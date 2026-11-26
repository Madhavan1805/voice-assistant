from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioChunk:
    pcm: bytes          # mono int16 little-endian
    sample_rate: int


class TTSEngine(ABC):
    name: str = "tts"

    @abstractmethod
    def supports(self, language: str) -> bool:
        """Can this engine speak ``language`` right now (voice installed, etc.)?"""

    @abstractmethod
    async def synthesize(self, text: str, language: str) -> AudioChunk:
        """Raise :class:`TTSError` on failure."""

    async def warmup(self) -> None:
        """Pre-load voices so the first reply isn't slow (optional)."""

    async def aclose(self) -> None:
        """Release resources (optional)."""
