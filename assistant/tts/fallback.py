"""Ordered TTS chain: first engine that supports the language and works wins."""
from __future__ import annotations

import asyncio
import logging

from ..errors import TTSError
from ..utils.retry import CircuitBreaker
from .base import AudioChunk, TTSEngine

log = logging.getLogger(__name__)


class FallbackTTS(TTSEngine):
    name = "fallback-tts"

    def __init__(
        self,
        engines: list[TTSEngine],
        *,
        timeout_s: float = 15.0,
        breaker_threshold: int = 3,
        breaker_reset_s: float = 60.0,
    ) -> None:
        if not engines:
            raise ValueError("FallbackTTS needs at least one engine")
        self._engines = engines
        self._timeout = timeout_s
        self._breakers = {e.name: CircuitBreaker(breaker_threshold, breaker_reset_s) for e in engines}

    def supports(self, language: str) -> bool:
        return any(e.supports(language) for e in self._engines)

    async def synthesize(self, text: str, language: str) -> AudioChunk:
        errors: list[str] = []
        for engine in self._engines:
            if not engine.supports(language):
                continue
            breaker = self._breakers[engine.name]
            if not breaker.allow():
                errors.append(f"{engine.name}: circuit open")
                continue
            try:
                chunk = await asyncio.wait_for(engine.synthesize(text, language), self._timeout)
            except (TTSError, asyncio.TimeoutError) as exc:
                breaker.record_failure()
                reason = "timeout" if isinstance(exc, asyncio.TimeoutError) else str(exc)
                log.warning("TTS engine %s failed: %s", engine.name, reason)
                errors.append(f"{engine.name}: {reason}")
                continue
            breaker.record_success()
            return chunk
        raise TTSError(f"no TTS engine could speak {language!r}: " + ("; ".join(errors) or "none installed"))

    async def warmup(self) -> None:
        for engine in self._engines:
            try:
                await engine.warmup()
            except Exception as exc:  # noqa: BLE001 - best-effort
                log.warning("TTS warmup of %s failed: %s", engine.name, exc)

    async def aclose(self) -> None:
        for engine in self._engines:
            await engine.aclose()
