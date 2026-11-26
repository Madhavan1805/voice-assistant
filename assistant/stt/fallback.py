"""Ordered chain of STT engines with per-engine timeout and circuit breaker."""
from __future__ import annotations

import asyncio
import logging

from ..errors import STTError, TranscriptionEmpty
from ..events import Transcript, Utterance
from ..utils.retry import CircuitBreaker
from .base import STTEngine

log = logging.getLogger(__name__)


class FallbackSTT(STTEngine):
    name = "fallback-stt"

    def __init__(
        self,
        engines: list[STTEngine],
        *,
        timeout_s: float = 25.0,
        breaker_threshold: int = 3,
        breaker_reset_s: float = 60.0,
    ) -> None:
        if not engines:
            raise ValueError("FallbackSTT needs at least one engine")
        self._engines = engines
        self._timeout = timeout_s
        self._breakers = {e.name: CircuitBreaker(breaker_threshold, breaker_reset_s) for e in engines}

    async def transcribe(self, utterance: Utterance) -> Transcript:
        errors: list[str] = []
        for engine in self._engines:
            breaker = self._breakers[engine.name]
            if not breaker.allow():
                errors.append(f"{engine.name}: circuit open")
                continue
            try:
                result = await asyncio.wait_for(engine.transcribe(utterance), self._timeout)
            except TranscriptionEmpty:
                # The engine worked and heard nothing. Don't pay for / trust a 2nd opinion on noise.
                breaker.record_success()
                raise
            except (STTError, asyncio.TimeoutError) as exc:
                breaker.record_failure()
                reason = "timeout" if isinstance(exc, asyncio.TimeoutError) else str(exc)
                log.warning("STT engine %s failed: %s", engine.name, reason)
                errors.append(f"{engine.name}: {reason}")
                continue
            breaker.record_success()
            return result
        raise STTError("all STT engines failed: " + "; ".join(errors))

    async def warmup(self) -> None:
        for engine in self._engines:
            try:
                await engine.warmup()
            except Exception as exc:  # noqa: BLE001 - warmup is best-effort
                log.warning("warmup of %s failed: %s", engine.name, exc)

    async def aclose(self) -> None:
        for engine in self._engines:
            await engine.aclose()
