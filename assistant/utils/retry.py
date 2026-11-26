"""Retry with exponential backoff + a small circuit breaker for flaky backends."""
from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Awaitable, Callable, TypeVar

log = logging.getLogger(__name__)
T = TypeVar("T")


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.3,
    max_delay: float = 4.0,
    jitter: float = 0.2,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Call ``fn`` up to ``attempts`` times. CancelledError is never swallowed."""
    attempts = max(1, attempts)
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except retry_on as exc:
            if attempt == attempts:
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            delay *= 1 + random.uniform(-jitter, jitter)
            log.debug("attempt %d/%d failed (%s); retrying in %.2fs", attempt, attempts, exc, delay)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


class CircuitBreaker:
    """closed → (N failures) → open → (reset timeout) → half-open → closed/open.

    Stops us hammering a dead backend on every utterance and lets the fallback
    chain skip straight to the next engine.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        reset_timeout_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = failure_threshold
        self._reset = reset_timeout_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self._reset:
            return "half-open"
        return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold or self.state == "half-open":
            self._opened_at = self._clock()
