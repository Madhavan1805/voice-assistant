"""Lets skills speak *unprompted* (timer finished, reminder due).

Skills hold an :class:`Announcer`; whoever owns the output (the pipeline, or the text
REPL) binds a sink to it. Without a sink the message is logged, never lost silently.
"""
from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger(__name__)


class Announcer:
    def __init__(self) -> None:
        self._sink: Callable[[str, str | None], None] | None = None

    def bind(self, sink: Callable[[str, str | None], None]) -> None:
        self._sink = sink

    def say(self, text: str, language: str | None = None) -> None:
        if self._sink is None:
            log.warning("announcement with nobody listening: %s", text)
            return
        try:
            self._sink(text, language)
        except Exception:  # noqa: BLE001 - an announcement must never crash a skill task
            log.exception("announcement sink failed")
