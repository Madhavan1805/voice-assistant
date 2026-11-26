"""Turns a stream of VAD decisions into complete utterances.

State machine (all sync in :meth:`feed`, so it is trivially unit-testable):

    IDLE ──(trigger_ms of consecutive speech)──▶ ACTIVE ──(end_silence_ms)──▶ emit
      ▲   pre-roll ring buffer keeps the audio    │  max_utterance_s ─────────▶ emit (forced)
      └───────────── that led up to the trigger ◀─┘

Too-short blips (coughs, clicks) are discarded via ``min_speech_ms``.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import AsyncIterator, Callable

import numpy as np

from ..config import VADCfg
from ..events import Utterance
from .input import AudioSource
from .vad import VADEngine

log = logging.getLogger(__name__)


class UtteranceSegmenter:
    def __init__(self, vad: VADEngine, cfg: VADCfg, *, sample_rate: int = 16000, frame_ms: int = 30) -> None:
        self._vad = vad
        self._sr = sample_rate
        self._frame_ms = frame_ms
        self._frame_samples = sample_rate * frame_ms // 1000
        f = lambda ms: max(1, ms // frame_ms)  # noqa: E731
        self._trigger = f(cfg.trigger_ms)
        self._end_silence = f(cfg.end_silence_ms)
        self._tail = f(cfg.tail_ms)
        self._min_speech = f(cfg.min_speech_ms)
        self._max_frames = int(cfg.max_utterance_s * 1000 / frame_ms)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=f(cfg.pre_roll_ms))
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._run = 0
        self._silence = 0
        self._speech_frames = 0
        self._frames: list[np.ndarray] = []
        self._pre_roll.clear()
        self._vad.reset()

    def feed(self, frame: np.ndarray) -> Utterance | None:
        is_speech = self._vad.is_speech(frame)

        if not self._active:
            self._pre_roll.append(frame)
            self._run = self._run + 1 if is_speech else 0
            if self._run >= self._trigger:
                self._active = True
                self._frames = list(self._pre_roll)
                self._pre_roll.clear()
                self._speech_frames = self._run
                self._silence = 0
            return None

        self._frames.append(frame)
        if is_speech:
            self._silence = 0
            self._speech_frames += 1
        else:
            self._silence += 1

        if self._silence >= self._end_silence:
            return self._finish(trim_silence=True)
        if len(self._frames) >= self._max_frames:
            log.debug("utterance hit max length; forcing cut")
            self._vad.relax()
            return self._finish(trim_silence=False)
        return None

    def _finish(self, *, trim_silence: bool) -> Utterance | None:
        frames, speech, silence = self._frames, self._speech_frames, self._silence
        self.reset()
        if speech < self._min_speech:
            log.debug("discarded short blip (%d ms)", speech * self._frame_ms)
            return None
        if trim_silence and silence > self._tail:
            frames = frames[: len(frames) - (silence - self._tail)]
        return Utterance(audio=np.concatenate(frames), sample_rate=self._sr)

    async def utterances(
        self, source: AudioSource, *, muted: Callable[[], bool] | None = None
    ) -> AsyncIterator[Utterance]:
        """Yield utterances from ``source``.

        While ``muted()`` is true (assistant is speaking) frames are discarded so
        it never transcribes its own voice.
        """
        async for frame in source.frames():
            if len(frame) != self._frame_samples:
                continue
            if muted is not None and muted():
                if self._active or self._run or self._pre_roll:
                    self.reset()
                continue
            utt = self.feed(frame)
            if utt is not None:
                yield utt
