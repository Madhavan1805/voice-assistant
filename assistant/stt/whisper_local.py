"""Local speech-to-text with faster-whisper (CTranslate2).

Runs in a worker thread so the event loop (and the mic queue) never blocks.
Handles Tamil / English / Tanglish via Whisper's language detection.
"""
from __future__ import annotations

import asyncio
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..config import WhisperCfg
from ..errors import STTError, TranscriptionEmpty
from ..events import Transcript, Utterance
from ..utils.text import normalize
from .base import STTEngine

log = logging.getLogger(__name__)

# Phrases Whisper famously "hears" in silence/noise. Dropped only when the model
# itself is unsure, so a genuine "thank you" in a conversation still gets through.
_PHANTOMS = {
    "thank you", "thanks", "thanks for watching", "thank you for watching",
    "you", "bye", "okay", "subtitles by the amara org community",
}


def _looks_phantom(text: str, confidence: float) -> bool:
    return confidence < 0.55 and normalize(text) in _PHANTOMS


class FasterWhisperSTT(STTEngine):
    name = "faster-whisper"

    def __init__(self, cfg: WhisperCfg, language: str = "auto") -> None:
        self._cfg = cfg
        self._language = None if language in ("", "auto") else language
        self._model = None
        self._load_lock = threading.Lock()
        # ONE worker: a timed-out job can't be killed, so parallel jobs would pile up and
        # starve the CPU (and the microphone). Serialising keeps the machine responsive.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")

    def _ensure_model(self):
        with self._load_lock:
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise STTError("faster-whisper is not installed") from exc
                log.info("loading Whisper model %r (%s)…", self._cfg.model, self._cfg.compute_type)
                self._model = WhisperModel(
                    self._cfg.model, device=self._cfg.device, compute_type=self._cfg.compute_type
                )
        return self._model

    async def warmup(self) -> None:
        await asyncio.get_running_loop().run_in_executor(self._executor, self._ensure_model)

    def _transcribe_sync(self, audio: np.ndarray) -> Transcript:
        model = self._ensure_model()
        pcm = audio.astype(np.float32) / 32768.0
        segments, info = model.transcribe(
            pcm,
            language=self._language,
            beam_size=self._cfg.beam_size,
            vad_filter=self._cfg.vad_filter,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            temperature=0.0,  # no fallback re-decoding: on noisy audio it made each clip 5-10x slower
        )
        kept = [s for s in segments if s.no_speech_prob < 0.6]  # consume the generator
        text = " ".join(s.text.strip() for s in kept).strip()
        if not kept or not text:
            raise TranscriptionEmpty("no speech detected")
        avg_logprob = sum(s.avg_logprob for s in kept) / len(kept)
        confidence = max(0.0, min(1.0, math.exp(avg_logprob)))
        if _looks_phantom(text, confidence):
            raise TranscriptionEmpty(f"discarded likely hallucination: {text!r}")
        return Transcript(text=text, language=info.language, confidence=confidence, engine=self.name)

    async def transcribe(self, utterance: Utterance) -> Transcript:
        try:
            return await asyncio.get_running_loop().run_in_executor(
                self._executor, self._transcribe_sync, utterance.audio
            )
        except (STTError, TranscriptionEmpty):
            raise
        except Exception as exc:  # noqa: BLE001 - model/runtime failures
            raise STTError(f"faster-whisper failed: {exc}") from exc

    async def aclose(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
