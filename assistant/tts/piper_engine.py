"""Piper TTS: fast, high-quality, fully local neural voices (ONNX, CPU-friendly).

Download a voice (.onnx + .onnx.json) from https://huggingface.co/rhasspy/piper-voices
and point ``tts.piper.voices`` at it. The adapter below handles both the 1.2.x
(``synthesize_stream_raw``) and the 1.3+ (``synthesize`` → AudioChunk) Python APIs.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

from ..config import PiperCfg
from ..errors import TTSError
from .base import AudioChunk, TTSEngine

log = logging.getLogger(__name__)


class PiperTTS(TTSEngine):
    name = "piper"

    def __init__(self, cfg: PiperCfg) -> None:
        self._paths = {lang: Path(p) for lang, p in cfg.voices.items()}
        self._voices: dict[str, object] = {}
        self._lock = threading.Lock()

    def supports(self, language: str) -> bool:
        path = self._paths.get(language)
        return path is not None and path.exists()

    def _voice(self, language: str):
        with self._lock:
            voice = self._voices.get(language)
            if voice is None:
                try:
                    from piper import PiperVoice
                except ImportError as exc:
                    raise TTSError("piper-tts is not installed") from exc
                path = self._paths.get(language)
                if path is None or not path.exists():
                    raise TTSError(f"no Piper voice file for language {language!r}")
                log.info("loading Piper voice %s", path)
                voice = PiperVoice.load(str(path))
                self._voices[language] = voice
            return voice

    def _synth_sync(self, text: str, language: str) -> AudioChunk:
        voice = self._voice(language)
        rate = int(voice.config.sample_rate)
        if hasattr(voice, "synthesize_stream_raw"):            # piper-tts 1.2.x
            pcm = b"".join(voice.synthesize_stream_raw(text))
        else:                                                  # piper-tts >= 1.3
            pcm = b"".join(chunk.audio_int16_bytes for chunk in voice.synthesize(text))
        if not pcm:
            raise TTSError("piper produced no audio")
        return AudioChunk(pcm, rate)

    async def warmup(self) -> None:
        for lang in self._paths:
            if self.supports(lang):
                await asyncio.to_thread(self._voice, lang)

    async def synthesize(self, text: str, language: str) -> AudioChunk:
        try:
            return await asyncio.to_thread(self._synth_sync, text, language)
        except TTSError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"piper failed: {exc}") from exc
