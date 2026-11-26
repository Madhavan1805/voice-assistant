"""eSpeak-NG: robotic but tiny, instant, and speaks 100+ languages incl. Tamil.
Used as the fallback voice when Piper has no voice for a language (or isn't installed)."""
from __future__ import annotations

import asyncio
import logging
import shutil

from ..audio.wavutil import wav_to_pcm16
from ..config import EspeakCfg
from ..errors import TTSError
from .base import AudioChunk, TTSEngine

log = logging.getLogger(__name__)


class EspeakTTS(TTSEngine):
    name = "espeak"

    def __init__(self, cfg: EspeakCfg, timeout_s: float = 10.0) -> None:
        self._cfg = cfg
        self._timeout = timeout_s

    def supports(self, language: str) -> bool:
        return language in self._cfg.voices and shutil.which(self._cfg.binary) is not None

    async def synthesize(self, text: str, language: str) -> AudioChunk:
        voice = self._cfg.voices.get(language)
        if voice is None:
            raise TTSError(f"espeak has no voice mapped for {language!r}")
        try:
            # Text goes via stdin so it can never be parsed as a command-line option.
            proc = await asyncio.create_subprocess_exec(
                self._cfg.binary, "-v", voice, "-s", str(self._cfg.speed), "--stdout",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise TTSError(f"cannot run {self._cfg.binary}: {exc}") from exc
        try:
            out, err = await asyncio.wait_for(proc.communicate(text.encode("utf-8")), self._timeout)
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise TTSError("espeak timed out") from exc
        if proc.returncode != 0:
            raise TTSError(f"espeak exited {proc.returncode}: {err.decode(errors='replace')[:200]}")
        try:
            pcm, rate = wav_to_pcm16(out)
        except ValueError as exc:
            raise TTSError(f"espeak returned unreadable audio: {exc}") from exc
        if not pcm:
            raise TTSError("espeak produced no audio")
        return AudioChunk(pcm, rate)
