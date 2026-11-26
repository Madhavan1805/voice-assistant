"""Cloud STT fallback via any OpenAI-compatible ``/audio/transcriptions`` endpoint
(OpenAI, Groq, a self-hosted whisper server, …)."""
from __future__ import annotations

import logging
import os

from ..audio.wavutil import pcm16_to_wav
from ..config import CloudSTTCfg
from ..errors import STTError, TranscriptionEmpty
from ..events import Transcript, Utterance
from .base import STTEngine

log = logging.getLogger(__name__)


class CloudWhisperSTT(STTEngine):
    name = "cloud-whisper"

    def __init__(self, cfg: CloudSTTCfg, language: str = "auto") -> None:
        try:
            import httpx
        except ImportError as exc:
            raise STTError("httpx is not installed") from exc
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise STTError(f"{cfg.api_key_env} is not set")
        self._cfg = cfg
        self._language = None if language in ("", "auto") else language
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {key}"},
            timeout=cfg.timeout_s,
        )
        self._httpx = httpx

    async def transcribe(self, utterance: Utterance) -> Transcript:
        wav = pcm16_to_wav(utterance.audio.tobytes(), utterance.sample_rate)
        data = {"model": self._cfg.model, "response_format": "json"}
        if self._language:
            data["language"] = self._language
        try:
            resp = await self._client.post(
                "/audio/transcriptions", data=data, files={"file": ("speech.wav", wav, "audio/wav")}
            )
            resp.raise_for_status()
            text = (resp.json().get("text") or "").strip()
        except (self._httpx.HTTPError, ValueError) as exc:
            raise STTError(f"cloud STT failed: {exc}") from exc
        if not text:
            raise TranscriptionEmpty("cloud STT returned no text")
        return Transcript(text=text, language=self._language, confidence=0.7, engine=self.name)

    async def aclose(self) -> None:
        await self._client.aclose()
