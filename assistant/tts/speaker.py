"""Speech output: sentence-level streaming for low perceived latency.

While sentence N is playing, sentence N+1 is already being synthesised, so a long
answer starts speaking after the *first* sentence is ready, not the whole reply.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Protocol

import numpy as np

from ..errors import TTSError
from ..utils.text import clean_for_speech, detect_language, split_sentences
from .base import AudioChunk, TTSEngine

log = logging.getLogger(__name__)


class Speaker(Protocol):
    async def say(self, text: str, language: str | None = None) -> None: ...
    async def warmup(self) -> None: ...
    async def aclose(self) -> None: ...


class AudioPlayer(Protocol):
    async def play(self, chunk: AudioChunk) -> None: ...
    async def stop(self) -> None: ...


class SoundDevicePlayer:
    async def play(self, chunk: AudioChunk) -> None:
        await asyncio.to_thread(self._play_blocking, chunk)

    @staticmethod
    def _play_blocking(chunk: AudioChunk) -> None:
        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:
            raise TTSError(f"audio output unavailable: {exc}") from exc
        try:
            sd.play(np.frombuffer(chunk.pcm, dtype=np.int16), chunk.sample_rate)
            sd.wait()
        except Exception as exc:  # noqa: BLE001
            raise TTSError(f"playback failed: {exc}") from exc

    async def stop(self) -> None:
        try:
            import sounddevice as sd

            sd.stop()  # unblocks the sd.wait() running in the worker thread
        except Exception:  # noqa: BLE001
            pass


class NullPlayer:
    """Records chunks instead of playing them (tests / headless)."""

    def __init__(self) -> None:
        self.played: list[AudioChunk] = []

    async def play(self, chunk: AudioChunk) -> None:
        self.played.append(chunk)

    async def stop(self) -> None:
        pass


class SpeechOutput:
    def __init__(self, engine: TTSEngine, player: AudioPlayer, *, max_sentence_chars: int = 220) -> None:
        self._engine = engine
        self._player = player
        self._max_chars = max_sentence_chars

    async def warmup(self) -> None:
        await self._engine.warmup()

    async def say(self, text: str, language: str | None = None) -> None:
        text = clean_for_speech(text)
        if not text:
            return
        lang = language or detect_language(text)
        sentences = split_sentences(text, self._max_chars)
        queue: asyncio.Queue[AudioChunk | Exception | None] = asyncio.Queue(maxsize=2)

        async def produce() -> None:
            try:
                for sentence in sentences:
                    await queue.put(await self._engine.synthesize(sentence, lang))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - hand the failure to the consumer
                await queue.put(exc)
            else:
                await queue.put(None)

        producer = asyncio.create_task(produce())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item if isinstance(item, TTSError) else TTSError(str(item)) from item
                await self._player.play(item)
        except BaseException:
            producer.cancel()
            await self._player.stop()
            raise
        finally:
            await asyncio.gather(producer, return_exceptions=True)

    async def interrupt(self) -> None:
        await self._player.stop()

    async def aclose(self) -> None:
        await self._engine.aclose()


class ConsoleSpeaker:
    """Degraded mode when no TTS engine is installed: just print the reply."""

    async def say(self, text: str, language: str | None = None) -> None:
        print(f"assistant> {clean_for_speech(text)}", flush=True)

    async def warmup(self) -> None:
        pass

    async def aclose(self) -> None:
        pass
