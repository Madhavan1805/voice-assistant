"""Continuous audio capture.

``MicrophoneStream`` bridges PortAudio's callback thread into asyncio through a
bounded queue (oldest frames are dropped under back-pressure rather than letting
latency grow) and transparently re-opens the device if it disappears or stalls.
"""
from __future__ import annotations

import asyncio
import logging
import wave
from pathlib import Path
from typing import AsyncIterator, Protocol

import numpy as np

from ..config import AudioCfg
from ..errors import AudioError, AudioUnavailable

log = logging.getLogger(__name__)


class AudioSource(Protocol):
    sample_rate: int
    frame_samples: int

    def frames(self) -> AsyncIterator[np.ndarray]:
        """Yield fixed-size mono int16 frames."""
        ...


class ArrayAudioSource:
    """Replays a numpy array / WAV file as frames — for tests and ``--wav`` demos."""

    def __init__(
        self,
        samples: np.ndarray,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        realtime: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * frame_ms // 1000
        self._samples = samples.astype(np.int16, copy=False)
        self._frame_s = frame_ms / 1000 if realtime else 0.0

    @classmethod
    def from_wav(cls, path: str | Path, frame_ms: int = 30, realtime: bool = False) -> "ArrayAudioSource":
        with wave.open(str(path), "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise AudioError("WAV must be mono 16-bit")
            rate = w.getframerate()
            samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return cls(samples, rate, frame_ms, realtime)

    async def frames(self) -> AsyncIterator[np.ndarray]:
        n = self.frame_samples
        for i in range(0, len(self._samples) - n + 1, n):
            yield self._samples[i : i + n]
            await asyncio.sleep(self._frame_s)


class MicrophoneStream:
    """sounddevice-backed mic with back-pressure handling and auto-reopen."""

    def __init__(self, cfg: AudioCfg) -> None:
        self._cfg = cfg
        self.sample_rate = cfg.sample_rate
        self.frame_samples = cfg.sample_rate * cfg.frame_ms // 1000
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[np.ndarray] | None = None
        self._stream = None
        self._dropped = 0

    # -- PortAudio thread -> asyncio ---------------------------------------
    def _callback(self, indata, frames, time_info, status) -> None:  # audio thread!
        if status:
            log.debug("audio status: %s", status)
        frame = np.frombuffer(indata, dtype=np.int16).copy()
        try:
            self._loop.call_soon_threadsafe(self._push, frame)  # type: ignore[union-attr]
        except RuntimeError:
            pass  # event loop already closed (shutdown)

    def _push(self, frame: np.ndarray) -> None:
        q = self._queue
        assert q is not None
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._dropped += 1
            if self._dropped % 100 == 1:
                log.warning("audio queue full, dropping frames (%d so far) — pipeline too slow?", self._dropped)
        q.put_nowait(frame)

    # -- device lifecycle --------------------------------------------------
    def _open(self) -> None:
        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:  # OSError: PortAudio library missing
            raise AudioUnavailable(
                f"sounddevice/PortAudio unavailable: {exc}",
                user_message=None,
            ) from exc
        try:
            self._stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self.frame_samples,
                channels=1,
                dtype="int16",
                device=self._cfg.device,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise AudioError(f"cannot open microphone: {exc}") from exc

    def _close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                log.debug("error closing mic stream", exc_info=True)

    async def frames(self) -> AsyncIterator[np.ndarray]:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self._cfg.queue_frames)
        failures = 0
        delay = self._cfg.reopen_delay_s
        while True:
            try:
                self._open()
                log.info("microphone open (%d Hz, %d-sample frames)", self.sample_rate, self.frame_samples)
                failures, delay = 0, self._cfg.reopen_delay_s
                while True:
                    try:
                        frame = await asyncio.wait_for(self._queue.get(), self._cfg.stall_timeout_s)
                    except asyncio.TimeoutError:
                        raise AudioError("microphone stalled (no frames)") from None
                    yield frame
            except AudioUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001 - CancelledError/GeneratorExit are BaseException
                failures += 1
                log.warning("microphone error: %s [%d/%d]", exc, failures, self._cfg.max_reopen_attempts)
                if failures >= self._cfg.max_reopen_attempts:
                    raise AudioError("microphone unavailable after repeated retries") from exc
            finally:
                self._close()
            await asyncio.sleep(delay)
            delay = min(delay * 2, 15.0)
            while not self._queue.empty():  # discard stale audio from before the gap
                self._queue.get_nowait()


def list_devices() -> str:
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        return f"sounddevice unavailable: {exc}"
    return str(sd.query_devices())
