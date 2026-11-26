"""Voice Activity Detection that copes with ambient noise.

``AdaptiveEnergyVAD`` tracks a moving noise floor (fast to decay, slow to rise) and
uses hysteresis (higher bar to *start* speech than to *continue* it), so a fan,
AC hum or distant TV gets absorbed into the floor instead of triggering the
assistant. When ``webrtcvad`` is installed we AND it with the energy gate: the
energy gate rejects steady noise WebRTC mislabels as speech, WebRTC rejects loud
non-speech transients (door slam, keyboard) the energy gate would accept.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from ..config import AudioCfg, VADCfg

log = logging.getLogger(__name__)


class VADEngine(ABC):
    @abstractmethod
    def is_speech(self, frame: np.ndarray) -> bool: ...

    def reset(self) -> None:
        """Forget speech state (keep learned noise floor)."""

    def relax(self) -> None:
        """Called when an utterance was force-cut; re-learn what 'quiet' is."""


def frame_rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(frame.astype(np.float32)))))


class AdaptiveEnergyVAD(VADEngine):
    def __init__(
        self,
        frame_ms: int = 30,
        *,
        min_rms: float = 120.0,
        start_ratio: float = 3.0,
        stop_ratio: float = 1.8,
        calibration_ms: int = 600,
        alpha_down: float = 0.10,
        alpha_up: float = 0.01,
    ) -> None:
        self.min_rms = min_rms
        self.start_ratio = start_ratio
        self.stop_ratio = stop_ratio
        self._calib_frames = max(1, calibration_ms // frame_ms)
        self._alpha_down = alpha_down
        self._alpha_up = alpha_up
        self.noise_floor: float | None = None
        self._seen = 0
        self._speaking = False
        self._recent: deque[float] = deque(maxlen=50)

    def is_speech(self, frame: np.ndarray) -> bool:
        rms = frame_rms(frame)
        self._recent.append(rms)

        if self._seen < self._calib_frames:  # assume the room is quiet at start-up
            self._seen += 1
            floor = self.noise_floor if self.noise_floor is not None else rms
            self.noise_floor = floor + (rms - floor) / self._seen
            return False

        floor = self.noise_floor or 1.0
        ratio = self.stop_ratio if self._speaking else self.start_ratio
        speaking = rms > max(self.min_rms, floor * ratio)
        if not speaking:
            alpha = self._alpha_down if rms < floor else self._alpha_up
            self.noise_floor = floor + alpha * (rms - floor)
        self._speaking = speaking
        return speaking

    def reset(self) -> None:
        self._speaking = False

    def relax(self) -> None:
        if self._recent:
            self.noise_floor = float(np.mean(self._recent))
        self._speaking = False


class WebRtcVAD(VADEngine):
    def __init__(self, sample_rate: int, aggressiveness: int = 2) -> None:
        import webrtcvad  # ImportError handled by build_vad

        self._vad = webrtcvad.Vad(aggressiveness)
        self._rate = sample_rate

    def is_speech(self, frame: np.ndarray) -> bool:
        try:
            return bool(self._vad.is_speech(frame.tobytes(), self._rate))
        except Exception:  # noqa: BLE001 - wrong frame length etc.
            return False


class HybridVAD(VADEngine):
    def __init__(self, energy: AdaptiveEnergyVAD, webrtc: WebRtcVAD) -> None:
        self._energy = energy
        self._webrtc = webrtc

    def is_speech(self, frame: np.ndarray) -> bool:
        e = self._energy.is_speech(frame)  # always evaluate: keeps the noise floor learning
        w = self._webrtc.is_speech(frame)
        return e and w

    def reset(self) -> None:
        self._energy.reset()

    def relax(self) -> None:
        self._energy.relax()


def build_vad(vad: VADCfg, audio: AudioCfg) -> VADEngine:
    energy = AdaptiveEnergyVAD(
        audio.frame_ms,
        min_rms=vad.min_rms,
        start_ratio=vad.start_ratio,
        stop_ratio=vad.stop_ratio,
        calibration_ms=vad.calibration_ms,
    )
    if vad.backend == "energy":
        return energy
    try:
        webrtc = WebRtcVAD(audio.sample_rate, vad.webrtc_aggressiveness)
    except ImportError:
        if vad.backend == "webrtc":
            raise
        log.info("webrtcvad not installed — using adaptive energy VAD only")
        return energy
    return HybridVAD(energy, webrtc)
