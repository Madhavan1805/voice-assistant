from __future__ import annotations

import numpy as np

from assistant.audio.input import ArrayAudioSource
from assistant.audio.segmenter import UtteranceSegmenter
from assistant.audio.vad import AdaptiveEnergyVAD
from assistant.config import VADCfg

from .conftest import noise, tone

SR, FRAME = 16000, 480  # 30 ms


def _segmenter(**vad_kw) -> UtteranceSegmenter:
    return UtteranceSegmenter(AdaptiveEnergyVAD(30), VADCfg(**vad_kw), sample_rate=SR, frame_ms=30)


def _run(seg: UtteranceSegmenter, audio: np.ndarray):
    out = []
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        u = seg.feed(audio[i : i + FRAME])
        if u is not None:
            out.append(u)
    return out


def test_detects_speech_in_noise_and_pads_with_preroll():
    audio = np.concatenate([noise(SR), tone(SR) + noise(SR), noise(SR, seed=2)])
    utts = _run(_segmenter(), audio)
    assert len(utts) == 1
    assert 0.9 <= utts[0].duration_s <= 1.7  # ~1 s of speech + pre-roll + tail


def test_steady_noise_never_triggers():
    assert _run(_segmenter(), noise(SR * 6, amp=60)) == []


def test_noise_floor_adapts_to_a_loud_hum():
    seg = _segmenter()
    hum = noise(SR * 8, amp=200)  # loud fan; well above the min_rms floor
    assert _run(seg, hum) == []
    assert seg._vad.noise_floor > 100


def test_short_blip_is_discarded():
    audio = np.concatenate([noise(SR), tone(int(SR * 0.12)), noise(SR, seed=3)])
    assert _run(_segmenter(), audio) == []


def test_two_utterances_separated_by_silence():
    gap = noise(SR, seed=4)
    audio = np.concatenate([noise(SR), tone(SR), gap, tone(SR), noise(SR, seed=5)])
    assert len(_run(_segmenter(), audio)) == 2


def test_nonstop_loud_sound_is_force_cut_once_then_absorbed_as_noise():
    # 5 s of unbroken loudness (a TV, a drill): emit one capped utterance, then relearn the
    # noise floor so the same sound doesn't re-trigger the assistant over and over.
    audio = np.concatenate([noise(SR), tone(SR * 5), noise(SR, seed=6)])
    utts = _run(_segmenter(max_utterance_s=2.0), audio)
    assert len(utts) == 1
    assert utts[0].duration_s <= 2.2


async def test_muted_frames_are_dropped():
    audio = np.concatenate([noise(SR), tone(SR), noise(SR, seed=7)])
    seg = _segmenter()
    got = [u async for u in seg.utterances(ArrayAudioSource(audio), muted=lambda: True)]
    assert got == []
    seg = _segmenter()
    got = [u async for u in seg.utterances(ArrayAudioSource(audio), muted=lambda: False)]
    assert len(got) == 1
