"""End-to-end pipeline tests with fake audio, STT and TTS (no hardware, no models)."""
from __future__ import annotations

import asyncio

import numpy as np

from assistant.audio.input import ArrayAudioSource
from assistant.audio.segmenter import UtteranceSegmenter
from assistant.audio.vad import AdaptiveEnergyVAD
from assistant.config import AssistantCfg, VADCfg
from assistant.core.wake import WakeGate
from assistant.errors import STTError, TranscriptionEmpty, TTSError
from assistant.events import Transcript
from assistant.pipeline import SpeechPipeline
from assistant.skills.time_skill import TimeSkill
from assistant.stt.base import STTEngine

from .conftest import noise, tone

SR = 16000


class ScriptedSTT(STTEngine):
    name = "scripted"

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)

    async def transcribe(self, utterance):
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


class RecordingSpeaker:
    def __init__(self, fail=False):
        self.said, self.fail = [], fail

    async def say(self, text, language=None):
        if self.fail:
            raise TTSError("speaker broken")
        self.said.append(text)

    async def warmup(self): ...
    async def aclose(self): ...


def _audio(n_utterances: int) -> np.ndarray:
    parts = [noise(SR)]
    for i in range(n_utterances):
        parts += [tone(SR), noise(SR, seed=i + 10)]
    return np.concatenate(parts)


def _pipeline(stt, speaker, dispatcher, n_utterances, wake_required=True):
    vcfg = VADCfg()
    return SpeechPipeline(
        source=ArrayAudioSource(_audio(n_utterances)),
        segmenter=UtteranceSegmenter(AdaptiveEnergyVAD(30), vcfg, sample_rate=SR, frame_ms=30),
        stt=stt,
        dispatcher=dispatcher,
        speaker=speaker,
        wake=WakeGate(["hey madhav"], required=wake_required),
        cfg=AssistantCfg(post_speech_cooldown_s=0.0),
    )


async def _run(p):
    await asyncio.wait_for(p.run(), timeout=15)


async def test_full_flow_wake_phrase_to_spoken_reply(make_assistant):
    d, _ = make_assistant(None, TimeSkill("Asia/Kolkata"))
    speaker = RecordingSpeaker()
    stt = ScriptedSTT(Transcript("Hey Madhav, what time is it?", "en", 0.9))
    await _run(_pipeline(stt, speaker, d, 1))
    assert len(speaker.said) == 1 and speaker.said[0].startswith("It's")


async def test_speech_without_wake_phrase_is_ignored(make_assistant):
    d, _ = make_assistant(None, TimeSkill())
    speaker = RecordingSpeaker()
    stt = ScriptedSTT(Transcript("what time is it", "en", 0.9))
    await _run(_pipeline(stt, speaker, d, 1))
    assert speaker.said == []


async def test_wake_only_then_followup_without_wake_phrase(make_assistant):
    d, _ = make_assistant(None, TimeSkill())
    speaker = RecordingSpeaker()
    stt = ScriptedSTT(Transcript("Hey Madhav", "en", 0.9), Transcript("what time is it", "en", 0.9))
    await _run(_pipeline(stt, speaker, d, 2))
    assert speaker.said[0] == "Yes?"
    assert speaker.said[1].startswith("It's")


async def test_stt_failures_are_survivable_and_reported_once(make_assistant):
    d, _ = make_assistant(None, TimeSkill())
    speaker = RecordingSpeaker()
    stt = ScriptedSTT(
        STTError("engine down"),
        TranscriptionEmpty("noise"),
        STTError("engine down"),
        Transcript("Hey Madhav what time is it", "en", 0.9),
    )
    await _run(_pipeline(stt, speaker, d, 4))
    assert sum("trouble hearing" in s for s in speaker.said) == 1   # rate-limited, not spammy
    assert any(s.startswith("It's") for s in speaker.said)          # pipeline recovered


async def test_low_confidence_asks_user_to_repeat(make_assistant):
    d, _ = make_assistant(None, TimeSkill())
    speaker = RecordingSpeaker()
    stt = ScriptedSTT(Transcript("Hey Madhav mumble mumble", "en", 0.05))
    await _run(_pipeline(stt, speaker, d, 1))
    assert speaker.said == ["Sorry, I didn't catch that. Could you say it again?"]


async def test_tts_failure_does_not_kill_the_pipeline(make_assistant):
    d, _ = make_assistant(None, TimeSkill())
    speaker = RecordingSpeaker(fail=True)
    stt = ScriptedSTT(
        Transcript("Hey Madhav what time is it", "en", 0.9),
        Transcript("Hey Madhav what time is it", "en", 0.9),
    )
    await _run(_pipeline(stt, speaker, d, 2))   # completes without raising


async def test_stage_crash_is_supervised_and_restarts(make_assistant):
    d, _ = make_assistant(None, TimeSkill())
    speaker = RecordingSpeaker()
    stt = ScriptedSTT(
        RuntimeError("model blew up"),                       # unexpected bug in an engine
        Transcript("Hey Madhav what time is it", "en", 0.9),
    )
    await _run(_pipeline(stt, speaker, d, 2))
    assert any(s.startswith("It's") for s in speaker.said)
