from __future__ import annotations

import asyncio

import numpy as np
import pytest

from assistant.audio.wavutil import pcm16_to_wav, wav_to_pcm16
from assistant.errors import LLMError, STTError, TranscriptionEmpty, TTSError
from assistant.events import Transcript, Utterance
from assistant.nlp.intent import extract_json
from assistant.nlp.llm import FallbackLLM
from assistant.stt.base import STTEngine
from assistant.stt.fallback import FallbackSTT
from assistant.tts.base import AudioChunk, TTSEngine
from assistant.tts.fallback import FallbackTTS
from assistant.tts.speaker import NullPlayer, SpeechOutput
from assistant.utils.retry import CircuitBreaker, retry_async
from assistant.utils.text import clean_for_speech, detect_language, split_sentences

from .conftest import FakeLLM

UTT = Utterance(np.zeros(1600, dtype=np.int16))


class StubSTT(STTEngine):
    def __init__(self, name, outcome):
        self.name, self.outcome, self.calls = name, outcome, 0

    async def transcribe(self, utterance):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


# -- STT fallback ------------------------------------------------------
async def test_stt_falls_back_when_primary_fails():
    good = Transcript("hello", "en", 0.9, "b")
    chain = FallbackSTT([StubSTT("a", STTError("down")), StubSTT("b", good)])
    assert (await chain.transcribe(UTT)).text == "hello"


async def test_stt_empty_is_not_retried_on_other_engines():
    second = StubSTT("b", Transcript("x"))
    chain = FallbackSTT([StubSTT("a", TranscriptionEmpty("silence")), second])
    with pytest.raises(TranscriptionEmpty):
        await chain.transcribe(UTT)
    assert second.calls == 0


async def test_stt_all_fail_and_circuit_opens():
    a, b = StubSTT("a", STTError("down")), StubSTT("b", STTError("down"))
    chain = FallbackSTT([a, b], breaker_threshold=2, breaker_reset_s=999)
    for _ in range(4):
        with pytest.raises(STTError):
            await chain.transcribe(UTT)
    assert a.calls == 2 and b.calls == 2   # breakers stopped hammering the dead engines


async def test_sole_stt_engine_is_never_locked_out():
    a = StubSTT("a", STTError("down"))
    chain = FallbackSTT([a], breaker_threshold=1, breaker_reset_s=999)
    for _ in range(3):
        with pytest.raises(STTError):
            await chain.transcribe(UTT)
    assert a.calls == 3


async def test_stt_timeout_counts_as_failure():
    class Hang(StubSTT):
        async def transcribe(self, utterance):
            await asyncio.sleep(5)

    chain = FallbackSTT([Hang("slow", None), StubSTT("ok", Transcript("hi"))], timeout_s=0.05)
    assert (await chain.transcribe(UTT)).text == "hi"


# -- LLM fallback ------------------------------------------------------
async def test_llm_retries_then_falls_back():
    flaky = FakeLLM(LLMError("1"), LLMError("2"))
    flaky.name = "flaky"
    backup = FakeLLM("from backup")
    backup.name = "backup"
    llm = FallbackLLM([flaky, backup])
    assert await llm.complete([{"role": "user", "content": "hi"}]) == "from backup"
    assert len(flaky.calls) == 2   # initial try + 1 retry


async def test_llm_all_fail_raises():
    a = FakeLLM(LLMError("x"), LLMError("x"))
    with pytest.raises(LLMError):
        await FallbackLLM([a]).complete([{"role": "user", "content": "hi"}])


def test_extract_json_tolerates_prose_and_fences():
    assert extract_json('Sure! ```json\n{"skill": "time"}\n``` done') == {"skill": "time"}
    assert extract_json("no json here") is None
    assert extract_json("{broken") is None


# -- retry / breaker ---------------------------------------------------
async def test_retry_async_eventually_succeeds_and_gives_up():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("nope")
        return "ok"

    assert await retry_async(flaky, attempts=3, base_delay=0.001) == "ok"
    calls["n"] = -10
    with pytest.raises(ValueError):
        await retry_async(flaky, attempts=2, base_delay=0.001)


def test_circuit_breaker_half_open_recovery():
    t = {"now": 0.0}
    cb = CircuitBreaker(2, 30, clock=lambda: t["now"])
    cb.record_failure(); cb.record_failure()
    assert cb.state == "open" and not cb.allow()
    t["now"] = 31
    assert cb.state == "half-open" and cb.allow()
    cb.record_success()
    assert cb.state == "closed"


# -- TTS ---------------------------------------------------------------
class StubTTS(TTSEngine):
    def __init__(self, name, langs=("en",), fail=False):
        self.name, self.langs, self.fail, self.spoken = name, langs, fail, []

    def supports(self, language):
        return language in self.langs

    async def synthesize(self, text, language):
        if self.fail:
            raise TTSError("broken")
        self.spoken.append(text)
        return AudioChunk(b"\x00\x00" * 10, 22050)


async def test_tts_chain_picks_engine_by_language_and_health():
    piper, espeak = StubTTS("piper", ("en",), fail=True), StubTTS("espeak", ("en", "ta"))
    chain = FallbackTTS([piper, espeak])
    await chain.synthesize("hi", "en")           # piper fails → espeak
    await chain.synthesize("வணக்கம்", "ta")       # piper doesn't support ta
    assert espeak.spoken == ["hi", "வணக்கம்"]
    with pytest.raises(TTSError):
        await chain.synthesize("bonjour", "fr")


async def test_speech_output_streams_sentence_by_sentence_and_cleans_text():
    engine, player = StubTTS("e"), NullPlayer()
    out = SpeechOutput(engine, player)
    await out.say("**Hello** there. Visit https://example.com now! Third one?", "en")
    assert engine.spoken == ["Hello there.", "Visit link now!", "Third one?"]
    assert len(player.played) == 3


async def test_speech_output_failure_surfaces_as_tts_error():
    out = SpeechOutput(FallbackTTS([StubTTS("e", fail=True)]), NullPlayer())
    with pytest.raises(TTSError):
        await out.say("hello", "en")


# -- text utils / wav --------------------------------------------------
def test_text_helpers():
    assert detect_language("இப்போ நேரம் என்ன") == "ta"
    assert detect_language("what time is it") == "en"
    assert clean_for_speech("# Title\n- item **bold** `x` 😀") == "Title\nitem bold x"
    assert split_sentences("One. Two! " + "word " * 100, 50)[0] == "One."
    assert all(len(s) <= 50 for s in split_sentences("word " * 100, 50))


def test_wav_roundtrip_and_streamed_header():
    pcm = np.arange(100, dtype=np.int16).tobytes()
    wav = pcm16_to_wav(pcm, 22050)
    assert wav_to_pcm16(wav) == (pcm, 22050)
    # espeak-style streamed WAV: data chunk size is 0xFFFFFFFF
    streamed = wav[:40] + b"\xff\xff\xff\xff" + wav[44:]
    assert wav_to_pcm16(streamed) == (pcm, 22050)
    with pytest.raises(ValueError):
        wav_to_pcm16(b"garbage")
