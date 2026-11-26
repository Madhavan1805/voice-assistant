from __future__ import annotations

import numpy as np

from assistant.config import WhisperCfg
from assistant.stt.whisper_local import FasterWhisperSTT, _is_repetitive


class FakeModel:
    def detect_language(self, pcm):
        return "hi", 0.5, [("hi", 0.5), ("ta", 0.2), ("en", 0.25), ("ml", 0.05)]


def test_language_detection_is_restricted_to_allowed_languages():
    stt = FasterWhisperSTT(WhisperCfg(), "auto", ["en", "ta"])
    assert stt._pick_language(FakeModel(), np.zeros(10, dtype=np.float32)) == "en"


def test_fixed_language_skips_detection():
    stt = FasterWhisperSTT(WhisperCfg(), "ta", ["en", "ta"])
    assert stt._pick_language(object(), np.zeros(10, dtype=np.float32)) == "ta"


def test_looping_garbage_is_flagged_but_real_sentences_are_not():
    assert _is_repetitive("Pallipam Dhani " + "ay" * 150)
    assert _is_repetitive("आब " + "वव " * 60)
    assert not _is_repetitive("Hey Kandy, what time is it right now in Chennai?")
    assert not _is_repetitive("So I was thinking we could go to the beach this weekend and then eat dinner.")
