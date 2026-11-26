from __future__ import annotations

import pytest

from assistant.config import Config, WindowsTTSCfg
from assistant.errors import TTSError
from assistant.tts.windows_engine import WindowsSapiTTS


def test_windows_engine_is_inert_off_windows():
    engine = WindowsSapiTTS(WindowsTTSCfg())
    assert not engine.available and not engine.supports("en")


async def test_windows_engine_refuses_unsupported_language():
    with pytest.raises(TTSError):
        await WindowsSapiTTS(WindowsTTSCfg()).synthesize("hi", "en")


def test_shipped_config_lists_windows_engine():
    assert "windows" in Config.load("config.yaml").tts.order
