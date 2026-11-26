from __future__ import annotations

import pytest

from assistant.config import Config
from assistant.errors import ConfigError


def test_defaults_are_valid():
    Config().validate()


def test_shipped_config_yaml_loads():
    cfg = Config.load("config.yaml")
    assert cfg.stt.whisper.model == "small"
    assert cfg.skills.smart_home.devices["fan"] == "switch.fan"


def test_unknown_key_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("audio:\n  sample_rat: 16000\n")
    with pytest.raises(ConfigError, match="sample_rat"):
        Config.load(p)


def test_invalid_values_are_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("audio:\n  sample_rate: 44100\n")
    with pytest.raises(ConfigError):
        Config.load(p)
    p.write_text("vad:\n  start_ratio: 1.0\n  stop_ratio: 2.0\n")
    with pytest.raises(ConfigError, match="hysteresis"):
        Config.load(p)


def test_missing_explicit_file_errors():
    with pytest.raises(ConfigError):
        Config.load("/nonexistent/config.yaml")
