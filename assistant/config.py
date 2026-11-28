"""Typed configuration loaded from ``config.yaml`` (+ ``.env`` for secrets).

Unknown keys raise :class:`ConfigError` so typos fail loudly instead of being
silently ignored. Secrets are read from environment variables, never from YAML.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

from .errors import ConfigError


@dataclass
class AssistantCfg:
    name: str = "Kandy"
    wake_phrases: list[str] = field(
        default_factory=lambda: ["hey kandy", "hi kandy", "ok kandy", "kandy", "hey candy", "hi candy", "candy"]
    )
    wake_required: bool = True
    wake_threshold: float = 0.8       # fuzzy match ratio (STT mishears names)
    followup_window_s: float = 15.0   # no wake word needed right after a reply
    history_turns: int = 10
    skill_timeout_s: float = 20.0
    min_confidence: float = 0.3       # STT confidence below this → "say again"
    post_speech_cooldown_s: float = 0.4
    max_utterance_age_s: float = 10.0  # older queued speech is dropped, not transcribed


@dataclass
class AudioCfg:
    sample_rate: int = 16000
    frame_ms: int = 30
    device: str | int | None = None
    queue_frames: int = 200           # ~6 s of buffer before frames are dropped
    stall_timeout_s: float = 3.0      # no frames for this long → reopen mic
    reopen_delay_s: float = 1.0
    max_reopen_attempts: int = 8


@dataclass
class VADCfg:
    backend: str = "auto"             # auto | webrtc | energy
    webrtc_aggressiveness: int = 2    # 0-3, higher = stricter
    min_rms: float = 120.0            # absolute floor (int16 scale)
    start_ratio: float = 3.0          # speech starts at noise_floor × this
    stop_ratio: float = 1.8           # …and stops below noise_floor × this
    calibration_ms: int = 600
    pre_roll_ms: int = 300
    trigger_ms: int = 90
    end_silence_ms: int = 700
    tail_ms: int = 150
    min_speech_ms: int = 250
    max_utterance_s: float = 15.0


@dataclass
class WhisperCfg:
    model: str = "small"              # tiny / base / small / medium / large-v3
    device: str = "auto"
    compute_type: str = "int8"
    beam_size: int = 1
    vad_filter: bool = False


@dataclass
class CloudSTTCfg:
    enabled: bool = False
    base_url: str = "https://api.openai.com/v1"   # any OpenAI-compatible server
    model: str = "whisper-1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 15.0


@dataclass
class STTCfg:
    language: str = "auto"            # auto | en | ta
    languages: list[str] = field(default_factory=lambda: ["en", "ta"])  # auto-detect picks only from these
    timeout_s: float = 40.0
    breaker_threshold: int = 3
    breaker_reset_s: float = 60.0
    whisper: WhisperCfg = field(default_factory=WhisperCfg)
    cloud: CloudSTTCfg = field(default_factory=CloudSTTCfg)


@dataclass
class OllamaCfg:
    enabled: bool = True
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    timeout_s: float = 30.0


@dataclass
class GroqCfg:
    enabled: bool = True              # skipped automatically if no API key
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "llama-3.3-70b-versatile"
    api_key_env: str = "GROQ_API_KEY"
    timeout_s: float = 15.0


@dataclass
class AnthropicCfg:
    enabled: bool = True              # skipped automatically if no API key
    model: str = "claude-sonnet-5-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    timeout_s: float = 20.0


@dataclass
class LLMCfg:
    order: list[str] = field(default_factory=lambda: ["groq", "ollama", "anthropic"])
    max_tokens: int = 300
    retries: int = 1
    breaker_threshold: int = 3
    breaker_reset_s: float = 60.0
    llm_routing: bool = True          # let the LLM route commands rules didn't catch
    groq: GroqCfg = field(default_factory=GroqCfg)
    ollama: OllamaCfg = field(default_factory=OllamaCfg)
    anthropic: AnthropicCfg = field(default_factory=AnthropicCfg)


@dataclass
class PiperCfg:
    enabled: bool = True
    voices: dict[str, str] = field(
        default_factory=lambda: {"en": "models/piper/en_US-lessac-medium.onnx"}
    )


@dataclass
class EspeakCfg:
    enabled: bool = True
    binary: str = "espeak-ng"
    voices: dict[str, str] = field(default_factory=lambda: {"en": "en-us", "ta": "ta"})
    speed: int = 165


@dataclass
class WindowsTTSCfg:
    enabled: bool = True
    rate: int = 0                     # -10 (slow) … 10 (fast)


@dataclass
class TTSCfg:
    order: list[str] = field(default_factory=lambda: ["piper", "windows", "espeak"])
    timeout_s: float = 15.0
    max_sentence_chars: int = 220
    breaker_threshold: int = 3
    breaker_reset_s: float = 60.0
    piper: PiperCfg = field(default_factory=PiperCfg)
    windows: WindowsTTSCfg = field(default_factory=WindowsTTSCfg)
    espeak: EspeakCfg = field(default_factory=EspeakCfg)


@dataclass
class SystemSkillCfg:
    enabled: bool = True
    # app alias -> argv (no shell!). Edit for your OS.
    apps: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class SmartHomeCfg:
    enabled: bool = False
    backend: str = "mock"             # mock | homeassistant
    base_url: str = "http://homeassistant.local:8123"
    token_env: str = "HOME_ASSISTANT_TOKEN"
    timeout_s: float = 5.0
    devices: dict[str, str] = field(default_factory=dict)  # spoken name -> entity_id


@dataclass
class WhatsAppCfg:
    enabled: bool = False
    profile_dir: str = ".wa_profile"  # keeps you logged in (scan the QR once)
    headless: bool = False            # must be False for the first QR login
    login_timeout_s: float = 90.0
    send_timeout_s: float = 20.0
    contacts: dict[str, str] = field(default_factory=dict)  # spoken name -> +91…


@dataclass
class WeatherCfg:
    default_city: str = "Chennai"
    timeout_s: float = 6.0


@dataclass
class SkillsCfg:
    timezone: str = "Asia/Kolkata"
    routines_file: str = "routines.yaml"
    system: SystemSkillCfg = field(default_factory=SystemSkillCfg)
    smart_home: SmartHomeCfg = field(default_factory=SmartHomeCfg)
    whatsapp: WhatsAppCfg = field(default_factory=WhatsAppCfg)
    weather: WeatherCfg = field(default_factory=WeatherCfg)


@dataclass
class Config:
    log_level: str = "INFO"
    assistant: AssistantCfg = field(default_factory=AssistantCfg)
    audio: AudioCfg = field(default_factory=AudioCfg)
    vad: VADCfg = field(default_factory=VADCfg)
    stt: STTCfg = field(default_factory=STTCfg)
    llm: LLMCfg = field(default_factory=LLMCfg)
    tts: TTSCfg = field(default_factory=TTSCfg)
    skills: SkillsCfg = field(default_factory=SkillsCfg)

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "Config":
        _load_dotenv()
        if path is None or not Path(path).exists():
            if path is not None:
                raise ConfigError(f"config file not found: {path}")
            cfg = cls()
        else:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover
                raise ConfigError("PyYAML is required to read config.yaml") from exc
            try:
                data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
            if not isinstance(data, dict):
                raise ConfigError(f"{path}: top level must be a mapping")
            cfg = _from_dict(cls, data, "config")
        cfg.validate()
        return cfg

    def validate(self) -> None:
        a, v = self.audio, self.vad
        if a.sample_rate not in (8000, 16000, 32000, 48000):
            raise ConfigError("audio.sample_rate must be 8000/16000/32000/48000 (WebRTC VAD limit)")
        if a.frame_ms not in (10, 20, 30):
            raise ConfigError("audio.frame_ms must be 10, 20 or 30 (WebRTC VAD limit)")
        if not 0 <= v.webrtc_aggressiveness <= 3:
            raise ConfigError("vad.webrtc_aggressiveness must be 0-3")
        if v.stop_ratio >= v.start_ratio:
            raise ConfigError("vad.stop_ratio must be lower than vad.start_ratio (hysteresis)")
        if not self.assistant.wake_phrases and self.assistant.wake_required:
            raise ConfigError("assistant.wake_phrases is empty but wake_required is true")


def _from_dict(cls: type, data: dict[str, Any], where: str) -> Any:
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(known)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        hint = hints[name]
        if is_dataclass(hint):
            if value is None:
                continue
            if not isinstance(value, dict):
                raise ConfigError(f"{where}.{name}: expected a mapping")
            value = _from_dict(hint, value, f"{where}.{name}")
        kwargs[name] = value
    return cls(**kwargs)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)
