"""Composition root: the only place that knows about every concrete class.

Everything optional degrades gracefully — a missing library or model file means that
backend is skipped (with a log line saying why), never a crash at start-up.
"""
from __future__ import annotations

import importlib.util
import logging
import shutil
import sys
from dataclasses import dataclass

from .audio.input import AudioSource, MicrophoneStream
from .audio.segmenter import UtteranceSegmenter
from .audio.vad import build_vad
from .config import Config
from .core.announce import Announcer
from .core.context import ConversationContext
from .core.dispatcher import Dispatcher
from .core.skill import SkillRegistry
from .core.wake import WakeGate
from .errors import ConfigError, LLMError, STTError
from .nlp.intent import IntentParser
from .nlp.llm import AnthropicClient, FallbackLLM, GroqClient, LLMClient, OllamaClient
from .pipeline import SpeechPipeline
from .skills.calc import CalcSkill
from .skills.knowledge import KnowledgeSkill
from .skills.routines import RoutineSkill, load_routines
from .skills.smalltalk import SmallTalkSkill
from .skills.smart_home import SmartHomeSkill, build_backend
from .skills.system import SystemSkill
from .skills.time_skill import TimeSkill
from .skills.timer import TimerSkill
from .skills.weather import WeatherSkill
from .skills.web import WebSkill
from .skills.whatsapp import WhatsAppSkill
from .stt.base import STTEngine
from .stt.cloud import CloudWhisperSTT
from .stt.fallback import FallbackSTT
from .stt.whisper_local import FasterWhisperSTT
from .tts.espeak_engine import EspeakTTS
from .tts.fallback import FallbackTTS
from .tts.piper_engine import PiperTTS
from .tts.windows_engine import WindowsSapiTTS
from .tts.speaker import ConsoleSpeaker, SoundDevicePlayer, Speaker, SpeechOutput

log = logging.getLogger(__name__)


@dataclass
class Assistant:
    """Everything needed for text mode (no audio hardware required)."""

    cfg: Config
    llm: LLMClient | None
    registry: SkillRegistry
    dispatcher: Dispatcher
    announcer: Announcer

    async def aclose(self) -> None:
        await self.registry.aclose()
        if self.llm is not None:
            await self.llm.aclose()


def build_llm(cfg: Config) -> LLMClient | None:
    clients: list[LLMClient] = []
    for name in cfg.llm.order:
        try:
            if name == "groq" and cfg.llm.groq.enabled:
                clients.append(GroqClient(cfg.llm.groq))
            elif name == "ollama" and cfg.llm.ollama.enabled:
                clients.append(OllamaClient(cfg.llm.ollama))
            elif name == "anthropic" and cfg.llm.anthropic.enabled:
                clients.append(AnthropicClient(cfg.llm.anthropic))
            elif name not in ("groq", "ollama", "anthropic"):
                log.warning("unknown LLM backend %r in llm.order", name)
        except LLMError as exc:
            log.info("LLM backend %r skipped: %s", name, exc)
    if not clients:
        log.warning("no LLM backend available — general questions will get an 'offline' reply")
        return None
    log.info("LLM chain: %s", " → ".join(c.name for c in clients))
    return FallbackLLM(clients, cfg.llm)


def build_assistant(cfg: Config) -> Assistant:
    llm = build_llm(cfg)
    registry = SkillRegistry()

    routines = RoutineSkill(load_routines(cfg.skills.routines_file))
    registry.register(routines)

    if cfg.skills.whatsapp.enabled:
        registry.register(WhatsAppSkill(cfg.skills.whatsapp))
    if cfg.skills.smart_home.enabled:
        try:
            registry.register(SmartHomeSkill(cfg.skills.smart_home, build_backend(cfg.skills.smart_home)))
        except ConfigError as exc:
            log.warning("smart_home skill disabled: %s", exc)
    if cfg.skills.system.enabled:
        registry.register(SystemSkill(cfg.skills.system))
    registry.register(TimeSkill(cfg.skills.timezone))
    announcer = Announcer()
    registry.register(TimerSkill(announcer))
    registry.register(WeatherSkill(cfg.skills.weather))
    registry.register(WebSkill())
    registry.register(CalcSkill())
    registry.register(SmallTalkSkill(cfg.assistant.name))
    registry.register(KnowledgeSkill(llm, cfg.assistant.name, cfg.llm.max_tokens))

    parser = IntentParser(registry, llm, llm_routing=cfg.llm.llm_routing)
    context = ConversationContext(cfg.assistant.history_turns)
    dispatcher = Dispatcher(
        registry, parser, context, skill_timeout_s=cfg.assistant.skill_timeout_s
    )
    routines.bind(dispatcher.run_intent)
    log.info("skills: %s", ", ".join(s.name for s in registry.all()))
    return Assistant(cfg, llm, registry, dispatcher, announcer)


def build_stt(cfg: Config) -> STTEngine:
    engines: list[STTEngine] = []
    if importlib.util.find_spec("faster_whisper") is not None:
        engines.append(FasterWhisperSTT(cfg.stt.whisper, cfg.stt.language, cfg.stt.languages))
    else:
        log.warning("faster-whisper is not installed; local speech recognition unavailable")
    if cfg.stt.cloud.enabled:
        try:
            engines.append(CloudWhisperSTT(cfg.stt.cloud, cfg.stt.language))
        except STTError as exc:
            log.warning("cloud STT disabled: %s", exc)
    if not engines:
        raise ConfigError(
            "No speech-to-text engine available. `pip install faster-whisper` "
            "(or enable stt.cloud in config.yaml)."
        )
    return FallbackSTT(
        engines,
        timeout_s=cfg.stt.timeout_s,
        breaker_threshold=cfg.stt.breaker_threshold,
        breaker_reset_s=cfg.stt.breaker_reset_s,
    )


def build_speaker(cfg: Config) -> Speaker:
    engines = []
    for name in cfg.tts.order:
        if name == "piper" and cfg.tts.piper.enabled:
            if importlib.util.find_spec("piper") is None:
                log.info("piper-tts not installed; skipping Piper")
                continue
            piper = PiperTTS(cfg.tts.piper)
            if not any(piper.supports(lang) for lang in cfg.tts.piper.voices):
                log.warning("no Piper voice file found (see README → Piper voices); skipping Piper")
                continue
            engines.append(piper)
        elif name == "windows" and cfg.tts.windows.enabled:
            if sys.platform != "win32":
                continue
            windows = WindowsSapiTTS(cfg.tts.windows, cfg.tts.timeout_s)
            if not windows.available:
                log.warning("Windows speech voices not found; skipping")
                continue
            engines.append(windows)
        elif name == "espeak" and cfg.tts.espeak.enabled:
            if shutil.which(cfg.tts.espeak.binary) is None:
                log.info("%s not found; skipping eSpeak", cfg.tts.espeak.binary)
                continue
            engines.append(EspeakTTS(cfg.tts.espeak))
    if not engines:
        log.warning("no TTS engine available — replies will be printed instead of spoken")
        return ConsoleSpeaker()
    log.info("TTS chain: %s", " → ".join(e.name for e in engines))
    chain = FallbackTTS(
        engines,
        timeout_s=cfg.tts.timeout_s,
        breaker_threshold=cfg.tts.breaker_threshold,
        breaker_reset_s=cfg.tts.breaker_reset_s,
    )
    return SpeechOutput(chain, SoundDevicePlayer(), max_sentence_chars=cfg.tts.max_sentence_chars)


def build_pipeline(
    assistant: Assistant,
    stt: STTEngine,
    speaker: Speaker,
    source: AudioSource | None = None,
) -> SpeechPipeline:
    cfg = assistant.cfg
    source = source or MicrophoneStream(cfg.audio)
    segmenter = UtteranceSegmenter(
        build_vad(cfg.vad, cfg.audio),
        cfg.vad,
        sample_rate=cfg.audio.sample_rate,
        frame_ms=cfg.audio.frame_ms,
    )
    wake = WakeGate(
        cfg.assistant.wake_phrases,
        required=cfg.assistant.wake_required,
        threshold=cfg.assistant.wake_threshold,
        followup_window_s=cfg.assistant.followup_window_s,
    )
    pipeline = SpeechPipeline(
        source=source,
        segmenter=segmenter,
        stt=stt,
        dispatcher=assistant.dispatcher,
        speaker=speaker,
        wake=wake,
        cfg=cfg.assistant,
    )
    assistant.announcer.bind(pipeline.announce)   # timers/reminders can speak on their own
    return pipeline
