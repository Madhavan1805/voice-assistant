"""The speech pipeline: mic → VAD → STT → wake gate → dispatcher → TTS.

Four stages run as independent asyncio tasks connected by bounded queues:

    capture ──utt_q──▶ stt ──cmd_q──▶ think ──speech_q──▶ speak
   (mic+VAD)        (Whisper)       (dispatcher)           (Piper)

Why queues: a slow LLM answer never stalls the microphone, and each stage can fail
and restart on its own. Each stage runs under a supervisor that restarts it with
exponential backoff, so a crash in one component never ends the assistant.

Half-duplex: while the assistant speaks, mic frames are discarded so it can't hear
(and answer) itself. Barge-in (interrupting it mid-sentence) would need echo
cancellation and is deliberately out of scope.
"""
from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Callable

from .audio.input import AudioSource
from .audio.segmenter import UtteranceSegmenter
from .config import AssistantCfg
from .core.dispatcher import Dispatcher
from .core.types import SkillResult
from .core.wake import WakeGate
from .errors import AudioUnavailable, STTError, TranscriptionEmpty, TTSError
from .events import Command, Utterance
from .messages import msg
from .stt.base import STTEngine
from .tts.speaker import Speaker
from .utils.text import detect_language

log = logging.getLogger(__name__)


class State(str, Enum):
    IDLE = "idle"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"


def _put_drop_oldest(q: asyncio.Queue, item) -> None:
    """Keep latency bounded: when a stage falls behind, discard the stalest item."""
    if q.full():
        try:
            q.get_nowait()
            log.warning("pipeline queue full; dropped oldest item")
        except asyncio.QueueEmpty:
            pass
    q.put_nowait(item)


class SpeechPipeline:
    def __init__(
        self,
        *,
        source: AudioSource,
        segmenter: UtteranceSegmenter,
        stt: STTEngine,
        dispatcher: Dispatcher,
        speaker: Speaker,
        wake: WakeGate,
        cfg: AssistantCfg,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = source
        self._segmenter = segmenter
        self._stt = stt
        self._dispatcher = dispatcher
        self._speaker = speaker
        self._wake = wake
        self._cfg = cfg
        self._clock = clock

        self._utt_q: asyncio.Queue[Utterance | None] = asyncio.Queue(maxsize=4)
        self._cmd_q: asyncio.Queue[Command | None] = asyncio.Queue(maxsize=4)
        self._speech_q: asyncio.Queue[SkillResult | None] = asyncio.Queue(maxsize=8)
        self._stop = asyncio.Event()
        self._speaking = asyncio.Event()
        self._notified: dict[str, float] = {}
        self._listeners: list[Callable[[State], None]] = []
        self.state = State.IDLE

    # -- public API -------------------------------------------------------
    def on_state(self, callback: Callable[[State], None]) -> None:
        """Subscribe to state changes (e.g. to drive an LED or a status line)."""
        self._listeners.append(callback)

    def request_stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        """Run until the audio source ends or :meth:`request_stop` is called."""
        try:
            await self._speaker.warmup()
            await self._stt.warmup()
        except Exception as exc:  # noqa: BLE001 - warmup is an optimisation only
            log.warning("warmup failed (first reply will be slower): %s", exc)

        stages = {
            "capture": self._capture_stage,
            "stt": self._stt_stage,
            "think": self._think_stage,
            "speak": self._speak_stage,
        }
        tasks = [asyncio.create_task(self._supervise(n, f), name=f"stage-{n}") for n, f in stages.items()]
        stopper = asyncio.create_task(self._stop.wait(), name="stop-waiter")
        log.info("pipeline running")
        try:
            pending = set(tasks)
            while pending and not stopper.done():
                _, waiting = await asyncio.wait(pending | {stopper}, return_when=asyncio.FIRST_COMPLETED)
                pending = waiting - {stopper}
        finally:
            for task in [*tasks, stopper]:
                task.cancel()
            await asyncio.gather(*tasks, stopper, return_exceptions=True)
            log.info("pipeline stopped")

    # -- supervision ------------------------------------------------------
    async def _supervise(self, name: str, stage: Callable[[], "asyncio.Future | object"]) -> None:
        backoff, crashes = 0.5, 0
        while not self._stop.is_set():
            started = self._clock()
            try:
                await stage()  # type: ignore[misc]
                return  # finished normally (finite audio source)
            except asyncio.CancelledError:
                raise
            except AudioUnavailable as exc:
                log.critical("audio backend unavailable, shutting down: %s", exc)
                self._stop.set()
                return
            except Exception:  # noqa: BLE001
                crashes = 1 if self._clock() - started > 30 else crashes + 1
                log.exception("stage %r crashed (#%d); restarting in %.1fs", name, crashes, backoff)
                if name == "capture" and crashes >= 3:
                    self._notify("mic", "mic_trouble", "en", cooldown=120)
                await asyncio.sleep(backoff)
                backoff = 0.5 if crashes == 1 else min(backoff * 2, 15.0)

    def _set_state(self, state: State) -> None:
        if state is not self.state:
            self.state = state
            for cb in self._listeners:
                try:
                    cb(state)
                except Exception:  # noqa: BLE001 - listeners must not break the pipeline
                    log.exception("state listener failed")

    def _notify(self, key: str, text_key: str, lang: str | None, *, cooldown: float) -> None:
        """Speak a canned problem message, rate-limited so failures don't become spam."""
        now = self._clock()
        if now - self._notified.get(key, float("-inf")) < cooldown:
            return
        self._notified[key] = now
        _put_drop_oldest(self._speech_q, SkillResult(msg(text_key, lang), ok=False, language=lang or "en"))

    # -- stages -----------------------------------------------------------
    async def _capture_stage(self) -> None:
        async for utt in self._segmenter.utterances(self._source, muted=self._speaking.is_set):
            log.debug("utterance captured (%.2fs)", utt.duration_s)
            _put_drop_oldest(self._utt_q, utt)
        await self._utt_q.put(None)  # source exhausted (only happens with file sources)

    async def _stt_stage(self) -> None:
        while True:
            utt = await self._utt_q.get()
            if utt is None:
                await self._cmd_q.put(None)
                return
            age = time.time() - utt.captured_at
            if age > self._cfg.max_utterance_age_s:
                log.warning("dropping stale utterance (%.0fs old) — STT is behind", age)
                continue
            self._set_state(State.TRANSCRIBING)
            try:
                transcript = await self._stt.transcribe(utt)
            except TranscriptionEmpty as exc:
                log.debug("nothing recognised: %s", exc)
                self._set_state(State.IDLE)
                continue
            except STTError as exc:
                log.warning("speech recognition failed: %s", exc)
                self._notify("stt", "stt_trouble", "en", cooldown=60)
                self._set_state(State.IDLE)
                continue
            except Exception:  # noqa: BLE001 - an engine bug must not kill the stage
                log.exception("unexpected STT error")
                self._notify("stt", "stt_trouble", "en", cooldown=60)
                self._set_state(State.IDLE)
                continue

            log.info("heard: %r [%s, conf %.2f, %s]", transcript.text, transcript.language,
                     transcript.confidence, transcript.engine)
            lang = transcript.language if transcript.language in ("en", "ta") else detect_language(transcript.text)
            gate = self._wake.check(transcript.text)
            if not gate.accepted:
                log.debug("ignored (no wake phrase)")
                self._set_state(State.IDLE)
                continue
            if gate.woke and not gate.command:  # just "Hey Madhav" → acknowledge, then listen
                _put_drop_oldest(self._speech_q, SkillResult(msg("listening", lang), language=lang))
                continue
            if transcript.confidence < self._cfg.min_confidence:
                self._notify("lowconf", "didnt_catch", lang, cooldown=5)
                self._set_state(State.IDLE)
                continue
            await self._cmd_q.put(Command(gate.command, lang))

    async def _think_stage(self) -> None:
        while True:
            cmd = await self._cmd_q.get()
            if cmd is None:
                await self._speech_q.put(None)
                return
            self._set_state(State.THINKING)
            result = await self._dispatcher.handle(cmd.text, cmd.language)  # never raises
            log.info("reply: %r", result.speech)
            await self._speech_q.put(result)

    async def _speak_stage(self) -> None:
        while True:
            item = await self._speech_q.get()
            if item is None:
                return
            if not item.speech:
                self._set_state(State.IDLE)
                continue
            self._speaking.set()
            self._set_state(State.SPEAKING)
            try:
                await self._speaker.say(item.speech, item.language)
            except TTSError as exc:
                log.error("speech output failed: %s", exc)
                print(f"assistant> {item.speech}", flush=True)   # you still see the reply
            finally:
                await asyncio.sleep(self._cfg.post_speech_cooldown_s)  # let the room echo die
                self._speaking.clear()
                self._wake.extend()  # follow-up questions don't need the wake word
                self._set_state(State.IDLE)
