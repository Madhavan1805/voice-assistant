# Voice Assistant (local-first, async)

An Alexa-style assistant in Python: continuous listening with noise-robust VAD → local
speech-to-text → intent routing → skills → local text-to-speech. Cloud services are
optional fallbacks, never requirements.

```
mic ─▶ VAD/segmenter ─▶ STT ─▶ wake gate ─▶ Dispatcher ─▶ TTS ─▶ speaker
      (adaptive noise)  (Whisper) (fuzzy)   ├ routines      (Piper → eSpeak)
                                            ├ system / smart_home / whatsapp
                                            ├ time
                                            └ knowledge (Ollama → Claude)
```

## Layout

| Path | Role |
|---|---|
| `assistant/audio/` | `input.py` mic stream (auto-reopen), `vad.py` adaptive-noise VAD, `segmenter.py` utterance state machine |
| `assistant/stt/` | faster-whisper (local), OpenAI-compatible cloud, `fallback.py` chain with circuit breakers |
| `assistant/nlp/` | `llm.py` Ollama/Anthropic with fallback, `intent.py` rules → LLM router → fallback |
| `assistant/core/` | `dispatcher.py` (never raises), `skill.py` plug-in API, `wake.py`, `context.py` |
| `assistant/skills/` | time, knowledge, routines, system, smart_home, whatsapp |
| `assistant/tts/` | Piper, eSpeak-NG, fallback chain, sentence-streaming `speaker.py` |
| `assistant/pipeline.py` | 4 supervised async stages joined by bounded queues |
| `assistant/bootstrap.py` | composition root; missing optional deps degrade gracefully |

## Run

```bash
python -m venv .venv && source .venv/bin/activate
sudo apt install libportaudio2 espeak-ng        # Linux; Tamil works via espeak-ng
pip install -r requirements.txt
cp .env.example .env                            # add ANTHROPIC_API_KEY if you want the cloud fallback

python main.py --text          # no mic needed: type commands
python main.py --list-devices
python main.py                 # live voice. Say: "Hey Madhav, what time is it?"
```

Local LLM: install [Ollama](https://ollama.com), then `ollama pull llama3.2:3b`.

### Piper voices
Download a `.onnx` and its `.onnx.json` from
<https://huggingface.co/rhasspy/piper-voices> into `models/piper/` and match the path in
`config.yaml` (`tts.piper.voices`). Without one, the assistant uses eSpeak-NG.

## Reliability design
- **Noise:** moving noise floor + start/stop hysteresis; WebRTC VAD is ANDed in when installed.
- **STT:** per-engine timeout and circuit breaker; "heard nothing" is distinct from "engine failed".
- **LLM:** retry, then next backend; if all fail the user hears a friendly message.
- **Pipeline:** each stage is supervised and restarts with backoff; the mic reopens itself.
- **Safety:** system commands come only from the `apps` allow-list (argv, no shell); WhatsApp
  sends only to configured contacts and only after you say "yes" to a read-back.

## Add a skill
Subclass `Skill` (`assistant/core/skill.py`), give it `name`, `description`, `patterns`,
implement `handle()`, and register it in `assistant/bootstrap.py`.

## Known limits
- Wake word is matched on the transcript (every utterance is transcribed). For lower CPU use,
  swap `WakeGate` for openWakeWord/Porcupine.
- Half-duplex: no barge-in while it speaks.
- WhatsApp Web selectors break when WhatsApp changes its DOM; automation may breach its ToS.
- Audio, Whisper, Piper, Selenium and Home Assistant paths are unit-tested with fakes; they
  haven't been exercised against real hardware yet.

## Tests
`pip install -r requirements-dev.txt && pytest`
