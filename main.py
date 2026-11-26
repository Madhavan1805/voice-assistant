#!/usr/bin/env python3
"""Entry point.

    python main.py                 # live voice mode (mic → speaker)
    python main.py --text          # type commands instead of speaking (no audio hardware needed)
    python main.py --wav clip.wav  # run a 16 kHz mono WAV through the full pipeline
    python main.py --list-devices  # show audio devices for audio.device in config.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from assistant.audio.input import ArrayAudioSource, list_devices
from assistant.bootstrap import (
    Assistant,
    build_assistant,
    build_pipeline,
    build_speaker,
    build_stt,
)
from assistant.config import Config
from assistant.errors import AssistantError


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local-first voice assistant")
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--text", action="store_true", help="type commands instead of speaking")
    mode.add_argument("--wav", metavar="FILE", help="process a 16 kHz mono 16-bit WAV file")
    mode.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args()


def _setup_logging(level: str, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "urllib3", "selenium", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def _text_repl(assistant: Assistant) -> None:
    print("Text mode — type what you would say (wake phrase not needed). Ctrl-D to quit.")
    loop = asyncio.get_running_loop()
    while True:
        try:
            line = await loop.run_in_executor(None, input, "you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line.strip():
            result = await assistant.dispatcher.handle(line)
            print(f"assistant> {result.speech}")


async def _amain(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    _setup_logging(cfg.log_level, args.verbose)

    if args.list_devices:
        print(list_devices())
        return 0

    assistant = build_assistant(cfg)
    try:
        if args.text:
            await _text_repl(assistant)
            return 0

        stt = build_stt(cfg)
        speaker = build_speaker(cfg)
        source = None
        if args.wav:
            source = ArrayAudioSource.from_wav(args.wav, cfg.audio.frame_ms)
        pipeline = build_pipeline(assistant, stt, speaker, source)

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, pipeline.request_stop)
            except NotImplementedError:  # Windows
                pass
        print(f"Listening… say “{cfg.assistant.wake_phrases[0]}” then your command. Ctrl-C to quit.")
        try:
            await pipeline.run()
        finally:
            await stt.aclose()
            await speaker.aclose()
        return 0
    finally:
        await assistant.aclose()


def main() -> int:
    args = _parse_args()
    try:
        return asyncio.run(_amain(args))
    except AssistantError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
