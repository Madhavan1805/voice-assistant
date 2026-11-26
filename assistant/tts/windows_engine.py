"""Windows built-in speech (SAPI via PowerShell): nothing to install.

Gives decent English voices out of the box, and Tamil too if the Windows Tamil
language pack (with its voice) is installed. Each sentence spawns one short-lived
PowerShell process, so latency is a bit higher than Piper's in-process synthesis.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ..audio.wavutil import wav_to_pcm16
from ..config import WindowsTTSCfg
from ..errors import TTSError
from .base import AudioChunk, TTSEngine

log = logging.getLogger(__name__)

_CREATE_NO_WINDOW = 0x08000000

_LIST_PS1 = (
    "Add-Type -AssemblyName System.Speech;"
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
    "$s.GetInstalledVoices() | Where-Object { $_.Enabled } | "
    "ForEach-Object { $_.VoiceInfo.Culture.TwoLetterISOLanguageName }"
)

_SPEAK_PS1 = r"""param([string]$lang, [string]$out, [int]$rate)
Add-Type -AssemblyName System.Speech
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$text = [Console]::In.ReadToEnd()
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice = $s.GetInstalledVoices() | Where-Object {
    $_.Enabled -and $_.VoiceInfo.Culture.TwoLetterISOLanguageName -eq $lang
} | Select-Object -First 1
if ($null -eq $voice) { exit 3 }
$s.SelectVoice($voice.VoiceInfo.Name)
$s.Rate = $rate
$s.SetOutputToWaveFile($out)
$s.Speak($text)
$s.Dispose()
"""


class WindowsSapiTTS(TTSEngine):
    name = "windows"

    def __init__(self, cfg: WindowsTTSCfg, timeout_s: float = 15.0) -> None:
        self._cfg = cfg
        self._timeout = timeout_s
        self._ps = shutil.which("powershell") or shutil.which("pwsh") if sys.platform == "win32" else None
        self._langs: set[str] = set()
        self._script: Path | None = None
        if self._ps:
            self._langs = self._detect_languages()
            if self._langs:
                folder = Path(tempfile.gettempdir()) / "voice_assistant_tts"
                folder.mkdir(exist_ok=True)
                self._script = folder / "speak.ps1"
                self._script.write_text(_SPEAK_PS1, encoding="utf-8")
                log.info("Windows voices available for: %s", ", ".join(sorted(self._langs)))

    @property
    def available(self) -> bool:
        return bool(self._langs)

    def _detect_languages(self) -> set[str]:
        try:
            out = subprocess.run(
                [self._ps, "-NoProfile", "-NonInteractive", "-Command", _LIST_PS1],
                capture_output=True, text=True, timeout=20,
                creationflags=_CREATE_NO_WINDOW,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("could not list Windows voices: %s", exc)
            return set()
        return {line.strip().lower() for line in out.splitlines() if line.strip()}

    def supports(self, language: str) -> bool:
        return language in self._langs

    async def synthesize(self, text: str, language: str) -> AudioChunk:
        if not self.supports(language) or self._script is None:
            raise TTSError(f"no Windows voice for {language!r}")
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            proc = await asyncio.create_subprocess_exec(
                self._ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                "-File", str(self._script), language, wav_path, str(self._cfg.rate),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=_CREATE_NO_WINDOW,
            )
            try:
                _, err = await asyncio.wait_for(proc.communicate(text.encode("utf-8")), self._timeout)
            except asyncio.TimeoutError as exc:
                proc.kill()
                await proc.wait()
                raise TTSError("Windows speech timed out") from exc
            if proc.returncode != 0:
                raise TTSError(f"powershell exited {proc.returncode}: {err.decode(errors='replace')[:200]}")
            try:
                pcm, rate = wav_to_pcm16(Path(wav_path).read_bytes())
            except (OSError, ValueError) as exc:
                raise TTSError(f"unreadable Windows speech audio: {exc}") from exc
            if not pcm:
                raise TTSError("Windows speech produced no audio")
            return AudioChunk(pcm, rate)
        except OSError as exc:
            raise TTSError(f"cannot run powershell: {exc}") from exc
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
