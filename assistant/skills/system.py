"""Local system control: open allow-listed apps, change volume.

Security: commands come ONLY from the ``apps`` allow-list in config.yaml (argv lists,
never a shell string) or from fixed volume templates. Spoken text is never executed.
"""
from __future__ import annotations

import asyncio
import difflib
import logging
import shutil
import sys
from typing import Callable

from ..config import SystemSkillCfg
from ..core.context import ConversationContext
from ..core.skill import Skill
from ..core.types import Intent, SkillResult
from ..errors import SkillError
from ..utils.text import normalize

log = logging.getLogger(__name__)


def volume_command(
    action: str,
    level: int | None = None,
    platform: str = sys.platform,
    which: Callable[[str], str | None] = shutil.which,
) -> list[str] | None:
    """Build the OS command for a volume action, or ``None`` if unsupported here."""
    if platform.startswith("linux"):
        if which("pactl"):
            sink = "@DEFAULT_SINK@"
            return {
                "set": ["pactl", "set-sink-volume", sink, f"{level}%"],
                "up": ["pactl", "set-sink-volume", sink, "+10%"],
                "down": ["pactl", "set-sink-volume", sink, "-10%"],
                "mute": ["pactl", "set-sink-mute", sink, "1"],
                "unmute": ["pactl", "set-sink-mute", sink, "0"],
            }.get(action)
        if which("amixer"):
            return {
                "set": ["amixer", "-q", "sset", "Master", f"{level}%"],
                "up": ["amixer", "-q", "sset", "Master", "10%+"],
                "down": ["amixer", "-q", "sset", "Master", "10%-"],
                "mute": ["amixer", "-q", "sset", "Master", "mute"],
                "unmute": ["amixer", "-q", "sset", "Master", "unmute"],
            }.get(action)
    if platform == "darwin":
        script = {
            "set": f"set volume output volume {level}",
            "up": "set volume output volume ((output volume of (get volume settings)) + 10)",
            "down": "set volume output volume ((output volume of (get volume settings)) - 10)",
            "mute": "set volume with output muted",
            "unmute": "set volume without output muted",
        }.get(action)
        return ["osascript", "-e", script] if script else None
    return None


class SystemSkill(Skill):
    name = "system"
    description = "Opens an app or website, or changes the computer's volume."
    slot_hints = '{"app": "<app name>"} or {"action": "set|up|down|mute|unmute", "level": "0-100"}'
    priority = 30
    patterns = (
        r"^(?:please\s+)?(?:open|launch|start)\s+(?:the\s+)?(?P<app>[\w .-]+?)(?:\s+please)?$",
        r"^(?P<app>[\w .-]+?)\s+(?:open\s+pannu|open\s+pannunga|thira)$",
        r"\bvolume\s+(?:to\s+)?(?P<level>\d{1,3})\s*(?:%|percent)?$",
        r"\bvolume\s+(?P<action>up|down)\b",
        r"^(?:please\s+)?(?P<action>unmute|mute)\b",
    )

    def __init__(self, cfg: SystemSkillCfg) -> None:
        super().__init__()
        self._apps = {normalize(k): v for k, v in cfg.apps.items() if v}
        self._children: set[asyncio.Task] = set()

    def _resolve_app(self, spoken: str) -> str | None:
        key = normalize(spoken)
        key = key[4:] if key.startswith("the ") else key
        if key in self._apps:
            return key
        close = difflib.get_close_matches(key, list(self._apps), n=1, cutoff=0.8)
        return close[0] if close else None

    def match(self, text: str) -> dict[str, str] | None:
        slots = super().match(text)
        if slots is None:
            return None
        # "open the garage door" must fall through to other skills, not hit us.
        if "app" in slots and self._resolve_app(slots["app"]) is None:
            return None
        if "level" in slots:
            slots["action"] = "set"
        return slots

    async def handle(self, intent: Intent, ctx: ConversationContext) -> SkillResult:
        slots = intent.slots
        if "app" in slots:
            return await self._open_app(slots["app"])
        action = slots.get("action") or ("set" if "level" in slots else "")
        if action:
            return await self._volume(action, slots.get("level"))
        return SkillResult("What would you like me to do on the computer?", ok=False)

    async def _open_app(self, spoken: str) -> SkillResult:
        alias = self._resolve_app(spoken)
        if alias is None:
            return SkillResult(f"I don't know an app called {spoken}.", ok=False)
        argv = self._apps[alias]
        if shutil.which(argv[0]) is None:
            raise SkillError(f"{argv[0]} not found", user_message=f"{alias} isn't installed on this computer.")
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            raise SkillError(str(exc), user_message=f"I couldn't open {alias}.") from exc
        task = asyncio.create_task(proc.wait())   # reap the child; don't block on it
        self._children.add(task)
        task.add_done_callback(self._children.discard)
        return SkillResult(f"Opening {alias}.")

    async def _volume(self, action: str, level: str | None) -> SkillResult:
        lvl = None
        if action == "set":
            try:
                lvl = max(0, min(100, int(level or "")))
            except ValueError:
                return SkillResult("What volume level, from zero to one hundred?", ok=False)
        cmd = volume_command(action, lvl)
        if cmd is None:
            return SkillResult("Volume control isn't supported on this system yet.", ok=False)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
            )
            _, err = await asyncio.wait_for(proc.communicate(), timeout=5)
        except (OSError, asyncio.TimeoutError) as exc:
            raise SkillError(str(exc), user_message="I couldn't change the volume.") from exc
        if proc.returncode != 0:
            raise SkillError(err.decode(errors="replace"), user_message="I couldn't change the volume.")
        return SkillResult(
            {
                "set": f"Volume set to {lvl} percent.",
                "up": "Volume up.",
                "down": "Volume down.",
                "mute": "Muted.",
                "unmute": "Unmuted.",
            }.get(action, "Done.")
        )
