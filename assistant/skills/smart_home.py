"""Smart-home control (Home Assistant REST API, or a mock for dry runs).

Home Assistant is the pragmatic local-first choice: it already speaks to Zigbee,
Matter, Tuya, MQTT, etc. over your LAN, so this skill only needs one HTTP call.
"""
from __future__ import annotations

import difflib
import logging
import os
import re
from abc import ABC, abstractmethod

from ..config import SmartHomeCfg
from ..core.context import ConversationContext
from ..core.skill import Skill
from ..core.types import Intent, SkillResult
from ..errors import ConfigError, SkillError
from ..utils.retry import retry_async
from ..utils.text import normalize

log = logging.getLogger(__name__)

_STATES = {"on": "on", "true": "on", "1": "on", "off": "off", "false": "off", "0": "off"}


class SmartHomeBackend(ABC):
    @abstractmethod
    async def call(self, entity_id: str, action: str, **params: int) -> None:
        """``action`` is 'on', 'off' or 'brightness' (params: level=0-100)."""

    async def aclose(self) -> None:
        """Release network resources (optional)."""


class MockBackend(SmartHomeBackend):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, int]]] = []

    async def call(self, entity_id: str, action: str, **params: int) -> None:
        log.info("[mock smart home] %s %s %s", entity_id, action, params)
        self.calls.append((entity_id, action, params))


class HomeAssistantBackend(SmartHomeBackend):
    def __init__(self, base_url: str, token: str, timeout_s: float = 5.0) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ConfigError("httpx is required for the Home Assistant backend") from exc
        self._httpx = httpx
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_s,
        )

    async def call(self, entity_id: str, action: str, **params: int) -> None:
        if action == "brightness":
            if not entity_id.startswith("light."):
                raise SkillError("brightness on non-light", user_message="Only lights support brightness.")
            path = "/api/services/light/turn_on"
            payload = {"entity_id": entity_id, "brightness_pct": params["level"]}
        else:  # the generic `homeassistant` domain works for lights, switches, fans, …
            path = f"/api/services/homeassistant/turn_{action}"
            payload = {"entity_id": entity_id}
        try:
            resp = await self._client.post(path, json=payload)
            resp.raise_for_status()
        except self._httpx.HTTPError as exc:
            raise SkillError(
                f"home assistant call failed: {exc}",
                user_message="I couldn't reach your smart home hub.",
            ) from exc

    async def aclose(self) -> None:
        await self._client.aclose()


def build_backend(cfg: SmartHomeCfg) -> SmartHomeBackend:
    if cfg.backend == "mock":
        return MockBackend()
    if cfg.backend == "homeassistant":
        token = os.environ.get(cfg.token_env)
        if not token:
            raise ConfigError(f"{cfg.token_env} is not set")
        return HomeAssistantBackend(cfg.base_url, token, cfg.timeout_s)
    raise ConfigError(f"unknown smart_home.backend: {cfg.backend!r}")


def _norm_device(name: str) -> str:
    n = re.sub(r"\b(?:please|now|the|my)\b", " ", normalize(name))
    return re.sub(r"\s+", " ", n).strip()


class SmartHomeSkill(Skill):
    name = "smart_home"
    description = "Turns smart-home devices (lights, fans, switches) on or off, or sets light brightness."
    slot_hints = '{"device": "<spoken device name>", "state": "on|off"} or {"device": "...", "level": "0-100"}'
    priority = 40
    patterns = (
        r"\b(?:turn|switch)\s+(?P<state>on|off)\s+(?:the\s+)?(?P<device>.+)$",
        r"\b(?:turn|switch)\s+(?:the\s+)?(?P<device>.+?)\s+(?P<state>on|off)$",
        r"^(?:the\s+)?(?P<device>.+?)\s+(?P<state>on|off)\s+(?:pannu|pannunga|pannidu|pannu\s+da|podu)$",
        r"\bset\s+(?:the\s+)?(?P<device>.+?)\s+(?:brightness\s+)?to\s+(?P<level>\d{1,3})\s*(?:%|percent)?$",
    )

    def __init__(self, cfg: SmartHomeCfg, backend: SmartHomeBackend) -> None:
        super().__init__()
        self._backend = backend
        self._by_name = {_norm_device(k): v for k, v in cfg.devices.items()}

    def _resolve(self, spoken: str) -> str | None:
        key = _norm_device(spoken)
        if key in self._by_name:
            return self._by_name[key]
        close = difflib.get_close_matches(key, list(self._by_name), n=1, cutoff=0.75)
        return self._by_name[close[0]] if close else None

    def match(self, text: str) -> dict[str, str] | None:
        slots = super().match(text)
        # Unknown device → not ours ("turn on the music" may belong to another skill).
        if slots is None or (slots.get("device") and self._resolve(slots["device"]) is None):
            return None
        return slots

    async def handle(self, intent: Intent, ctx: ConversationContext) -> SkillResult:
        slots = intent.slots
        device = slots.get("device")
        if not device:
            return SkillResult("Which device?", ok=False)
        entity = self._resolve(device)
        if entity is None:
            return SkillResult(f"I don't know a device called {device}.", ok=False)

        if "level" in slots:
            try:
                level = max(0, min(100, int(slots["level"])))
            except ValueError:
                return SkillResult("What brightness, from zero to one hundred?", ok=False)
            await self._send(entity, "brightness", level=level)
            return SkillResult(f"Okay, {device} at {level} percent.")

        state = _STATES.get(slots.get("state", "").strip().lower())
        if state is None:
            return SkillResult(f"Should I turn {device} on or off?", ok=False)
        await self._send(entity, state)
        return SkillResult(f"Okay, {device} {state}.")

    async def _send(self, entity: str, action: str, **params: int) -> None:
        await retry_async(
            lambda: self._backend.call(entity, action, **params),
            attempts=2,
            base_delay=0.3,
            retry_on=(SkillError,),
        )

    async def aclose(self) -> None:
        await self._backend.aclose()
