"""Weather via Open-Meteo (free, no API key)."""
from __future__ import annotations

import logging
import re

from ..config import WeatherCfg
from ..core.context import ConversationContext
from ..core.skill import Skill
from ..core.types import Intent, SkillResult
from ..errors import SkillError

log = logging.getLogger(__name__)

_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_CODES = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "foggy", 48: "foggy",
    51: "drizzling", 53: "drizzling", 55: "drizzling", 56: "drizzling", 57: "drizzling",
    61: "raining", 63: "raining", 65: "raining heavily", 66: "raining", 67: "raining",
    71: "snowing", 73: "snowing", 75: "snowing heavily", 77: "snowing",
    80: "showery", 81: "showery", 82: "showery with heavy rain", 85: "snowing", 86: "snowing",
    95: "thunderstorming", 96: "thunderstorming", 99: "thunderstorming",
}
_TRAILING = re.compile(r"\s+(?:today|now|right now|tonight|tomorrow|please|like)\s*$", re.I)


class WeatherSkill(Skill):
    name = "weather"
    description = "Current weather and today's forecast for a city."
    slot_hints = '{"city": "<city name, optional>"}'
    priority = 35
    patterns = (
        r"\b(?:weather|temperature|forecast)\b(?:.*?\b(?:in|at|for)\s+(?P<city>[A-Za-z][A-Za-z .'-]*))?",
        r"\b(?:will it|is it going to|is it gonna)\s+rain\b(?:.*?\b(?:in|at)\s+(?P<city>[A-Za-z][A-Za-z .'-]*))?",
    )

    def __init__(self, cfg: WeatherCfg) -> None:
        super().__init__()
        self._cfg = cfg

    async def _get_json(self, url: str, params: dict) -> dict:
        import httpx

        async with httpx.AsyncClient(timeout=self._cfg.timeout_s) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

    async def handle(self, intent: Intent, ctx: ConversationContext) -> SkillResult:
        city = _TRAILING.sub("", intent.slots.get("city", "")).strip(" .?!") or self._cfg.default_city
        try:
            geo = await self._get_json(_GEO_URL, {"name": city, "count": 1})
            places = geo.get("results") or []
            if not places:
                return SkillResult(f"I couldn't find a place called {city}.", ok=False)
            place = places[0]
            data = await self._get_json(_FORECAST_URL, {
                "latitude": place["latitude"], "longitude": place["longitude"],
                "current": "temperature_2m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto", "forecast_days": 1,
            })
            cur, day = data["current"], data["daily"]
            temp = round(cur["temperature_2m"])
            sky = _CODES.get(int(cur["weather_code"]), "unsettled")
            hi, lo = round(day["temperature_2m_max"][0]), round(day["temperature_2m_min"][0])
            rain = day.get("precipitation_probability_max", [None])[0]
        except SkillError:
            raise
        except Exception as exc:  # noqa: BLE001 - network, JSON shape, missing httpx…
            raise SkillError(f"weather lookup failed: {exc}", user_message="I couldn't get the weather right now.") from exc

        name = place.get("name", city)
        text = f"Right now in {name} it's {temp} degrees and {sky}. Today's high is {hi}, low is {lo}"
        if rain is not None:
            text += f", with a {round(rain)} percent chance of rain"
        return SkillResult(text + ".")
