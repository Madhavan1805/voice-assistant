from __future__ import annotations

import asyncio

import pytest

from assistant.config import WeatherCfg
from assistant.core.announce import Announcer
from assistant.core.context import ConversationContext
from assistant.core.types import Intent
from assistant.errors import SkillError
from assistant.skills.calc import CalcSkill, safe_eval
from assistant.skills.smalltalk import SmallTalkSkill
from assistant.skills.timer import TimerSkill, format_duration, parse_duration
from assistant.skills.weather import WeatherSkill
from assistant.skills.web import WebSkill

CTX = ConversationContext()


def _intent(skill, slots, text=""):
    return Intent(skill.name, slots, 1.0, "rules", text)


# -- calc ----------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("25 times 4", 100), ("12 plus 7", 19), ("100 divided by 8", 12.5), ("20 percent of 300", 60),
    ("2 to the power of 10", 1024), ("1,000 minus 1", 999), ("(3 + 4) * 2", 14), ("7 x 6", 42),
])
def test_safe_eval_values(text, expected):
    assert safe_eval(text) == expected


@pytest.mark.parametrize("text", [
    "__import__('os').system('ls')", "the capital of france", "5 divided by 0", "9 ** 9 ** 9", "", "open(1)",
])
def test_safe_eval_rejects_non_maths(text):
    assert safe_eval(text) is None


async def test_calc_skill_claims_only_real_maths():
    skill = CalcSkill()
    assert skill.match("what is the capital of france") is None
    slots = skill.match("what's 25 times 4?")
    assert slots is not None
    assert (await skill.handle(_intent(skill, slots), CTX)).speech == "That's 100."


# -- durations / timers ----------------------------------------------------
@pytest.mark.parametrize("text,seconds", [
    ("5 minutes", 300), ("an hour", 3600), ("half an hour", 1800), ("1 hour 30 minutes", 5400),
    ("an hour and a half", 5400), ("ten minutes", 600), ("90 seconds", 90), ("2.5 hours", 9000),
])
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


def test_parse_duration_rejects_nonsense_and_huge():
    assert parse_duration("soon") is None
    assert parse_duration("30 hours") is None


def test_format_duration():
    assert format_duration(300) == "5 minutes"
    assert format_duration(5400) == "1 hour 30 minutes"
    assert format_duration(1) == "1 second"


async def test_timer_fires_announces_and_can_be_cancelled():
    heard: list[str] = []
    ann = Announcer()
    ann.bind(lambda text, lang=None: heard.append(text))
    skill = TimerSkill(ann)

    slots = skill.match("set a timer for 1 second")
    assert slots == {"when": "1 second"}
    assert (await skill.handle(_intent(skill, slots), CTX)).speech == "Okay, timer set for 1 second."
    await asyncio.sleep(1.3)
    assert heard == ["Time's up! Your 1 second timer is done."]

    slots = skill.match("remind me in 10 minutes to call amma")
    assert slots == {"when": "10 minutes", "what": "call amma"}
    await skill.handle(_intent(skill, slots), CTX)
    assert "call amma" in (await skill.handle(_intent(skill, skill.match("list my timers")), CTX)).speech
    assert (await skill.handle(_intent(skill, skill.match("cancel all timers")), CTX)).speech == "Okay, cancelled."
    await skill.aclose()
    assert heard == ["Time's up! Your 1 second timer is done."]   # cancelled reminder never fired


def test_timer_does_not_claim_unparseable_durations():
    assert TimerSkill(Announcer()).match("set a timer for banana") is None


def test_announcer_without_sink_does_not_raise():
    Announcer().say("nobody is listening")


# -- weather ---------------------------------------------------------------
class FakeWeather(WeatherSkill):
    def __init__(self, fail=False, empty=False):
        super().__init__(WeatherCfg(default_city="Chennai"))
        self.fail, self.empty, self.calls = fail, empty, []

    async def _get_json(self, url, params):
        self.calls.append(params)
        if self.fail:
            raise RuntimeError("network down")
        if "name" in params:
            return {"results": []} if self.empty else {"results": [{"name": params["name"], "latitude": 13.0, "longitude": 80.2}]}
        return {
            "current": {"temperature_2m": 29.4, "weather_code": 2},
            "daily": {"temperature_2m_max": [31.2], "temperature_2m_min": [25.1], "precipitation_probability_max": [40]},
        }


async def test_weather_reports_city_and_defaults_to_config():
    skill = FakeWeather()
    slots = skill.match("what's the weather in Madurai today")
    assert slots["city"].strip().lower().startswith("madurai")
    out = (await skill.handle(_intent(skill, {"city": "Madurai today"}), CTX)).speech
    assert out == "Right now in Madurai it's 29 degrees and partly cloudy. Today's high is 31, low is 25, with a 40 percent chance of rain."
    out = (await skill.handle(_intent(skill, {}), CTX)).speech
    assert "Chennai" in out and skill.match("what is the weather") == {}


async def test_weather_failures_are_spoken_not_raised():
    with pytest.raises(SkillError) as err:
        await FakeWeather(fail=True).handle(_intent(WeatherSkill(WeatherCfg()), {}), CTX)
    assert "couldn't get the weather" in err.value.user_message
    r = await FakeWeather(empty=True).handle(_intent(WeatherSkill(WeatherCfg()), {"city": "Nowhereville"}), CTX)
    assert not r.ok and "Nowhereville" in r.speech


# -- web ---------------------------------------------------------------------
async def test_web_skill_opens_encoded_urls_and_reports_failure():
    opened: list[str] = []
    skill = WebSkill(opener=lambda url: opened.append(url) or True)
    slots = skill.match("play Kadhal Kasukuthu on youtube")
    assert slots == {"play": "Kadhal Kasukuthu"}
    await skill.handle(_intent(skill, slots), CTX)
    await skill.handle(_intent(skill, skill.match("search for best biryani & more")), CTX)
    assert opened[0] == "https://www.youtube.com/results?search_query=Kadhal+Kasukuthu"
    assert opened[1] == "https://www.google.com/search?q=best+biryani+%26+more"
    with pytest.raises(SkillError):
        await WebSkill(opener=lambda url: False).handle(_intent(skill, {"play": "x"}), CTX)


# -- small talk ---------------------------------------------------------------
@pytest.mark.parametrize("text,key", [
    ("hello", "greet"), ("hey machan", "greet"), ("how are you", "how"), ("who are you", "who"),
    ("thank you so much", "thanks"), ("what else can you do", "capabilities"), ("tell me a joke", "joke"),
    ("flip a coin", "coin"),
])
async def test_smalltalk_routes(text, key):
    skill = SmallTalkSkill("Kandy")
    slots = skill.match(text)
    assert slots and key in slots
    assert (await skill.handle(_intent(skill, slots), CTX)).speech


def test_smalltalk_ignores_real_questions():
    assert SmallTalkSkill().match("what is the capital of france") is None


# -- end to end through the dispatcher ------------------------------------------
async def test_dispatcher_routes_new_skills_without_any_llm(make_assistant):
    d, _ = make_assistant(None, TimerSkill(Announcer()), CalcSkill(), SmallTalkSkill(), WebSkill(opener=lambda u: True))
    assert (await d.handle("what is 6 times 7")).speech == "That's 42."
    assert (await d.handle("set a timer for 5 minutes")).speech == "Okay, timer set for 5 minutes."
    assert "Kandy" in (await d.handle("who are you")).speech
    assert (await d.handle("cancel the timer")).speech == "Okay, cancelled."
