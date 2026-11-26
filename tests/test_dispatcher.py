from __future__ import annotations

import asyncio

from assistant.core.dispatcher import interpret_confirmation
from assistant.core.skill import Skill
from assistant.core.types import Intent, PendingAction, SkillResult
from assistant.errors import LLMError, SkillError
from assistant.skills.routines import Routine
from assistant.skills.time_skill import TimeSkill

from .conftest import FakeLLM


class Boom(Skill):
    name = "boom"
    patterns = (r"^explode$",)

    async def handle(self, intent, ctx):
        raise RuntimeError("kaboom")


class Friendly(Skill):
    name = "friendly"
    patterns = (r"^fail nicely$",)

    async def handle(self, intent, ctx):
        raise SkillError("x", user_message="That device is unplugged.")


class Slow(Skill):
    name = "slow"
    patterns = (r"^be slow$",)

    async def handle(self, intent, ctx):
        await asyncio.sleep(5)
        return SkillResult("done")


class Risky(Skill):
    name = "risky"
    patterns = (r"^do the risky thing$",)

    def __init__(self):
        super().__init__()
        self.ran = 0

    async def handle(self, intent, ctx):
        async def go():
            self.ran += 1
            return SkillResult("risky thing done")

        return SkillResult("Really?", pending=PendingAction(go, "Okay, not doing it."))


async def test_rule_match_skips_the_llm(make_assistant):
    llm = FakeLLM("should not be used")
    d, _ = make_assistant(llm, TimeSkill("Asia/Kolkata"))
    r = await d.handle("what time is it")
    assert r.ok and r.speech.startswith("It's")
    assert llm.calls == []


async def test_plain_question_goes_straight_to_knowledge(make_assistant):
    llm = FakeLLM("Paris is the capital of France.")
    d, _ = make_assistant(llm)
    r = await d.handle("what is the capital of france")
    assert r.speech == "Paris is the capital of France."
    assert len(llm.calls) == 1          # no separate routing call


async def test_llm_router_picks_a_skill_for_unmatched_commands(make_assistant):
    llm = FakeLLM('{"skill": "time", "slots": {"kind": "date"}, "confidence": 0.9}')
    d, _ = make_assistant(llm, TimeSkill("Asia/Kolkata"))
    r = await d.handle("remind me which day we are on")
    assert r.speech.startswith("Today is")


async def test_router_garbage_falls_back_to_knowledge(make_assistant):
    llm = FakeLLM("lol not json", "Sure, here you go.")
    d, _ = make_assistant(llm, TimeSkill())
    r = await d.handle("blah blah")
    assert r.speech == "Sure, here you go."


async def test_llm_down_gives_spoken_offline_message_not_a_crash(make_assistant):
    d, _ = make_assistant(FakeLLM(LLMError("boom"), LLMError("boom")))
    r = await d.handle("tell me a joke")
    assert not r.ok and "brain" in r.speech


async def test_skill_crash_is_contained(make_assistant):
    d, _ = make_assistant(None, Boom())
    r = await d.handle("explode")
    assert not r.ok and "went wrong" in r.speech
    assert (await d.handle("tell me a joke")).speech  # dispatcher still alive


async def test_skill_error_user_message_is_spoken(make_assistant):
    d, _ = make_assistant(None, Friendly())
    assert (await d.handle("fail nicely")).speech == "That device is unplugged."


async def test_skill_timeout(make_assistant):
    d, _ = make_assistant(None, Slow())
    d._timeout = 0.05
    r = await d.handle("be slow")
    assert not r.ok and "too long" in r.speech


async def test_confirmation_yes_runs_the_action(make_assistant):
    risky = Risky()
    d, _ = make_assistant(None, risky)
    r = await d.handle("do the risky thing")
    assert r.speech == "Really?" and risky.ran == 0
    assert (await d.handle("yes please")).speech == "risky thing done"
    assert risky.ran == 1


async def test_confirmation_no_cancels(make_assistant):
    risky = Risky()
    d, _ = make_assistant(None, risky)
    await d.handle("do the risky thing")
    assert (await d.handle("no")).speech == "Okay, not doing it."
    assert risky.ran == 0 and d.context.pending is None


async def test_confirmation_expires_and_other_speech_cancels(make_assistant):
    risky = Risky()
    d, _ = make_assistant(None, risky)
    clock = {"t": 0.0}
    d._clock = lambda: clock["t"]
    await d.handle("do the risky thing")
    clock["t"] = 999
    await d.handle("what time is it")       # also: a new command drops the pending action
    assert risky.ran == 0
    await d.handle("do the risky thing")
    await d.handle("explode")                # unrelated command → pending cleared
    await d.handle("yes")
    assert risky.ran == 0


async def test_routine_runs_steps_and_skips_missing_skills(make_assistant):
    routine = Routine(
        "morning",
        ("good morning",),
        ({"say": "Hello!"}, {"skill": "time"}, {"skill": "ghost"}),
    )
    d, _ = make_assistant(None, TimeSkill("Asia/Kolkata"), routines=[routine])
    r = await d.handle("good morning")
    assert r.speech.startswith("Hello! It's")


def test_interpret_confirmation():
    assert interpret_confirmation("Yes, please.") is True
    assert interpret_confirmation("aama") is True
    assert interpret_confirmation("venam") is False
    assert interpret_confirmation("no don't") is False
    assert interpret_confirmation("what time is it") is None
    assert interpret_confirmation("") is None
