from __future__ import annotations

from assistant.core.wake import WakeGate


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _gate(clock=None, **kw) -> WakeGate:
    return WakeGate(["hey madhav", "madhav"], clock=clock or Clock(), **kw)


def test_strips_wake_phrase_and_keeps_command_case():
    r = _gate().check("Hey Madhav, what's the time?")
    assert (r.accepted, r.woke, r.command) == (True, True, "what's the time?")


def test_wake_only_has_empty_command():
    r = _gate().check("Hey Madhav")
    assert r.accepted and r.woke and r.command == ""


def test_tolerates_stt_misspelling():
    assert _gate().check("hey madhov turn on the light").woke


def test_ignores_speech_without_wake_phrase():
    assert not _gate().check("what a lovely day").accepted


def test_followup_window_opens_and_expires():
    clock = Clock()
    g = _gate(clock, followup_window_s=8)
    assert not g.check("and tomorrow?").accepted
    g.extend()
    clock.t += 5
    r = g.check("and tomorrow?")
    assert r.accepted and not r.woke and r.command == "and tomorrow?"
    clock.t += 10
    assert not g.check("and tomorrow?").accepted


def test_wake_not_required_accepts_everything():
    assert _gate(required=False).check("anything at all").accepted


def test_tamil_command_after_wake_phrase():
    r = _gate().check("Hey Madhav, இப்போ நேரம் என்ன?")
    assert r.woke and r.command == "இப்போ நேரம் என்ன?"
