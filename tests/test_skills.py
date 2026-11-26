from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from assistant.config import SmartHomeCfg, SystemSkillCfg, WhatsAppCfg
from assistant.core.context import ConversationContext
from assistant.core.types import Intent
from assistant.skills.smart_home import MockBackend, SmartHomeSkill
from assistant.skills.system import SystemSkill, volume_command
from assistant.skills.time_skill import TimeSkill
from assistant.skills.whatsapp import WhatsAppSkill

CTX = ConversationContext()


def _intent(skill, slots=None, text=""):
    return Intent(skill.name, slots or {}, 1.0, "rules", text)


# -- smart home ----------------------------------------------------------
def _home():
    cfg = SmartHomeCfg(enabled=True, devices={"living room light": "light.living", "fan": "switch.fan"})
    backend = MockBackend()
    return SmartHomeSkill(cfg, backend), backend


async def test_smart_home_turn_on_variants():
    skill, backend = _home()
    for phrase in ("turn on the living room light", "turn the fan off", "switch off the fan please"):
        slots = skill.match(phrase)
        assert slots is not None, phrase
        r = await skill.handle(_intent(skill, slots), CTX)
        assert r.ok, phrase
    assert [(e, a) for e, a, _ in backend.calls] == [
        ("light.living", "on"), ("switch.fan", "off"), ("switch.fan", "off")
    ]


async def test_smart_home_brightness():
    skill, backend = _home()
    slots = skill.match("set the living room light to 40 percent")
    await skill.handle(_intent(skill, slots), CTX)
    assert backend.calls[0] == ("light.living", "brightness", {"level": 40})


async def test_smart_home_unknown_device_is_not_claimed():
    skill, _ = _home()
    assert skill.match("turn on the music") is None


# -- system --------------------------------------------------------------
def test_system_open_only_allowlisted_apps():
    skill = SystemSkill(SystemSkillCfg(apps={"browser": ["xdg-open", "x"]}))
    assert skill.match("open the browser") == {"app": "browser"}
    assert skill.match("open rm -rf") is None
    assert skill.match("open the garage door") is None


def test_volume_commands():
    which = lambda name: "/usr/bin/" + name  # noqa: E731
    assert volume_command("set", 30, "linux", which)[:2] == ["pactl", "set-sink-volume"]
    assert volume_command("up", None, "darwin", which)[0] == "osascript"
    assert volume_command("up", None, "win32", which) is None


def test_system_volume_level_slot_implies_set():
    skill = SystemSkill(SystemSkillCfg())
    assert skill.match("set volume to 40") == {"level": "40", "action": "set"}


# -- whatsapp ------------------------------------------------------------
class FakeWA:
    def __init__(self):
        self.sent = []

    async def send(self, phone, text):
        self.sent.append((phone, text))

    async def aclose(self):
        pass


async def test_whatsapp_requires_confirmation_before_sending():
    wa = FakeWA()
    skill = WhatsAppSkill(WhatsAppCfg(enabled=True, contacts={"amma": "+91 98765 43210"}), wa)
    slots = skill.match("send a message to amma saying I'll be late")
    assert slots == {"contact": "amma", "message": "I'll be late"}
    r = await skill.handle(_intent(skill, slots), CTX)
    assert r.pending is not None and wa.sent == []      # nothing sent yet
    done = await r.pending.execute()
    assert wa.sent == [("+919876543210", "I'll be late")] and done.ok


async def test_whatsapp_unknown_contact_and_bad_numbers():
    skill = WhatsAppSkill(WhatsAppCfg(enabled=True, contacts={"amma": "+919876543210", "bad": "12"}), FakeWA())
    r = await skill.handle(_intent(skill, {"contact": "stranger", "message": "hi"}), CTX)
    assert not r.ok and r.pending is None
    assert "bad" not in skill._contacts


# -- time ----------------------------------------------------------------
async def test_time_and_date():
    fixed = datetime(2026, 11, 22, 9, 5, tzinfo=ZoneInfo("Asia/Kolkata"))
    skill = TimeSkill(now=lambda: fixed)
    assert (await skill.handle(_intent(skill, {}, "what time is it"), CTX)).speech == "It's 9:05 AM."
    assert (await skill.handle(_intent(skill, {}, "what's the date"), CTX)).speech == "Today is Sunday, November 22nd."
