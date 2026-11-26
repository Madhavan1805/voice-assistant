"""Canned spoken messages (English + Tamil)."""
from __future__ import annotations

_MESSAGES: dict[str, dict[str, str]] = {
    "didnt_catch": {
        "en": "Sorry, I didn't catch that. Could you say it again?",
        "ta": "மன்னிக்கவும், புரியவில்லை. மீண்டும் சொல்லுங்கள்.",
    },
    "listening": {"en": "Yes?", "ta": "சொல்லுங்கள்?"},
    "offline": {
        "en": "I can't reach my brain right now. Please try again in a moment.",
        "ta": "இப்போது என்னால் பதில் சொல்ல முடியவில்லை. கொஞ்சம் கழித்து முயற்சி செய்யுங்கள்.",
    },
    "stt_trouble": {
        "en": "I'm having trouble hearing you right now.",
        "ta": "எனக்கு உங்கள் குரல் சரியாகக் கேட்கவில்லை.",
    },
    "mic_trouble": {
        "en": "I can't access the microphone.",
        "ta": "மைக்ரோஃபோனை அணுக முடியவில்லை.",
    },
    "timeout": {
        "en": "That's taking too long. Please try again.",
        "ta": "இது அதிக நேரம் எடுக்கிறது. மீண்டும் முயற்சி செய்யுங்கள்.",
    },
    "skill_error": {
        "en": "Sorry, something went wrong with that.",
        "ta": "மன்னிக்கவும், ஏதோ தவறு நடந்தது.",
    },
    "cancelled": {"en": "Okay, cancelled.", "ta": "சரி, ரத்து செய்துவிட்டேன்."},
}


def msg(key: str, lang: str | None = "en") -> str:
    variants = _MESSAGES[key]
    return variants.get(lang or "en") or variants["en"]
