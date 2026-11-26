"""Text helpers shared across modules.

NOTE: Tamil vowel signs are combining marks (not ``\\w``), so we only ever strip
ASCII punctuation — never use ``\\W`` on user text.
"""
from __future__ import annotations

import re
import string

_PUNCT = str.maketrans({c: " " for c in string.punctuation if c != "'"})
_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+")
_CODE_BLOCK = re.compile(r"```.*?```", re.S)
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️]")
_BULLET = re.compile(r"^[ \t]*(?:[-*•]|\d+[.)])[ \t]+", re.M)
_MD_CHARS = re.compile(r"[*_`#>~|]+")
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
_QUESTION_START = re.compile(
    r"^(?:who|what|when|where|why|how|which|is|are|am|was|were|do|does|did|can|could|"
    r"will|would|should|tell me|explain|define|describe|give me|"
    r"enna|yaar|yaaru|yen|eppadi|epdi|enga|epo|eppo)\b",
    re.I,
)


def normalize(text: str) -> str:
    """Lowercase, drop ASCII punctuation (keeps apostrophes), collapse spaces."""
    return _WS.sub(" ", text.lower().translate(_PUNCT)).strip()


def light_clean(text: str) -> str:
    """Collapse whitespace and trim trailing punctuation but PRESERVE case."""
    return _WS.sub(" ", text).strip().rstrip(".!?।, ")


def detect_language(text: str) -> str:
    """'ta' if the text is mostly Tamil script, otherwise 'en'."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return "en"
    tamil = sum("஀" <= c <= "௿" for c in chars)
    return "ta" if tamil / len(chars) > 0.3 else "en"


def is_question(text: str) -> bool:
    t = text.strip()
    return t.endswith("?") or bool(_QUESTION_START.match(t))


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def clean_for_speech(text: str) -> str:
    """Strip markdown, URLs, code and emoji so the TTS doesn't read symbols."""
    text = _CODE_BLOCK.sub(" ", text)
    text = _URL.sub("link", text)
    text = _EMOJI.sub("", text)
    text = _BULLET.sub("", text)
    text = _MD_CHARS.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def split_sentences(text: str, max_chars: int = 220) -> list[str]:
    """Split into speakable chunks so the first sentence can start playing early."""
    out: list[str] = []
    for part in _SENT_SPLIT.split(text):
        part = part.strip()
        while len(part) > max_chars:
            cut = part.rfind(" ", 0, max_chars)
            if cut < 20:
                cut = max_chars
            out.append(part[:cut].strip(" ,"))
            part = part[cut:].strip(" ,")
        if part:
            out.append(part)
    return out
