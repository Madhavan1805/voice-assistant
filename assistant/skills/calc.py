"""Spoken maths ('what is 25 times 4', '20 percent of 300').

Evaluated with a tiny AST walker — never eval() — so spoken text can't run code.
"""
from __future__ import annotations

import ast
import operator
import re

from ..core.context import ConversationContext
from ..core.skill import Skill
from ..core.types import Intent, SkillResult

_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_WORD_OPS = [
    (r"\bto the power of\b", "**"), (r"\braised to\b", "**"), (r"\bmultiplied by\b", "*"),
    (r"\bdivided by\b", "/"), (r"\bplus\b", "+"), (r"\bminus\b", "-"), (r"\btimes\b", "*"),
    (r"\binto\b", "*"), (r"\bover\b", "/"), (r"\bmod(?:ulo)?\b", "%"),
    (r"(?<=\d)\s*[x×]\s*(?=\d)", "*"), (r"÷", "/"), (r"\^", "**"),
]
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(\d+(?:\.\d+)?)")


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(b) > 100 or abs(a) > 1e9):
            raise ValueError("power too large")
        result = _BIN[type(node.op)](a, b)
        if abs(result) > 1e18:
            raise ValueError("result too large")
        return result
    raise ValueError("unsupported expression")


def safe_eval(text: str) -> float | None:
    s = text.lower().strip().rstrip("?.! ")
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
    s = _PERCENT.sub(lambda m: f"({m.group(1)}/100*{m.group(2)})", s)
    for rx, rep in _WORD_OPS:
        s = re.sub(rx, f" {rep} ", s)
    s = s.strip()
    if not s or not re.fullmatch(r"[\d\s.+\-*/%()]+", s):
        return None
    try:
        return _eval(ast.parse(s, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError):
        return None


def format_number(n: float) -> str:
    return str(int(round(n))) if abs(n - round(n)) < 1e-9 else f"{n:.6g}"


class CalcSkill(Skill):
    name = "calc"
    description = "Does arithmetic: add, subtract, multiply, divide, percentages."
    slot_hints = '{"expr": "<the maths expression as spoken>"}'
    priority = 24
    patterns = (
        r"^(?:please\s+)?(?:what(?:'s| is)|calculate|compute|how much is|solve|tell me)\s+(?P<expr>.+)$",
        r"^(?P<expr>\d.*)$",
    )

    def match(self, text: str) -> dict[str, str] | None:
        slots = super().match(text)
        if slots is None or safe_eval(slots.get("expr", "")) is None:
            return None   # "what is the capital of France" is not ours
        return slots

    async def handle(self, intent: Intent, ctx: ConversationContext) -> SkillResult:
        value = safe_eval(intent.slots.get("expr", ""))
        if value is None:
            return SkillResult("Sorry, I couldn't work that out.", ok=False)
        return SkillResult(f"That's {format_number(value)}.")
