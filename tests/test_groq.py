from __future__ import annotations

import json

import pytest

httpx = pytest.importorskip("httpx")   # runs wherever requirements.txt is installed

from assistant.config import Config, GroqCfg
from assistant.errors import LLMError
from assistant.nlp.llm import GroqClient


def _client(handler, monkeypatch, key="gsk_test"):
    monkeypatch.setenv("GROQ_API_KEY", key)
    return GroqClient(GroqCfg(), transport=httpx.MockTransport(handler))


async def test_groq_sends_openai_style_request_and_parses_reply(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "  Vanakkam machan!  "}}]})

    out = await _client(handler, monkeypatch).complete(
        [{"role": "user", "content": "hi"}], system="Be friendly", max_tokens=50
    )
    assert out == "Vanakkam machan!"
    assert seen["auth"] == "Bearer gsk_test"
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["body"]["model"] == "llama-3.3-70b-versatile"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "Be friendly"}
    assert seen["body"]["max_tokens"] == 50


@pytest.mark.parametrize("status,hint", [(401, "rejected"), (429, "rate limit"), (500, "HTTP 500")])
async def test_groq_http_errors_become_llm_errors_without_leaking_the_key(monkeypatch, status, hint):
    client = _client(lambda r: httpx.Response(status, json={"error": "x"}), monkeypatch, key="gsk_SECRET")
    with pytest.raises(LLMError) as err:
        await client.complete([{"role": "user", "content": "hi"}])
    assert hint in str(err.value) and "gsk_SECRET" not in str(err.value)


async def test_groq_empty_or_malformed_reply_is_an_error(monkeypatch):
    for body in ({"choices": [{"message": {"content": ""}}]}, {"choices": []}, {"oops": 1}):
        client = _client(lambda r, b=body: httpx.Response(200, json=b), monkeypatch)
        with pytest.raises(LLMError):
            await client.complete([{"role": "user", "content": "hi"}])


def test_groq_needs_a_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        GroqClient(GroqCfg())


def test_shipped_config_puts_groq_first():
    assert Config.load("config.yaml").llm.order[0] == "groq"
