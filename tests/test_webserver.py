from __future__ import annotations

import asyncio
import http.client
import json
import threading

import pytest

from assistant.bootstrap import build_assistant
from assistant.config import Config
from assistant.webserver import WebServer


@pytest.fixture
def server(tmp_path):
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    cfg = Config()
    cfg.llm.ollama.enabled = False
    cfg.llm.anthropic.enabled = False
    cfg.skills.routines_file = str(tmp_path / "none.yaml")
    assistant = build_assistant(cfg)
    srv = WebServer(assistant, loop, port=0)
    port = srv.start()
    yield srv, port, assistant
    srv.stop()
    loop.call_soon_threadsafe(loop.stop)


def call(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = {"Host": f"127.0.0.1:{port}"}
    if body is not None:
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    conn.request(method, path, body=json.dumps(body) if isinstance(body, (dict, list)) else body, headers=h)
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, dict(r.getheaders()), data


def test_serves_the_page_with_security_headers(server):
    _, port, _ = server
    status, headers, body = call(port, "GET", "/")
    assert status == 200 and b"<title>Kandy</title>" in body
    assert "default-src 'self'" in headers["Content-Security-Policy"]


def test_status_reports_skills_and_missing_llm(server):
    _, port, _ = server
    status, _, body = call(port, "GET", "/api/status")
    info = json.loads(body)
    assert status == 200 and info["llm"] is False and "timer" in info["skills"] and "knowledge" not in info["skills"]


def test_chat_runs_the_dispatcher(server):
    _, port, _ = server
    status, _, body = call(port, "POST", "/api/chat", {"text": "what is 6 times 7"})
    assert status == 200 and json.loads(body)["reply"] == "That's 42."
    status, _, body = call(port, "POST", "/api/chat", {"text": "tell me about black holes"})
    reply = json.loads(body)
    assert status == 200 and reply["ok"] is False and "brain" in reply["reply"]


def test_rejects_foreign_origin_and_host(server):
    _, port, _ = server
    assert call(port, "POST", "/api/chat", {"text": "hi"}, {"Origin": "https://evil.example"})[0] == 403
    assert call(port, "GET", "/api/status", None, {"Host": "evil.example"})[0] == 403
    assert call(port, "POST", "/api/chat", {"text": "hi"}, {"Origin": f"http://localhost:{port}"})[0] == 200


def test_rejects_bad_requests(server):
    _, port, _ = server
    assert call(port, "POST", "/api/chat", {"text": "  "})[0] == 400
    assert call(port, "POST", "/api/chat", "not json", {"Content-Type": "application/json"})[0] == 400
    assert call(port, "POST", "/api/chat", "x=1", {"Content-Type": "text/plain"})[0] == 415
    assert call(port, "POST", "/api/chat", "x" * 20000, {"Content-Type": "application/json"})[0] == 413
    assert call(port, "GET", "/nope")[0] == 404


def test_announcements_are_delivered_once(server):
    _, port, assistant = server
    assistant.announcer.say("Time's up!")
    events = json.loads(call(port, "GET", "/api/events")[2])["events"]
    assert events == [{"text": "Time's up!", "language": None}]
    assert json.loads(call(port, "GET", "/api/events")[2])["events"] == []


def test_reset_clears_conversation_memory(server):
    _, port, assistant = server
    call(port, "POST", "/api/chat", {"text": "hello"})
    assert assistant.dispatcher.context.messages()
    assert call(port, "POST", "/api/reset", {})[0] == 200
    assert not assistant.dispatcher.context.messages()
