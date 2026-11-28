"""Tiny local web server for the browser UI (``python main.py --web``).

Standard library only. It binds to 127.0.0.1 and rejects foreign Host/Origin headers,
because skills can open apps and send messages — a random website must never be able
to talk to it (CSRF / DNS-rebinding).

Endpoints
    GET  /              the chat page (web/index.html)
    GET  /api/status    {name, llm, skills}
    GET  /api/events    announcements since last poll (timers, reminders)
    POST /api/chat      {"text": "..."} -> {"reply", "ok", "language"}
    POST /api/reset     clear conversation memory
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .bootstrap import Assistant

log = logging.getLogger(__name__)

MAX_BODY = 8 * 1024
MAX_TEXT = 500
CHAT_TIMEOUT_S = 90
INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"

_CSP = (
    "default-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; script-src 'self' 'unsafe-inline'; "
    "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
)


class WebServer:
    def __init__(
        self,
        assistant: "Assistant",
        loop: asyncio.AbstractEventLoop,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        index: Path = INDEX,
    ) -> None:
        self._assistant = assistant
        self._loop = loop
        self._host = host
        self._port = port
        self._index = index
        self._lock = asyncio.Lock()
        self._events: deque[dict] = deque(maxlen=50)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port = port

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> int:
        self._assistant.announcer.bind(self._on_announce)
        self._httpd = ThreadingHTTPServer((self._host, self._port), self._make_handler())
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="webui", daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    # -- assistant bridge --------------------------------------------------
    def _on_announce(self, text: str, language: str | None = None) -> None:
        self._events.append({"text": text, "language": language})

    async def _chat(self, text: str) -> dict:
        async with self._lock:
            result = await self._assistant.dispatcher.handle(text)
        return {"reply": result.speech, "ok": result.ok, "language": result.language}

    def _status(self) -> dict:
        a = self._assistant
        return {
            "name": a.cfg.assistant.name,
            "llm": a.llm is not None,
            "skills": [s.name for s in a.registry.all() if not s.is_fallback],
        }

    def _drain_events(self) -> list[dict]:
        out = []
        while self._events:
            out.append(self._events.popleft())
        return out

    # -- HTTP ---------------------------------------------------------------
    def _allowed_hosts(self) -> set[str]:
        return {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):  # noqa: A003
                log.debug("web: " + fmt, *args)

            # helpers
            def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code: int, obj: dict | list) -> None:
                self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

            def _guard(self) -> bool:
                host = self.headers.get("Host", "")
                origin = self.headers.get("Origin")
                if host not in server._allowed_hosts():
                    self._json(403, {"error": "forbidden host"})
                    return False
                if origin is not None and origin not in {f"http://{h}" for h in server._allowed_hosts()}:
                    self._json(403, {"error": "forbidden origin"})
                    return False
                return True

            # routes
            def do_GET(self) -> None:  # noqa: N802
                if not self._guard():
                    return
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    try:
                        body = server._index.read_bytes()
                    except OSError:
                        self._json(500, {"error": "web/index.html is missing"})
                        return
                    self._send(200, body, "text/html; charset=utf-8", {"Content-Security-Policy": _CSP})
                elif path == "/api/status":
                    self._json(200, server._status())
                elif path == "/api/events":
                    self._json(200, {"events": server._drain_events()})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                if not self._guard():
                    return
                path = self.path.split("?", 1)[0]
                if not self.headers.get("Content-Type", "").startswith("application/json"):
                    self._json(415, {"error": "send application/json"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = -1
                if not 0 <= length <= MAX_BODY:
                    self._json(413, {"error": "body too large"})
                    return
                try:
                    data = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(data, dict):
                        raise ValueError
                except ValueError:
                    self._json(400, {"error": "invalid JSON"})
                    return

                if path == "/api/reset":
                    server._assistant.dispatcher.context.clear()
                    self._json(200, {"ok": True})
                elif path == "/api/chat":
                    text = data.get("text")
                    if not isinstance(text, str) or not text.strip():
                        self._json(400, {"error": "text is required"})
                        return
                    fut = asyncio.run_coroutine_threadsafe(server._chat(text.strip()[:MAX_TEXT]), server._loop)
                    try:
                        self._json(200, fut.result(timeout=CHAT_TIMEOUT_S))
                    except Exception:  # noqa: BLE001
                        fut.cancel()
                        log.exception("chat request failed")
                        self._json(500, {"error": "assistant failed", "reply": "Something went wrong on my side.", "ok": False})
                else:
                    self._json(404, {"error": "not found"})

        return Handler
