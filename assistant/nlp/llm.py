"""LLM backends (local Ollama first, Anthropic as cloud fallback) behind one interface."""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

from ..config import AnthropicCfg, LLMCfg, OllamaCfg
from ..errors import LLMError
from ..utils.retry import CircuitBreaker, retry_async

log = logging.getLogger(__name__)


class LLMClient(ABC):
    name: str = "llm"

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        system: str = "",
        max_tokens: int = 300,
        temperature: float = 0.3,
    ) -> str:
        """Return the assistant's reply text. Raise :class:`LLMError` on any failure."""

    async def aclose(self) -> None:
        """Release network resources (optional)."""


class OllamaClient(LLMClient):
    name = "ollama"

    def __init__(self, cfg: OllamaCfg) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise LLMError("httpx is not installed") from exc
        self._cfg = cfg
        self._httpx = httpx
        self._client = httpx.AsyncClient(base_url=cfg.base_url.rstrip("/"), timeout=cfg.timeout_s)

    async def complete(self, messages, *, system="", max_tokens=300, temperature=0.3) -> str:
        payload = {
            "model": self._cfg.model,
            "messages": ([{"role": "system", "content": system}] if system else []) + messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            resp = await self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            text = (resp.json().get("message") or {}).get("content", "")
        except (self._httpx.HTTPError, ValueError) as exc:
            raise LLMError(f"ollama request failed: {exc}") from exc
        if not text.strip():
            raise LLMError("ollama returned an empty response")
        return text.strip()

    async def aclose(self) -> None:
        await self._client.aclose()


class AnthropicClient(LLMClient):
    name = "anthropic"

    def __init__(self, cfg: AnthropicCfg) -> None:
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise LLMError(f"{cfg.api_key_env} is not set")
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise LLMError("anthropic package is not installed") from exc
        self._cfg = cfg
        self._client = AsyncAnthropic(api_key=key, timeout=cfg.timeout_s, max_retries=1)

    async def complete(self, messages, *, system="", max_tokens=300, temperature=0.3) -> str:
        # The API requires the conversation to start with a user turn.
        while messages and messages[0]["role"] != "user":
            messages = messages[1:]
        if not messages:
            raise LLMError("no user message to answer")
        try:
            resp = await self._client.messages.create(
                model=self._cfg.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or None,
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001 - SDK raises many APIError subclasses
            raise LLMError(f"anthropic request failed: {exc}") from exc
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        if not text:
            raise LLMError("anthropic returned an empty response")
        return text

    async def aclose(self) -> None:
        await self._client.close()


class FallbackLLM(LLMClient):
    """Try backends in order; retry transient errors; skip backends whose breaker is open."""

    name = "fallback-llm"

    def __init__(self, clients: list[LLMClient], cfg: LLMCfg | None = None) -> None:
        if not clients:
            raise ValueError("FallbackLLM needs at least one client")
        cfg = cfg or LLMCfg()
        self._clients = clients
        self._retries = cfg.retries
        self._breakers = {
            c.name: CircuitBreaker(cfg.breaker_threshold, cfg.breaker_reset_s) for c in clients
        }

    async def complete(self, messages, *, system="", max_tokens=300, temperature=0.3) -> str:
        errors: list[str] = []
        for client in self._clients:
            breaker = self._breakers[client.name]
            if not breaker.allow():
                errors.append(f"{client.name}: circuit open")
                continue
            try:
                text = await retry_async(
                    lambda c=client: c.complete(
                        messages, system=system, max_tokens=max_tokens, temperature=temperature
                    ),
                    attempts=self._retries + 1,
                    base_delay=0.4,
                    retry_on=(LLMError,),
                )
            except LLMError as exc:
                breaker.record_failure()
                log.warning("LLM backend %s failed: %s", client.name, exc)
                errors.append(f"{client.name}: {exc}")
                continue
            breaker.record_success()
            return text
        raise LLMError("all LLM backends failed: " + "; ".join(errors))

    async def aclose(self) -> None:
        for client in self._clients:
            await client.aclose()
