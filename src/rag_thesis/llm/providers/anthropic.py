"""Anthropic adapter (``anthropic`` SDK, ``pip install rag-thesis[anthropic]``), Messages API.

Differences from the other adapters, all handled here:

- system messages go into the separate ``system`` field;
- ``max_tokens`` is required, so ``max_output_tokens`` falls back to ``default_max_tokens``;
- there is no native JSON mode: ``json_mode`` adds ``JSON_INSTRUCTION`` to the system prompt and
  the shared tolerant parser (``ChatModel.complete_json``) does the rest;
- ``seed`` is not supported by the API and is not sent;
- usage has no total; it is input + output tokens.
"""

from __future__ import annotations

import os
import time
from typing import Any

from rag_thesis.llm.base import ChatModel
from rag_thesis.llm.errors import (
    LLMAuthError,
    LLMBadRequestError,
    LLMContentFilterError,
    LLMError,
    LLMTransientError,
    error_for_status,
    is_network_failure,
)
from rag_thesis.llm.types import ChatRequest, ChatResponse, Usage

JSON_INSTRUCTION = "Respond with a single JSON object only, with no other text before or after it."


class AnthropicChatModel(ChatModel):
    """One Claude model. ``client`` is a built ``anthropic.Anthropic`` (tests pass a fake)."""

    def __init__(
        self,
        *,
        model: str,
        provider: str = "anthropic",
        api_key_env: str | None = "ANTHROPIC_API_KEY",
        base_url: str | None = None,
        default_max_tokens: int = 1024,
        client: Any | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.default_max_tokens = default_max_tokens
        self._client = client

    def complete(self, request: ChatRequest) -> ChatResponse:
        kwargs = self.to_request_kwargs(request)
        client = self._get_client()
        start = time.perf_counter()
        try:
            response = client.messages.create(**kwargs)
        except Exception as exc:
            raise map_anthropic_error(exc) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        blocks = getattr(response, "content", None) or []
        text = "".join(str(getattr(b, "text", "")) for b in blocks if getattr(b, "type", None) == "text").strip()
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            raise LLMContentFilterError(f"{self.provider} refused to answer (stop_reason=refusal)")

        usage = getattr(response, "usage", None)
        return ChatResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            usage=Usage(
                int(getattr(usage, "input_tokens", 0) or 0),
                int(getattr(usage, "output_tokens", 0) or 0),
            ),
            latency_ms=latency_ms,
            finish_reason=str(stop_reason) if stop_reason is not None else None,
        )

    def to_request_kwargs(self, request: ChatRequest) -> dict[str, Any]:
        """Map a ``ChatRequest`` to ``messages.create`` keyword arguments."""
        system_parts = [m.content.strip() for m in request.messages if m.role == "system" and m.content.strip()]
        if request.json_mode:
            system_parts.append(JSON_INSTRUCTION)
        messages = [{"role": m.role, "content": m.content} for m in request.messages if m.role != "system"]
        if not messages:
            raise LLMBadRequestError("Anthropic needs at least one non-system message.")

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_output_tokens or self.default_max_tokens,
            "messages": messages,
            "temperature": request.temperature,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if request.timeout_s is not None:
            kwargs["timeout"] = request.timeout_s
        return kwargs

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.api_key_env:
                raise LLMAuthError(f"No api_key_env is configured for provider {self.provider!r}.")
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                raise LLMAuthError(f"Set the environment variable {self.api_key_env} to call {self.provider}.")
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError("Anthropic needs the anthropic package: pip install rag-thesis[anthropic]") from exc
            self._client = anthropic.Anthropic(api_key=api_key, base_url=self.base_url)
        return self._client


def map_anthropic_error(exc: Exception) -> LLMError:
    """Map an ``anthropic`` SDK exception (``APIStatusError`` has ``status_code``; 529 = overloaded)."""
    message = f"{type(exc).__name__}: {exc}"
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return error_for_status(status, message)
    if is_network_failure(exc):  # APITimeoutError, APIConnectionError
        return LLMTransientError(message)
    return LLMError(message)
