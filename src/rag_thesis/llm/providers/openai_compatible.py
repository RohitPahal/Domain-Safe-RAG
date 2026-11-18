"""OpenAI-compatible adapter (``openai`` SDK, ``pip install rag-thesis[openai]``).

One adapter for every server that speaks the OpenAI Chat Completions API, chosen by ``base_url``:
OpenAI itself, Groq, Together, OpenRouter, and local servers such as Ollama, vLLM or LM Studio
(``api_key_env=None`` when the server needs no key). With only ``temperature`` set, the call is
exactly what the old client sent: ``model``, ``messages``, ``temperature``.
"""

from __future__ import annotations

import os
import time
from typing import Any

from rag_thesis.llm.base import ChatModel
from rag_thesis.llm.errors import (
    LLMAuthError,
    LLMContentFilterError,
    LLMError,
    LLMQuotaExceededError,
    LLMTransientError,
    error_for_status,
    is_network_failure,
)
from rag_thesis.llm.types import ChatRequest, ChatResponse, Usage

# Placeholder key for local servers that need none (the SDK refuses an empty key).
_NO_KEY = "not-needed"


class OpenAICompatibleChatModel(ChatModel):
    """One model on an OpenAI-compatible server. ``client`` is a built ``openai.OpenAI`` (tests: a fake).

    ``max_tokens_param`` names the output-limit field: ``max_tokens`` (Groq, Ollama, vLLM, older
    OpenAI models) or ``max_completion_tokens`` (newer OpenAI models).
    """

    def __init__(
        self,
        *,
        model: str,
        provider: str = "openai",
        api_key_env: str | None = "OPENAI_API_KEY",
        base_url: str | None = None,
        max_tokens_param: str = "max_tokens",
        client: Any | None = None,
    ) -> None:
        if max_tokens_param not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("max_tokens_param must be 'max_tokens' or 'max_completion_tokens'")
        self.provider = provider
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.max_tokens_param = max_tokens_param
        self._client = client

    def complete(self, request: ChatRequest) -> ChatResponse:
        kwargs = self.to_request_kwargs(request)
        client = self._get_client()
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise map_openai_error(exc) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        choices = getattr(response, "choices", None) or []
        first = choices[0] if choices else None
        text = _content_to_text(getattr(getattr(first, "message", None), "content", "")).strip()
        finish_reason = getattr(first, "finish_reason", None)
        if not text and finish_reason == "content_filter":
            raise LLMContentFilterError(f"{self.provider} filtered the output (finish_reason=content_filter)")

        usage = getattr(response, "usage", None)
        total_tokens = getattr(usage, "total_tokens", None)
        return ChatResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            usage=Usage(
                int(getattr(usage, "prompt_tokens", 0) or 0),
                int(getattr(usage, "completion_tokens", 0) or 0),
                int(total_tokens) if total_tokens is not None else None,
            ),
            latency_ms=latency_ms,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
        )

    def to_request_kwargs(self, request: ChatRequest) -> dict[str, Any]:
        """Map a ``ChatRequest`` to ``chat.completions.create`` keyword arguments."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": message.role, "content": message.content} for message in request.messages],
            "temperature": request.temperature,
        }
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if request.max_output_tokens is not None:
            kwargs[self.max_tokens_param] = request.max_output_tokens
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.timeout_s is not None:
            kwargs["timeout"] = request.timeout_s
        return kwargs

    def _get_client(self) -> Any:
        if self._client is None:
            api_key = _NO_KEY
            if self.api_key_env is not None:
                api_key = os.getenv(self.api_key_env) or ""
                if not api_key:
                    raise LLMAuthError(f"Set the environment variable {self.api_key_env} to call {self.provider}.")
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError(
                    "OpenAI-compatible providers need the openai package: pip install rag-thesis[openai]"
                ) from exc
            self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        return self._client


def map_openai_error(exc: Exception) -> LLMError:
    """Map an ``openai`` SDK exception (``APIStatusError`` has ``status_code``) to a normalised error."""
    message = f"{type(exc).__name__}: {exc}"
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status == 429 and "insufficient_quota" in str(exc):
            return LLMQuotaExceededError(message, status_code=status)
        return error_for_status(status, message)
    if is_network_failure(exc):  # APITimeoutError, APIConnectionError
        return LLMTransientError(message)
    return LLMError(message)


def _content_to_text(content: Any) -> str:
    """Message content is a string, or a list of parts (strings or ``{"text": ...}`` objects)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            text = item if isinstance(item, str) else (item.get("text") if isinstance(item, dict) else None)
            text = text if text is not None else getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""
