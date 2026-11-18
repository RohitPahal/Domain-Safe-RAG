"""Gemini adapter (``google-genai`` SDK, ``pip install rag-thesis[gemini]``).

With only ``temperature`` set, the request is exactly what the thesis runs sent: system messages
joined into ``system_instruction``, other messages as ``contents`` (empty ones skipped), and
``response_mime_type=application/json`` when JSON is asked for (now explicit via ``json_mode``
instead of the word "json" in the prompt).
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
    LLMQuotaExceededError,
    LLMTransientError,
    error_for_status,
    is_network_failure,
)
from rag_thesis.llm.types import ChatRequest, ChatResponse, Usage

# Finish reasons meaning the output was blocked, not merely short.
_BLOCKED_FINISH_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY"}
# Gemini 429s that mean a hard quota is used up: retrying soon won't help.
_QUOTA_MARKERS = ("per day", "limit: 0")


class GeminiChatModel(ChatModel):
    """One Gemini model. ``client`` is an already-built ``genai.Client`` (tests pass a fake)."""

    def __init__(
        self,
        *,
        model: str,
        provider: str = "gemini",
        api_key_env: str | None = "GEMINI_API_KEY",
        client: Any | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key_env = api_key_env
        self._client = client

    def complete(self, request: ChatRequest) -> ChatResponse:
        contents, config = to_gemini_request(request)
        client = self._get_client()
        start = time.perf_counter()
        try:
            response = client.models.generate_content(model=self.model, contents=contents, config=config)
        except Exception as exc:
            raise map_gemini_error(exc) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        text = str(getattr(response, "text", "") or "").strip()
        finish_reason = _finish_reason(response)
        block_reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        if not text and (block_reason or finish_reason in _BLOCKED_FINISH_REASONS):
            raise LLMContentFilterError(f"Gemini blocked the output (finish_reason={finish_reason}, {block_reason=})")

        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        total_tokens = getattr(usage, "total_token_count", None)
        return ChatResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            usage=Usage(prompt_tokens, completion_tokens, int(total_tokens) if total_tokens is not None else None),
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.api_key_env:
                raise LLMAuthError(f"No api_key_env is configured for provider {self.provider!r}.")
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                raise LLMAuthError(f"Set the environment variable {self.api_key_env} to call {self.provider}.")
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError("Gemini needs the google-genai package: pip install rag-thesis[gemini]") from exc
            self._client = genai.Client(api_key=api_key)
        return self._client


def to_gemini_request(request: ChatRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map a ``ChatRequest`` to Gemini ``contents`` and ``config`` (plain dicts the SDK accepts)."""
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in request.messages:
        content = message.content.strip()
        if not content:
            continue
        if message.role == "system":
            system_parts.append(content)
        else:
            role = "model" if message.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": content}]})
    if not contents:
        raise LLMBadRequestError("Gemini needs at least one non-system message with content.")

    config: dict[str, Any] = {"temperature": request.temperature}
    if system_parts:
        config["system_instruction"] = "\n\n".join(system_parts)
    if request.json_mode:
        config["response_mime_type"] = "application/json"
    if request.max_output_tokens is not None:
        config["max_output_tokens"] = request.max_output_tokens
    if request.seed is not None:
        config["seed"] = request.seed
    if request.timeout_s is not None:
        config["http_options"] = {"timeout": int(request.timeout_s * 1000)}  # milliseconds
    return contents, config


def map_gemini_error(exc: Exception) -> LLMError:
    """Map a google-genai exception (``APIError`` has an int ``code``) to a normalised error."""
    message = f"{type(exc).__name__}: {exc}"
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        if code == 429 and any(marker in str(exc).lower() for marker in _QUOTA_MARKERS):
            return LLMQuotaExceededError(message, status_code=code)
        return error_for_status(code, message)
    if is_network_failure(exc):
        return LLMTransientError(message)
    return LLMError(message)


def _finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    if reason is None:
        return None
    return str(getattr(reason, "value", reason))
