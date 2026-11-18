"""Composition root of the LLM layer: the only place that builds concrete chat models.

``build_chat_model(role, settings, ...)`` looks up the role's provider type in ``ADAPTERS``, builds
the adapter and wraps it (outermost first)::

    RoleParameters -> Caching -> UsageTracking -> OfflineGuard -> Retrying -> adapter

- ``RoleParameters`` applies the role's configured temperature / output limit / timeout / seed, so
  callers only say *what* to ask (messages, ``json_mode``), never how;
- ``Caching`` (when ``llm.cache_path`` is set): repeats and replays cost no call;
- ``UsageTracking`` (when a ``CallLog`` is given): one record per real call;
- ``OfflineGuard``: real calls only with ``allow_llm_calls``;
- ``Retrying``: the one retry policy, with the role's ``max_retries``.

Adding a provider = one adapter module + one ``ADAPTERS`` entry (or ``register_provider``).
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable

from rag_thesis.llm.base import ChatModel
from rag_thesis.llm.errors import LLMAuthError
from rag_thesis.llm.providers.anthropic import AnthropicChatModel
from rag_thesis.llm.providers.gemini import GeminiChatModel
from rag_thesis.llm.providers.openai_compatible import OpenAICompatibleChatModel
from rag_thesis.llm.settings import LLMSettings, RoleSettings
from rag_thesis.llm.types import ChatRequest, ChatResponse
from rag_thesis.llm.wrappers import (
    CachingChatModel,
    CallLog,
    ChatModelWrapper,
    OfflineGuardChatModel,
    RetryingChatModel,
    UsageTrackingChatModel,
)

# Provider type (``llm.providers.<name>.type``) -> adapter class.
ADAPTERS: dict[str, Callable[..., ChatModel]] = {
    "gemini": GeminiChatModel,
    "openai_compatible": OpenAICompatibleChatModel,
    "anthropic": AnthropicChatModel,
}


def register_provider(type_name: str, adapter: Callable[..., ChatModel]) -> None:
    """Make another adapter available under ``type_name`` (it must accept the adapter arguments)."""
    ADAPTERS[type_name] = adapter


class RoleParametersChatModel(ChatModelWrapper):
    """Sets the role's configured call parameters on every request (callers set only messages + JSON)."""

    def __init__(self, inner: ChatModel, *, settings: RoleSettings) -> None:
        super().__init__(inner)
        self.settings = settings

    def complete(self, request: ChatRequest) -> ChatResponse:
        configured = dataclasses.replace(
            request,
            temperature=self.settings.temperature,
            max_output_tokens=self.settings.max_output_tokens,
            timeout_s=self.settings.timeout_s,
            seed=self.settings.seed,
        )
        return self.inner.complete(configured)


def build_chat_model(
    role: str,
    settings: LLMSettings,
    *,
    allow_llm_calls: bool,
    call_log: CallLog | None = None,
) -> ChatModel:
    """Build the fully wrapped chat model for ``role`` from the config's LLM settings.

    With ``allow_llm_calls`` the API key variable is checked now, before any work is done.
    """
    role_settings = settings.role(role)
    provider = role_settings.provider
    if provider.type not in ADAPTERS:
        raise ValueError(f"No adapter for provider type {provider.type!r} (llm.providers.{provider.name}).")
    if allow_llm_calls and provider.api_key_env and not os.getenv(provider.api_key_env):
        raise LLMAuthError(
            f"LLM calls are allowed but {provider.api_key_env} is not set (needed by the {role!r} role, "
            f"provider {provider.name!r}). Set it in the environment or .env."
        )

    adapter_kwargs = {"model": role_settings.model, "provider": provider.name, "api_key_env": provider.api_key_env}
    if provider.base_url is not None:
        adapter_kwargs["base_url"] = provider.base_url
    model: ChatModel = ADAPTERS[provider.type](**adapter_kwargs, **provider.options)

    model = RetryingChatModel(model, max_retries=role_settings.max_retries)
    model = OfflineGuardChatModel(model, allow_llm_calls=allow_llm_calls)
    if call_log is not None:
        model = UsageTrackingChatModel(model, role=role, call_log=call_log)
    if settings.cache_path:
        model = CachingChatModel(model, cache_path=settings.cache_path)
    return RoleParametersChatModel(model, settings=role_settings)
