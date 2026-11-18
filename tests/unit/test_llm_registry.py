"""LLM registry: the one composition root builds and wraps every role's chat model."""

from __future__ import annotations

import pytest

from rag_thesis.llm import registry
from rag_thesis.llm.errors import LLMAuthError, LLMCallsDisabledError, LLMTransientError
from rag_thesis.llm.registry import ADAPTERS, RoleParametersChatModel, build_chat_model, register_provider
from rag_thesis.llm.settings import resolve_llm_settings
from rag_thesis.llm.testing import FakeChatModel
from rag_thesis.llm.types import ChatMessage, ChatRequest
from rag_thesis.llm.wrappers import (
    CachingChatModel,
    CallLog,
    OfflineGuardChatModel,
    RetryingChatModel,
    UsageTrackingChatModel,
)

REQUEST = ChatRequest(messages=[ChatMessage("user", "Hi")], temperature=0.9, json_mode=True)


class ScriptedAdapter(FakeChatModel):
    """A registered adapter type for tests; records the arguments the registry passes."""

    built: list[dict] = []
    script: list = []

    def __init__(self, *, model, provider, api_key_env, **options):
        super().__init__(list(ScriptedAdapter.script), provider=provider, model=model)
        ScriptedAdapter.built.append({"model": model, "provider": provider, "api_key_env": api_key_env, **options})


@pytest.fixture
def fake_type(monkeypatch):
    monkeypatch.setattr(registry, "ADAPTERS", dict(ADAPTERS))
    register_provider("scripted", ScriptedAdapter)
    ScriptedAdapter.built = []
    ScriptedAdapter.script = []
    return ScriptedAdapter


def _settings(*, cache_path=None, api_key_env="RAG_THESIS_TEST_KEY", **role_overrides):
    block = {
        "providers": {"prov": {"type": "scripted", "api_key_env": api_key_env, "extra_option": 1}},
        "roles": {"answer": {"provider": "prov", "model": "answer-model", **role_overrides}},
    }
    if cache_path:
        block["cache_path"] = str(cache_path)
    return resolve_llm_settings({"llm": block})


def _chain(model):
    chain = [type(model)]
    while hasattr(model, "inner"):
        model = model.inner
        chain.append(type(model))
    return chain


def test_built_in_provider_types():
    assert set(ADAPTERS) == {"gemini", "openai_compatible", "anthropic"}


def test_wrappers_are_composed_in_the_documented_order(fake_type, tmp_path):
    model = build_chat_model(
        "answer", _settings(cache_path=tmp_path / "c.jsonl"), allow_llm_calls=False, call_log=CallLog()
    )
    assert _chain(model) == [
        RoleParametersChatModel,
        CachingChatModel,
        UsageTrackingChatModel,
        OfflineGuardChatModel,
        RetryingChatModel,
        ScriptedAdapter,
    ]
    assert (model.provider, model.model) == ("prov", "answer-model")
    assert fake_type.built == [
        {"model": "answer-model", "provider": "prov", "api_key_env": "RAG_THESIS_TEST_KEY", "extra_option": 1}
    ]


def test_optional_wrappers_are_left_out_when_not_configured(fake_type):
    model = build_chat_model("answer", _settings(), allow_llm_calls=False)
    assert CachingChatModel not in _chain(model) and UsageTrackingChatModel not in _chain(model)


def test_role_parameters_come_from_the_config_not_the_caller(fake_type, monkeypatch):
    monkeypatch.setenv("RAG_THESIS_TEST_KEY", "x")
    fake_type.script = ["ok"]
    settings = _settings(max_output_tokens=32, seed=5, timeout_s=9)
    model = build_chat_model("answer", settings, allow_llm_calls=True)
    model.complete(REQUEST)

    sent = model.inner.inner.inner.requests[0]  # params -> guard -> retry -> adapter
    assert (sent.temperature, sent.max_output_tokens, sent.seed, sent.timeout_s) == (0.0, 32, 5, 9)
    assert sent.json_mode is True and sent.messages == REQUEST.messages  # what to ask stays the caller's


def test_without_permission_no_real_call_is_made(fake_type):
    fake_type.script = ["never sent"]
    model = build_chat_model("answer", _settings(), allow_llm_calls=False)
    with pytest.raises(LLMCallsDisabledError):
        model.complete(REQUEST)


def test_cache_replays_offline_and_real_calls_are_tracked_and_retried(fake_type, monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_THESIS_TEST_KEY", "x")
    monkeypatch.setattr(RetryingChatModel, "delay_before_retry", lambda self, n: 0.0)
    fake_type.script = [LLMTransientError("503"), "BRCA1"]
    cache_path = tmp_path / "c.jsonl"
    call_log = CallLog()
    live = build_chat_model("answer", _settings(cache_path=cache_path), allow_llm_calls=True, call_log=call_log)
    assert live.complete(REQUEST).text == "BRCA1"  # retried once, then cached
    assert [c["status"] for c in call_log.as_dicts()] == ["ok"]  # one real call as seen by the caller

    fake_type.script = []
    replay = build_chat_model("answer", _settings(cache_path=cache_path), allow_llm_calls=False, call_log=call_log)
    assert replay.complete(REQUEST).text == "BRCA1"  # offline, from the cache
    assert len(call_log) == 1  # cache hits are not real calls


def test_missing_api_key_fails_at_build_time_only_when_calls_are_allowed(fake_type, monkeypatch):
    monkeypatch.delenv("RAG_THESIS_TEST_KEY", raising=False)
    build_chat_model("answer", _settings(), allow_llm_calls=False)  # replay-only: no key needed
    with pytest.raises(LLMAuthError, match="RAG_THESIS_TEST_KEY"):
        build_chat_model("answer", _settings(), allow_llm_calls=True)
    build_chat_model("answer", _settings(api_key_env=None), allow_llm_calls=True)  # e.g. local Ollama


def test_unconfigured_role_and_unknown_type_fail_clearly(fake_type, monkeypatch):
    with pytest.raises(ValueError, match="'judge' role"):
        build_chat_model("judge", _settings(), allow_llm_calls=False)
    monkeypatch.setattr(registry, "ADAPTERS", {})
    with pytest.raises(ValueError, match="No adapter"):
        build_chat_model("answer", _settings(), allow_llm_calls=False)


def test_real_adapters_are_built_from_the_thesis_config(monkeypatch):
    settings = resolve_llm_settings({"llm_provider": "gemini", "llm_answer_model": "gemini-2.5-flash"})
    model = build_chat_model("answer", settings, allow_llm_calls=False)
    adapter = model
    while hasattr(adapter, "inner"):
        adapter = adapter.inner
    assert type(adapter).__name__ == "GeminiChatModel"
    assert (adapter.provider, adapter.model, adapter.api_key_env) == ("gemini", "gemini-2.5-flash", "GEMINI_API_KEY")
