"""ChatModel wrappers: each cross-cutting behaviour is written once."""

from __future__ import annotations

import dataclasses
import random

import pytest

from rag_thesis import protection
from rag_thesis.llm.errors import LLMAuthError, LLMCallsDisabledError, LLMRateLimitError, LLMTransientError
from rag_thesis.llm.testing import FakeChatModel
from rag_thesis.llm.types import ChatMessage, ChatRequest, ChatResponse, Usage
from rag_thesis.llm.wrappers import (
    CachingChatModel,
    CallLog,
    OfflineGuardChatModel,
    RetryingChatModel,
    UsageTrackingChatModel,
)
from rag_thesis.protection import ProtectedPathError, WriteGuard

REQUEST = ChatRequest(messages=[ChatMessage("user", "Hi")])


def _retrying(replies, **kwargs):
    sleeps: list[float] = []
    inner = FakeChatModel(replies, provider="p", model="m")
    model = RetryingChatModel(inner, sleep=sleeps.append, jitter=0.0, **kwargs)
    return model, inner, sleeps


def test_retrying_keeps_the_inner_identity():
    model, _, _ = _retrying([])
    assert (model.provider, model.model) == ("p", "m")


def test_retryable_errors_are_retried_with_exponential_backoff():
    model, inner, sleeps = _retrying([LLMTransientError("503"), LLMRateLimitError("429"), "ok"])
    assert model.complete(REQUEST).text == "ok"
    assert len(inner.requests) == 3
    assert sleeps == [5.0, 10.0]


def test_gives_up_after_max_retries_with_the_last_error():
    errors = [LLMTransientError(f"fail {n}") for n in range(4)]
    model, inner, sleeps = _retrying(errors, max_retries=3)
    with pytest.raises(LLMTransientError, match="fail 3"):
        model.complete(REQUEST)
    assert len(inner.requests) == 4
    assert sleeps == [5.0, 10.0, 20.0]


@pytest.mark.parametrize("error", [LLMAuthError("bad key"), ValueError("a bug, not an LLM error")])
def test_non_retryable_errors_are_raised_at_once(error):
    model, inner, sleeps = _retrying([error, "never reached"])
    with pytest.raises(type(error)):
        model.complete(REQUEST)
    assert len(inner.requests) == 1
    assert sleeps == []


def test_zero_retries_means_one_attempt():
    model, inner, _ = _retrying([LLMTransientError("x")], max_retries=0)
    with pytest.raises(LLMTransientError):
        model.complete(REQUEST)
    assert len(inner.requests) == 1


def test_delay_is_capped_and_jitter_stays_within_bounds():
    model = RetryingChatModel(FakeChatModel(), base_delay_s=5, max_delay_s=12, jitter=0.5, rng=random.Random(1))
    for retry_number, base in [(1, 5.0), (2, 10.0), (3, 12.0), (8, 12.0)]:
        delay = model.delay_before_retry(retry_number)
        assert base <= delay <= base * 1.5
    with pytest.raises(ValueError):
        RetryingChatModel(FakeChatModel(), max_retries=-1)


def test_offline_guard_blocks_calls_unless_allowed():
    blocked_inner = FakeChatModel(["never sent"], provider="p", model="m")
    blocked = OfflineGuardChatModel(blocked_inner, allow_llm_calls=False)
    with pytest.raises(LLMCallsDisabledError, match="p/m"):
        blocked.complete(REQUEST)
    assert blocked_inner.requests == []  # nothing reached the provider

    allowed = OfflineGuardChatModel(FakeChatModel(["sent"]), allow_llm_calls=True)
    assert allowed.complete(REQUEST).text == "sent"


def test_offline_guard_error_is_not_retried():
    inner = FakeChatModel(["never sent"])
    model = RetryingChatModel(OfflineGuardChatModel(inner, allow_llm_calls=False), sleep=lambda s: None)
    with pytest.raises(LLMCallsDisabledError):
        model.complete(REQUEST)


def test_usage_tracking_records_successful_and_failed_calls():
    call_log = CallLog()
    inner = FakeChatModel(["ok", LLMAuthError("bad key")], provider="p", model="m", usage=Usage(7, 3, 12))
    model = UsageTrackingChatModel(inner, role="answer", call_log=call_log)

    model.complete(REQUEST)
    with pytest.raises(LLMAuthError):
        model.complete(REQUEST)

    first, second = call_log.as_dicts()
    assert first["role"] == "answer" and first["provider"] == "p" and first["model"] == "m"
    assert (first["status"], first["prompt_tokens"], first["completion_tokens"], first["total_tokens"]) == (
        "ok",
        7,
        3,
        12,
    )
    assert first["error"] is None and first["latency_ms"] >= 0
    assert second["status"] == "error" and second["total_tokens"] == 0
    assert second["error"] == "LLMAuthError: bad key"
    assert len(call_log) == 2 and call_log.as_dicts(start=1) == [second]


def test_one_call_log_is_shared_by_several_roles():
    call_log = CallLog()
    rewrite = UsageTrackingChatModel(FakeChatModel(["r"]), role="rewrite", call_log=call_log)
    answer = UsageTrackingChatModel(FakeChatModel(["a"]), role="answer", call_log=call_log)
    rewrite.complete(REQUEST)
    answer.complete(REQUEST)
    assert [call["role"] for call in call_log.as_dicts()] == ["rewrite", "answer"]


def test_calls_blocked_by_the_offline_guard_are_not_recorded():
    call_log = CallLog()
    guarded = OfflineGuardChatModel(FakeChatModel(["never sent"]), allow_llm_calls=False)
    model = UsageTrackingChatModel(guarded, role="judge", call_log=call_log)
    with pytest.raises(LLMCallsDisabledError):
        model.complete(REQUEST)
    assert len(call_log) == 0


def test_cache_serves_repeats_without_calling_the_inner_model(tmp_path):
    cache_path = tmp_path / "llm.jsonl"
    reply = ChatResponse(text="BRCA1", provider="p", model="m", usage=Usage(10, 2, 15), finish_reason="STOP")
    inner = FakeChatModel([reply], provider="p", model="m")
    model = CachingChatModel(inner, cache_path=cache_path)

    assert model.complete(REQUEST) == reply
    assert model.complete(REQUEST) == reply
    assert len(inner.requests) == 1

    # A new process (fresh wrapper, same file) replays it too, with no inner model call at all.
    replay = CachingChatModel(FakeChatModel([], provider="p", model="m"), cache_path=cache_path)
    assert replay.complete(REQUEST) == reply


def test_cache_key_covers_provider_model_and_output_changing_parameters(tmp_path):
    cache_path = tmp_path / "llm.jsonl"
    CachingChatModel(FakeChatModel(["first"], provider="p", model="m"), cache_path=cache_path).complete(REQUEST)

    variants = [
        ("other-provider", "m", REQUEST),  # same model name on another provider
        ("p", "other-model", REQUEST),
        ("p", "m", dataclasses.replace(REQUEST, json_mode=True)),
        ("p", "m", dataclasses.replace(REQUEST, temperature=0.5)),
        ("p", "m", dataclasses.replace(REQUEST, max_output_tokens=64)),
        ("p", "m", dataclasses.replace(REQUEST, seed=1)),
        ("p", "m", ChatRequest(messages=[ChatMessage("user", "Hi!")])),
    ]
    for provider, model_name, request in variants:
        inner = FakeChatModel(["fresh"], provider=provider, model=model_name)
        assert CachingChatModel(inner, cache_path=cache_path).complete(request).text == "fresh"
        assert len(inner.requests) == 1

    # The timeout can't change the reply, so it is not part of the key.
    inner = FakeChatModel([], provider="p", model="m")
    timed = dataclasses.replace(REQUEST, timeout_s=5)
    assert CachingChatModel(inner, cache_path=cache_path).complete(timed).text == "first"


def test_failed_calls_are_not_cached(tmp_path):
    cache_path = tmp_path / "llm.jsonl"
    inner = FakeChatModel([LLMAuthError("bad key"), "ok"])
    model = CachingChatModel(inner, cache_path=cache_path)
    with pytest.raises(LLMAuthError):
        model.complete(REQUEST)
    assert not cache_path.exists()
    assert model.complete(REQUEST).text == "ok"


def test_cache_with_offline_guard_is_the_replay_model(tmp_path):
    cache_path = tmp_path / "llm.jsonl"
    CachingChatModel(FakeChatModel(["cached"]), cache_path=cache_path).complete(REQUEST)

    offline = FakeChatModel([])
    replay = CachingChatModel(OfflineGuardChatModel(offline, allow_llm_calls=False), cache_path=cache_path)
    assert replay.complete(REQUEST).text == "cached"
    with pytest.raises(LLMCallsDisabledError):
        replay.complete(ChatRequest(messages=[ChatMessage("user", "not cached")]))
    assert offline.requests == []


def test_protected_cache_replays_but_refuses_a_miss_before_the_call(tmp_path, monkeypatch):
    folder = tmp_path / "protected"
    folder.mkdir()
    cache_path = folder / "llm.jsonl"
    CachingChatModel(FakeChatModel(["cached"]), cache_path=cache_path).complete(REQUEST)
    monkeypatch.setattr(protection, "DEFAULT_GUARD", WriteGuard([folder]))

    inner = FakeChatModel(["never sent"])
    model = CachingChatModel(inner, cache_path=cache_path)
    assert model.complete(REQUEST).text == "cached"  # reading is allowed
    with pytest.raises(ProtectedPathError):
        model.complete(ChatRequest(messages=[ChatMessage("user", "new")]))
    assert inner.requests == []  # refused before the real call
