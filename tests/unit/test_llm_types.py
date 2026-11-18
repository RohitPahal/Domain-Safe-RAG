"""Provider-neutral LLM types: defaults, immutability and input checks."""

from __future__ import annotations

import dataclasses

import pytest

from rag_thesis.llm.types import ChatMessage, ChatRequest, ChatResponse, Usage


def _request(**overrides) -> ChatRequest:
    return ChatRequest(messages=[ChatMessage("system", "Be brief."), ChatMessage("user", "Hi")], **overrides)


def test_request_defaults_are_deterministic_and_explicit():
    request = _request()
    assert request.temperature == 0.0
    assert request.json_mode is False
    assert request.max_output_tokens is None
    assert request.seed is None
    assert request.timeout_s is None


def test_request_stores_messages_as_a_tuple_and_is_immutable():
    messages = [ChatMessage("user", "Hi")]
    request = ChatRequest(messages=messages)
    messages.append(ChatMessage("user", "changed later"))

    assert request.messages == (ChatMessage("user", "Hi"),)
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.temperature = 1.0  # type: ignore[misc]


def test_equal_requests_are_equal_and_hashable():
    assert _request(json_mode=True) == _request(json_mode=True)
    assert hash(_request(seed=7)) == hash(_request(seed=7))
    assert _request(json_mode=True) != _request()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"messages": []}, ValueError),
        ({"messages": "Hi"}, TypeError),
        ({"messages": [{"role": "user", "content": "Hi"}]}, TypeError),
        ({"messages": [ChatMessage("user", "Hi")], "temperature": -0.1}, ValueError),
        ({"messages": [ChatMessage("user", "Hi")], "max_output_tokens": 0}, ValueError),
        ({"messages": [ChatMessage("user", "Hi")], "timeout_s": 0}, ValueError),
    ],
)
def test_malformed_requests_fail_when_built(kwargs, error):
    with pytest.raises(error):
        ChatRequest(**kwargs)


def test_message_rejects_unknown_role_and_non_text_content():
    with pytest.raises(ValueError, match="role"):
        ChatMessage("model", "Hi")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="content"):
        ChatMessage("user", None)  # type: ignore[arg-type]


def test_usage_total_defaults_to_prompt_plus_completion():
    assert Usage(prompt_tokens=10, completion_tokens=5).total_tokens == 15
    assert Usage().total_tokens == 0
    # A provider-reported total is kept as is (some count extra tokens, e.g. for reasoning).
    assert Usage(prompt_tokens=10, completion_tokens=5, total_tokens=40).total_tokens == 40
    with pytest.raises(ValueError):
        Usage(prompt_tokens=-1)


def test_response_defaults_and_checks():
    response = ChatResponse(text="insulin", provider="fake", model="fake-model")
    assert response.usage == Usage()
    assert response.latency_ms == 0.0
    assert response.finish_reason is None

    with pytest.raises(ValueError):
        ChatResponse(text="x", provider="", model="fake-model")
    with pytest.raises(TypeError):
        ChatResponse(text=None, provider="fake", model="fake-model")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ChatResponse(text="x", provider="fake", model="m", latency_ms=-1)
