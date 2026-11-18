"""Provider adapter contract: every adapter must pass the same tests.

Each adapter gets a small harness: a fake SDK client (records the call, returns a canned response
or raises), plus how to build that SDK's response and errors. No network, no SDK needed. Adapter-
specific checks (exact request mapping, real SDK exception classes) follow the shared contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from rag_thesis.llm.base import ChatModel
from rag_thesis.llm.errors import (
    LLMAuthError,
    LLMBadRequestError,
    LLMContentFilterError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMTransientError,
)
from rag_thesis.llm.providers.anthropic import JSON_INSTRUCTION, AnthropicChatModel
from rag_thesis.llm.providers.gemini import GeminiChatModel, to_gemini_request
from rag_thesis.llm.providers.openai_compatible import OpenAICompatibleChatModel
from rag_thesis.llm.types import ChatMessage, ChatRequest

SYSTEM = "You answer biomedical questions."
USER = "Which gene is mutated?"
REQUEST = ChatRequest(messages=[ChatMessage("system", SYSTEM), ChatMessage("user", USER)])


class FakeSDKError(Exception):
    """Stands in for an SDK's HTTP error: carries the status the way that SDK does."""


class ReadTimeout(Exception):
    """Same class name as httpx's timeout, which the adapters recognise as a network failure."""


# --- Harnesses --------------------------------------------------------------------------------


class _Recorder:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


@dataclass
class Harness:
    name: str
    reports_total: bool = True  # does the provider report its own total token count?

    def client(self, outcome: Any) -> Any: ...
    def adapter(self, client: Any | None, **kwargs: Any) -> ChatModel: ...
    def response(self, text: str, prompt: int, completion: int, total: int | None, finish: str) -> Any: ...
    def http_error(self, status: int, message: str = "error") -> Exception: ...
    def sent_json_mode(self, call: dict[str, Any]) -> bool: ...
    def sent_system(self, call: dict[str, Any]) -> str | None: ...
    def sent_user(self, call: dict[str, Any]) -> str: ...


class GeminiHarness(Harness):
    def client(self, outcome):
        recorder = _Recorder(outcome)
        return SimpleNamespace(models=SimpleNamespace(generate_content=recorder), recorder=recorder)

    def adapter(self, client, **kwargs):
        return GeminiChatModel(model="test-model", client=client, **kwargs)

    def response(self, text, prompt, completion, total, finish):
        usage = SimpleNamespace(prompt_token_count=prompt, candidates_token_count=completion, total_token_count=total)
        candidate = SimpleNamespace(finish_reason=SimpleNamespace(value=finish))
        return SimpleNamespace(text=text, usage_metadata=usage, candidates=[candidate], prompt_feedback=None)

    def http_error(self, status, message="error"):
        error = FakeSDKError(message)
        error.code = status  # google-genai APIError carries the HTTP status as `code`
        return error

    def sent_json_mode(self, call):
        return call["config"].get("response_mime_type") == "application/json"

    def sent_system(self, call):
        return call["config"].get("system_instruction")

    def sent_user(self, call):
        return call["contents"][-1]["parts"][0]["text"]


class OpenAIHarness(Harness):
    def client(self, outcome):
        recorder = _Recorder(outcome)
        completions = SimpleNamespace(create=recorder)
        return SimpleNamespace(chat=SimpleNamespace(completions=completions), recorder=recorder)

    def adapter(self, client, **kwargs):
        return OpenAICompatibleChatModel(model="test-model", client=client, **kwargs)

    def response(self, text, prompt, completion, total, finish):
        choice = SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish.lower())
        usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
        return SimpleNamespace(choices=[choice], usage=usage)

    def http_error(self, status, message="error"):
        error = FakeSDKError(message)
        error.status_code = status  # openai APIStatusError carries the HTTP status as `status_code`
        return error

    def sent_json_mode(self, call):
        return call.get("response_format") == {"type": "json_object"}

    def sent_system(self, call):
        return next((m["content"] for m in call["messages"] if m["role"] == "system"), None)

    def sent_user(self, call):
        return call["messages"][-1]["content"]


class AnthropicHarness(Harness):
    def client(self, outcome):
        recorder = _Recorder(outcome)
        return SimpleNamespace(messages=SimpleNamespace(create=recorder), recorder=recorder)

    def adapter(self, client, **kwargs):
        return AnthropicChatModel(model="test-model", client=client, **kwargs)

    def response(self, text, prompt, completion, total, finish):
        blocks = [SimpleNamespace(type="text", text=text)]
        usage = SimpleNamespace(input_tokens=prompt, output_tokens=completion)  # no total in this API
        return SimpleNamespace(content=blocks, usage=usage, stop_reason="end_turn" if finish == "STOP" else finish)

    def http_error(self, status, message="error"):
        error = FakeSDKError(message)
        error.status_code = status  # anthropic APIStatusError carries the HTTP status as `status_code`
        return error

    def sent_json_mode(self, call):
        return JSON_INSTRUCTION in call.get("system", "")

    def sent_system(self, call):
        return call.get("system")

    def sent_user(self, call):
        return call["messages"][-1]["content"]


HARNESSES = [
    GeminiHarness("gemini"),
    OpenAIHarness("openai_compatible"),
    AnthropicHarness("anthropic", reports_total=False),
]


@pytest.fixture(params=HARNESSES, ids=lambda h: h.name)
def harness(request):
    return request.param


def _call(harness: Harness, outcome: Any, request: ChatRequest = REQUEST, **kwargs: Any):
    client = harness.client(outcome)
    response = harness.adapter(client, **kwargs).complete(request)
    return response, client.recorder.calls[0]


# --- Shared contract --------------------------------------------------------------------------


def test_returns_text_usage_identity_and_finish_reason(harness):
    response, _ = _call(harness, harness.response("  BRCA1 \n", 11, 3, 20, "STOP"), provider="my-provider")
    assert response.text == "BRCA1"
    expected_total = 20 if harness.reports_total else 14  # a reported total is kept, else the sum
    usage = response.usage
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (11, 3, expected_total)
    assert response.provider == "my-provider" and response.model == "test-model"
    assert response.finish_reason is not None and response.latency_ms >= 0


def test_total_tokens_default_to_prompt_plus_completion(harness):
    response, _ = _call(harness, harness.response("x", 4, 2, None, "STOP"))
    assert response.usage.total_tokens == 6


def test_sends_model_system_and_user_messages(harness):
    _, call = _call(harness, harness.response("x", 1, 1, 2, "STOP"))
    assert call["model"] == "test-model"
    assert harness.sent_system(call) == SYSTEM
    assert harness.sent_user(call) == USER


def test_json_mode_is_explicit(harness):
    _, plain = _call(harness, harness.response("x", 1, 1, 2, "STOP"))
    json_request = ChatRequest(messages=REQUEST.messages, json_mode=True)
    _, json_call = _call(harness, harness.response("{}", 1, 1, 2, "STOP"), json_request)
    assert not harness.sent_json_mode(plain)
    assert harness.sent_json_mode(json_call)


def test_word_json_in_the_prompt_does_not_switch_json_mode_on(harness):
    request = ChatRequest(messages=[ChatMessage("user", "Return JSON only.")])
    _, call = _call(harness, harness.response("{}", 1, 1, 2, "STOP"), request)
    assert not harness.sent_json_mode(call)  # P2: never inferred from prompt text


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, LLMRateLimitError),
        (401, LLMAuthError),
        (403, LLMAuthError),
        (400, LLMBadRequestError),
        (404, LLMBadRequestError),
        (500, LLMTransientError),
        (503, LLMTransientError),
    ],
)
def test_sdk_http_errors_are_normalised(harness, status, expected):
    with pytest.raises(expected) as info:
        _call(harness, harness.http_error(status))
    assert info.value.status_code == status
    assert isinstance(info.value.__cause__, FakeSDKError)  # the original error is kept


def test_timeouts_are_transient(harness):
    with pytest.raises(LLMTransientError):
        _call(harness, ReadTimeout("read timed out"))


def test_missing_api_key_fails_before_any_network_call(harness, monkeypatch):
    monkeypatch.delenv("RAG_THESIS_TEST_KEY", raising=False)
    model = harness.adapter(None, api_key_env="RAG_THESIS_TEST_KEY")
    with pytest.raises(LLMAuthError, match="RAG_THESIS_TEST_KEY"):
        model.complete(REQUEST)


# --- Gemini specifics -------------------------------------------------------------------------


# What the thesis client (`llm/client.py`, since deleted) built for the messages below, recorded
# from that code before it was deleted.
_THESIS_CONTENTS = [
    {"role": "user", "parts": [{"text": "Question: What is BRCA1?"}]},
    {"role": "user", "parts": [{"text": "Again"}]},
]
THESIS_GEMINI_REQUESTS = {
    False: (_THESIS_CONTENTS, {"temperature": 0, "system_instruction": "Answer with one line."}),
    True: (
        _THESIS_CONTENTS,
        {
            "temperature": 0,
            "system_instruction": "Rewrite the question. Return JSON only.",
            "response_mime_type": "application/json",
        },
    ),
}


@pytest.mark.parametrize("json_mode", [False, True])
def test_gemini_request_is_identical_to_the_thesis_client(json_mode):
    # The old client switched JSON on when "json" appeared in the prompt; callers now set
    # json_mode explicitly for exactly those prompts (rewrite, judge).
    system = "Rewrite the question. Return JSON only." if json_mode else "Answer with one line."
    messages = [
        {"role": "system", "content": f"  {system} "},
        {"role": "user", "content": "Question: What is BRCA1?"},
        {"role": "assistant", "content": "  "},  # empty messages are skipped by both
        {"role": "user", "content": "Again"},
    ]
    request = ChatRequest(messages=[ChatMessage(m["role"], m["content"]) for m in messages], json_mode=json_mode)
    assert to_gemini_request(request) == THESIS_GEMINI_REQUESTS[json_mode]


def test_gemini_maps_new_parameters_only_when_set():
    request = ChatRequest(messages=[ChatMessage("user", "Hi")], max_output_tokens=64, seed=7, timeout_s=2.5)
    _, config = to_gemini_request(request)
    assert config == {"temperature": 0.0, "max_output_tokens": 64, "seed": 7, "http_options": {"timeout": 2500}}


def test_gemini_needs_a_non_system_message():
    with pytest.raises(LLMBadRequestError):
        to_gemini_request(ChatRequest(messages=[ChatMessage("system", "only a system prompt")]))


def test_gemini_daily_quota_is_not_retryable():
    harness = GeminiHarness("gemini")
    with pytest.raises(LLMQuotaExceededError):
        _call(harness, harness.http_error(429, "Quota exceeded for metric: requests per day, limit: 0"))


def test_gemini_blocked_output_is_a_content_filter_error():
    harness = GeminiHarness("gemini")
    with pytest.raises(LLMContentFilterError):
        _call(harness, harness.response("", 5, 0, 5, "SAFETY"))
    # An empty but unblocked reply is returned as is (callers decide, e.g. NOT_FOUND handling).
    response, _ = _call(harness, harness.response("", 5, 0, 5, "STOP"))
    assert response.text == ""


@pytest.mark.parametrize(("status", "expected"), [(429, LLMRateLimitError), (503, LLMTransientError)])
def test_gemini_maps_the_real_sdk_exceptions(status, expected):
    errors = pytest.importorskip("google.genai.errors")
    error_class = errors.ClientError if status < 500 else errors.ServerError
    real = error_class(status, {"error": {"code": status, "message": "boom", "status": "X"}})
    harness = GeminiHarness("gemini")
    with pytest.raises(expected):
        _call(harness, real)


# --- OpenAI-compatible specifics --------------------------------------------------------------


def test_openai_request_is_identical_to_the_thesis_client():
    messages = [{"role": "system", "content": "Return JSON only."}, {"role": "user", "content": " Question: x "}]
    # Recorded from the thesis client (`llm/client.py`, since deleted) before it was deleted.
    thesis_call = {"model": "test-model", "messages": messages, "temperature": 0}

    harness = OpenAIHarness("openai_compatible")
    request = ChatRequest(messages=[ChatMessage(m["role"], m["content"]) for m in messages])
    _, call = _call(harness, harness.response("x", 1, 1, 2, "STOP"), request)
    assert call == thesis_call


def test_openai_maps_new_parameters_only_when_set():
    request = ChatRequest(messages=[ChatMessage("user", "Hi")], max_output_tokens=64, seed=7, timeout_s=2.5)
    kwargs = OpenAICompatibleChatModel(model="m", client=object()).to_request_kwargs(request)
    assert kwargs == {
        "model": "m",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.0,
        "max_tokens": 64,
        "seed": 7,
        "timeout": 2.5,
    }
    newer = OpenAICompatibleChatModel(model="m", client=object(), max_tokens_param="max_completion_tokens")
    assert newer.to_request_kwargs(request)["max_completion_tokens"] == 64
    with pytest.raises(ValueError):
        OpenAICompatibleChatModel(model="m", max_tokens_param="tokens")


def test_openai_reads_list_content_and_detects_filtered_output():
    harness = OpenAIHarness("openai_compatible")
    parts = harness.response("", 1, 1, 2, "STOP")
    parts.choices[0].message.content = [{"type": "text", "text": "BR"}, SimpleNamespace(text="CA1")]
    response, _ = _call(harness, parts)
    assert response.text == "BRCA1"
    with pytest.raises(LLMContentFilterError):
        _call(harness, harness.response("", 1, 0, 1, "content_filter"))


def test_openai_insufficient_quota_is_not_retryable():
    harness = OpenAIHarness("openai_compatible")
    with pytest.raises(LLMQuotaExceededError):
        _call(harness, harness.http_error(429, "Error code: 429 - {'error': {'code': 'insufficient_quota'}}"))


def test_openai_local_server_needs_no_key(monkeypatch):
    pytest.importorskip("openai")
    model = OpenAICompatibleChatModel(model="m", provider="ollama", api_key_env=None, base_url="http://localhost:1/v1")
    client = model._get_client()  # builds the SDK client; no request is sent
    assert str(client.base_url).startswith("http://localhost:1/v1")


@pytest.mark.parametrize(
    ("make_error", "expected"),
    [
        (
            lambda openai, httpx, req: openai.RateLimitError(
                "slow down", response=httpx.Response(429, request=req), body=None
            ),
            LLMRateLimitError,
        ),
        (
            lambda openai, httpx, req: openai.AuthenticationError(
                "bad key", response=httpx.Response(401, request=req), body=None
            ),
            LLMAuthError,
        ),
        (
            lambda openai, httpx, req: openai.InternalServerError(
                "oops", response=httpx.Response(500, request=req), body=None
            ),
            LLMTransientError,
        ),
        (lambda openai, httpx, req: openai.APITimeoutError(request=req), LLMTransientError),
        (lambda openai, httpx, req: openai.APIConnectionError(request=req), LLMTransientError),
    ],
)
def test_openai_maps_the_real_sdk_exceptions(make_error, expected):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    error = make_error(openai, httpx, httpx.Request("POST", "http://test.invalid/v1/chat/completions"))
    harness = OpenAIHarness("openai_compatible")
    with pytest.raises(expected):
        _call(harness, error)


# --- Anthropic specifics ----------------------------------------------------------------------


def test_anthropic_request_mapping():
    request = ChatRequest(
        messages=[ChatMessage("system", " Be strict. "), ChatMessage("user", "Q"), ChatMessage("assistant", "A")],
        json_mode=True,
        seed=3,  # not supported by the API: not sent
        timeout_s=9,
    )
    kwargs = AnthropicChatModel(model="m", client=object(), default_max_tokens=512).to_request_kwargs(request)
    assert kwargs == {
        "model": "m",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}],
        "temperature": 0.0,
        "system": f"Be strict.\n\n{JSON_INSTRUCTION}",
        "timeout": 9,
    }
    limited = ChatRequest(messages=[ChatMessage("user", "Q")], max_output_tokens=64)
    plain = AnthropicChatModel(model="m", client=object()).to_request_kwargs(limited)
    assert plain["max_tokens"] == 64 and "system" not in plain


def test_anthropic_needs_a_non_system_message():
    with pytest.raises(LLMBadRequestError):
        AnthropicChatModel(model="m", client=object()).to_request_kwargs(
            ChatRequest(messages=[ChatMessage("system", "only a system prompt")])
        )


def test_anthropic_joins_text_blocks_and_detects_refusals():
    harness = AnthropicHarness("anthropic", reports_total=False)
    reply = harness.response("", 1, 1, None, "STOP")
    reply.content = [
        SimpleNamespace(type="thinking", thinking="hidden"),
        SimpleNamespace(type="text", text="BR"),
        SimpleNamespace(type="text", text="CA1"),
    ]
    response, _ = _call(harness, reply)
    assert response.text == "BRCA1" and response.finish_reason == "end_turn"
    with pytest.raises(LLMContentFilterError):
        _call(harness, harness.response("", 1, 0, None, "refusal"))


def test_anthropic_overloaded_is_transient():
    harness = AnthropicHarness("anthropic", reports_total=False)
    with pytest.raises(LLMTransientError):
        _call(harness, harness.http_error(529, "overloaded_error"))
