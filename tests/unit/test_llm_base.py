"""ChatModel base: complete_json is implemented once for every provider."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from rag_thesis.llm.base import JSONOutputError, extract_json_object
from rag_thesis.llm.testing import FakeChatModel
from rag_thesis.llm.types import ChatMessage, ChatRequest, ChatResponse, Usage

REQUEST = ChatRequest(messages=[ChatMessage("user", "Rewrite this.")])


class Rewrite(BaseModel):
    rewrite: str


@pytest.mark.parametrize(
    "text",
    [
        '{"rewrite": "aspirin"}',
        '  {"rewrite": "aspirin"}\n',
        '```json\n{"rewrite": "aspirin"}\n```',
        '```\n{"rewrite": "aspirin"}\n```',
        'Here you go: {"rewrite": "aspirin"} Hope it helps.',
    ],
)
def test_extract_json_object_tolerates_fences_and_surrounding_text(text):
    assert extract_json_object(text) == {"rewrite": "aspirin"}


@pytest.mark.parametrize("text", ["", "   ", "no json here", '{"rewrite": ', "[1, 2]", '"just a string"'])
def test_extract_json_object_rejects_non_objects(text):
    with pytest.raises(JSONOutputError):
        extract_json_object(text)


def test_complete_json_forces_json_mode_and_validates():
    model = FakeChatModel(['{"rewrite": "aspirin synonym"}'], usage=Usage(prompt_tokens=3, completion_tokens=2))
    parsed, response = model.complete_json(REQUEST, Rewrite)

    assert parsed == Rewrite(rewrite="aspirin synonym")
    assert response.usage.total_tokens == 5
    assert model.requests[0].json_mode is True
    assert REQUEST.json_mode is False  # the caller's request is not changed


def test_complete_json_raises_on_schema_mismatch_and_keeps_the_text():
    model = FakeChatModel(['{"other": 1}'])
    with pytest.raises(JSONOutputError, match="Rewrite") as info:
        model.complete_json(REQUEST, Rewrite)
    assert info.value.text == '{"other": 1}'


def test_fake_chat_model_scripts_replies_errors_and_fails_when_exhausted():
    full = ChatResponse(text="full", provider="p", model="m")
    model = FakeChatModel(["first", full, RuntimeError("boom")])

    assert model.complete(REQUEST).text == "first"
    assert model.complete(REQUEST) is full
    with pytest.raises(RuntimeError, match="boom"):
        model.complete(REQUEST)
    with pytest.raises(AssertionError, match="no scripted reply"):
        model.complete(REQUEST)
    assert len(model.requests) == 4
