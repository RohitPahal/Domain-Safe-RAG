"""Answer generation: one call per window, an explicit outcome, one retry policy.

Which windows are asked (top-1, then top-3 after an empty or NOT_FOUND reply) is decided by the
graph's edges; see the path and snapshot tests in test_pipeline_behaviour.py.
"""

from __future__ import annotations

import pytest

from rag_thesis.llm.errors import LLMTransientError
from rag_thesis.llm.testing import FakeChatModel
from rag_thesis.llm.types import Usage
from rag_thesis.llm.wrappers import RetryingChatModel
from rag_thesis.pipeline.generation import AnswerGenerator, add_usage, postprocess_answer
from rag_thesis.prompts.answer import build_answer_messages
from rag_thesis.protection import ProtectedPathError

CHUNKS = {
    "c1": {"text": "BRCA1 is a tumour suppressor."},
    "c2": {"text": "TP53 is mutated in many cancers."},
    "c3": {"text": "Other text."},
}
TOP_1 = [{"chunk_id": "c1"}]
TOP_3 = [{"chunk_id": cid} for cid in ("c1", "c2", "c3")]


def _answer(model, window=TOP_1, chunks=CHUNKS):
    return AnswerGenerator(model).answer(qid="q1", question="Which gene?", window=window, chunks_by_id=chunks)


def test_an_answer_cites_every_chunk_of_its_window():
    model = FakeChatModel(["TP53"], usage=Usage(10, 2, 12))
    outcome = _answer(model, window=TOP_3)
    assert (outcome.status, outcome.answer_text, outcome.citations) == ("answered", "TP53", ["c1", "c2", "c3"])
    assert outcome.usage == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    (request,) = model.requests
    contexts = ["[c1] BRCA1 is a tumour suppressor.", "[c2] TP53 is mutated in many cancers.", "[c3] Other text."]
    assert list(request.messages) == build_answer_messages("Which gene?", contexts)
    assert request.json_mode is False


@pytest.mark.parametrize("reply", ["NOT_FOUND", "  ", "not_found."])
def test_no_answer_is_not_found_and_its_tokens_still_count(reply):
    outcome = _answer(FakeChatModel([reply], usage=Usage(5, 1, 6)))
    assert (outcome.status, outcome.answer_text, outcome.citations, outcome.error) == ("not_found", None, [], None)
    assert outcome.usage == {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}


def test_a_provider_failure_is_recorded_with_its_reason_and_not_retried_here(caplog):
    caplog.set_level("WARNING")
    model = FakeChatModel([LLMTransientError("503 overloaded"), "never asked"], provider="p", model="m")
    outcome = _answer(model)
    assert (outcome.status, outcome.usage) == ("failed", {})
    assert outcome.error == "LLMTransientError: 503 overloaded"
    assert len(model.requests) == 1  # no retry loop here
    assert "qid=q1 model=p/m" in caplog.text


def test_retries_come_from_the_model_wrapper():
    inner = FakeChatModel([LLMTransientError("503"), "BRCA1"])
    outcome = _answer(RetryingChatModel(inner, sleep=lambda s: None))
    assert outcome.status == "answered" and len(inner.requests) == 2


def test_missing_chunk_text_is_a_recorded_failure_without_a_call():
    model = FakeChatModel([])
    outcome = _answer(model, chunks={})
    assert outcome.status == "failed" and "chunk store" in outcome.error
    assert model.requests == []


def test_chunks_missing_from_the_store_are_left_out_of_the_prompt_but_still_cited():
    model = FakeChatModel(["TP53"])
    outcome = _answer(model, window=TOP_3, chunks={"c1": CHUNKS["c1"], "c3": CHUNKS["c3"]})
    assert outcome.citations == ["c1", "c2", "c3"]  # unchanged thesis behaviour
    assert "[c2]" not in model.requests[0].messages[-1].content


@pytest.mark.parametrize(
    "error", [TypeError("a code bug"), ValueError("a code bug"), ProtectedPathError("protected cache")]
)
def test_anything_but_a_provider_failure_stops_the_run(error):
    # Only provider failures become a failed status; anything else stops the run.
    with pytest.raises(type(error)):
        _answer(FakeChatModel([error]))


def test_add_usage_sums_returned_calls_only():
    first = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    second = {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 9}
    assert add_usage(first, second) == {"prompt_tokens": 17, "completion_tokens": 3, "total_tokens": 21}
    assert add_usage(first, {}) == first  # a failed second call spent nothing
    assert add_usage({}, {}) == {}  # no call returned: no usage at all


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("BRCA1", "BRCA1"),
        ("Answer: BRCA1.", "BRCA1"),
        ("BRCA1\nbecause the chunk says so", "BRCA1"),
        ('{"exact_answer": "TP53"}', "TP53"),
        ("```json\n{\n}\n```", None),
        ("not_found", None),
        ("", None),
    ],
)
def test_postprocess_answer(reply, expected):
    assert postprocess_answer(reply) == expected
