from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_thesis.cli.run import app as run_rag_app
from rag_thesis.llm.base import ChatModel
from rag_thesis.llm.errors import LLMAuthError, LLMBadRequestError, LLMRateLimitError, LLMTransientError
from rag_thesis.llm.types import ChatResponse, Usage
from rag_thesis.pipeline.workflow import build_rag_run_context, run_rag_workflow, run_rag_workflow_with_context
from rag_thesis.schemas import RunLogLine

RUNNER = CliRunner()


class FakeRetriever:
    def __init__(self, results_by_query: dict[str, list[dict]]) -> None:
        self.results_by_query = results_by_query

    def retrieve(self, query: str, k: int) -> list[dict]:
        return [dict(item) for item in self.results_by_query.get(query, [])[:k]]


def _reply(text, prompt_tokens=0, completion_tokens=0, total_tokens=0, raw_response=None) -> ChatResponse:
    usage = Usage(prompt_tokens, completion_tokens, total_tokens)
    return ChatResponse(text=text, provider="groq", model="llama-3.1-8b-instant", usage=usage)


class FakeLLMClient(ChatModel):
    """One scripted model for every role (routes on the prompt); `calls` records each request."""

    provider = "groq"
    model = "llama-3.1-8b-instant"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, request):
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        return self.chat(self.model, messages, request.temperature)

    def chat(self, model, messages, temperature=0):
        self.calls.append({"model": model, "messages": messages, "temperature": temperature})
        system_prompt = messages[0]["content"]
        if 'Return JSON only in the form {"rewrite": "..."}.' in system_prompt:
            return _reply(
                text='{"rewrite":"What is the role of BRCA1 in breast cancer?"}',
                prompt_tokens=10,
                completion_tokens=4,
                total_tokens=14,
                raw_response={"mocked": True},
            )
        if "Generate exactly four retrieval rewrites" in system_prompt:
            return _reply(
                text=(
                    '{"bm25_lexical":"BRCA1 breast cancer role",'
                    '"dense_semantic":"role of BRCA1 in breast cancer",'
                    '"abbreviation_expanded":"BRCA1 breast cancer function",'
                    '"minimal_entity_locked":"BRCA1 role in breast cancer"}'
                ),
                prompt_tokens=12,
                completion_tokens=6,
                total_tokens=18,
                raw_response={"mocked": True},
            )
        return _reply(
            text="BRCA1 is involved in breast cancer biology according to the retrieved chunks.",
            prompt_tokens=15,
            completion_tokens=8,
            total_tokens=23,
            raw_response={"mocked": True},
        )


class SequentialLLMClient(FakeLLMClient):
    def __init__(self, responses: list[object]) -> None:
        super().__init__()
        self._responses = list(responses)

    def chat(self, model, messages, temperature=0):
        self.calls.append({"model": model, "messages": messages, "temperature": temperature})
        next_response = self._responses[len(self.calls) - 1]
        if isinstance(next_response, Exception):
            raise next_response
        return next_response


class StaticLLMClient(FakeLLMClient):
    def __init__(self, response: ChatResponse) -> None:
        super().__init__()
        self._response = response

    def chat(self, model, messages, temperature=0):
        self.calls.append({"model": model, "messages": messages, "temperature": temperature})
        return self._response


def _roles(model) -> dict | None:
    """The same model for every query-time role (None: no models, as without --generate)."""
    return None if model is None else {"rewrite": model, "repair": model, "answer": model}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _config() -> dict:
    return {
        "rrf_k": 60,
        "llm_provider": "groq",
        "llm_api_key_env": "GROQ_API_KEY",
        "llm_api_base": "https://api.groq.com/openai/v1",
        "llm_rewrite_model": "llama-3.1-8b-instant",
        "llm_answer_model": "llama-3.1-8b-instant",
        "openai_rewrite_model": "gpt-4.1-mini",
        "openai_answer_model": "gpt-4.1-mini",
    }


def _write_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "rrf_k: 60",
                "llm_provider: groq",
                "llm_api_key_env: GROQ_API_KEY",
                "llm_api_base: https://api.groq.com/openai/v1",
                "llm_rewrite_model: llama-3.1-8b-instant",
                "llm_answer_model: llama-3.1-8b-instant",
                "openai_rewrite_model: gpt-4.1-mini",
                "openai_answer_model: gpt-4.1-mini",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _questions() -> list[dict]:
    return [
        {
            "qid": "q-1",
            "question": "What is the role of BRCA1 in breast cancer?",
            "doc_id": "doc-1",
            "answers": [{"text": "BRCA1", "answer_start": 20}],
        }
    ]


def _questions_two() -> list[dict]:
    return [
        {
            "qid": "q-1",
            "question": "What is the role of BRCA1 in breast cancer?",
            "doc_id": "doc-1",
            "answers": [{"text": "BRCA1", "answer_start": 20}],
        },
        {
            "qid": "q-2",
            "question": "What is the role of BRCA1 in breast cancer?",
            "doc_id": "doc-1",
            "answers": [{"text": "BRCA1", "answer_start": 20}],
        },
    ]


def _chunks() -> list[dict]:
    return [
        {
            "chunk_id": "doc-1::0",
            "doc_id": "doc-1",
            "chunk_index": 0,
            "text": "BRCA1 is involved in DNA repair and breast cancer biology.",
            "char_start": 0,
            "char_end": 60,
            "metadata": {"source": "test"},
        },
        {
            "chunk_id": "doc-2::0",
            "doc_id": "doc-2",
            "chunk_index": 0,
            "text": "Breast cancer pathways often involve BRCA1 and related genes.",
            "char_start": 0,
            "char_end": 65,
            "metadata": {"source": "test"},
        },
        {
            "chunk_id": "doc-3::0",
            "doc_id": "doc-3",
            "chunk_index": 0,
            "text": "General oncology background information.",
            "char_start": 0,
            "char_end": 39,
            "metadata": {"source": "test"},
        },
        {
            "chunk_id": "doc-4::0",
            "doc_id": "doc-4",
            "chunk_index": 0,
            "text": "Additional BRCA1 evidence relevant to tumor suppression.",
            "char_start": 0,
            "char_end": 56,
            "metadata": {"source": "test"},
        },
    ]


def _bm25_backend() -> FakeRetriever:
    return FakeRetriever(
        {
            "What is the role of BRCA1 in breast cancer?": [
                {"chunk_id": "doc-1::0", "doc_id": "doc-1", "score": 3.0, "rank": 1},
                {"chunk_id": "doc-2::0", "doc_id": "doc-2", "score": 2.0, "rank": 2},
                {"chunk_id": "doc-3::0", "doc_id": "doc-3", "score": 1.0, "rank": 3},
            ],
            "BRCA1 breast cancer role": [
                {"chunk_id": "doc-1::0", "doc_id": "doc-1", "score": 3.0, "rank": 1},
                {"chunk_id": "doc-2::0", "doc_id": "doc-2", "score": 2.0, "rank": 2},
            ],
            "role of BRCA1 in breast cancer": [
                {"chunk_id": "doc-2::0", "doc_id": "doc-2", "score": 3.0, "rank": 1},
                {"chunk_id": "doc-4::0", "doc_id": "doc-4", "score": 2.0, "rank": 2},
            ],
            "BRCA1 breast cancer function": [
                {"chunk_id": "doc-1::0", "doc_id": "doc-1", "score": 2.5, "rank": 1},
                {"chunk_id": "doc-4::0", "doc_id": "doc-4", "score": 2.0, "rank": 2},
            ],
            "BRCA1 role in breast cancer": [
                {"chunk_id": "doc-4::0", "doc_id": "doc-4", "score": 3.0, "rank": 1},
                {"chunk_id": "doc-2::0", "doc_id": "doc-2", "score": 2.0, "rank": 2},
            ],
        }
    )


def _dense_backend() -> FakeRetriever:
    return FakeRetriever(
        {
            "What is the role of BRCA1 in breast cancer?": [
                {"chunk_id": "doc-2::0", "doc_id": "doc-2", "score": 0.95, "rank": 1},
                {"chunk_id": "doc-1::0", "doc_id": "doc-1", "score": 0.90, "rank": 2},
                {"chunk_id": "doc-4::0", "doc_id": "doc-4", "score": 0.80, "rank": 3},
            ],
            "BRCA1 breast cancer role": [
                {"chunk_id": "doc-2::0", "doc_id": "doc-2", "score": 0.95, "rank": 1},
                {"chunk_id": "doc-1::0", "doc_id": "doc-1", "score": 0.91, "rank": 2},
            ],
            "role of BRCA1 in breast cancer": [
                {"chunk_id": "doc-1::0", "doc_id": "doc-1", "score": 0.96, "rank": 1},
                {"chunk_id": "doc-4::0", "doc_id": "doc-4", "score": 0.92, "rank": 2},
            ],
            "BRCA1 breast cancer function": [
                {"chunk_id": "doc-4::0", "doc_id": "doc-4", "score": 0.95, "rank": 1},
                {"chunk_id": "doc-1::0", "doc_id": "doc-1", "score": 0.90, "rank": 2},
            ],
            "BRCA1 role in breast cancer": [
                {"chunk_id": "doc-1::0", "doc_id": "doc-1", "score": 0.97, "rank": 1},
                {"chunk_id": "doc-2::0", "doc_id": "doc-2", "score": 0.93, "rank": 2},
            ],
        }
    )


def test_retrieval_only_mode_works_without_api_key_and_writes_valid_log(tmp_path) -> None:
    questions_path = tmp_path / "questions.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    log_path = tmp_path / "run_log.jsonl"
    _write_jsonl(questions_path, _questions())
    _write_jsonl(chunks_path, _chunks())

    state = run_rag_workflow(
        qid="q-1",
        retriever_mode="bm25",
        rewrite_mode="none",
        guard_mode="off",
        k=2,
        questions_path=questions_path,
        chunks_path=chunks_path,
        config=_config(),
        bm25_backend=_bm25_backend(),
        generation_enabled=False,
        log_path=log_path,
    )

    assert state["answer_text"] is None
    assert state["citations"] == []
    assert len(state["retrieval_results"]) == 2
    RunLogLine.model_validate(json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[0]))


def test_workflow_does_not_crash_without_api_key_for_multi_guard_on(tmp_path, monkeypatch) -> None:
    questions_path = tmp_path / "questions.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    log_path = tmp_path / "run_log.jsonl"
    _write_jsonl(questions_path, _questions())
    _write_jsonl(chunks_path, _chunks())
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    import rag_thesis.pipeline.nodes as rag_nodes_module

    monkeypatch.setattr(
        rag_nodes_module,
        "rewrite_query",
        lambda **kwargs: (_ for _ in ()).throw(LLMAuthError("Set the environment variable GROQ_API_KEY to call groq.")),
    )

    state = run_rag_workflow(
        qid="q-1",
        retriever_mode="bm25",
        rewrite_mode="multi",
        guard_mode="on",
        k=3,
        questions_path=questions_path,
        chunks_path=chunks_path,
        config=_config(),
        bm25_backend=_bm25_backend(),
        log_path=log_path,
    )

    assert state["rewrites"] == ["What is the role of BRCA1 in breast cancer?"]
    assert state["rewrite_status"] == "failed" and "GROQ_API_KEY" in state["rewrite_error"]
    assert state["answer_text"] is None
    assert state["citations"] == []
    assert state["guard_report"]["guard_mode"] == "on"
    assert state["retrieval_results"]
    RunLogLine.model_validate(json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[0]))


def test_all_combinations_produce_valid_run_logs_on_toy_data(tmp_path) -> None:
    questions_path = tmp_path / "questions.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    log_path = tmp_path / "run_log.jsonl"
    _write_jsonl(questions_path, _questions())
    _write_jsonl(chunks_path, _chunks())

    combinations = [
        (retriever, rewrite, guard)
        for retriever in ("bm25", "dense", "hybrid")
        for rewrite in ("none", "single", "multi")
        for guard in ("off", "on")
    ]
    for retriever, rewrite, guard in combinations:
        state = run_rag_workflow(
            qid="q-1",
            retriever_mode=retriever,
            rewrite_mode=rewrite,
            guard_mode=guard,
            k=3,
            questions_path=questions_path,
            chunks_path=chunks_path,
            config=_config(),
            bm25_backend=_bm25_backend(),
            dense_backend=_dense_backend(),
            chat_models=_roles(FakeLLMClient()),
            generation_enabled=False,
            log_path=log_path,
        )
        assert state["retrieval_results"]
        assert state["answer_text"] is None
        assert state["citations"] == []
        assert state["config_snapshot"]["retriever_mode"] == retriever
        assert state["config_snapshot"]["rewrite_mode"] == rewrite
        assert state["config_snapshot"]["guard_mode"] == guard

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(combinations)
    for line in lines:
        RunLogLine.model_validate(json.loads(line))


def test_guard_off_preserves_unsafe_rewrite_but_guard_on_falls_back_to_original(tmp_path, monkeypatch) -> None:
    questions_path = tmp_path / "questions.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    _write_jsonl(questions_path, _questions())
    _write_jsonl(chunks_path, _chunks())
    monkeypatch.setenv("GROQ_API_KEY", "present-but-should-not-be-used")

    import rag_thesis.pipeline.nodes as rag_nodes_module

    monkeypatch.setattr(
        rag_nodes_module,
        "rewrite_query",
        lambda **kwargs: ["What is the role of BRCA1 stage 2 in breast cancer?"],
    )

    guard_off_state = run_rag_workflow(
        qid="q-1",
        retriever_mode="bm25",
        rewrite_mode="single",
        guard_mode="off",
        k=3,
        questions_path=questions_path,
        chunks_path=chunks_path,
        config=_config(),
        bm25_backend=_bm25_backend(),
        generation_enabled=False,
        log_path=tmp_path / "guard_off.jsonl",
    )
    guard_on_state = run_rag_workflow(
        qid="q-1",
        retriever_mode="bm25",
        rewrite_mode="single",
        guard_mode="on",
        k=3,
        questions_path=questions_path,
        chunks_path=chunks_path,
        config=_config(),
        bm25_backend=_bm25_backend(),
        generation_enabled=False,
        log_path=tmp_path / "guard_on.jsonl",
    )

    assert guard_off_state["rewrites"] == ["What is the role of BRCA1 stage 2 in breast cancer?"]
    assert guard_off_state["guard_report"]["guard_mode"] == "off"
    assert guard_on_state["rewrites"] == ["What is the role of BRCA1 in breast cancer?"]
    assert guard_on_state["guard_report"]["guard_mode"] == "on"
    assert guard_on_state["guard_report"]["reports"][0]["final_action"] == "fallback_original"


def test_multi_rewrite_retrieval_merges_results_deterministically(tmp_path) -> None:
    questions_path = tmp_path / "questions.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    _write_jsonl(questions_path, _questions())
    _write_jsonl(chunks_path, _chunks())

    first = run_rag_workflow(
        qid="q-1",
        retriever_mode="bm25",
        rewrite_mode="multi",
        guard_mode="off",
        k=4,
        questions_path=questions_path,
        chunks_path=chunks_path,
        config=_config(),
        bm25_backend=_bm25_backend(),
        chat_models=_roles(FakeLLMClient()),
        generation_enabled=False,
        log_path=tmp_path / "first.jsonl",
    )
    second = run_rag_workflow(
        qid="q-1",
        retriever_mode="bm25",
        rewrite_mode="multi",
        guard_mode="off",
        k=4,
        questions_path=questions_path,
        chunks_path=chunks_path,
        config=_config(),
        bm25_backend=_bm25_backend(),
        chat_models=_roles(FakeLLMClient()),
        generation_enabled=False,
        log_path=tmp_path / "second.jsonl",
    )

    assert [result["chunk_id"] for result in first["retrieval_results"]] == ["doc-2::0", "doc-4::0", "doc-1::0"]
    assert [result["chunk_id"] for result in second["retrieval_results"]] == ["doc-2::0", "doc-4::0", "doc-1::0"]


def test_generation_citations_are_subset_of_retrieved_chunk_ids(tmp_path) -> None:
    questions_path = tmp_path / "questions.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    log_path = tmp_path / "run_log.jsonl"
    _write_jsonl(questions_path, _questions())
    _write_jsonl(chunks_path, _chunks())

    state = run_rag_workflow(
        qid="q-1",
        retriever_mode="hybrid",
        rewrite_mode="none",
        guard_mode="on",
        k=3,
        questions_path=questions_path,
        chunks_path=chunks_path,
        config=_config(),
        bm25_backend=_bm25_backend(),
        dense_backend=_dense_backend(),
        chat_models=_roles(FakeLLMClient()),
        generation_enabled=True,
        log_path=log_path,
    )

    retrieved_chunk_ids = {result["chunk_id"] for result in state["retrieval_results"]}
    assert state["answer_text"] is not None
    assert set(state["citations"]).issubset(retrieved_chunk_ids)
    assert state["usage"]["total_tokens"] == 23
    assert set(json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[0])["citations"]).issubset(
        retrieved_chunk_ids
    )


def test_generation_makes_one_call_per_window_retries_belong_to_the_model(tmp_path) -> None:
    # The workflow's own retry loop is gone: retries happen once, in RetryingChatModel
    # (see tests/unit/test_generation.py). A failed call is recorded at once.
    client = SequentialLLMClient([LLMRateLimitError("429 Too Many Requests from provider"), _ANSWER])
    state = _run_with_status(tmp_path, client=client)
    assert state["generation_status"] == "failed"
    assert "429 Too Many Requests" in state["generation_error"]
    assert len(client.calls) == 1


def test_generation_does_not_retry_non_transient_failures(tmp_path, monkeypatch, caplog) -> None:
    questions_path = tmp_path / "questions.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    log_path = tmp_path / "run_log.jsonl"
    _write_jsonl(questions_path, _questions())
    _write_jsonl(chunks_path, _chunks())

    import rag_thesis.pipeline.workflow as rag_workflow_module

    sleep_calls: list[int] = []
    monkeypatch.setattr(rag_workflow_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    caplog.set_level("WARNING")

    client = SequentialLLMClient([LLMBadRequestError("invalid request body for answer generation")])

    state = run_rag_workflow(
        qid="q-1",
        retriever_mode="hybrid",
        rewrite_mode="none",
        guard_mode="on",
        k=3,
        questions_path=questions_path,
        chunks_path=chunks_path,
        config=_config(),
        bm25_backend=_bm25_backend(),
        dense_backend=_dense_backend(),
        chat_models=_roles(client),
        generation_enabled=True,
        log_path=log_path,
    )

    assert state["answer_text"] is None
    assert state["citations"] == []
    assert state["usage"] == {}
    assert sleep_calls == []
    assert len(client.calls) == 1
    assert "invalid request body for answer generation" in caplog.text


def test_generation_postprocesses_first_line_and_answer_label(tmp_path) -> None:
    questions_path = tmp_path / "questions.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    log_path = tmp_path / "run_log.jsonl"
    _write_jsonl(questions_path, _questions())
    _write_jsonl(chunks_path, _chunks())

    client = StaticLLMClient(
        _reply(
            text="Answer: Alpha-synuclein\nThis is supported by the retrieved chunks.",
            prompt_tokens=15,
            completion_tokens=8,
            total_tokens=23,
            raw_response={"mocked": True},
        )
    )

    state = run_rag_workflow(
        qid="q-1",
        retriever_mode="hybrid",
        rewrite_mode="none",
        guard_mode="on",
        k=3,
        questions_path=questions_path,
        chunks_path=chunks_path,
        config=_config(),
        bm25_backend=_bm25_backend(),
        dense_backend=_dense_backend(),
        chat_models=_roles(client),
        generation_enabled=True,
        log_path=log_path,
    )

    assert state["answer_text"] == "Alpha-synuclein"
    assert state["citations"] == [state["retrieval_results"][0]["chunk_id"]]
    assert state["usage"]["total_tokens"] == 23
    assert len(client.calls) == 1
    assert (
        "Return exactly the entity, type, binary choice, number, or mutation/defect asked by the question."
        in client.calls[0]["messages"][0]["content"]
    )
    assert (
        "If the question asks for a mutation or defect, return the mutation or defect, not merely the associated gene "
        "name." in client.calls[0]["messages"][0]["content"]
    )


def test_generation_maps_not_found_to_unanswered_state(tmp_path) -> None:
    questions_path = tmp_path / "questions.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    log_path = tmp_path / "run_log.jsonl"
    _write_jsonl(questions_path, _questions())
    _write_jsonl(chunks_path, _chunks())

    client = SequentialLLMClient(
        [
            _reply(
                text="NOT_FOUND",
                prompt_tokens=15,
                completion_tokens=4,
                total_tokens=19,
                raw_response={"mocked": True},
            ),
            _reply(
                text="NOT_FOUND",
                prompt_tokens=18,
                completion_tokens=5,
                total_tokens=23,
                raw_response={"mocked": True},
            ),
        ]
    )

    state = run_rag_workflow(
        qid="q-1",
        retriever_mode="hybrid",
        rewrite_mode="none",
        guard_mode="on",
        k=3,
        questions_path=questions_path,
        chunks_path=chunks_path,
        config=_config(),
        bm25_backend=_bm25_backend(),
        dense_backend=_dense_backend(),
        chat_models=_roles(client),
        generation_enabled=True,
        log_path=log_path,
    )

    assert state["answer_text"] is None
    assert state["citations"] == []
    # Both NOT_FOUND calls were made, so their tokens count (15+18, 4+5, 19+23). Earlier
    # this was {} and the spent tokens were missing from cost reports.
    assert state["usage"] == {"prompt_tokens": 33, "completion_tokens": 9, "total_tokens": 42}
    assert state["generation_status"] == "not_found"
    assert len(client.calls) == 2


def test_generation_falls_back_from_top1_not_found_to_top3(tmp_path) -> None:
    questions_path = tmp_path / "questions.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    log_path = tmp_path / "run_log.jsonl"
    _write_jsonl(questions_path, _questions())
    _write_jsonl(chunks_path, _chunks())

    client = SequentialLLMClient(
        [
            _reply(
                text="NOT_FOUND",
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
                raw_response={"mocked": True},
            ),
            _reply(
                text="BRCA1",
                prompt_tokens=20,
                completion_tokens=3,
                total_tokens=23,
                raw_response={"mocked": True},
            ),
        ]
    )

    state = run_rag_workflow(
        qid="q-1",
        retriever_mode="hybrid",
        rewrite_mode="none",
        guard_mode="on",
        k=3,
        questions_path=questions_path,
        chunks_path=chunks_path,
        config=_config(),
        bm25_backend=_bm25_backend(),
        dense_backend=_dense_backend(),
        chat_models=_roles(client),
        generation_enabled=True,
        log_path=log_path,
    )

    assert state["answer_text"] == "BRCA1"
    assert state["citations"] == [result["chunk_id"] for result in state["retrieval_results"][:3]]
    assert state["usage"] == {
        "prompt_tokens": 30,
        "completion_tokens": 5,
        "total_tokens": 35,
    }
    assert len(client.calls) == 2


def test_preloaded_context_reuses_questions_chunks_backends_and_llm_once(tmp_path, monkeypatch) -> None:
    questions_path = tmp_path / "questions.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    _write_jsonl(questions_path, _questions_two())
    _write_jsonl(chunks_path, _chunks())

    import rag_thesis.pipeline.workflow as rag_workflow_module

    original_load_questions = rag_workflow_module.load_questions_by_id
    original_load_chunks = rag_workflow_module.load_chunks_by_id
    load_counts = {
        "questions": 0,
        "chunks": 0,
        "bm25": 0,
        "dense": 0,
        "llm": 0,
    }

    def counted_load_questions(path: Path):
        load_counts["questions"] += 1
        return original_load_questions(path)

    def counted_load_chunks(path: Path):
        load_counts["chunks"] += 1
        return original_load_chunks(path)

    def counted_load_bm25(_path: Path):
        load_counts["bm25"] += 1
        return _bm25_backend()

    def counted_load_dense(_path: Path, query_cache_dir=None):
        load_counts["dense"] += 1
        return _dense_backend()

    def counted_build_chat_model(role, settings, *, allow_llm_calls, call_log=None):
        load_counts["llm"] += 1
        return FakeLLMClient()

    monkeypatch.setattr(rag_workflow_module, "load_questions_by_id", counted_load_questions)
    monkeypatch.setattr(rag_workflow_module, "load_chunks_by_id", counted_load_chunks)
    monkeypatch.setattr(rag_workflow_module, "load_bm25", counted_load_bm25)
    monkeypatch.setattr(rag_workflow_module, "load_chroma_index", counted_load_dense)
    monkeypatch.setattr(rag_workflow_module, "build_chat_model", counted_build_chat_model)

    context = build_rag_run_context(
        questions_path=questions_path,
        chunks_path=chunks_path,
        retriever_mode="hybrid",
        config=_config(),
        bm25_dir=tmp_path / "bm25",
        vector_dir=tmp_path / "vector",
        allow_llm_calls=True,
    )

    assert load_counts == {
        "questions": 1,
        "chunks": 1,
        "bm25": 1,
        "dense": 1,
        "llm": 3,  # rewrite, repair, answer: built once per context, not per question
    }

    first = run_rag_workflow_with_context(
        qid="q-1",
        retriever_mode="hybrid",
        rewrite_mode="none",
        guard_mode="off",
        k=3,
        context=context,
        generation_enabled=False,
        log_path=tmp_path / "first.jsonl",
    )
    second = run_rag_workflow_with_context(
        qid="q-2",
        retriever_mode="hybrid",
        rewrite_mode="none",
        guard_mode="off",
        k=3,
        context=context,
        generation_enabled=False,
        log_path=tmp_path / "second.jsonl",
    )

    assert load_counts == {
        "questions": 1,
        "chunks": 1,
        "bm25": 1,
        "dense": 1,
        "llm": 3,  # rewrite, repair, answer: built once per context, not per question
    }
    assert first["qid"] == "q-1"
    assert second["qid"] == "q-2"
    assert first["retrieval_results"]
    assert second["retrieval_results"]


def test_run_log_writing_is_append_safe_and_cli_smoke(tmp_path, monkeypatch) -> None:
    questions_path = tmp_path / "questions.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    config_path = tmp_path / "config.yaml"
    log_path = tmp_path / "run_log.jsonl"
    _write_jsonl(questions_path, _questions())
    _write_jsonl(chunks_path, _chunks())
    _write_config(config_path)

    import rag_thesis.cli.run as run_rag_script

    monkeypatch.setattr(
        run_rag_script,
        "run_rag_workflow",
        lambda **kwargs: run_rag_workflow(
            **kwargs,
            config=_config(),
            bm25_backend=_bm25_backend(),
            dense_backend=_dense_backend(),
            chat_models=_roles(FakeLLMClient()),
            generation_enabled=False,
        ),
    )

    args = [
        "--qid",
        "q-1",
        "--retriever",
        "hybrid",
        "--rewrite",
        "multi",
        "--guard",
        "on",
        "--k",
        "3",
        "--questions-path",
        str(questions_path),
        "--chunks-path",
        str(chunks_path),
        "--config-path",
        str(config_path),
        "--log-path",
        str(log_path),
    ]
    first = RUNNER.invoke(run_rag_app, args)
    second = RUNNER.invoke(run_rag_app, args)

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert "retrieved_results:" in first.output
    assert len(log_path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_pipeline_runs_on_real_langgraph() -> None:
    """The workflow must use LangGraph itself, never a home-made substitute engine."""
    from langgraph.graph.state import CompiledStateGraph

    from rag_thesis.pipeline.graph import build_rag_graph

    graph = build_rag_graph()
    assert isinstance(graph, CompiledStateGraph)
    assert {"rewrite_node", "guard_node", "retrieve_node", "generate_node", "log_node"} <= set(graph.nodes)


# --- Run-log status fields: every fallback is recorded ---------------------

_ANSWER = _reply(text="BRCA1", prompt_tokens=10, completion_tokens=2, total_tokens=12, raw_response=None)
_NOT_FOUND = _reply(text="NOT_FOUND", prompt_tokens=10, completion_tokens=2, total_tokens=12, raw_response=None)


def _run_with_status(tmp_path, *, client, rewrite_mode="none", generation_enabled=True, bm25=None, chunks=None):
    """Run one question through the real pipeline; the rewrite cache lives in tmp_path (never outputs/)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    questions_path, chunks_path, log_path = tmp_path / "q.jsonl", tmp_path / "c.jsonl", tmp_path / "run_log.jsonl"
    _write_jsonl(questions_path, _questions())
    _write_jsonl(chunks_path, _chunks() if chunks is None else chunks)
    state = run_rag_workflow(
        qid="q-1",
        retriever_mode="bm25",
        rewrite_mode=rewrite_mode,
        guard_mode="off",
        k=3,
        questions_path=questions_path,
        chunks_path=chunks_path,
        config={**_config(), "rewrite_cache_path": str(tmp_path / "rewrites.jsonl")},
        bm25_backend=bm25 or _bm25_backend(),
        chat_models=_roles(client),
        generation_enabled=generation_enabled,
        log_path=log_path,
    )
    logged = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    for field in ("rewrite_status", "rewrite_error", "generation_status", "generation_error"):
        assert logged[field] == state.get(field), field  # the run log carries exactly the state's statuses
    return state


def test_status_when_nothing_fails(tmp_path):
    state = _run_with_status(tmp_path, client=StaticLLMClient(_ANSWER))
    assert state["rewrite_status"] == "not_requested"
    assert state["generation_status"] == "answered"
    assert state["generation_error"] is None


def test_status_when_generation_is_disabled(tmp_path):
    state = _run_with_status(tmp_path, client=None, generation_enabled=False)
    assert state["generation_status"] == "disabled" and state["answer_text"] is None


def test_status_when_retrieval_finds_nothing(tmp_path):
    state = _run_with_status(tmp_path, client=StaticLLMClient(_ANSWER), bm25=FakeRetriever({}))
    assert state["generation_status"] == "no_retrieval"


def test_rewrite_ok_and_rewrite_failure_are_both_recorded(tmp_path, caplog):
    ok = _run_with_status(tmp_path / "ok", client=FakeLLMClient(), rewrite_mode="multi", generation_enabled=False)
    assert ok["rewrite_status"] == "ok" and ok["rewrite_error"] is None

    caplog.set_level("WARNING")
    failing = SequentialLLMClient([LLMTransientError("provider unavailable")])
    failed = _run_with_status(tmp_path / "fail", client=failing, rewrite_mode="multi", generation_enabled=False)
    assert failed["rewrite_status"] == "failed"
    assert "provider unavailable" in failed["rewrite_error"]
    assert failed["rewrites"] == [_questions()[0]["question"]]  # still falls back to the original question
    assert "Rewrite failed for qid=q-1" in caplog.text


def test_generation_failure_is_recorded_not_mistaken_for_not_found(tmp_path):
    state = _run_with_status(tmp_path, client=SequentialLLMClient([LLMBadRequestError("invalid request body")]))
    assert state["answer_text"] is None
    assert state["generation_status"] == "failed"
    assert "invalid request body" in state["generation_error"]
    assert state["usage"] == {}  # the only call failed, so no tokens were spent


def test_failure_after_a_not_found_window_keeps_the_spent_tokens(tmp_path):
    client = SequentialLLMClient([_NOT_FOUND, LLMBadRequestError("invalid request body")])
    state = _run_with_status(tmp_path, client=client)
    assert state["generation_status"] == "failed"
    assert state["usage"] == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}


def test_missing_chunk_text_is_a_recorded_failure(tmp_path):
    state = _run_with_status(tmp_path, client=StaticLLMClient(_ANSWER), chunks=[_chunks()[-1]])
    assert state["generation_status"] == "failed"
    assert "were found in the chunk store" in state["generation_error"]


# --- Run log: usage + per-call llm_calls ------------------------------------------------


def test_run_log_records_usage_and_every_llm_call_per_question(tmp_path):
    from rag_thesis.llm.wrappers import CallLog, UsageTrackingChatModel

    questions_path, chunks_path, log_path = tmp_path / "q.jsonl", tmp_path / "c.jsonl", tmp_path / "log.jsonl"
    _write_jsonl(questions_path, _questions_two())
    _write_jsonl(chunks_path, _chunks())
    call_log = CallLog()
    fake = FakeLLMClient()
    models = {role: UsageTrackingChatModel(fake, role=role, call_log=call_log) for role in ("rewrite", "answer")}
    context = build_rag_run_context(
        questions_path=questions_path,
        chunks_path=chunks_path,
        retriever_mode="bm25",
        config={**_config(), "rewrite_cache_path": str(tmp_path / "rewrites.jsonl")},
        bm25_backend=_bm25_backend(),
        chat_models=models,
        call_log=call_log,
    )
    for qid in ("q-1", "q-2"):
        run_rag_workflow_with_context(
            qid=qid,
            retriever_mode="bm25",
            rewrite_mode="single",
            guard_mode="off",
            k=3,
            context=context,
            log_path=log_path,
        )

    first, second = (json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines())
    for row in (first, second):
        RunLogLine.model_validate(row)
        assert "openai_usage" not in row  # the misleading legacy name is no longer written
        assert [call["role"] for call in row["llm_calls"]] == ["rewrite", "answer"]  # this question's calls only
        assert all(call["provider"] == "groq" and call["status"] == "ok" for call in row["llm_calls"])
    assert first["usage"] == {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23}  # answer tokens
    assert first["llm_calls"][0]["total_tokens"] == 14  # the rewrite call is logged too


def test_legacy_run_logs_with_openai_usage_still_load():
    from rag_thesis.schemas import run_log_usage

    legacy = {
        "qid": "q",
        "query_original": "Q?",
        "query_rewrites": ["Q?"],
        "guard_report": {},
        "retrieval_results": [],
        "answer_text": None,
        "citations": [],
        "latency_ms": 1.0,
        "openai_usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        "config_snapshot": {},
    }
    row = RunLogLine.model_validate(legacy).model_dump()
    assert run_log_usage(row) == {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
    assert run_log_usage({"usage": {"total_tokens": 9}, "openai_usage": None}) == {"total_tokens": 9}
    assert run_log_usage({}) == {}


def test_no_models_are_built_without_permission():
    from rag_thesis.pipeline.workflow import build_pipeline_chat_models

    assert build_pipeline_chat_models(_config(), allow_llm_calls=False) == ({}, None)


def test_graph_is_compiled_once_and_shared_by_every_question(tmp_path, monkeypatch):
    from rag_thesis.pipeline import graph as graph_module

    builds = []
    real_build = graph_module.build_rag_graph

    def counting_build():
        builds.append(1)
        return real_build()

    monkeypatch.setattr(graph_module, "build_rag_graph", counting_build)
    graph_module.compiled_rag_graph.cache_clear()
    try:
        questions_path, chunks_path = tmp_path / "q.jsonl", tmp_path / "c.jsonl"
        _write_jsonl(questions_path, _questions_two())
        _write_jsonl(chunks_path, _chunks())
        context = build_rag_run_context(
            questions_path=questions_path,
            chunks_path=chunks_path,
            retriever_mode="bm25",
            config=_config(),
            bm25_backend=_bm25_backend(),
        )
        for qid in ("q-1", "q-2"):
            state = run_rag_workflow_with_context(
                qid=qid,
                retriever_mode="bm25",
                rewrite_mode="none",
                guard_mode="off",
                k=3,
                context=context,
                log_path=tmp_path / "log.jsonl",
            )
            assert state["qid"] == qid and state["retrieval_results"]
        assert len(builds) == 1
    finally:
        graph_module.compiled_rag_graph.cache_clear()  # don't leave a test-built graph behind


@pytest.mark.parametrize(
    "error",
    [TypeError("a code bug"), KeyError("a code bug"), ValueError("a code bug"), ImportError("no SDK installed")],
)
def test_unexpected_rewrite_errors_stop_the_run(tmp_path, error):
    # Only provider failures and invalid replies become a failed status.
    # Anything else must not look like an LLM failure in the run log.
    client = SequentialLLMClient([error])
    with pytest.raises(type(error)):
        _run_with_status(tmp_path, client=client, rewrite_mode="multi", generation_enabled=False)
    assert not (tmp_path / "run_log.jsonl").exists()


def test_a_corrupt_rewrite_cache_stops_the_run(tmp_path):
    # Otherwise every question would quietly fall back to the original: a "multi" run that is really "none".
    (tmp_path / "rewrites.jsonl").write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON in cache file"):
        _run_with_status(tmp_path, client=FakeLLMClient(), rewrite_mode="multi", generation_enabled=False)
    assert not (tmp_path / "run_log.jsonl").exists()


def test_a_config_without_a_rewrite_role_stops_the_run(tmp_path):
    questions_path, chunks_path = tmp_path / "q.jsonl", tmp_path / "c.jsonl"
    _write_jsonl(questions_path, _questions())
    _write_jsonl(chunks_path, _chunks())
    config = {"rrf_k": 60, "rewrite_cache_path": str(tmp_path / "rewrites.jsonl")}  # no LLM settings at all
    with pytest.raises(ValueError, match="'rewrite' role"):
        run_rag_workflow(
            qid="q-1",
            retriever_mode="bm25",
            rewrite_mode="multi",
            guard_mode="off",
            k=3,
            questions_path=questions_path,
            chunks_path=chunks_path,
            config=config,
            bm25_backend=_bm25_backend(),
            log_path=tmp_path / "run_log.jsonl",
        )
    assert not (tmp_path / "run_log.jsonl").exists()


def test_a_bug_during_generation_stops_the_run(tmp_path):
    client = SequentialLLMClient([TypeError("a code bug")])
    with pytest.raises(TypeError):
        _run_with_status(tmp_path, client=client)
    assert not (tmp_path / "run_log.jsonl").exists()


def test_an_invalid_cached_rewrite_record_stops_the_run(tmp_path):
    # A cached record is not a model reply: three rewrites where four belong means a corrupt cache.
    from rag_thesis.llm.cache import build_rewrite_cache_key
    from rag_thesis.rewrite.rewriter import REWRITE_PROMPT_VERSION

    key = build_rewrite_cache_key(
        qid="q-1",
        query_original=_questions()[0]["question"],
        rewrite_mode="multi",
        rewrite_model="llama-3.1-8b-instant",
        prompt_version=REWRITE_PROMPT_VERSION,
        provider="groq",
    )
    record = {"cache_key": key, "rewrites": ["a", "b", "c"]}
    (tmp_path / "rewrites.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid cached rewrite record"):
        _run_with_status(tmp_path, client=FakeLLMClient(), rewrite_mode="multi", generation_enabled=False)
    assert not (tmp_path / "run_log.jsonl").exists()


def test_structured_rewrite_mode_runs_through_the_pipeline(tmp_path) -> None:
    """R6: the structured mode gives the pipeline the same four rewrites as multi (plus a cached structure)."""
    questions_path = tmp_path / "questions.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    _write_jsonl(questions_path, _questions())
    _write_jsonl(chunks_path, _chunks())
    structured_reply = json.dumps(
        {
            "entities": ["BRCA1"],
            "abbreviations": [],
            "answer_type": "role",
            "constraints": ["in breast cancer"],
            "rewrites": {
                "bm25_lexical": "BRCA1 breast cancer role",
                "dense_semantic": "role of BRCA1 in breast cancer",
                "abbreviation_expanded": "BRCA1 breast cancer function",
                "minimal_entity_locked": "BRCA1 role in breast cancer",
            },
        }
    )
    client = StaticLLMClient(_reply(structured_reply))
    config = {**_config(), "rewrite_cache_path": str(tmp_path / "rewrites.jsonl")}
    state = run_rag_workflow(
        qid="q-1",
        retriever_mode="bm25",
        rewrite_mode="structured",
        guard_mode="off",
        k=4,
        questions_path=questions_path,
        chunks_path=chunks_path,
        config=config,
        bm25_backend=_bm25_backend(),
        chat_models={"rewrite": client},
        generation_enabled=False,
        log_path=tmp_path / "run.jsonl",
    )
    assert state["rewrite_status"] == "ok" and len(state["rewrites"]) == 4
    assert "entities" in client.calls[0]["messages"][0]["content"]  # the structured prompt was sent
    log = json.loads((tmp_path / "run.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert log["config_snapshot"]["rewrite_mode"] == "structured" and log["query_rewrites"] == state["rewrites"]


def test_rewrite_gate_threshold_comes_from_the_config_per_retriever() -> None:
    from rag_thesis.pipeline.workflow import _gate_threshold

    config = {"rewrite_gate_thresholds": {"bm25": 30, "dense": 0.6}}
    assert _gate_threshold(config, "bm25") == 30.0 and _gate_threshold(config, "hybrid") is None
    assert _gate_threshold({}, "bm25") is None
    with pytest.raises(ValueError, match="must be a number"):
        _gate_threshold({"rewrite_gate_thresholds": {"bm25": "high"}}, "bm25")
    with pytest.raises(ValueError, match="must map a retriever"):
        _gate_threshold({"rewrite_gate_thresholds": [30]}, "bm25")


def test_no_gate_without_rewriting(tmp_path) -> None:
    questions_path, chunks_path = tmp_path / "questions.jsonl", tmp_path / "chunks.jsonl"
    _write_jsonl(questions_path, _questions())
    _write_jsonl(chunks_path, _chunks())
    state = run_rag_workflow(
        qid="q-1",
        retriever_mode="bm25",
        rewrite_mode="none",
        guard_mode="off",
        k=4,
        questions_path=questions_path,
        chunks_path=chunks_path,
        config={**_config(), "rewrite_gate_thresholds": {"bm25": 0.0}},
        bm25_backend=_bm25_backend(),
        generation_enabled=False,
        log_path=tmp_path / "run.jsonl",
    )
    assert "rewrite_gate" not in state and state["rewrite_status"] == "not_requested"
    assert "rewrite_gate" not in json.loads((tmp_path / "run.jsonl").read_text(encoding="utf-8"))
