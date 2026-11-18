import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from rag_thesis.cli.validate import app
from rag_thesis.schemas import (
    AnswerSpan,
    ChunkLine,
    DocumentLine,
    GoldEvidenceLine,
    QuestionLine,
    RunLogLine,
)


@pytest.mark.parametrize(
    ("model_cls", "payload"),
    [
        (
            AnswerSpan,
            {
                "text": "insulin",
                "answer_start": 3,
            },
        ),
        (
            DocumentLine,
            {
                "doc_id": "doc-001",
                "text": "Document body",
                "metadata": {"source": "bioasq"},
            },
        ),
        (
            ChunkLine,
            {
                "chunk_id": "chunk-001",
                "doc_id": "doc-001",
                "chunk_index": 0,
                "text": "Chunk body",
                "char_start": 0,
                "char_end": 10,
                "metadata": {"section": "abstract"},
            },
        ),
        (
            QuestionLine,
            {
                "qid": "q-001",
                "question": "What is insulin?",
                "doc_id": "doc-001",
                "answers": [{"text": "a hormone", "answer_start": 0}],
            },
        ),
        (
            GoldEvidenceLine,
            {
                "qid": "q-001",
                "gold_doc_id": "doc-001",
                "gold_chunk_ids": ["chunk-001"],
                "mapping_status": "mapped",
            },
        ),
        (
            RunLogLine,
            {
                "qid": "q-001",
                "query_original": "What is insulin?",
                "query_rewrites": ["insulin definition"],
                "guard_report": {"status": "pass"},
                "retrieval_results": [{"doc_id": "doc-001", "score": 0.9}],
                "answer_text": "Insulin is a hormone.",
                "citations": ["chunk-001"],
                "latency_ms": 123.4,
                "openai_usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "config_snapshot": {"retrieve_k": 10},
            },
        ),
    ],
)
def test_valid_schema_examples_pass(model_cls, payload):
    instance = model_cls.model_validate(payload)
    assert instance is not None


@pytest.mark.parametrize(
    ("model_cls", "payload"),
    [
        (
            AnswerSpan,
            {
                "text": "insulin",
                "answer_start": -1,
            },
        ),
        (
            DocumentLine,
            {
                "doc_id": "",
                "text": "Document body",
                "metadata": {"source": "bioasq"},
            },
        ),
        (
            ChunkLine,
            {
                "chunk_id": "chunk-001",
                "doc_id": "doc-001",
                "chunk_index": 0,
                "text": "Chunk body",
                "char_start": 10,
                "char_end": 5,
                "metadata": {"section": "abstract"},
            },
        ),
        (
            QuestionLine,
            {
                "qid": "q-001",
                "question": "",
                "doc_id": "doc-001",
                "answers": [{"text": "a hormone", "answer_start": 0}],
            },
        ),
        (
            GoldEvidenceLine,
            {
                "qid": "q-001",
                "gold_doc_id": "",
                "gold_chunk_ids": ["chunk-001"],
            },
        ),
        (
            RunLogLine,
            {
                "qid": "q-001",
                "query_original": "What is insulin?",
                "query_rewrites": ["insulin definition"],
                "guard_report": {"status": "pass"},
                "retrieval_results": [{"doc_id": "doc-001", "score": 0.9}],
                "answer_text": "",
                "citations": ["chunk-001"],
                "latency_ms": 123.4,
                "openai_usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "config_snapshot": {"retrieve_k": 10},
            },
        ),
    ],
)
def test_invalid_schema_examples_fail(model_cls, payload):
    with pytest.raises(ValidationError):
        model_cls.model_validate(payload)


@pytest.mark.parametrize(
    ("model_cls", "payload"),
    [
        (
            AnswerSpan,
            {
                "text": "insulin",
                "answer_start": 1,
                "extra_field": "not allowed",
            },
        ),
        (
            DocumentLine,
            {
                "doc_id": "doc-001",
                "text": "Document body",
                "metadata": {"source": "bioasq"},
                "extra_field": "not allowed",
            },
        ),
        (
            ChunkLine,
            {
                "chunk_id": "chunk-001",
                "doc_id": "doc-001",
                "chunk_index": 0,
                "text": "Chunk body",
                "char_start": 0,
                "char_end": 10,
                "metadata": {"section": "abstract"},
                "extra_field": "not allowed",
            },
        ),
        (
            QuestionLine,
            {
                "qid": "q-001",
                "question": "What is insulin?",
                "doc_id": "doc-001",
                "answers": [{"text": "a hormone", "answer_start": 0}],
                "extra_field": "not allowed",
            },
        ),
        (
            GoldEvidenceLine,
            {
                "qid": "q-001",
                "gold_doc_id": "doc-001",
                "gold_chunk_ids": ["chunk-001"],
                "extra_field": "not allowed",
            },
        ),
        (
            RunLogLine,
            {
                "qid": "q-001",
                "query_original": "What is insulin?",
                "query_rewrites": ["insulin definition"],
                "guard_report": {"status": "pass"},
                "retrieval_results": [{"doc_id": "doc-001", "score": 0.9}],
                "answer_text": "Insulin is a hormone.",
                "citations": ["chunk-001"],
                "latency_ms": 123.4,
                "openai_usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "config_snapshot": {"retrieve_k": 10},
                "extra_field": "not allowed",
            },
        ),
    ],
)
def test_schemas_forbid_extra_fields(model_cls, payload):
    with pytest.raises(ValidationError):
        model_cls.model_validate(payload)


def test_validate_artifacts_exits_nonzero_on_invalid_input(tmp_path):
    processed_dir = tmp_path / "processed"
    eval_dir = tmp_path / "eval"
    processed_dir.mkdir()
    eval_dir.mkdir()

    (processed_dir / "documents.jsonl").write_text(
        '{"doc_id":"doc-001","text":"Document body","metadata":{"source":"bioasq"}}\n',
        encoding="utf-8",
    )
    (processed_dir / "chunks.jsonl").write_text(
        '{"chunk_id":"chunk-001","doc_id":"doc-001","chunk_index":0,"text":"Chunk '
        'body","char_start":0,"char_end":10,"metadata":{"section":"abstract"}}\n',
        encoding="utf-8",
    )
    (eval_dir / "questions.jsonl").write_text(
        '{"qid":"q-001","question":"What is insulin?","doc_id":"doc-001","answers":[{"text":"a '
        'hormone","answer_start":0}]}\n',
        encoding="utf-8",
    )
    (eval_dir / "gold_evidence.jsonl").write_text(
        '{"qid":"q-001","gold_doc_id":"doc-001","gold_chunk_ids":["chunk-001"]}\n',
        encoding="utf-8",
    )
    (eval_dir / "run_log.jsonl").write_text(
        '{"qid":"q-001","query_original":"What is insulin?","query_rewrites":["insulin '
        'definition"],"guard_report":{"status":"pass"},"retrieval_results":[{"doc_id":"doc-001","score":0.9}],"answer_text":"","citations":["chunk-001"],"latency_ms":123.4,"openai_usage":{"prompt_tokens":10,"completion_tokens":5},"config_snapshot":{"retrieve_k":10}}\n',
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--processed-dir",
            str(processed_dir),
            "--eval-dir",
            str(eval_dir),
        ],
    )

    assert result.exit_code != 0
    assert "line 1" in result.output
    assert "answer_text" in result.output
