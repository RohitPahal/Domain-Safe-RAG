from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import rag_thesis.cli.run_batch as run_batch_cli
import rag_thesis.pipeline.run_batch as run_rag_batch_script
from rag_thesis.pipeline.workflow import RAGRunContext

RUNNER = CliRunner()


def test_select_qids_is_deterministic() -> None:
    ordered_qids = ["q-1", "q-2", "q-3", "q-4", "q-5"]

    first = run_rag_batch_script.select_qids(ordered_qids, random_n=3, seed=42)
    second = run_rag_batch_script.select_qids(ordered_qids, random_n=3, seed=42)

    assert first == second
    assert len(first) == 3
    assert len(set(first)) == 3


def test_run_rag_batch_builds_context_once_and_reuses_it(monkeypatch, tmp_path) -> None:
    calls = {
        "build_context": 0,
        "workflow": 0,
    }
    seen_context_ids: list[int] = []

    def fake_build_context(**kwargs):
        calls["build_context"] += 1
        assert kwargs["retriever_mode"] == "hybrid"
        assert kwargs["allow_llm_calls"] is False
        return RAGRunContext(
            config={"rrf_k": 60},
            questions_by_id={
                "q-1": {"qid": "q-1", "question": "Question 1"},
                "q-2": {"qid": "q-2", "question": "Question 2"},
                "q-3": {"qid": "q-3", "question": "Question 3"},
            },
            chunks_by_id={},
            bm25_backend=object(),
            dense_backend=object(),
        )

    def fake_run_with_context(**kwargs):
        calls["workflow"] += 1
        seen_context_ids.append(id(kwargs["context"]))
        return {
            "qid": kwargs["qid"],
            "retrieval_results": [{"chunk_id": "doc-1::0"}],
            "answer_text": None,
        }

    monkeypatch.setattr(run_rag_batch_script, "build_rag_run_context", fake_build_context)
    monkeypatch.setattr(run_rag_batch_script, "run_rag_workflow_with_context", fake_run_with_context)

    progress_lines: list[tuple[int, int, str]] = []
    states = run_rag_batch_script.run_rag_batch(
        retriever_mode="hybrid",
        rewrite_mode="none",
        guard_mode="off",
        k=10,
        questions_path=tmp_path / "questions.jsonl",
        chunks_path=tmp_path / "chunks.jsonl",
        config_path=tmp_path / "config.yaml",
        bm25_dir=tmp_path / "bm25",
        vector_dir=tmp_path / "vector",
        log_path=tmp_path / "run_log.jsonl",
        first_n=2,
        progress_callback=lambda index, total, state: progress_lines.append((index, total, state["qid"])),
    )

    assert calls["build_context"] == 1
    assert calls["workflow"] == 2
    assert len(set(seen_context_ids)) == 1
    assert [state["qid"] for state in states] == ["q-1", "q-2"]
    assert progress_lines == [(1, 2, "q-1"), (2, 2, "q-2")]


def test_a_fatal_error_stops_the_batch_at_once(monkeypatch, tmp_path) -> None:
    """No question runs, and no API call is spent, after a fatal error. This is why batches loop
    over the compiled graph instead of using `graph.batch`, which keeps running the remaining questions."""
    started: list[str] = []

    def fake_run_with_context(**kwargs):
        started.append(kwargs["qid"])
        if kwargs["qid"] == "q-2":
            raise RuntimeError("fatal")
        return {"qid": kwargs["qid"]}

    questions = {qid: {"qid": qid, "question": "Q"} for qid in ("q-1", "q-2", "q-3")}
    context = RAGRunContext(config={"rrf_k": 60}, questions_by_id=questions, chunks_by_id={})
    monkeypatch.setattr(run_rag_batch_script, "build_rag_run_context", lambda **kwargs: context)
    monkeypatch.setattr(run_rag_batch_script, "run_rag_workflow_with_context", fake_run_with_context)

    with pytest.raises(RuntimeError, match="fatal"):
        run_rag_batch_script.run_rag_batch(
            retriever_mode="bm25",
            rewrite_mode="none",
            guard_mode="off",
            k=10,
            questions_path=tmp_path / "questions.jsonl",
            chunks_path=tmp_path / "chunks.jsonl",
            config_path=tmp_path / "config.yaml",
            bm25_dir=tmp_path / "bm25",
            vector_dir=tmp_path / "vector",
            log_path=tmp_path / "run_log.jsonl",
        )
    assert started == ["q-1", "q-2"]


def test_run_rag_batch_cli_smoke(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_run_rag_batch(**kwargs):
        captured.update(kwargs)
        return [
            {"qid": "q-2", "retrieval_results": [{"chunk_id": "doc-1::0"}], "answer_text": "A1"},
            {"qid": "q-1", "retrieval_results": [{"chunk_id": "doc-2::0"}], "answer_text": None},
        ]

    monkeypatch.setattr(run_batch_cli, "run_rag_batch", fake_run_rag_batch)  # the CLI's reference

    args = [
        "--retriever",
        "hybrid",
        "--rewrite",
        "none",
        "--guard",
        "off",
        "--random-n",
        "2",
        "--seed",
        "7",
        "--questions-path",
        str(tmp_path / "questions.jsonl"),
        "--chunks-path",
        str(tmp_path / "chunks.jsonl"),
        "--config-path",
        str(tmp_path / "config.yaml"),
        "--log-path",
        str(tmp_path / "isolated.jsonl"),
    ]

    for path in (tmp_path / "questions.jsonl", tmp_path / "chunks.jsonl", tmp_path / "config.yaml"):
        path.write_text("placeholder\n", encoding="utf-8")

    result = RUNNER.invoke(run_batch_cli.app, args)

    assert result.exit_code == 0
    assert "preloading_context: start" in result.output
    assert "preloading_context: done" in result.output
    assert "total_qids: 2" in result.output
    assert captured["retriever_mode"] == "hybrid"
    assert captured["rewrite_mode"] == "none"
    assert captured["guard_mode"] == "off"
    assert captured["random_n"] == 2
    assert captured["seed"] == 7
    assert captured["log_path"] == Path(tmp_path / "isolated.jsonl")
