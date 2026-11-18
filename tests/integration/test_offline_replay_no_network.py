"""Replaying the official benchmark config makes zero network calls.

The real pipeline runs all 120 questions with the official `config_snapshot.yaml` (its rewrite
cache read-only) while every socket connection is blocked. Without permission no LLM model is
built; rewrites come from the cache. The multi-rewrite BM25 run must reproduce the official
`bm25_multi_guard_off` rewrites and top-10 lists exactly. Skips where data or results are absent.
"""

from __future__ import annotations

import json
import socket

import pytest

from rag_thesis.config import load_config
from rag_thesis.paths import PROJECT_ROOT
from rag_thesis.pipeline.workflow import build_rag_run_context, run_rag_workflow_with_context

OFFICIAL = PROJECT_ROOT / "outputs" / "final_benchmark" / "subset120_caseB_gem25flash_gem25flash_gem25pro_20260416"
QUESTIONS = PROJECT_ROOT / "data" / "eval" / "questions.jsonl"
CHUNKS = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
BM25_DIR = PROJECT_ROOT / "indexes" / "bm25"
REQUIRED = [OFFICIAL / "config_snapshot.yaml", OFFICIAL / "subset120" / "runs" / "bm25_multi_guard_off.jsonl"]
REQUIRED += [QUESTIONS, CHUNKS, BM25_DIR / "meta.json"]

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in REQUIRED), reason="official results, data or BM25 index not present"
)


@pytest.fixture
def no_network(monkeypatch):
    """Every attempt to open a network connection fails the test."""
    attempts: list[object] = []

    def refuse(*args, **kwargs):
        attempts.append(args)
        raise AssertionError(f"network access attempted: {args}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    return attempts


def _replay(tmp_path, guard_mode: str) -> list[dict]:
    context = build_rag_run_context(
        questions_path=QUESTIONS,
        chunks_path=CHUNKS,
        retriever_mode="bm25",
        config=load_config(OFFICIAL / "config_snapshot.yaml"),
        bm25_dir=BM25_DIR,
        allow_llm_calls=False,
    )
    assert context.chat_models == {}  # no permission: no LLM model is built at all
    log_path = tmp_path / f"replay_{guard_mode}.jsonl"
    for qid in context.questions_by_id:
        run_rag_workflow_with_context(
            qid=qid,
            retriever_mode="bm25",
            rewrite_mode="multi",
            guard_mode=guard_mode,
            k=10,
            context=context,
            log_path=log_path,
        )
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


def test_official_multi_rewrite_replay_is_exact_and_offline(tmp_path, no_network):
    rows = _replay(tmp_path, "off")
    official = {}
    with (OFFICIAL / "subset120" / "runs" / "bm25_multi_guard_off.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            official[row["qid"]] = row

    assert len(rows) == 120 and no_network == []
    for row in rows:
        expected = official[row["qid"]]
        assert row["query_rewrites"] == expected["query_rewrites"], row["qid"]
        assert [r["chunk_id"] for r in row["retrieval_results"]] == [
            r["chunk_id"] for r in expected["retrieval_results"]
        ], row["qid"]
        assert (row["rewrite_status"], row["generation_status"], row["llm_calls"]) == ("ok", "disabled", [])


def test_guard_on_replay_is_offline_and_never_calls_a_repair_model(tmp_path, no_network):
    rows = _replay(tmp_path, "on")
    assert len(rows) == 120 and no_network == []
    actions = {report["final_action"] for row in rows for report in row["guard_report"]["reports"]}
    assert actions <= {"pass", "fallback_original"}  # unsafe rewrites fall back: no repair model offline
    assert all(row["llm_calls"] == [] for row in rows)
