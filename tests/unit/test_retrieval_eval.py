from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_thesis.cli.eval_retrieval import app as eval_app
from rag_thesis.evaluation.retrieval import (
    bioasq_question_id,
    compute_first_relevant_rank,
    compute_hit_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
    evaluate_retriever,
    load_logged_queries,
    question_level_gold,
)

RUNNER = CliRunner()


class FakeRetriever:
    def __init__(self, results_by_query: dict[str, list[dict]]) -> None:
        self.results_by_query = results_by_query

    def retrieve(self, query: str, k: int) -> list[dict]:
        return [dict(item) for item in self.results_by_query.get(query, [])[:k]]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _write_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "eval_k_list: [5, 10, 20]",
                "rrf_k: 60",
                "retrieve_k: 10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _questions() -> list[dict]:
    return [
        {
            "qid": "q-1",
            "question": "alpha beta gamma",
            "doc_id": "doc-1",
            "answers": [{"text": "alpha", "answer_start": 0}],
        },
        {
            "qid": "q-2",
            "question": "delta epsilon",
            "doc_id": "doc-2",
            "answers": [{"text": "delta", "answer_start": 0}],
        },
    ]


def _gold() -> list[dict]:
    return [
        {
            "qid": "q-1",
            "gold_doc_id": "doc-1",
            "gold_chunk_ids": ["doc-1::0", "doc-1::1"],
            "mapping_status": "full_cover",
        },
        {"qid": "q-2", "gold_doc_id": "doc-2", "gold_chunk_ids": ["doc-2::0"], "mapping_status": "full_cover"},
    ]


def _bm25_backend() -> FakeRetriever:
    return FakeRetriever(
        {
            "alpha beta gamma": [
                {"chunk_id": "doc-1::0", "doc_id": "doc-1", "score": 3.0, "rank": 1},
                {"chunk_id": "doc-3::0", "doc_id": "doc-3", "score": 2.0, "rank": 2},
                {"chunk_id": "doc-1::1", "doc_id": "doc-1", "score": 1.5, "rank": 3},
            ],
            "delta epsilon": [
                {"chunk_id": "doc-4::0", "doc_id": "doc-4", "score": 4.0, "rank": 1},
                {"chunk_id": "doc-2::0", "doc_id": "doc-2", "score": 3.0, "rank": 2},
            ],
        }
    )


def _dense_backend() -> FakeRetriever:
    return FakeRetriever(
        {
            "alpha beta gamma": [
                {"chunk_id": "doc-3::0", "doc_id": "doc-3", "score": 0.95, "rank": 1},
                {"chunk_id": "doc-1::1", "doc_id": "doc-1", "score": 0.90, "rank": 2},
                {"chunk_id": "doc-1::0", "doc_id": "doc-1", "score": 0.80, "rank": 3},
            ],
            "delta epsilon": [
                {"chunk_id": "doc-2::0", "doc_id": "doc-2", "score": 0.99, "rank": 1},
                {"chunk_id": "doc-4::0", "doc_id": "doc-4", "score": 0.20, "rank": 2},
            ],
        }
    )


def test_metric_helpers_compute_expected_values() -> None:
    retrieved = ["a", "b", "c", "d"]
    gold = ["c", "d", "e"]

    assert compute_recall_at_k(retrieved, gold, 2) == 0.0
    assert compute_recall_at_k(retrieved, gold, 3) == 1.0 / 3.0
    assert compute_recall_at_k(retrieved, gold, 4) == 2.0 / 3.0
    assert compute_first_relevant_rank(retrieved, gold) == 3


def test_ndcg_rewards_gold_chunks_near_the_top() -> None:
    assert compute_ndcg_at_k(["g1", "x", "y"], ["g1"], 10) == 1.0  # the only gold chunk first
    assert compute_ndcg_at_k(["x", "g1"], ["g1"], 10) == pytest.approx(1 / math.log2(3))
    assert compute_ndcg_at_k(["x", "y"], ["g1"], 10) == 0.0
    # Two gold chunks: ideal = 1 + 1/log2(3); found at ranks 1 and 3.
    assert compute_ndcg_at_k(["g1", "x", "g2"], ["g1", "g2"], 10) == pytest.approx(
        (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    )
    assert compute_ndcg_at_k(["x", "g1"], ["g1"], 1) == 0.0  # outside the cut-off
    assert compute_ndcg_at_k(["g1"], [], 10) == 0.0


def test_evaluate_retriever_computes_recall_and_mrr_correctly() -> None:
    metrics_row, detail_rows = evaluate_retriever(
        retriever="bm25",
        questions=_questions(),
        gold_by_qid={row["qid"]: row for row in _gold()},
        config={"rrf_k": 60},
        eval_k_list=[5, 10, 20],
        bm25_backend=_bm25_backend(),
    )

    assert metrics_row["retriever"] == "bm25"
    assert metrics_row["num_questions"] == 2
    assert metrics_row["Recall@5"] == 1.0
    assert metrics_row["Recall@10"] == 1.0
    assert metrics_row["Recall@20"] == 1.0
    assert metrics_row["MRR"] == 0.75
    # q-1: gold at ranks 1 and 3 of 2 gold chunks; q-2: the only gold chunk at rank 2.
    expected_ndcg = ((1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)) + 1 / math.log2(3)) / 2
    assert metrics_row["nDCG@10"] == pytest.approx(expected_ndcg, abs=1e-6)
    assert detail_rows[0]["first_relevant_rank"] == 1
    assert detail_rows[1]["first_relevant_rank"] == 2


def test_run_retrieval_eval_cli_generates_reports_without_network(tmp_path, monkeypatch) -> None:
    questions_path = tmp_path / "questions.jsonl"
    gold_path = tmp_path / "gold_evidence.jsonl"
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "reports"
    bm25_dir = tmp_path / "bm25"
    vector_dir = tmp_path / "vector"
    bm25_dir.mkdir()
    vector_dir.mkdir()
    _write_jsonl(questions_path, _questions())
    _write_jsonl(gold_path, _gold())
    _write_config(config_path)

    import rag_thesis.evaluation.retrieval as eval_script

    monkeypatch.setattr(eval_script, "load_bm25", lambda path: _bm25_backend())
    monkeypatch.setattr(eval_script, "load_chroma_index", lambda path, query_cache_dir=None: _dense_backend())

    result = RUNNER.invoke(
        eval_app,
        [
            "--retriever",
            "all",
            "--questions-path",
            str(questions_path),
            "--gold-path",
            str(gold_path),
            "--config-path",
            str(config_path),
            "--bm25-dir",
            str(bm25_dir),
            "--vector-dir",
            str(vector_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert "metrics_path:" in result.output

    metrics_rows = list(csv.DictReader((output_dir / "retrieval_metrics.csv").open("r", encoding="utf-8")))
    details_rows = list(csv.DictReader((output_dir / "retrieval_details.csv").open("r", encoding="utf-8")))
    summary_text = (output_dir / "summary.md").read_text(encoding="utf-8")

    assert [row["retriever"] for row in metrics_rows] == ["bm25", "dense", "hybrid"]
    assert len(details_rows) == 6
    assert details_rows[0]["qid"] == "q-1"
    assert details_rows[0]["retriever"] == "bm25"
    assert details_rows[0]["retrieved_chunk_ids"].startswith("[")
    assert json.loads(details_rows[0]["queries"]) == ["alpha beta gamma"]  # by default: the question itself
    assert "| Retriever | Questions | Recall@5 | Recall@10 | Recall@20 | MRR | nDCG@10 |" in summary_text
    assert "Queries: the questions themselves." in summary_text


def _run_log_row(qid: str, question: str, rewrites: list[str]) -> dict:
    """A minimal valid run-log line (RunLogLine) with the given post-guard rewrites."""
    return {
        "qid": qid,
        "query_original": question,
        "query_rewrites": rewrites,
        "guard_report": {"guard_mode": "off", "final_rewrites": rewrites, "reports": []},
        "retrieval_results": [],
        "answer_text": None,
        "citations": [],
        "latency_ms": 1.0,
        "config_snapshot": {},
    }


def test_logged_queries_are_fused_with_rrf_like_the_pipeline() -> None:
    backend = FakeRetriever(
        {
            "query a": [
                {"chunk_id": "doc-1::0", "doc_id": "doc-1", "score": 2.0, "rank": 1},
                {"chunk_id": "doc-9::0", "doc_id": "doc-9", "score": 1.0, "rank": 2},
            ],
            "query b": [
                {"chunk_id": "doc-9::0", "doc_id": "doc-9", "score": 2.0, "rank": 1},
                {"chunk_id": "doc-8::0", "doc_id": "doc-8", "score": 1.0, "rank": 2},
            ],
        }
    )
    _, detail_rows = evaluate_retriever(
        retriever="bm25",
        questions=_questions()[:1],
        gold_by_qid={row["qid"]: row for row in _gold()},
        config={"rrf_k": 60},
        eval_k_list=[5],
        bm25_backend=backend,
        queries_by_qid={"q-1": ["query a", "query b"]},
    )
    # RRF: doc-9 = 1/62 + 1/61 (found by both) > doc-1 = 1/61 > doc-8 = 1/62.
    assert json.loads(detail_rows[0]["retrieved_chunk_ids"]) == ["doc-9::0", "doc-1::0", "doc-8::0"]
    assert json.loads(detail_rows[0]["queries"]) == ["query a", "query b"]
    assert detail_rows[0]["first_relevant_rank"] == 2


def test_a_question_without_logged_queries_fails_loudly() -> None:
    with pytest.raises(ValueError, match="No logged queries for qid: q-2"):
        evaluate_retriever(
            retriever="bm25",
            questions=_questions(),
            gold_by_qid={row["qid"]: row for row in _gold()},
            config={"rrf_k": 60},
            eval_k_list=[5],
            bm25_backend=_bm25_backend(),
            queries_by_qid={"q-1": ["alpha beta gamma"]},
        )


def test_logged_queries_are_read_like_the_pipeline_reads_its_rewrites(tmp_path) -> None:
    log_path = tmp_path / "run.jsonl"
    _write_jsonl(
        log_path,
        [
            _run_log_row("q-1", "alpha beta gamma", [" query a ", "query a", "query b"]),
            _run_log_row("q-2", "delta epsilon", []),
        ],
    )
    assert load_logged_queries(log_path) == {"q-1": ["query a", "query b"], "q-2": ["delta epsilon"]}


def test_a_run_log_with_a_repeated_qid_is_refused(tmp_path) -> None:
    log_path = tmp_path / "run.jsonl"
    _write_jsonl(log_path, [_run_log_row("q-1", "alpha", ["a"]), _run_log_row("q-1", "alpha", ["b"])])
    with pytest.raises(ValueError, match="appears more than once"):
        load_logged_queries(log_path)


def test_eval_cli_replays_the_queries_of_a_run_log(tmp_path, monkeypatch) -> None:
    questions_path = tmp_path / "questions.jsonl"
    gold_path = tmp_path / "gold_evidence.jsonl"
    config_path = tmp_path / "config.yaml"
    log_path = tmp_path / "run.jsonl"
    output_dir = tmp_path / "reports"
    (tmp_path / "bm25").mkdir()
    _write_jsonl(questions_path, _questions())
    _write_jsonl(gold_path, _gold())
    _write_config(config_path)
    # q-2 is retrieved with a rewrite that finds its gold chunk first.
    _write_jsonl(log_path, [_run_log_row("q-1", "alpha beta gamma", []), _run_log_row("q-2", "delta epsilon", ["d"])])
    backend = _bm25_backend()
    backend.results_by_query["d"] = [{"chunk_id": "doc-2::0", "doc_id": "doc-2", "score": 1.0, "rank": 1}]

    import rag_thesis.evaluation.retrieval as eval_script

    monkeypatch.setattr(eval_script, "load_bm25", lambda path: backend)
    result = RUNNER.invoke(
        eval_app,
        [
            "--retriever",
            "bm25",
            "--questions-path",
            str(questions_path),
            "--gold-path",
            str(gold_path),
            "--config-path",
            str(config_path),
            "--bm25-dir",
            str(tmp_path / "bm25"),
            "--output-dir",
            str(output_dir),
            "--queries-from-log",
            str(log_path),
        ],
    )

    assert result.exit_code == 0, result.output
    details = list(csv.DictReader((output_dir / "retrieval_details.csv").open("r", encoding="utf-8")))
    assert [json.loads(row["queries"]) for row in details] == [["alpha beta gamma"], ["d"]]
    assert details[1]["first_relevant_rank"] == "1"
    assert "replayed from run log" in (output_dir / "summary.md").read_text(encoding="utf-8")


def test_hit_at_k_is_success_at_k() -> None:
    assert compute_hit_at_k(3, 5) == 1.0
    assert compute_hit_at_k(6, 5) == 0.0
    assert compute_hit_at_k(None, 5) == 0.0
    metrics_row, detail_rows = evaluate_retriever(
        retriever="bm25",
        questions=_questions(),
        gold_by_qid={row["qid"]: row for row in _gold()},
        config={"rrf_k": 60},
        eval_k_list=[1, 5],
        bm25_backend=_bm25_backend(),
    )
    assert [(row["Hit@1"], row["Recall@1"]) for row in detail_rows] == [(1.0, 0.5), (0.0, 0.0)]  # q-1: 1 of 2 gold
    assert metrics_row["Hit@1"] == 0.5 and metrics_row["Hit@5"] == 1.0


def test_question_level_gold_joins_the_snippets_of_one_bioasq_question() -> None:
    assert bioasq_question_id("54df695b1388e8454a000004_016") == "54df695b1388e8454a000004"
    gold_rows = [
        {"qid": "Q1_001", "gold_doc_id": "a", "gold_chunk_ids": ["a::0"]},
        {"qid": "Q2_001", "gold_doc_id": "b", "gold_chunk_ids": ["b::1"]},
        {"qid": "Q1_002", "gold_doc_id": "c", "gold_chunk_ids": ["c::0", "c::1"]},
        {"qid": "Q1_003", "gold_doc_id": "a", "gold_chunk_ids": ["a::0"]},  # same snippet asked again
    ]
    widened = question_level_gold(gold_rows)
    assert widened["Q1_001"]["gold_chunk_ids"] == ["a::0", "c::0", "c::1"]
    assert widened["Q1_002"]["gold_chunk_ids"] == ["c::0", "c::1", "a::0"]  # own chunks first
    assert widened["Q2_001"]["gold_chunk_ids"] == ["b::1"]
    assert widened["Q1_002"]["gold_doc_id"] == "c" and gold_rows[2]["gold_chunk_ids"] == ["c::0", "c::1"]
