"""Retrieval latency benchmark: timing helpers, report files, protection and the CLI."""

from __future__ import annotations

import csv
import json

import pytest
from typer.testing import CliRunner

from rag_thesis import protection
from rag_thesis.cli import bench_retrieval as bench_cli
from rag_thesis.evaluation import bench
from rag_thesis.protection import ProtectedPathError, WriteGuard


class CountingRetriever:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[dict]:
        self.queries.append((query, k))
        return []


def test_time_queries_asks_each_query_once():
    retriever = CountingRetriever()
    times_ms = bench.time_queries(retriever, ["a", "b", "c"], 5)
    assert retriever.queries == [("a", 5), ("b", 5), ("c", 5)]
    assert len(times_ms) == 3 and all(t >= 0 for t in times_ms)


def test_latency_row_statistics():
    row = bench.latency_row("dense", 20, 1.23456, [50.0, 1.0, 2.0, 3.0, 4.0])  # the first query stands apart
    assert row == {
        "retriever": "dense",
        "queries": 5,
        "k": 20,
        "index_load_s": 1.235,
        "first_query_ms": 50.0,
        "mean_ms": 2.5,
        "p50_ms": 2.5,
        "p95_ms": 3.85,
        "max_ms": 4.0,
    }
    assert bench.latency_row("bm25", 5, 0.0, [7.0])["mean_ms"] == 7.0  # one query: no "rest" to average


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        "".join(
            json.dumps(
                {
                    "qid": f"q{n}",
                    "question": f"Question {n}?",
                    "doc_id": "d1",
                    "answers": [{"text": "x", "answer_start": 0}],
                }
            )
            + "\n"
            for n in range(3)
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text("rrf_k: 60\neval_k_list: [5, 10]\n", encoding="utf-8")
    loads: list[str] = []

    def fake_bm25(path):
        loads.append("bm25")
        return CountingRetriever()

    def fake_dense(path, query_cache_dir=None):
        loads.append("dense")
        return CountingRetriever()

    monkeypatch.setattr(bench, "load_bm25", fake_bm25)
    monkeypatch.setattr(bench, "load_chroma_index", fake_dense)
    return {"questions": questions, "config": config, "loads": loads, "dir": tmp_path}


def _run(inputs, output_dir, **kwargs):
    return bench.run_retrieval_benchmark(
        retriever=kwargs.pop("retriever", "all"),
        questions_path=inputs["questions"],
        config_path=inputs["config"],
        bm25_dir=inputs["dir"] / "bm25",
        vector_dir=inputs["dir"] / "chroma",
        output_dir=output_dir,
        **kwargs,
    )


def test_benchmark_loads_each_index_once_and_writes_the_report(inputs):
    result = _run(inputs, inputs["dir"] / "reports")
    assert inputs["loads"] == ["bm25", "dense"]  # shared by the three retrievers
    assert [row["retriever"] for row in result["rows"]] == ["bm25", "dense", "hybrid"]
    assert all(row["queries"] == 3 and row["k"] == 10 for row in result["rows"])  # k = largest eval_k_list value
    with result["csv_path"].open(encoding="utf-8") as handle:
        assert [row["retriever"] for row in csv.DictReader(handle)] == ["bm25", "dense", "hybrid"]
    assert "| retriever | queries | k |" in result["summary_path"].read_text(encoding="utf-8")


def test_benchmark_loads_only_what_the_retriever_needs(inputs):
    result = _run(inputs, inputs["dir"] / "reports", retriever="bm25", k=3, max_qids=2)
    assert inputs["loads"] == ["bm25"]
    assert result["rows"][0]["queries"] == 2 and result["rows"][0]["k"] == 3


def test_benchmark_refuses_a_protected_output_dir_before_loading(inputs, monkeypatch):
    protected = inputs["dir"] / "protected"
    protected.mkdir()
    monkeypatch.setattr(protection, "DEFAULT_GUARD", WriteGuard([protected]))
    with pytest.raises(ProtectedPathError):
        _run(inputs, protected / "reports")
    assert inputs["loads"] == []


def test_cli_prints_the_table(inputs, monkeypatch):
    rows = [bench.latency_row(name, 20, 0.1, [1.0]) for name in ("bm25", "dense", "hybrid")]
    monkeypatch.setattr(
        bench_cli,
        "run_retrieval_benchmark",
        lambda **kwargs: {"rows": rows, "csv_path": "latency.csv", "summary_path": "latency.md"},
    )
    result = CliRunner().invoke(
        bench_cli.app,
        ["--questions-path", str(inputs["questions"]), "--config-path", str(inputs["config"])],
    )
    assert result.exit_code == 0, result.output
    assert "retriever  queries  k" in result.output and "hybrid  1  20" in result.output
    assert "csv_path: latency.csv" in result.output
