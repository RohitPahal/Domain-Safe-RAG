"""tools/compare_corpora.py: the 9 logged conditions on several corpora, side by side (fakes only)."""

from __future__ import annotations

import json
from pathlib import Path

import tools.compare_corpora as compare
from tools.compare_corpora import CONDITIONS, compare_corpora, summary_markdown

QUESTION = {"qid": "q1", "question": "which gene?", "doc_id": "d", "answers": [{"text": "x", "answer_start": 0}]}
GOLD = {"qid": "q1", "gold_doc_id": "d", "gold_chunk_ids": ["d::0"]}


class FakeIndex:
    """Ranks the gold chunk first in the small corpus and below 10 distractors in the big one."""

    def __init__(self, big: bool) -> None:
        self.big = big

    def retrieve(self, query: str, k: int) -> list[dict]:
        ids = [f"other-{i}::0" for i in range(10)] + ["d::0"] if self.big else ["d::0"]
        return [
            {"chunk_id": chunk_id, "doc_id": chunk_id.split("::")[0], "score": 1.0 / rank, "rank": rank}
            for rank, chunk_id in enumerate(ids[:k], start=1)
        ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _run_log_row(rewrites: list[str]) -> dict:
    return {
        "qid": "q1",
        "query_original": "which gene?",
        "query_rewrites": rewrites,
        "guard_report": {},
        "retrieval_results": [],
        "answer_text": None,
        "citations": [],
        "latency_ms": 1.0,
        "config_snapshot": {},
    }


def test_compares_every_condition_on_every_corpus(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for condition in CONDITIONS:
        rewrites = ["which gene?"] if condition.endswith("_none") else ["gene a", "gene b"]
        _write_jsonl(runs_dir / f"{condition}.jsonl", [_run_log_row(rewrites)])
    _write_jsonl(tmp_path / "questions.jsonl", [QUESTION])
    _write_jsonl(tmp_path / "gold.jsonl", [GOLD])
    (tmp_path / "config.yaml").write_text("retrieve_k: 10\nrrf_k: 60\n", encoding="utf-8")
    monkeypatch.setattr(compare, "load_bm25", lambda path: FakeIndex(big="full" in str(path)))
    monkeypatch.setattr(
        compare, "load_chroma_index", lambda path, query_cache_dir=None: FakeIndex(big="full" in str(path))
    )

    metrics, details, k = compare_corpora(
        [("subset", tmp_path / "subset"), ("full", tmp_path / "full")],
        runs_dir,
        tmp_path / "questions.jsonl",
        tmp_path / "gold.jsonl",
        tmp_path / "config.yaml",
    )

    assert k == 10
    assert [(row["corpus"], row["condition"]) for row in metrics] == [
        (corpus, condition) for corpus in ("subset", "full") for condition in CONDITIONS
    ]
    assert {row["Recall@10"] for row in metrics if row["corpus"] == "subset"} == {1.0}
    assert {row["Recall@10"] for row in metrics if row["corpus"] == "full"} == {0.0}  # pushed out of the top-10
    assert len(details) == 2 * len(CONDITIONS)
    table = summary_markdown(metrics, details, k)
    assert "post-thesis" in table
    assert "| bm25_none | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0 / 1 |" in table
