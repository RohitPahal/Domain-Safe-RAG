"""tools/hyde_eval.py (R8): query sets, leaking passages, the report (fakes only)."""

from __future__ import annotations

import json

import pytest
from tools.hyde_eval import leaking_qids, mrr_by_set, query_sets, report_markdown

from rag_thesis.rewrite.rewriter import load_cached_passages

QUESTIONS = [
    {"qid": "Q1_001", "question": "Which gene causes CF?", "doc_id": "g", "answers": [{"text": "CFTR"}]},
    {"qid": "Q2_001", "question": "What is ATP?", "doc_id": "g", "answers": [{"text": "energy carrier"}]},
]
PASSAGES = {"Q1_001": "Mutations in CFTR cause CF.", "Q2_001": "ATP is a nucleotide found in cells."}


class Fake:
    """Passages find the gold chunk; the bare questions don't."""

    name = "fake"

    def retrieve(self, query, k):
        chunk = "g::0" if query in PASSAGES.values() else "x::0"
        return [{"chunk_id": chunk, "doc_id": chunk[0], "score": 1.0, "rank": 1}]


def test_query_sets_and_leaks():
    sets = query_sets(QUESTIONS, PASSAGES)
    assert sets["hyde"]["Q1_001"] == [PASSAGES["Q1_001"]]
    assert sets["question+hyde"]["Q2_001"] == ["What is ATP?", PASSAGES["Q2_001"]]
    assert leaking_qids(QUESTIONS, PASSAGES) == {"Q1_001"}  # the passage states CFTR


def test_evaluation_and_report_split_out_leaking_passages():
    gold = {q["qid"]: {"qid": q["qid"], "gold_doc_id": "g", "gold_chunk_ids": ["g::0"]} for q in QUESTIONS}
    backends = {"bm25_backend": Fake(), "dense_backend": Fake()}
    mrr = mrr_by_set(QUESTIONS, gold, PASSAGES, backends, {"retrieve_k": 10, "rrf_k": 60}, "bm25")
    assert mrr["question"] == [0.0, 0.0] and mrr["hyde"] == [1.0, 1.0]
    report = report_markdown({("full", "bm25"): mrr}, QUESTIONS, {"Q1_001"})
    assert "| full | bm25 | all (2) |" in report and "| full | bm25 | no leak (1) |" in report
    assert "1 passages contain a gold answer" in report


def test_cached_passages_are_read_by_mode(tmp_path):
    path = tmp_path / "rewrites.jsonl"
    records = [
        {"cache_key": "a", "qid": "Q1_001", "rewrite_mode": "hyde", "rewrites": ["a passage"]},
        {"cache_key": "b", "qid": "Q1_001", "rewrite_mode": "multi", "rewrites": ["1", "2", "3", "4"]},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    assert load_cached_passages(path, ["Q1_001"]) == {"Q1_001": "a passage"}
    with pytest.raises(ValueError, match="no cached hyde rewrites"):
        load_cached_passages(path, ["Q2_001"])
