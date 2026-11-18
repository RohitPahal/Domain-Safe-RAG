"""tools/gate_eval.py (R5): never / always / gated query sets and their evaluation (fakes only)."""

from __future__ import annotations

from tools.gate_eval import evaluate, report_markdown, strategies

from rag_thesis.rewrite.rewriter import MULTI_REWRITE_ORDER

QUESTIONS = [
    {"qid": "Q1_001", "question": "sure question", "doc_id": "g", "answers": []},
    {"qid": "Q2_001", "question": "unsure question", "doc_id": "g", "answers": []},
]
REWRITES = {q["qid"]: {s: f"{q['question']} {s}" for s in MULTI_REWRITE_ORDER} for q in QUESTIONS}


class Fake:
    """The sure question scores 0.9 and finds the gold chunk; everything else scores 0.1."""

    name = "fake"

    def retrieve(self, query, k):
        if query == "sure question":
            return [{"chunk_id": "g::0", "doc_id": "g", "score": 0.9, "rank": 1}]
        chunk = "g::0" if query.startswith("unsure question ") else "x::0"  # rewrites find it
        return [{"chunk_id": chunk, "doc_id": chunk[0], "score": 0.1, "rank": 1}]


def test_the_gate_rewrites_only_unsure_questions():
    sets = strategies(QUESTIONS, REWRITES, Fake(), 10, threshold=0.5)
    assert sets["never"]["Q2_001"] == ["unsure question"]
    assert sets["gated"]["Q1_001"] == ["sure question"]  # confident: question alone
    assert sets["gated"]["Q2_001"] == sets["always"]["Q2_001"] and len(sets["always"]["Q2_001"]) == 5


def test_evaluation_and_report():
    gold = {q["qid"]: {"qid": q["qid"], "gold_doc_id": "g", "gold_chunk_ids": ["g::0"]} for q in QUESTIONS}
    backends = {"bm25_backend": Fake(), "dense_backend": Fake()}
    result = evaluate(QUESTIONS, gold, REWRITES, backends, {"retrieve_k": 10, "rrf_k": 60}, 0.5, "bm25")
    assert result["rewrite_share"] == 0.5
    assert [row["MRR"] for row in result["rows"]["never"]] == [1.0, 0.0]
    assert [row["MRR"] for row in result["rows"]["gated"]] == [1.0, 1.0]  # the rewrites rescue Q2
    report = report_markdown({("full", "bm25"): result}, QUESTIONS, 10, {"thresholds": {"bm25": 0.5}})
    assert "| full | bm25 |" in report and "50% |" in report
