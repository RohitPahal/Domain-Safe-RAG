"""tools/tune_fusion_weights.py: question weight chosen on dev, reported on test (fakes only)."""

from __future__ import annotations

from tools.tune_fusion_weights import (
    GRID,
    question_and_rewrites,
    question_weights,
    report_markdown,
    score_on_test,
    tune,
)

from rag_thesis.rewrite.rewriter import MULTI_REWRITE_ORDER

QUESTIONS = [{"qid": f"Q{i}_001", "question": f"question {i}", "doc_id": "g", "answers": []} for i in range(3)]
REWRITES = {q["qid"]: {slot: f"{q['question']} {slot}" for slot in MULTI_REWRITE_ORDER} for q in QUESTIONS}
GOLD = {q["qid"]: {"qid": q["qid"], "gold_doc_id": "g", "gold_chunk_ids": ["g::0"]} for q in QUESTIONS}
CONFIG = {"retrieve_k": 10, "rrf_k": 60}


class QuestionFindsGold:
    """The question itself finds the gold chunk; every rewrite finds the same distractor."""

    name = "bm25"

    def retrieve(self, query, k):
        chunk = "g::0" if query.startswith("question") and query.split()[-1].isdigit() else "x::0"
        return [{"chunk_id": chunk, "doc_id": chunk[0], "score": 1.0, "rank": 1}]


BACKENDS = {"bm25_backend": QuestionFindsGold(), "dense_backend": QuestionFindsGold()}


def test_query_sets_and_weights():
    queries = question_and_rewrites(QUESTIONS[:1], REWRITES)
    assert queries["Q0_001"][0] == "question 0" and len(queries["Q0_001"]) == 5
    assert question_weights(QUESTIONS[:1], 3.0) == {"Q0_001": {"question 0": 3.0}}


def test_the_smallest_weight_that_lets_the_question_win_is_chosen():
    # Gold scores w/61, the distractor 4/61: w=4 ties and the tie-break (chunk id) puts gold first.
    tuned = tune(QUESTIONS, GOLD, REWRITES, BACKENDS, CONFIG)
    for retriever in ("bm25", "dense", "hybrid"):
        assert tuned[retriever]["chosen"] == 4.0
        assert tuned[retriever]["scores"][1.0] == 0.5 and tuned[retriever]["scores"][6.0] == 1.0
    rows = score_on_test(QUESTIONS, GOLD, REWRITES, BACKENDS, CONFIG, {r: 4.0 for r in ("bm25", "dense", "hybrid")})
    assert [row["retriever"] for row in rows] == ["bm25", "dense", "hybrid"]
    assert rows[0]["plain"][0] == 0.5 and rows[0]["weighted"][0] == 1.0 and rows[0]["weighted_vs_plain"][0] == 0.5
    report = report_markdown(tuned, {"full": rows}, n_dev=3, n_test=3)
    assert "| bm25 | " in report and f"w={GRID[-1]:g}" in report and "## Test qids, full corpus" in report
