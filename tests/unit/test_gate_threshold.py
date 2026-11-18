"""retrieval/confidence.py and tools/gate_threshold.py (R5): the signal and its dev threshold."""

from __future__ import annotations

import pytest
from tools.gate_threshold import auc_ci, measure, report_markdown, youden_threshold

from rag_thesis.retrieval.confidence import is_confident, top1_score


def test_top1_score_and_confidence():
    results = [{"score": 0.8}, {"score": 0.5}]
    assert top1_score(results) == 0.8 and top1_score([]) is None
    assert is_confident(results, 0.8) and not is_confident(results, 0.81)
    assert not is_confident([], -1.0)  # nothing retrieved: never confident


def test_youden_threshold_separates_low_scores_from_high():
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    failed = [True, True, False, False, False, False]
    best = youden_threshold(scores, failed)
    assert best["threshold"] == pytest.approx(0.25)  # rewrite the two low scores, keep the rest
    assert (best["sensitivity"], best["specificity"], best["youden_j"]) == (1.0, 1.0, 1.0)
    assert best["rewrite_share"] == pytest.approx(2 / 6)
    with pytest.raises(ValueError, match="both"):
        youden_threshold([0.1, 0.2], [True, True])


def test_auc_ci_brackets_the_auc():
    scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    failed = [True, True, False, True, False, False, True, False]
    value, low, high = auc_ci(scores, failed)
    assert low <= value <= high and 0.0 <= low and high <= 1.0


class Fake:
    name = "fake"

    def __init__(self, ranking):
        self.ranking = ranking

    def retrieve(self, query, k):
        chunk, score = self.ranking[query]
        return [{"chunk_id": chunk, "doc_id": chunk.split("::")[0], "score": score, "rank": 1}][:k]


def test_measure_and_report_per_retriever():
    questions = [{"qid": f"Q{i}_001", "question": f"q{i}"} for i in range(4)]
    gold = {q["qid"]: {"gold_chunk_ids": ["g::0"]} for q in questions}
    # sure and right vs unsure and wrong
    ranking = {"q0": ("g::0", 0.9), "q1": ("g::0", 0.8), "q2": ("x::0", 0.2), "q3": ("x::0", 0.1)}
    backends = {"bm25": Fake(ranking), "dense": Fake(ranking)}
    measured = measure(questions, gold, backends, {"retrieve_k": 10, "rrf_k": 60})
    assert measured["bm25"]["failed"] == 2 and measured["bm25"]["auc"][0] == 1.0
    assert measured["dense"]["threshold"] == pytest.approx(0.5)
    assert "| bm25 | 2/4 | 1.000" in report_markdown(measured, 4)
