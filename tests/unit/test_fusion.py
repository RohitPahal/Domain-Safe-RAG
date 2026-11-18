"""pipeline/fusion.py: plain and weighted Reciprocal Rank Fusion of several queries (R2)."""

from __future__ import annotations

import pytest

from rag_thesis.evaluation.retrieval import evaluate_retriever
from rag_thesis.pipeline.fusion import merge_multi_results, retrieve_queries


def _hits(*chunk_ids: str) -> list[dict]:
    return [
        {"chunk_id": cid, "doc_id": cid.split("::")[0], "score": 1.0, "rank": rank, "sources": ["bm25"]}
        for rank, cid in enumerate(chunk_ids, start=1)
    ]


# The question finds a::0 first; both rewrites prefer b::0.
PER_QUERY = [("question", _hits("a::0", "b::0")), ("rewrite 1", _hits("b::0", "c::0")), ("rewrite 2", _hits("b::0"))]


def test_plain_rrf_sums_reciprocal_ranks():
    fused = merge_multi_results(PER_QUERY, out_k=3, k_rrf=60)
    assert [r["chunk_id"] for r in fused] == ["b::0", "a::0", "c::0"]
    assert fused[0]["score"] == pytest.approx(1 / 62 + 1 / 61 + 1 / 61)
    assert fused[0]["queries"] == ["question", "rewrite 1", "rewrite 2"]


def test_weights_of_one_change_nothing():
    plain = merge_multi_results(PER_QUERY, out_k=3, k_rrf=60)
    assert merge_multi_results(PER_QUERY, out_k=3, k_rrf=60, weights={"question": 1.0, "rewrite 1": 1.0}) == plain


def test_a_heavier_question_can_win():
    # The question alone finds a::0; two rewrites find b::0. Plain RRF: b (2/61) beats a (1/61).
    per_query = [("question", _hits("a::0")), ("rewrite 1", _hits("b::0")), ("rewrite 2", _hits("b::0"))]
    assert merge_multi_results(per_query, out_k=2, k_rrf=60)[0]["chunk_id"] == "b::0"
    fused = merge_multi_results(per_query, out_k=2, k_rrf=60, weights={"question": 3.0})
    assert [r["chunk_id"] for r in fused] == ["a::0", "b::0"]  # 3/61 > 2/61
    assert fused[0]["score"] == pytest.approx(3 / 61)
    with pytest.raises(ValueError, match="non-negative"):
        merge_multi_results(per_query, out_k=2, k_rrf=60, weights={"question": -1.0})


class Fake:
    name = "bm25"

    def retrieve(self, query, k):
        return {"q": _hits("a::0"), "r": _hits("b::0"), "s": _hits("b::0")}[query][:k]


def test_weights_reach_retrieval_and_evaluation():
    plain = retrieve_queries(Fake(), ["q", "r", "s"], 2, retriever_mode="bm25", k_rrf=60)
    weighted = retrieve_queries(Fake(), ["q", "r", "s"], 2, retriever_mode="bm25", k_rrf=60, weights={"q": 3.0})
    assert (plain[0]["chunk_id"], weighted[0]["chunk_id"]) == ("b::0", "a::0")
    question = {"qid": "Q_001", "question": "q", "doc_id": "a", "answers": []}
    gold = {"Q_001": {"qid": "Q_001", "gold_doc_id": "a", "gold_chunk_ids": ["a::0"]}}
    common = dict(retriever="bm25", questions=[question], gold_by_qid=gold, config={"rrf_k": 60}, eval_k_list=[1])
    queries = {"Q_001": ["q", "r", "s"]}
    unweighted, _ = evaluate_retriever(bm25_backend=Fake(), queries_by_qid=queries, **common)
    heavy, _ = evaluate_retriever(
        bm25_backend=Fake(), queries_by_qid=queries, weights_by_qid={"Q_001": {"q": 3.0}}, **common
    )
    assert (unweighted["Hit@1"], heavy["Hit@1"]) == (0.0, 1.0)
