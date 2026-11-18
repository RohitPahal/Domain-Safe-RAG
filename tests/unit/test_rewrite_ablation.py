"""tools/rewrite_ablation.py: query-set variants, memoised retrieval, cache reading (fakes only)."""

from __future__ import annotations

import json

import pytest
from tools.rewrite_ablation import query_sets, routed_rows, summarize

from rag_thesis.retrieval.memo import MemoRetriever
from rag_thesis.rewrite.rewriter import MULTI_REWRITE_ORDER, load_cached_rewrites

QUESTION = {"qid": "Q_001", "question": "What is ATP?", "doc_id": "d", "answers": [{"text": "x", "answer_start": 0}]}
REWRITES = {
    "Q_001": dict(zip(MULTI_REWRITE_ORDER, ["atp", "ATP meaning", "adenosine triphosphate", "atp"], strict=True))
}


def test_query_sets_cover_r1_r7_and_single_types():
    sets = query_sets([QUESTION], REWRITES, guard_on={"Q_001": ["atp", "What is ATP?"]})
    q = {name: by_qid["Q_001"] for name, by_qid in sets.items()}
    assert q["question"] == ["What is ATP?"]
    assert q["rewrites"] == ["atp", "ATP meaning", "adenosine triphosphate"]  # de-duplicated like the pipeline
    assert q["question+rewrites"] == ["What is ATP?", "atp", "ATP meaning", "adenosine triphosphate"]
    assert q["question+guard-on"] == ["What is ATP?", "atp"]
    assert q["without-abbreviation_expanded"] == ["atp", "ATP meaning"]
    assert q["without-bm25_lexical"] == ["ATP meaning", "adenosine triphosphate", "atp"]  # the twin stays
    assert q["only-dense_semantic"] == ["ATP meaning"]
    assert len(sets) == 5 + 2 * len(MULTI_REWRITE_ORDER)


def test_memo_retriever_asks_once_per_query_and_depth():
    calls = []

    class Inner:
        name = "bm25"

        def retrieve(self, query, k):
            calls.append((query, k))
            return [{"chunk_id": "c::0", "doc_id": "c", "score": 1.0, "rank": 1}]

    memo = MemoRetriever(Inner())
    first = memo.retrieve("a", 10)
    first[0]["rank"] = 99  # callers can't change the remembered result
    assert memo.retrieve("a", 10)[0]["rank"] == 1
    memo.retrieve("a", 20)
    assert calls == [("a", 10), ("a", 20)] and memo.name == "bm25"


def test_cached_rewrites_reads_multi_records_and_requires_every_qid(tmp_path):
    path = tmp_path / "rewrites.jsonl"
    records = [
        {"cache_key": "k1", "qid": "Q_001", "rewrite_mode": "multi", "rewrites": ["a", "b", "c", "d"]},
        {"cache_key": "k2", "qid": "Q_001", "rewrite_mode": "single", "rewrites": ["s"]},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    assert load_cached_rewrites(path, ["Q_001"]) == {"Q_001": dict(zip(MULTI_REWRITE_ORDER, "abcd", strict=True))}
    with pytest.raises(ValueError, match="no cached multi rewrites"):
        load_cached_rewrites(path, ["Q_001", "Q_002"])


def test_summarize_compares_leave_one_out_with_the_full_set_and_the_rest_with_the_question():
    def rows(mrr):
        return [{"MRR": value, "Hit@10": float(value > 0)} for value in mrr]

    data = {}
    for variant, mrr in (("question", [0.5, 1.0]), ("rewrites", [1.0, 1.0]), ("without-bm25_lexical", [0.5, 0.5])):
        for gold in ("question", "strict"):
            data[("full", "bm25", variant, gold)] = rows(mrr)
    summary = {row["variant"]: row for row in summarize(data, ["Q1", "Q2"], 10)}
    assert summary["question"]["diff_question_mrr"] is None
    assert summary["rewrites"]["reference"] == "question" and summary["rewrites"]["diff_question_mrr"][0] == 0.25
    assert summary["without-bm25_lexical"]["reference"] == "rewrites"
    assert summary["without-bm25_lexical"]["diff_question_mrr"][0] == -0.5


class Recording:
    """Answers every query with one chunk named after itself, and records the queries it saw."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.seen: list[str] = []

    def retrieve(self, query, k):
        self.seen.append(query)
        return [{"chunk_id": f"{self.name}::0", "doc_id": self.name, "score": 1.0, "rank": 1}]


def test_routing_sends_each_rewrite_type_to_its_retriever():
    bm25, dense = Recording("b"), Recording("d")
    golds = {"question": {"Q_001": {"gold_chunk_ids": ["d::0"]}}, "strict": {"Q_001": {"gold_chunk_ids": ["x::0"]}}}
    rows = routed_rows([QUESTION], golds, REWRITES, {"bm25_backend": bm25, "dense_backend": dense}, 10, 60, True)
    # question -> hybrid (both); bm25_lexical "atp" -> BM25 only; dense_semantic -> dense only;
    # abbreviation_expanded -> hybrid; minimal_entity_locked repeats "atp" and is dropped.
    assert bm25.seen == ["What is ATP?", "atp", "adenosine triphosphate"]
    assert dense.seen == ["What is ATP?", "ATP meaning", "adenosine triphosphate"]
    assert rows["question"][0]["Hit@10"] == 1.0 and rows["strict"][0]["MRR"] == 0.0


def test_dictionary_query_sets_are_added_when_expansions_are_given():
    sets = query_sets(
        [QUESTION], REWRITES, {"Q_001": ["atp"]}, expanded={"Q_001": "What is ATP (adenosine triphosphate)?"}
    )
    assert sets["dictionary"]["Q_001"] == ["What is ATP (adenosine triphosphate)?"]
    assert sets["question+dictionary"]["Q_001"] == ["What is ATP?", "What is ATP (adenosine triphosphate)?"]
    unchanged = query_sets([QUESTION], REWRITES, {"Q_001": ["atp"]}, expanded={"Q_001": "What is ATP?"})
    assert unchanged["question+dictionary"]["Q_001"] == ["What is ATP?"]  # nothing to expand: same as the question


def test_structured_sets_keep_what_the_structured_guard_lets_through():
    from tools.rewrite_ablation import structured_sets

    from rag_thesis.guard.aliases import AbbreviationDictionary
    from rag_thesis.rewrite.rewriter import StructuredRewrite

    question = {"qid": "Q_001", "question": "What is ATP?", "doc_id": "d", "answers": [{"text": "energy"}]}
    structure = StructuredRewrite.model_validate(
        {
            "entities": ["ATP", "BRCA1"],  # BRCA1 is invented
            "abbreviations": [{"short": "ATP", "long": "adenosine triphosphate"}],
            "answer_type": "molecule",
            "constraints": [],
            "rewrites": dict(
                zip(
                    MULTI_REWRITE_ORDER,
                    ["ATP definition", "What is adenosine triphosphate?", "ATP and TP53", "ATP energy carrier"],
                    strict=True,
                )
            ),
        }
    )
    sets, notes = structured_sets([question], {"Q_001": structure}, AbbreviationDictionary({}), REWRITES)
    assert sets["Q_001"]["all"] == structure.rewrite_list()
    assert sets["Q_001"]["guarded"] == ["ATP definition", "What is adenosine triphosphate?", "ATP energy carrier"]
    assert "blocks 1 of 4 structured rewrites (by check: added_entities 1)" in notes[1]
    assert "1 structure fields are not in their question" in notes[1]
    assert "1 of 4 structured vs 0 of 4 multi" in notes[1]  # "ATP energy carrier" contains the answer
    variants = query_sets([QUESTION], REWRITES, {"Q_001": ["atp"]}, structured=sets)
    assert variants["question+structured-guarded"]["Q_001"][0] == "What is ATP?"
    assert variants["structured"]["Q_001"] == structure.rewrite_list()
