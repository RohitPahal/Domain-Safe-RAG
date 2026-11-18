"""The retriever interface (retrieval/base.py), HybridRetriever and make_retriever (factory.py)."""

from __future__ import annotations

import pytest

from rag_thesis.retrieval.base import Retriever
from rag_thesis.retrieval.bm25 import BM25Index, build_bm25
from rag_thesis.retrieval.chroma_store import ChromaDenseIndex
from rag_thesis.retrieval.factory import RETRIEVER_MODES, make_retriever
from rag_thesis.retrieval.hybrid import HybridRetriever


class FixedRetriever(Retriever):
    """A toy backend that always returns the same ranked chunk ids (stands in for a new backend)."""

    name = "fixed"

    def __init__(self, chunk_ids: list[str]) -> None:
        self.chunk_ids = chunk_ids
        self.calls = 0

    def retrieve(self, query: str, k: int) -> list[dict]:
        self.calls += 1
        return [
            {"chunk_id": cid, "doc_id": cid.split("::")[0], "score": 1.0 / rank, "rank": rank}
            for rank, cid in enumerate(self.chunk_ids[:k], start=1)
        ]


def test_every_backend_is_a_retriever():
    assert issubclass(BM25Index, Retriever) and BM25Index.name == "bm25"
    assert issubclass(ChromaDenseIndex, Retriever) and ChromaDenseIndex.name == "dense"
    assert issubclass(HybridRetriever, Retriever) and HybridRetriever.name == "hybrid"
    assert RETRIEVER_MODES == ("bm25", "dense", "hybrid")


def test_retriever_is_an_interface():
    with pytest.raises(TypeError):
        Retriever()  # abstract: retrieve() must be implemented

    class Incomplete(Retriever):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_factory_returns_the_loaded_index_for_single_modes():
    bm25, dense = FixedRetriever(["a::0"]), FixedRetriever(["b::0"])
    assert make_retriever("bm25", bm25=bm25) is bm25
    assert make_retriever("dense", dense=dense) is dense


def test_factory_builds_hybrid_from_both_indexes():
    bm25, dense = FixedRetriever(["a::0", "c::0"]), FixedRetriever(["b::0", "c::0"])
    hybrid = make_retriever("hybrid", bm25=bm25, dense=dense, k_rrf=60)
    assert isinstance(hybrid, HybridRetriever) and hybrid.k_rrf == 60
    results = hybrid.retrieve("q", 3)
    assert results[0]["chunk_id"] == "c::0"  # found by both, so it ranks first under RRF
    assert results[0]["sources"] == "both"


def test_factory_errors_are_clear():
    with pytest.raises(ValueError, match="Unknown retriever mode 'sparse'"):
        make_retriever("sparse", bm25=FixedRetriever([]))
    with pytest.raises(ValueError, match="needs a BM25 index"):
        make_retriever("bm25")
    with pytest.raises(ValueError, match="needs a dense index"):
        make_retriever("hybrid", bm25=FixedRetriever([]))


def test_hybrid_rejects_backends_without_retrieve_and_skips_work_for_k_0():
    with pytest.raises(TypeError, match="dense must define"):
        HybridRetriever(bm25=FixedRetriever([]), dense=object())
    bm25, dense = FixedRetriever(["a::0"]), FixedRetriever(["b::0"])
    assert HybridRetriever(bm25=bm25, dense=dense).retrieve("q", 0) == []
    assert bm25.calls == dense.calls == 0


def test_a_new_backend_plugs_in_without_changing_callers():
    """Extensibility: a real BM25 index and a brand-new backend combine through the same factory."""
    bm25 = build_bm25(
        [
            {"chunk_id": "d1::0", "doc_id": "d1", "text": "brca1 breast cancer gene"},
            {"chunk_id": "d2::0", "doc_id": "d2", "text": "insulin and diabetes"},
        ]
    )
    new_backend = FixedRetriever(["d1::0", "d2::0"])
    results = make_retriever("hybrid", bm25=bm25, dense=new_backend).retrieve("brca1 gene", 2)
    assert [r["chunk_id"] for r in results][0] == "d1::0"
    assert new_backend.calls == 1
