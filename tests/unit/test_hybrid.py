from __future__ import annotations

from typing import Any

from rag_thesis.retrieval.hybrid import HybridRetriever, rrf_fuse


class FakeBackend:
    def __init__(self, results_by_query: dict[str, list[dict[str, Any]]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        self.calls.append((query, k))
        results = self.results_by_query.get(query, [])
        return [dict(result) for result in results[:k]]


def test_retrieve_hybrid_returns_unique_ids_and_accurate_sources() -> None:
    query = "cat dog"
    bm25_backend = FakeBackend(
        {
            query: [
                {"chunk_id": "chunk-a", "doc_id": "doc-a", "score": 3.0, "rank": 1},
                {"chunk_id": "chunk-b", "doc_id": "doc-b", "score": 2.0, "rank": 2},
                {"chunk_id": "chunk-c", "doc_id": "doc-c", "score": 1.0, "rank": 3},
            ]
        }
    )
    dense_backend = FakeBackend(
        {
            query: [
                {"chunk_id": "chunk-d", "doc_id": "doc-d", "score": 0.9, "rank": 1},
                {"chunk_id": "chunk-b", "doc_id": "doc-b", "score": 0.8, "rank": 2},
                {"chunk_id": "chunk-e", "doc_id": "doc-e", "score": 0.7, "rank": 3},
            ]
        }
    )

    results = HybridRetriever(bm25=bm25_backend, dense=dense_backend, k_rrf=60).retrieve(query, k=4)

    assert bm25_backend.calls == [(query, 4)]
    assert dense_backend.calls == [(query, 4)]
    assert len(results) == 4
    assert len({result["chunk_id"] for result in results}) == 4
    assert [result["rank"] for result in results] == [1, 2, 3, 4]
    assert all(result["doc_id"] for result in results)
    assert results[0]["chunk_id"] == "chunk-b"
    assert results[0]["sources"] == "both"
    assert results[1]["chunk_id"] == "chunk-a"
    assert results[1]["sources"] == "bm25"
    assert results[2]["chunk_id"] == "chunk-d"
    assert results[2]["sources"] == "dense"


def test_retrieve_hybrid_differs_from_bm25_only_for_toy_query() -> None:
    query = "cat dog"
    bm25_results = [
        {"chunk_id": "chunk-a", "doc_id": "doc-a", "score": 3.0, "rank": 1},
        {"chunk_id": "chunk-b", "doc_id": "doc-b", "score": 2.0, "rank": 2},
        {"chunk_id": "chunk-c", "doc_id": "doc-c", "score": 1.0, "rank": 3},
    ]
    dense_results = [
        {"chunk_id": "chunk-d", "doc_id": "doc-d", "score": 0.9, "rank": 1},
        {"chunk_id": "chunk-b", "doc_id": "doc-b", "score": 0.8, "rank": 2},
        {"chunk_id": "chunk-e", "doc_id": "doc-e", "score": 0.7, "rank": 3},
    ]
    bm25_backend = FakeBackend({query: bm25_results})
    dense_backend = FakeBackend({query: dense_results})

    hybrid_results = HybridRetriever(bm25=bm25_backend, dense=dense_backend, k_rrf=60).retrieve(query, k=3)

    assert [result["chunk_id"] for result in hybrid_results] != [result["chunk_id"] for result in bm25_results[:3]]
    assert [result["chunk_id"] for result in hybrid_results] == ["chunk-b", "chunk-a", "chunk-d"]


def test_rrf_fuse_is_deterministic_under_ties() -> None:
    list_a = [
        {"chunk_id": "chunk-b", "doc_id": "doc-b", "score": 1.0, "rank": 1},
        {"chunk_id": "chunk-d", "doc_id": "doc-d", "score": 0.5, "rank": 2},
    ]
    list_b = [
        {"chunk_id": "chunk-a", "doc_id": "doc-a", "score": 1.0, "rank": 1},
        {"chunk_id": "chunk-c", "doc_id": "doc-c", "score": 0.5, "rank": 2},
    ]

    results = rrf_fuse(list_a, list_b, k_rrf=60, out_k=4)

    assert [result["chunk_id"] for result in results] == ["chunk-a", "chunk-b", "chunk-c", "chunk-d"]
    assert [result["rank"] for result in results] == [1, 2, 3, 4]


def test_rrf_fuse_uses_explicit_ranks_merges_duplicates_and_respects_out_k() -> None:
    list_a = [
        {"chunk_id": "chunk-z", "doc_id": "doc-z", "score": 0.2, "rank": 3},
        {"chunk_id": "chunk-a", "doc_id": "doc-a", "score": 0.9, "rank": 1},
        {"chunk_id": "chunk-a", "doc_id": "doc-a", "score": 0.1, "rank": 4},
    ]
    list_b = [
        {"chunk_id": "chunk-b", "doc_id": "doc-b", "score": 0.8, "rank": 1},
        {"chunk_id": "chunk-c", "doc_id": "doc-c", "score": 0.7, "rank": 2},
    ]

    results = rrf_fuse(list_a, list_b, k_rrf=60, out_k=3)

    assert len(results) == 3
    assert [result["chunk_id"] for result in results] == ["chunk-a", "chunk-b", "chunk-c"]
    assert [result["sources"] for result in results] == ["bm25", "dense", "dense"]
    assert all(result["chunk_id"] != "chunk-z" for result in results)
