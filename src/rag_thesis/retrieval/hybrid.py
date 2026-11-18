"""Hybrid retrieval: Reciprocal Rank Fusion (RRF) of a BM25 and a dense retriever.

RRF score of a chunk = sum over the input lists of 1 / (k_rrf + rank). It uses only ranks,
so BM25 and cosine scores (different scales) never need to be normalised against each other.
"""

from __future__ import annotations

from typing import Any

from rag_thesis.retrieval.base import Retriever


def rrf_fuse(
    list_a: list[dict[str, Any]],
    list_b: list[dict[str, Any]],
    k_rrf: int = 60,
    out_k: int = 10,
) -> list[dict[str, Any]]:
    if out_k <= 0:
        return []
    if k_rrf < 0:
        raise ValueError("k_rrf must be non-negative.")

    fused: dict[str, dict[str, Any]] = {}

    for source_name, results in (("bm25", list_a), ("dense", list_b)):
        normalized_results = _normalize_source_results(results, source_name)
        for result in normalized_results:
            chunk_id = result["chunk_id"]
            doc_id = result["doc_id"]
            entry = fused.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "score": 0.0,
                    "_sources": set(),
                },
            )
            if entry["doc_id"] != doc_id:
                raise ValueError(f"Conflicting doc_id values found for chunk_id {chunk_id}.")

            entry["score"] += 1.0 / float(k_rrf + result["rank"])
            entry["_sources"].add(source_name)

    ranked = sorted(
        fused.values(),
        key=lambda item: (-float(item["score"]), str(item["chunk_id"])),
    )

    results: list[dict[str, Any]] = []
    for rank, item in enumerate(ranked[:out_k], start=1):
        sources = item["_sources"]
        if sources == {"bm25", "dense"}:
            source_label = "both"
        elif sources == {"bm25"}:
            source_label = "bm25"
        else:
            source_label = "dense"

        results.append(
            {
                "chunk_id": item["chunk_id"],
                "score": float(item["score"]),
                "rank": rank,
                "doc_id": item["doc_id"],
                "sources": source_label,
            }
        )

    return results


class HybridRetriever(Retriever):
    """Fuses the top-``k`` of a BM25 and a dense retriever with RRF and keeps the best ``k``.

    Each side retrieves ``k`` candidates, so the fused top-``k`` depends on ``k`` itself
    (retrieving 20 then cutting to 10 is not the same as retrieving 10).
    """

    name = "hybrid"

    def __init__(self, bm25: Retriever, dense: Retriever, k_rrf: int = 60) -> None:
        for label, backend in (("bm25", bm25), ("dense", dense)):
            if not callable(getattr(backend, "retrieve", None)):
                raise TypeError(f"{label} must define a callable retrieve(query, k) method.")
        self.bm25 = bm25
        self.dense = dense
        self.k_rrf = k_rrf

    def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        if k <= 0:
            return []
        bm25_results = self.bm25.retrieve(query, k)
        dense_results = self.dense.retrieve(query, k)
        return rrf_fuse(bm25_results, dense_results, k_rrf=self.k_rrf, out_k=k)


def _normalize_source_results(results: list[dict[str, Any]], source_name: str) -> list[dict[str, Any]]:
    best_by_chunk_id: dict[str, dict[str, Any]] = {}

    for result in results:
        chunk_id = result.get("chunk_id")
        doc_id = result.get("doc_id")
        rank = result.get("rank")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"{source_name} results contain an invalid chunk_id.")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(f"{source_name} results contain an invalid doc_id for chunk {chunk_id}.")
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"{source_name} results contain an invalid rank for chunk {chunk_id}.")

        existing = best_by_chunk_id.get(chunk_id)
        if existing is None:
            best_by_chunk_id[chunk_id] = {"chunk_id": chunk_id, "doc_id": doc_id, "rank": rank}
            continue

        if existing["doc_id"] != doc_id:
            raise ValueError(f"Conflicting doc_id values found for chunk_id {chunk_id}.")
        if rank < existing["rank"]:
            best_by_chunk_id[chunk_id] = {"chunk_id": chunk_id, "doc_id": doc_id, "rank": rank}

    return sorted(
        best_by_chunk_id.values(),
        key=lambda item: (int(item["rank"]), str(item["chunk_id"])),
    )
