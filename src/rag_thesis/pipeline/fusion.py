"""Retrieval results of a query: normalise each list, then fuse several queries with RRF.

``retrieve_queries`` is the one implementation of "retrieve with a question's query set", used by
the pipeline's retrieve step and by ``eval-retrieval`` when it replays a run's queries.
"""

from __future__ import annotations

from typing import Any

from rag_thesis.retrieval.base import Retriever


def unique_in_order(values: list[str]) -> list[str]:
    """``values`` stripped, without empty strings and without repeats, in their first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def retrieve_queries(
    retriever: Retriever,
    queries: list[str],
    k: int,
    *,
    retriever_mode: str,
    k_rrf: int,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Top-``k`` for each query; a single query keeps its own ranking, several are fused with RRF
    (weighted by ``weights``, query text -> weight, if given)."""
    per_query = [(query, normalize_results(retriever.retrieve(query, k), retriever_mode, query)) for query in queries]
    if len(per_query) == 1:
        return per_query[0][1][:k]
    return merge_multi_results(per_query, out_k=k, k_rrf=k_rrf, weights=weights)


def normalize_results(results: list[dict[str, Any]], retriever_mode: str, rewrite_query: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for result in results:
        chunk_id = result.get("chunk_id")
        doc_id = result.get("doc_id")
        rank = result.get("rank")
        score = result.get("score")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("retrieval results contain an invalid chunk_id.")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(f"retrieval result for {chunk_id} is missing a valid doc_id.")
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError(f"retrieval result for {chunk_id} is missing a valid rank.")
        if not isinstance(score, (int, float)):
            raise ValueError(f"retrieval result for {chunk_id} is missing a valid score.")
        normalized.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "score": float(score),
                "rank": rank,
                "sources": _normalize_sources(result, retriever_mode),
                "queries": [rewrite_query],
            }
        )
    return normalized


def _normalize_sources(result: dict[str, Any], retriever_mode: str) -> list[str]:
    raw_sources = result.get("sources")
    if isinstance(raw_sources, list):
        return sorted({str(item) for item in raw_sources if str(item).strip()})
    if isinstance(raw_sources, str) and raw_sources.strip():
        return ["bm25", "dense"] if raw_sources == "both" else [raw_sources]
    source = result.get("source")
    if isinstance(source, str) and source.strip():
        return [source]
    return ["bm25", "dense"] if retriever_mode == "hybrid" else [retriever_mode]


def merge_multi_results(
    per_rewrite: list[tuple[str, list[dict[str, Any]]]],
    *,
    out_k: int,
    k_rrf: int,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion of several queries' result lists: a chunk scores the sum over the
    lists of ``weight / (k_rrf + rank)``. ``weights`` maps a query's text to its weight (R2, e.g.
    the original question above its rewrites); a query without one weighs 1.0, so without weights
    this is plain RRF. Ties break on the best rank, then on the chunk id."""
    weights = weights or {}
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("RRF weights must be non-negative.")
    fused: dict[str, dict[str, Any]] = {}
    for query, results in per_rewrite:
        weight = float(weights.get(query, 1.0))
        for result in results:
            entry = fused.setdefault(
                result["chunk_id"],
                {
                    "chunk_id": result["chunk_id"],
                    "doc_id": result["doc_id"],
                    "score": 0.0,
                    "_best_rank": result["rank"],
                    "_sources": set(),
                    "_queries": [],
                },
            )
            if entry["doc_id"] != result["doc_id"]:
                raise ValueError(f"Conflicting doc_id values found for chunk_id {result['chunk_id']}.")
            entry["score"] += weight / float(k_rrf + result["rank"])
            entry["_best_rank"] = min(int(entry["_best_rank"]), int(result["rank"]))
            entry["_sources"].update(result.get("sources", []))
            if query not in entry["_queries"]:
                entry["_queries"].append(query)
    ranked = sorted(
        fused.values(), key=lambda item: (-float(item["score"]), int(item["_best_rank"]), str(item["chunk_id"]))
    )
    final_results: list[dict[str, Any]] = []
    for rank, result in enumerate(ranked[:out_k], start=1):
        final_results.append(
            {
                "chunk_id": result["chunk_id"],
                "doc_id": result["doc_id"],
                "score": float(result["score"]),
                "rank": rank,
                "sources": sorted(result["_sources"]),
                "queries": list(result["_queries"]),
            }
        )
    return final_results
