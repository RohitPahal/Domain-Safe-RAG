"""The one place that turns a retriever mode ("bm25" / "dense" / "hybrid") into a retriever.

Both the pipeline and the retrieval evaluation call ``make_retriever``, so they can never
disagree about what a mode means. Index loading stays with the caller (indexes are loaded
once and shared, e.g. "hybrid" reuses the already loaded BM25 and dense indexes).
"""

from __future__ import annotations

from rag_thesis.retrieval.base import Retriever
from rag_thesis.retrieval.hybrid import HybridRetriever

RETRIEVER_MODES = ("bm25", "dense", "hybrid")


def make_retriever(
    mode: str,
    *,
    bm25: Retriever | None = None,
    dense: Retriever | None = None,
    k_rrf: int = 60,
) -> Retriever:
    """Return the retriever for ``mode``, built from already-loaded indexes.

    Raises ``ValueError`` for an unknown mode or when an index the mode needs is missing.
    """
    if mode not in RETRIEVER_MODES:
        raise ValueError(f"Unknown retriever mode '{mode}'. Choose one of: {', '.join(RETRIEVER_MODES)}.")
    if mode in ("bm25", "hybrid") and bm25 is None:
        raise ValueError(f"'{mode}' retrieval needs a BM25 index, but none was loaded.")
    if mode in ("dense", "hybrid") and dense is None:
        raise ValueError(f"'{mode}' retrieval needs a dense index, but none was loaded.")

    if mode == "bm25":
        return bm25
    if mode == "dense":
        return dense
    return HybridRetriever(bm25=bm25, dense=dense, k_rrf=k_rrf)
