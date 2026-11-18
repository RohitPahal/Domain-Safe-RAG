"""How sure is a retrieval? The signal that gates rewriting (R5: rewrite only when unsure).

The retriever's own top-1 score: BM25's score, dense's cosine similarity or hybrid's RRF score, so
each retriever has its own scale and its own threshold. Measured on the dev split (full corpus,
original questions only) it predicts "no relevant chunk at rank 1" better than the
score margin to rank 2 and about as well as BM25-dense agreement, which would need both retrievers.
"""

from __future__ import annotations

from typing import Any


def top1_score(results: list[dict[str, Any]]) -> float | None:
    """The rank-1 score, or ``None`` when nothing was retrieved (treat as unsure)."""
    return float(results[0]["score"]) if results else None


def is_confident(results: list[dict[str, Any]], threshold: float) -> bool:
    """Confident when the top-1 score reaches ``threshold``; no results are never confident."""
    score = top1_score(results)
    return score is not None and score >= threshold
