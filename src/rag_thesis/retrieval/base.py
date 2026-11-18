"""The interface every retriever follows.

A retriever turns a query into its top-``k`` chunks, best first. Each result is a dict with:

- ``chunk_id`` (str) and ``doc_id`` (str): which chunk, from which document;
- ``score`` (float): higher is better (scale depends on the retriever);
- ``rank`` (int): 1-based position;
- ``sources`` (hybrid only): which retrievers found the chunk.

Implementations: ``BM25Index`` (bm25.py), ``ChromaDenseIndex`` (chroma_store.py),
``HybridRetriever`` (hybrid.py). Code that *uses* retrieval gets one from
``factory.make_retriever`` and only calls ``retrieve``, so adding a new backend (a reranker,
another vector store) means writing one new class, not changing its callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Retriever(ABC):
    """Anything that can answer ``retrieve(query, k)``."""

    #: Short identifier, e.g. "bm25", "dense", "hybrid".
    name: str = "retriever"

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        """Return the top-``k`` chunks for ``query``, best first ([] if ``k <= 0`` or the query is blank)."""
