"""A retriever wrapper that remembers results, for offline analyses that ask the same query often.

Ablations and tuning grids retrieve the same query texts many times (one question in many query
sets, one rewrite under many weights); results don't change between calls, so each (query, k) is
retrieved once. Callers get copies, so they can't alter what is remembered.
"""

from __future__ import annotations

from typing import Any

from rag_thesis.retrieval.base import Retriever


class MemoRetriever(Retriever):
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "retriever")
        self.results: dict[tuple[str, int], list[dict[str, Any]]] = {}

    def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        if (query, k) not in self.results:
            self.results[(query, k)] = self.inner.retrieve(query, k)
        return [dict(result) for result in self.results[(query, k)]]
