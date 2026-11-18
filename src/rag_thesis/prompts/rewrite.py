"""Query-rewrite prompt (role ``rewrite``): one rewrite (``single``) or four typed rewrites (``multi``).

The reply must be a JSON object: ``{"rewrite": ...}`` for single, or the four keys
bm25_lexical, dense_semantic, abbreviation_expanded, minimal_entity_locked for multi.
"""

from __future__ import annotations

from rag_thesis.llm.types import ChatMessage

VERSION = "rewrite_v1"


def build_rewrite_messages(query_original: str, rewrite_mode: str) -> list[ChatMessage]:
    if rewrite_mode == "single":
        return [
            ChatMessage(
                "system",
                "Rewrite the user question for retrieval. Preserve intent and do not add new constraints. "
                'Return JSON only in the form {"rewrite": "..."}.',
            ),
            ChatMessage("user", f"Question: {query_original}"),
        ]

    return [
        ChatMessage(
            "system",
            "Generate exactly four retrieval rewrites that preserve intent and do not add new constraints. "
            "Return JSON only with these keys in this order: "
            "bm25_lexical, dense_semantic, abbreviation_expanded, minimal_entity_locked.",
        ),
        ChatMessage("user", f"Question: {query_original}"),
    ]
