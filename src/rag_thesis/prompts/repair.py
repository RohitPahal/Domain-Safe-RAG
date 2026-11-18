"""Guard-repair prompt (role ``repair``): fix an unsafe rewrite so it keeps the critical tokens.

The reply is plain text: the repaired query only.
"""

from __future__ import annotations

from rag_thesis.llm.types import ChatMessage

VERSION = "repair_v1"


def build_repair_messages(
    *, original_query: str, candidate_rewrite: str, critical_tokens: list[str]
) -> list[ChatMessage]:
    critical_tokens_text = ", ".join(critical_tokens) if critical_tokens else "[none]"
    return [
        ChatMessage(
            "system",
            "Repair the rewritten biomedical query so it preserves the original intent and preserves the "
            "critical tokens exactly. Do not add new numbers, ids, dates, quoted phrases, or negation "
            "constraints. Return only the repaired query text.",
        ),
        ChatMessage(
            "user",
            f"Original query: {original_query}\n"
            f"Candidate rewrite: {candidate_rewrite}\n"
            f"Critical tokens to preserve exactly: {critical_tokens_text}",
        ),
    ]
