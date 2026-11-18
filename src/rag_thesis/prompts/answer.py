"""Answer-generation prompt (role ``answer``): the minimal exact answer span from retrieved chunks.

The reply is one line: the answer, or exactly ``NOT_FOUND_ANSWER`` when the chunks don't state it.
"""

from __future__ import annotations

from rag_thesis.llm.types import ChatMessage

VERSION = "answer_v1"
NOT_FOUND_ANSWER = "NOT_FOUND"


def build_answer_messages(question: str, contexts: list[str]) -> list[ChatMessage]:
    """``contexts`` are the chunk texts as ``"[chunk_id] text"`` lines."""
    return [
        ChatMessage(
            "system",
            "You answer biomedical factoid questions using only the provided retrieved chunks. "
            "Return exactly one line containing only the minimal exact answer span supported by the "
            "evidence. "
            "Return exactly the entity, type, binary choice, number, or mutation/defect asked by the "
            "question. "
            "If the question asks for a number, return only the number and its necessary unit if "
            "present. "
            "If the question asks for a mutation or defect, return the mutation or defect, not merely "
            "the associated gene name. "
            "If the question asks for a binary choice such as synonymous versus non-synonymous, return "
            "only that choice. "
            "Do not include explanations, reasoning, full sentences, labels, bullets, or question "
            "restatement. "
            "If the answer is not explicitly stated in the retrieved chunks, return exactly this token "
            f"and nothing else: {NOT_FOUND_ANSWER}.",
        ),
        ChatMessage(
            "user",
            f"Question: {question}\n\n"
            "Retrieved chunks:\n"
            + "\n\n".join(contexts)
            + "\n\nOutput requirement: reply with exactly one line containing either the minimal exact "
            f"answer span or this exact token: {NOT_FOUND_ANSWER}.",
        ),
    ]
