"""Structured query-rewrite prompt (role ``rewrite``, mode ``structured``; R6).

Besides the four typed rewrites of the ``multi`` mode, the model states what every rewrite must
keep: the question's entities and constraints (copied verbatim), its abbreviations with their long
forms, and the expected answer type. The guard then checks each rewrite against these fields
(``guard.structured``) instead of guessing critical tokens with patterns.

The reply must be a JSON object with the keys entities, abbreviations, answer_type, constraints
and rewrites (itself an object with bm25_lexical, dense_semantic, abbreviation_expanded,
minimal_entity_locked).
"""

from __future__ import annotations

from rag_thesis.llm.types import ChatMessage

VERSION = "rewrite_structured_v1"

SYSTEM = (
    "You rewrite biomedical questions for document retrieval. First analyse the question, then write "
    "four retrieval rewrites. Return JSON only, with exactly these keys:\n"
    '- "entities": the genes, proteins, diseases, drugs, organisms, procedures and other named things in '
    "the question, each copied verbatim from the question;\n"
    '- "abbreviations": a list of {"short": ..., "long": ...} for each abbreviation in the question, '
    "with its long form (an empty list if there is none);\n"
    '- "answer_type": the kind of answer asked for, in a few words (e.g. "gene", "disease", "number", '
    '"list of drugs", "yes/no");\n'
    '- "constraints": phrases of the question that restrict the answer (organism, negation, condition, '
    "time, quantity), each copied verbatim; an empty list if there is none;\n"
    '- "rewrites": an object with the keys bm25_lexical, dense_semantic, abbreviation_expanded, '
    "minimal_entity_locked, in this order.\n"
    "Every rewrite must keep every entity and every constraint (an abbreviation may be replaced by its "
    "long form), must ask for the same answer type, and must not add any fact, name or answer that the "
    "question does not contain."
)


def build_structured_rewrite_messages(query_original: str) -> list[ChatMessage]:
    return [ChatMessage("system", SYSTEM), ChatMessage("user", f"Question: {query_original}")]
