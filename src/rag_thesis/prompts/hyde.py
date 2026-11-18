"""HyDE prompt (role ``rewrite``, mode ``hyde``; R8): a hypothetical passage to retrieve with.

Instead of rewriting the question, the model writes a short passage that could answer it, in the
style of the corpus (a biomedical abstract); its embedding is usually closer to real answering
passages than the question's is. The passage is the model's guess, so it can contain the answer
from the model's own knowledge: it is used only to retrieve, never shown as evidence, and the
evaluation reports how often it leaks a gold answer (tools/hyde_eval.py).

The reply must be a JSON object: ``{"passage": "..."}``.
"""

from __future__ import annotations

from rag_thesis.llm.types import ChatMessage

VERSION = "hyde_v1"

SYSTEM = (
    "Write a short hypothetical passage (2 to 4 sentences) in the style of a biomedical research "
    "abstract that would answer the user's question. Keep every gene, protein, disease, drug, organism, "
    "number and abbreviation of the question exactly as written. It is used only to search a document "
    "collection, so plausible wording matters more than certainty; do not mention that it is hypothetical. "
    'Return JSON only in the form {"passage": "..."}.'
)


def build_hyde_messages(query_original: str) -> list[ChatMessage]:
    return [ChatMessage("system", SYSTEM), ChatMessage("user", f"Question: {query_original}")]
