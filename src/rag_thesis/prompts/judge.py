"""LLM-judge prompt (role ``judge``): strict, reference-guided grading of one predicted answer.

The reply must be one JSON object with overall_verdict, overall_reason, faithfulness_label,
faithfulness_score, relevance_label, relevance_score, supported_text, unsupported_text.
"""

from __future__ import annotations

import json

from rag_thesis.llm.types import ChatMessage

VERSION = "judge_v2"

SYSTEM_PROMPT = (
    "You are a strict biomedical QA judge.\n"
    "Use only the question, gold answers, predicted answer, and evidence snippets.\n"
    "Do not use outside knowledge.\n"
    "Return exactly one JSON object with these keys and no prose outside the JSON:\n"
    "overall_verdict, overall_reason, faithfulness_label, faithfulness_score, "
    "relevance_label, relevance_score, supported_text, unsupported_text.\n"
    "Allowed overall_verdict values: CORRECT, PARTIALLY_CORRECT, INCORRECT, UNSUPPORTED.\n"
    "Allowed faithfulness_label values: FAITHFUL, PARTIALLY_FAITHFUL, UNFAITHFUL.\n"
    "Allowed relevance_label values: RELEVANT, PARTIALLY_RELEVANT, IRRELEVANT.\n"
    "Exact score mapping:\n"
    "- FAITHFUL = 1.0\n"
    "- PARTIALLY_FAITHFUL = 0.5\n"
    "- UNFAITHFUL = 0.0\n"
    "- RELEVANT = 1.0\n"
    "- PARTIALLY_RELEVANT = 0.5\n"
    "- IRRELEVANT = 0.0\n"
    "Overall rubric:\n"
    "- CORRECT: semantically matches an acceptable gold answer, including clear alias or abbreviation "
    "equivalence.\n"
    "- PARTIALLY_CORRECT: close but incomplete, too broad, too specific, or answer type only partly matches.\n"
    "- INCORRECT: wrong answer.\n"
    "- UNSUPPORTED: answer is not justified by the evidence, is speculative, or should not have been answered from "
    "the evidence shown.\n"
    "Faithfulness rubric, judged only against the retrieved evidence:\n"
    "- FAITHFUL: all material claims in the predicted answer are supported by the evidence.\n"
    "- PARTIALLY_FAITHFUL: the core answer is supported, but there are extra unsupported or overstated details.\n"
    "- UNFAITHFUL: the key answer claim is unsupported by or contradicted by the evidence.\n"
    "Relevance rubric:\n"
    "- RELEVANT: directly answers the question.\n"
    "- PARTIALLY_RELEVANT: on-topic but incomplete, vague, or only partly answers the question.\n"
    "- IRRELEVANT: does not actually answer the question or is mostly off-topic.\n"
    "supported_text must be the minimal part of the predicted answer grounded in the evidence.\n"
    "unsupported_text must be the minimal unsupported or contradicted part of the predicted answer.\n"
    "Always include supported_text and unsupported_text, even when they are empty strings.\n"
    "Keep overall_reason to one short sentence.\n"
    "If predicted_answer is empty or NOT_FOUND, return INCORRECT when the evidence explicitly supports a gold "
    "answer; otherwise return UNSUPPORTED."
)


def build_judge_messages(
    *,
    question: str,
    predicted_answer: str,
    gold_answers: list[str],
    evidence_chunks: list[dict[str, str]],
) -> list[ChatMessage]:
    """``evidence_chunks``: dicts with ``chunk_id``, ``source`` (citation / retrieved) and ``text``."""
    gold_block = json.dumps(gold_answers, ensure_ascii=False)
    evidence_lines: list[str] = []
    for index, chunk in enumerate(evidence_chunks, start=1):
        evidence_lines.append(f"[{index}] chunk_id={chunk['chunk_id']} source={chunk['source']}\n{chunk['text']}")
    evidence_block = "\n\n".join(evidence_lines) if evidence_lines else "[none]"

    user_content = (
        f"prompt_version: {VERSION}\n"
        f"question: {question}\n"
        f"predicted_answer: {predicted_answer or '[empty]'}\n"
        f"gold_answers: {gold_block}\n"
        f"evidence_snippets:\n{evidence_block}\n"
    )
    return [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", user_content)]
