"""Domain-safe guard for query rewrites.

Checks that an LLM rewrite keeps the question's critical tokens (ids, numbers, abbreviations,
...) and adds no new sensitive tokens or negation. An unsafe rewrite gets one LLM repair attempt;
if that fails or is still unsafe, the original question is used instead.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from rag_thesis.llm.base import ChatModel
from rag_thesis.llm.errors import LLMError
from rag_thesis.llm.types import ChatRequest
from rag_thesis.prompts.repair import build_repair_messages

LOGGER = logging.getLogger(__name__)


class EmptyRepairError(ValueError):
    """The repair model replied with nothing usable."""


# What a repair can legitimately fail with: a provider failure or an empty reply. Both fall back to
# the original question. Anything else, such as a code bug or a missing SDK, stops the run: it must
# never look like a failed repair in the run log.
EXPECTED_REPAIR_ERRORS = (LLMError, EmptyRepairError)

QUOTED_PATTERN = re.compile(r'"[^"\n]+"|\'[^\'\n]+\'')
DATE_PATTERN = re.compile(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b")
PERCENT_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%")
SLASH_PATTERN = re.compile(r"\b[A-Za-z0-9-]+(?:/[A-Za-z0-9-]+)+\b")
HYPHENATED_PATTERN = re.compile(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b")
CODE_PATTERN = re.compile(r"\b(?:[A-Z]{2,}[0-9]+[A-Za-z0-9]*|[A-Z]+[0-9]+|[a-z][A-Za-z]*[0-9]+[A-Za-z0-9]*)\b")
ALLCAPS_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
TITLECASE_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
DECIMAL_PATTERN = re.compile(r"\b\d+\.\d+\b")
NUMBER_PATTERN = re.compile(r"\b\d+\b")
NEGATION_PATTERNS = [
    re.compile(r"\bno\b", re.IGNORECASE),
    re.compile(r"\bnot\b", re.IGNORECASE),
    re.compile(r"\bwithout\b", re.IGNORECASE),
    re.compile(r"\bexcept\b", re.IGNORECASE),
    re.compile(r"\bexcluding\b", re.IGNORECASE),
    re.compile(r"\bexclude\b", re.IGNORECASE),
    re.compile(r"\babsence of\b", re.IGNORECASE),
    re.compile(r"\black of\b", re.IGNORECASE),
    re.compile(r"\bnegative for\b", re.IGNORECASE),
]
PATTERN_SPECS = [
    ("quoted", QUOTED_PATTERN),
    ("date", DATE_PATTERN),
    ("percent", PERCENT_PATTERN),
    ("slash", SLASH_PATTERN),
    ("hyphenated", HYPHENATED_PATTERN),
    ("code", CODE_PATTERN),
    ("allcaps", ALLCAPS_PATTERN),
    ("titlecase", TITLECASE_PATTERN),
    ("decimal", DECIMAL_PATTERN),
    ("number", NUMBER_PATTERN),
]
PATTERN_PRIORITY = {token_type: index for index, (token_type, _) in enumerate(PATTERN_SPECS)}
SENSITIVE_ADDED_TOKEN_TYPES = {
    "quoted",
    "date",
    "percent",
    "slash",
    "hyphenated",
    "code",
    "allcaps",
    "decimal",
    "number",
}


@dataclass(frozen=True)
class CriticalToken:
    text: str
    token_type: str
    start: int
    end: int


@dataclass(frozen=True)
class SafetyAssessment:
    is_safe: bool
    original_critical_tokens: list[str]
    rewrite_critical_tokens: list[str]
    missing_tokens: list[str]
    added_tokens: list[str]
    negation_added: bool
    reason_codes: list[str]
    rewrite_token_types: dict[str, str]


def extract_critical_tokens(text: str) -> list[CriticalToken]:
    candidates: list[CriticalToken] = []
    for token_type, pattern in PATTERN_SPECS:
        for match in pattern.finditer(text):
            candidates.append(
                CriticalToken(
                    text=match.group(0),
                    token_type=token_type,
                    start=match.start(),
                    end=match.end(),
                )
            )

    candidates.sort(
        key=lambda item: (
            item.start,
            PATTERN_PRIORITY[item.token_type],
            -(item.end - item.start),
            item.text,
        )
    )
    accepted: list[CriticalToken] = []
    seen_texts: set[str] = set()
    for candidate in candidates:
        if any(_overlaps(candidate, existing) for existing in accepted):
            continue
        if candidate.text in seen_texts:
            continue
        accepted.append(candidate)
        seen_texts.add(candidate.text)

    accepted.sort(key=lambda item: (item.start, item.end, item.text))
    return accepted


def validate_and_repair_rewrite(
    original_query: str,
    candidate_rewrite: str,
    repair_model: ChatModel | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return ``(query_to_use, report)`` for one candidate rewrite.

    ``final_action`` is one of ``pass`` / ``repaired`` / ``rejected`` / ``fallback_original``.
    The report's top-level fields (``is_safe``, ``reason_codes``, ``missing_tokens``, ...) describe
    the assessment of the **final** query. ``initial_assessment`` always describes the candidate
    exactly as the rewriter produced it, i.e. why it was flagged, which the final fields can't
    show once a repair succeeded. ``repair_error`` holds the reason when the repair call failed.

    ``repair_model`` is the ``repair`` role's ``ChatModel``; without one, an unsafe rewrite falls
    back to the original question (``REPAIR_UNAVAILABLE``).
    """
    normalized_original = original_query.strip()
    normalized_candidate = candidate_rewrite.strip()
    if not normalized_original:
        raise ValueError("original_query must be a non-empty string.")
    if not normalized_candidate:
        raise ValueError("candidate_rewrite must be a non-empty string.")

    initial_assessment = assess_rewrite(normalized_original, normalized_candidate)
    if initial_assessment.is_safe:
        return normalized_candidate, _build_guard_report(
            initial_assessment,
            initial_assessment=initial_assessment,
            repair_attempted=False,
            final_action="pass",
        )

    if repair_model is None:
        return normalized_original, _build_guard_report(
            initial_assessment,
            initial_assessment=initial_assessment,
            repair_attempted=False,
            final_action="fallback_original",
            extra_reason_codes=["REPAIR_UNAVAILABLE"],
        )

    try:
        repaired_query = _attempt_repair(
            original_query=normalized_original,
            candidate_rewrite=normalized_candidate,
            original_critical_tokens=initial_assessment.original_critical_tokens,
            repair_model=repair_model,
        )
    except EXPECTED_REPAIR_ERRORS as exc:
        repair_error = f"{type(exc).__name__}: {exc}"
        LOGGER.warning("Guard repair failed; using the original question. %s", repair_error)
        return normalized_original, _build_guard_report(
            initial_assessment,
            initial_assessment=initial_assessment,
            repair_attempted=True,
            final_action="fallback_original",
            extra_reason_codes=["REPAIR_FAILED"],
            repair_error=repair_error,
        )

    repaired_assessment = assess_rewrite(normalized_original, repaired_query)
    if repaired_assessment.is_safe:
        return repaired_query, _build_guard_report(
            repaired_assessment,
            initial_assessment=initial_assessment,
            repair_attempted=True,
            final_action="repaired",
        )

    return normalized_original, _build_guard_report(
        repaired_assessment,
        initial_assessment=initial_assessment,
        repair_attempted=True,
        final_action="rejected",
        extra_reason_codes=["REPAIR_UNSAFE"],
    )


def assess_rewrite(original_query: str, rewrite_query: str) -> SafetyAssessment:
    """The guard's check of one rewrite, without repair: critical tokens missing or sensitive ones
    added, and negation added. Public so analysis tools can recompute the guard's first verdict."""
    original_tokens = extract_critical_tokens(original_query)
    rewrite_tokens = extract_critical_tokens(rewrite_query)
    original_texts = [token.text for token in original_tokens]
    rewrite_texts = [token.text for token in rewrite_tokens]
    rewrite_token_types = {token.text: token.token_type for token in rewrite_tokens}

    rewrite_text_set = set(rewrite_texts)
    original_text_set = set(original_texts)
    missing_tokens = [token for token in original_texts if token not in rewrite_text_set]
    added_tokens = [token for token in rewrite_texts if token not in original_text_set]
    negation_added = not _has_negation(original_query) and _has_negation(rewrite_query)

    reason_codes: list[str] = []
    if missing_tokens:
        reason_codes.append("MISSING_CRITICAL_TOKENS")

    added_sensitive_tokens = [
        token for token in added_tokens if rewrite_token_types.get(token) in SENSITIVE_ADDED_TOKEN_TYPES
    ]
    if added_sensitive_tokens:
        reason_codes.append("ADDED_SENSITIVE_TOKENS")

    if negation_added:
        reason_codes.append("NEGATION_ADDED")

    return SafetyAssessment(
        is_safe=not reason_codes,
        original_critical_tokens=original_texts,
        rewrite_critical_tokens=rewrite_texts,
        missing_tokens=missing_tokens,
        added_tokens=added_tokens,
        negation_added=negation_added,
        reason_codes=reason_codes,
        rewrite_token_types=rewrite_token_types,
    )


def _build_guard_report(
    assessment: SafetyAssessment,
    *,
    initial_assessment: SafetyAssessment,
    repair_attempted: bool,
    final_action: str,
    extra_reason_codes: list[str] | None = None,
    repair_error: str | None = None,
) -> dict[str, Any]:
    reason_codes = list(assessment.reason_codes)
    for code in extra_reason_codes or []:
        if code not in reason_codes:
            reason_codes.append(code)

    return {
        "is_safe": assessment.is_safe,
        "original_critical_tokens": list(assessment.original_critical_tokens),
        "rewrite_critical_tokens": list(assessment.rewrite_critical_tokens),
        "missing_tokens": list(assessment.missing_tokens),
        "added_tokens": list(assessment.added_tokens),
        "negation_added": assessment.negation_added,
        "repair_attempted": repair_attempted,
        "final_action": final_action,
        "reason_codes": reason_codes,
        # Why the candidate (as the rewriter produced it) was or wasn't flagged; kept even when a
        # repair succeeded and the final fields above are clean.
        "initial_assessment": {
            "is_safe": initial_assessment.is_safe,
            "reason_codes": list(initial_assessment.reason_codes),
            "missing_tokens": list(initial_assessment.missing_tokens),
            "added_tokens": list(initial_assessment.added_tokens),
            "negation_added": initial_assessment.negation_added,
        },
        "repair_error": repair_error,
    }


def _attempt_repair(
    *,
    original_query: str,
    candidate_rewrite: str,
    original_critical_tokens: list[str],
    repair_model: ChatModel,
) -> str:
    messages = build_repair_messages(
        original_query=original_query,
        candidate_rewrite=candidate_rewrite,
        critical_tokens=original_critical_tokens,
    )
    response = repair_model.complete(ChatRequest(messages=messages))
    repaired_text = response.text.strip()
    if not repaired_text:
        raise EmptyRepairError("Repair response was empty.")
    return repaired_text


def _has_negation(text: str) -> bool:
    return any(pattern.search(text) for pattern in NEGATION_PATTERNS)


def _overlaps(left: CriticalToken, right: CriticalToken) -> bool:
    return left.start < right.end and right.start < left.end
