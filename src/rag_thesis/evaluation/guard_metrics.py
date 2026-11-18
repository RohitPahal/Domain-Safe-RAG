"""Guard and leakage metrics of run logs, counted over distinct rewrites.

BioASQ asks one question once per snippet, so repeated questions get identical rewrites; each
distinct (question, rewrite) pair counts once. Initial reasons are recomputed with the guard's own
check (``guard.domain_safe.assess_rewrite``): logs written before the initial assessment was logged
kept only the post-repair reasons, and recomputing gives the codes the guard saw.

A rewrite *leaks* when it contains a gold answer the question doesn't (as a whole word sequence,
case-insensitive): the rewriter put its own knowledge, here the answer, into the query.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rag_thesis.guard.checks import contains, words
from rag_thesis.guard.domain_safe import assess_rewrite

ACTIONS = ("pass", "repaired", "rejected", "fallback_original")
REASONS = ("MISSING_CRITICAL_TOKENS", "ADDED_SENSITIVE_TOKENS", "NEGATION_ADDED")


def answer_leaks(question: str, rewrite: str, answers: list[str]) -> list[str]:
    """The gold answers that ``rewrite`` contains and ``question`` doesn't."""
    question_words, rewrite_words = words(question), words(rewrite)
    return [
        answer
        for answer in dict.fromkeys(answers)
        if contains(rewrite_words, words(answer)) and not contains(question_words, words(answer))
    ]


def guard_metric_rows(
    run_rows: list[dict[str, Any]],
    group_key: Callable[[dict[str, Any]], Any],
    answers_by_qid: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """One row per run group (``group_key(row).to_dict()``) that rewrites: distinct rewrites, the
    guard's final actions and initial reasons, and how many rewrites / final queries leak."""
    groups: dict[Any, dict[str, Any]] = {}
    for row in run_rows:
        key = group_key(row)
        if key.rewrite_mode == "none":
            continue
        group = groups.setdefault(key, {"candidates": {}, "finals": set()})
        question = str(row["query_original"])
        reports = (row.get("guard_report") or {}).get("reports") or []
        candidates = [(report["candidate_rewrite"], report.get("final_action")) for report in reports]
        if not reports:  # guard off: the logged queries are the rewrites themselves
            candidates = [(rewrite, None) for rewrite in row["query_rewrites"]]
        for candidate, action in candidates:
            group["candidates"].setdefault((row["qid"], question, candidate), action)
        group["finals"].update((row["qid"], question, final) for final in row["query_rewrites"])
    return [_group_row(key, group, answers_by_qid) for key, group in sorted(groups.items())]


def _group_row(key: Any, group: dict[str, Any], answers_by_qid: dict[str, list[str]] | None) -> dict[str, Any]:
    distinct: dict[tuple[str, str], str | None] = {}
    qid_of: dict[tuple[str, str], str] = {}
    for (qid, question, candidate), action in group["candidates"].items():
        distinct.setdefault((question, candidate), action)
        qid_of.setdefault((question, candidate), qid)
    row: dict[str, Any] = {**key.to_dict(), "distinct_rewrites": len(distinct)}
    for action in ACTIONS:
        row[action] = sum(value == action for value in distinct.values())
    for reason in REASONS:
        row[f"initial_{reason}"] = sum(reason in assess_rewrite(q, c).reason_codes for q, c in distinct)
    if answers_by_qid is None:
        row["leaking_rewrites"] = row["leaking_final_queries"] = ""
        return row
    row["leaking_rewrites"] = sum(bool(answer_leaks(q, c, answers_by_qid.get(qid_of[(q, c)], []))) for q, c in distinct)
    finals = {(question, final): qid for qid, question, final in group["finals"]}
    row["leaking_final_queries"] = sum(
        bool(answer_leaks(q, f, answers_by_qid.get(qid, []))) for (q, f), qid in finals.items()
    )
    return row


def guard_markdown(rows: list[dict[str, Any]]) -> list[str]:
    """The summary section for ``guard_metric_rows``."""
    if not rows:
        return []
    header = ["retriever", "rewrite", "guard", "distinct rewrites", *ACTIONS, "initial missing", "initial added"]
    header += ["initial negation", "leaking rewrites", "leaking final queries"]
    lines = [
        "",
        "## Guard and leakage (distinct rewrites)",
        "",
        "Each distinct (question, rewrite) pair counts once. Initial reasons: the guard's check of the rewrite "
        "as the rewriter wrote it (recomputed). Leaking: contains a gold answer the question doesn't.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * 3 + ["---:"] * (len(header) - 3)) + " |",
    ]
    for row in rows:
        values = [row["retriever_mode"], row["rewrite_mode"], row["guard_mode"], row["distinct_rewrites"]]
        values += [row[action] for action in ACTIONS] + [row[f"initial_{reason}"] for reason in REASONS]
        values += [row["leaking_rewrites"], row["leaking_final_queries"]]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return lines
