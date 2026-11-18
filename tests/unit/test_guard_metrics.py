"""evaluation/guard_metrics.py: guard actions, initial reasons and leaks over distinct rewrites."""

from __future__ import annotations

from rag_thesis.evaluation.guard_metrics import answer_leaks, guard_markdown, guard_metric_rows
from rag_thesis.evaluation.report import build_group_key

QUESTION = "Which gene is mutated in BRCA1 carriers?"


def _row(qid: str, mode: str, guard: str, reports: list[dict], finals: list[str]) -> dict:
    return {
        "qid": qid,
        "query_original": QUESTION,
        "query_rewrites": finals,
        "guard_report": {"guard_mode": guard, "reports": reports},
        "config_snapshot": {"retriever_mode": "bm25", "rewrite_mode": mode, "guard_mode": guard, "k": 10},
    }


def test_answer_leaks_needs_the_whole_answer_and_ignores_answers_in_the_question():
    assert answer_leaks("Which syndrome?", "Bazex syndrome synonyms", ["Bazex syndrome", "Bazex"]) == [
        "Bazex syndrome",
        "Bazex",
    ]
    assert answer_leaks("Is BRCA1 mutated?", "BRCA1 mutation", ["BRCA1"]) == []  # already in the question
    assert answer_leaks("Which receptor?", "TNFR levels", ["TNF"]) == []  # not a whole word


def test_rows_count_distinct_rewrites_actions_reasons_and_leaks():
    reports = [
        {"candidate_rewrite": "BRCA1 carriers mutated gene", "final_action": "pass"},
        {"candidate_rewrite": "Which gene is mutated in carriers?", "final_action": "repaired"},  # drops BRCA1
        {"candidate_rewrite": "Is TP53 mutated in BRCA1 carriers?", "final_action": "rejected"},  # adds TP53
    ]
    rows = [
        _row("Q_001", "multi", "on", reports, ["BRCA1 carriers mutated gene", QUESTION]),
        _row("Q_002", "multi", "on", reports, ["BRCA1 carriers mutated gene", QUESTION]),  # same question again
        _row("Q_001", "multi", "off", [], ["BRCA1 carriers mutated gene", "TP53 in BRCA1 carriers"]),
        _row("Q_001", "none", "off", [], [QUESTION]),  # no rewriting: no guard row
    ]
    on, off = sorted(
        guard_metric_rows(rows, build_group_key, {"Q_001": ["TP53"], "Q_002": ["TP53"]}),
        key=lambda row: row["guard_mode"],
        reverse=True,
    )
    assert (on["guard_mode"], on["distinct_rewrites"]) == ("on", 3)  # Q_002's twins count once
    assert (on["pass"], on["repaired"], on["rejected"], on["fallback_original"]) == (1, 1, 1, 0)
    assert on["initial_MISSING_CRITICAL_TOKENS"] == 1 and on["initial_ADDED_SENSITIVE_TOKENS"] == 1
    assert on["leaking_rewrites"] == 1 and on["leaking_final_queries"] == 0  # the guard removed the leak
    assert (off["distinct_rewrites"], off["pass"], off["leaking_rewrites"]) == (2, 0, 1)
    section = "\n".join(guard_markdown([on, off]))
    assert (
        "## Guard and leakage (distinct rewrites)" in section and "| bm25 | multi | on | 3 | 1 | 1 | 1 | 0 |" in section
    )


def test_without_answers_leaks_are_left_blank():
    rows = [_row("Q_001", "multi", "off", [], ["BRCA1 carriers"])]
    (row,) = guard_metric_rows(rows, build_group_key)
    assert row["leaking_rewrites"] == "" and row["leaking_final_queries"] == ""
    assert guard_markdown([]) == []
