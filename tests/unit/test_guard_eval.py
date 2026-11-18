"""tools/guard_eval.py: reviewed rows, guard scores against owner labels, the report."""

from __future__ import annotations

import csv

import pytest
from tools.guard_eval import report_markdown, reviewed_rows, scores


def _write_labels(path, labels):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "qid", "question", "rewrite", "owner_label"])
        writer.writeheader()
        for index, label in enumerate(labels, start=1):
            writer.writerow(
                {
                    "row_id": f"g{index:03d}",
                    "qid": f"Q{index}_001",
                    "question": "q",
                    "rewrite": "r",
                    "owner_label": label,
                }
            )


def test_only_reviewed_rows_count_and_labels_are_checked(tmp_path):
    path = tmp_path / "guard_labels.csv"
    assert reviewed_rows(path) == []  # no file yet
    _write_labels(path, ["ok", "", " Flag "])
    assert [row["row_id"] for row in reviewed_rows(path)] == ["g001", "g003"]
    _write_labels(path, ["ok", "maybe"])
    with pytest.raises(ValueError, match="maybe"):
        reviewed_rows(path)


def _row(qid, label, v1, warn, block):
    return {"qid": qid, "owner_label": label, "v1": v1, "v2_names_warn": warn, "v2_names_block": block}


ROWS = [
    _row("A_001", "flag", True, True, True),  # real problem, all flag
    _row("B_001", "flag", False, False, True),  # an injected name: only v2 with names blocking flags it
    _row("C_001", "ok", True, False, False),  # v1 false alarm, v2 passes
    _row("D_001", "ok", True, False, False),
    _row("E_001", "ok", False, False, False),
]


def test_scores_count_the_confusion_matrix_and_rates():
    v1 = scores(ROWS, "v1")
    assert (v1["tp"], v1["fp"], v1["fn"], v1["tn"]) == (1, 2, 1, 1)
    assert v1["precision"][0] == pytest.approx(1 / 3) and v1["recall"][0] == 0.5
    assert v1["false_alarm_rate"][0] == pytest.approx(2 / 3)
    block = scores(ROWS, "v2_names_block")
    assert (block["recall"][0], block["false_alarm_rate"][0], block["precision"][0]) == (1.0, 0.0, 1.0)
    nothing = scores([_row("E_001", "ok", False, False, False)], "v1")
    assert nothing["precision"] is None and nothing["recall"] is None  # no flags, no real problems


def test_report_with_and_without_labels():
    all_rows = [{**row, "question": "q", "rewrite": "r"} for row in ROWS]
    assert "No reviewed rows yet" in report_markdown(all_rows, [])
    report = report_markdown(all_rows, ROWS)
    assert "5 reviewed rows (2 flag, 3 ok)" in report
    assert "| v1 | 1 | 2 | 1 | 1 |" in report
    assert "v2_names_block: false alarms removed 2, added 0" in report
    assert "real problems newly caught 1, newly missed 0" in report
