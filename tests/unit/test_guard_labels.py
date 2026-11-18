"""tools/guard_labels.py: suggested labels from rules independent of the guard; the review file."""

from __future__ import annotations

import csv

import pytest
from tools.guard_labels import _has_owner_input, identifiers, sample_rows, suggest_label

from rag_thesis.guard.aliases import AbbreviationDictionary
from rag_thesis.rewrite.rewriter import MULTI_REWRITE_ORDER

ALIASES = AbbreviationDictionary.from_texts(["oculocutaneous albinism type 1 (OCA1) is common"])


def label(question: str, rewrite: str, answers: tuple[str, ...] = ("zzz",)) -> tuple[str, str]:
    suggestion = suggest_label(question, rewrite, list(answers), ALIASES)
    return suggestion["suggested_label"], suggestion["suggested_category"]


def test_identifiers_are_tokens_with_digits_or_two_capitals_split_at_brackets():
    assert identifiers("Which enzyme causes oculocutaneous 1 (OCA1)-type albinism in DNA [18F]x?") == [
        "1",
        "OCA1",
        "DNA",
        "18F",
    ]
    assert identifiers("Which gene-defect underlies the Vel-blood group?") == []  # ordinary compounds


@pytest.mark.parametrize(
    ("question", "rewrite", "answers", "expected"),
    [
        (
            "What is another name for Acrokeratosis paraneoplastica?",
            "Bazex syndrome or Acrokeratosis paraneoplastica",
            ("Bazex syndrome",),
            ("flag", "adds_answer"),
        ),
        (
            "Which genes are not mutated in MDS?",
            "Which genes are mutated in MDS?",
            ("zzz",),
            ("flag", "negation_change"),
        ),
        ("How many TAp73 isoforms exist?", "List the TAp73 isoforms", ("zzz",), ("flag", "answer_type_drift")),
        ("How many TAp73 isoforms exist?", "Quantify the TAp73 isoforms that exist", ("zzz",), ("ok", "")),
        ("Which kinase does KN-93 inhibit?", "Which kinase is inhibited?", ("zzz",), ("flag", "drops_critical")),
        ("Which kinase does KN-93 inhibit?", "Which kinase does KN 93 inhibit?", ("zzz",), ("ok", "")),
        ("What causes OCA1 albinism?", "What causes oculocutaneous albinism type 1?", ("zzz",), ("ok", "")),
        ("Where is DAT1 located?", "Where is the dopamine transporter 1 located?", ("zzz",), ("check", "expanded")),
        (
            "Which SWI/SNF subunit binds?",
            "Which SWItch/Sucrose Non-Fermentable subunit binds?",
            ("zzz",),
            ("check", "expanded"),
        ),  # "Non" inside the expansion is not a negation
        ("Which enzyme maintains methylation?", "Which protein keeps methylation?", ("zzz",), ("check", "paraphrase")),
        (
            "Which enzyme maintains methylation?",
            "Which enzyme maintains methylation in Escherichia?",
            ("zzz",),
            ("check", "adds_entity"),
        ),
    ],
)
def test_suggestions(question, rewrite, answers, expected):
    assert label(question, rewrite, answers) == expected


def test_one_row_per_distinct_question_and_rewrite():
    question = {"question": "What causes OCA1 albinism?", "answers": [{"text": "TYR", "answer_start": 0}]}
    questions = [{**question, "qid": "Q_001"}, {**question, "qid": "Q_002"}]
    four = ["What causes OCA1 albinism?", "OCA1 cause", "What causes oculocutaneous albinism type 1?", "OCA1"]
    rows = sample_rows(questions, {"Q_001": four, "Q_002": four}, ALIASES)
    assert len(rows) == 4 and {row["qid"] for row in rows} == {"Q_001"}
    assert all(row["also_qids"] == "Q_002" for row in rows)
    assert sorted(row["rewrite_type"] for row in rows) == sorted(MULTI_REWRITE_ORDER)
    assert sorted(row["row_id"] for row in rows) == ["g001", "g002", "g003", "g004"]


def test_a_file_with_owner_labels_is_recognised(tmp_path):
    path = tmp_path / "guard_labels.csv"
    assert not _has_owner_input(path)
    rows = [{"row_id": "g001", "owner_label": ""}, {"row_id": "g002", "owner_label": "ok"}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "owner_label"])
        writer.writeheader()
        writer.writerows(rows[:1])
    assert not _has_owner_input(path)  # blank labels: safe to regenerate
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=["row_id", "owner_label"]).writerow(rows[1])
    assert _has_owner_input(path)
