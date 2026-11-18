"""guard/structured.py (R6): rewrites checked field by field against the stated structure."""

from __future__ import annotations

import pytest

from rag_thesis.guard.checks import BLOCK, WARN, is_blocked
from rag_thesis.guard.structured import assess_structured, check_rewrite, check_structure
from rag_thesis.rewrite.rewriter import MULTI_REWRITE_ORDER, StructuredRewrite

QUESTION = "How many TAp73 isoforms are known in humans?"


def _structure(rewrites, entities=("TAp73",), constraints=("in humans",), abbreviations=None):
    return StructuredRewrite.model_validate(
        {
            "entities": list(entities),
            "abbreviations": abbreviations
            if abbreviations is not None
            else [{"short": "TAp73", "long": "transactivation-competent p73"}],
            "answer_type": "number",
            "constraints": list(constraints),
            "rewrites": dict(zip(MULTI_REWRITE_ORDER, rewrites, strict=True)),
        }
    )


GOOD = [
    "How many TAp73 isoforms exist in humans?",
    "Number of TAp73 variants identified in humans",
    "How many transactivation-competent p73 isoforms are found in humans?",  # the stated long form
    "TAp73 isoforms in humans: how many?",
]


def test_rewrites_that_keep_the_structure_pass():
    structure_violations, per_rewrite = assess_structured(QUESTION, _structure(GOOD))
    assert structure_violations == [] and per_rewrite == [[], [], [], []]


@pytest.mark.parametrize(
    ("rewrite", "check", "severity"),
    [
        ("How many TAp73 isoforms are known?", "constraints", BLOCK),  # "in humans" dropped
        ("How many p73 isoforms exist in humans?", "entities", BLOCK),  # TAp73 lost, no long form
        ("List the TAp73 isoforms identified in humans", "answer_type", BLOCK),
        ("How many TAp73 isoforms are not identified in humans?", "negation", BLOCK),
        ("How many TAp73 isoforms bind TP53 in humans?", "added_entities", BLOCK),
        ("How many TAp73 isoforms exist in humans and Escherichia?", "added_entities", WARN),
    ],
)
def test_each_field_is_checked(rewrite, check, severity):
    violations = check_rewrite(QUESTION, _structure(GOOD), rewrite)
    assert {(v.check, v.severity) for v in violations} == {(check, severity)}


def test_an_invented_entity_is_reported_and_not_trusted():
    structure = _structure(GOOD, entities=("TAp73", "BRCA1"))
    (violation,) = check_structure(QUESTION, structure)
    assert violation.severity == WARN and "BRCA1" in violation.detail
    # BRCA1 is not required in the rewrites, and a rewrite that adds it is flagged as an addition.
    assert check_rewrite(QUESTION, structure, GOOD[0]) == []
    added = check_rewrite(QUESTION, structure, "How many TAp73 isoforms bind BRCA1 in humans?")
    assert is_blocked(added) and added[0].detail == "adds BRCA1"


def test_names_severity_is_validated():
    with pytest.raises(ValueError, match="names"):
        assess_structured(QUESTION, _structure(GOOD), names="sometimes")
