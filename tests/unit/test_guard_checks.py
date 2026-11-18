"""guard/checks.py (guard v2): the documented v1 false alarms and the missed injection, plus each check."""

from __future__ import annotations

import pytest

from rag_thesis.guard.aliases import AbbreviationDictionary
from rag_thesis.guard.checks import BLOCK, WARN, assess, identifiers, is_blocked
from rag_thesis.guard.domain_safe import assess_rewrite

EMPTY = AbbreviationDictionary({})
TAP73 = "How many TAp73 isoforms are known in humans?"
PHILADELPHIA = (
    "Which tyrosine kinase in Philadelphia- chromosome positive chronic myelogenous leukemia "
    "does Imatinib (Gleevec) inhibit?"
)
DNA = "Which enzyme maintains DNA (cytosine-5-)-methylation?"
DAT1 = "On which chromosome region is the human dopamine transporter gene (DAT1) located?"
FIPA = "Mutations in which gene are linked to familial isolated pituitary adenoma?"
BAZEX = "What is another name for Acrokeratosis paraneoplastica?"

# Real cached rewrites the v1 guard flags although they keep the meaning.
V1_FALSE_ALARMS = [
    (TAP73, "How many transactivation-competent p73 isoforms are known in humans?"),
    (FIPA, "Mutations in which gene are linked to FIPA (familial isolated pituitary adenoma)?"),
    (
        PHILADELPHIA,
        "Identify the protein kinase inhibited by Imatinib (Gleevec) in Philadelphia chromosome-positive "
        "chronic myeloid leukemia.",
    ),
    (PHILADELPHIA, "Imatinib Gleevec target tyrosine kinase Philadelphia chromosome CML"),
    (DNA, "Which enzyme maintains deoxyribonucleic acid (cytosine-5-)-methylation?"),
    (DAT1, "What is the chromosomal location of the human dopamine transporter 1 gene?"),
]


@pytest.mark.parametrize(("question", "rewrite"), V1_FALSE_ALARMS)
def test_v1_false_alarms_are_not_blocked(question, rewrite):
    assert not assess_rewrite(question, rewrite).is_safe  # v1 flags it
    assert not is_blocked(assess(question, rewrite, EMPTY))  # v2 doesn't block (at most warns)


def test_the_bazex_injection_v1_missed_is_caught_when_names_block():
    rewrite = "Synonyms for Bazex syndrome, also known as Acrokeratosis paraneoplastica"
    assert assess_rewrite(BAZEX, rewrite).is_safe  # v1 lets the leaked answer through
    assert [v.detail for v in assess(BAZEX, rewrite, EMPTY)] == ["adds Bazex"]
    assert not is_blocked(assess(BAZEX, rewrite, EMPTY))  # names warn by default
    assert is_blocked(assess(BAZEX, rewrite, EMPTY, names=BLOCK))


@pytest.mark.parametrize(
    ("question", "rewrite", "check", "severity"),
    [
        ("Which kinase does KN-93 inhibit?", "Which kinase is inhibited?", "identifiers", BLOCK),
        ("Which kinase does KN-93 inhibit?", "Which kinase does KN 93 inhibit?", None, None),  # spelling only
        (DNA, "Which enzyme maintains deoxyribonucleic acid methylation at cytosine 5?", "identifiers", WARN),
        ("Which genes are not mutated in MDS?", "Which genes are mutated in MDS?", "negation", BLOCK),
        ("Which genes are mutated in MDS?", "Which genes are never mutated in MDS?", "negation", BLOCK),
        (TAP73, "List the TAp73 isoforms identified in humans", "answer_type", BLOCK),
        (TAP73, "Number of TAp73 isoforms identified in humans", None, None),
        (
            "Which gene causes Marfan syndrome?",
            "Which gene, FBN1 or TP53, causes Marfan syndrome?",
            "added_entities",
            BLOCK,
        ),
    ],
)
def test_each_check(question, rewrite, check, severity):
    violations = assess(question, rewrite, EMPTY)
    if check is None:
        assert violations == []
    else:
        assert {(v.check, v.severity) for v in violations} == {(check, severity)}


def test_corpus_aliases_count_as_kept_and_the_selection_works():
    aliases = AbbreviationDictionary.from_texts(["oculocutaneous albinism type 1 (OCA1) is common"])
    question, rewrite = "What causes OCA1 albinism?", "What causes oculocutaneous albinism type 1?"
    assert assess(question, rewrite, aliases) == []
    assert assess(question, rewrite, EMPTY, checks=("negation",)) == []
    with pytest.raises(ValueError, match="names"):
        assess(question, rewrite, aliases, names="maybe")
    assert identifiers("U12-dependent DNA (OCA1)-type p73 Turner") == ["U12-dependent", "DNA", "OCA1", "p73"]
