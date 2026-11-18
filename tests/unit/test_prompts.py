"""Prompts: each prompt's exact text is pinned to its version.

The fingerprints were taken from the code that produced the thesis runs, before the prompts moved
to ``rag_thesis.prompts``, so this also proves the move changed no text. If a prompt changes on
purpose: bump its ``VERSION`` (that changes cache keys) and update the pinned version + fingerprint.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from rag_thesis.prompts import answer, answer_v2, hyde, judge, repair, rewrite, rewrite_structured


def _fingerprint(messages) -> str:
    pairs = [[message.role, message.content] for message in messages]
    return hashlib.sha256(json.dumps(pairs, ensure_ascii=False).encode("utf-8")).hexdigest()


EVIDENCE = [{"chunk_id": "c1", "source": "citation", "text": "BRCA1 is mutated."}]

CASES = {
    "rewrite/single": (
        rewrite,
        lambda: rewrite.build_rewrite_messages("Which gene is mutated in BRCA?", "single"),
    ),
    "rewrite/multi": (
        rewrite,
        lambda: rewrite.build_rewrite_messages("Which gene is mutated in BRCA?", "multi"),
    ),
    "repair/tokens": (
        repair,
        lambda: repair.build_repair_messages(
            original_query="Is TP53 mutated in 50% of cancers?",
            candidate_rewrite="Is p53 altered in cancers?",
            critical_tokens=["TP53", "50%"],
        ),
    ),
    "repair/no_tokens": (
        repair,
        lambda: repair.build_repair_messages(
            original_query="Is TP53 mutated in 50% of cancers?",
            candidate_rewrite="Is p53 altered in cancers?",
            critical_tokens=[],
        ),
    ),
    "answer/two_chunks": (
        answer,
        lambda: answer.build_answer_messages("Which gene is mutated?", ["[c1] BRCA1 is mutated.", "[c2] Other text."]),
    ),
    "answer_v2/two_passages": (
        answer_v2,
        lambda: answer_v2.build_answer_messages(
            "Which gene is mutated?", [("c1", "BRCA1 is mutated."), ("c2", "Other text.")]
        ),
    ),
    "judge/answered": (
        judge,
        lambda: judge.build_judge_messages(
            question="Which gene?", predicted_answer="BRCA1", gold_answers=["BRCA1"], evidence_chunks=EVIDENCE
        ),
    ),
    "hyde": (hyde, lambda: hyde.build_hyde_messages("Which gene is mutated in BRCA?")),
    "rewrite_structured": (
        rewrite_structured,
        lambda: rewrite_structured.build_structured_rewrite_messages("Which gene is mutated in BRCA?"),
    ),
    "judge/empty": (
        judge,
        lambda: judge.build_judge_messages(
            question="Which gene?",
            predicted_answer="",
            gold_answers=["BRCA1", "breast cancer 1"],
            evidence_chunks=[],
        ),
    ),
}

# (version, fingerprint) per case.
PINNED = {
    "rewrite/single": ("rewrite_v1", "606c7041deb989e85ca823d0a8053720ad8b51d18b626bad4f9323396de0d9d1"),
    "rewrite/multi": ("rewrite_v1", "32bb72794b7b6aa09bbc5bebfaad4f24d81dad1b9cf1751733a1f4a375cb4a32"),
    "repair/tokens": ("repair_v1", "ca72e6bb763174cf5a3cca184a5cb933c1b078c3e0a2e7532e87f4129c0f6fd1"),
    "repair/no_tokens": ("repair_v1", "5bae2696adf83e0cac90bfe7eb556ce365c34d00d97c5c5d680e8fbe70db8282"),
    "answer/two_chunks": ("answer_v1", "c755f160ce88c4b66a95dd865900040ae8e9a76e61d8b8b6dcf54fa0ccee11fb"),
    "answer_v2/two_passages": (
        "answer_v2",
        "1d7d8eaebe04942c84fa0d070090b275ca241a959d61f0a2cf0b09f027e7e277",
    ),
    "judge/answered": ("judge_v2", "7bd83ce5eb7f27269062d0bfb313fb4959fc7ba27c8fd750ecffe06b4286340e"),
    "hyde": ("hyde_v1", "6a54fa005ff286b4eaa2e749b67c09a0c91270c72b7b516303b87c96cadfd9eb"),
    "rewrite_structured": (
        "rewrite_structured_v1",
        "820c8dc63f61ca81afb657e3a616534ffef08405db509c1bc9da58f0778cee40",
    ),
    "judge/empty": ("judge_v2", "16c0f823fe88756417c8e6ae3a5567ac50f56d43432708fa7b4712b1294b0f2e"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_prompt_text_matches_its_pinned_version(case):
    module, build = CASES[case]
    version, fingerprint = PINNED[case]
    assert module.VERSION == version, f"{case}: version changed; update PINNED after reviewing the new text"
    assert _fingerprint(build()) == fingerprint, f"{case}: prompt text changed without a version bump"


def test_prompts_are_plain_provider_neutral_messages():
    for module, build in CASES.values():
        roles = [message.role for message in build()]
        assert roles == ["system", "user"], module.__name__
