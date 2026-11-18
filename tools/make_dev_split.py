"""Dev split for tuning (e.g. R2 fusion weights) that never touches the frozen test questions.

BioASQ asks one question once per snippet (qid = ``<question id>_<snippet>``), so the split is made
at the question level: every BioASQ question of the 120 test qids is excluded, then a seeded sample
of the remaining questions is drawn with one QA pair each (independent questions, and no rewrite
calls spent on repeats). Questions and gold evidence come from the full-corpus build, whose corpus
holds their contexts, so the dev split is evaluated on the full corpus. No LLM calls.

Usage:
    python tools/make_dev_split.py --out-dir data/dev
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from rag_thesis.data.corpus import write_jsonl
from rag_thesis.evaluation.retrieval import bioasq_question_id
from rag_thesis.io import load_gold_evidence, load_questions

N_QUESTIONS = 120
SEED = 42


def dev_split(
    questions: list[dict[str, Any]], test_qids: list[str], n_questions: int = N_QUESTIONS, seed: int = SEED
) -> list[dict[str, Any]]:
    """``n_questions`` BioASQ questions none of ``test_qids`` asks, one QA pair each, sorted by qid."""
    excluded = {bioasq_question_id(qid) for qid in test_qids}
    by_question: dict[str, list[dict[str, Any]]] = {}
    for question in questions:
        question_id = bioasq_question_id(question["qid"])
        if question_id not in excluded:
            by_question.setdefault(question_id, []).append(question)
    rng = random.Random(seed)
    chosen = rng.sample(sorted(by_question), min(n_questions, len(by_question)))
    picked = [rng.choice(sorted(by_question[question_id], key=lambda q: q["qid"])) for question_id in chosen]
    return sorted(picked, key=lambda q: q["qid"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=Path("data/full/eval/questions.jsonl"))
    parser.add_argument("--gold", type=Path, default=Path("data/full/eval/gold_evidence.jsonl"))
    parser.add_argument("--test-questions", type=Path, default=Path("data/eval/questions.jsonl"))
    parser.add_argument("--n", type=int, default=N_QUESTIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=Path("data/dev"))
    args = parser.parse_args()

    test_qids = [q["qid"] for q in load_questions(args.test_questions)]
    dev = dev_split(load_questions(args.questions), test_qids, args.n, args.seed)
    gold = load_gold_evidence(args.gold)
    write_jsonl(args.out_dir / "questions.jsonl", dev)
    write_jsonl(args.out_dir / "gold_evidence.jsonl", [gold[q["qid"]] for q in dev])
    print(f"dev split: {len(dev)} qids from {len(dev)} BioASQ questions, none of the test set's -> {args.out_dir}")


if __name__ == "__main__":
    main()
