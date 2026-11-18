"""tools/make_dev_split.py: question-level dev split, disjoint from the test questions."""

from __future__ import annotations

from tools.make_dev_split import dev_split

from rag_thesis.evaluation.retrieval import bioasq_question_id


def _qa(qid: str) -> dict:
    return {"qid": qid, "question": f"question {bioasq_question_id(qid)}", "doc_id": "d", "answers": []}


QUESTIONS = [_qa(f"Q{q}_{s:03d}") for q in range(10) for s in range(1, 4)]  # 10 questions x 3 snippets


def test_dev_split_excludes_test_questions_and_takes_one_pair_each():
    dev = dev_split(QUESTIONS, test_qids=["Q0_002", "Q1_001"], n_questions=5, seed=7)
    ids = [bioasq_question_id(q["qid"]) for q in dev]
    assert len(dev) == 5 and len(set(ids)) == 5  # one QA pair per question
    assert not {"Q0", "Q1"} & set(ids)  # every snippet of a test question is excluded
    assert [q["qid"] for q in dev] == sorted(q["qid"] for q in dev)


def test_dev_split_is_deterministic_and_caps_at_what_is_available():
    assert dev_split(QUESTIONS, ["Q0_001"], 5, seed=7) == dev_split(QUESTIONS, ["Q0_001"], 5, seed=7)
    assert dev_split(QUESTIONS, ["Q0_001"], 5, seed=7) != dev_split(QUESTIONS, ["Q0_001"], 5, seed=8)
    assert len(dev_split(QUESTIONS, ["Q0_001"], 50)) == 9
