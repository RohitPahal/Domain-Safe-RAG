from __future__ import annotations

import csv
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from rag_thesis.io import load_gold_evidence as read_gold_evidence
from rag_thesis.protection import ensure_writable
from rag_thesis.schemas import QuestionLine, RunLogLine

ARTICLES_PATTERN = re.compile(r"\b(a|an|the)\b")
PUNCT_TRANSLATION = str.maketrans({character: " " for character in string.punctuation})


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    spaced_punctuation = lowered.translate(PUNCT_TRANSLATION)
    no_articles = ARTICLES_PATTERN.sub(" ", spaced_punctuation)
    return " ".join(no_articles.split())


def exact_match_normalized(prediction: str, gold_answer: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold_answer) else 0.0


def token_f1_score(prediction: str, gold_answer: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()
    if not prediction_tokens or not gold_tokens:
        return 1.0 if prediction_tokens == gold_tokens else 0.0

    overlap = Counter(prediction_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0

    precision = common / float(len(prediction_tokens))
    recall = common / float(len(gold_tokens))
    return 2.0 * precision * recall / (precision + recall)


def lenient_containment_score(prediction: str, gold_answer: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()
    if not prediction_tokens or not gold_tokens:
        return 1.0 if prediction_tokens == gold_tokens else 0.0
    if prediction_tokens == gold_tokens:
        return 1.0
    return (
        1.0
        if _token_sequence_contains(prediction_tokens, gold_tokens)
        or _token_sequence_contains(gold_tokens, prediction_tokens)
        else 0.0
    )


def load_gold_answers(questions_path: Path) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    answers_by_question: dict[str, list[str]] = {}

    with questions_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Empty line found in {questions_path} at line {line_number}.")
            payload = QuestionLine.model_validate(json.loads(line)).model_dump()
            rows.append(payload)
            question_text = str(payload["question"]).strip()
            answers = answers_by_question.setdefault(question_text, [])
            for answer in payload.get("answers", []):
                answer_text = str(answer["text"]).strip()
                if answer_text and answer_text not in answers:
                    answers.append(answer_text)

    by_qid: dict[str, dict[str, Any]] = {}
    for payload in rows:
        qid = str(payload["qid"])
        question_text = str(payload["question"]).strip()
        qid_answers = [
            str(answer["text"]).strip() for answer in payload.get("answers", []) if str(answer["text"]).strip()
        ]
        grouped_answers = list(answers_by_question.get(question_text, []))
        by_qid[qid] = {
            "qid": qid,
            "question": question_text,
            "qid_answers": qid_answers,
            "gold_answers": grouped_answers or qid_answers,
        }
    return by_qid


def load_gold_evidence(gold_evidence_path: Path | None) -> dict[str, dict[str, Any]]:
    """Gold evidence keyed by qid, or {} when no gold file is given or it doesn't exist (optional input)."""
    if gold_evidence_path is None or not gold_evidence_path.exists():
        return {}
    return read_gold_evidence(gold_evidence_path)


def load_run_log(run_log_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with run_log_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = RunLogLine.model_validate(json.loads(line)).model_dump()
            payload["_line_number"] = line_number
            rows.append(payload)
    return rows


def score_prediction(prediction: str | None, gold_answers: list[str]) -> dict[str, Any]:
    prediction_text = (prediction or "").strip()
    best_gold = ""
    best_exact_match = 0.0
    best_token_f1 = 0.0
    best_lenient = 0.0
    best_index = -1

    for index, gold_answer in enumerate(gold_answers):
        exact_match = exact_match_normalized(prediction_text, gold_answer)
        token_f1 = token_f1_score(prediction_text, gold_answer)
        lenient = lenient_containment_score(prediction_text, gold_answer)
        candidate_key = (exact_match, token_f1, lenient, -index)
        best_key = (best_exact_match, best_token_f1, best_lenient, -best_index)
        if index == 0 or candidate_key > best_key:
            best_gold = gold_answer
            best_exact_match = exact_match
            best_token_f1 = token_f1
            best_lenient = lenient
            best_index = index

    return {
        "best_matching_gold": best_gold,
        "exact_match": best_exact_match,
        "token_f1": best_token_f1,
        "lenient_containment": best_lenient,
    }


def _token_sequence_contains(container_tokens: list[str], contained_tokens: list[str]) -> bool:
    if len(contained_tokens) > len(container_tokens):
        return False
    window_size = len(contained_tokens)
    for start_index in range(len(container_tokens) - window_size + 1):
        if container_tokens[start_index : start_index + window_size] == contained_tokens:
            return True
    return False


def evaluate_answer_quality(
    *,
    run_log_path: Path,
    questions_path: Path,
    gold_evidence_path: Path | None,
    reports_dir: Path,
) -> dict[str, Any]:
    # Fail fast, before any work: never write into a protected folder.
    ensure_writable(reports_dir)
    run_rows = load_run_log(run_log_path)
    if not run_rows:
        raise ValueError(f"No run log rows were found in {run_log_path}.")

    gold_by_qid = load_gold_answers(questions_path)
    evidence_by_qid = load_gold_evidence(gold_evidence_path)

    per_question_rows: list[dict[str, Any]] = []
    total = len(run_rows)
    answered = 0
    total_em = 0.0
    total_f1 = 0.0
    total_lenient = 0.0

    for row in run_rows:
        qid = str(row["qid"])
        gold_record = gold_by_qid.get(qid)
        if gold_record is None:
            raise ValueError(f"Gold answers not found for qid: {qid}")

        prediction = row.get("answer_text")
        if isinstance(prediction, str) and prediction.strip():
            answered += 1

        scores = score_prediction(prediction, list(gold_record["gold_answers"]))
        total_em += float(scores["exact_match"])
        total_f1 += float(scores["token_f1"])
        total_lenient += float(scores["lenient_containment"])

        retrieved_top_chunk_ids = [
            str(result.get("chunk_id")).strip()
            for result in row.get("retrieval_results", [])
            if str(result.get("chunk_id", "")).strip()
        ]
        citations = [str(citation).strip() for citation in row.get("citations", []) if str(citation).strip()]
        gold_evidence = evidence_by_qid.get(qid, {})
        gold_chunk_ids = [str(chunk_id) for chunk_id in gold_evidence.get("gold_chunk_ids", [])]

        per_question_rows.append(
            {
                "qid": qid,
                "question": gold_record["question"],
                "predicted_answer": prediction or "",
                "gold_answers": json.dumps(gold_record["gold_answers"], ensure_ascii=False),
                "best_matching_gold": scores["best_matching_gold"],
                "exact_match": int(scores["exact_match"]),
                "token_f1": round(float(scores["token_f1"]), 6),
                "lenient_containment": int(scores["lenient_containment"]),
                "citations": json.dumps(citations, ensure_ascii=False),
                "retrieved_top_chunk_ids": json.dumps(retrieved_top_chunk_ids, ensure_ascii=False),
                "gold_chunk_ids": json.dumps(gold_chunk_ids, ensure_ascii=False),
            }
        )

    unanswered = total - answered
    summary = {
        "total_questions": total,
        "answered": answered,
        "unanswered": unanswered,
        "answer_rate": round(answered / float(total), 6),
        "exact_match_normalized": round(total_em / float(total), 6),
        "token_f1": round(total_f1 / float(total), 6),
        "lenient_containment_rate": round(total_lenient / float(total), 6),
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    output_stem = run_log_path.stem
    per_question_path = reports_dir / f"{output_stem}_answer_quality.csv"
    summary_path = reports_dir / f"{output_stem}_answer_quality_summary.md"

    with per_question_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "qid",
                "question",
                "predicted_answer",
                "gold_answers",
                "best_matching_gold",
                "exact_match",
                "token_f1",
                "lenient_containment",
                "citations",
                "retrieved_top_chunk_ids",
                "gold_chunk_ids",
            ],
        )
        writer.writeheader()
        for row in per_question_rows:
            writer.writerow(row)

    summary_lines = [
        "# Answer Quality Summary",
        "",
        f"Run log: `{run_log_path}`",
        f"Gold answers source: `{questions_path}`",
        f"Gold evidence source: `{gold_evidence_path}`"
        if gold_evidence_path is not None
        else "Gold evidence source: `none`",
        "",
        "Gold-answer structure used:",
        "- Primary gold answers are loaded from `data/eval/questions.jsonl`.",
        "- Each current `qid` contains one `answers` span entry.",
        "- Distinct answer texts are also unioned across identical question text so repeated question variants share "
        "all observed acceptable gold strings.",
        "",
        f"- total_questions: {summary['total_questions']}",
        f"- answered: {summary['answered']}",
        f"- unanswered: {summary['unanswered']}",
        f"- answer_rate: {summary['answer_rate']:.6f}",
        f"- exact_match_normalized: {summary['exact_match_normalized']:.6f}",
        f"- token_f1: {summary['token_f1']:.6f}",
        f"- lenient_containment_rate: {summary['lenient_containment_rate']:.6f}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "summary": summary,
        "per_question_path": per_question_path,
        "summary_path": summary_path,
    }
