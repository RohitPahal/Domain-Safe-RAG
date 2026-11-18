from __future__ import annotations

import csv
import dataclasses
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rag_thesis.config import load_config
from rag_thesis.evaluation.answers import load_gold_answers, load_run_log
from rag_thesis.io import load_chunks_by_id
from rag_thesis.llm.base import ChatModel, extract_json_object
from rag_thesis.llm.errors import LLMCallsDisabledError
from rag_thesis.llm.registry import build_chat_model
from rag_thesis.llm.settings import LLMSettings, resolve_llm_settings
from rag_thesis.llm.types import ChatRequest
from rag_thesis.prompts import judge as judge_prompt
from rag_thesis.prompts.judge import build_judge_messages
from rag_thesis.protection import ensure_writable

OVERALL_VERDICTS = ("CORRECT", "PARTIALLY_CORRECT", "INCORRECT", "UNSUPPORTED")
JUDGE_LABELS = OVERALL_VERDICTS
FAITHFULNESS_LABELS = ("FAITHFUL", "PARTIALLY_FAITHFUL", "UNFAITHFUL")
RELEVANCE_LABELS = ("RELEVANT", "PARTIALLY_RELEVANT", "IRRELEVANT")
FAITHFULNESS_SCORE_MAP = {
    "FAITHFUL": 1.0,
    "PARTIALLY_FAITHFUL": 0.5,
    "UNFAITHFUL": 0.0,
}
RELEVANCE_SCORE_MAP = {
    "RELEVANT": 1.0,
    "PARTIALLY_RELEVANT": 0.5,
    "IRRELEVANT": 0.0,
}
MAX_EVIDENCE_CHUNKS = 3
MAX_SNIPPET_CHARS = 700
PROMPT_VERSION = judge_prompt.VERSION


def parse_judge_response(response_text: str) -> dict[str, Any]:
    payload = extract_json_object(response_text)  # the shared tolerant parser (code fences, text around)

    overall_verdict = str(payload.get("overall_verdict", "")).strip().upper()
    overall_reason = str(payload.get("overall_reason", "")).strip() or "No reason provided."
    faithfulness_label = str(payload.get("faithfulness_label", "")).strip().upper()
    relevance_label = str(payload.get("relevance_label", "")).strip().upper()

    if overall_verdict not in OVERALL_VERDICTS:
        raise ValueError(f"Invalid overall_verdict '{overall_verdict}' in response: {response_text}")
    if faithfulness_label not in FAITHFULNESS_LABELS:
        raise ValueError(f"Invalid faithfulness_label '{faithfulness_label}' in response: {response_text}")
    if relevance_label not in RELEVANCE_LABELS:
        raise ValueError(f"Invalid relevance_label '{relevance_label}' in response: {response_text}")

    if "faithfulness_score" not in payload:
        raise ValueError(f"Missing faithfulness_score in response: {response_text}")
    if "relevance_score" not in payload:
        raise ValueError(f"Missing relevance_score in response: {response_text}")
    if "supported_text" not in payload or "unsupported_text" not in payload:
        raise ValueError(f"supported_text and unsupported_text must both be present: {response_text}")

    try:
        faithfulness_score = float(payload["faithfulness_score"])
        relevance_score = float(payload["relevance_score"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Judge scores must be numeric: {response_text}") from exc

    expected_faithfulness_score = FAITHFULNESS_SCORE_MAP[faithfulness_label]
    expected_relevance_score = RELEVANCE_SCORE_MAP[relevance_label]
    if faithfulness_score != expected_faithfulness_score:
        raise ValueError(
            f"faithfulness_score {faithfulness_score} does not match label {faithfulness_label}: {response_text}"
        )
    if relevance_score != expected_relevance_score:
        raise ValueError(f"relevance_score {relevance_score} does not match label {relevance_label}: {response_text}")

    supported_text_raw = payload["supported_text"]
    unsupported_text_raw = payload["unsupported_text"]
    if not isinstance(supported_text_raw, str) or not isinstance(unsupported_text_raw, str):
        raise ValueError(f"supported_text and unsupported_text must both be strings: {response_text}")

    return {
        "overall_verdict": overall_verdict,
        "overall_reason": overall_reason,
        "judge_label": overall_verdict,
        "judge_reason": overall_reason,
        "faithfulness_label": faithfulness_label,
        "faithfulness_score": faithfulness_score,
        "relevance_label": relevance_label,
        "relevance_score": relevance_score,
        "supported_text": supported_text_raw,
        "unsupported_text": unsupported_text_raw,
    }


def choose_evidence_chunks(run_row: dict[str, Any], chunks_by_id: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    candidate_chunk_ids: list[tuple[str, str]] = []
    seen: set[str] = set()

    for citation in run_row.get("citations", []):
        chunk_id = str(citation).strip()
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            candidate_chunk_ids.append((chunk_id, "citation"))

    if not candidate_chunk_ids:
        for result in run_row.get("retrieval_results", []):
            chunk_id = str(result.get("chunk_id", "")).strip()
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                candidate_chunk_ids.append((chunk_id, "retrieved"))
            if len(candidate_chunk_ids) >= MAX_EVIDENCE_CHUNKS:
                break

    evidence_chunks: list[dict[str, str]] = []
    for chunk_id, source in candidate_chunk_ids[:MAX_EVIDENCE_CHUNKS]:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        snippet = str(chunk["text"]).strip()
        if len(snippet) > MAX_SNIPPET_CHARS:
            snippet = snippet[: MAX_SNIPPET_CHARS - 3].rstrip() + "..."
        evidence_chunks.append(
            {
                "chunk_id": chunk_id,
                "source": source,
                "text": snippet,
            }
        )
    return evidence_chunks


def summarize_judge_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall_counts = Counter(str(row.get("overall_verdict") or row.get("judge_label")) for row in rows)
    faithfulness_counts = Counter(str(row["faithfulness_label"]) for row in rows)
    relevance_counts = Counter(str(row["relevance_label"]) for row in rows)
    total = len(rows)
    summary = {
        "total_questions": total,
    }
    for label in OVERALL_VERDICTS:
        count = int(overall_counts.get(label, 0))
        summary[f"{label.lower()}_count"] = count
        summary[f"{label.lower()}_rate"] = round(count / float(total), 6) if total else 0.0
    for label in FAITHFULNESS_LABELS:
        count = int(faithfulness_counts.get(label, 0))
        summary[f"{label.lower()}_count"] = count
        summary[f"{label.lower()}_rate"] = round(count / float(total), 6) if total else 0.0
    for label in RELEVANCE_LABELS:
        count = int(relevance_counts.get(label, 0))
        summary[f"{label.lower()}_count"] = count
        summary[f"{label.lower()}_rate"] = round(count / float(total), 6) if total else 0.0
    faithfulness_scores = [float(row["faithfulness_score"]) for row in rows]
    relevance_scores = [float(row["relevance_score"]) for row in rows]
    faithful_and_relevant_count = sum(
        1 for row in rows if str(row["faithfulness_label"]) == "FAITHFUL" and str(row["relevance_label"]) == "RELEVANT"
    )
    summary["faithfulness_mean_score"] = round(sum(faithfulness_scores) / float(total), 6) if total else 0.0
    summary["relevance_mean_score"] = round(sum(relevance_scores) / float(total), 6) if total else 0.0
    summary["faithful_and_relevant_count"] = faithful_and_relevant_count
    summary["faithful_and_relevant_rate"] = round(faithful_and_relevant_count / float(total), 6) if total else 0.0
    return summary


def evaluate_with_judge(
    *,
    run_log_path: Path,
    questions_path: Path,
    chunks_path: Path,
    config_path: Path,
    reports_dir: Path,
    judge_model: str | None = None,
    chat_model: ChatModel | None = None,
    allow_llm_calls: bool = False,
    first_n: int | None = None,
    output_stem: str | None = None,
) -> dict[str, Any]:
    """Judge a run log. ``chat_model`` is the judge model (tests); otherwise it is built from the
    config's ``judge`` role (``judge_model`` overrides its model name). Without ``allow_llm_calls``
    judging can only replay cached judge responses (``llm.cache_path``)."""
    # Fail fast, before any judge (LLM) call: never write into a protected folder.
    ensure_writable(reports_dir)
    config = load_config(config_path)
    judge = chat_model or _build_judge_model(config, judge_model=judge_model, allow_llm_calls=allow_llm_calls)

    gold_by_qid = load_gold_answers(questions_path)
    chunks_by_id = load_chunks_by_id(chunks_path)
    run_rows = load_run_log(run_log_path)
    if not run_rows:
        raise ValueError(f"No run log rows were found in {run_log_path}.")
    if first_n is not None:
        if first_n <= 0:
            raise ValueError("first_n must be greater than zero when provided.")
        run_rows = run_rows[:first_n]

    judged_rows: list[dict[str, Any]] = []
    for row in run_rows:
        qid = str(row["qid"])
        gold_record = gold_by_qid.get(qid)
        if gold_record is None:
            raise ValueError(f"Gold answers not found for qid: {qid}")

        evidence_chunks = choose_evidence_chunks(row, chunks_by_id)
        messages = build_judge_messages(
            question=str(gold_record["question"]),
            predicted_answer=str(row.get("answer_text") or "").strip(),
            gold_answers=list(gold_record["gold_answers"]),
            evidence_chunks=evidence_chunks,
        )
        response = judge.complete(ChatRequest(messages=messages, json_mode=True))
        parsed = parse_judge_response(response.text)

        judged_rows.append(
            {
                "qid": qid,
                "question": gold_record["question"],
                "predicted_answer": row.get("answer_text") or "",
                "gold_answers": json.dumps(gold_record["gold_answers"], ensure_ascii=False),
                "overall_verdict": parsed["overall_verdict"],
                "overall_reason": parsed["overall_reason"],
                "judge_label": parsed["judge_label"],
                "judge_reason": parsed["judge_reason"],
                "faithfulness_label": parsed["faithfulness_label"],
                "faithfulness_score": parsed["faithfulness_score"],
                "relevance_label": parsed["relevance_label"],
                "relevance_score": parsed["relevance_score"],
                "supported_text": parsed["supported_text"],
                "unsupported_text": parsed["unsupported_text"],
            }
        )

    summary = summarize_judge_rows(judged_rows)

    reports_dir.mkdir(parents=True, exist_ok=True)
    resolved_output_stem = output_stem or run_log_path.stem
    per_question_path = reports_dir / f"{resolved_output_stem}_judge.csv"
    summary_path = reports_dir / f"{resolved_output_stem}_judge_summary.md"

    with per_question_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "qid",
                "question",
                "predicted_answer",
                "gold_answers",
                "overall_verdict",
                "overall_reason",
                "judge_label",
                "judge_reason",
                "faithfulness_label",
                "faithfulness_score",
                "relevance_label",
                "relevance_score",
                "supported_text",
                "unsupported_text",
            ],
        )
        writer.writeheader()
        for row in judged_rows:
            writer.writerow(row)

    summary_lines = [
        "# LLM Judge Summary",
        "",
        f"Run log: `{run_log_path}`",
        f"Gold answers source: `{questions_path}`",
        f"Chunks source: `{chunks_path}`",
        f"Judge model: `{judge.model}` (provider `{judge.provider}`)",
        f"Prompt version: `{PROMPT_VERSION}`",
        "",
        "Judge rubric:",
        "- CORRECT: semantically matches an acceptable gold answer, including clear alias or abbreviation equivalence.",
        "- PARTIALLY_CORRECT: close but incomplete, too broad, too specific, or answer type only partly matches.",
        "- INCORRECT: wrong answer.",
        "- UNSUPPORTED: answer is not justified by the evidence, is speculative, or should not have been answered from "
        "the evidence shown.",
        "- FAITHFUL: all material claims in the predicted answer are supported by the retrieved evidence.",
        "- PARTIALLY_FAITHFUL: the core answer is supported, but unsupported or overstated details were added.",
        "- UNFAITHFUL: the key claim is unsupported by or contradicted by the retrieved evidence.",
        "- RELEVANT: directly answers the question.",
        "- PARTIALLY_RELEVANT: on-topic but incomplete, vague, or only partly answers the question.",
        "- IRRELEVANT: does not actually answer the question or is mostly off-topic.",
        "",
        f"- total_questions: {summary['total_questions']}",
    ]
    for label in OVERALL_VERDICTS:
        count_key = f"{label.lower()}_count"
        rate_key = f"{label.lower()}_rate"
        summary_lines.append(f"- {count_key}: {summary[count_key]}")
        summary_lines.append(f"- {rate_key}: {summary[rate_key]:.6f}")
    for label in FAITHFULNESS_LABELS:
        count_key = f"{label.lower()}_count"
        rate_key = f"{label.lower()}_rate"
        summary_lines.append(f"- {count_key}: {summary[count_key]}")
        summary_lines.append(f"- {rate_key}: {summary[rate_key]:.6f}")
    summary_lines.append(f"- faithfulness_mean_score: {summary['faithfulness_mean_score']:.6f}")
    for label in RELEVANCE_LABELS:
        count_key = f"{label.lower()}_count"
        rate_key = f"{label.lower()}_rate"
        summary_lines.append(f"- {count_key}: {summary[count_key]}")
        summary_lines.append(f"- {rate_key}: {summary[rate_key]:.6f}")
    summary_lines.append(f"- relevance_mean_score: {summary['relevance_mean_score']:.6f}")
    summary_lines.append(f"- faithful_and_relevant_count: {summary['faithful_and_relevant_count']}")
    summary_lines.append(f"- faithful_and_relevant_rate: {summary['faithful_and_relevant_rate']:.6f}")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "judge_model": judge.model,
        "judge_provider": judge.provider,
        "summary": summary,
        "per_question_path": per_question_path,
        "summary_path": summary_path,
    }


def _build_judge_model(config: dict[str, Any], *, judge_model: str | None, allow_llm_calls: bool) -> ChatModel:
    settings = resolve_llm_settings(config)
    if judge_model:
        role = dataclasses.replace(settings.role("judge"), model=judge_model)
        settings = LLMSettings(roles={**settings.roles, "judge": role}, cache_path=settings.cache_path)
    if not allow_llm_calls and not settings.cache_path:
        raise LLMCallsDisabledError(
            "Judging calls an LLM: pass --allow-llm-calls (or configure llm.cache_path to replay cached judgements)."
        )
    return build_chat_model("judge", settings, allow_llm_calls=allow_llm_calls)
