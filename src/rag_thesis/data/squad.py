from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_thesis.protection import ensure_writable

BUCKET_ORDER = ("synonym_entity", "numeric", "abbrev_short", "other")
ALL_CAPS_PATTERN = re.compile(r"^[A-Z]{2,}$")


@dataclass(frozen=True)
class QaRecord:
    item_index: int
    paragraph_index: int
    qa_index: int
    qid: str
    question: str
    bucket: str


def load_squad_dataset(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Dataset root must be a JSON object.")

    return payload


def classify_question(question: str) -> str:
    lowered = question.lower()
    if "synonym" in lowered:
        return "synonym_entity"
    if any(character.isdigit() or character == "%" for character in question):
        return "numeric"

    tokens = question.split()
    if len(tokens) <= 7:
        return "abbrev_short"

    for token in tokens:
        normalized = re.sub(r"[^A-Za-z0-9]", "", token)
        if len(normalized) >= 2 and ALL_CAPS_PATTERN.fullmatch(normalized):
            return "abbrev_short"

    return "other"


def analyze_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "data_items": 0,
        "paragraphs": 0,
        "total_qas": 0,
        "preview_pairs": [],
        "empty_questions": 0,
        "empty_answers": 0,
        "invalid_answer_start": 0,
        "answer_text_mismatch": 0,
        "malformed_data_items": 0,
        "malformed_paragraphs": 0,
        "malformed_qas": 0,
        "malformed_answers": 0,
    }

    data_items = payload.get("data")
    if not isinstance(data_items, list):
        return summary

    summary["data_items"] = len(data_items)

    for item in data_items:
        if not isinstance(item, dict):
            summary["malformed_data_items"] += 1
            continue

        paragraphs = item.get("paragraphs")
        if not isinstance(paragraphs, list):
            summary["malformed_data_items"] += 1
            continue

        for paragraph in paragraphs:
            if not isinstance(paragraph, dict):
                summary["malformed_paragraphs"] += 1
                continue

            summary["paragraphs"] += 1
            context = paragraph.get("context")
            qas = paragraph.get("qas")
            if not isinstance(context, str) or not isinstance(qas, list):
                summary["malformed_paragraphs"] += 1
                continue

            for qa in qas:
                if not isinstance(qa, dict):
                    summary["malformed_qas"] += 1
                    continue

                summary["total_qas"] += 1
                qid = qa.get("id")
                question = qa.get("question")
                answers = qa.get("answers")
                qa_is_malformed = False

                if len(summary["preview_pairs"]) < 3 and isinstance(qid, str) and isinstance(question, str):
                    summary["preview_pairs"].append((qid, question))

                if not isinstance(qid, str) or not qid.strip():
                    qa_is_malformed = True

                if not isinstance(question, str):
                    qa_is_malformed = True
                elif not question.strip():
                    summary["empty_questions"] += 1

                if not isinstance(answers, list):
                    qa_is_malformed = True

                if qa_is_malformed:
                    summary["malformed_qas"] += 1

                if not isinstance(answers, list):
                    continue

                if not answers:
                    summary["empty_answers"] += 1
                    continue

                for answer in answers:
                    if not isinstance(answer, dict):
                        summary["malformed_answers"] += 1
                        continue

                    text = answer.get("text")
                    answer_start = answer.get("answer_start")

                    if not isinstance(text, str):
                        summary["malformed_answers"] += 1
                        continue

                    if not text.strip():
                        summary["empty_answers"] += 1

                    if not isinstance(answer_start, int) or answer_start < 0:
                        summary["invalid_answer_start"] += 1
                        continue

                    answer_end = answer_start + len(text)
                    if answer_start > len(context) or answer_end > len(context):
                        summary["invalid_answer_start"] += 1
                        continue

                    if context[answer_start:answer_end] != text:
                        summary["answer_text_mismatch"] += 1

    return summary


def extract_valid_qa_records(payload: dict[str, Any]) -> list[QaRecord]:
    records: list[QaRecord] = []
    seen_qids: set[str] = set()
    data_items = payload.get("data")
    if not isinstance(data_items, list):
        return records

    for item_index, item in enumerate(data_items):
        if not isinstance(item, dict):
            continue

        paragraphs = item.get("paragraphs")
        if not isinstance(paragraphs, list):
            continue

        for paragraph_index, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, dict):
                continue

            qas = paragraph.get("qas")
            if not isinstance(qas, list):
                continue

            for qa_index, qa in enumerate(qas):
                if not isinstance(qa, dict):
                    continue

                qid = qa.get("id")
                question = qa.get("question")
                answers = qa.get("answers")
                if not isinstance(qid, str) or not qid.strip():
                    continue
                if qid in seen_qids:
                    continue
                if not isinstance(question, str) or not question.strip():
                    continue
                if not _has_non_empty_answers(answers):
                    continue

                seen_qids.add(qid)
                records.append(
                    QaRecord(
                        item_index=item_index,
                        paragraph_index=paragraph_index,
                        qa_index=qa_index,
                        qid=qid,
                        question=question,
                        bucket=classify_question(question),
                    )
                )

    return records


def bucket_counts(records: list[QaRecord]) -> dict[str, int]:
    counts = {bucket: 0 for bucket in BUCKET_ORDER}
    for record in records:
        counts[record.bucket] += 1
    return counts


def select_qa_records(records: list[QaRecord], n_qas: int, mode: str, seed: int) -> list[QaRecord]:
    if n_qas <= 0 or not records:
        return []

    target = min(n_qas, len(records))
    if mode == "first":
        return records[:target]

    if mode == "random":
        generator = random.Random(seed)
        shuffled = list(records)
        generator.shuffle(shuffled)
        return shuffled[:target]

    if mode == "stratified":
        generator = random.Random(seed)
        grouped = {bucket: [] for bucket in BUCKET_ORDER}
        for record in records:
            grouped[record.bucket].append(record)

        for bucket in BUCKET_ORDER:
            generator.shuffle(grouped[bucket])

        selected: list[QaRecord] = []
        while len(selected) < target:
            added_any = False
            for bucket in BUCKET_ORDER:
                if grouped[bucket] and len(selected) < target:
                    selected.append(grouped[bucket].pop(0))
                    added_any = True
            if not added_any:
                break

        return selected

    raise ValueError(f"Unsupported selection mode: {mode}")


def build_subset_payload(payload: dict[str, Any], selected_records: list[QaRecord]) -> dict[str, Any]:
    selected_positions = {(record.item_index, record.paragraph_index, record.qa_index) for record in selected_records}
    subset_payload = copy.deepcopy(payload)
    data_items = payload.get("data")
    if not isinstance(data_items, list):
        subset_payload["data"] = []
        return subset_payload

    subset_payload["data"] = []
    for item_index, item in enumerate(data_items):
        if not isinstance(item, dict):
            continue

        paragraphs = item.get("paragraphs")
        if not isinstance(paragraphs, list):
            continue

        kept_paragraphs: list[dict[str, Any]] = []
        for paragraph_index, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, dict):
                continue

            qas = paragraph.get("qas")
            if not isinstance(qas, list):
                continue

            kept_qas = [
                copy.deepcopy(qa)
                for qa_index, qa in enumerate(qas)
                if (item_index, paragraph_index, qa_index) in selected_positions
            ]
            if kept_qas:
                paragraph_copy = copy.deepcopy(paragraph)
                paragraph_copy["qas"] = kept_qas
                kept_paragraphs.append(paragraph_copy)

        if kept_paragraphs:
            item_copy = copy.deepcopy(item)
            item_copy["paragraphs"] = kept_paragraphs
            subset_payload["data"].append(item_copy)

    return subset_payload


def save_squad_dataset(path: Path, payload: dict[str, Any]) -> None:
    ensure_writable(path)  # never write into a protected folder (see protection.py)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _has_non_empty_answers(answers: Any) -> bool:
    if not isinstance(answers, list) or not answers:
        return False

    for answer in answers:
        if not isinstance(answer, dict):
            continue
        text = answer.get("text")
        answer_start = answer.get("answer_start")
        if isinstance(text, str) and text.strip() and isinstance(answer_start, int) and answer_start >= 0:
            return True
    return False
