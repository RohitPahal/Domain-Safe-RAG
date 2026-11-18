from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_thesis.data.squad import load_squad_dataset
from rag_thesis.ids import deterministic_hash
from rag_thesis.protection import ensure_writable
from rag_thesis.schemas import (
    AnswerSpan,
    ChunkLine,
    DocumentLine,
    GoldEvidenceLine,
    QuestionLine,
)


@dataclass(frozen=True)
class ChunkWindow:
    chunk_id: str
    chunk_index: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class MappingResult:
    gold_chunk_ids: list[str]
    mapping_status: str
    failure_reasons: list[str]


def build_corpus_artifacts(
    input_path: Path,
    processed_dir: Path,
    eval_dir: Path,
    report_dir: Path,
    tokenizer: Any,
    tokenizer_name: str,
    chunk_tokens: int = 180,
    chunk_overlap: int = 40,
) -> dict[str, int]:
    payload = load_squad_dataset(input_path)
    for directory in (processed_dir, eval_dir, report_dir):
        ensure_writable(directory)  # never write into a protected folder (see protection.py)
    processed_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    documents_path = processed_dir / "documents.jsonl"
    chunks_path = processed_dir / "chunks.jsonl"
    questions_path = eval_dir / "questions.jsonl"
    gold_path = eval_dir / "gold_evidence.jsonl"
    failures_path = report_dir / "mapping_failures.jsonl"

    context_index: dict[str, dict[str, Any]] = {}
    document_lines: list[dict[str, Any]] = []
    chunk_lines: list[dict[str, Any]] = []
    question_lines: list[dict[str, Any]] = []
    gold_lines: list[dict[str, Any]] = []
    failure_lines: list[dict[str, Any]] = []

    data_items = payload.get("data")
    if not isinstance(data_items, list):
        raise ValueError("Input dataset must contain a top-level 'data' list.")

    for item_index, item in enumerate(data_items):
        if not isinstance(item, dict):
            continue

        paragraphs = item.get("paragraphs")
        if not isinstance(paragraphs, list):
            continue

        for paragraph_index, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, dict):
                continue

            context = paragraph.get("context")
            qas = paragraph.get("qas")
            if not isinstance(context, str) or not context:
                continue
            if not isinstance(qas, list):
                continue

            if context not in context_index:
                doc_id = deterministic_hash(context)
                document_line = DocumentLine.model_validate(
                    {
                        "doc_id": doc_id,
                        "text": context,
                        "metadata": {
                            "context_char_length": len(context),
                        },
                    }
                )
                built_chunk_lines, chunk_windows = build_chunks_for_document(
                    context=context,
                    doc_id=doc_id,
                    tokenizer=tokenizer,
                    tokenizer_name=tokenizer_name,
                    chunk_tokens=chunk_tokens,
                    chunk_overlap=chunk_overlap,
                )
                context_index[context] = {
                    "doc_id": doc_id,
                    "chunk_windows": chunk_windows,
                }
                document_lines.append(document_line.model_dump())
                chunk_lines.extend(chunk.model_dump() for chunk in built_chunk_lines)

            doc_id = context_index[context]["doc_id"]
            chunk_windows = context_index[context]["chunk_windows"]

            for qa_index, qa in enumerate(qas):
                if not isinstance(qa, dict):
                    continue

                qid = qa.get("id")
                question = qa.get("question")
                raw_answers = qa.get("answers")
                if not isinstance(qid, str) or not qid.strip():
                    continue
                if not isinstance(question, str) or not question.strip():
                    continue
                if not isinstance(raw_answers, list):
                    raw_answers = []

                answers = _normalize_answers(raw_answers)
                question_line = QuestionLine.model_validate(
                    {
                        "qid": qid,
                        "question": question,
                        "doc_id": doc_id,
                        "answers": [answer.model_dump() for answer in answers],
                    }
                )
                question_lines.append(question_line.model_dump())

                mapping = map_answers_to_chunks(
                    context=context,
                    answers=answers,
                    chunk_windows=chunk_windows,
                )
                gold_line = GoldEvidenceLine.model_validate(
                    {
                        "qid": qid,
                        "gold_doc_id": doc_id,
                        "gold_chunk_ids": mapping.gold_chunk_ids,
                        "mapping_status": mapping.mapping_status,
                    }
                )
                gold_lines.append(gold_line.model_dump())

                if mapping.mapping_status == "failed":
                    failure_lines.append(
                        {
                            "qid": qid,
                            "doc_id": doc_id,
                            "item_index": item_index,
                            "paragraph_index": paragraph_index,
                            "qa_index": qa_index,
                            "answers": [answer.model_dump() for answer in answers],
                            "failure_reasons": mapping.failure_reasons,
                        }
                    )

    write_jsonl(documents_path, document_lines)
    write_jsonl(chunks_path, chunk_lines)
    write_jsonl(questions_path, question_lines)
    write_jsonl(gold_path, gold_lines)
    write_jsonl(failures_path, failure_lines)

    return {
        "documents": len(document_lines),
        "chunks": len(chunk_lines),
        "questions": len(question_lines),
        "gold_evidence": len(gold_lines),
        "mapping_failures": len(failure_lines),
    }


def build_chunks_for_document(
    context: str,
    doc_id: str,
    tokenizer: Any,
    tokenizer_name: str,
    chunk_tokens: int,
    chunk_overlap: int,
) -> tuple[list[ChunkLine], list[ChunkWindow]]:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_tokens:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_tokens.")

    token_spans = _token_offsets_for_context(context=context, tokenizer=tokenizer)
    if not token_spans:
        chunk_line = ChunkLine.model_validate(
            {
                "chunk_id": f"{doc_id}::0",
                "doc_id": doc_id,
                "chunk_index": 0,
                "text": context,
                "char_start": 0,
                "char_end": len(context),
                "metadata": {
                    "token_start": 0,
                    "token_end": 0,
                    "token_count": 0,
                    "tokenizer_name": tokenizer_name,
                },
            }
        )
        return [chunk_line], [
            ChunkWindow(chunk_id=chunk_line.chunk_id, chunk_index=0, char_start=0, char_end=len(context))
        ]

    if len(token_spans) <= chunk_tokens:
        chunk_line = ChunkLine.model_validate(
            {
                "chunk_id": f"{doc_id}::0",
                "doc_id": doc_id,
                "chunk_index": 0,
                "text": context,
                "char_start": 0,
                "char_end": len(context),
                "metadata": {
                    "token_start": 0,
                    "token_end": len(token_spans),
                    "token_count": len(token_spans),
                    "tokenizer_name": tokenizer_name,
                },
            }
        )
        return [chunk_line], [
            ChunkWindow(chunk_id=chunk_line.chunk_id, chunk_index=0, char_start=0, char_end=len(context))
        ]

    chunk_lines: list[ChunkLine] = []
    chunk_windows: list[ChunkWindow] = []
    step = chunk_tokens - chunk_overlap
    start_token_index = 0
    chunk_index = 0

    while start_token_index < len(token_spans):
        end_token_index = min(start_token_index + chunk_tokens, len(token_spans))
        char_start = token_spans[start_token_index][0]
        char_end = token_spans[end_token_index - 1][1]
        chunk_line = ChunkLine.model_validate(
            {
                "chunk_id": f"{doc_id}::{chunk_index}",
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "text": context[char_start:char_end],
                "char_start": char_start,
                "char_end": char_end,
                "metadata": {
                    "token_start": start_token_index,
                    "token_end": end_token_index,
                    "token_count": end_token_index - start_token_index,
                    "tokenizer_name": tokenizer_name,
                },
            }
        )
        chunk_lines.append(chunk_line)
        chunk_windows.append(
            ChunkWindow(
                chunk_id=chunk_line.chunk_id,
                chunk_index=chunk_index,
                char_start=char_start,
                char_end=char_end,
            )
        )

        if end_token_index == len(token_spans):
            break

        start_token_index += step
        chunk_index += 1

    return chunk_lines, chunk_windows


def map_answers_to_chunks(
    context: str,
    answers: list[AnswerSpan],
    chunk_windows: list[ChunkWindow],
) -> MappingResult:
    if not answers:
        return MappingResult(gold_chunk_ids=[], mapping_status="failed", failure_reasons=["no_valid_answers"])

    selected_windows: list[ChunkWindow] = []
    saw_overlap_only = False
    failure_reasons: list[str] = []

    for answer in answers:
        answer_start = answer.answer_start
        answer_end = answer_start + len(answer.text)
        if answer_end > len(context):
            failure_reasons.append(f"answer_span_out_of_bounds:{answer_start}:{answer_end}:{len(context)}")
            continue

        full_cover = [
            chunk for chunk in chunk_windows if chunk.char_start <= answer_start and answer_end <= chunk.char_end
        ]
        if full_cover:
            selected_windows.extend(full_cover)
            continue

        overlaps = [chunk for chunk in chunk_windows if chunk.char_start < answer_end and answer_start < chunk.char_end]
        if overlaps:
            selected_windows.extend(overlaps)
            saw_overlap_only = True
            continue

        failure_reasons.append(f"no_chunk_overlap:{answer_start}:{answer_end}")

    deduplicated = {chunk.chunk_id: chunk for chunk in selected_windows}
    gold_chunk_ids = [chunk.chunk_id for chunk in sorted(deduplicated.values(), key=lambda chunk: chunk.chunk_index)]

    if failure_reasons:
        return MappingResult(
            gold_chunk_ids=gold_chunk_ids,
            mapping_status="failed",
            failure_reasons=failure_reasons,
        )
    if saw_overlap_only:
        return MappingResult(
            gold_chunk_ids=gold_chunk_ids,
            mapping_status="overlap_only",
            failure_reasons=[],
        )
    return MappingResult(
        gold_chunk_ids=gold_chunk_ids,
        mapping_status="full_cover",
        failure_reasons=[],
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_writable(path)  # never write into a protected folder (see protection.py)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _normalize_answers(raw_answers: list[Any]) -> list[AnswerSpan]:
    answers: list[AnswerSpan] = []
    for raw_answer in raw_answers:
        if not isinstance(raw_answer, dict):
            continue

        text = raw_answer.get("text")
        answer_start = raw_answer.get("answer_start")
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(answer_start, int) or answer_start < 0:
            continue

        answers.append(
            AnswerSpan.model_validate(
                {
                    "text": text,
                    "answer_start": answer_start,
                }
            )
        )

    return answers


def _token_offsets_for_context(context: str, tokenizer: Any) -> list[tuple[int, int]]:
    encoded = tokenizer(
        context,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        max_length=None,
        verbose=False,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError("Tokenizer must return offset mappings for deterministic chunk projection.")

    normalized_offsets: list[tuple[int, int]] = []
    for start, end in offsets:
        if end <= start:
            continue
        normalized_offsets.append((int(start), int(end)))

    return normalized_offsets


def load_embed_tokenizer(model_name: str):
    """Load the embedding model's own tokenizer; chunk sizes are counted in its wordpiece tokens."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except ImportError:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        return model.tokenizer
