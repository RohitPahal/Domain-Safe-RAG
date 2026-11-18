"""Reading the project's JSONL artifacts: one place, one set of rules.

Every artifact (questions, chunks, gold evidence) is a JSONL file: one JSON object per line,
validated with its Pydantic model from ``schemas.py``. All loaders here apply the same rules:

- an empty line is an error (it usually means a truncated or hand-edited file);
- invalid JSON or an invalid record is an error that names the file and line number;
- loaders that return a dict keyed by id refuse duplicate ids (instead of silently keeping
  only the last record);
- an empty chunks file is an error (no retrieval is possible without chunks).

Records are returned as plain dicts (``model_dump()``), as the rest of the code expects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from rag_thesis.schemas import ChunkLine, GoldEvidenceLine, QuestionLine


def read_jsonl(path: Path | str, model: type[BaseModel], max_records: int | None = None) -> list[dict[str, Any]]:
    """Read ``path`` line by line, validate each line with ``model``, return the records as dicts.

    Stops after ``max_records`` records if given.
    """
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if max_records is not None and len(records) >= max_records:
                break
            if not line.strip():
                raise ValueError(f"Empty line found in {path} at line {line_number}.")
            try:
                records.append(model.model_validate(json.loads(line)).model_dump())
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"Invalid {model.__name__} record in {path} at line {line_number}: {exc}") from exc
    return records


def index_by(records: list[dict[str, Any]], key: str, source: Path | str) -> dict[str, dict[str, Any]]:
    """Map ``record[key] -> record``; raise on a duplicate key instead of silently overwriting."""
    by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        value = str(record[key])
        if value in by_key:
            raise ValueError(f"Duplicate {key} '{value}' in {source}.")
        by_key[value] = record
    return by_key


def load_questions(path: Path | str, max_qids: int | None = None) -> list[dict[str, Any]]:
    """Questions in file order (optionally only the first ``max_qids``)."""
    return read_jsonl(path, QuestionLine, max_records=max_qids)


def load_questions_by_id(path: Path | str) -> dict[str, dict[str, Any]]:
    """Questions keyed by ``qid``."""
    return index_by(load_questions(path), "qid", path)


def load_chunks(path: Path | str) -> list[dict[str, Any]]:
    """Chunks in file order; an empty file is an error."""
    chunks = read_jsonl(path, ChunkLine)
    if not chunks:
        raise ValueError(f"No chunks found in {path}.")
    return chunks


def load_chunks_by_id(path: Path | str) -> dict[str, dict[str, Any]]:
    """Chunks keyed by ``chunk_id``."""
    return index_by(load_chunks(path), "chunk_id", path)


def load_gold_evidence(path: Path | str) -> dict[str, dict[str, Any]]:
    """Gold evidence keyed by ``qid``."""
    return index_by(read_jsonl(path, GoldEvidenceLine), "qid", path)
