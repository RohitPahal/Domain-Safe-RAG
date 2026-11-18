"""Validate processed and evaluation JSONL artifacts against their Pydantic schemas.

The command-line wrapper is ``rag_thesis.cli.validate`` (``rag-thesis validate``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rag_thesis.schemas import (
    ChunkLine,
    DocumentLine,
    GoldEvidenceLine,
    QuestionLine,
    RunLogLine,
)

try:
    import orjson
except ImportError:  # pragma: no cover
    import json

    class _OrjsonCompat:
        JSONDecodeError = json.JSONDecodeError

        @staticmethod
        def loads(value: bytes) -> Any:
            return json.loads(value.decode("utf-8"))

    orjson = _OrjsonCompat()


SCHEMA_BY_FILENAME = {
    "documents.jsonl": DocumentLine,
    "chunks.jsonl": ChunkLine,
    "questions.jsonl": QuestionLine,
    "gold_evidence.jsonl": GoldEvidenceLine,
    "run_log.jsonl": RunLogLine,
}


def _format_validation_error(error: dict[str, Any]) -> str:
    location = ".".join(str(part) for part in error.get("loc", ()))
    message = error.get("msg", "validation error")
    return f"{location}: {message}" if location else message


def _iter_jsonl_lines(
    path: Path, schema: type[DocumentLine | ChunkLine | QuestionLine | GoldEvidenceLine | RunLogLine]
) -> tuple[int, int, list[str]]:
    total = 0
    valid = 0
    errors: list[str] = []

    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                errors.append(f"{path}: line {line_number}: empty line is not allowed")
                continue

            total += 1

            try:
                payload = orjson.loads(raw_line)
            except orjson.JSONDecodeError as exc:
                errors.append(f"{path}: line {line_number}: invalid JSON: {exc}")
                continue

            try:
                schema.model_validate(payload)
            except ValidationError as exc:
                detail = "; ".join(_format_validation_error(error) for error in exc.errors())
                errors.append(f"{path}: line {line_number}: {detail}")
                continue

            valid += 1

    return total, valid, errors


@dataclass(frozen=True)
class ValidationSummary:
    """Record counts and error messages from one validation of the artifact folders."""

    total: int
    valid: int
    invalid: int
    errors: list[str] = field(default_factory=list)


def validate_artifacts(processed_dir: Path, eval_dir: Path) -> ValidationSummary:
    """Check that the required artifacts exist and that every JSONL line matches its schema."""
    total = 0
    valid = 0
    invalid = 0
    errors: list[str] = []

    required_file_map = {
        processed_dir: ("documents.jsonl", "chunks.jsonl"),
        eval_dir: ("questions.jsonl", "gold_evidence.jsonl"),
    }
    optional_file_map = {
        eval_dir: ("run_log.jsonl",),
    }

    required_files = {
        base_dir / filename for base_dir, filenames in required_file_map.items() for filename in filenames
    }
    optional_files = {
        base_dir / filename for base_dir, filenames in optional_file_map.items() for filename in filenames
    }
    allowed_files = required_files | optional_files

    for expected_file in sorted(required_files):
        if not expected_file.exists():
            errors.append(f"Missing required file: {expected_file}")

    for base_dir in (processed_dir, eval_dir):
        for path in sorted(base_dir.glob("*.jsonl")):
            if path not in allowed_files:
                errors.append(f"Unsupported artifact file: {path}")
                continue

            schema = SCHEMA_BY_FILENAME[path.name]
            file_total, file_valid, file_errors = _iter_jsonl_lines(path, schema)
            total += file_total
            valid += file_valid
            invalid += len(file_errors)
            errors.extend(file_errors)

    return ValidationSummary(total=total, valid=valid, invalid=invalid, errors=errors)
