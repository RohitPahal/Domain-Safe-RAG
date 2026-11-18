from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def _reject_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class AnswerSpan(StrictBaseModel):
    text: str
    answer_start: int = Field(ge=0)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _reject_empty_string(value, "text")


class DocumentLine(StrictBaseModel):
    doc_id: str
    text: str
    metadata: dict[str, Any]

    @field_validator("doc_id", "text")
    @classmethod
    def validate_non_empty_fields(cls, value: str, info: Any) -> str:
        return _reject_empty_string(value, info.field_name)


class ChunkLine(StrictBaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int = Field(ge=0)
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    metadata: dict[str, Any]

    @field_validator("chunk_id", "doc_id", "text")
    @classmethod
    def validate_non_empty_fields(cls, value: str, info: Any) -> str:
        return _reject_empty_string(value, info.field_name)

    @model_validator(mode="after")
    def validate_char_window(self) -> ChunkLine:
        if self.char_end < self.char_start:
            raise ValueError("char_end must be greater than or equal to char_start")
        return self


class QuestionLine(StrictBaseModel):
    qid: str
    question: str
    doc_id: str
    answers: list[AnswerSpan]

    @field_validator("qid", "question", "doc_id")
    @classmethod
    def validate_non_empty_fields(cls, value: str, info: Any) -> str:
        return _reject_empty_string(value, info.field_name)


class GoldEvidenceLine(StrictBaseModel):
    qid: str
    gold_doc_id: str
    gold_chunk_ids: list[str]
    mapping_status: str | None = None

    @field_validator("qid", "gold_doc_id")
    @classmethod
    def validate_non_empty_ids(cls, value: str, info: Any) -> str:
        return _reject_empty_string(value, info.field_name)

    @field_validator("gold_chunk_ids")
    @classmethod
    def validate_chunk_ids(cls, value: list[str]) -> list[str]:
        for chunk_id in value:
            _reject_empty_string(chunk_id, "gold_chunk_ids[]")
        return value

    @field_validator("mapping_status")
    @classmethod
    def validate_mapping_status(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _reject_empty_string(value, "mapping_status")


class RunLogLine(StrictBaseModel):
    qid: str
    query_original: str
    query_rewrites: list[str]
    guard_report: dict[str, Any]
    retrieval_results: list[dict[str, Any]]
    answer_text: str | None
    citations: list[str]
    latency_ms: float = Field(ge=0)
    # Answer-generation tokens. Older logs (e.g. the official benchmark) call it
    # `openai_usage` (even for Gemini); readers use `run_log_usage(row)` for either.
    usage: dict[str, Any] | None = None
    openai_usage: dict[str, Any] | None = None
    # One record per real LLM call of this question (role, provider, model, tokens, latency, status).
    llm_calls: list[dict[str, Any]] | None = None
    config_snapshot: dict[str, Any]
    # What happened in the rewrite and generation steps, so a fallback is never silent.
    # Optional: logs written before these fields existed (e.g. the official benchmark) stay valid.
    rewrite_status: Literal["not_requested", "ok", "failed", "skipped_confident"] | None = None
    rewrite_error: str | None = None  # set when rewrite_status == "failed"
    # R5: the rewrite gate's decision (confidence, threshold, rewritten); only when the gate ran.
    rewrite_gate: dict[str, Any] | None = None
    # Guardrail findings (question / passages / answer) and what was blocked; only when guardrails are on.
    guardrail: dict[str, Any] | None = None
    generation_status: (
        Literal["no_retrieval", "disabled", "answered", "not_found", "refused", "blocked", "failed"] | None
    ) = None
    generation_error: str | None = None  # set when generation_status == "failed"

    @field_validator("qid", "query_original")
    @classmethod
    def validate_non_empty_fields(cls, value: str, info: Any) -> str:
        return _reject_empty_string(value, info.field_name)

    @field_validator("answer_text")
    @classmethod
    def validate_answer_text(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _reject_empty_string(value, "answer_text")

    @field_validator("query_rewrites", "citations")
    @classmethod
    def validate_string_lists(cls, value: list[str], info: Any) -> list[str]:
        for item in value:
            _reject_empty_string(item, f"{info.field_name}[]")
        return value


def run_log_usage(row: dict[str, Any]) -> dict[str, Any]:
    """Answer-generation token usage of a run-log row, new (`usage`) or legacy (`openai_usage`) logs."""
    return dict(row.get("usage") or row.get("openai_usage") or {})
