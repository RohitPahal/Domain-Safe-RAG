"""Shared JSONL artifact loaders (rag_thesis/io.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_thesis.evaluation.answers import load_gold_evidence as load_optional_gold_evidence
from rag_thesis.io import (
    load_chunks,
    load_chunks_by_id,
    load_gold_evidence,
    load_questions,
    load_questions_by_id,
    read_jsonl,
)
from rag_thesis.schemas import QuestionLine


def _question(qid: str) -> dict:
    return {
        "qid": qid,
        "question": f"Question {qid}?",
        "doc_id": "d1",
        "answers": [{"text": "BRCA1", "answer_start": 0}],
    }


def _chunk(chunk_id: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": "d1",
        "chunk_index": 0,
        "text": "BRCA1 text",
        "char_start": 0,
        "char_end": 10,
        "metadata": {},
    }


def _write(path: Path, rows: list, raw_lines: list[str] | None = None) -> Path:
    lines = [json.dumps(row) for row in rows] + (raw_lines or [])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_read_jsonl_validates_and_returns_plain_dicts(tmp_path):
    rows = read_jsonl(_write(tmp_path / "q.jsonl", [_question("q1"), _question("q2")]), QuestionLine)
    assert [row["qid"] for row in rows] == ["q1", "q2"]
    assert isinstance(rows[0], dict)


def test_max_questions_stops_early(tmp_path):
    path = _write(tmp_path / "q.jsonl", [_question("q1"), _question("q2")], raw_lines=["not json"])
    assert [q["qid"] for q in load_questions(path, max_qids=2)] == ["q1", "q2"]  # never reads the bad line


@pytest.mark.parametrize(
    "bad_line, message",
    [
        ("", "Empty line found .* at line 2"),
        ("{not json", "Invalid QuestionLine record .* at line 2"),
        (json.dumps({"qid": "q2"}), "Invalid QuestionLine record .* at line 2"),  # missing fields
    ],
    ids=["empty_line", "invalid_json", "invalid_record"],
)
def test_errors_name_the_file_and_line(tmp_path, bad_line, message):
    path = tmp_path / "q.jsonl"
    path.write_text(
        json.dumps(_question("q1")) + "\n" + bad_line + "\n" + json.dumps(_question("q3")) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=message):
        load_questions(path)


def test_keyed_loaders_refuse_duplicate_ids(tmp_path):
    questions = _write(tmp_path / "q.jsonl", [_question("q1"), _question("q1")])
    assert len(load_questions(questions)) == 2  # the list loader keeps file order as-is
    with pytest.raises(ValueError, match="Duplicate qid 'q1'"):
        load_questions_by_id(questions)

    chunks = _write(tmp_path / "c.jsonl", [_chunk("d1::0"), _chunk("d1::0")])
    with pytest.raises(ValueError, match="Duplicate chunk_id 'd1::0'"):
        load_chunks_by_id(chunks)


def test_chunk_loaders(tmp_path):
    path = _write(tmp_path / "c.jsonl", [_chunk("d1::0"), _chunk("d1::1")])
    assert [c["chunk_id"] for c in load_chunks(path)] == ["d1::0", "d1::1"]
    assert set(load_chunks_by_id(path)) == {"d1::0", "d1::1"}

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No chunks found"):
        load_chunks(empty)


def test_gold_evidence_loaders(tmp_path):
    path = _write(tmp_path / "g.jsonl", [{"qid": "q1", "gold_doc_id": "d1", "gold_chunk_ids": ["d1::0"]}])
    assert load_gold_evidence(path)["q1"]["gold_chunk_ids"] == ["d1::0"]
    # eval-answers treats the gold file as optional:
    assert load_optional_gold_evidence(None) == {}
    assert load_optional_gold_evidence(tmp_path / "missing.jsonl") == {}
    assert load_optional_gold_evidence(path) == load_gold_evidence(path)
