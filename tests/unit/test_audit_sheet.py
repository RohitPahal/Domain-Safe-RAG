from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.generate_audit_sheet import AUDIT_CONFIGS, _fallback_output_path, write_audit_csv
from tools.generate_audit_sheet import app as audit_app
from typer.testing import CliRunner

RUNNER = CliRunner()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _write_config(path: Path) -> None:
    path.write_text("retrieve_k: 3\n", encoding="utf-8")


def _questions() -> list[dict]:
    return [
        {
            "qid": "q-1",
            "question": "What is BRCA1?",
            "doc_id": "doc-1",
            "answers": [{"text": "BRCA1", "answer_start": 0}],
        },
        {
            "qid": "q-2",
            "question": "What is IL-6?",
            "doc_id": "doc-2",
            "answers": [{"text": "IL-6", "answer_start": 0}],
        },
    ]


def _gold() -> list[dict]:
    return [
        {"qid": "q-1", "gold_doc_id": "doc-1", "gold_chunk_ids": ["doc-1::0"], "mapping_status": "full_cover"},
        {"qid": "q-2", "gold_doc_id": "doc-2", "gold_chunk_ids": ["doc-2::0"], "mapping_status": "full_cover"},
    ]


def test_generate_audit_sheet_runs_only_two_intended_configs_and_writes_expected_columns(tmp_path, monkeypatch) -> None:
    questions_path = tmp_path / "questions.jsonl"
    gold_path = tmp_path / "gold_evidence.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "audit_manual_sheet.csv"
    run_log_path = tmp_path / "audit_runs.jsonl"
    bm25_dir = tmp_path / "bm25"
    vector_dir = tmp_path / "vector"
    bm25_dir.mkdir()
    vector_dir.mkdir()
    chunks_path.write_text("", encoding="utf-8")
    _write_jsonl(questions_path, _questions())
    _write_jsonl(gold_path, _gold())
    _write_config(config_path)

    import tools.generate_audit_sheet as audit_script

    calls: list[tuple[str, str, str, str]] = []
    monkeypatch.setattr(audit_script, "load_bm25", lambda path: object())
    monkeypatch.setattr(audit_script, "load_chroma_index", lambda path, query_cache_dir=None: object())

    def fake_run_rag_workflow(**kwargs):
        calls.append(
            (
                kwargs["qid"],
                kwargs["retriever_mode"],
                kwargs["rewrite_mode"],
                kwargs["guard_mode"],
                kwargs.get("allow_llm_calls"),
                kwargs.get("generation_enabled"),
            )
        )
        if kwargs["qid"] == "q-1" and kwargs["rewrite_mode"] == "none":
            return {
                "retrieval_results": [{"chunk_id": "doc-1::0", "doc_id": "doc-1", "score": 1.0, "rank": 1}],
                "answer_text": "BRCA1 answer",
                "citations": ["doc-1::0"],
            }
        return {
            "retrieval_results": [{"chunk_id": "doc-x::0", "doc_id": "doc-x", "score": 1.0, "rank": 1}],
            "answer_text": None,
            "citations": [],
        }

    monkeypatch.setattr(audit_script, "run_rag_workflow", fake_run_rag_workflow)

    result = RUNNER.invoke(
        audit_app,
        [
            "--questions-path",
            str(questions_path),
            "--gold-path",
            str(gold_path),
            "--chunks-path",
            str(chunks_path),
            "--config-path",
            str(config_path),
            "--bm25-dir",
            str(bm25_dir),
            "--vector-dir",
            str(vector_dir),
            "--output-path",
            str(output_path),
            "--run-log-path",
            str(run_log_path),
        ],
    )

    assert result.exit_code == 0
    assert "audit_sheet_path:" in result.output

    rows = list(csv.DictReader(output_path.open("r", encoding="utf-8")))
    assert len(rows) == len(_questions()) * len(AUDIT_CONFIGS)
    assert list(rows[0].keys()) == [
        "qid",
        "question",
        "gold_answer",
        "gold_chunk_ids",
        "retrieved_chunk_ids",
        "predicted_answer",
        "evidence_found",
        "citations_valid",
        "config",
        "notes",
    ]
    assert rows[0]["config"] == "A_hybrid_none_guard_off"
    assert rows[1]["config"] == "B_hybrid_multi_guard_on"
    assert rows[0]["evidence_found"] == "true"
    assert rows[0]["citations_valid"] == "true"
    assert rows[1]["evidence_found"] == "false"
    assert rows[1]["citations_valid"] == "true"
    assert all(call[:4][1:] in {("hybrid", "none", "off"), ("hybrid", "multi", "on")} for call in calls)
    assert all(call[4] is False for call in calls)
    assert all(call[5] is None for call in calls)
    assert len(calls) == len(_questions()) * len(AUDIT_CONFIGS)


def test_citations_valid_is_false_when_citations_escape_retrieved_set(tmp_path, monkeypatch) -> None:
    questions_path = tmp_path / "questions.jsonl"
    gold_path = tmp_path / "gold_evidence.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "audit_manual_sheet.csv"
    run_log_path = tmp_path / "audit_runs.jsonl"
    bm25_dir = tmp_path / "bm25"
    vector_dir = tmp_path / "vector"
    bm25_dir.mkdir()
    vector_dir.mkdir()
    chunks_path.write_text("", encoding="utf-8")
    _write_jsonl(questions_path, _questions()[:1])
    _write_jsonl(gold_path, _gold()[:1])
    _write_config(config_path)

    import tools.generate_audit_sheet as audit_script

    monkeypatch.setattr(audit_script, "load_bm25", lambda path: object())
    monkeypatch.setattr(audit_script, "load_chroma_index", lambda path, query_cache_dir=None: object())
    monkeypatch.setattr(
        audit_script,
        "run_rag_workflow",
        lambda **kwargs: {
            "retrieval_results": [{"chunk_id": "doc-1::0", "doc_id": "doc-1", "score": 1.0, "rank": 1}],
            "answer_text": "BRCA1 answer",
            "citations": ["doc-missing::0"],
        },
    )

    result = RUNNER.invoke(
        audit_app,
        [
            "--questions-path",
            str(questions_path),
            "--gold-path",
            str(gold_path),
            "--chunks-path",
            str(chunks_path),
            "--config-path",
            str(config_path),
            "--bm25-dir",
            str(bm25_dir),
            "--vector-dir",
            str(vector_dir),
            "--output-path",
            str(output_path),
            "--run-log-path",
            str(run_log_path),
        ],
    )

    assert result.exit_code == 0
    rows = list(csv.DictReader(output_path.open("r", encoding="utf-8")))
    assert [row["citations_valid"] for row in rows] == ["false", "false"]


def test_write_audit_csv_falls_back_when_target_is_write_locked(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "audit_manual_sheet.csv"
    original_open = Path.open

    def fake_open(self, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if self == output_path and mode == "w":
            raise PermissionError("locked")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fake_open)

    actual_path = write_audit_csv(
        output_path,
        [
            {
                "qid": "q-1",
                "question": "What is BRCA1?",
                "gold_answer": "BRCA1",
                "gold_chunk_ids": '["doc-1::0"]',
                "retrieved_chunk_ids": '["doc-1::0"]',
                "predicted_answer": "",
                "evidence_found": "true",
                "citations_valid": "true",
                "config": "A_hybrid_none_guard_off",
                "notes": "",
            }
        ],
    )

    assert actual_path == _fallback_output_path(output_path)
    rows = list(csv.DictReader(actual_path.open("r", encoding="utf-8")))
    assert rows[0]["qid"] == "q-1"
