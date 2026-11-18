from __future__ import annotations

import csv
import json
from pathlib import Path

from rag_thesis.evaluation.answers import (
    evaluate_answer_quality,
    exact_match_normalized,
    lenient_containment_score,
    load_gold_answers,
    normalize_answer,
    score_prediction,
    token_f1_score,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_normalize_answer_is_case_punctuation_whitespace_and_article_insensitive() -> None:
    assert normalize_answer(" The, Alpha-Synuclein!  ") == "alpha synuclein"
    assert normalize_answer("An   epidermal growth factor receptor") == "epidermal growth factor receptor"


def test_exact_match_and_token_f1_use_normalized_forms() -> None:
    assert exact_match_normalized("Alpha-synuclein", "alpha synuclein") == 1.0
    assert token_f1_score("Alpha synuclein protein", "alpha synuclein") == 0.8


def test_lenient_containment_detects_short_gold_inside_longer_prediction() -> None:
    assert lenient_containment_score("Facioscapulohumeral dystrophy (FSHD)", "FSHD") == 1.0
    assert lenient_containment_score("connective tissue", "ribosomal protein genes") == 0.0
    assert lenient_containment_score("TAp73 isoforms are discussed here", "7") == 0.0


def test_load_gold_answers_unions_distinct_answers_for_identical_question_text(tmp_path) -> None:
    questions_path = tmp_path / "questions.jsonl"
    _write_jsonl(
        questions_path,
        [
            {
                "qid": "q-1",
                "question": "Which gene causes Mowat-Wilson syndrome?",
                "doc_id": "doc-1",
                "answers": [{"text": "ZFHX1B", "answer_start": 1}],
            },
            {
                "qid": "q-2",
                "question": "Which gene causes Mowat-Wilson syndrome?",
                "doc_id": "doc-2",
                "answers": [{"text": "ZEB2", "answer_start": 2}],
            },
        ],
    )

    gold_by_qid = load_gold_answers(questions_path)

    assert gold_by_qid["q-1"]["gold_answers"] == ["ZFHX1B", "ZEB2"]
    assert gold_by_qid["q-2"]["gold_answers"] == ["ZFHX1B", "ZEB2"]


def test_score_prediction_uses_best_matching_gold() -> None:
    result = score_prediction("ZEB2", ["ZFHX1B", "ZEB2"])

    assert result["best_matching_gold"] == "ZEB2"
    assert result["exact_match"] == 1.0
    assert result["token_f1"] == 1.0


def test_evaluate_answer_quality_generates_outputs_and_expected_metrics(tmp_path) -> None:
    run_log_path = tmp_path / "run_log.jsonl"
    questions_path = tmp_path / "questions.jsonl"
    gold_evidence_path = tmp_path / "gold_evidence.jsonl"
    reports_dir = tmp_path / "reports"

    _write_jsonl(
        questions_path,
        [
            {
                "qid": "q-1",
                "question": "What protein mainly makes up Lewy bodies?",
                "doc_id": "doc-1",
                "answers": [{"text": "Alpha-synuclein", "answer_start": 0}],
            },
            {
                "qid": "q-2",
                "question": "What protein mainly makes up Lewy bodies?",
                "doc_id": "doc-2",
                "answers": [{"text": "alpha synuclein", "answer_start": 5}],
            },
            {
                "qid": "q-3",
                "question": "How many TAp73 isoforms are known in humans?",
                "doc_id": "doc-3",
                "answers": [{"text": "7", "answer_start": 9}],
            },
        ],
    )
    _write_jsonl(
        gold_evidence_path,
        [
            {
                "qid": "q-1",
                "gold_doc_id": "doc-1",
                "gold_chunk_ids": ["doc-1::0"],
                "mapping_status": "full_cover",
            },
            {
                "qid": "q-2",
                "gold_doc_id": "doc-2",
                "gold_chunk_ids": ["doc-2::0"],
                "mapping_status": "full_cover",
            },
            {
                "qid": "q-3",
                "gold_doc_id": "doc-3",
                "gold_chunk_ids": ["doc-3::0"],
                "mapping_status": "full_cover",
            },
        ],
    )
    _write_jsonl(
        run_log_path,
        [
            {
                "qid": "q-1",
                "query_original": "What protein mainly makes up Lewy bodies?",
                "query_rewrites": ["What protein mainly makes up Lewy bodies?"],
                "guard_report": {},
                "retrieval_results": [{"chunk_id": "doc-1::0"}],
                "answer_text": "Alpha synuclein",
                "citations": ["doc-1::0"],
                "latency_ms": 10.0,
                "openai_usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "config_snapshot": {"retriever_mode": "hybrid", "rewrite_mode": "none", "guard_mode": "off", "k": 10},
            },
            {
                "qid": "q-3",
                "query_original": "How many TAp73 isoforms are known in humans?",
                "query_rewrites": ["How many TAp73 isoforms are known in humans?"],
                "guard_report": {},
                "retrieval_results": [{"chunk_id": "doc-3::0"}],
                "answer_text": None,
                "citations": [],
                "latency_ms": 12.0,
                "openai_usage": {},
                "config_snapshot": {"retriever_mode": "hybrid", "rewrite_mode": "none", "guard_mode": "off", "k": 10},
            },
        ],
    )

    outputs = evaluate_answer_quality(
        run_log_path=run_log_path,
        questions_path=questions_path,
        gold_evidence_path=gold_evidence_path,
        reports_dir=reports_dir,
    )

    assert outputs["summary"] == {
        "total_questions": 2,
        "answered": 1,
        "unanswered": 1,
        "answer_rate": 0.5,
        "exact_match_normalized": 0.5,
        "token_f1": 0.5,
        "lenient_containment_rate": 0.5,
    }

    per_question_path = outputs["per_question_path"]
    summary_path = outputs["summary_path"]
    assert per_question_path.exists()
    assert summary_path.exists()

    with per_question_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["best_matching_gold"] == "Alpha-synuclein"
    assert rows[0]["lenient_containment"] == "1"
    assert rows[1]["token_f1"] == "0.0"

    summary_text = summary_path.read_text(encoding="utf-8")
    assert "exact_match_normalized: 0.500000" in summary_text
    assert "token_f1: 0.500000" in summary_text
