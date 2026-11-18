from __future__ import annotations

import csv
import json
from pathlib import Path

from rag_thesis.evaluation.report import run_qa_report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _write_audit_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_row(
    *,
    qid: str,
    retriever_mode: str,
    rewrite_mode: str,
    guard_mode: str,
    k: int,
    answer_model: str,
    answer_text: str | None,
    citations: list[str],
    retrieval_chunk_ids: list[str],
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict:
    total_tokens = prompt_tokens + completion_tokens
    retrieval_results = [
        {
            "chunk_id": chunk_id,
            "doc_id": chunk_id.split("::", 1)[0],
            "rank": index,
            "score": float(len(retrieval_chunk_ids) - index + 1),
        }
        for index, chunk_id in enumerate(retrieval_chunk_ids, start=1)
    ]
    return {
        "qid": qid,
        "query_original": f"Question {qid}",
        "query_rewrites": [f"Question {qid}"],
        "guard_report": {"guard_mode": guard_mode, "reports": []},
        "retrieval_results": retrieval_results,
        "answer_text": answer_text,
        "citations": citations,
        "latency_ms": latency_ms,
        "openai_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        if total_tokens
        else {},
        "config_snapshot": {
            "retriever_mode": retriever_mode,
            "rewrite_mode": rewrite_mode,
            "guard_mode": guard_mode,
            "k": k,
            "llm_answer_model": answer_model,
        },
    }


def test_run_qa_report_aggregates_metrics_and_prefers_canonical_audit_sheet(tmp_path) -> None:
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    runs_dir.mkdir()
    reports_dir.mkdir()

    _write_jsonl(
        runs_dir / "run_log.jsonl",
        [
            _run_row(
                qid="q-1",
                retriever_mode="hybrid",
                rewrite_mode="multi",
                guard_mode="on",
                k=10,
                answer_model="model-a",
                answer_text="Answered.",
                citations=["doc-1::0"],
                retrieval_chunk_ids=["doc-1::0", "doc-2::0"],
                latency_ms=100.0,
                prompt_tokens=100,
                completion_tokens=50,
            ),
            _run_row(
                qid="q-2",
                retriever_mode="hybrid",
                rewrite_mode="multi",
                guard_mode="on",
                k=10,
                answer_model="model-a",
                answer_text=None,
                citations=[],
                retrieval_chunk_ids=["doc-3::0", "doc-4::0"],
                latency_ms=300.0,
                prompt_tokens=0,
                completion_tokens=0,
            ),
        ],
    )
    _write_jsonl(
        runs_dir / "extra_runs.jsonl",
        [
            _run_row(
                qid="q-3",
                retriever_mode="hybrid",
                rewrite_mode="none",
                guard_mode="off",
                k=10,
                answer_model="model-a",
                answer_text="Another answer.",
                citations=["doc-missing::0"],
                retrieval_chunk_ids=["doc-5::0", "doc-6::0"],
                latency_ms=50.0,
                prompt_tokens=20,
                completion_tokens=10,
            )
        ],
    )

    _write_audit_csv(
        reports_dir / "audit_manual_sheet.csv",
        [
            {
                "qid": "q-1",
                "question": "Question q-1",
                "gold_answer": "A",
                "gold_chunk_ids": '["doc-1::0"]',
                "retrieved_chunk_ids": '["doc-1::0"]',
                "predicted_answer": "",
                "evidence_found": "true",
                "citations_valid": "true",
                "config": "A_hybrid_none_guard_off",
                "notes": "",
            },
            {
                "qid": "q-1",
                "question": "Question q-1",
                "gold_answer": "A",
                "gold_chunk_ids": '["doc-1::0"]',
                "retrieved_chunk_ids": '["doc-1::0"]',
                "predicted_answer": "",
                "evidence_found": "true",
                "citations_valid": "true",
                "config": "B_hybrid_multi_guard_on",
                "notes": "",
            },
            {
                "qid": "q-2",
                "question": "Question q-2",
                "gold_answer": "A",
                "gold_chunk_ids": '["doc-x::0"]',
                "retrieved_chunk_ids": '["doc-y::0"]',
                "predicted_answer": "",
                "evidence_found": "false",
                "citations_valid": "true",
                "config": "B_hybrid_multi_guard_on",
                "notes": "",
            },
        ],
    )
    _write_audit_csv(
        reports_dir / "audit_manual_sheet.generated.csv",
        [
            {
                "qid": "q-1",
                "question": "Question q-1",
                "gold_answer": "A",
                "gold_chunk_ids": '["doc-1::0"]',
                "retrieved_chunk_ids": '["doc-z::0"]',
                "predicted_answer": "",
                "evidence_found": "false",
                "citations_valid": "true",
                "config": "B_hybrid_multi_guard_on",
                "notes": "",
            }
        ],
    )

    outputs = run_qa_report(
        runs_dir=runs_dir,
        reports_dir=reports_dir,
        prompt_price_per_1k_usd=0.01,
        completion_price_per_1k_usd=0.02,
    )

    assert outputs["audit_path"] == reports_dir / "audit_manual_sheet.csv"

    qa_rows = list(csv.DictReader((reports_dir / "qa_metrics.csv").open("r", encoding="utf-8")))
    cost_rows = list(csv.DictReader((reports_dir / "cost_latency.csv").open("r", encoding="utf-8")))
    summary_text = (reports_dir / "final_ablation_summary.md").read_text(encoding="utf-8")

    assert list(qa_rows[0].keys()) == [
        "retriever_mode",
        "rewrite_mode",
        "guard_mode",
        "k",
        "answer_model",
        "total_questions_attempted",
        "total_answered",
        "unanswered_rate",
        "citation_valid_rate",
        "evidence_found_rate",
    ]
    assert list(cost_rows[0].keys()) == [
        "retriever_mode",
        "rewrite_mode",
        "guard_mode",
        "k",
        "answer_model",
        "total_questions_attempted",
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_tokens",
        "avg_tokens_per_question",
        "estimated_cost_usd",
    ]

    hybrid_multi_row = next(
        row for row in qa_rows if row["retriever_mode"] == "hybrid" and row["rewrite_mode"] == "multi"
    )
    assert hybrid_multi_row["guard_mode"] == "on"
    assert hybrid_multi_row["total_questions_attempted"] == "2"
    assert hybrid_multi_row["total_answered"] == "1"
    assert hybrid_multi_row["unanswered_rate"] == "0.5"
    assert hybrid_multi_row["citation_valid_rate"] == "1.0"
    assert hybrid_multi_row["evidence_found_rate"] == "0.5"

    hybrid_none_row = next(
        row for row in qa_rows if row["retriever_mode"] == "hybrid" and row["rewrite_mode"] == "none"
    )
    assert hybrid_none_row["guard_mode"] == "off"
    assert hybrid_none_row["citation_valid_rate"] == "0.0"
    assert hybrid_none_row["evidence_found_rate"] == "1.0"

    hybrid_multi_cost = next(
        row for row in cost_rows if row["retriever_mode"] == "hybrid" and row["rewrite_mode"] == "multi"
    )
    assert hybrid_multi_cost["avg_latency_ms"] == "200.0"
    assert hybrid_multi_cost["p50_latency_ms"] == "200.0"
    assert hybrid_multi_cost["p95_latency_ms"] == "290.0"
    assert hybrid_multi_cost["total_prompt_tokens"] == "100"
    assert hybrid_multi_cost["total_completion_tokens"] == "50"
    assert hybrid_multi_cost["total_tokens"] == "150"
    assert hybrid_multi_cost["avg_tokens_per_question"] == "75.0"
    assert hybrid_multi_cost["estimated_cost_usd"] == "0.002"

    assert "Audit sheet selection rule: preferred canonical audit file." in summary_text
    assert "Audit sheet used: `" in summary_text
    assert "audit_manual_sheet.csv`." in summary_text
    assert "# Final Ablation Summary" in summary_text
    assert "## QA Behavior" in summary_text
    assert "## Cost and Latency" in summary_text


def test_run_qa_report_degrades_gracefully_without_audit_sheet(tmp_path) -> None:
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    runs_dir.mkdir()
    reports_dir.mkdir()

    _write_jsonl(
        runs_dir / "run_log.jsonl",
        [
            _run_row(
                qid="q-1",
                retriever_mode="bm25",
                rewrite_mode="none",
                guard_mode="off",
                k=5,
                answer_model="model-b",
                answer_text=None,
                citations=[],
                retrieval_chunk_ids=["doc-1::0"],
                latency_ms=25.0,
                prompt_tokens=0,
                completion_tokens=0,
            )
        ],
    )

    outputs = run_qa_report(runs_dir=runs_dir, reports_dir=reports_dir)

    assert outputs["audit_path"] is None
    qa_rows = list(csv.DictReader((reports_dir / "qa_metrics.csv").open("r", encoding="utf-8")))
    cost_rows = list(csv.DictReader((reports_dir / "cost_latency.csv").open("r", encoding="utf-8")))
    summary_text = (reports_dir / "final_ablation_summary.md").read_text(encoding="utf-8")

    assert qa_rows[0]["retriever_mode"] == "bm25"
    assert qa_rows[0]["unanswered_rate"] == "1.0"
    assert qa_rows[0]["citation_valid_rate"] == "0.0"
    assert qa_rows[0]["evidence_found_rate"] == ""
    assert cost_rows[0]["estimated_cost_usd"] == "0.0"
    assert "Audit sheet used: `none`." in summary_text


def test_run_qa_report_uses_generated_audit_fallback_and_keeps_k_and_model_groups_separate(tmp_path) -> None:
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    runs_dir.mkdir()
    reports_dir.mkdir()

    _write_jsonl(
        runs_dir / "run_log.jsonl",
        [
            _run_row(
                qid="q-1",
                retriever_mode="hybrid",
                rewrite_mode="multi",
                guard_mode="on",
                k=5,
                answer_model="model-a",
                answer_text="Answer one.",
                citations=["doc-1::0"],
                retrieval_chunk_ids=["doc-1::0", "doc-2::0"],
                latency_ms=10.0,
                prompt_tokens=10,
                completion_tokens=5,
            ),
            _run_row(
                qid="q-2",
                retriever_mode="hybrid",
                rewrite_mode="multi",
                guard_mode="on",
                k=10,
                answer_model="model-b",
                answer_text="Answer two.",
                citations=["doc-3::0"],
                retrieval_chunk_ids=["doc-3::0", "doc-4::0"],
                latency_ms=20.0,
                prompt_tokens=20,
                completion_tokens=10,
            ),
        ],
    )

    _write_audit_csv(
        reports_dir / "audit_manual_sheet.generated.csv",
        [
            {
                "qid": "q-1",
                "question": "Question q-1",
                "gold_answer": "A",
                "gold_chunk_ids": '["doc-1::0"]',
                "retrieved_chunk_ids": '["doc-1::0"]',
                "predicted_answer": "",
                "evidence_found": "true",
                "citations_valid": "true",
                "config": "B_hybrid_multi_guard_on",
                "notes": "",
            },
            {
                "qid": "q-2",
                "question": "Question q-2",
                "gold_answer": "A",
                "gold_chunk_ids": '["doc-9::0"]',
                "retrieved_chunk_ids": '["doc-8::0"]',
                "predicted_answer": "",
                "evidence_found": "false",
                "citations_valid": "true",
                "config": "B_hybrid_multi_guard_on",
                "notes": "",
            },
        ],
    )

    outputs = run_qa_report(runs_dir=runs_dir, reports_dir=reports_dir)

    assert outputs["audit_path"] == reports_dir / "audit_manual_sheet.generated.csv"

    qa_rows = list(csv.DictReader((reports_dir / "qa_metrics.csv").open("r", encoding="utf-8")))
    assert len(qa_rows) == 2
    assert {(row["k"], row["answer_model"]) for row in qa_rows} == {("5", "model-a"), ("10", "model-b")}
    assert all(row["evidence_found_rate"] == "0.5" for row in qa_rows)

    summary_text = (reports_dir / "final_ablation_summary.md").read_text(encoding="utf-8")
    assert "Audit sheet selection rule: generated audit fallback." in summary_text
    assert "audit_manual_sheet.generated.csv`." in summary_text


def test_group_key_reads_the_answer_model_from_old_and_new_config_snapshots() -> None:
    from rag_thesis.evaluation.report import build_group_key

    new_block = {
        "llm": {
            "providers": {"gemini": {"type": "gemini", "api_key_env": "GEMINI_API_KEY"}},
            "roles": {"answer": {"provider": "gemini", "model": "gemini-2.5-flash"}},
        }
    }
    flat = {"llm_provider": "gemini", "llm_answer_model": "gemini-2.5-flash"}
    openai_era = {"openai_answer_model": "gpt-4.1-mini"}  # no llm_provider: read the flat key directly
    assert build_group_key({"config_snapshot": new_block}).answer_model == "gemini-2.5-flash"
    assert build_group_key({"config_snapshot": flat}).answer_model == "gemini-2.5-flash"
    assert build_group_key({"config_snapshot": openai_era}).answer_model == "gpt-4.1-mini"
    assert build_group_key({"config_snapshot": {}}).answer_model == "unknown"
