from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import rag_thesis.evaluation.judge as judge_module
from rag_thesis.evaluation.judge import (
    build_judge_messages,
    evaluate_with_judge,
    parse_judge_response,
    summarize_judge_rows,
)
from rag_thesis.llm.errors import LLMAuthError, LLMCallsDisabledError
from rag_thesis.llm.testing import FakeChatModel
from rag_thesis.llm.types import Usage


def _fake_judge(responses: list[str]) -> FakeChatModel:
    return FakeChatModel(responses, provider="gemini", model="gemini-2.5-pro", usage=Usage(12, 4, 16))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_build_judge_messages_includes_question_gold_and_evidence() -> None:
    messages = build_judge_messages(
        question="Which gene-defect underlies the Vel-blood group?",
        predicted_answer="SMIM1",
        gold_answers=["a 17 nucleotide deletion"],
        evidence_chunks=[
            {
                "chunk_id": "doc-1::0",
                "source": "citation",
                "text": "Vel-negative individuals have a 17 nucleotide deletion in SMIM1.",
            }
        ],
    )

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert "CORRECT" in messages[0].content
    assert "UNSUPPORTED" in messages[0].content
    assert "faithfulness_label" in messages[0].content
    assert "relevance_label" in messages[0].content
    assert "supported_text" in messages[0].content
    assert "unsupported_text" in messages[0].content
    assert "Which gene-defect underlies the Vel-blood group?" in messages[1].content
    assert '["a 17 nucleotide deletion"]' in messages[1].content
    assert "chunk_id=doc-1::0 source=citation" in messages[1].content


def test_parse_judge_response_accepts_json_and_code_fences() -> None:
    parsed_json = parse_judge_response(
        '{"overall_verdict":"CORRECT","overall_reason":"Alias '
        'match.","faithfulness_label":"FAITHFUL","faithfulness_score":1.0,"relevance_label":"RELEVANT","relevance_score":1.0,"supported_text":"ZEB2","unsupported_text":""}'
    )
    parsed_fenced = parse_judge_response(
        "```json\n"
        '{"overall_verdict":"PARTIALLY_CORRECT","overall_reason":"Close but too '
        'broad.","faithfulness_label":"PARTIALLY_FAITHFUL","faithfulness_score":0.5,"relevance_label":"PARTIALLY_RELEVANT","relevance_score":0.5,"supported_text":"deletion","unsupported_text":"SMIM1"}'
        "\n```"
    )

    assert parsed_json == {
        "overall_verdict": "CORRECT",
        "overall_reason": "Alias match.",
        "judge_label": "CORRECT",
        "judge_reason": "Alias match.",
        "faithfulness_label": "FAITHFUL",
        "faithfulness_score": 1.0,
        "relevance_label": "RELEVANT",
        "relevance_score": 1.0,
        "supported_text": "ZEB2",
        "unsupported_text": "",
    }
    assert parsed_fenced == {
        "overall_verdict": "PARTIALLY_CORRECT",
        "overall_reason": "Close but too broad.",
        "judge_label": "PARTIALLY_CORRECT",
        "judge_reason": "Close but too broad.",
        "faithfulness_label": "PARTIALLY_FAITHFUL",
        "faithfulness_score": 0.5,
        "relevance_label": "PARTIALLY_RELEVANT",
        "relevance_score": 0.5,
        "supported_text": "deletion",
        "unsupported_text": "SMIM1",
    }


def test_parse_judge_response_rejects_invalid_labels_or_scores() -> None:
    invalid_score = (
        '{"overall_verdict":"CORRECT","overall_reason":"Alias match.","faithfulness_label":"FAITHFUL",'
        '"faithfulness_score":0.5,"relevance_label":"RELEVANT","relevance_score":1.0,'
        '"supported_text":"ZEB2","unsupported_text":""}'
    )
    invalid_relevance = (
        '{"overall_verdict":"CORRECT","overall_reason":"Alias match.","faithfulness_label":"FAITHFUL",'
        '"faithfulness_score":1.0,"relevance_label":"UNKNOWN","relevance_score":1.0,'
        '"supported_text":"ZEB2","unsupported_text":""}'
    )

    try:
        parse_judge_response(invalid_score)
    except ValueError as exc:
        assert "faithfulness_score" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected invalid score to raise ValueError.")

    try:
        parse_judge_response(invalid_relevance)
    except ValueError as exc:
        assert "relevance_label" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected invalid relevance label to raise ValueError.")


def test_summarize_judge_rows_aggregates_counts_and_rates() -> None:
    summary = summarize_judge_rows(
        [
            {
                "overall_verdict": "CORRECT",
                "faithfulness_label": "FAITHFUL",
                "faithfulness_score": 1.0,
                "relevance_label": "RELEVANT",
                "relevance_score": 1.0,
            },
            {
                "overall_verdict": "CORRECT",
                "faithfulness_label": "FAITHFUL",
                "faithfulness_score": 1.0,
                "relevance_label": "PARTIALLY_RELEVANT",
                "relevance_score": 0.5,
            },
            {
                "overall_verdict": "UNSUPPORTED",
                "faithfulness_label": "UNFAITHFUL",
                "faithfulness_score": 0.0,
                "relevance_label": "IRRELEVANT",
                "relevance_score": 0.0,
            },
            {
                "overall_verdict": "INCORRECT",
                "faithfulness_label": "PARTIALLY_FAITHFUL",
                "faithfulness_score": 0.5,
                "relevance_label": "PARTIALLY_RELEVANT",
                "relevance_score": 0.5,
            },
        ]
    )

    assert summary == {
        "total_questions": 4,
        "correct_count": 2,
        "correct_rate": 0.5,
        "partially_correct_count": 0,
        "partially_correct_rate": 0.0,
        "incorrect_count": 1,
        "incorrect_rate": 0.25,
        "unsupported_count": 1,
        "unsupported_rate": 0.25,
        "faithful_count": 2,
        "faithful_rate": 0.5,
        "partially_faithful_count": 1,
        "partially_faithful_rate": 0.25,
        "unfaithful_count": 1,
        "unfaithful_rate": 0.25,
        "relevant_count": 1,
        "relevant_rate": 0.25,
        "partially_relevant_count": 2,
        "partially_relevant_rate": 0.5,
        "irrelevant_count": 1,
        "irrelevant_rate": 0.25,
        "faithfulness_mean_score": 0.625,
        "relevance_mean_score": 0.5,
        "faithful_and_relevant_count": 1,
        "faithful_and_relevant_rate": 0.25,
    }


def test_evaluate_with_judge_writes_outputs_and_uses_gold_answers(tmp_path) -> None:
    run_log_path = tmp_path / "run_log.jsonl"
    questions_path = tmp_path / "questions.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    config_path = tmp_path / "config.yaml"
    reports_dir = tmp_path / "reports"

    _write_jsonl(
        questions_path,
        [
            {
                "qid": "q-1",
                "question": "Which gene causes Mowat-Wilson syndrome?",
                "doc_id": "doc-1",
                "answers": [{"text": "ZFHX1B", "answer_start": 0}],
            },
            {
                "qid": "q-2",
                "question": "Which gene causes Mowat-Wilson syndrome?",
                "doc_id": "doc-2",
                "answers": [{"text": "ZEB2", "answer_start": 0}],
            },
        ],
    )
    _write_jsonl(
        chunks_path,
        [
            {
                "chunk_id": "doc-1::0",
                "doc_id": "doc-1",
                "chunk_index": 0,
                "text": "ZEB2, also known as ZFHX1B, is responsible for Mowat-Wilson syndrome.",
                "char_start": 0,
                "char_end": 76,
                "metadata": {},
            }
        ],
    )
    _write_jsonl(
        run_log_path,
        [
            {
                "qid": "q-1",
                "query_original": "Which gene causes Mowat-Wilson syndrome?",
                "query_rewrites": ["Which gene causes Mowat-Wilson syndrome?"],
                "guard_report": {},
                "retrieval_results": [{"chunk_id": "doc-1::0"}],
                "answer_text": "ZEB2",
                "citations": ["doc-1::0"],
                "latency_ms": 10.0,
                "openai_usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "config_snapshot": {"retriever_mode": "hybrid", "rewrite_mode": "none", "guard_mode": "off", "k": 10},
            },
            {
                "qid": "q-2",
                "query_original": "Which gene causes Mowat-Wilson syndrome?",
                "query_rewrites": ["Which gene causes Mowat-Wilson syndrome?"],
                "guard_report": {},
                "retrieval_results": [{"chunk_id": "doc-1::0"}],
                "answer_text": None,
                "citations": [],
                "latency_ms": 11.0,
                "openai_usage": {},
                "config_snapshot": {"retriever_mode": "hybrid", "rewrite_mode": "none", "guard_mode": "off", "k": 10},
            },
        ],
    )
    config_path.write_text(
        "llm_provider: gemini\nllm_answer_model: gemini-2.5-flash\nllm_judge_model: gemini-2.5-pro\n",
        encoding="utf-8",
    )
    fake_client = _fake_judge(
        [
            '{"overall_verdict":"CORRECT","overall_reason":"ZEB2 is an alias of '
            'ZFHX1B.","faithfulness_label":"FAITHFUL","faithfulness_score":1.0,"relevance_label":"RELEVANT","relevance_score":1.0,"supported_text":"ZEB2","unsupported_text":""}',
            '{"overall_verdict":"UNSUPPORTED","overall_reason":"No supported answer was '
            'produced.","faithfulness_label":"UNFAITHFUL","faithfulness_score":0.0,"relevance_label":"IRRELEVANT","relevance_score":0.0,"supported_text":"","unsupported_text":""}',
        ]
    )

    outputs = evaluate_with_judge(
        run_log_path=run_log_path,
        questions_path=questions_path,
        chunks_path=chunks_path,
        config_path=config_path,
        reports_dir=reports_dir,
        chat_model=fake_client,
        first_n=2,
        output_stem="__smoke_judge_q2",
    )

    assert outputs["judge_model"] == "gemini-2.5-pro"
    assert outputs["summary"] == {
        "total_questions": 2,
        "correct_count": 1,
        "correct_rate": 0.5,
        "partially_correct_count": 0,
        "partially_correct_rate": 0.0,
        "incorrect_count": 0,
        "incorrect_rate": 0.0,
        "unsupported_count": 1,
        "unsupported_rate": 0.5,
        "faithful_count": 1,
        "faithful_rate": 0.5,
        "partially_faithful_count": 0,
        "partially_faithful_rate": 0.0,
        "unfaithful_count": 1,
        "unfaithful_rate": 0.5,
        "relevant_count": 1,
        "relevant_rate": 0.5,
        "partially_relevant_count": 0,
        "partially_relevant_rate": 0.0,
        "irrelevant_count": 1,
        "irrelevant_rate": 0.5,
        "faithfulness_mean_score": 0.5,
        "relevance_mean_score": 0.5,
        "faithful_and_relevant_count": 1,
        "faithful_and_relevant_rate": 0.5,
    }
    assert len(fake_client.requests) == 2
    assert fake_client.requests[0].json_mode is True  # explicit, not inferred from the prompt

    with outputs["per_question_path"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["overall_verdict"] == "CORRECT"
    assert rows[0]["judge_label"] == "CORRECT"
    assert rows[0]["faithfulness_label"] == "FAITHFUL"
    assert rows[0]["relevance_label"] == "RELEVANT"
    assert rows[0]["supported_text"] == "ZEB2"
    assert rows[0]["unsupported_text"] == ""
    assert rows[1]["overall_verdict"] == "UNSUPPORTED"
    assert rows[1]["judge_label"] == "UNSUPPORTED"
    assert '["ZFHX1B", "ZEB2"]' == rows[0]["gold_answers"]
    assert outputs["per_question_path"].name == "__smoke_judge_q2_judge.csv"
    assert outputs["summary_path"].name == "__smoke_judge_q2_judge_summary.md"

    summary_text = outputs["summary_path"].read_text(encoding="utf-8")
    assert "Judge model: `gemini-2.5-pro`" in summary_text
    assert "- correct_rate: 0.500000" in summary_text
    assert "- faithful_rate: 0.500000" in summary_text
    assert "- relevance_mean_score: 0.500000" in summary_text
    assert "- faithful_and_relevant_rate: 0.500000" in summary_text


# --- `rag-thesis judge` CLI wrapper ----------------------------------------


def test_judge_cli_prints_every_verdict_without_calling_an_llm(tmp_path, monkeypatch):
    """Runs the real CLI with the evaluation replaced by a fake result, so no LLM is called."""
    from typer.testing import CliRunner

    import rag_thesis.cli.judge as judge_cli
    from rag_thesis.evaluation.judge import OVERALL_VERDICTS

    summary = {"total_questions": 4, "faithfulness_mean_score": 0.75, "relevance_mean_score": 1.0}
    summary["faithful_and_relevant_rate"] = 0.5
    for label in OVERALL_VERDICTS:
        summary[f"{label.lower()}_count"] = 1
        summary[f"{label.lower()}_rate"] = 0.25
    seen_kwargs: list[dict] = []

    def fake_evaluate(**kwargs):
        seen_kwargs.append(kwargs)
        return {
            "summary": summary,
            "judge_model": "fake-judge",
            "judge_provider": "fake",
            "per_question_path": tmp_path / "q.csv",
            "summary_path": tmp_path / "s.md",
        }

    monkeypatch.setattr(judge_cli, "evaluate_with_judge", fake_evaluate)
    files = {name: tmp_path / name for name in ("run.jsonl", "questions.jsonl", "chunks.jsonl", "config.yaml")}
    for path in files.values():
        path.write_text("", encoding="utf-8")

    result = CliRunner().invoke(
        judge_cli.app,
        [
            "--run-log-path",
            str(files["run.jsonl"]),
            "--questions-path",
            str(files["questions.jsonl"]),
            "--chunks-path",
            str(files["chunks.jsonl"]),
            "--config-path",
            str(files["config.yaml"]),
            "--reports-dir",
            str(tmp_path / "reports"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "judge_model: fake-judge" in result.output
    for label in OVERALL_VERDICTS:
        assert f"{label.lower()}_count: 1" in result.output
    assert "faithful_and_relevant_rate: 0.500000" in result.output
    assert seen_kwargs[0]["allow_llm_calls"] is False  # real judge calls only with --allow-llm-calls


# --- Permission: judging calls an LLM only when explicitly allowed ---------------------

LEGACY_JUDGE_CONFIG = "llm_provider: gemini\nllm_answer_model: gemini-2.5-flash\nllm_judge_model: gemini-2.5-pro\n"
NEW_JUDGE_CONFIG = (
    "llm:\n"
    "  cache_path: cache.jsonl\n"
    "  providers: {gemini: {type: gemini, api_key_env: GEMINI_API_KEY}}\n"
    "  roles: {judge: {provider: gemini, model: gemini-2.5-pro}}\n"
)


def _evaluate(tmp_path, config_text: str, **kwargs):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    return evaluate_with_judge(
        run_log_path=tmp_path / "missing_run.jsonl",  # never read: the permission check comes first
        questions_path=tmp_path / "missing_questions.jsonl",
        chunks_path=tmp_path / "missing_chunks.jsonl",
        config_path=config_path,
        reports_dir=tmp_path / "reports",
        **kwargs,
    )


def test_judge_refuses_real_calls_without_permission_before_any_work(tmp_path):
    with pytest.raises(LLMCallsDisabledError, match="--allow-llm-calls"):
        _evaluate(tmp_path, LEGACY_JUDGE_CONFIG)


def test_judge_with_permission_checks_the_api_key_before_any_work(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMAuthError, match="GEMINI_API_KEY"):
        _evaluate(tmp_path, LEGACY_JUDGE_CONFIG, allow_llm_calls=True)


def test_judge_model_override_and_cache_replay_without_permission(tmp_path, monkeypatch):
    built = {}

    def fake_build(role, settings, *, allow_llm_calls):
        built.update(role=role, model=settings.role(role).model, allow=allow_llm_calls, cache=settings.cache_path)
        raise RuntimeError("stop after building")

    monkeypatch.setattr(judge_module, "build_chat_model", fake_build)
    with pytest.raises(RuntimeError, match="stop after building"):
        _evaluate(tmp_path, NEW_JUDGE_CONFIG, judge_model="other-judge")
    # A cache is configured, so a replay-only judge model is built (no permission, no key needed).
    assert built == {"role": "judge", "model": "other-judge", "allow": False, "cache": "cache.jsonl"}
