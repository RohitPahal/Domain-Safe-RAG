"""rewrite_batch and `rag-thesis rewrite-batch`: fill the rewrite cache for a question set."""

from __future__ import annotations

import json

from typer.testing import CliRunner

import rag_thesis.cli.rewrite_batch as cli_module
from rag_thesis.cli.rewrite_batch import app
from rag_thesis.llm.testing import FakeChatModel
from rag_thesis.rewrite.rewriter import MULTI_REWRITE_ORDER, load_cached_rewrites, load_cached_structures, rewrite_batch

QUESTIONS = [
    {"qid": "D1_001", "question": "What is ATP?", "doc_id": "d", "answers": []},
    {"qid": "D2_001", "question": "Which gene causes CF?", "doc_id": "d", "answers": []},
]


def _reply(prefix: str) -> str:
    return json.dumps({slot: f"{prefix} {slot}" for slot in MULTI_REWRITE_ORDER})


def _config(tmp_path) -> dict:
    return {
        "rewrite_cache_path": str(tmp_path / "rewrites.jsonl"),
        "llm_provider": "gemini",
        "llm_rewrite_model": "fake-model",
    }


def test_misses_are_rewritten_into_the_cache_and_replayed_without_a_model(tmp_path):
    config = _config(tmp_path)
    model = FakeChatModel([_reply("a"), _reply("b")], provider="gemini")
    rows = rewrite_batch(QUESTIONS, config=config, rewrite_mode="multi", chat_model=model)
    assert [row["status"] for row in rows] == ["ok", "ok"] and len(model.requests) == 2
    assert rows[0]["rewrites"][0] == "a bm25_lexical"
    replay = rewrite_batch(QUESTIONS, config=config, rewrite_mode="multi")  # no model: cache hits only
    assert [row["rewrites"] for row in replay] == [row["rewrites"] for row in rows]


def test_without_a_model_misses_are_reported_and_a_bad_reply_fails_without_stopping(tmp_path):
    config = _config(tmp_path)
    assert [row["status"] for row in rewrite_batch(QUESTIONS, config=config, rewrite_mode="multi")] == [
        "not_cached",
        "not_cached",
    ]
    model = FakeChatModel(["not json at all", _reply("b")], provider="gemini")
    rows = rewrite_batch(QUESTIONS, config=config, rewrite_mode="multi", chat_model=model)
    assert [row["status"] for row in rows] == ["failed", "ok"]
    assert rows[0]["error"].startswith("JSONOutputError")


def test_cli_dry_run_and_permitted_run(tmp_path, monkeypatch):
    questions = tmp_path / "questions.jsonl"
    questions.write_text("".join(json.dumps(q) + "\n" for q in QUESTIONS), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"rewrite_cache_path: {(tmp_path / 'rewrites.jsonl').as_posix()}\nllm_provider: gemini\n"
        "llm_rewrite_model: fake-model\nllm_api_key_env: GEMINI_API_KEY\n",
        encoding="utf-8",
    )
    args = ["--questions-path", str(questions), "--config-path", str(config)]
    dry = CliRunner().invoke(app, args)
    assert dry.exit_code == 0, dry.output
    assert "not_cached: 2" in dry.output and "llm_calls: 0" in dry.output and "--allow-llm-calls" in dry.output

    built = []

    def fake_build(role, settings, *, allow_llm_calls, call_log=None):
        built.append((role, allow_llm_calls))
        return FakeChatModel([_reply("a"), _reply("b")], provider="gemini")

    monkeypatch.setattr(cli_module, "build_chat_model", fake_build)
    run = CliRunner().invoke(app, [*args, "--allow-llm-calls"])
    assert run.exit_code == 0, run.output
    assert built == [("rewrite", True)] and "in_cache: 2" in run.output


def _structured_reply(prefix: str) -> str:
    return json.dumps(
        {
            "entities": ["ATP"],
            "abbreviations": [{"short": "ATP", "long": "adenosine triphosphate"}],
            "answer_type": "molecule",
            "constraints": [],
            "rewrites": {slot: f"{prefix} {slot}" for slot in MULTI_REWRITE_ORDER},
        }
    )


def test_structured_mode_caches_rewrites_and_their_structure(tmp_path):
    config = _config(tmp_path)
    model = FakeChatModel([_structured_reply("s")], provider="gemini")
    (row,) = rewrite_batch(QUESTIONS[:1], config=config, rewrite_mode="structured", chat_model=model)
    assert row["status"] == "ok" and row["rewrites"][2] == "s abbreviation_expanded"
    assert "rewrite_structured_v1" not in model.requests[0].messages[0].content  # the prompt, not its version
    assert "entities" in model.requests[0].messages[0].content and model.requests[0].json_mode
    (structure,) = load_cached_structures(tmp_path / "rewrites.jsonl", ["D1_001"]).values()
    assert structure.entities == ["ATP"] and structure.abbreviations[0].long == "adenosine triphosphate"
    assert structure.rewrite_list() == row["rewrites"]
    assert load_cached_rewrites(tmp_path / "rewrites.jsonl", ["D1_001"], "structured")["D1_001"]["bm25_lexical"] == (
        "s bm25_lexical"
    )
    # the multi and structured caches are separate: a multi lookup misses
    assert rewrite_batch(QUESTIONS[:1], config=config, rewrite_mode="multi")[0]["status"] == "not_cached"
    # a reply without the structure fields is unusable: failed, the batch goes on
    bad = FakeChatModel([_reply("x")], provider="gemini")
    assert rewrite_batch(QUESTIONS[1:], config=config, rewrite_mode="structured", chat_model=bad)[0]["status"] == (
        "failed"
    )


def test_hyde_mode_caches_one_hypothetical_passage(tmp_path):
    config = _config(tmp_path)
    model = FakeChatModel([json.dumps({"passage": "ATP is the main energy carrier of the cell."})], provider="gemini")
    (row,) = rewrite_batch(QUESTIONS[:1], config=config, rewrite_mode="hyde", chat_model=model)
    assert row["status"] == "ok" and row["rewrites"] == ["ATP is the main energy carrier of the cell."]
    assert "hypothetical passage" in model.requests[0].messages[0].content and model.requests[0].json_mode
    assert rewrite_batch(QUESTIONS[:1], config=config, rewrite_mode="hyde")[0]["rewrites"] == row["rewrites"]  # replay
    assert rewrite_batch(QUESTIONS[:1], config=config, rewrite_mode="single")[0]["status"] == "not_cached"
    blank = FakeChatModel(['{"passage": "  "}'], provider="gemini")
    assert rewrite_batch(QUESTIONS[1:], config=config, rewrite_mode="hyde", chat_model=blank)[0]["status"] == "failed"
