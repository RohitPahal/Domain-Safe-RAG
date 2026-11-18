"""Write protection (rag_thesis/protection.py).

Every test points the guard at a temporary "protected" folder via `protected_dir`, so no test
ever writes near the real outputs/final_benchmark/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from rag_thesis import protection
from rag_thesis.paths import PROJECT_ROOT
from rag_thesis.protection import DEFAULT_GUARD, PROTECTED_DIRS, ProtectedPathError, WriteGuard


@pytest.fixture
def protected_dir(tmp_path, monkeypatch) -> Path:
    """A temporary protected folder; the package-wide guard is swapped to protect it."""
    folder = tmp_path / "protected"
    folder.mkdir()
    monkeypatch.setattr(protection, "DEFAULT_GUARD", WriteGuard([folder]))
    return folder


def _assert_untouched(folder: Path, expected: set[str] | None = None) -> None:
    assert {p.name for p in folder.iterdir()} == (expected or set())


# --- The rule itself ----------------------------------------------------------------------


def test_guard_protects_the_folder_and_everything_inside(tmp_path):
    folder = tmp_path / "protected"
    guard = WriteGuard([folder])
    assert guard.is_protected(folder)
    assert guard.is_protected(folder / "file.jsonl")
    assert guard.is_protected(folder / "a" / "b" / "c.csv")
    assert guard.is_protected(tmp_path / "other" / ".." / "protected" / "x.txt")  # `..` is resolved


def test_guard_allows_everything_else(tmp_path):
    guard = WriteGuard([tmp_path / "protected"])
    assert not guard.is_protected(tmp_path / "protected_v2" / "x.txt")  # same prefix, different folder
    assert not guard.is_protected(tmp_path / "outputs" / "runs" / "log.jsonl")
    assert guard.check(tmp_path / "ok.txt") == tmp_path / "ok.txt"


def test_check_raises_a_clear_error(tmp_path):
    guard = WriteGuard([tmp_path / "protected"])
    with pytest.raises(ProtectedPathError, match="read-only"):
        guard.check(tmp_path / "protected" / "x.txt")
    assert issubclass(ProtectedPathError, PermissionError)


@pytest.mark.skipif(sys.platform != "win32", reason="paths are case-insensitive only on Windows")
def test_guard_ignores_case_on_windows(tmp_path):
    guard = WriteGuard([tmp_path / "protected"])
    assert guard.is_protected(Path(str(tmp_path / "PROTECTED" / "x.txt")))


def test_guard_sees_through_symlinks(tmp_path):
    folder = tmp_path / "protected"
    folder.mkdir()
    link = tmp_path / "innocent_looking_link"
    try:
        os.symlink(folder, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("creating symlinks is not permitted here")
    assert WriteGuard([folder]).is_protected(link / "x.txt")


def test_the_official_results_are_protected_by_default():
    assert "outputs/final_benchmark" in PROTECTED_DIRS
    official = PROJECT_ROOT / "outputs" / "final_benchmark"
    assert DEFAULT_GUARD.is_protected(official / "anything" / "file.jsonl")  # a check only, no write
    assert not DEFAULT_GUARD.is_protected(PROJECT_ROOT / "outputs" / "cache" / "rewrites.jsonl")


# --- Every writer asks the guard and writes nothing ----------------------------------------


def test_rewrite_cache_replays_read_only_but_refuses_new_records(protected_dir):
    from rag_thesis.llm.cache import append_cache_record

    cache = protected_dir / "rewrites.jsonl"
    existing = {"cache_key": "k1", "rewrites": ["a"]}
    cache.write_text(json.dumps(existing) + "\n", encoding="utf-8")  # set up as if it were official

    assert append_cache_record(cache, existing) is False  # already cached: read-only, allowed
    with pytest.raises(ProtectedPathError):
        append_cache_record(cache, {"cache_key": "k2", "rewrites": ["b"]})
    assert cache.read_text(encoding="utf-8").splitlines() == [json.dumps(existing)]


def test_run_log_writer_refuses(protected_dir):
    from rag_thesis.pipeline.nodes import _append_jsonl

    with pytest.raises(ProtectedPathError):
        _append_jsonl(protected_dir / "runs" / "log.jsonl", {"qid": "q"})
    _assert_untouched(protected_dir)


def test_bm25_save_refuses(protected_dir):
    from rag_thesis.retrieval.bm25 import build_bm25

    index = build_bm25([{"chunk_id": "d::0", "doc_id": "d", "text": "cat sat"}])
    with pytest.raises(ProtectedPathError):
        index.save_bm25(protected_dir / "bm25")
    _assert_untouched(protected_dir)


def test_chroma_build_refuses_before_loading_the_model(protected_dir, monkeypatch):
    from rag_thesis.retrieval import chroma_store

    def must_not_load(*args, **kwargs):
        raise AssertionError("the model was loaded before the protection check")

    monkeypatch.setattr(chroma_store, "load_sentence_transformer", must_not_load)
    chunks = [{"chunk_id": "d::0", "doc_id": "d", "text": "cat sat"}]
    with pytest.raises(ProtectedPathError):
        chroma_store.build_chroma_index(
            chunks, {"embed_model": "m", "embed_max_seq_length": 256}, protected_dir / "chroma"
        )
    _assert_untouched(protected_dir)


def _write_retrieval_eval_csv(path: Path) -> None:
    from rag_thesis.evaluation.retrieval import write_csv

    write_csv(path, [{"a": 1}], ["a"])


def _write_report_csv(path: Path) -> None:
    from rag_thesis.evaluation.report import write_csv

    write_csv(path, [{"a": 1}], ["a"])


def _write_corpus_jsonl(path: Path) -> None:
    from rag_thesis.data.corpus import write_jsonl

    write_jsonl(path, [{"a": 1}])


def _write_squad_json(path: Path) -> None:
    from rag_thesis.data.squad import save_squad_dataset

    save_squad_dataset(path, {"data": []})


@pytest.mark.parametrize(
    "writer", [_write_retrieval_eval_csv, _write_report_csv, _write_corpus_jsonl, _write_squad_json]
)
def test_file_writers_refuse(protected_dir, writer):
    with pytest.raises(ProtectedPathError):
        writer(protected_dir / "sub" / "out.file")
    _assert_untouched(protected_dir)


def test_answer_evaluation_refuses_before_reading_anything(protected_dir, tmp_path):
    from rag_thesis.evaluation.answers import evaluate_answer_quality

    with pytest.raises(ProtectedPathError):  # not FileNotFoundError: the check comes first
        evaluate_answer_quality(
            run_log_path=tmp_path / "missing.jsonl",
            questions_path=tmp_path / "missing_questions.jsonl",
            gold_evidence_path=None,
            reports_dir=protected_dir / "reports",
        )
    _assert_untouched(protected_dir)


def test_judge_refuses_before_any_llm_call(protected_dir, tmp_path):
    from rag_thesis.evaluation.judge import evaluate_with_judge
    from rag_thesis.llm.testing import FakeChatModel

    with pytest.raises(ProtectedPathError):
        evaluate_with_judge(
            run_log_path=tmp_path / "missing.jsonl",
            questions_path=tmp_path / "q.jsonl",
            chunks_path=tmp_path / "c.jsonl",
            config_path=tmp_path / "config.yaml",
            reports_dir=protected_dir / "judge",
            chat_model=FakeChatModel([]),  # fails if it is ever called
        )
    _assert_untouched(protected_dir)


# --- Fail fast in the rewriter, and never hidden by the pipeline's fallback ------------------

REWRITE_CONFIG = {"llm_provider": "gemini", "llm_rewrite_model": "gemini-2.5-flash"}


class _RecordingClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("an LLM call was made before the protection check")


def test_rewriter_replays_a_protected_cache_but_refuses_a_miss_before_calling_the_llm(protected_dir):
    from rag_thesis.llm.cache import build_rewrite_cache_key
    from rag_thesis.llm.testing import FakeChatModel
    from rag_thesis.rewrite.rewriter import REWRITE_PROMPT_VERSION, QueryRewriter

    cache = protected_dir / "rewrites.jsonl"
    key = build_rewrite_cache_key(
        qid="q1",
        query_original="What is BRCA1?",
        rewrite_mode="single",
        rewrite_model="gemini-2.5-flash",
        prompt_version=REWRITE_PROMPT_VERSION,
    )
    cache.write_text(json.dumps({"cache_key": key, "rewrites": ["BRCA1 function"]}) + "\n", encoding="utf-8")
    model = FakeChatModel([], provider="gemini", model="gemini-2.5-flash")  # fails if it is ever called
    rewriter = QueryRewriter(config=REWRITE_CONFIG, chat_model=model, cache_path=cache)

    # The record has a legacy key (no provider), like the official cache: still replayed.
    assert rewriter.rewrite(qid="q1", query_original="What is BRCA1?", rewrite_mode="single") == ["BRCA1 function"]
    with pytest.raises(ProtectedPathError):
        rewriter.rewrite(qid="q2", query_original="What is TP53?", rewrite_mode="single")
    assert model.requests == []


def test_pipeline_rewrite_fallback_never_hides_a_protection_error(protected_dir):
    from langgraph.runtime import Runtime

    from rag_thesis.llm.testing import FakeChatModel
    from rag_thesis.pipeline.nodes import RAGRunContext, rewrite_node

    state = {"qid": "q2", "query_original": "What is TP53?", "rewrite_mode": "single"}
    context = RAGRunContext(
        config={**REWRITE_CONFIG, "rewrite_cache_path": str(protected_dir / "rewrites.jsonl")},
        questions_by_id={},
        chunks_by_id={},
        chat_models={"rewrite": FakeChatModel([])},  # fails if it is ever called
    )
    with pytest.raises(ProtectedPathError):
        rewrite_node(state, Runtime(context=context))


def test_pipeline_refuses_a_protected_log_path_before_running(protected_dir):
    from rag_thesis.pipeline.workflow import RAGRunContext, run_rag_workflow_with_context

    context = RAGRunContext(
        config={"rrf_k": 60},
        questions_by_id={"q1": {"qid": "q1", "question": "What is BRCA1?"}},
        chunks_by_id={},
        bm25_backend=object(),  # never used: the run is refused before retrieval
    )
    with pytest.raises(ProtectedPathError):
        run_rag_workflow_with_context(
            qid="q1",
            retriever_mode="bm25",
            rewrite_mode="none",
            guard_mode="off",
            k=10,
            context=context,
            log_path=protected_dir / "runs" / "log.jsonl",
        )
    _assert_untouched(protected_dir)
