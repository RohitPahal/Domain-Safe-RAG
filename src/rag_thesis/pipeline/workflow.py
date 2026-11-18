from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rag_thesis import guardrails
from rag_thesis.config import load_config
from rag_thesis.io import load_chunks_by_id, load_questions_by_id
from rag_thesis.llm.base import ChatModel
from rag_thesis.llm.registry import build_chat_model
from rag_thesis.llm.settings import resolve_llm_settings
from rag_thesis.llm.wrappers import CallLog
from rag_thesis.pipeline.graph import compiled_rag_graph
from rag_thesis.pipeline.nodes import RAGRunContext, RAGWorkflowState
from rag_thesis.protection import ensure_writable
from rag_thesis.retrieval.bm25 import load_bm25
from rag_thesis.retrieval.chroma_store import load_chroma_index

RUN_LOG_FILENAME = "run_log.jsonl"
VALID_RETRIEVERS = {"bm25", "dense", "hybrid"}
VALID_REWRITES = {"none", "single", "multi", "structured", "hyde"}
VALID_GUARDS = {"off", "on"}
# LLM roles used at query time (the judge runs separately, on saved run logs).
PIPELINE_ROLES = ("rewrite", "repair", "answer")


def build_rag_run_context(
    *,
    questions_path: Path,
    chunks_path: Path,
    retriever_mode: str,
    config_path: Path | None = None,
    config: dict[str, Any] | None = None,
    bm25_dir: Path | None = None,
    vector_dir: Path | None = None,
    bm25_backend: Any | None = None,
    dense_backend: Any | None = None,
    chat_models: dict[str, ChatModel] | None = None,
    call_log: CallLog | None = None,
    allow_llm_calls: bool = False,
) -> RAGRunContext:
    """Load everything a batch of questions shares, once.

    ``chat_models`` (role -> model) may be given directly (tests); otherwise, with
    ``allow_llm_calls``, one model per configured pipeline role is built through the registry,
    all recording into one ``CallLog``. Without permission no model is built.
    """
    _validate_mode("retriever_mode", retriever_mode, VALID_RETRIEVERS)

    if not config and config_path is None:
        raise ValueError("config or config_path must be provided.")
    resolved_config = config or load_config(config_path)
    questions_by_id = load_questions_by_id(questions_path)
    chunks_by_id = load_chunks_by_id(chunks_path)

    if chat_models is None:
        chat_models, call_log = build_pipeline_chat_models(resolved_config, allow_llm_calls=allow_llm_calls)

    resolved_bm25 = bm25_backend
    if retriever_mode in {"bm25", "hybrid"} and resolved_bm25 is None:
        if bm25_dir is None:
            raise ValueError("bm25_dir is required when bm25_backend is not provided.")
        resolved_bm25 = load_bm25(bm25_dir)

    resolved_dense = dense_backend
    if retriever_mode in {"dense", "hybrid"} and resolved_dense is None:
        if vector_dir is None:
            raise ValueError("vector_dir is required when dense_backend is not provided.")
        resolved_dense = load_chroma_index(vector_dir, query_cache_dir=resolved_config.get("query_embedding_cache_dir"))

    return RAGRunContext(
        config=resolved_config,
        questions_by_id=questions_by_id,
        chunks_by_id=chunks_by_id,
        bm25_backend=resolved_bm25,
        dense_backend=resolved_dense,
        chat_models=dict(chat_models),
        call_log=call_log,
    )


def build_pipeline_chat_models(
    config: dict[str, Any], *, allow_llm_calls: bool
) -> tuple[dict[str, ChatModel], CallLog | None]:
    """The query-time role models from the config's LLM settings (none without permission)."""
    if not allow_llm_calls:
        return {}, None
    settings = resolve_llm_settings(config)
    call_log = CallLog()
    models = {
        role: build_chat_model(role, settings, allow_llm_calls=True, call_log=call_log)
        for role in PIPELINE_ROLES
        if role in settings.roles
    }
    return models, call_log


def run_rag_workflow(
    *,
    qid: str,
    retriever_mode: str,
    rewrite_mode: str,
    guard_mode: str,
    k: int,
    questions_path: Path,
    chunks_path: Path,
    config_path: Path | None = None,
    config: dict[str, Any] | None = None,
    bm25_dir: Path | None = None,
    vector_dir: Path | None = None,
    log_path: Path | None = None,
    bm25_backend: Any | None = None,
    dense_backend: Any | None = None,
    chat_models: dict[str, ChatModel] | None = None,
    allow_llm_calls: bool = False,
    generation_enabled: bool | None = None,
) -> RAGWorkflowState:
    context = build_rag_run_context(
        questions_path=questions_path,
        chunks_path=chunks_path,
        retriever_mode=retriever_mode,
        config_path=config_path,
        config=config,
        bm25_dir=bm25_dir,
        vector_dir=vector_dir,
        bm25_backend=bm25_backend,
        dense_backend=dense_backend,
        chat_models=chat_models,
        allow_llm_calls=allow_llm_calls,
    )
    return run_rag_workflow_with_context(
        qid=qid,
        retriever_mode=retriever_mode,
        rewrite_mode=rewrite_mode,
        guard_mode=guard_mode,
        k=k,
        context=context,
        log_path=log_path,
        generation_enabled=generation_enabled,
    )


def run_rag_workflow_with_context(
    *,
    qid: str,
    retriever_mode: str,
    rewrite_mode: str,
    guard_mode: str,
    k: int,
    context: RAGRunContext,
    log_path: Path | None = None,
    generation_enabled: bool | None = None,
) -> RAGWorkflowState:
    _validate_mode("retriever_mode", retriever_mode, VALID_RETRIEVERS)
    _validate_mode("rewrite_mode", rewrite_mode, VALID_REWRITES)
    _validate_mode("guard_mode", guard_mode, VALID_GUARDS)
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    question = context.questions_by_id.get(qid)
    if question is None:
        raise ValueError(f"QID not found: {qid}")
    if retriever_mode in {"bm25", "hybrid"} and context.bm25_backend is None:
        raise ValueError("bm25 backend is not available in the provided context.")
    if retriever_mode in {"dense", "hybrid"} and context.dense_backend is None:
        raise ValueError("dense backend is not available in the provided context.")

    # Generation runs when an answer model exists; `generation_enabled=False` turns it off. (There is no
    # way to force it without a model: there would be nothing to ask.)
    enabled = generation_enabled is not False and "answer" in context.chat_models
    initial_state: RAGWorkflowState = {
        "qid": qid,
        "query_original": str(question["question"]),
        "rewrite_mode": rewrite_mode,
        "guard_mode": guard_mode,
        "retriever_mode": retriever_mode,
        "rewrites": [str(question["question"])],
        "rewrite_status": "not_requested",  # the rewrite node replaces this when it runs
        "guard_report": {},
        "retrieval_results": [],
        "answer_text": None,
        "citations": [],
        "latency_ms": 0.0,
        "usage": {},
        "config_snapshot": _build_config_snapshot(
            context.config,
            retriever_mode=retriever_mode,
            rewrite_mode=rewrite_mode,
            guard_mode=guard_mode,
            k=k,
        ),
        "k": k,
        # Calls before this index belong to earlier questions of the same batch.
        "call_log_start": len(context.call_log) if context.call_log is not None else 0,
        "generation_enabled": enabled,
        # Checked before the run starts, so no LLM call is spent on a run whose log can't be written.
        "log_path": str(ensure_writable(log_path or (Path("outputs") / "runs" / RUN_LOG_FILENAME))),
        "start_time": time.perf_counter(),
    }
    if _guardrails_enabled(context.config):
        findings = guardrails.check_question(str(question["question"]))
        initial_state["guardrail"] = {
            "question": [asdict(finding) for finding in findings],
            "question_blocked": guardrails.is_blocked(findings),
        }
    threshold = _gate_threshold(context.config, retriever_mode)
    if threshold is not None and rewrite_mode != "none":
        initial_state["gate_threshold"] = threshold
    return compiled_rag_graph().invoke(initial_state, context=context)


def _build_config_snapshot(
    config: dict[str, Any], *, retriever_mode: str, rewrite_mode: str, guard_mode: str, k: int
) -> dict[str, Any]:
    snapshot = dict(config)
    snapshot.update({"retriever_mode": retriever_mode, "rewrite_mode": rewrite_mode, "guard_mode": guard_mode, "k": k})
    return snapshot


def _guardrails_enabled(config: dict[str, Any]) -> bool:
    """``guardrails: true`` turns the checks of ``rag_thesis.guardrails`` on; a config without the key
    (every thesis-era config) runs without them."""
    value = config.get("guardrails", False)
    if not isinstance(value, bool):
        raise ValueError(f"guardrails must be true or false, not {value!r}.")
    return value


def _gate_threshold(config: dict[str, Any], retriever_mode: str) -> float | None:
    """R5: the configured rewrite-gate threshold for this retriever (``rewrite_gate_thresholds``, e.g.
    from tools/gate_threshold.py), or ``None`` when rewriting isn't gated."""
    thresholds = config.get("rewrite_gate_thresholds") or {}
    if not isinstance(thresholds, dict):
        raise ValueError("rewrite_gate_thresholds must map a retriever (bm25, dense, hybrid) to a number.")
    value = thresholds.get(retriever_mode)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"rewrite_gate_thresholds.{retriever_mode} must be a number, not {value!r}.")
    return float(value)


def _validate_mode(field_name: str, value: str, valid_values: set[str]) -> None:
    if value not in valid_values:
        raise ValueError(f"{field_name} must be one of: {', '.join(sorted(valid_values))}")
