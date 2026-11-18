"""State, run context and nodes of the query-time graph.

``RAGWorkflowState`` is plain data about one question. ``RAGRunContext`` holds what every question
of a run shares (models, retrievers, chunks, config); LangGraph passes it to each node at run time
(``runtime.context``), so it is never part of the state. Each node returns its state update.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from langgraph.runtime import Runtime

from rag_thesis import guardrails
from rag_thesis.guard.domain_safe import validate_and_repair_rewrite
from rag_thesis.llm.base import ChatModel
from rag_thesis.llm.wrappers import CallLog
from rag_thesis.pipeline.fusion import retrieve_queries, unique_in_order
from rag_thesis.pipeline.generation import AnswerGenerator, add_usage, answer_prompt_version
from rag_thesis.protection import ensure_writable
from rag_thesis.retrieval.confidence import is_confident, top1_score
from rag_thesis.retrieval.factory import make_retriever
from rag_thesis.rewrite.rewriter import EXPECTED_REWRITE_ERRORS, rewrite_query
from rag_thesis.schemas import RunLogLine

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RAGRunContext:
    config: dict[str, Any]
    questions_by_id: dict[str, dict[str, Any]]
    chunks_by_id: dict[str, dict[str, Any]]
    bm25_backend: Any | None = None
    dense_backend: Any | None = None
    # One ChatModel per role (missing role = that step can't call an LLM: rewrite replays the
    # cache only, repair is unavailable, generation is disabled) and the log of their real calls.
    chat_models: dict[str, ChatModel] = field(default_factory=dict)
    call_log: CallLog | None = None


class RAGWorkflowState(TypedDict, total=False):
    qid: str
    query_original: str
    rewrite_mode: str
    guard_mode: str
    retriever_mode: str
    rewrites: list[str]
    guard_report: dict[str, Any]
    retrieval_results: list[dict[str, Any]]
    answer_text: str | None
    citations: list[str]
    latency_ms: float
    usage: dict[str, Any]
    llm_calls: list[dict[str, Any]]
    config_snapshot: dict[str, Any]
    k: int
    call_log_start: int
    generation_enabled: bool
    log_path: str
    start_time: float
    rewrite_status: str
    rewrite_error: str | None
    generation_status: str
    generation_error: str | None
    gate_threshold: float | None  # R5: rewrite only below this top-1 score (None: no gate)
    rewrite_gate: dict[str, Any]
    guardrail: dict[str, Any]  # findings of rag_thesis.guardrails; present only when guardrails are on


def gate_node(state: RAGWorkflowState, runtime: Runtime[RAGRunContext]) -> dict[str, Any]:
    """R5: retrieve with the question first; rewrite only if the retrieval is unsure (its top-1 score
    is below the threshold). A confident retrieval is used as it is: no rewrite, no guard."""
    question = str(state["query_original"]).strip()
    results = _retrieve(state, runtime, [question])
    threshold = float(state["gate_threshold"])
    confident = is_confident(results, threshold)
    gate = {"confidence": top1_score(results), "threshold": threshold, "rewritten": not confident}
    if not confident:
        return {"rewrite_gate": gate}
    return {
        "rewrite_gate": gate,
        "rewrites": [question],
        "rewrite_status": "skipped_confident",
        "guard_report": {"guard_mode": state["guard_mode"], "final_rewrites": [question], "reports": []},
        "retrieval_results": results,
    }


def rewrite_node(state: RAGWorkflowState, runtime: Runtime[RAGRunContext]) -> dict[str, Any]:
    query_original = str(state["query_original"])
    try:
        rewrites = rewrite_query(
            config=dict(runtime.context.config),
            qid=str(state["qid"]),
            query_original=query_original,
            rewrite_mode=str(state["rewrite_mode"]),
            chat_model=runtime.context.chat_models.get("rewrite"),
        )
    except EXPECTED_REWRITE_ERRORS as exc:
        return {"rewrite_status": "failed", "rewrite_error": f"{type(exc).__name__}: {exc}"}
    rewrites = unique_in_order(rewrites) or [query_original]
    return {"rewrites": rewrites, "rewrite_status": "ok"}


def rewrite_fallback_node(state: RAGWorkflowState) -> dict[str, Any]:
    """A failed rewrite doesn't stop the run: continue with the original question, and say why."""
    LOGGER.warning(
        "Rewrite failed for qid=%s; using the original question. %s", state["qid"], state.get("rewrite_error")
    )
    return {"rewrites": [str(state["query_original"])]}


def skip_guard_node(state: RAGWorkflowState) -> dict[str, Any]:
    """Guard off: the rewrites go to retrieval unchecked, and the report says so."""
    rewrites = _rewrites_or_original(state)
    return {"rewrites": rewrites, "guard_report": {"guard_mode": "off", "final_rewrites": rewrites, "reports": []}}


def guard_node(state: RAGWorkflowState, runtime: Runtime[RAGRunContext]) -> dict[str, Any]:
    """Guard on: check every rewrite, repair an unsafe one once, else fall back to the original."""
    rewrites = _rewrites_or_original(state)
    reports: list[dict[str, Any]] = []
    final_rewrites: list[str] = []
    repair_model = runtime.context.chat_models.get("repair")
    for rewrite in rewrites:
        final_query, report = validate_and_repair_rewrite(str(state["query_original"]), rewrite, repair_model)
        final_rewrites.append(final_query)
        reports.append({"candidate_rewrite": rewrite, "final_query": final_query, **report})
    final_rewrites = unique_in_order(final_rewrites) or [str(state["query_original"])]
    return {
        "rewrites": final_rewrites,
        "guard_report": {
            "guard_mode": "on",
            "final_rewrites": final_rewrites,
            "reports": reports,
            "all_safe": all(report["is_safe"] for report in reports),
        },
    }


def retrieve_node(state: RAGWorkflowState, runtime: Runtime[RAGRunContext]) -> dict[str, Any]:
    queries = _rewrites_or_original(state)
    if state.get("rewrite_gate", {}).get("rewritten"):
        # R5: the gate already found the question unsure; keep it in the set with its rewrites (R1).
        queries = unique_in_order([str(state["query_original"]), *queries])
    return {"retrieval_results": _retrieve(state, runtime, queries)}


def _retrieve(state: RAGWorkflowState, runtime: Runtime[RAGRunContext], queries: list[str]) -> list[dict[str, Any]]:
    """Top-k for the queries with this run's retriever, fused as in ``fusion.retrieve_queries``."""
    context = runtime.context
    rrf_k = int(context.config.get("rrf_k", 60))
    mode = str(state["retriever_mode"])
    retriever = make_retriever(mode, bm25=context.bm25_backend, dense=context.dense_backend, k_rrf=rrf_k)
    return retrieve_queries(retriever, queries, int(state["k"]), retriever_mode=mode, k_rrf=rrf_k)


def refuse_node(state: RAGWorkflowState) -> dict[str, Any]:
    """Guardrails blocked the question before anything ran: no rewrite, no retrieval, no model call."""
    reasons = "; ".join(finding["detail"] for finding in state["guardrail"]["question"])
    LOGGER.warning("Question refused by the guardrails for qid=%s: %s", state["qid"], reasons)
    return {
        "answer_text": None,
        "citations": [],
        "usage": {},
        "generation_status": "refused",
        "generation_error": f"question blocked by guardrails: {reasons}",
    }


def skip_generation_node(state: RAGWorkflowState) -> dict[str, Any]:
    """No answer is asked for: nothing was retrieved, or generation is off (not wanted or no model)."""
    status = "no_retrieval" if not state.get("retrieval_results") else "disabled"
    return {"answer_text": None, "citations": [], "usage": {}, "generation_status": status}


def generate_node(state: RAGWorkflowState, runtime: Runtime[RAGRunContext]) -> dict[str, Any]:
    """Ask with the top-1 chunk."""
    return _ask(state, runtime, window_size=1, usage_so_far={})


def generate_retry_node(state: RAGWorkflowState, runtime: Runtime[RAGRunContext]) -> dict[str, Any]:
    """The top-1 chunk gave no answer: ask once more with the top-3 chunks. Tokens of both calls count."""
    return _ask(state, runtime, window_size=3, usage_so_far=dict(state.get("usage", {})))


def _ask(
    state: RAGWorkflowState, runtime: Runtime[RAGRunContext], *, window_size: int, usage_so_far: dict[str, int]
) -> dict[str, Any]:
    context = runtime.context
    generator = AnswerGenerator(context.chat_models["answer"], answer_prompt_version(context.config))
    outcome = generator.answer(
        qid=str(state["qid"]),
        question=str(state["query_original"]),
        window=list(state["retrieval_results"])[:window_size],
        chunks_by_id=context.chunks_by_id,
    )
    update: dict[str, Any] = {
        "answer_text": outcome.answer_text,
        "citations": outcome.citations,
        "usage": add_usage(usage_so_far, outcome.usage),
        "generation_status": outcome.status,
    }
    if outcome.error is not None:
        update["generation_error"] = outcome.error
    if "guardrail" in state:
        update.update(_check_generation(state, context, window_size, outcome.answer_text))
    return update


def _check_generation(
    state: RAGWorkflowState, context: RAGRunContext, window_size: int, answer_text: str | None
) -> dict[str, Any]:
    """Guardrails after a generation call: instruction-like text in the passages is recorded, and an
    answer that isn't a plain span of those passages is blocked (no answer, no citations)."""
    window = list(state["retrieval_results"])[:window_size]
    passages = [
        str(context.chunks_by_id[r["chunk_id"]]["text"]) for r in window if r["chunk_id"] in context.chunks_by_id
    ]
    record = dict(state["guardrail"])
    record["passages"] = [asdict(f) for passage in passages for f in guardrails.check_passage(passage)]
    update: dict[str, Any] = {"guardrail": record}
    if answer_text is None:
        return update
    findings = guardrails.check_answer(answer_text, passages)
    record["answer"] = [asdict(finding) for finding in findings]
    if guardrails.is_blocked(findings):
        reasons = "; ".join(finding.detail for finding in findings)
        LOGGER.warning("Answer blocked by the guardrails for qid=%s: %s", state["qid"], reasons)
        record["blocked_answer"] = answer_text
        update.update(
            answer_text=None,
            citations=[],
            generation_status="blocked",
            generation_error=f"answer blocked by guardrails: {reasons}",
        )
    return update


def log_node(state: RAGWorkflowState, runtime: Runtime[RAGRunContext]) -> dict[str, Any]:
    latency_ms = round(max(0.0, (time.perf_counter() - float(state["start_time"])) * 1000.0), 3)
    retrieved_chunk_ids = {result["chunk_id"] for result in state.get("retrieval_results", [])}
    citations = list(state.get("citations", []))
    if any(citation not in retrieved_chunk_ids for citation in citations):
        raise ValueError("All citations must be a subset of the retrieved chunk_ids.")

    call_log = runtime.context.call_log
    payload = RunLogLine.model_validate(
        {
            "qid": state["qid"],
            "query_original": state["query_original"],
            "query_rewrites": list(state.get("rewrites", [])),
            "guard_report": dict(state.get("guard_report", {})),
            "retrieval_results": list(state.get("retrieval_results", [])),
            "answer_text": state.get("answer_text"),
            "citations": citations,
            "latency_ms": latency_ms,
            "usage": dict(state.get("usage", {})),
            "llm_calls": call_log.as_dicts(int(state.get("call_log_start", 0))) if call_log is not None else [],
            "config_snapshot": dict(state["config_snapshot"]),
            "rewrite_status": state.get("rewrite_status"),
            "rewrite_error": state.get("rewrite_error"),
            "generation_status": state.get("generation_status"),
            "generation_error": state.get("generation_error"),
            "rewrite_gate": state.get("rewrite_gate"),
            "guardrail": state.get("guardrail"),
        }
    ).model_dump()
    payload.pop("openai_usage")  # legacy field (read from old logs only); new logs write `usage`
    if payload["rewrite_gate"] is None:
        payload.pop("rewrite_gate")  # only gated runs record the gate
    if payload["guardrail"] is None:
        payload.pop("guardrail")  # only runs with guardrails on record them
    _append_jsonl(Path(state["log_path"]), payload)
    return {
        "latency_ms": latency_ms,
        "answer_text": payload["answer_text"],
        "citations": payload["citations"],
        "usage": payload["usage"],
        "llm_calls": payload["llm_calls"],
        # Return the statuses too, so the final state is as complete as the run log.
        "rewrite_status": payload["rewrite_status"],
        "rewrite_error": payload["rewrite_error"],
        "generation_status": payload["generation_status"],
        "generation_error": payload["generation_error"],
    }


def _rewrites_or_original(state: RAGWorkflowState) -> list[str]:
    """The rewrites to use, stripped and de-duplicated in order; the original question if there are none."""
    return unique_in_order(state.get("rewrites", []) or [str(state["query_original"])])


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_writable(path)  # never write into a protected folder (see protection.py)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
