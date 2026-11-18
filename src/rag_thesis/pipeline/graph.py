"""Wiring of the query-time LangGraph: which nodes exist and the edges between them."""

from __future__ import annotations

from functools import lru_cache

# LangGraph is a required dependency (pyproject.toml). It is imported directly on purpose:
# if it is missing, the pipeline must fail loudly rather than silently run on a home-made
# substitute engine with different behaviour.
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rag_thesis.pipeline.nodes import (
    RAGRunContext,
    RAGWorkflowState,
    gate_node,
    generate_node,
    generate_retry_node,
    guard_node,
    log_node,
    refuse_node,
    retrieve_node,
    rewrite_fallback_node,
    rewrite_node,
    skip_generation_node,
    skip_guard_node,
)

GUARD_STEPS = ["guard_node", "skip_guard_node"]


def route_to_guard(state: RAGWorkflowState) -> str:
    return "guard_node" if state["guard_mode"] == "on" else "skip_guard_node"


def route_start(state: RAGWorkflowState) -> str:
    if state.get("guardrail", {}).get("question_blocked"):
        return "refuse_node"  # guardrails: nothing runs for a blocked question
    if state["rewrite_mode"] == "none":
        return route_to_guard(state)
    return "gate_node" if state.get("gate_threshold") is not None else "rewrite_node"


def route_after_gate(state: RAGWorkflowState) -> str:
    """R5: an unsure retrieval goes on to rewriting; a confident one straight to generation."""
    return "rewrite_node" if state["rewrite_gate"]["rewritten"] else route_after_retrieve(state)


def route_after_rewrite(state: RAGWorkflowState) -> str:
    return "rewrite_fallback_node" if state["rewrite_status"] == "failed" else route_to_guard(state)


def route_after_retrieve(state: RAGWorkflowState) -> str:
    if state.get("retrieval_results") and state.get("generation_enabled"):
        return "generate_node"
    return "skip_generation_node"


def route_after_generate(state: RAGWorkflowState) -> str:
    # The top-1 chunk gave no answer and there are more chunks: ask again with the top-3.
    if state["generation_status"] == "not_found" and len(state["retrieval_results"]) > 1:
        return "generate_retry_node"
    return "log_node"


def build_rag_graph() -> CompiledStateGraph:
    graph = StateGraph(RAGWorkflowState, context_schema=RAGRunContext)
    graph.add_node("refuse_node", refuse_node)
    graph.add_node("gate_node", gate_node)
    graph.add_node("rewrite_node", rewrite_node)
    graph.add_node("rewrite_fallback_node", rewrite_fallback_node)
    graph.add_node("guard_node", guard_node)
    graph.add_node("skip_guard_node", skip_guard_node)
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("generate_node", generate_node)
    graph.add_node("generate_retry_node", generate_retry_node)
    graph.add_node("skip_generation_node", skip_generation_node)
    graph.add_node("log_node", log_node)
    graph.add_conditional_edges(START, route_start, ["refuse_node", "gate_node", "rewrite_node", *GUARD_STEPS])
    graph.add_conditional_edges(
        "gate_node", route_after_gate, ["rewrite_node", "generate_node", "skip_generation_node"]
    )
    graph.add_conditional_edges("rewrite_node", route_after_rewrite, ["rewrite_fallback_node", *GUARD_STEPS])
    graph.add_conditional_edges("rewrite_fallback_node", route_to_guard, GUARD_STEPS)
    graph.add_edge("guard_node", "retrieve_node")
    graph.add_edge("skip_guard_node", "retrieve_node")
    graph.add_conditional_edges("retrieve_node", route_after_retrieve, ["generate_node", "skip_generation_node"])
    graph.add_conditional_edges("generate_node", route_after_generate, ["generate_retry_node", "log_node"])
    graph.add_edge("generate_retry_node", "log_node")
    graph.add_edge("skip_generation_node", "log_node")
    graph.add_edge("refuse_node", "log_node")
    graph.add_edge("log_node", END)
    return graph.compile()


@lru_cache(maxsize=1)
def compiled_rag_graph() -> CompiledStateGraph:
    """The one compiled graph of this process. It holds no run data, so every question shares it;
    what differs per run reaches the nodes through the runtime context."""
    return build_rag_graph()
