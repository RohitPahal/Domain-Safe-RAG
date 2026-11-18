from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

from rag_thesis.pipeline.workflow import (
    RAGWorkflowState,
    build_rag_run_context,
    run_rag_workflow_with_context,
)

DEFAULT_RANDOM_SEED = 42


def select_qids(
    ordered_qids: list[str],
    *,
    first_n: int | None = None,
    random_n: int | None = None,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[str]:
    if first_n is not None and random_n is not None:
        raise ValueError("Specify only one of first_n or random_n.")
    if first_n is not None:
        return ordered_qids[: min(first_n, len(ordered_qids))]
    if random_n is not None:
        sample_size = min(random_n, len(ordered_qids))
        return random.Random(seed).sample(ordered_qids, sample_size)
    return list(ordered_qids)


def run_rag_batch(
    *,
    retriever_mode: str,
    rewrite_mode: str,
    guard_mode: str,
    k: int,
    questions_path: Path,
    chunks_path: Path,
    config_path: Path,
    bm25_dir: Path,
    vector_dir: Path,
    log_path: Path,
    first_n: int | None = None,
    random_n: int | None = None,
    seed: int = DEFAULT_RANDOM_SEED,
    generate: bool = False,
    progress_callback: Callable[[int, int, RAGWorkflowState], None] | None = None,
) -> list[RAGWorkflowState]:
    context = build_rag_run_context(
        questions_path=questions_path,
        chunks_path=chunks_path,
        retriever_mode=retriever_mode,
        config_path=config_path,
        bm25_dir=bm25_dir,
        vector_dir=vector_dir,
        allow_llm_calls=generate,
    )
    selected_qids = select_qids(
        list(context.questions_by_id.keys()),
        first_n=first_n,
        random_n=random_n,
        seed=seed,
    )

    states: list[RAGWorkflowState] = []
    total = len(selected_qids)
    for index, qid in enumerate(selected_qids, start=1):
        state = run_rag_workflow_with_context(
            qid=qid,
            retriever_mode=retriever_mode,
            rewrite_mode=rewrite_mode,
            guard_mode=guard_mode,
            k=k,
            context=context,
            log_path=log_path,
            generation_enabled=True if generate else None,
        )
        states.append(state)
        if progress_callback is not None:
            progress_callback(index, total, state)
    return states
