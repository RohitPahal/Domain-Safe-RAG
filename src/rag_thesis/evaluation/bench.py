"""Retrieval latency benchmark (``rag-thesis bench-retrieval``).

Measures speed only, never quality: how long loading the indexes takes, how long the first query
takes (one-off costs such as GPU set-up or loading the encoder land there), and the per-query time
after that, for each retriever over the evaluation questions.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

from rag_thesis.config import load_config
from rag_thesis.evaluation.retrieval import VALID_RETRIEVERS, resolve_eval_k_list
from rag_thesis.io import load_questions
from rag_thesis.protection import ensure_writable
from rag_thesis.retrieval.base import Retriever
from rag_thesis.retrieval.bm25 import load_bm25
from rag_thesis.retrieval.chroma_store import load_chroma_index
from rag_thesis.retrieval.factory import RETRIEVER_MODES, make_retriever

LATENCY_FILENAME = "retrieval_latency.csv"
SUMMARY_FILENAME = "retrieval_latency.md"
COLUMNS = ["retriever", "queries", "k", "index_load_s", "first_query_ms", "mean_ms", "p50_ms", "p95_ms", "max_ms"]


def time_queries(retriever: Retriever, queries: list[str], k: int) -> list[float]:
    """Milliseconds per query, in order (each query is asked exactly once)."""
    times_ms = []
    for query in queries:
        start = time.perf_counter()
        retriever.retrieve(query, k)
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return times_ms


def latency_row(name: str, k: int, load_s: float, times_ms: list[float]) -> dict[str, Any]:
    """The first query on its own (it pays one-off costs); statistics over the rest."""
    first_ms = times_ms[0] if times_ms else 0.0
    values = np.asarray(times_ms[1:] or times_ms or [0.0], dtype=float)
    return {
        "retriever": name,
        "queries": len(times_ms),
        "k": k,
        "index_load_s": round(load_s, 3),
        "first_query_ms": round(first_ms, 2),
        "mean_ms": round(float(values.mean()), 2),
        "p50_ms": round(float(np.percentile(values, 50)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
        "max_ms": round(float(values.max()), 2),
    }


def run_retrieval_benchmark(
    *,
    retriever: str,
    questions_path: Path,
    config_path: Path,
    bm25_dir: Path,
    vector_dir: Path,
    output_dir: Path,
    k: int | None = None,
    max_qids: int | None = None,
) -> dict[str, Any]:
    """Time each selected retriever over the questions; write a CSV and a Markdown summary."""
    if retriever not in VALID_RETRIEVERS:
        raise ValueError(f"retriever must be one of: {', '.join(VALID_RETRIEVERS)}")
    ensure_writable(output_dir)  # before any loading: never write into a protected folder
    config = load_config(config_path)
    top_k = k if k is not None else max(resolve_eval_k_list(config))
    queries = [str(question["question"]) for question in load_questions(questions_path, max_qids=max_qids)]
    selected = list(RETRIEVER_MODES) if retriever == "all" else [retriever]

    load_s = {"bm25": 0.0, "dense": 0.0}
    backends: dict[str, Any] = {"bm25": None, "dense": None}
    if any(name in {"bm25", "hybrid"} for name in selected):
        start = time.perf_counter()
        backends["bm25"] = load_bm25(bm25_dir)
        load_s["bm25"] = time.perf_counter() - start
    if any(name in {"dense", "hybrid"} for name in selected):
        start = time.perf_counter()
        # No query-embedding cache: every dense query is encoded, so the numbers are the retrievers' own cost
        # and don't depend on how warm a cache happens to be.
        backends["dense"] = load_chroma_index(vector_dir)
        load_s["dense"] = time.perf_counter() - start

    rows = []
    for name in selected:
        backend = make_retriever(
            name, bm25=backends["bm25"], dense=backends["dense"], k_rrf=int(config.get("rrf_k", 60))
        )
        times_ms = time_queries(backend, queries, top_k)
        # Hybrid needs both indexes; they are loaded once per run and shared with the other retrievers.
        name_load_s = sum(load_s[index] for index in ("bm25", "dense") if index == name or name == "hybrid")
        rows.append(latency_row(name, top_k, name_load_s, times_ms))

    csv_path, summary_path = output_dir / LATENCY_FILENAME, output_dir / SUMMARY_FILENAME
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Retrieval latency",
        "",
        f"{len(queries)} questions, top-{top_k}. `index_load_s`: loading the indexes the retriever needs "
        "(once per run, shared); `first_query_ms`: the first query, which pays one-off costs; "
        "the other columns cover the rest. The query-embedding cache is not used: every dense query is encoded.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(COLUMNS)) + " |",
        *("| " + " | ".join(str(row[column]) for column in COLUMNS) + " |" for row in rows),
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": rows, "csv_path": csv_path, "summary_path": summary_path}
