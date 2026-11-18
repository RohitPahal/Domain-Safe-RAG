"""Retrieval must reproduce the official thesis results exactly.

Compares the real indexes (indexes/bm25, indexes/chroma) against the golden baseline
extracted from the official run logs (tools/extract_golden.py), for all 9 configs
(bm25 / dense / hybrid x no rewrite, multi-rewrite guard off, multi-rewrite guard on). Each
question is retrieved with the queries the official run used (multi-rewrite: its post-guard
rewrites, fused with RRF), through ``retrieve_queries``, the code the pipeline and
``eval-retrieval --queries-from-log`` use. Two views per config:

- evaluation view: top-20 per question (each query to depth 20), as in the thesis retrieval table;
- pipeline view: top-10 with retrieval depth 10, as fed to answer generation.
  (For hybrid and multi-rewrite the two views differ, because RRF fusion depends on the depth.
  For guard on, the thesis table's guard run repaired a few rewrites differently from the
  pipeline run, so each view has its own logged queries.)

Runs offline (Hugging Face offline mode). Skipped where the local-only golden file or the
indexes don't exist, e.g. in public CI.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "retrieval_baseline.jsonl"
BM25_DIR = REPO_ROOT / "indexes" / "bm25"
CHROMA_DIR = REPO_ROOT / "indexes" / "chroma"
RRF_K = 60  # rrf_k of the official config
CONFIGS = [
    f"{retriever}_{condition}"
    for retriever in ("bm25", "dense", "hybrid")
    for condition in ("none", "multi_guard_off", "multi_guard_on")
]

pytestmark = [
    # Importing the ML stack dominates this module's run time (~16 s warm), not the model
    # weights (~0.4 s). Quick loop without it: pytest -m "not slow".
    pytest.mark.slow,
    pytest.mark.skipif(
        not (GOLDEN_PATH.is_file() and BM25_DIR.is_dir() and (CHROMA_DIR / "chroma.sqlite3").is_file()),
        reason="local golden baseline or retrieval indexes not present",
    ),
]


@pytest.fixture(scope="module")
def golden() -> dict[str, list[dict]]:
    """Golden records grouped by config (e.g. ``hybrid_multi_guard_on``)."""
    by_config: dict[str, list[dict]] = {config: [] for config in CONFIGS}
    with GOLDEN_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            by_config[record["config"]].append(record)
    return by_config


@pytest.fixture(scope="module")
def retrievers(tmp_path_factory):
    """Load the real indexes once per module, without network access.

    The Chroma index is read from a temp copy: ChromaDB rewrites its index files whenever a
    collection is opened, even for a pure read, and tests must never
    modify real project data.
    """
    chroma_copy = tmp_path_factory.mktemp("parity") / "chroma"
    shutil.copytree(CHROMA_DIR, chroma_copy)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HF_HUB_OFFLINE", "1")
        patch.setenv("TRANSFORMERS_OFFLINE", "1")
        from rag_thesis.pipeline.fusion import retrieve_queries
        from rag_thesis.retrieval.bm25 import load_bm25
        from rag_thesis.retrieval.chroma_store import load_chroma_index
        from rag_thesis.retrieval.factory import make_retriever

        bm25 = load_bm25(BM25_DIR)
        # A temp query cache: each distinct query is encoded live once; its repeats (other configs,
        # the other view) reuse the exact stored vector, which keeps this module fast.
        dense = load_chroma_index(chroma_copy, query_cache_dir=chroma_copy.parent / "query_cache")

    # Same path as the pipeline and eval-retrieval: the factory decides what a mode means.
    retrievers_by_mode = {
        mode: make_retriever(mode, bm25=bm25, dense=dense, k_rrf=RRF_K) for mode in ("bm25", "dense", "hybrid")
    }

    def search(retriever: str, queries: list[str], k: int) -> list[str]:
        results = retrieve_queries(retrievers_by_mode[retriever], queries, k, retriever_mode=retriever, k_rrf=RRF_K)
        return [result["chunk_id"] for result in results]

    return search


def _mismatches(records: list[dict], search, k: int, queries_field: str, golden_field: str) -> list[str]:
    retriever = records[0]["retriever_mode"]
    return [record["qid"] for record in records if search(retriever, record[queries_field], k) != record[golden_field]]


@pytest.mark.parametrize("config", CONFIGS)
def test_evaluation_view_top20_matches_official(config, golden, retrievers):
    records = golden[config]
    assert len(records) == 120
    mismatches = _mismatches(records, retrievers, 20, "eval_queries", "eval_top20_chunk_ids")
    assert not mismatches, f"{config}: {len(mismatches)}/120 top-20 lists differ, e.g. {mismatches[:3]}"


@pytest.mark.parametrize("config", CONFIGS)
def test_pipeline_view_top10_matches_official(config, golden, retrievers):
    records = golden[config]
    assert len(records) == 120
    mismatches = _mismatches(records, retrievers, 10, "queries", "pipeline_top_chunk_ids")
    assert not mismatches, f"{config}: {len(mismatches)}/120 pipeline top-10 lists differ, e.g. {mismatches[:3]}"
