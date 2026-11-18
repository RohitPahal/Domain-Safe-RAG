"""The official benchmark's rewrite cache replays completely, read-only, with no model at all.

Its records were written before the provider was part of the cache key, so every hit here
goes through the legacy-key lookup. Skips where the official results are absent (e.g. public CI).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rag_thesis.config import load_config
from rag_thesis.llm.cache import load_cache_records
from rag_thesis.rewrite.rewriter import QueryRewriter

OFFICIAL = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "final_benchmark"
    / "subset120_caseB_gem25flash_gem25flash_gem25pro_20260416"
)
CACHE = OFFICIAL / "cache" / "rewrites.jsonl"

pytestmark = pytest.mark.skipif(not CACHE.is_file(), reason="official benchmark cache not present")


def test_all_official_rewrites_replay_with_zero_calls_and_no_writes():
    before = hashlib.sha256(CACHE.read_bytes()).hexdigest()
    records = list(load_cache_records(CACHE).values())
    rewriter = QueryRewriter(config=load_config(OFFICIAL / "config_snapshot.yaml"), cache_path=CACHE)  # no model

    for record in records:
        replayed = rewriter.rewrite(
            qid=record["qid"], query_original=record["query_original"], rewrite_mode=record["rewrite_mode"]
        )
        assert replayed == record["rewrites"], record["qid"]

    assert len(records) == 120
    assert hashlib.sha256(CACHE.read_bytes()).hexdigest() == before  # read-only
