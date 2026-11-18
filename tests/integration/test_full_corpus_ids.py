"""The frozen 120 evaluation questions keep their gold evidence in the full corpus.

Documents and chunks get content-hash ids, so a question's doc_id, chunk ids and gold chunk ids
come from its own context only. This checks it on the real data: every evaluation question, its
gold record and every chunk of the subset corpus (data/eval, data/processed) are identical in the
full-corpus build (data/full, see data/README.md). Skipped where the local data doesn't exist,
e.g. in public CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSET = {"eval": REPO_ROOT / "data" / "eval", "processed": REPO_ROOT / "data" / "processed"}
FULL = {"eval": REPO_ROOT / "data" / "full" / "eval", "processed": REPO_ROOT / "data" / "full" / "processed"}
ARTIFACTS = [
    ("eval", "gold_evidence.jsonl", "qid"),
    ("eval", "questions.jsonl", "qid"),
    ("processed", "chunks.jsonl", "chunk_id"),
]

pytestmark = pytest.mark.skipif(
    not all((root[folder] / name).is_file() for root in (SUBSET, FULL) for folder, name, _ in ARTIFACTS),
    reason="local subset or full-corpus data not present",
)


def _by_key(path: Path, key: str) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_key = {row[key]: row for row in rows}
    assert len(by_key) == len(rows), f"duplicate {key} in {path}"
    return by_key


@pytest.mark.parametrize(("folder", "name", "key"), ARTIFACTS)
def test_every_subset_record_is_identical_in_the_full_corpus(folder, name, key):
    subset = _by_key(SUBSET[folder] / name, key)
    full = _by_key(FULL[folder] / name, key)
    differing = [item for item, row in subset.items() if full.get(item) != row]
    assert not differing, f"{name}: {len(differing)}/{len(subset)} records differ or are missing, e.g. {differing[:3]}"
