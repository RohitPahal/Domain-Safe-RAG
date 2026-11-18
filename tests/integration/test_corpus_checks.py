"""Guards on the real corpora: no chunk is truncated by the encoder, every gold mapping resolves.

For each built corpus (data/processed, and data/full/processed if built):
- every chunk, tokenized as the dense encoder sees it (with its special tokens), fits in
  ``embed_max_seq_length``, so the encoder never silently drops the end of a chunk;
- every question has a gold mapping that didn't fail, and every gold chunk id exists in the corpus.

Skipped where the local data doesn't exist, e.g. in public CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPORA = {
    "subset": (REPO_ROOT / "data" / "processed", REPO_ROOT / "data" / "eval"),
    "full": (REPO_ROOT / "data" / "full" / "processed", REPO_ROOT / "data" / "full" / "eval"),
}
PRESENT = [
    name
    for name, (processed, eval_dir) in CORPORA.items()
    if (processed / "chunks.jsonl").is_file() and (eval_dir / "gold_evidence.jsonl").is_file()
]

pytestmark = pytest.mark.skipif(not PRESENT, reason="no local corpus data")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def config() -> dict:
    from rag_thesis.config import load_config

    return load_config(REPO_ROOT / "configs" / "default.yaml")


@pytest.mark.slow  # loads the encoder's tokenizer
@pytest.mark.parametrize("corpus", PRESENT)
def test_no_chunk_is_truncated_by_the_encoder(corpus, config):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HF_HUB_OFFLINE", "1")
        patch.setenv("TRANSFORMERS_OFFLINE", "1")
        from rag_thesis.data.corpus import load_embed_tokenizer

        tokenizer = load_embed_tokenizer(config["embed_model"])
    texts = [row["text"] for row in _rows(CORPORA[corpus][0] / "chunks.jsonl")]
    lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]]
    limit = int(config["embed_max_seq_length"])
    too_long = [length for length in lengths if length > limit]
    assert not too_long, f"{corpus}: {len(too_long)} chunks exceed {limit} tokens (longest {max(too_long)})"


@pytest.mark.parametrize("corpus", PRESENT)
def test_every_gold_mapping_resolves_to_existing_chunks(corpus):
    processed, eval_dir = CORPORA[corpus]
    chunk_ids = {row["chunk_id"] for row in _rows(processed / "chunks.jsonl")}
    gold = _rows(eval_dir / "gold_evidence.jsonl")
    questions = {row["qid"] for row in _rows(eval_dir / "questions.jsonl")}
    assert {row["qid"] for row in gold} == questions
    failed = [row["qid"] for row in gold if row.get("mapping_status") == "failed" or not row["gold_chunk_ids"]]
    assert not failed, f"{corpus}: {len(failed)} questions without a gold mapping, e.g. {failed[:3]}"
    dangling = [cid for row in gold for cid in row["gold_chunk_ids"] if cid not in chunk_ids]
    assert not dangling, f"{corpus}: gold chunk ids missing from the corpus, e.g. {dangling[:3]}"
