"""Tests for the ChromaDB dense index (real Chroma on a temp dir, fake encoder, no network)."""

import json
import re
import sys

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from rag_thesis.cli.build_index import app as build_indexes_app
from rag_thesis.retrieval import chroma_store
from rag_thesis.retrieval.chroma_store import (
    COLLECTION_NAME,
    _open_client,
    build_chroma_index,
    load_chroma_index,
    stored_embeddings,
)
from rag_thesis.retrieval.embeddings import chunk_records, encode_texts, load_sentence_transformer
from rag_thesis.retrieval.embeddings import resolve_device as _resolve_device

TOKEN_PATTERN = re.compile(r"[a-z]+")
VOCAB = ["cat", "dog", "bird", "friends"]
CONFIG = {
    "embed_model": "fake-model",
    "embed_batch_size": 8,
    "embed_max_seq_length": 256,
}


class FakeEncoder:
    """Bag-of-words vectors over a tiny vocabulary; enough to test ranking."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.max_seq_length = None

    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True):
        rows = []
        for text in texts:
            tokens = TOKEN_PATTERN.findall(text.lower())
            vector = np.array([tokens.count(word) for word in VOCAB] + [0.1], dtype=np.float32)
            rows.append(vector / np.linalg.norm(vector))
        return np.vstack(rows)


@pytest.fixture(autouse=True)
def fake_encoder(monkeypatch):
    monkeypatch.setattr(chroma_store, "load_sentence_transformer", lambda name, device: FakeEncoder(name, device))


def _chunks(texts):
    return [{"chunk_id": f"doc-{i}::0", "doc_id": f"doc-{i}", "text": text} for i, text in enumerate(texts)]


SAMPLE = _chunks(["cat sat on mat", "dog barked loudly", "cat and dog are friends", "bird flew away"])


def test_retrieve_ranks_and_formats_results(tmp_path):
    index = build_chroma_index(SAMPLE, CONFIG, tmp_path / "chroma")
    results = index.retrieve("bird", k=3)

    assert [r["rank"] for r in results] == [1, 2, 3]
    assert results[0]["chunk_id"] == "doc-3::0"
    assert results[0]["doc_id"] == "doc-3"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)  # same direction: cosine distance 0
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert set(results[0]) == {"chunk_id", "score", "rank", "doc_id"}


def test_retrieve_edge_cases(tmp_path):
    index = build_chroma_index(SAMPLE, CONFIG, tmp_path / "chroma")
    assert index.retrieve("cat", k=0) == []
    assert index.retrieve("cat", k=-1) == []
    assert index.retrieve("   ", k=3) == []
    assert len(index.retrieve("cat", k=100)) == len(SAMPLE)  # k larger than the collection


def test_load_round_trip_gives_identical_results(tmp_path):
    built = build_chroma_index(SAMPLE, CONFIG, tmp_path / "chroma")
    loaded = load_chroma_index(tmp_path / "chroma")
    assert loaded.retrieve("cat dog", k=4) == built.retrieve("cat dog", k=4)
    assert loaded.meta["num_items"] == len(SAMPLE)


def test_collection_uses_cosine_and_index_settings(tmp_path):
    index = build_chroma_index(SAMPLE, {**CONFIG, "hnsw_m": 32, "hnsw_ef_construction": 200}, tmp_path / "chroma")
    hnsw = index.collection.configuration_json["hnsw"]
    assert hnsw["space"] == "cosine"
    assert hnsw["max_neighbors"] == 32
    assert hnsw["ef_construction"] == 200


def test_stored_embeddings_are_the_encoder_output_in_row_order(tmp_path):
    index = build_chroma_index(SAMPLE, CONFIG, tmp_path / "chroma")
    expected = FakeEncoder("fake-model").encode([c["text"] for c in SAMPLE])
    np.testing.assert_allclose(stored_embeddings(index), expected, rtol=1e-6)


def test_rebuild_replaces_old_records(tmp_path):
    build_chroma_index(SAMPLE, CONFIG, tmp_path / "chroma")
    rebuilt = build_chroma_index(SAMPLE[:2], CONFIG, tmp_path / "chroma")
    assert rebuilt.collection.count() == 2
    assert load_chroma_index(tmp_path / "chroma").collection.count() == 2


def test_ties_are_broken_by_row_order(tmp_path):
    duplicates = _chunks(["cat", "cat", "cat"])
    index = build_chroma_index(duplicates, CONFIG, tmp_path / "chroma")
    assert [r["chunk_id"] for r in index.retrieve("cat", k=3)] == ["doc-0::0", "doc-1::0", "doc-2::0"]


def test_invalid_chunks_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        build_chroma_index([], CONFIG, tmp_path / "chroma")
    with pytest.raises(ValueError, match="Duplicate"):
        build_chroma_index([SAMPLE[0], SAMPLE[0]], CONFIG, tmp_path / "chroma")
    with pytest.raises(ValueError):
        build_chroma_index([{"chunk_id": "a", "doc_id": "d", "text": ""}], CONFIG, tmp_path / "chroma")


def test_load_errors_are_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_chroma_index(tmp_path / "does-not-exist")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="build-index --dense"):
        load_chroma_index(empty_dir)


def test_load_detects_record_count_mismatch(tmp_path):
    index = build_chroma_index(SAMPLE, CONFIG, tmp_path / "chroma")
    index.collection.delete(ids=["doc-0::0"])
    with pytest.raises(ValueError, match="rebuild"):
        load_chroma_index(tmp_path / "chroma")


def test_telemetry_is_disabled_and_collection_name(tmp_path):
    client = _open_client(tmp_path / "chroma")
    assert client.get_settings().anonymized_telemetry is False
    assert COLLECTION_NAME == "chunks"


# --- `rag-thesis build-index --dense` ----------------------------------------


RUNNER = CliRunner()


def _cli_inputs(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "".join(
            json.dumps({**c, "chunk_index": 0, "char_start": 0, "char_end": len(c["text"]), "metadata": {}}) + "\n"
            for c in SAMPLE
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "embed_model: fake-model\nembed_batch_size: 8\nembed_max_seq_length: 256\n", encoding="utf-8"
    )
    return ["--chunks-path", str(chunks_path), "--config-path", str(config_path)]


def test_build_index_dense_cli_writes_a_loadable_chroma_index(tmp_path):
    index_dir = tmp_path / "chroma"
    result = RUNNER.invoke(build_indexes_app, ["--dense", *_cli_inputs(tmp_path), "--index-dir", str(index_dir)])

    assert result.exit_code == 0, result.output
    assert f"dense_chunks_indexed: {len(SAMPLE)}" in result.output
    loaded = load_chroma_index(index_dir)
    assert loaded.collection.count() == len(SAMPLE)
    assert loaded.retrieve("bird", k=1)[0]["chunk_id"] == "doc-3::0"


def test_build_index_dense_cli_refuses_an_old_hnswlib_folder(tmp_path):
    legacy_dir = tmp_path / "vector"
    legacy_dir.mkdir()
    (legacy_dir / "index.bin").write_bytes(b"old hnswlib index")

    result = RUNNER.invoke(build_indexes_app, ["--dense", *_cli_inputs(tmp_path), "--index-dir", str(legacy_dir)])

    assert result.exit_code != 0
    assert "old hnswlib index" in result.output
    assert sorted(p.name for p in legacy_dir.iterdir()) == ["index.bin"]  # nothing written into it


# --- Embedding device selection ---------------------------------------------


def test_resolve_device_options(monkeypatch):
    assert _resolve_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="embed_device"):
        _resolve_device("gpu0")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="no CUDA GPU"):
        _resolve_device("cuda")  # never silently falls back to the CPU

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _resolve_device("auto") == "cuda"
    assert _resolve_device("cuda") == "cuda"


def test_build_records_the_embedding_device(tmp_path):
    index = build_chroma_index(SAMPLE, {**CONFIG, "embed_device": "cpu"}, tmp_path / "chroma")
    assert index.meta["embed_device"] == "cpu"
    assert load_chroma_index(tmp_path / "chroma").meta["embed_device"] == "cpu"


# --- Loading never writes into the wrong folder ------------------------------


def test_load_refuses_an_old_hnswlib_folder_without_writing(tmp_path):
    legacy_dir = tmp_path / "vector"
    legacy_dir.mkdir()
    (legacy_dir / "index.bin").write_bytes(b"old hnswlib index")
    with pytest.raises(ValueError, match="old hnswlib index"):
        load_chroma_index(legacy_dir)
    assert sorted(p.name for p in legacy_dir.iterdir()) == ["index.bin"]


def test_load_of_an_empty_folder_creates_no_database(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No Chroma database"):
        load_chroma_index(empty_dir)
    assert list(empty_dir.iterdir()) == []


# --- Embedding helpers --------------------------------------------------------


def test_missing_sentence_transformers_gives_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # make the import fail
    with pytest.raises(RuntimeError, match="sentence-transformers is required"):
        load_sentence_transformer("any-model", device="cpu")


def test_encode_texts_returns_float32_unit_rows():
    matrix = encode_texts(FakeEncoder("fake-model"), ["cat", "dog bird"], batch_size=2)
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0, rtol=1e-6)


def test_chunk_records_splits_and_validates():
    assert chunk_records(SAMPLE[:2]) == (
        ["doc-0::0", "doc-1::0"],
        ["doc-0", "doc-1"],
        [SAMPLE[0]["text"], SAMPLE[1]["text"]],
    )
    with pytest.raises(ValueError, match="doc_id"):
        chunk_records([{"chunk_id": "a", "doc_id": "", "text": "x"}])


def test_loading_an_index_defers_the_encoder_until_a_query_needs_it(tmp_path, monkeypatch):
    build_chroma_index(SAMPLE, CONFIG, tmp_path / "chroma")
    loads = []

    def counting_loader(name, device):
        loads.append(name)
        return FakeEncoder(name, device)

    monkeypatch.setattr(chroma_store, "load_sentence_transformer", counting_loader)
    index = load_chroma_index(tmp_path / "chroma")
    assert loads == []  # opening the index loads no model
    assert index.retrieve("   ", 3) == [] and loads == []  # nothing to encode, nothing loaded
    first = index.retrieve("cat on a mat", 3)
    second = index.retrieve("dog barked", 3)
    assert loads == [CONFIG["embed_model"]]  # loaded once, on the first real query
    assert first[0]["chunk_id"] == "doc-0::0" and second[0]["chunk_id"] == "doc-1::0"
    assert index.model.max_seq_length == CONFIG["embed_max_seq_length"]


def test_an_index_needs_an_encoder_or_a_loader():
    with pytest.raises(ValueError, match="encoder"):
        chroma_store.ChromaDenseIndex(collection=object(), model=None, meta={})
