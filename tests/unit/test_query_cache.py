"""Query-embedding cache: exact vectors, safe writes, and a dense index that never re-encodes."""

from __future__ import annotations

import numpy as np
import pytest

from rag_thesis import protection
from rag_thesis.evaluation import bench
from rag_thesis.evaluation import retrieval as retrieval_eval
from rag_thesis.pipeline import workflow
from rag_thesis.protection import ProtectedPathError, WriteGuard
from rag_thesis.retrieval import chroma_store
from rag_thesis.retrieval.chroma_store import build_chroma_index, load_chroma_index
from rag_thesis.retrieval.query_cache import QueryEmbeddingCache

CONFIG = {"embed_model": "fake-model", "embed_batch_size": 8, "embed_max_seq_length": 256, "embed_device": "cpu"}
CHUNKS = [
    {"chunk_id": f"doc-{i}::0", "doc_id": f"doc-{i}", "text": text}
    for i, text in enumerate(["cat sat on mat", "dog barked loudly", "bird flew away"])
]
VOCAB = ["cat", "dog", "bird", "mat", "barked", "flew"]
QUESTION_LINE = '{"qid": "q1", "question": "Q?", "doc_id": "d", "answers": [{"text": "x", "answer_start": 0}]}'
CHUNK_LINE = (
    '{"chunk_id": "d::0", "doc_id": "d", "chunk_index": 0, "text": "x", "char_start": 0, "char_end": 1, "metadata": {}}'
)
GOLD_LINE = '{"qid": "q1", "gold_doc_id": "d", "gold_chunk_ids": ["d::0"]}'


class FakeEncoder:
    max_seq_length = None

    def encode(self, texts, **kwargs):
        rows = [np.array([text.count(word) for word in VOCAB] + [0.1], dtype=np.float32) for text in texts]
        return np.vstack([row / np.linalg.norm(row) for row in rows])


@pytest.fixture
def loads(monkeypatch):
    calls: list[str] = []

    def counting_loader(name, device):
        calls.append(name)
        return FakeEncoder()

    monkeypatch.setattr(chroma_store, "load_sentence_transformer", counting_loader)
    return calls


def test_key_covers_model_length_and_text():
    base = QueryEmbeddingCache.key("m", 256, "What is BRCA1?")
    assert base == QueryEmbeddingCache.key("m", 256, "What is BRCA1?")  # stable
    others = {
        QueryEmbeddingCache.key("other", 256, "What is BRCA1?"),
        QueryEmbeddingCache.key("m", 128, "What is BRCA1?"),
        QueryEmbeddingCache.key("m", 256, "What is BRCA2?"),
    }
    assert base not in others and len(others) == 3


def test_vectors_round_trip_exactly(tmp_path):
    cache = QueryEmbeddingCache(tmp_path / "cache")
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32) / np.float32(3.7)
    assert cache.get("k") is None
    cache.put("k", vector)
    stored = cache.get("k")
    assert stored.dtype == np.float32 and np.array_equal(stored, vector)  # bit for bit
    assert sorted(path.name for path in (tmp_path / "cache").iterdir()) == ["k.npy"]  # no temp file left


def test_an_unreadable_cache_file_fails_loudly_and_names_itself(tmp_path):
    (tmp_path / "k.npy").write_bytes(b"")  # e.g. left behind by a crash
    with pytest.raises(ValueError, match="k.npy"):
        QueryEmbeddingCache(tmp_path).get("k")


def test_a_corrupt_cache_file_fails_loudly(tmp_path):
    cache = QueryEmbeddingCache(tmp_path)
    np.save(tmp_path / "k.npy", np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="Corrupt query-embedding cache"):
        cache.get("k")


def test_a_protected_cache_dir_is_read_but_never_written(tmp_path, monkeypatch):
    folder = tmp_path / "protected"
    QueryEmbeddingCache(folder).put("k", np.ones(3, dtype=np.float32))
    monkeypatch.setattr(protection, "DEFAULT_GUARD", WriteGuard([folder]))
    assert QueryEmbeddingCache(folder).get("k") is not None
    with pytest.raises(ProtectedPathError):
        QueryEmbeddingCache(folder).put("other", np.ones(3, dtype=np.float32))


def test_a_cached_query_is_never_re_encoded_and_needs_no_encoder(tmp_path, loads):
    build_chroma_index(CHUNKS, CONFIG, tmp_path / "chroma")
    loads.clear()
    cache_dir = tmp_path / "query_cache"

    first_run = load_chroma_index(tmp_path / "chroma", query_cache_dir=cache_dir)
    expected = first_run.retrieve("cat on a mat", 3)
    assert loads == ["fake-model"]  # cache miss: encoded once and stored
    assert len(list(cache_dir.glob("*.npy"))) == 1

    replay = load_chroma_index(tmp_path / "chroma", query_cache_dir=cache_dir)
    assert replay.retrieve("cat on a mat", 3) == expected
    assert loads == ["fake-model"]  # replay: the vector came from the cache, no encoder loaded


def test_without_a_cache_dir_nothing_is_stored(tmp_path, loads):
    build_chroma_index(CHUNKS, CONFIG, tmp_path / "chroma")
    index = load_chroma_index(tmp_path / "chroma")
    assert index.query_cache is None
    index.retrieve("cat on a mat", 3)
    assert not (tmp_path / "query_cache").exists()


@pytest.fixture
def inputs(tmp_path):
    """Small valid input files, so each reader gets as far as loading the dense index."""
    files = {name: tmp_path / f"{name}.jsonl" for name in ("questions", "chunks", "gold")}
    files["questions"].write_text(QUESTION_LINE + "\n", encoding="utf-8")
    files["chunks"].write_text(CHUNK_LINE + "\n", encoding="utf-8")
    files["gold"].write_text(GOLD_LINE + "\n", encoding="utf-8")
    files["config"] = tmp_path / "config.yaml"
    files["config"].write_text("rrf_k: 60\nquery_embedding_cache_dir: my/cache\n", encoding="utf-8")
    return files


def _capture_cache_dir(module, monkeypatch) -> list:
    seen: list = []

    def fake_load(index_dir, query_cache_dir=None):
        seen.append(query_cache_dir)
        raise RuntimeError("stop after loading")

    monkeypatch.setattr(module, "load_chroma_index", fake_load)
    return seen


def test_the_pipeline_uses_the_configured_cache_dir(inputs, tmp_path, monkeypatch):
    seen = _capture_cache_dir(workflow, monkeypatch)
    with pytest.raises(RuntimeError, match="stop after loading"):
        workflow.build_rag_run_context(
            questions_path=inputs["questions"],
            chunks_path=inputs["chunks"],
            retriever_mode="dense",
            config_path=inputs["config"],
            vector_dir=tmp_path / "chroma",
        )
    assert seen == ["my/cache"]


def test_retrieval_evaluation_uses_the_configured_cache_dir(inputs, tmp_path, monkeypatch):
    seen = _capture_cache_dir(retrieval_eval, monkeypatch)
    with pytest.raises(RuntimeError, match="stop after loading"):
        retrieval_eval.run_retrieval_evaluation(
            retriever="dense",
            questions_path=inputs["questions"],
            gold_path=inputs["gold"],
            config_path=inputs["config"],
            bm25_dir=tmp_path,
            vector_dir=tmp_path / "chroma",
            output_dir=tmp_path / "out",
        )
    assert seen == ["my/cache"]


def test_the_benchmark_measures_without_the_cache(inputs, tmp_path, monkeypatch):
    seen = _capture_cache_dir(bench, monkeypatch)
    with pytest.raises(RuntimeError, match="stop after loading"):
        bench.run_retrieval_benchmark(
            retriever="dense",
            questions_path=inputs["questions"],
            config_path=inputs["config"],
            bm25_dir=tmp_path,
            vector_dir=tmp_path / "chroma",
            output_dir=tmp_path / "out",
        )
    assert seen == [None]  # timings must not depend on how warm a cache is


def test_a_miss_in_a_protected_cache_is_refused_before_the_encoder_loads(tmp_path, loads, monkeypatch):
    build_chroma_index(CHUNKS, CONFIG, tmp_path / "chroma")
    loads.clear()
    folder = tmp_path / "protected_cache"
    folder.mkdir()
    monkeypatch.setattr(protection, "DEFAULT_GUARD", WriteGuard([folder]))
    index = load_chroma_index(tmp_path / "chroma", query_cache_dir=folder)
    with pytest.raises(ProtectedPathError):
        index.retrieve("cat on a mat", 3)
    assert loads == []  # refused before any expensive work
