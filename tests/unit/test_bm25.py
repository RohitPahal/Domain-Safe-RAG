import json
import math
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_thesis.cli.build_index import app as build_indexes_app
from rag_thesis.retrieval.bm25 import (
    CHUNK_IDS_FILENAME,
    DEFAULT_BM25_VARIANT,
    DOC_IDS_FILENAME,
    META_FILENAME,
    TOKENIZED_CORPUS_FILENAME,
    LuceneBM25,
    build_bm25,
    load_bm25,
    make_bm25_scorer,
)

RUNNER = CliRunner()


def _sample_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "doc-1::0",
            "doc_id": "doc-1",
            "chunk_index": 0,
            "text": "cat sat on mat",
            "char_start": 0,
            "char_end": 14,
            "metadata": {"source": "test"},
        },
        {
            "chunk_id": "doc-1::1",
            "doc_id": "doc-1",
            "chunk_index": 1,
            "text": "dog barked loudly",
            "char_start": 15,
            "char_end": 32,
            "metadata": {"source": "test"},
        },
        {
            "chunk_id": "doc-2::0",
            "doc_id": "doc-2",
            "chunk_index": 0,
            "text": "cat and dog became friends",
            "char_start": 0,
            "char_end": 26,
            "metadata": {"source": "test"},
        },
        {
            "chunk_id": "doc-3::0",
            "doc_id": "doc-3",
            "chunk_index": 0,
            "text": "bird flew away",
            "char_start": 0,
            "char_end": 14,
            "metadata": {"source": "test"},
        },
    ]


def _write_chunks_jsonl(path: Path, chunks: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk))
            handle.write("\n")


def test_bm25_retrieve_returns_unique_chunk_ids_and_doc_ids(tmp_path):
    index = build_bm25(_sample_chunks())

    results = index.retrieve("cat friends", k=3)

    assert len(results) == 3
    assert len({result["chunk_id"] for result in results}) == 3
    assert [result["rank"] for result in results] == [1, 2, 3]
    assert all("doc_id" in result and result["doc_id"] for result in results)
    assert results[0]["score"] >= results[1]["score"] >= results[2]["score"]
    assert results[0]["chunk_id"] == "doc-2::0"
    assert results[0]["doc_id"] == "doc-2"


def test_bm25_save_and_load_preserve_ranking(tmp_path):
    index = build_bm25(_sample_chunks())
    index_dir = tmp_path / "bm25"
    index.save_bm25(index_dir)

    loaded_index = load_bm25(index_dir)
    original_results = index.retrieve("cat dog", k=4)
    loaded_results = loaded_index.retrieve("cat dog", k=4)

    assert loaded_results == original_results


def test_build_indexes_cli_creates_expected_files(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    index_dir = tmp_path / "bm25"
    _write_chunks_jsonl(chunks_path, _sample_chunks())

    result = RUNNER.invoke(
        build_indexes_app,
        [
            "--bm25",
            "--chunks-path",
            str(chunks_path),
            "--index-dir",
            str(index_dir),
        ],
    )

    assert result.exit_code == 0
    assert not (index_dir / "bm25.pkl").exists()  # no pickle is written any more
    assert (index_dir / CHUNK_IDS_FILENAME).exists()
    assert (index_dir / DOC_IDS_FILENAME).exists()
    assert (index_dir / TOKENIZED_CORPUS_FILENAME).exists()
    assert (index_dir / META_FILENAME).exists()

    loaded_index = load_bm25(index_dir)
    results = loaded_index.retrieve("dog", k=3)
    assert len(results) == 3
    assert all(result["doc_id"] for result in results)


# --- Explicit BM25 variants -------------------------------------------------


def test_default_variant_is_lucene_thesis_variant():
    index = build_bm25(_sample_chunks())
    assert DEFAULT_BM25_VARIANT == "lucene"
    assert index.variant == "lucene"
    assert isinstance(index.bm25, LuceneBM25)


def test_lucene_scores_match_formula():
    corpus = [["cat", "sat"], ["dog"], ["cat", "dog", "cat"]]
    scorer = LuceneBM25(corpus)
    n_docs, avgdl, k1, b = 3, 2.0, 1.5, 0.75
    idf_cat = math.log(1 + (n_docs - 2 + 0.5) / (2 + 0.5))
    tf, dl = 2, 3
    expected = idf_cat * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    assert scorer.get_scores(["cat"])[2] == pytest.approx(expected)


def test_okapi_variant_is_explicit_opt_in():
    pytest.importorskip("rank_bm25")
    index = build_bm25(_sample_chunks(), variant="okapi")
    assert index.variant == "okapi"
    assert type(index.bm25).__name__ == "BM25Okapi"


def test_okapi_without_package_fails_instead_of_falling_back(monkeypatch):
    monkeypatch.setitem(sys.modules, "rank_bm25", None)
    with pytest.raises(RuntimeError, match="rank-bm25"):
        make_bm25_scorer([["a"]], "okapi")


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="Unknown BM25 variant"):
        make_bm25_scorer([["a"]], "bm42")


def test_load_uses_meta_variant_and_explicit_override(tmp_path):
    pytest.importorskip("rank_bm25")
    index_dir = tmp_path / "bm25"
    build_bm25(_sample_chunks(), variant="okapi").save_bm25(index_dir)
    assert json.loads((index_dir / META_FILENAME).read_text(encoding="utf-8"))["variant"] == "okapi"
    assert load_bm25(index_dir).variant == "okapi"
    assert load_bm25(index_dir, variant="lucene").variant == "lucene"


def test_legacy_index_without_variant_loads_as_lucene_and_ignores_pickle(tmp_path):
    index_dir = tmp_path / "bm25"
    build_bm25(_sample_chunks()).save_bm25(index_dir)
    meta_path = index_dir / META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.pop("variant")
    meta["backend"] = "BM25Okapi"  # what the legacy index meta says
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    (index_dir / "bm25.pkl").write_bytes(b"not a real pickle")  # a stale legacy file must be ignored

    loaded = load_bm25(index_dir)
    assert loaded.variant == "lucene"
    assert isinstance(loaded.bm25, LuceneBM25)
