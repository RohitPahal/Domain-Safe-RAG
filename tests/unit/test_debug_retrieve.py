import json
from pathlib import Path

import tools.debug_retrieve as debug_retrieve
from tools.debug_retrieve import app
from typer.testing import CliRunner

RUNNER = CliRunner()


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query: str, k: int):
        return [dict(result) for result in self.results[:k]]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _chunks() -> list[dict]:
    return [
        {
            "chunk_id": "doc-1::0",
            "doc_id": "doc-1",
            "chunk_index": 0,
            "text": "Aspirin is also called acetylsalicylic acid. It is widely used as a pain reliever.",
            "char_start": 0,
            "char_end": 86,
            "metadata": {"source": "test"},
        },
        {
            "chunk_id": "doc-2::0",
            "doc_id": "doc-2",
            "chunk_index": 0,
            "text": "Ibuprofen is a nonsteroidal anti-inflammatory drug used for pain and fever.",
            "char_start": 0,
            "char_end": 80,
            "metadata": {"source": "test"},
        },
    ]


def _questions() -> list[dict]:
    return [
        {
            "qid": "q-1",
            "question": "What is the synonym of aspirin?",
            "doc_id": "doc-1",
            "answers": [{"text": "acetylsalicylic acid", "answer_start": 24}],
        }
    ]


def _gold() -> list[dict]:
    return [
        {
            "qid": "q-1",
            "gold_doc_id": "doc-1",
            "gold_chunk_ids": ["doc-1::0"],
            "mapping_status": "full_cover",
        }
    ]


def _config_path(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("rrf_k: 60\n", encoding="utf-8")
    return config_path


def _index_dirs(tmp_path: Path) -> tuple[Path, Path]:
    bm25_dir = tmp_path / "bm25"
    vector_dir = tmp_path / "vector"
    bm25_dir.mkdir()
    vector_dir.mkdir()
    return bm25_dir, vector_dir


def test_query_mode_runs_for_all_backends(monkeypatch, tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    _write_jsonl(chunks_path, _chunks())
    config_path = _config_path(tmp_path)
    bm25_dir, vector_dir = _index_dirs(tmp_path)

    monkeypatch.setattr(
        debug_retrieve,
        "load_bm25",
        lambda path: FakeRetriever(
            [
                {"chunk_id": "doc-1::0", "doc_id": "doc-1", "score": 1.0, "rank": 1},
                {"chunk_id": "doc-2::0", "doc_id": "doc-2", "score": 0.5, "rank": 2},
            ]
        ),
    )
    monkeypatch.setattr(
        debug_retrieve,
        "load_chroma_index",
        lambda path: FakeRetriever(
            [
                {"chunk_id": "doc-1::0", "doc_id": "doc-1", "score": 0.9, "rank": 1},
                {"chunk_id": "doc-2::0", "doc_id": "doc-2", "score": 0.4, "rank": 2},
            ]
        ),
    )

    for mode in ("bm25", "dense", "hybrid"):
        result = RUNNER.invoke(
            app,
            [
                "--query",
                "What is the synonym of aspirin?",
                "--mode",
                mode,
                "--k",
                "2",
                "--chunks-path",
                str(chunks_path),
                "--bm25-dir",
                str(bm25_dir),
                "--vector-dir",
                str(vector_dir),
                "--config-path",
                str(config_path),
            ],
        )

        assert result.exit_code == 0
        assert "Preview:" in result.output
        assert "Total Results Shown: 2" in result.output


def test_qid_mode_loads_question_and_marks_gold(monkeypatch, tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    questions_path = tmp_path / "questions.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    config_path = _config_path(tmp_path)
    bm25_dir, vector_dir = _index_dirs(tmp_path)
    _write_jsonl(chunks_path, _chunks())
    _write_jsonl(questions_path, _questions())
    _write_jsonl(gold_path, _gold())

    monkeypatch.setattr(
        debug_retrieve,
        "load_bm25",
        lambda path: FakeRetriever(
            [
                {"chunk_id": "doc-1::0", "doc_id": "doc-1", "score": 1.0, "rank": 1},
                {"chunk_id": "doc-2::0", "doc_id": "doc-2", "score": 0.5, "rank": 2},
            ]
        ),
    )

    result = RUNNER.invoke(
        app,
        [
            "--qid",
            "q-1",
            "--mode",
            "bm25",
            "--k",
            "2",
            "--chunks-path",
            str(chunks_path),
            "--questions-path",
            str(questions_path),
            "--gold-path",
            str(gold_path),
            "--bm25-dir",
            str(bm25_dir),
            "--vector-dir",
            str(vector_dir),
            "--config-path",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert "QID: q-1" in result.output
    assert "Question: What is the synonym of aspirin?" in result.output
    assert "Gold Chunk IDs: ['doc-1::0']" in result.output
    assert "Gold Hit: yes" in result.output
    assert "Any Gold Found in Top-2: yes" in result.output
    assert "Matching Gold Chunk IDs: ['doc-1::0']" in result.output
    assert "acetylsalicylic acid" in result.output


def test_invalid_argument_combinations_fail_cleanly(monkeypatch, tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    questions_path = tmp_path / "questions.jsonl"
    config_path = _config_path(tmp_path)
    bm25_dir, vector_dir = _index_dirs(tmp_path)
    _write_jsonl(chunks_path, _chunks())
    _write_jsonl(questions_path, _questions())

    monkeypatch.setattr(debug_retrieve, "load_bm25", lambda path: FakeRetriever([]))
    monkeypatch.setattr(debug_retrieve, "load_chroma_index", lambda path: FakeRetriever([]))

    both_result = RUNNER.invoke(
        app,
        [
            "--query",
            "test query",
            "--qid",
            "q-1",
            "--mode",
            "bm25",
            "--chunks-path",
            str(chunks_path),
            "--questions-path",
            str(questions_path),
            "--bm25-dir",
            str(bm25_dir),
            "--vector-dir",
            str(vector_dir),
            "--config-path",
            str(config_path),
        ],
    )
    neither_result = RUNNER.invoke(
        app,
        [
            "--mode",
            "bm25",
            "--chunks-path",
            str(chunks_path),
            "--questions-path",
            str(questions_path),
            "--bm25-dir",
            str(bm25_dir),
            "--vector-dir",
            str(vector_dir),
            "--config-path",
            str(config_path),
        ],
    )
    missing_qid_result = RUNNER.invoke(
        app,
        [
            "--qid",
            "missing-qid",
            "--mode",
            "bm25",
            "--chunks-path",
            str(chunks_path),
            "--questions-path",
            str(questions_path),
            "--bm25-dir",
            str(bm25_dir),
            "--vector-dir",
            str(vector_dir),
            "--config-path",
            str(config_path),
        ],
    )

    assert both_result.exit_code != 0
    assert "exactly one of --query or --qid" in both_result.output
    assert neither_result.exit_code != 0
    assert "exactly one of --query or --qid" in neither_result.output
    assert missing_qid_result.exit_code != 0
    assert "QID not found" in missing_qid_result.output


def test_backend_load_failure_is_reported_cleanly(monkeypatch, tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    config_path = _config_path(tmp_path)
    bm25_dir, vector_dir = _index_dirs(tmp_path)
    _write_jsonl(chunks_path, _chunks())

    def _raise_missing_dependency(path: Path):
        raise ModuleNotFoundError("rank_bm25")

    monkeypatch.setattr(debug_retrieve, "load_bm25", _raise_missing_dependency)

    result = RUNNER.invoke(
        app,
        [
            "--query",
            "test query",
            "--mode",
            "bm25",
            "--chunks-path",
            str(chunks_path),
            "--bm25-dir",
            str(bm25_dir),
            "--vector-dir",
            str(vector_dir),
            "--config-path",
            str(config_path),
        ],
    )

    assert result.exit_code != 0
    assert "Failed to load the BM25 backend" in result.output
