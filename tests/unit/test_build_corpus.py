import json
import re
from pathlib import Path

from typer.testing import CliRunner

import rag_thesis.cli.build_corpus
from rag_thesis.cli.build_corpus import app as build_corpus_app
from rag_thesis.cli.validate import app as validate_artifacts_app
from rag_thesis.data.corpus import build_chunks_for_document, build_corpus_artifacts, map_answers_to_chunks
from rag_thesis.ids import deterministic_hash
from rag_thesis.schemas import AnswerSpan

RUNNER = CliRunner()
TOKEN_PATTERN = re.compile(r"\S+")


class FakeTokenizer:
    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        truncation: bool = False,
        **kwargs,
    ) -> dict:
        matches = list(TOKEN_PATTERN.finditer(text))
        return {
            "input_ids": list(range(len(matches))),
            "offset_mapping": [(match.start(), match.end()) for match in matches],
        }


class SubwordLikeTokenizer:
    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        truncation: bool = False,
        **kwargs,
    ) -> dict:
        offsets: list[tuple[int, int]] = []
        for match in TOKEN_PATTERN.finditer(text):
            token_start = match.start()
            token_end = match.end()
            midpoint = min(token_start + 2, token_end)
            offsets.append((token_start, midpoint))
            if midpoint < token_end:
                offsets.append((midpoint, token_end))
        return {
            "input_ids": list(range(len(offsets))),
            "offset_mapping": offsets,
        }


def _install_fake_tokenizer(monkeypatch) -> None:
    monkeypatch.setattr(rag_thesis.cli.build_corpus, "load_embed_tokenizer", lambda *args, **kwargs: FakeTokenizer())


def _build_subset_fixture() -> dict:
    repeated_context = "Alpha protein regulates beta cells."
    long_tokens = [f"tok{i:03d}" for i in range(520)]
    long_context = " ".join(long_tokens)
    long_answer_text = " ".join(long_tokens[430:435])
    long_answer_start = long_context.index(long_answer_text)

    return {
        "version": "1.0",
        "data": [
            {
                "title": "Doc A",
                "paragraphs": [
                    {
                        "context": repeated_context,
                        "qas": [
                            {
                                "id": "q-short-success",
                                "question": "Which protein regulates beta cells?",
                                "answers": [
                                    {
                                        "text": "Alpha protein",
                                        "answer_start": 0,
                                    }
                                ],
                            },
                            {
                                "id": "q-short-failed",
                                "question": "Which text is outside the document?",
                                "answers": [
                                    {
                                        "text": "missing span",
                                        "answer_start": 999,
                                    }
                                ],
                            },
                        ],
                    }
                ],
            },
            {
                "title": "Doc B",
                "paragraphs": [
                    {
                        "context": repeated_context,
                        "qas": [
                            {
                                "id": "q-short-duplicate-context",
                                "question": "What kind of cells are regulated?",
                                "answers": [
                                    {
                                        "text": "beta cells",
                                        "answer_start": 24,
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "context": long_context,
                        "qas": [
                            {
                                "id": "q-long-success",
                                "question": "Which token span is used in the long document?",
                                "answers": [
                                    {
                                        "text": long_answer_text,
                                        "answer_start": long_answer_start,
                                    }
                                ],
                            }
                        ],
                    },
                ],
            },
        ],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_build_corpus_produces_expected_artifacts_and_passes_validator(tmp_path, monkeypatch):
    _install_fake_tokenizer(monkeypatch)
    input_path = tmp_path / "bioasq_subset.json"
    processed_dir = tmp_path / "processed"
    eval_dir = tmp_path / "eval"
    report_dir = tmp_path / "reports"
    _write_json(input_path, _build_subset_fixture())

    result = RUNNER.invoke(
        build_corpus_app,
        [
            "--in",
            str(input_path),
            "--processed-dir",
            str(processed_dir),
            "--eval-dir",
            str(eval_dir),
            "--report-dir",
            str(report_dir),
        ],
    )

    assert result.exit_code == 0
    assert "documents: 2" in result.output
    assert "questions: 4" in result.output
    assert "gold_evidence: 4" in result.output
    assert "mapping_failures: 1" in result.output

    documents = _read_jsonl(processed_dir / "documents.jsonl")
    chunks = _read_jsonl(processed_dir / "chunks.jsonl")
    questions = _read_jsonl(eval_dir / "questions.jsonl")
    gold = _read_jsonl(eval_dir / "gold_evidence.jsonl")
    failures = _read_jsonl(report_dir / "mapping_failures.jsonl")

    assert len(documents) == 2
    repeated_context = "Alpha protein regulates beta cells."
    repeated_doc_id = deterministic_hash(repeated_context)
    assert any(document["doc_id"] == repeated_doc_id for document in documents)
    assert sum(1 for document in documents if document["doc_id"] == repeated_doc_id) == 1

    chunks_by_doc: dict[str, list[dict]] = {}
    doc_text_by_id = {document["doc_id"]: document["text"] for document in documents}
    for chunk in chunks:
        chunks_by_doc.setdefault(chunk["doc_id"], []).append(chunk)
        assert chunk["chunk_id"] == f"{chunk['doc_id']}::{chunk['chunk_index']}"
        assert 0 <= chunk["char_start"] <= chunk["char_end"] <= len(doc_text_by_id[chunk["doc_id"]])
        assert chunk["text"] == doc_text_by_id[chunk["doc_id"]][chunk["char_start"] : chunk["char_end"]]

    for doc_id, doc_chunks in chunks_by_doc.items():
        sorted_chunks = sorted(doc_chunks, key=lambda chunk: chunk["chunk_index"])
        starts = [chunk["char_start"] for chunk in sorted_chunks]
        indexes = [chunk["chunk_index"] for chunk in sorted_chunks]
        ids = [chunk["chunk_id"] for chunk in sorted_chunks]
        assert starts == sorted(starts)
        assert indexes == list(range(len(sorted_chunks)))
        assert ids == [f"{doc_id}::{index}" for index in indexes]

    long_doc_id = max(chunks_by_doc, key=lambda candidate: len(chunks_by_doc[candidate]))
    long_doc_chunks = sorted(chunks_by_doc[long_doc_id], key=lambda chunk: chunk["chunk_index"])
    assert [chunk["metadata"]["token_start"] for chunk in long_doc_chunks] == [0, 140, 280, 420]
    assert [chunk["metadata"]["token_count"] for chunk in long_doc_chunks] == [180, 180, 180, 100]
    assert all(
        chunk["metadata"]["tokenizer_name"] == "sentence-transformers/all-MiniLM-L6-v2" for chunk in long_doc_chunks
    )

    assert len(questions) == 4
    assert all(question["answers"] for question in questions)
    assert len({question["qid"] for question in questions}) == 4

    gold_by_qid = {line["qid"]: line for line in gold}
    assert gold_by_qid["q-short-success"]["mapping_status"] == "full_cover"
    assert gold_by_qid["q-long-success"]["mapping_status"] == "full_cover"
    assert gold_by_qid["q-short-failed"]["mapping_status"] == "failed"
    assert gold_by_qid["q-short-failed"]["gold_chunk_ids"] == []

    assert len(failures) == 1
    assert failures[0]["qid"] == "q-short-failed"

    validate_result = RUNNER.invoke(
        validate_artifacts_app,
        [
            "--processed-dir",
            str(processed_dir),
            "--eval-dir",
            str(eval_dir),
        ],
    )

    assert validate_result.exit_code == 0
    assert "Validation passed." in validate_result.output


def test_build_corpus_uses_raw_context_exactly_for_doc_id(tmp_path, monkeypatch):
    _install_fake_tokenizer(monkeypatch)
    context_with_whitespace = "  Alpha protein regulates beta cells.  "
    payload = {
        "version": "1.0",
        "data": [
            {
                "title": "Doc A",
                "paragraphs": [
                    {
                        "context": context_with_whitespace,
                        "qas": [
                            {
                                "id": "q1",
                                "question": "Which protein regulates beta cells?",
                                "answers": [{"text": "Alpha protein", "answer_start": 2}],
                            }
                        ],
                    }
                ],
            },
            {
                "title": "Doc B",
                "paragraphs": [
                    {
                        "context": context_with_whitespace,
                        "qas": [
                            {
                                "id": "q2",
                                "question": "Which cells are regulated?",
                                "answers": [{"text": "beta cells", "answer_start": 26}],
                            }
                        ],
                    }
                ],
            },
        ],
    }

    input_path = tmp_path / "bioasq_subset.json"
    processed_dir = tmp_path / "processed"
    eval_dir = tmp_path / "eval"
    report_dir = tmp_path / "reports"
    _write_json(input_path, payload)

    result = RUNNER.invoke(
        build_corpus_app,
        [
            "--in",
            str(input_path),
            "--processed-dir",
            str(processed_dir),
            "--eval-dir",
            str(eval_dir),
            "--report-dir",
            str(report_dir),
        ],
    )

    assert result.exit_code == 0
    documents = _read_jsonl(processed_dir / "documents.jsonl")
    assert len(documents) == 1
    assert documents[0]["doc_id"] == deterministic_hash(context_with_whitespace)
    assert documents[0]["text"] == context_with_whitespace


def test_overlap_only_mapping_falls_back_to_overlapping_chunks():
    context = "tok0 tok1 tok2 tok3 tok4 tok5"
    doc_id = deterministic_hash(context)
    tokenizer = FakeTokenizer()
    _, chunk_windows = build_chunks_for_document(
        context=context,
        doc_id=doc_id,
        tokenizer=tokenizer,
        tokenizer_name="fake-tokenizer",
        chunk_tokens=3,
        chunk_overlap=0,
    )
    answer_text = "tok2 tok3"
    answer_start = context.index(answer_text)

    mapping = map_answers_to_chunks(
        context=context,
        answers=[AnswerSpan(text=answer_text, answer_start=answer_start)],
        chunk_windows=chunk_windows,
    )

    assert mapping.mapping_status == "overlap_only"
    assert mapping.gold_chunk_ids == [f"{doc_id}::0", f"{doc_id}::1"]


def test_build_corpus_chunk_boundaries_are_deterministic(tmp_path, monkeypatch):
    _install_fake_tokenizer(monkeypatch)
    input_path = tmp_path / "bioasq_subset.json"
    processed_dir_a = tmp_path / "processed_a"
    eval_dir_a = tmp_path / "eval_a"
    report_dir_a = tmp_path / "reports_a"
    processed_dir_b = tmp_path / "processed_b"
    eval_dir_b = tmp_path / "eval_b"
    report_dir_b = tmp_path / "reports_b"
    _write_json(input_path, _build_subset_fixture())

    result_a = RUNNER.invoke(
        build_corpus_app,
        [
            "--in",
            str(input_path),
            "--processed-dir",
            str(processed_dir_a),
            "--eval-dir",
            str(eval_dir_a),
            "--report-dir",
            str(report_dir_a),
        ],
    )
    result_b = RUNNER.invoke(
        build_corpus_app,
        [
            "--in",
            str(input_path),
            "--processed-dir",
            str(processed_dir_b),
            "--eval-dir",
            str(eval_dir_b),
            "--report-dir",
            str(report_dir_b),
        ],
    )

    assert result_a.exit_code == 0
    assert result_b.exit_code == 0
    assert _read_jsonl(processed_dir_a / "chunks.jsonl") == _read_jsonl(processed_dir_b / "chunks.jsonl")


def test_build_chunks_follow_tokenizer_offsets_not_whitespace_tokens():
    context = "alpha beta"
    doc_id = deterministic_hash(context)
    tokenizer = SubwordLikeTokenizer()

    chunks, _ = build_chunks_for_document(
        context=context,
        doc_id=doc_id,
        tokenizer=tokenizer,
        tokenizer_name="subword-like",
        chunk_tokens=2,
        chunk_overlap=0,
    )

    assert len(chunks) == 2
    assert chunks[0].text == "alpha"
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == 5
    assert chunks[0].metadata["token_count"] == 2
    assert chunks[1].text == "beta"
    assert chunks[1].char_start == 6
    assert chunks[1].char_end == 10
    assert chunks[1].metadata["token_count"] == 2


def test_gold_ids_do_not_depend_on_the_rest_of_the_corpus(tmp_path):
    """A question's doc, chunks and gold ids come from its own context only (content hashes), so the
    frozen evaluation questions keep the same gold ids when the corpus grows (subset -> full)."""
    subset = _build_subset_fixture()
    full = json.loads(json.dumps(subset))
    other_context = "Gamma kinase binds delta receptors in liver cells."
    other_qa = {
        "id": "q-other",
        "question": "What binds delta receptors?",
        "answers": [{"text": "Gamma kinase", "answer_start": 0}],
    }
    full["data"].insert(0, {"title": "Doc C", "paragraphs": [{"context": other_context, "qas": [other_qa]}]})
    builds = {}
    for name, payload in (("full", full), ("subset", subset)):
        input_path = tmp_path / f"{name}.json"
        _write_json(input_path, payload)
        build_corpus_artifacts(
            input_path=input_path,
            processed_dir=tmp_path / name / "processed",
            eval_dir=tmp_path / name / "eval",
            report_dir=tmp_path / name / "reports",
            tokenizer=FakeTokenizer(),
            tokenizer_name="fake-tokenizer",
            chunk_tokens=180,
            chunk_overlap=40,
        )
        builds[name] = {
            artifact: {row.get("qid", row.get("chunk_id")): row for row in _read_jsonl(tmp_path / name / path)}
            for artifact, path in (
                ("gold", "eval/gold_evidence.jsonl"),
                ("questions", "eval/questions.jsonl"),
                ("chunks", "processed/chunks.jsonl"),
            )
        }
    subset_build, full_build = builds["subset"], builds["full"]
    assert len(subset_build["chunks"]) > 2 and len(full_build["chunks"]) > len(subset_build["chunks"])
    for artifact in ("gold", "questions", "chunks"):
        for key, row in subset_build[artifact].items():
            assert full_build[artifact][key] == row, (artifact, key)
