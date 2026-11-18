"""tools/retriever_audit.py: duplicates, verdicts and the report, on fakes (no real data)."""

from __future__ import annotations

import json
from pathlib import Path

import tools.retriever_audit as audit
from tools.retriever_audit import RETRIEVERS, duplicate_pairs, verdict_lines

from rag_thesis.evaluation.retrieval import question_level_gold

WORDS = [f"w{i}" for i in range(30)]


def _chunk(chunk_id: str, text: str) -> dict:
    doc_id = chunk_id.split("::")[0]
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "chunk_index": 0,
        "text": text,
        "char_start": 0,
        "char_end": len(text),
        "metadata": {"token_count": len(text.split())},
    }


def test_duplicate_pairs_finds_exact_and_near_copies_across_documents():
    chunks = [
        _chunk("a::0", " ".join(WORDS)),
        _chunk("b::0", " ".join(WORDS).upper() + "."),  # same words: exact after normalising
        _chunk("c::0", " ".join(WORDS[:-1] + ["other"])),  # one word changed: Jaccard 25/27
        _chunk("d::0", "an unrelated passage about something else entirely here"),
    ]
    exact, near = duplicate_pairs(chunks)
    assert exact == [("a::0", "b::0")]
    assert near == [("a::0", "b::0"), ("a::0", "c::0"), ("b::0", "c::0")]


def test_verdicts_follow_the_paired_intervals():
    comparisons = [
        {"comparison": "full: dense - bm25", "metric": "question MRR", "diff": (-0.3, -0.4, -0.2)},
        {"comparison": "full: hybrid - bm25", "metric": "question MRR", "diff": (-0.05, -0.1, 0.01)},
        {"comparison": "full: hybrid - dense", "metric": "question MRR", "diff": (0.25, 0.1, 0.4)},
        {"comparison": "full: dense - bm25", "metric": "strict MRR", "diff": (0.9, 0.8, 1.0)},  # other metric
    ]
    lines = verdict_lines(comparisons, ["full"], "question MRR")
    assert lines == [
        "- **bm25:** full: better than dense (+0.300), level with hybrid (+0.050).",
        "- **dense:** full: worse than bm25 (-0.300), worse than hybrid (-0.250).",
        "- **hybrid:** full: level with bm25 (-0.050), better than dense (+0.250).",
    ]


class SiblingFirstIndex:
    """Ranks every query's sibling snippet (doc s) first and its own snippet (doc o) second."""

    def retrieve(self, query: str, k: int) -> list[dict]:
        return [
            {"chunk_id": cid, "doc_id": cid.split("::")[0], "score": 1.0 / rank, "rank": rank}
            for rank, cid in enumerate(["s::0", "o::0"][:k], start=1)
        ]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_the_report_scores_both_golds_and_lists_the_checks(tmp_path, monkeypatch):
    questions = [
        {"qid": "Q_001", "question": "which gene?", "doc_id": "o", "answers": [{"text": "x", "answer_start": 0}]},
        {"qid": "Q_002", "question": "which gene?", "doc_id": "s", "answers": [{"text": "x", "answer_start": 0}]},
    ]
    gold_rows = [
        {"qid": "Q_001", "gold_doc_id": "o", "gold_chunk_ids": ["o::0"], "mapping_status": "full_cover"},
        {"qid": "Q_002", "gold_doc_id": "s", "gold_chunk_ids": ["s::0"], "mapping_status": "full_cover"},
    ]
    data = tmp_path / "data"
    _write(data / "processed" / "chunks.jsonl", [_chunk("o::0", "own snippet"), _chunk("s::0", "sibling snippet")])
    _write(data / "eval" / "gold_evidence.jsonl", gold_rows)
    monkeypatch.setattr(audit, "load_bm25", lambda path: SiblingFirstIndex())
    monkeypatch.setattr(audit, "load_chroma_index", lambda path, query_cache_dir=None: SiblingFirstIndex())
    strict = {row["qid"]: row for row in gold_rows}
    golds = {"strict": strict, "question": question_level_gold(gold_rows)}
    config = {"retrieve_k": 10, "rrf_k": 60}

    results = {"tiny": audit.evaluate_corpus(tmp_path / "indexes", questions, golds, config)}
    checks = {"tiny": audit.corpus_checks(data, golds)}
    clusters = ["Q", "Q"]
    metrics = audit.metric_rows(results, clusters, 10)
    report = audit.report_markdown(checks, metrics, audit.paired(results, clusters, 10), results, questions, 10)

    by_key = {(row["retriever"], row["metric"]): row["mean"] for row in metrics}
    for retriever in RETRIEVERS:
        assert by_key[(retriever, "strict MRR")] == 0.75  # Q_001: own snippet at rank 2; Q_002: rank 1
        assert by_key[(retriever, "question MRR")] == 1.0  # the sibling snippet counts too
    assert checks["tiny"]["chunks"] == 2 and checks["tiny"]["mapping_status"] == {"full_cover": 2}
    assert "2 frozen evaluation qids (1 BioASQ questions)" in report
    for section in ("## Corpus checks", "## Metrics", "## Paired differences", "## Verdict per retriever"):
        assert section in report
