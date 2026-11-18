"""Retriever audit: retrieval quality with confidence intervals, corpus checks and failures.

For each corpus (e.g. the closed subset and the full corpus) and retriever (BM25, dense, hybrid),
the frozen evaluation questions are retrieved with their own text at pipeline depth
(``retrieve_k``) and scored against two golds: strict (the question's own snippet) and
question-level (every snippet of the same BioASQ question, ``evaluation.retrieval.question_level_gold``).
Each mean has a 95% bootstrap CI that resamples BioASQ questions, not qids (``evaluation.stats``);
retrievers and corpora are compared with paired CIs of the difference. Also: corpus checks (gold
mapping, chunk lengths, duplicates), the questions a retriever misses, and a verdict per corpus.
No LLM calls. Writes into ``--output-dir``:

  retriever_audit.md             the report (a post-thesis result)
  retriever_audit_metrics.csv    one row per corpus x retriever x metric: mean, CI low, CI high
  retriever_audit_questions.csv  one row per corpus x retriever x question: first relevant rank per gold

Usage:
    python tools/retriever_audit.py --corpus subset indexes data --corpus full indexes/full data/full \\
        --output-dir outputs/reports
(each corpus: NAME, an index folder with ``bm25/`` and ``chroma/``, a data folder with ``processed/``
and ``eval/``)
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from rag_thesis.config import load_config
from rag_thesis.evaluation.retrieval import (
    bioasq_question_id,
    evaluate_retriever,
    question_level_gold,
    write_csv,
)
from rag_thesis.evaluation.stats import bootstrap_ci, paired_bootstrap_ci
from rag_thesis.io import load_chunks, load_gold_evidence, load_questions
from rag_thesis.protection import ensure_writable
from rag_thesis.retrieval.bm25 import load_bm25
from rag_thesis.retrieval.chroma_store import load_chroma_index

RETRIEVERS = ("bm25", "dense", "hybrid")
# (gold, depth, metric): the report's metrics. Depth "k" = retrieve_k (pipeline depth); 20 only for Recall@20.
METRICS = [
    ("strict", "k", "Recall@5"),
    ("strict", "k", "Recall@{k}"),
    ("strict", 20, "Recall@20"),
    ("strict", "k", "MRR"),
    ("strict", "k", "nDCG@10"),
    ("question", "k", "Hit@{k}"),
    ("question", "k", "MRR"),
]
COMPARED = [("strict", "MRR"), ("question", "Hit@{k}"), ("question", "MRR")]
NEAR_DUPLICATE_JACCARD = 0.8
SHINGLE_WORDS = 5


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def duplicate_pairs(chunks: list[dict[str, Any]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Exact duplicates (same normalised text) and near duplicates (5-word shingles, Jaccard >= 0.8;
    exact ones included). Shingles shared by more than 50 chunks are boilerplate and skipped."""
    by_text: dict[str, list[str]] = defaultdict(list)
    shingles: dict[str, set[str]] = {}
    for chunk in chunks:
        words = _words(chunk["text"])
        by_text[" ".join(words)].append(chunk["chunk_id"])
        span = max(1, len(words) - SHINGLE_WORDS + 1)
        shingles[chunk["chunk_id"]] = {" ".join(words[i : i + SHINGLE_WORDS]) for i in range(span)}
    exact = [pair for ids in by_text.values() for pair in combinations(sorted(ids), 2)]
    holders: dict[str, list[str]] = defaultdict(list)
    for chunk_id, chunk_shingles in shingles.items():
        for shingle in chunk_shingles:
            holders[shingle].append(chunk_id)
    shared: Counter[tuple[str, str]] = Counter()
    for ids in holders.values():
        if 1 < len(ids) <= 50:
            shared.update(combinations(sorted(ids), 2))
    near = [
        (a, b) for (a, b), n in sorted(shared.items()) if n / len(shingles[a] | shingles[b]) >= NEAR_DUPLICATE_JACCARD
    ]
    return exact, near


def corpus_checks(data_root: Path, golds: dict[str, dict[str, dict]]) -> dict[str, Any]:
    """Size, gold mapping, chunk length and duplicates of one corpus, and whether a duplicate pair
    splits an evaluation question's gold (one side gold, the other not)."""
    chunks = load_chunks(data_root / "processed" / "chunks.jsonl")
    corpus_gold = load_gold_evidence(data_root / "eval" / "gold_evidence.jsonl")
    exact, near = duplicate_pairs(chunks)
    splits = sum(
        (a in set(gold["gold_chunk_ids"])) != (b in set(gold["gold_chunk_ids"]))
        for by_qid in golds.values()
        for gold in by_qid.values()
        for a, b in near
    )
    return {
        "documents": len({chunk["doc_id"] for chunk in chunks}),
        "chunks": len(chunks),
        "longest_chunk_wordpieces": max(int(chunk["metadata"]["token_count"]) for chunk in chunks),
        "qa_pairs": len(corpus_gold),
        "mapping_status": dict(Counter(str(row.get("mapping_status")) for row in corpus_gold.values())),
        "two_gold_chunks": sum(len(row["gold_chunk_ids"]) > 1 for row in corpus_gold.values()),
        "exact_duplicate_pairs": len(exact),
        "near_duplicate_pairs": len(near),
        "gold_splitting_pairs": splits,
    }


def evaluate_corpus(
    index_root: Path, questions: list[dict], golds: dict[str, dict[str, dict]], config: dict[str, Any]
) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    """Per-question rows for every retriever x gold x depth the report needs."""
    k = int(config["retrieve_k"])
    backends = {
        "bm25_backend": load_bm25(index_root / "bm25"),
        "dense_backend": load_chroma_index(
            index_root / "chroma", query_cache_dir=config.get("query_embedding_cache_dir")
        ),
    }
    rows: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for retriever in RETRIEVERS:
        for gold_name, depth in sorted({(gold, k if d == "k" else d) for gold, d, _ in METRICS}):
            _, rows[(retriever, gold_name, depth)] = evaluate_retriever(
                retriever=retriever,
                questions=questions,
                gold_by_qid=golds[gold_name],
                config=config,
                eval_k_list=sorted({5, depth}),
                **backends,
            )
    return rows


def metric_rows(results: dict[str, dict], clusters: list[str], k: int) -> list[dict[str, Any]]:
    """Mean and 95% CI of every corpus x retriever x metric."""
    out = []
    for corpus, rows in results.items():
        for retriever in RETRIEVERS:
            for gold, depth, metric in METRICS:
                name = metric.format(k=k)
                values = [row[name] for row in rows[(retriever, gold, k if depth == "k" else depth)]]
                mean, low, high = bootstrap_ci(values, clusters=clusters)
                label = f"{gold} {name}" if name.startswith(("Hit", "MRR")) else name
                out.append(
                    {"corpus": corpus, "retriever": retriever, "metric": label, "mean": mean, "low": low, "high": high}
                )
    return out


def paired(results: dict[str, dict], clusters: list[str], k: int) -> list[dict[str, Any]]:
    """Paired differences with 95% CIs: retriever vs retriever in each corpus, corpus vs the first."""
    out = []
    corpora = list(results)
    for gold, metric in COMPARED:
        name = metric.format(k=k)
        values = {
            (corpus, retriever): [row[name] for row in results[corpus][(retriever, gold, k)]]
            for corpus in corpora
            for retriever in RETRIEVERS
        }
        for corpus in corpora:
            for a, b in combinations(RETRIEVERS, 2):
                diff = paired_bootstrap_ci(values[(corpus, a)], values[(corpus, b)], clusters=clusters)
                out.append({"comparison": f"{corpus}: {b} - {a}", "metric": f"{gold} {name}", "diff": diff})
        for corpus in corpora[1:]:
            for retriever in RETRIEVERS:
                diff = paired_bootstrap_ci(
                    values[(corpora[0], retriever)], values[(corpus, retriever)], clusters=clusters
                )
                out.append(
                    {"comparison": f"{retriever}: {corpus} - {corpora[0]}", "metric": f"{gold} {name}", "diff": diff}
                )
    return out


def verdict_lines(comparisons: list[dict[str, Any]], corpora: list[str], metric: str) -> list[str]:
    """Per retriever and corpus: better than, worse than or level with each other retriever on
    ``metric``, by whether the paired 95% CI excludes 0."""
    diffs = {row["comparison"]: row["diff"] for row in comparisons if row["metric"] == metric}
    lines = []
    for retriever in RETRIEVERS:
        parts = []
        for corpus in corpora:
            verdicts = []
            for other in RETRIEVERS:
                if other == retriever:
                    continue
                key = f"{corpus}: {retriever} - {other}"
                sign = 1.0
                if key not in diffs:  # stored the other way round
                    key, sign = f"{corpus}: {other} - {retriever}", -1.0
                mean, low, high = (sign * value for value in diffs[key])
                low, high = min(low, high), max(low, high)
                word = "better than" if low > 0 else "worse than" if high < 0 else "level with"
                verdicts.append(f"{word} {other} ({mean:+.3f})")
            parts.append(f"{corpus}: " + ", ".join(verdicts))
        lines.append(f"- **{retriever}:** " + "; ".join(parts) + ".")
    return lines


def _ci(mean: float, low: float, high: float, signed: bool = False) -> str:
    fmt = "+.3f" if signed else ".3f"
    return f"{mean:{fmt}} [{low:{fmt}}, {high:{fmt}}]"


def report_markdown(
    checks: dict[str, dict],
    metrics: list[dict],
    comparisons: list[dict],
    results: dict[str, dict],
    questions: list[dict],
    k: int,
) -> str:
    n_questions = len({bioasq_question_id(q["qid"]) for q in questions})
    lines = [
        "# Retriever audit (post-thesis result, not a thesis number)",
        "",
        f"{len(questions)} frozen evaluation qids ({n_questions} BioASQ questions), each retrieved with its own text "
        f"at pipeline depth k={k} (Recall@20: depth 20). Strict gold: the question's own snippet. Question-level "
        "gold: every snippet of the same BioASQ question. Means with 95% bootstrap CIs that resample BioASQ "
        "questions (10,000 resamples, seed 0).",
        "",
        "## Corpus checks",
        "",
        "| check | " + " | ".join(checks) + " |",
        "| --- | " + " | ".join("---:" for _ in checks) + " |",
    ]
    for key in next(iter(checks.values())):
        lines.append(f"| {key} | " + " | ".join(str(c[key]) for c in checks.values()) + " |")
    labels = list(dict.fromkeys(row["metric"] for row in metrics))
    lines += [
        "",
        "## Metrics",
        "",
        "| corpus | retriever | " + " | ".join(labels) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in labels) + " |",
    ]
    by_key = {(row["corpus"], row["retriever"], row["metric"]): row for row in metrics}
    for corpus in results:
        for retriever in RETRIEVERS:
            cells = [_ci(*(by_key[(corpus, retriever, label)][f] for f in ("mean", "low", "high"))) for label in labels]
            lines.append(f"| {corpus} | {retriever} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Paired differences",
        "",
        "A difference is marked * when its 95% CI excludes 0.",
        "",
        "| comparison | metric | difference [95% CI] |",
        "| --- | --- | ---: |",
    ]
    for row in comparisons:
        mean, low, high = row["diff"]
        mark = " *" if low > 0 or high < 0 else ""
        lines.append(f"| {row['comparison']} | {row['metric']} | {_ci(mean, low, high, signed=True)}{mark} |")
    lines += [
        "",
        "## Verdict per retriever",
        "",
        "By question-level MRR (strict gold counts only one of a question's snippets, so it understates "
        "retrieval on a corpus that holds the others); better / worse only where the paired 95% CI excludes 0.",
        "",
        *verdict_lines(comparisons, list(results), "question MRR"),
        "",
        "## Ranking per corpus (by question-level MRR)",
        "",
    ]
    for corpus in results:
        ranked = sorted(RETRIEVERS, key=lambda r: -by_key[(corpus, r, "question MRR")]["mean"])
        parts = [
            f"{r} (question MRR {by_key[(corpus, r, 'question MRR')]['mean']:.3f}, "
            f"strict MRR {by_key[(corpus, r, 'strict MRR')]['mean']:.3f})"
            for r in ranked
        ]
        lines.append(f"- **{corpus}:** " + " > ".join(parts) + ".")
    lines += [
        "",
        f"## Questions with no question-level gold chunk in the top-{k}",
        "",
        "| corpus | qid | question | " + " | ".join(RETRIEVERS) + " |",
        "| --- | --- | --- | " + " | ".join("---" for _ in RETRIEVERS) + " |",
    ]
    text_by_qid = {q["qid"]: q["question"] for q in questions}
    for corpus, rows in results.items():
        ranks = {r: {row["qid"]: row["first_relevant_rank"] for row in rows[(r, "question", k)]} for r in RETRIEVERS}
        for qid in text_by_qid:
            found = [ranks[r][qid] for r in RETRIEVERS]
            if any(rank == "" for rank in found):
                cells = [str(rank) if rank != "" else "miss" for rank in found]
                lines.append(f"| {corpus} | {qid} | {text_by_qid[qid][:70]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--corpus", nargs=3, action="append", required=True, metavar=("NAME", "INDEX_ROOT", "DATA_ROOT")
    )
    parser.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    parser.add_argument("--gold", type=Path, default=Path("data/eval/gold_evidence.jsonl"))
    parser.add_argument(
        "--all-gold",
        type=Path,
        default=Path("data/full/eval/gold_evidence.jsonl"),
        help="gold of every QA pair, for question-level gold (the full corpus build has them all)",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports"))
    args = parser.parse_args()

    config = load_config(args.config)
    k = int(config["retrieve_k"])
    questions = load_questions(args.questions)
    widened = question_level_gold(list(load_gold_evidence(args.all_gold).values()))
    golds = {"strict": load_gold_evidence(args.gold), "question": {q["qid"]: widened[q["qid"]] for q in questions}}
    clusters = [bioasq_question_id(q["qid"]) for q in questions]
    checks, results = {}, {}
    for name, index_root, data_root in args.corpus:
        checks[name] = corpus_checks(Path(data_root), golds)
        results[name] = evaluate_corpus(Path(index_root), questions, golds, config)
    metrics = metric_rows(results, clusters, k)
    comparisons = paired(results, clusters, k)

    write_csv(
        args.output_dir / "retriever_audit_metrics.csv",
        metrics,
        ["corpus", "retriever", "metric", "mean", "low", "high"],
    )
    question_rows = [
        {
            "corpus": corpus,
            "retriever": r,
            "qid": s["qid"],
            "strict_rank": s["first_relevant_rank"],
            "question_rank": q["first_relevant_rank"],
        }
        for corpus, rows in results.items()
        for r in RETRIEVERS
        for s, q in zip(rows[(r, "strict", k)], rows[(r, "question", k)], strict=True)
    ]
    write_csv(args.output_dir / "retriever_audit_questions.csv", question_rows, list(question_rows[0]))
    report_path = args.output_dir / "retriever_audit.md"
    ensure_writable(report_path)  # never write into a protected folder (see protection.py)
    report = report_markdown(checks, metrics, comparisons, results, questions, k)
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
