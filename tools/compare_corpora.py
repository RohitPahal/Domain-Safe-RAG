"""Retrieval of the 9 official conditions on two or more corpora (e.g. subset vs full), offline.

Each condition replays the queries of its official run log (``none``: the question itself;
``multi``: the logged post-guard rewrites, guard-on repairs included), retrieved at pipeline depth
(``retrieve_k``) and fused exactly as in the pipeline (``evaluation.retrieval``). The questions and
gold evidence are the frozen evaluation set; their ids are the same in every corpus built from the
same data (content hashes). No LLM calls. Writes into ``--output-dir``:

  corpus_comparison.csv          one row per corpus x condition (Recall@5, Recall@k, MRR@k)
  corpus_comparison_details.csv  one row per corpus x condition x question (first relevant rank)
  corpus_comparison.md           side by side, with how many questions gained or lost a gold chunk
                                 in the top-k against the first corpus

Usage:
    python tools/compare_corpora.py --corpus subset indexes --corpus full indexes/full \\
        --output-dir outputs/reports/phase5_full_corpus
(each corpus folder holds ``bm25/`` and ``chroma/``)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rag_thesis.config import load_config
from rag_thesis.evaluation.retrieval import evaluate_retriever, load_logged_queries, write_csv
from rag_thesis.io import load_gold_evidence, load_questions
from rag_thesis.protection import ensure_writable
from rag_thesis.retrieval.bm25 import load_bm25
from rag_thesis.retrieval.chroma_store import load_chroma_index

DEFAULT_RUNS_DIR = Path(
    "outputs/final_benchmark/subset120_caseB_gem25flash_gem25flash_gem25pro_20260416/subset120/runs"
)
CONDITIONS = [
    f"{retriever}_{condition}"
    for retriever in ("bm25", "dense", "hybrid")
    for condition in ("none", "multi_guard_off", "multi_guard_on")
]


def compare_corpora(
    corpora: list[tuple[str, Path]], runs_dir: Path, questions_path: Path, gold_path: Path, config_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Metrics and per-question rows for every corpus x condition; also returns the depth k."""
    config = load_config(config_path)
    k = int(config["retrieve_k"])
    questions = load_questions(questions_path)
    gold_by_qid = load_gold_evidence(gold_path)
    queries = {condition: load_logged_queries(runs_dir / f"{condition}.jsonl") for condition in CONDITIONS}
    metrics: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for corpus, index_root in corpora:
        backends = {
            "bm25_backend": load_bm25(index_root / "bm25"),
            "dense_backend": load_chroma_index(
                index_root / "chroma", query_cache_dir=config.get("query_embedding_cache_dir")
            ),
        }
        for condition in CONDITIONS:
            row, rows = evaluate_retriever(
                retriever=condition.split("_")[0],
                questions=questions,
                gold_by_qid=gold_by_qid,
                config=config,
                eval_k_list=[5, k],
                queries_by_qid=queries[condition],
                **backends,
            )
            metrics.append({"corpus": corpus, "condition": condition, **row})
            details.extend(
                {
                    "corpus": corpus,
                    "condition": condition,
                    "qid": r["qid"],
                    "first_relevant_rank": r["first_relevant_rank"],
                }
                for r in rows
            )
    return metrics, details, k


def summary_markdown(metrics: list[dict[str, Any]], details: list[dict[str, Any]], k: int) -> str:
    """One table: each corpus' Recall@k and MRR@k per condition, and the questions that gained or
    lost a gold chunk in the top-k against the first corpus."""
    corpora = list(dict.fromkeys(row["corpus"] for row in metrics))
    reference = corpora[0]
    by_key = {(row["corpus"], row["condition"]): row for row in metrics}
    hit = {
        (row["corpus"], row["condition"], row["qid"]): row["first_relevant_rank"] != ""
        and int(row["first_relevant_rank"]) <= k
        for row in details
    }
    qids = sorted({row["qid"] for row in details})
    header = ["condition"] + [f"{c} R@{k}" for c in corpora] + [f"{c} MRR@{k}" for c in corpora]
    header += [f"{c} vs {reference}: gained / lost" for c in corpora[1:]]
    lines = [
        "# Retrieval by corpus (post-thesis result, not a thesis number)",
        "",
        f"{len(qids)} frozen evaluation questions; queries replayed from the official run logs; "
        f"pipeline depth k={k}. Gained / lost: questions with a gold chunk in the top-{k} in one corpus "
        "but not the other.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |",
    ]
    for condition in CONDITIONS:
        cells = [condition]
        cells += [f"{float(by_key[(c, condition)][f'Recall@{k}']):.4f}" for c in corpora]
        cells += [f"{float(by_key[(c, condition)]['MRR']):.4f}" for c in corpora]
        for corpus in corpora[1:]:
            gained = sum(hit[(corpus, condition, q)] and not hit[(reference, condition, q)] for q in qids)
            lost = sum(hit[(reference, condition, q)] and not hit[(corpus, condition, q)] for q in qids)
            cells.append(f"{gained} / {lost}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--corpus", nargs=2, action="append", required=True, metavar=("NAME", "INDEX_ROOT"), help="repeatable"
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    parser.add_argument("--gold", type=Path, default=Path("data/eval/gold_evidence.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    corpora = [(name, Path(root)) for name, root in args.corpus]
    metrics, details, k = compare_corpora(corpora, args.runs_dir, args.questions, args.gold, args.config)
    metric_fields = list(metrics[0])  # corpus, condition, then every metric evaluate_retriever reports
    write_csv(args.output_dir / "corpus_comparison.csv", metrics, metric_fields)
    write_csv(
        args.output_dir / "corpus_comparison_details.csv",
        details,
        ["corpus", "condition", "qid", "first_relevant_rank"],
    )
    summary_path = args.output_dir / "corpus_comparison.md"
    ensure_writable(summary_path)  # never write into a protected folder (see protection.py)
    summary = summary_markdown(metrics, details, k)
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
