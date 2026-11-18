"""Extract a golden retrieval baseline from the official benchmark run logs.

Reads the read-only official run logs and writes, per (config, qid), the queries
that were actually used for retrieval and the top-k chunk ids that came back.
Later refactors are checked against this file (parity tests). No API calls.

Usage:
    python tools/extract_golden.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

OFFICIAL_ROOT = Path("outputs/final_benchmark/subset120_caseB_gem25flash_gem25flash_gem25pro_20260416/subset120")
OFFICIAL_RUNS_DIR = OFFICIAL_ROOT / "runs"
# The official retrieval tables were produced by a separate evaluation that re-ran
# retrieval at depth 20 using the logged final queries. With RRF fusion, depth changes
# the top-10, so both views are stored: pipeline (k=10, fed the generator) and eval (k=20).
OFFICIAL_RETRIEVAL_DIR = OFFICIAL_ROOT / "retrieval"
GOLD_EVIDENCE_PATH = Path("data/eval/gold_evidence.jsonl")
OUTPUT_PATH = Path("tests/golden/retrieval_baseline.jsonl")
SUMMARY_PATH = Path("tests/golden/retrieval_baseline_summary.json")
CONFIGS = [
    f"{retriever}_{condition}"
    for retriever in ("bm25", "dense", "hybrid")
    for condition in ("none", "multi_guard_off", "multi_guard_on")
]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=OFFICIAL_RUNS_DIR)
    parser.add_argument("--retrieval-dir", type=Path, default=OFFICIAL_RETRIEVAL_DIR)
    parser.add_argument("--gold", type=Path, default=GOLD_EVIDENCE_PATH)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()

    gold = {row["qid"]: set(row["gold_chunk_ids"]) for row in _read_jsonl(args.gold)}
    records: list[dict] = []
    summary: dict[str, dict] = {}

    for config in CONFIGS:
        rows = _read_jsonl(args.runs_dir / f"{config}.jsonl")
        eval_rows = _read_eval_details(args.retrieval_dir / f"{config}_details.csv")
        pipeline_recall = 0.0
        pipeline_mrr = 0.0
        eval_recall = 0.0
        eval_mrr = 0.0
        for row in rows:
            qid = row["qid"]
            gold_ids = gold[qid]
            chunk_ids = [result["chunk_id"] for result in row["retrieval_results"]]
            eval_row = eval_rows[qid]
            pipeline_recall += _recall(chunk_ids, gold_ids, 10)
            pipeline_mrr += _reciprocal_rank(chunk_ids, gold_ids)
            eval_recall += _recall(eval_row["chunk_ids"], gold_ids, 10)
            eval_mrr += _reciprocal_rank(eval_row["chunk_ids"], gold_ids)
            snapshot = row["config_snapshot"]
            records.append(
                {
                    "config": config,
                    "qid": qid,
                    "retriever_mode": snapshot["retriever_mode"],
                    "rewrite_mode": snapshot["rewrite_mode"],
                    "guard_mode": snapshot["guard_mode"],
                    "pipeline_k": snapshot["k"],
                    "queries": row["query_rewrites"],
                    "pipeline_top_chunk_ids": chunk_ids,
                    "eval_queries": eval_row["queries"],
                    "eval_top20_chunk_ids": eval_row["chunk_ids"],
                }
            )
        n = len(rows)
        summary[config] = {
            "n": n,
            "pipeline_k10_recall_at_10": round(pipeline_recall / n, 6),
            "pipeline_k10_mrr": round(pipeline_mrr / n, 6),
            "eval_k20_recall_at_10": round(eval_recall / n, 6),
            "eval_k20_mrr": round(eval_mrr / n, 6),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with args.summary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"{'config':24s} {'pipe R@10':>10s} {'pipe MRR':>9s} {'eval R@10':>10s} {'eval MRR':>9s}")
    for config, m in summary.items():
        print(
            f"{config:24s} {m['pipeline_k10_recall_at_10']:10.6f} {m['pipeline_k10_mrr']:9.6f} "
            f"{m['eval_k20_recall_at_10']:10.6f} {m['eval_k20_mrr']:9.6f}"
        )


def _read_eval_details(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["qid"]: {
                "queries": json.loads(row["final_queries"]),
                "chunk_ids": json.loads(row["retrieved_chunk_ids"]),
            }
            for row in csv.DictReader(handle)
        }


def _recall(chunk_ids: list[str], gold_ids: set[str], k: int) -> float:
    return len(set(chunk_ids[:k]) & gold_ids) / len(gold_ids)


def _reciprocal_rank(chunk_ids: list[str], gold_ids: set[str]) -> float:
    first = next((rank for rank, cid in enumerate(chunk_ids, 1) if cid in gold_ids), None)
    return 1.0 / first if first else 0.0


if __name__ == "__main__":
    main()
