"""Offline analysis of guard decisions and multi-query rewrites in a benchmark folder.

Reads run logs + the rewrite cache (read-only) and reports:
  1. guard outcome per rewrite type, with the *initial* failure reasons recomputed
     (the logs keep only post-repair reasons for repaired rewrites),
  2. answer leakage (rewrite contains a gold answer absent from the question),
  3. negation coverage,
  4. which rewrite types find the gold chunk,
  5. per-question gains/losses vs the no-rewrite baseline,
  6. how often the original query is in the retrieval set,
  7. near-duplicate rewrites.
No API calls. Optional --json writes the numbers for tracking over time.

Usage:
    python tools/analyze_guard_rewrites.py [--json outputs/reports/guard_rewrite_analysis.json]
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

from rag_thesis.guard.domain_safe import _has_negation, assess_rewrite

DEFAULT_BENCHMARK = Path("outputs/final_benchmark/subset120_caseB_gem25flash_gem25flash_gem25pro_20260416")
SLOTS = ["bm25_lexical", "dense_semantic", "abbreviation_expanded", "minimal_entity_locked"]
RETRIEVERS = ("bm25", "dense", "hybrid")
ARTICLES = re.compile(r"\b(a|an|the)\b")
PUNCT = str.maketrans({c: " " for c in string.punctuation})


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _norm(text: str) -> str:
    return " ".join(ARTICLES.sub(" ", text.lower().translate(PUNCT)).split())


def _answers(question: dict) -> list[str]:
    raw = question.get("answers") or question.get("answer") or []
    raw = [raw] if isinstance(raw, str) else raw
    return [item["text"] if isinstance(item, dict) else str(item) for item in raw]


def _jaccard(a: str, b: str) -> float:
    left, right = set(_norm(a).split()), set(_norm(b).split())
    return len(left & right) / max(1, len(left | right))


def _hit(row: dict, gold_ids: set[str]) -> bool:
    return bool({r["chunk_id"] for r in row["retrieval_results"]} & gold_ids)


def analyze(benchmark: Path, questions_path: Path, gold_path: Path) -> dict:
    runs = benchmark / "subset120" / "runs"
    questions = {q["qid"]: q for q in _read_jsonl(questions_path)}
    gold = {g["qid"]: set(g["gold_chunk_ids"]) for g in _read_jsonl(gold_path)}
    cache = {c["qid"]: c["rewrites"] for c in _read_jsonl(benchmark / "cache" / "rewrites.jsonl")}
    out: dict = {}

    # 1. guard outcomes with recomputed initial reasons
    actions: dict[str, Counter] = defaultdict(Counter)
    initial_reasons: Counter = Counter()
    logged_empty_reasons = 0
    trigger_types: Counter = Counter()
    for row in _read_jsonl(runs / "dense_multi_guard_on.jsonl"):
        rewrites = cache[row["qid"]]
        for report in row["guard_report"]["reports"]:
            candidate = report["candidate_rewrite"]
            slot = SLOTS[rewrites.index(candidate)] if candidate in rewrites else "unknown"
            actions[slot][report["final_action"]] += 1
            if report["final_action"] == "pass":
                continue
            if not report["reason_codes"]:
                logged_empty_reasons += 1
            assessment = assess_rewrite(row["query_original"], candidate)
            initial_reasons.update(assessment.reason_codes)
            trigger_types.update(assessment.rewrite_token_types.get(t, "?") for t in assessment.added_tokens)
            trigger_types["missing"] += len(assessment.missing_tokens)
    out["guard_actions_per_slot"] = {slot: dict(actions[slot]) for slot in SLOTS}
    out["guard_initial_reasons"] = dict(initial_reasons)
    out["guard_flagged_with_empty_logged_reasons"] = logged_empty_reasons
    out["guard_trigger_token_types"] = dict(trigger_types)

    # 2. leakage
    leaks: Counter = Counter()
    leaked_qids: list[str] = []
    for qid, rewrites in cache.items():
        q_norm = f" {_norm(questions[qid]['question'])} "
        golds = [f" {_norm(a)} " for a in _answers(questions[qid]) if _norm(a)]
        leaked = False
        for slot, rewrite in zip(SLOTS, rewrites, strict=True):
            r_norm = f" {_norm(rewrite)} "
            if any(a in r_norm and a not in q_norm for a in golds):
                leaks[slot] += 1
                leaked = True
        if leaked:
            leaked_qids.append(qid)
    out["leakage"] = {
        "questions": len(leaked_qids),
        "total_questions": len(cache),
        "per_slot": dict(leaks),
        "qids": leaked_qids,
    }

    # 3. negation
    negated = [qid for qid in cache if _has_negation(questions[qid]["question"])]
    out["negation"] = {
        "questions_with_negation": len(negated),
        "rewrites_dropping_negation": sum(1 for qid in negated for r in cache[qid] if not _has_negation(r)),
    }

    # 4-6. retrieval contribution, gains/losses, original-query presence
    out["retrieval"] = {}
    for retriever in RETRIEVERS:
        by = {
            cond: {r["qid"]: r for r in _read_jsonl(runs / f"{retriever}_{cond}.jsonl")}
            for cond in ("none", "multi_guard_off", "multi_guard_on")
        }
        found: Counter = Counter()
        sole: Counter = Counter()
        for qid, row in by["multi_guard_off"].items():
            gold_rows = [r for r in row["retrieval_results"] if r["chunk_id"] in gold[qid]]
            if not gold_rows:
                continue
            finders = set().union(*(set(r["queries"]) for r in gold_rows))
            found.update(slot for slot, rw in zip(SLOTS, cache[qid], strict=True) if rw in finders)
            if len(finders) == 1:
                only = next(iter(finders))
                sole[SLOTS[cache[qid].index(only)] if only in cache[qid] else "original"] += 1
        stats = {"gold_found_by_slot": dict(found), "sole_finder": dict(sole)}
        for cond in ("multi_guard_off", "multi_guard_on"):
            base, test = by["none"], by[cond]
            stats[cond] = {
                "gained_vs_none": sum(_hit(test[q], gold[q]) and not _hit(base[q], gold[q]) for q in base),
                "lost_vs_none": sum(_hit(base[q], gold[q]) and not _hit(test[q], gold[q]) for q in base),
                "original_query_in_set": sum(r["query_original"].strip() in r["query_rewrites"] for r in test.values()),
            }
        out["retrieval"][retriever] = stats

    # 7. diversity
    out["near_duplicate_rewrite_pairs"] = {
        "count": sum(
            1 for rw in cache.values() for i in range(4) for j in range(i + 1, 4) if _jaccard(rw[i], rw[j]) > 0.8
        ),
        "of": 6 * len(cache),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    parser.add_argument("--gold", type=Path, default=Path("data/eval/gold_evidence.jsonl"))
    parser.add_argument(
        "--json", type=Path, default=None, help="optional output path (never inside the benchmark folder)"
    )
    args = parser.parse_args()
    result = analyze(args.benchmark, args.questions, args.gold)
    print(json.dumps(result, indent=2))
    if args.json:
        if args.benchmark.resolve() in args.json.resolve().parents:
            raise SystemExit("Refusing to write inside the protected benchmark folder.")
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
