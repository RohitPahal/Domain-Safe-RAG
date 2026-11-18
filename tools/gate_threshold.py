"""R5: choose, on the dev split, the confidence below which a question gets rewritten.

Each dev question is retrieved with its own text at pipeline depth on the full corpus (where the dev
contexts are). The confidence is the retriever's top-1 score (``retrieval.confidence``); a question
"fails" when no relevant chunk (question-level gold) is at rank 1. Per retriever: how well low
confidence predicts failure (AUC, 95% bootstrap CI) and the threshold that best separates the two
(Youden's J = sensitivity + specificity - 1; rewrite when the top-1 score is below it). No rewrites
are needed, so this runs offline. Writes rewrite_gate.json (for the config's ``rewrite_gate``) and
rewrite_gate.md into ``--output-dir``.

Usage:
    python tools/gate_threshold.py --output-dir outputs/reports/rewrite_gate
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from rag_thesis.config import load_config
from rag_thesis.evaluation.retrieval import compute_first_relevant_rank, question_level_gold
from rag_thesis.evaluation.stats import auc
from rag_thesis.io import load_gold_evidence, load_questions
from rag_thesis.protection import ensure_writable
from rag_thesis.retrieval.bm25 import load_bm25
from rag_thesis.retrieval.chroma_store import load_chroma_index
from rag_thesis.retrieval.confidence import top1_score
from rag_thesis.retrieval.factory import make_retriever

RETRIEVERS = ("bm25", "dense", "hybrid")
RESAMPLES = 2000


def youden_threshold(scores: list[float], failed: list[bool]) -> dict[str, float]:
    """The threshold t maximising sensitivity + specificity - 1 for "failed when score < t" (ties go
    to the lower t), with its sensitivity, specificity and the share of questions it would rewrite."""
    n_fail = sum(failed)
    n_ok = len(failed) - n_fail
    if n_fail == 0 or n_ok == 0:
        raise ValueError("need both failed and successful questions to choose a threshold.")
    values = sorted(set(scores))
    candidates = [values[0]] + [(a + b) / 2 for a, b in zip(values, values[1:], strict=False)] + [values[-1] + 1e-9]
    best: dict[str, float] | None = None
    for t in candidates:
        gated = [s < t for s in scores]
        sensitivity = sum(g and f for g, f in zip(gated, failed, strict=True)) / n_fail
        specificity = sum(not g and not f for g, f in zip(gated, failed, strict=True)) / n_ok
        j = sensitivity + specificity - 1
        if best is None or j > best["youden_j"]:
            best = {
                "threshold": t,
                "youden_j": j,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "rewrite_share": sum(gated) / len(gated),
            }
    assert best is not None
    return best


def auc_ci(scores: list[float], failed: list[bool], seed: int = 0) -> tuple[float, float, float]:
    """AUC of low confidence for failure, with a 95% bootstrap CI (questions resampled)."""
    negated = [-s for s in scores]
    rng = random.Random(seed)
    values = []
    for _ in range(RESAMPLES):
        idx = [rng.randrange(len(scores)) for _ in scores]
        value = auc([negated[i] for i in idx], [failed[i] for i in idx])
        if value == value:  # skip resamples without both classes (nan)
            values.append(value)
    values.sort()
    return auc(negated, failed), values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def measure(questions: list[dict], gold: dict[str, dict], backends: dict[str, Any], config: dict[str, Any]) -> dict:
    """Per retriever: the top-1 scores, which questions fail, AUC and the chosen threshold."""
    k, k_rrf = int(config["retrieve_k"]), int(config.get("rrf_k", 60))
    out = {}
    for retriever in RETRIEVERS:
        backend = make_retriever(retriever, bm25=backends["bm25"], dense=backends["dense"], k_rrf=k_rrf)
        scores, failed = [], []
        for question in questions:
            results = backend.retrieve(question["question"], k)
            score = top1_score(results)
            scores.append(float("-inf") if score is None else score)
            ids = [result["chunk_id"] for result in results]
            failed.append(compute_first_relevant_rank(ids, gold[question["qid"]]["gold_chunk_ids"]) != 1)
        out[retriever] = {"auc": auc_ci(scores, failed), "failed": sum(failed), **youden_threshold(scores, failed)}
    return out


def report_markdown(measured: dict, n: int) -> str:
    lines = [
        "# R5: when to rewrite (post-thesis result, not a thesis number)",
        "",
        f"Dev split ({n} questions, full corpus, original questions only). Confidence: the retriever's top-1 score; "
        "failure: no relevant chunk (question-level gold) at rank 1. Rewrite when the top-1 score is below the "
        "threshold (chosen by Youden's J).",
        "",
        "| retriever | failures | AUC [95% CI] | threshold | sensitivity | specificity | rewritten share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for retriever, m in measured.items():
        a = m["auc"]
        lines.append(
            f"| {retriever} | {m['failed']}/{n} | {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] | {m['threshold']:.6g} | "
            f"{m['sensitivity']:.3f} | {m['specificity']:.3f} | {m['rewrite_share']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dev-questions", type=Path, default=Path("data/dev/questions.jsonl"))
    parser.add_argument("--index-root", type=Path, default=Path("indexes/full"))
    parser.add_argument("--all-gold", type=Path, default=Path("data/full/eval/gold_evidence.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/rewrite_gate"))
    args = parser.parse_args()

    config = load_config(args.config)
    questions = load_questions(args.dev_questions)
    gold = question_level_gold(list(load_gold_evidence(args.all_gold).values()))
    backends = {
        "bm25": load_bm25(args.index_root / "bm25"),
        "dense": load_chroma_index(args.index_root / "chroma", query_cache_dir=config.get("query_embedding_cache_dir")),
    }
    measured = measure(questions, gold, backends, config)
    gate = {"signal": "top1_score", "thresholds": {r: m["threshold"] for r, m in measured.items()}}
    json_path, report_path = args.output_dir / "rewrite_gate.json", args.output_dir / "rewrite_gate.md"
    for path in (json_path, report_path):
        ensure_writable(path)  # never write into a protected folder (see protection.py)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    report = report_markdown(measured, len(questions))
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
