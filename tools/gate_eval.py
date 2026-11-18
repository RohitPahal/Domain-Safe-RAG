"""R5: does rewriting only when unsure pay off? Never vs always vs gated, on the test qids, offline.

For each corpus and retriever, three query sets per test question, retrieved at pipeline depth and
fused as in the pipeline: ``never`` (the question alone), ``always`` (the question and its four
cached rewrites) and ``gated`` (the question alone when its top-1 score reaches the retriever's
threshold from tools/gate_threshold.py, else the question and its rewrites, as the pipeline's gate
does). Question-level MRR and Hit@k with 95% CIs that resample BioASQ questions, paired
differences, and the share of questions the gate rewrites (the rewrite calls it would spend).
The thresholds were chosen on the full corpus; BM25's score scale depends on the corpus, so its
subset numbers use a threshold from another scale. No LLM calls.

Usage:
    python tools/gate_eval.py --corpus subset indexes --corpus full indexes/full \\
        --gate outputs/reports/rewrite_gate/rewrite_gate.json --output-dir outputs/reports/rewrite_gate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag_thesis.config import load_config
from rag_thesis.evaluation.retrieval import bioasq_question_id, evaluate_retriever, question_level_gold
from rag_thesis.evaluation.stats import bootstrap_ci, paired_bootstrap_ci
from rag_thesis.io import load_gold_evidence, load_questions
from rag_thesis.pipeline.fusion import unique_in_order
from rag_thesis.protection import ensure_writable
from rag_thesis.retrieval.bm25 import load_bm25
from rag_thesis.retrieval.chroma_store import load_chroma_index
from rag_thesis.retrieval.confidence import is_confident
from rag_thesis.retrieval.factory import make_retriever
from rag_thesis.retrieval.memo import MemoRetriever
from rag_thesis.rewrite.rewriter import load_cached_rewrites

BENCHMARK = Path("outputs/final_benchmark/subset120_caseB_gem25flash_gem25flash_gem25pro_20260416")
RETRIEVERS = ("bm25", "dense", "hybrid")


def strategies(
    questions: list[dict], rewrites: dict[str, dict[str, str]], retriever: Any, k: int, threshold: float
) -> dict[str, dict[str, list[str]]]:
    """never / always / gated: qid -> queries. ``retriever`` decides the gate on the question alone."""
    out: dict[str, dict[str, list[str]]] = {"never": {}, "always": {}, "gated": {}}
    for question in questions:
        qid, text = question["qid"], question["question"].strip()
        both = unique_in_order([text, *rewrites[qid].values()])
        out["never"][qid] = [text]
        out["always"][qid] = both
        out["gated"][qid] = [text] if is_confident(retriever.retrieve(text, k), threshold) else both
    return out


def evaluate(
    questions: list[dict],
    gold: dict[str, dict],
    rewrites: dict,
    backends: dict,
    config: dict,
    threshold: float,
    retriever: str,
) -> dict[str, Any]:
    """Per strategy: MRR and Hit@k rows; plus the gate's rewrite share."""
    k, k_rrf = int(config["retrieve_k"]), int(config.get("rrf_k", 60))
    deciding = make_retriever(retriever, bm25=backends["bm25_backend"], dense=backends["dense_backend"], k_rrf=k_rrf)
    sets = strategies(questions, rewrites, deciding, k, threshold)
    rows = {}
    for name, queries in sets.items():
        _, rows[name] = evaluate_retriever(
            retriever=retriever,
            questions=questions,
            gold_by_qid=gold,
            config=config,
            eval_k_list=[k],
            queries_by_qid=queries,
            **backends,
        )
    rewritten = sum(len(sets["gated"][q["qid"]]) > 1 for q in questions) / len(questions)
    return {"rows": rows, "rewrite_share": rewritten}


def _ci(v: tuple[float, float, float], signed: bool = False) -> str:
    fmt = "+.3f" if signed else ".3f"
    mark = " *" if signed and (v[1] > 0 or v[2] < 0) else ""
    return f"{v[0]:{fmt}} [{v[1]:{fmt}}, {v[2]:{fmt}}]{mark}"


def report_markdown(results: dict[tuple[str, str], dict], questions: list[dict], k: int, gate: dict) -> str:
    clusters = [bioasq_question_id(q["qid"]) for q in questions]
    lines = [
        "# R5: rewrite only when unsure (post-thesis result, not a thesis number)",
        "",
        f"{len(questions)} test qids, cached official rewrites, pipeline depth k={k}, question-level gold; "
        f"thresholds (top-1 score, chosen on the dev split): {json.dumps(gate['thresholds'])}. 95% CIs resample "
        "BioASQ questions; * = the paired CI excludes 0. BM25's threshold comes from the full corpus' score scale.",
        "",
        "| corpus | retriever | never MRR | always MRR | gated MRR | gated - never | gated - always | rewritten |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (corpus, retriever), result in results.items():
        mrr = {name: [row["MRR"] for row in rows] for name, rows in result["rows"].items()}
        lines.append(
            f"| {corpus} | {retriever} | {_ci(bootstrap_ci(mrr['never'], clusters=clusters))} | "
            f"{_ci(bootstrap_ci(mrr['always'], clusters=clusters))} | "
            f"{_ci(bootstrap_ci(mrr['gated'], clusters=clusters))} | "
            f"{_ci(paired_bootstrap_ci(mrr['never'], mrr['gated'], clusters=clusters), True)} | "
            f"{_ci(paired_bootstrap_ci(mrr['always'], mrr['gated'], clusters=clusters), True)} | "
            f"{result['rewrite_share']:.0%} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", nargs=2, action="append", required=True, metavar=("NAME", "INDEX_ROOT"))
    parser.add_argument("--gate", type=Path, default=Path("outputs/reports/rewrite_gate/rewrite_gate.json"))
    parser.add_argument("--rewrite-cache", type=Path, default=BENCHMARK / "cache" / "rewrites.jsonl")
    parser.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    parser.add_argument("--all-gold", type=Path, default=Path("data/full/eval/gold_evidence.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/rewrite_gate"))
    args = parser.parse_args()

    config = load_config(args.config)
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    questions = load_questions(args.questions)
    widened = question_level_gold(list(load_gold_evidence(args.all_gold).values()))
    gold = {q["qid"]: widened[q["qid"]] for q in questions}
    rewrites = load_cached_rewrites(args.rewrite_cache, [q["qid"] for q in questions])
    results = {}
    for corpus, root in args.corpus:
        backends = {
            "bm25_backend": MemoRetriever(load_bm25(Path(root) / "bm25")),
            "dense_backend": MemoRetriever(
                load_chroma_index(Path(root) / "chroma", query_cache_dir=config.get("query_embedding_cache_dir"))
            ),
        }
        for retriever in RETRIEVERS:
            threshold = float(gate["thresholds"][retriever])
            results[(corpus, retriever)] = evaluate(questions, gold, rewrites, backends, config, threshold, retriever)
    report_path = args.output_dir / "gate_eval.md"
    ensure_writable(report_path)  # never write into a protected folder (see protection.py)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = report_markdown(results, questions, int(config["retrieve_k"]), gate)
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
