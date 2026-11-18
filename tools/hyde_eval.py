"""R8: does retrieving with a hypothetical passage (HyDE) help? Offline, once the passages exist.

For each corpus and retriever, three query sets per test question, retrieved at pipeline depth and
fused as in the pipeline: ``question`` (alone), ``hyde`` (the cached hypothetical passage alone)
and ``question+hyde``. Question-level MRR with 95% CIs that resample BioASQ questions and paired
differences against the question, on all test qids and on those whose passage does **not** contain
a gold answer the question lacks: a passage that already states the answer shows what the model
knew, not what retrieval found, so the second view is the fair one. No LLM calls; generate the
passages first with ``rag-thesis rewrite-batch --mode hyde --questions-path data/eval/questions.jsonl
--allow-llm-calls``.

Usage:
    python tools/hyde_eval.py --corpus subset indexes --corpus full indexes/full
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rag_thesis.config import load_config
from rag_thesis.evaluation.guard_metrics import answer_leaks
from rag_thesis.evaluation.retrieval import bioasq_question_id, evaluate_retriever, question_level_gold
from rag_thesis.evaluation.stats import bootstrap_ci, paired_bootstrap_ci
from rag_thesis.io import load_gold_evidence, load_questions
from rag_thesis.protection import ensure_writable
from rag_thesis.retrieval.bm25 import load_bm25
from rag_thesis.retrieval.chroma_store import load_chroma_index
from rag_thesis.retrieval.memo import MemoRetriever
from rag_thesis.rewrite.rewriter import QueryRewriter, load_cached_passages

RETRIEVERS = ("bm25", "dense", "hybrid")
SETS = ("question", "hyde", "question+hyde")


def query_sets(questions: list[dict], passages: dict[str, str]) -> dict[str, dict[str, list[str]]]:
    """question / hyde / question+hyde: qid -> queries."""
    out: dict[str, dict[str, list[str]]] = {name: {} for name in SETS}
    for question in questions:
        qid, text = question["qid"], question["question"].strip()
        out["question"][qid] = [text]
        out["hyde"][qid] = [passages[qid]]
        out["question+hyde"][qid] = [text, passages[qid]]
    return out


def leaking_qids(questions: list[dict], passages: dict[str, str]) -> set[str]:
    """Questions whose passage contains a gold answer the question doesn't."""
    return {
        q["qid"]
        for q in questions
        if answer_leaks(q["question"], passages[q["qid"]], [str(a["text"]) for a in q["answers"]])
    }


def mrr_by_set(
    questions: list[dict], gold: dict, passages: dict[str, str], backends: dict, config: dict, retriever: str
) -> dict[str, list[float]]:
    """Per query set: each question's question-level reciprocal rank."""
    out = {}
    for name, queries in query_sets(questions, passages).items():
        _, rows = evaluate_retriever(
            retriever=retriever,
            questions=questions,
            gold_by_qid=gold,
            config=config,
            eval_k_list=[int(config["retrieve_k"])],
            queries_by_qid=queries,
            **backends,
        )
        out[name] = [row["MRR"] for row in rows]
    return out


def _ci(v: tuple[float, float, float], signed: bool = False) -> str:
    fmt = "+.3f" if signed else ".3f"
    mark = " *" if signed and (v[1] > 0 or v[2] < 0) else ""
    return f"{v[0]:{fmt}} [{v[1]:{fmt}}, {v[2]:{fmt}}]{mark}"


def report_markdown(
    results: dict[tuple[str, str], dict[str, list[float]]], questions: list[dict], leaks: set[str]
) -> str:
    lines = [
        "# R8: HyDE, retrieving with a hypothetical passage (post-thesis result, not a thesis number)",
        "",
        f"{len(questions)} test qids, question-level MRR at pipeline depth; 95% CIs resample BioASQ questions; "
        f"* = the paired CI excludes 0. {len(leaks)} passages contain a gold answer their question doesn't "
        "(the model's own knowledge); the 'no leak' rows leave those questions out.",
        "",
        "| corpus | retriever | qids | question | hyde | question+hyde | hyde - question | question+hyde - question |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (corpus, retriever), mrr in results.items():
        for label, keep in (("all", None), ("no leak", leaks)):
            idx = [i for i, q in enumerate(questions) if keep is None or q["qid"] not in keep]
            if not idx:
                continue
            clusters = [bioasq_question_id(questions[i]["qid"]) for i in idx]
            v = {name: [values[i] for i in idx] for name, values in mrr.items()}
            lines.append(
                f"| {corpus} | {retriever} | {label} ({len(idx)}) | "
                f"{_ci(bootstrap_ci(v['question'], clusters=clusters))} | "
                f"{_ci(bootstrap_ci(v['hyde'], clusters=clusters))} | "
                f"{_ci(bootstrap_ci(v['question+hyde'], clusters=clusters))} | "
                f"{_ci(paired_bootstrap_ci(v['question'], v['hyde'], clusters=clusters), True)} | "
                f"{_ci(paired_bootstrap_ci(v['question'], v['question+hyde'], clusters=clusters), True)} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", nargs=2, action="append", required=True, metavar=("NAME", "INDEX_ROOT"))
    parser.add_argument("--hyde-cache", type=Path, default=None, help="default: the config's rewrite cache")
    parser.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    parser.add_argument("--all-gold", type=Path, default=Path("data/full/eval/gold_evidence.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/hyde"))
    args = parser.parse_args()

    config = load_config(args.config)
    questions = load_questions(args.questions)
    cache = args.hyde_cache or QueryRewriter(config=config).cache_path
    try:
        passages = load_cached_passages(cache, [q["qid"] for q in questions])
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(
            f"HyDE passages missing ({exc}). Generate them first: rag-thesis rewrite-batch --mode hyde "
            "--questions-path data/eval/questions.jsonl --allow-llm-calls"
        ) from exc
    widened = question_level_gold(list(load_gold_evidence(args.all_gold).values()))
    gold = {q["qid"]: widened[q["qid"]] for q in questions}
    results: dict[tuple[str, str], Any] = {}
    for corpus, root in args.corpus:
        backends = {
            "bm25_backend": MemoRetriever(load_bm25(Path(root) / "bm25")),
            "dense_backend": MemoRetriever(
                load_chroma_index(Path(root) / "chroma", query_cache_dir=config.get("query_embedding_cache_dir"))
            ),
        }
        for retriever in RETRIEVERS:
            results[(corpus, retriever)] = mrr_by_set(questions, gold, passages, backends, config, retriever)
    report_path = args.output_dir / "hyde_eval.md"
    ensure_writable(report_path)  # never write into a protected folder (see protection.py)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = report_markdown(results, questions, leaking_qids(questions, passages))
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
