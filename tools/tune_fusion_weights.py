"""R2: tune the original question's weight in the multi-query RRF on the dev split, report it on test.

Each question is retrieved with itself and its four cached rewrites (R1), fused with weighted RRF
(``fusion.merge_multi_results``): the question weighs ``w``, every rewrite 1.0. For each retriever,
``w`` is chosen on the dev split (tools/make_dev_split.py; full corpus, where its contexts are) by
question-level MRR at pipeline depth, from a small grid (ties go to the smaller weight). The chosen
weight is then reported on the 120 frozen test qids, on every corpus, against plain RRF (``w`` = 1)
and against the question alone, with paired 95% CIs that resample BioASQ questions. The test set
is never used for choosing. No LLM calls: the dev rewrites must be in the rewrite cache first
(``rag-thesis rewrite-batch --allow-llm-calls``). Writes tune_fusion_weights.md and
fusion_weights.json into ``--output-dir``.

Usage:
    python tools/tune_fusion_weights.py --dev-corpus indexes/full \\
        --corpus subset indexes --corpus full indexes/full --output-dir outputs/reports/fusion_weights
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
from rag_thesis.retrieval.memo import MemoRetriever
from rag_thesis.rewrite.rewriter import QueryRewriter, load_cached_rewrites

BENCHMARK = Path("outputs/final_benchmark/subset120_caseB_gem25flash_gem25flash_gem25pro_20260416")
GRID = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
RETRIEVERS = ("bm25", "dense", "hybrid")


def question_and_rewrites(questions: list[dict], rewrites: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    """qid -> the question followed by its rewrites, de-duplicated like the pipeline."""
    return {q["qid"]: unique_in_order([q["question"], *rewrites[q["qid"]].values()]) for q in questions}


def question_weights(questions: list[dict], weight: float) -> dict[str, dict[str, float]]:
    """qid -> {the question's text: ``weight``}; every other query keeps weight 1.0."""
    return {q["qid"]: {q["question"].strip(): weight} for q in questions}


def _backends(index_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "bm25_backend": MemoRetriever(load_bm25(index_root / "bm25")),
        "dense_backend": MemoRetriever(
            load_chroma_index(index_root / "chroma", query_cache_dir=config.get("query_embedding_cache_dir"))
        ),
    }


def _mrr_rows(retriever, questions, gold, queries, weights, backends, config) -> list[float]:
    _, rows = evaluate_retriever(
        retriever=retriever,
        questions=questions,
        gold_by_qid=gold,
        config=config,
        eval_k_list=[int(config["retrieve_k"])],
        queries_by_qid=queries,
        weights_by_qid=weights,
        **backends,
    )
    return [row["MRR"] for row in rows]


def tune(questions, gold, rewrites, backends, config, grid=GRID) -> dict[str, dict[str, Any]]:
    """Per retriever: the dev MRR of every grid weight and the chosen weight."""
    queries = question_and_rewrites(questions, rewrites)
    out = {}
    for retriever in RETRIEVERS:
        scores = {
            w: sum(_mrr_rows(retriever, questions, gold, queries, question_weights(questions, w), backends, config))
            / len(questions)
            for w in grid
        }
        best = max(grid, key=lambda w: (round(scores[w], 9), -w))  # ties: the smaller weight
        out[retriever] = {"scores": scores, "chosen": best}
    return out


def score_on_test(questions, gold, rewrites, backends, config, chosen: dict[str, float]) -> list[dict[str, Any]]:
    """Per retriever: question alone, plain RRF and the chosen weight, with paired CIs."""
    clusters = [bioasq_question_id(q["qid"]) for q in questions]
    with_rewrites = question_and_rewrites(questions, rewrites)
    alone = {q["qid"]: [q["question"]] for q in questions}
    rows = []
    for retriever in RETRIEVERS:
        base = _mrr_rows(retriever, questions, gold, alone, None, backends, config)
        plain = _mrr_rows(retriever, questions, gold, with_rewrites, None, backends, config)
        weighted = _mrr_rows(
            retriever, questions, gold, with_rewrites, question_weights(questions, chosen[retriever]), backends, config
        )
        rows.append(
            {
                "retriever": retriever,
                "weight": chosen[retriever],
                "question": bootstrap_ci(base, clusters=clusters),
                "plain": bootstrap_ci(plain, clusters=clusters),
                "weighted": bootstrap_ci(weighted, clusters=clusters),
                "weighted_vs_plain": paired_bootstrap_ci(plain, weighted, clusters=clusters),
                "weighted_vs_question": paired_bootstrap_ci(base, weighted, clusters=clusters),
            }
        )
    return rows


def _ci(v: tuple[float, float, float], signed: bool = False) -> str:
    fmt = "+.3f" if signed else ".3f"
    mark = " *" if signed and (v[1] > 0 or v[2] < 0) else ""
    return f"{v[0]:{fmt}} [{v[1]:{fmt}}, {v[2]:{fmt}}]{mark}"


def report_markdown(tuned: dict[str, dict], tests: dict[str, list[dict]], n_dev: int, n_test: int) -> str:
    lines = [
        "# R2: weighted fusion of the question and its rewrites (post-thesis result, not a thesis number)",
        "",
        f"Weight of the original question (each rewrite weighs 1.0), chosen per retriever on the dev split "
        f"({n_dev} qids, full corpus) by question-level MRR; reported on the {n_test} test qids with 95% CIs that "
        "resample BioASQ questions (* = the paired CI excludes 0).",
        "",
        "## Dev split: question-level MRR per weight",
        "",
        "| retriever | " + " | ".join(f"w={w:g}" for w in GRID) + " | chosen |",
        "| --- | " + " | ".join("---:" for _ in GRID) + " | ---: |",
    ]
    for retriever, result in tuned.items():
        cells = [f"{result['scores'][w]:.4f}" for w in GRID]
        lines.append(f"| {retriever} | " + " | ".join(cells) + f" | {result['chosen']:g} |")
    for corpus, rows in tests.items():
        lines += [
            "",
            f"## Test qids, {corpus} corpus: question-level MRR",
            "",
            "| retriever | w | question alone | question + rewrites, plain RRF | weighted | weighted - plain | "
            "weighted - question |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                f"| {row['retriever']} | {row['weight']:g} | {_ci(row['question'])} | {_ci(row['plain'])} | "
                f"{_ci(row['weighted'])} | {_ci(row['weighted_vs_plain'], True)} | "
                f"{_ci(row['weighted_vs_question'], True)} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dev-questions", type=Path, default=Path("data/dev/questions.jsonl"))
    parser.add_argument("--dev-corpus", type=Path, default=Path("indexes/full"), help="index folder for dev")
    parser.add_argument("--dev-rewrite-cache", type=Path, default=None, help="default: the config's rewrite cache")
    parser.add_argument("--corpus", nargs=2, action="append", required=True, metavar=("NAME", "INDEX_ROOT"))
    parser.add_argument("--test-questions", type=Path, default=Path("data/eval/questions.jsonl"))
    parser.add_argument("--test-rewrite-cache", type=Path, default=BENCHMARK / "cache" / "rewrites.jsonl")
    parser.add_argument("--all-gold", type=Path, default=Path("data/full/eval/gold_evidence.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/fusion_weights"))
    args = parser.parse_args()

    config = load_config(args.config)
    widened = question_level_gold(list(load_gold_evidence(args.all_gold).values()))
    dev = load_questions(args.dev_questions)
    dev_cache = args.dev_rewrite_cache or QueryRewriter(config=config).cache_path
    try:
        dev_rewrites = load_cached_rewrites(dev_cache, [q["qid"] for q in dev])
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(
            f"Dev rewrites missing ({exc}). Generate them first: rag-thesis rewrite-batch --allow-llm-calls"
        ) from exc
    dev_gold = {q["qid"]: widened[q["qid"]] for q in dev}
    tuned = tune(dev, dev_gold, dev_rewrites, _backends(args.dev_corpus, config), config)
    chosen = {retriever: result["chosen"] for retriever, result in tuned.items()}

    test = load_questions(args.test_questions)
    test_gold = {q["qid"]: widened[q["qid"]] for q in test}
    test_rewrites = load_cached_rewrites(args.test_rewrite_cache, [q["qid"] for q in test])
    tests = {
        name: score_on_test(test, test_gold, test_rewrites, _backends(Path(root), config), config, chosen)
        for name, root in args.corpus
    }
    weights_path = args.output_dir / "fusion_weights.json"
    report_path = args.output_dir / "tune_fusion_weights.md"
    for path in (weights_path, report_path):
        ensure_writable(path)  # never write into a protected folder (see protection.py)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path.write_text(json.dumps({"question_weight": chosen}, indent=2) + "\n", encoding="utf-8")
    report = report_markdown(tuned, tests, len(dev), len(test))
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
