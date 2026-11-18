"""Which query sets help retrieval? Ablations over the cached multi-query rewrites, offline.

For every evaluation qid the official rewrite cache holds four rewrites, one per type
(``bm25_lexical``, ``dense_semantic``, ``abbreviation_expanded``, ``minimal_entity_locked``). Each
query set below is retrieved at pipeline depth and fused with RRF exactly as in the pipeline
(``evaluate_retriever``), on every corpus, and scored against strict and question-level gold with
95% CIs that resample BioASQ questions:

  question             the question alone (no rewriting)
  rewrites             the four rewrites, guard off (the thesis multi-query condition)
  question+rewrites    R1: the question as well (RAG-Fusion style)
  guard-on             the post-guard queries the official run logged (LLM repairs included)
  question+guard-on    R1 with the guard on
  without-<type>       R7: leave one rewrite type out of ``rewrites``
  only-<type>          one rewrite type alone
  routed               R3, hybrid only: each rewrite to the retriever its type was written for
                       (``bm25_lexical`` -> BM25, ``dense_semantic`` -> dense, the others -> hybrid)
  question+routed      R3 with the question (to hybrid) as well
  dictionary           R4: the question with its abbreviations expanded from the corpus
                       (``guard.aliases``, no LLM): "DAT1" -> "DAT1 (dopamine transporter 1)"
  question+dictionary  R4 with the plain question as well
  structured           R6 (with ``--structured-cache``): the four ``structured``-mode rewrites, guard off
  structured-guarded   R6: only the structured rewrites the structured guard (``guard.structured``) lets
                       through (the question if none)
  question+structured-guarded   R6 with the question as well

Paired differences in question-level MRR: ``without-<type>`` against ``rewrites`` (a type's marginal
contribution, positive = the set is better without it), ``routed`` / ``question+routed`` against
``rewrites`` / ``question+rewrites`` (routing against plain hybrid), every other set against ``question``.
No LLM calls. Writes rewrite_ablation.md and rewrite_ablation_metrics.csv into ``--output-dir``.

Usage:
    python tools/rewrite_ablation.py --corpus subset indexes --corpus full indexes/full \\
        --output-dir outputs/reports/rewrite_ablation
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from rag_thesis.config import load_config
from rag_thesis.evaluation.guard_metrics import answer_leaks
from rag_thesis.evaluation.retrieval import (
    bioasq_question_id,
    compute_first_relevant_rank,
    compute_hit_at_k,
    evaluate_retriever,
    load_logged_queries,
    question_level_gold,
    write_csv,
)
from rag_thesis.evaluation.stats import bootstrap_ci, paired_bootstrap_ci
from rag_thesis.guard.aliases import AbbreviationDictionary
from rag_thesis.guard.checks import is_blocked
from rag_thesis.guard.structured import assess_structured
from rag_thesis.io import load_gold_evidence, load_questions, read_jsonl
from rag_thesis.pipeline.fusion import merge_multi_results, normalize_results, unique_in_order
from rag_thesis.protection import ensure_writable
from rag_thesis.retrieval.bm25 import load_bm25
from rag_thesis.retrieval.chroma_store import load_chroma_index
from rag_thesis.retrieval.factory import make_retriever
from rag_thesis.retrieval.memo import MemoRetriever
from rag_thesis.rewrite.rewriter import (
    MULTI_REWRITE_ORDER,
    StructuredRewrite,
    load_cached_rewrites,
    load_cached_structures,
)
from rag_thesis.schemas import DocumentLine

BENCHMARK = Path("outputs/final_benchmark/subset120_caseB_gem25flash_gem25flash_gem25pro_20260416")
RETRIEVERS = ("bm25", "dense", "hybrid")
ROUTES = {"bm25_lexical": "bm25", "dense_semantic": "dense"}  # R3; every other rewrite type goes to hybrid
REFERENCES = {"routed": "rewrites", "question+routed": "question+rewrites"}


def structured_sets(
    questions: list[dict[str, Any]],
    structures: dict[str, StructuredRewrite],
    aliases: AbbreviationDictionary,
    multi_rewrites: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    """R6: qid -> {"all": the four structured rewrites, "guarded": those the structured guard lets
    through}, and report notes (blocked rewrites per check, invented structure fields, answer leaks
    in the structured and in the multi rewrites)."""
    sets: dict[str, dict[str, list[str]]] = {}
    blocked_by: Counter[str] = Counter()
    invented = blocked = leaks_structured = leaks_multi = 0
    for question in questions:
        qid, text = question["qid"], question["question"]
        answers = [str(answer["text"]) for answer in question["answers"]]
        structure = structures[qid]
        structure_violations, per_rewrite = assess_structured(text, structure, aliases)
        invented += len(structure_violations)
        rewrites = structure.rewrite_list()
        kept = [
            rewrite for rewrite, violations in zip(rewrites, per_rewrite, strict=True) if not is_blocked(violations)
        ]
        blocked += len(rewrites) - len(kept)
        for violations in per_rewrite:  # each blocked rewrite counts once per check that blocks it
            blocked_by.update({v.check for v in violations if is_blocked([v])})
        leaks_structured += sum(bool(answer_leaks(text, rewrite, answers)) for rewrite in rewrites)
        leaks_multi += sum(bool(answer_leaks(text, rewrite, answers)) for rewrite in multi_rewrites[qid].values())
        sets[qid] = {"all": rewrites, "guarded": kept}
    total = 4 * len(questions)
    notes = [
        "",
        f"R6: the structured guard blocks {blocked} of {total} structured rewrites (by check: "
        + (", ".join(f"{name} {count}" for name, count in blocked_by.most_common()) or "none")
        + f"); {invented} structure fields are not in their question; rewrites containing a gold answer the "
        f"question doesn't: {leaks_structured} of {total} structured vs {leaks_multi} of {total} multi.",
    ]
    return sets, notes


def query_sets(
    questions: list[dict[str, Any]],
    rewrites: dict[str, dict[str, str]],
    guard_on: dict[str, list[str]],
    expanded: dict[str, str] | None = None,
    structured: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Variant name -> qid -> its queries, de-duplicated in order like the pipeline's retrieve step."""
    variants: dict[str, dict[str, list[str]]] = {}
    for question in questions:
        qid, text = question["qid"], question["question"]
        by_type = rewrites[qid]
        sets = {
            "question": [text],
            "rewrites": list(by_type.values()),
            "question+rewrites": [text, *by_type.values()],
            "guard-on": guard_on[qid],
            "question+guard-on": [text, *guard_on[qid]],
        }
        for slot in MULTI_REWRITE_ORDER:
            sets[f"without-{slot}"] = [rewrite for other, rewrite in by_type.items() if other != slot]
        for slot in MULTI_REWRITE_ORDER:
            sets[f"only-{slot}"] = [by_type[slot]]
        if expanded is not None:
            sets["dictionary"] = [expanded[qid]]
            sets["question+dictionary"] = [text, expanded[qid]]
        if structured is not None:
            sets["structured"] = structured[qid]["all"]
            sets["structured-guarded"] = structured[qid]["guarded"]
            sets["question+structured-guarded"] = [text, *structured[qid]["guarded"]]
        for name, queries in sets.items():
            variants.setdefault(name, {})[qid] = unique_in_order(queries) or [text]
    return variants


def routed_rows(
    questions: list[dict[str, Any]],
    golds: dict[str, dict[str, dict]],
    rewrites: dict[str, dict[str, str]],
    backends: dict[str, Any],
    k: int,
    k_rrf: int,
    with_question: bool,
) -> dict[str, list[dict[str, Any]]]:
    """R3 for hybrid: each rewrite retrieved by the retriever its type targets, all lists fused with
    RRF as in the pipeline. Gold name -> per-question rows (``MRR``, ``Hit@k``)."""
    retrievers = {
        mode: make_retriever(mode, bm25=backends["bm25_backend"], dense=backends["dense_backend"], k_rrf=k_rrf)
        for mode in RETRIEVERS
    }
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in golds}
    for question in questions:
        qid = question["qid"]
        pairs = [("hybrid", question["question"])] if with_question else []
        pairs += [(ROUTES.get(slot, "hybrid"), rewrite) for slot, rewrite in rewrites[qid].items()]
        queries = unique_in_order([query for _, query in pairs])
        mode_of = {query.strip(): mode for mode, query in reversed(pairs)}  # first occurrence wins
        per_query = [
            (query, normalize_results(retrievers[mode_of[query]].retrieve(query, k), mode_of[query], query))
            for query in queries
        ]
        fused = merge_multi_results(per_query, out_k=k, k_rrf=k_rrf) if len(per_query) > 1 else per_query[0][1][:k]
        retrieved = [result["chunk_id"] for result in fused]
        for name, gold in golds.items():
            first = compute_first_relevant_rank(retrieved, list(gold[qid]["gold_chunk_ids"]))
            rows[name].append(
                {"qid": qid, "MRR": 0.0 if first is None else 1.0 / first, f"Hit@{k}": compute_hit_at_k(first, k)}
            )
    return rows


def evaluate_variants(
    corpora: list[tuple[str, Path]],
    questions: list[dict[str, Any]],
    golds: dict[str, dict[str, dict]],
    rewrites: dict[str, dict[str, str]],
    runs_dir: Path,
    config: dict[str, Any],
    expanded: dict[str, str] | None = None,
    structured: dict[str, dict[str, list[str]]] | None = None,
) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    """(corpus, retriever, variant, gold) -> per-question rows at pipeline depth."""
    k = int(config["retrieve_k"])
    rows: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for corpus, index_root in corpora:
        backends = {
            "bm25_backend": MemoRetriever(load_bm25(index_root / "bm25")),
            "dense_backend": MemoRetriever(
                load_chroma_index(index_root / "chroma", query_cache_dir=config.get("query_embedding_cache_dir"))
            ),
        }
        for retriever in RETRIEVERS:
            guard_on = load_logged_queries(runs_dir / f"{retriever}_multi_guard_on.jsonl")
            for variant, queries_by_qid in query_sets(questions, rewrites, guard_on, expanded, structured).items():
                for gold_name, gold in golds.items():
                    _, rows[(corpus, retriever, variant, gold_name)] = evaluate_retriever(
                        retriever=retriever,
                        questions=questions,
                        gold_by_qid=gold,
                        config=config,
                        eval_k_list=[k],
                        queries_by_qid=queries_by_qid,
                        **backends,
                    )
        k_rrf = int(config.get("rrf_k", 60))
        for variant, with_question in (("routed", False), ("question+routed", True)):
            for gold_name, gold_rows in routed_rows(
                questions, golds, rewrites, backends, k, k_rrf, with_question
            ).items():
                rows[(corpus, "hybrid", variant, gold_name)] = gold_rows
    return rows


def _values(rows: dict, corpus: str, retriever: str, variant: str, metric: str, gold: str = "question") -> list[float]:
    return [row[metric] for row in rows[(corpus, retriever, variant, gold)]]


def summarize(rows: dict[tuple[str, str, str, str], list[dict]], clusters: list[str], k: int) -> list[dict[str, Any]]:
    """One row per corpus x retriever x variant: means with CIs and the paired difference."""
    out = []
    for corpus, retriever, variant, gold in rows:
        if gold != "question":
            continue
        reference = "rewrites" if variant.startswith("without-") else REFERENCES.get(variant, "question")

        row = {"corpus": corpus, "retriever": retriever, "variant": variant, "reference": reference}
        row["question_mrr"] = bootstrap_ci(_values(rows, corpus, retriever, variant, "MRR"), clusters=clusters)
        row["question_hit"] = bootstrap_ci(_values(rows, corpus, retriever, variant, f"Hit@{k}"), clusters=clusters)
        row["strict_mrr"] = bootstrap_ci(_values(rows, corpus, retriever, variant, "MRR", "strict"), clusters=clusters)
        row["diff_question_mrr"] = (
            None
            if variant == reference
            else paired_bootstrap_ci(
                _values(rows, corpus, retriever, reference, "MRR"),
                _values(rows, corpus, retriever, variant, "MRR"),
                clusters=clusters,
            )
        )
        out.append(row)
    return out


def _ci(values: tuple[float, float, float] | None, signed: bool = False) -> str:
    if values is None:
        return "(reference)"
    fmt = "+.3f" if signed else ".3f"
    mean, low, high = values
    mark = " *" if signed and (low > 0 or high < 0) else ""
    return f"{mean:{fmt}} [{low:{fmt}}, {high:{fmt}}]{mark}"


def report_markdown(
    summary: list[dict[str, Any]], n_qids: int, n_questions: int, k: int, notes: list[str] | None = None
) -> str:
    lines = [
        "# Multi-query rewrite ablation (post-thesis result, not a thesis number)",
        "",
        f"{n_qids} frozen evaluation qids ({n_questions} BioASQ questions); cached official rewrites; pipeline "
        f"depth k={k}; queries fused with RRF as in the pipeline. Question-level gold counts every snippet of the "
        "same BioASQ question. 95% CIs resample BioASQ questions; * = the paired CI excludes 0. "
        "diff: `without-<type>` against `rewrites` (positive = better without that type), everything else against "
        "`question`.",
        *(notes or []),
    ]
    for corpus in dict.fromkeys(row["corpus"] for row in summary):
        for retriever in RETRIEVERS:
            lines += [
                "",
                f"## {corpus} / {retriever}",
                "",
                f"| query set | question MRR | question Hit@{k} | strict MRR | diff question MRR |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
            for row in summary:
                if row["corpus"] == corpus and row["retriever"] == retriever:
                    lines.append(
                        f"| {row['variant']} | {_ci(row['question_mrr'])} | {_ci(row['question_hit'])} | "
                        f"{_ci(row['strict_mrr'])} | {_ci(row['diff_question_mrr'], signed=True)} |"
                    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", nargs=2, action="append", required=True, metavar=("NAME", "INDEX_ROOT"))
    parser.add_argument("--rewrite-cache", type=Path, default=BENCHMARK / "cache" / "rewrites.jsonl")
    parser.add_argument("--runs-dir", type=Path, default=BENCHMARK / "subset120" / "runs")
    parser.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    parser.add_argument("--gold", type=Path, default=Path("data/eval/gold_evidence.jsonl"))
    parser.add_argument("--all-gold", type=Path, default=Path("data/full/eval/gold_evidence.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--documents", type=Path, default=Path("data/full/processed/documents.jsonl"), help="for the R4 dictionary"
    )
    parser.add_argument(
        "--structured-cache",
        type=Path,
        default=None,
        help="rewrite cache holding structured-mode rewrites of the test qids (R6); adds the structured query sets",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/rewrite_ablation"))
    args = parser.parse_args()

    config = load_config(args.config)
    k = int(config["retrieve_k"])
    questions = load_questions(args.questions)
    qids = [q["qid"] for q in questions]
    widened = question_level_gold(list(load_gold_evidence(args.all_gold).values()))
    golds = {"strict": load_gold_evidence(args.gold), "question": {qid: widened[qid] for qid in qids}}
    rewrites = load_cached_rewrites(args.rewrite_cache, qids)
    corpora = [(name, Path(root)) for name, root in args.corpus]
    dictionary = AbbreviationDictionary.from_texts(row["text"] for row in read_jsonl(args.documents, DocumentLine))
    expanded = {q["qid"]: dictionary.expand(q["question"]) for q in questions}
    changed = [q for q in questions if expanded[q["qid"]] != q["question"]]
    leaking = [
        q for q in changed if answer_leaks(q["question"], expanded[q["qid"]], [str(a["text"]) for a in q["answers"]])
    ]
    notes = [
        "",
        f"R4: the corpus dictionary ({len(dictionary)} short forms) expands an abbreviation in {len(changed)} of "
        f"{len(questions)} qids; {len(leaking)} of the expanded questions contain a gold answer the question doesn't.",
    ]
    structured = None
    if args.structured_cache is not None:
        try:
            structures = load_cached_structures(args.structured_cache, qids)
        except (ValueError, FileNotFoundError) as exc:
            raise SystemExit(
                f"Structured rewrites missing ({exc}). Generate them first: rag-thesis rewrite-batch --mode structured "
                "--questions-path data/eval/questions.jsonl --allow-llm-calls"
            ) from exc
        structured, structured_notes = structured_sets(questions, structures, dictionary, rewrites)
        notes += structured_notes
    rows = evaluate_variants(corpora, questions, golds, rewrites, args.runs_dir, config, expanded, structured)
    clusters = [bioasq_question_id(qid) for qid in qids]
    summary = summarize(rows, clusters, k)

    flat = [
        {
            **{key: row[key] for key in ("corpus", "retriever", "variant", "reference")},
            **{
                f"{name}_{part}": value
                for name in ("question_mrr", "question_hit", "strict_mrr")
                for part, value in zip(("mean", "low", "high"), row[name], strict=True)
            },
        }
        for row in summary
    ]
    write_csv(args.output_dir / "rewrite_ablation_metrics.csv", flat, list(flat[0]))
    report_path = args.output_dir / "rewrite_ablation.md"
    ensure_writable(report_path)  # never write into a protected folder (see protection.py)
    report = report_markdown(summary, len(qids), len(set(clusters)), k, notes)
    report_path.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
