"""Retrieval-only evaluation: Recall@k, MRR and nDCG@10 of each retriever against the gold chunks.

Strict relevance: only a question's gold chunks (from its own context) count. In a corpus that
holds other snippets for the same question, ``question_level_gold`` widens each qid's gold to every
QA pair of the same BioASQ question (the same question asked with other snippets). ``Hit@k`` (any
relevant chunk in the top-k) is reported for whichever gold is passed in.

By default every question is retrieved with its own text. With a run log
(``queries_log_path``), each question is retrieved with the queries that run used (its
post-guard rewrites), fused exactly as the pipeline fuses them, so any logged rewrite/guard
condition can be re-evaluated offline, e.g. on another corpus.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from rag_thesis.config import load_config
from rag_thesis.io import load_gold_evidence, load_questions, read_jsonl
from rag_thesis.pipeline.fusion import retrieve_queries, unique_in_order
from rag_thesis.protection import ensure_writable
from rag_thesis.retrieval.bm25 import load_bm25
from rag_thesis.retrieval.chroma_store import load_chroma_index
from rag_thesis.retrieval.factory import make_retriever
from rag_thesis.schemas import RunLogLine

VALID_RETRIEVERS = ("bm25", "dense", "hybrid", "all")
DETAILS_FILENAME = "retrieval_details.csv"
METRICS_FILENAME = "retrieval_metrics.csv"
SUMMARY_FILENAME = "summary.md"
NDCG_K = 10
NDCG_COLUMN = f"nDCG@{NDCG_K}"


def resolve_eval_k_list(config: dict[str, Any], override: list[int] | None = None) -> list[int]:
    raw_values = override if override is not None else config.get("eval_k_list", [5, 10, 20])
    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError("eval_k_list must be a non-empty list of positive integers.")
    normalized = sorted({int(value) for value in raw_values})
    if any(value <= 0 for value in normalized):
        raise ValueError("eval_k_list must contain only positive integers.")
    return normalized


def compute_recall_at_k(retrieved_chunk_ids: list[str], gold_chunk_ids: list[str], k: int) -> float:
    if not gold_chunk_ids:
        return 0.0
    hits = len(set(retrieved_chunk_ids[:k]).intersection(gold_chunk_ids))
    return hits / float(len(set(gold_chunk_ids)))


def compute_ndcg_at_k(retrieved_chunk_ids: list[str], gold_chunk_ids: list[str], k: int) -> float:
    """Binary-relevance nDCG@k: each gold chunk counts 1 (once), every other chunk 0. The ideal
    ranking puts all gold chunks first, so finding every gold chunk at the top scores 1.0."""
    gold_set = set(gold_chunk_ids)
    if not gold_set:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved_chunk_ids[:k], start=1)
        if chunk_id in gold_set
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold_set), k) + 1))
    return dcg / ideal


def bioasq_question_id(qid: str) -> str:
    """The BioASQ question of a qid. QA pairs are ``<question id>_<snippet number>``: one BioASQ
    question is asked once per snippet, so several qids (and contexts) share a question id."""
    return qid.rsplit("_", 1)[0]


def question_level_gold(gold_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """qid -> its gold record with ``gold_chunk_ids`` widened to the gold chunks of every QA pair of
    the same BioASQ question (own chunks first, then the others in file order, no repeats)."""
    by_question: dict[str, list[str]] = {}
    for row in gold_rows:
        chunk_ids = by_question.setdefault(bioasq_question_id(str(row["qid"])), [])
        chunk_ids.extend(chunk_id for chunk_id in row["gold_chunk_ids"] if chunk_id not in chunk_ids)
    widened: dict[str, dict[str, Any]] = {}
    for row in gold_rows:
        own = list(row["gold_chunk_ids"])
        siblings = [chunk_id for chunk_id in by_question[bioasq_question_id(str(row["qid"]))] if chunk_id not in own]
        widened[str(row["qid"])] = {**row, "gold_chunk_ids": own + siblings}
    return widened


def compute_hit_at_k(first_relevant_rank: int | None, k: int) -> float:
    """1.0 if a relevant chunk is in the top-k (success@k), else 0.0."""
    return 1.0 if first_relevant_rank is not None and first_relevant_rank <= k else 0.0


def compute_first_relevant_rank(retrieved_chunk_ids: list[str], gold_chunk_ids: list[str]) -> int | None:
    gold_set = set(gold_chunk_ids)
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in gold_set:
            return rank
    return None


def load_logged_queries(run_log_path: Path) -> dict[str, list[str]]:
    """qid -> the queries a run retrieved that question with (its logged post-guard rewrites)."""
    queries_by_qid: dict[str, list[str]] = {}
    for row in read_jsonl(run_log_path, RunLogLine):
        qid = str(row["qid"])
        if qid in queries_by_qid:
            raise ValueError(f"qid {qid} appears more than once in run log {run_log_path}.")
        # The same rule as the pipeline's retrieve step: rewrites de-duplicated, else the question.
        queries_by_qid[qid] = unique_in_order(row["query_rewrites"]) or [str(row["query_original"])]
    return queries_by_qid


def evaluate_retriever(
    *,
    retriever: str,
    questions: list[dict[str, Any]],
    gold_by_qid: dict[str, dict[str, Any]],
    config: dict[str, Any],
    eval_k_list: list[int],
    bm25_backend: Any | None = None,
    dense_backend: Any | None = None,
    queries_by_qid: dict[str, list[str]] | None = None,
    weights_by_qid: dict[str, dict[str, float]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Retrieve the top ``max(eval_k_list)`` per question and score it against the gold chunks.

    A question is retrieved with its own text, or with ``queries_by_qid[qid]`` if given (each
    query to the same depth, several fused with RRF, as in the pipeline; ``weights_by_qid[qid]``
    weights that fusion per query text).
    """
    max_k = max(eval_k_list)
    k_rrf = int(config.get("rrf_k", 60))
    backend = make_retriever(retriever, bm25=bm25_backend, dense=dense_backend, k_rrf=k_rrf)
    detail_rows: list[dict[str, Any]] = []
    recall_sums = {k: 0.0 for k in eval_k_list}
    hit_sums = {k: 0.0 for k in eval_k_list}
    mrr_sum = 0.0
    ndcg_sum = 0.0

    for question in questions:
        qid = str(question["qid"])
        gold = gold_by_qid.get(qid)
        if gold is None:
            raise ValueError(f"Gold evidence not found for qid: {qid}")

        if queries_by_qid is None:
            queries = [str(question["question"])]
        elif qid in queries_by_qid:
            queries = queries_by_qid[qid]
        else:
            raise ValueError(f"No logged queries for qid: {qid}")
        weights = (weights_by_qid or {}).get(qid)
        results = retrieve_queries(backend, queries, max_k, retriever_mode=retriever, k_rrf=k_rrf, weights=weights)
        retrieved_chunk_ids = [str(result["chunk_id"]) for result in results]
        gold_chunk_ids = [str(chunk_id) for chunk_id in gold["gold_chunk_ids"]]
        gold_set = set(gold_chunk_ids)
        relevant_chunk_ids = [chunk_id for chunk_id in retrieved_chunk_ids if chunk_id in gold_set]
        first_relevant_rank = compute_first_relevant_rank(retrieved_chunk_ids, gold_chunk_ids)
        reciprocal_rank = 0.0 if first_relevant_rank is None else 1.0 / float(first_relevant_rank)
        mrr_sum += reciprocal_rank
        ndcg = compute_ndcg_at_k(retrieved_chunk_ids, gold_chunk_ids, NDCG_K)
        ndcg_sum += ndcg

        row: dict[str, Any] = {
            "qid": qid,
            "question": str(question["question"]),
            "queries": json.dumps(queries, ensure_ascii=False),
            "retriever": retriever,
            "gold_doc_id": str(gold["gold_doc_id"]),
            "gold_chunk_ids": json.dumps(gold_chunk_ids, ensure_ascii=False),
            "retrieved_chunk_ids": json.dumps(retrieved_chunk_ids, ensure_ascii=False),
            "relevant_chunk_ids": json.dumps(relevant_chunk_ids, ensure_ascii=False),
            "first_relevant_rank": "" if first_relevant_rank is None else first_relevant_rank,
            "MRR": round(reciprocal_rank, 6),
            NDCG_COLUMN: round(ndcg, 6),
        }
        for k in eval_k_list:
            recall_value = compute_recall_at_k(retrieved_chunk_ids, gold_chunk_ids, k)
            recall_sums[k] += recall_value
            row[f"Recall@{k}"] = round(recall_value, 6)
            hit_sums[k] += compute_hit_at_k(first_relevant_rank, k)
            row[f"Hit@{k}"] = compute_hit_at_k(first_relevant_rank, k)
        detail_rows.append(row)

    num_questions = len(questions)
    metrics_row: dict[str, Any] = {
        "retriever": retriever,
        "num_questions": num_questions,
        "MRR": round(mrr_sum / float(num_questions), 6) if num_questions else 0.0,
        NDCG_COLUMN: round(ndcg_sum / float(num_questions), 6) if num_questions else 0.0,
    }
    for k in eval_k_list:
        metrics_row[f"Recall@{k}"] = round(recall_sums[k] / float(num_questions), 6) if num_questions else 0.0
    for k in eval_k_list:
        metrics_row[f"Hit@{k}"] = round(hit_sums[k] / float(num_questions), 6) if num_questions else 0.0
    return metrics_row, detail_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_writable(path)  # never write into a protected folder (see protection.py)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(
    path: Path, metrics_rows: list[dict[str, Any]], eval_k_list: list[int], queries_log_path: Path | None = None
) -> None:
    ensure_writable(path)  # never write into a protected folder (see protection.py)
    path.parent.mkdir(parents=True, exist_ok=True)
    header_columns = ["Retriever", "Questions", *[f"Recall@{k}" for k in eval_k_list], "MRR", NDCG_COLUMN]
    queries_note = (
        "the questions themselves"
        if queries_log_path is None
        else f"replayed from run log `{queries_log_path}` (each question's logged rewrites, fused with RRF)"
    )
    lines = [
        "# Retrieval Evaluation Summary",
        "",
        f"Queries: {queries_note}.",
        "",
        "| " + " | ".join(header_columns) + " |",
        "| " + " | ".join(["---"] * len(header_columns)) + " |",
    ]
    for row in metrics_rows:
        values = [
            str(row["retriever"]),
            str(row["num_questions"]),
            *[f"{float(row[f'Recall@{k}']):.6f}" for k in eval_k_list],
            f"{float(row['MRR']):.6f}",
            f"{float(row[NDCG_COLUMN]):.6f}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_retrieval_evaluation(
    *,
    retriever: str,
    questions_path: Path,
    gold_path: Path,
    config_path: Path,
    bm25_dir: Path,
    vector_dir: Path,
    output_dir: Path,
    max_qids: int | None = None,
    eval_k_list_override: list[int] | None = None,
    queries_log_path: Path | None = None,
) -> dict[str, Path]:
    if retriever not in VALID_RETRIEVERS:
        raise ValueError(f"retriever must be one of: {', '.join(VALID_RETRIEVERS)}")

    config = load_config(config_path)
    eval_k_list = resolve_eval_k_list(config, eval_k_list_override)
    questions = load_questions(questions_path, max_qids=max_qids)
    gold_by_qid = load_gold_evidence(gold_path)
    queries_by_qid = load_logged_queries(queries_log_path) if queries_log_path is not None else None
    selected_retrievers = ["bm25", "dense", "hybrid"] if retriever == "all" else [retriever]

    bm25_backend = load_bm25(bm25_dir) if any(name in {"bm25", "hybrid"} for name in selected_retrievers) else None
    dense_backend = (
        load_chroma_index(vector_dir, query_cache_dir=config.get("query_embedding_cache_dir"))
        if any(name in {"dense", "hybrid"} for name in selected_retrievers)
        else None
    )

    metrics_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for retriever_name in selected_retrievers:
        metrics_row, retriever_details = evaluate_retriever(
            retriever=retriever_name,
            questions=questions,
            gold_by_qid=gold_by_qid,
            config=config,
            eval_k_list=eval_k_list,
            bm25_backend=bm25_backend,
            dense_backend=dense_backend,
            queries_by_qid=queries_by_qid,
        )
        metrics_rows.append(metrics_row)
        detail_rows.extend(retriever_details)

    metrics_fieldnames = [
        "retriever",
        "num_questions",
        *[f"Recall@{k}" for k in eval_k_list],
        "MRR",
        NDCG_COLUMN,
        *[f"Hit@{k}" for k in eval_k_list],
    ]
    details_fieldnames = [
        "qid",
        "question",
        "queries",
        "retriever",
        "gold_doc_id",
        "gold_chunk_ids",
        "retrieved_chunk_ids",
        "relevant_chunk_ids",
        "first_relevant_rank",
        *[f"Recall@{k}" for k in eval_k_list],
        "MRR",
        NDCG_COLUMN,
        *[f"Hit@{k}" for k in eval_k_list],
    ]

    metrics_path = output_dir / METRICS_FILENAME
    details_path = output_dir / DETAILS_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    write_csv(metrics_path, metrics_rows, metrics_fieldnames)
    write_csv(details_path, detail_rows, details_fieldnames)
    write_summary(summary_path, metrics_rows, eval_k_list, queries_log_path)
    return {"metrics_path": metrics_path, "details_path": details_path, "summary_path": summary_path}
