from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import typer

from rag_thesis.config import load_config
from rag_thesis.io import load_gold_evidence, load_questions
from rag_thesis.paths import PROJECT_ROOT as REPO_ROOT
from rag_thesis.pipeline.workflow import run_rag_workflow
from rag_thesis.retrieval.bm25 import load_bm25
from rag_thesis.retrieval.chroma_store import load_chroma_index

DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "reports" / "audit_manual_sheet.csv"
DEFAULT_RUN_LOG_PATH = REPO_ROOT / "outputs" / "runs" / "audit_sheet_run_log.jsonl"
AUDIT_CONFIGS = (
    {"label": "A_hybrid_none_guard_off", "retriever_mode": "hybrid", "rewrite_mode": "none", "guard_mode": "off"},
    {"label": "B_hybrid_multi_guard_on", "retriever_mode": "hybrid", "rewrite_mode": "multi", "guard_mode": "on"},
)

app = typer.Typer(
    add_completion=False, help="Generate the deterministic manual audit sheet for the two required RAG configs."
)


def _serialize_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def _join_answers(question: dict[str, Any]) -> str:
    texts: list[str] = []
    for answer in question.get("answers", []):
        text = str(answer["text"]).strip()
        if text and text not in texts:
            texts.append(text)
    return " | ".join(texts)


def _to_bool_string(value: bool) -> str:
    return "true" if value else "false"


def generate_audit_rows(
    *,
    questions_path: Path,
    gold_path: Path,
    chunks_path: Path,
    config_path: Path,
    bm25_dir: Path,
    vector_dir: Path,
    run_log_path: Path,
    generate: bool = False,
) -> list[dict[str, Any]]:
    questions = load_questions(questions_path)
    gold_by_qid = load_gold_evidence(gold_path)
    config = load_config(config_path)
    k = int(config.get("retrieve_k", 10))
    bm25_backend = load_bm25(bm25_dir)
    dense_backend = load_chroma_index(vector_dir, query_cache_dir=config.get("query_embedding_cache_dir"))

    rows: list[dict[str, Any]] = []
    for question in questions:
        qid = str(question["qid"])
        gold = gold_by_qid.get(qid)
        if gold is None:
            raise ValueError(f"Gold evidence not found for qid: {qid}")

        gold_chunk_ids = [str(chunk_id) for chunk_id in gold["gold_chunk_ids"]]
        gold_set = set(gold_chunk_ids)
        for audit_config in AUDIT_CONFIGS:
            workflow_kwargs = {
                "qid": qid,
                "retriever_mode": str(audit_config["retriever_mode"]),
                "rewrite_mode": str(audit_config["rewrite_mode"]),
                "guard_mode": str(audit_config["guard_mode"]),
                "k": k,
                "questions_path": questions_path,
                "chunks_path": chunks_path,
                "config_path": config_path,
                "bm25_dir": bm25_dir,
                "vector_dir": vector_dir,
                "log_path": run_log_path,
                "bm25_backend": bm25_backend,
                "dense_backend": dense_backend,
                "allow_llm_calls": generate,
            }
            if generate:
                workflow_kwargs["generation_enabled"] = True

            state = run_rag_workflow(**workflow_kwargs)
            retrieved_chunk_ids = [str(result["chunk_id"]) for result in state.get("retrieval_results", [])]
            retrieved_set = set(retrieved_chunk_ids)
            citations = [str(citation) for citation in state.get("citations", [])]
            evidence_found = bool(retrieved_set.intersection(gold_set))
            citations_valid = set(citations).issubset(retrieved_set)

            rows.append(
                {
                    "qid": qid,
                    "question": str(question["question"]),
                    "gold_answer": _join_answers(question),
                    "gold_chunk_ids": _serialize_list(gold_chunk_ids),
                    "retrieved_chunk_ids": _serialize_list(retrieved_chunk_ids),
                    "predicted_answer": state.get("answer_text") or "",
                    "evidence_found": _to_bool_string(evidence_found),
                    "citations_valid": _to_bool_string(citations_valid),
                    "config": str(audit_config["label"]),
                    "notes": "",
                }
            )
    return rows


def _fallback_output_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.generated{path.suffix}")


def write_audit_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "qid",
        "question",
        "gold_answer",
        "gold_chunk_ids",
        "retrieved_chunk_ids",
        "predicted_answer",
        "evidence_found",
        "citations_valid",
        "config",
        "notes",
    ]
    output_path = path
    try:
        handle = output_path.open("w", encoding="utf-8", newline="")
    except PermissionError:
        output_path = _fallback_output_path(path)
        handle = output_path.open("w", encoding="utf-8", newline="")

    with handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def generate_audit_sheet(
    *,
    questions_path: Path,
    gold_path: Path,
    chunks_path: Path,
    config_path: Path,
    bm25_dir: Path,
    vector_dir: Path,
    output_path: Path,
    run_log_path: Path,
    generate: bool = False,
) -> Path:
    rows = generate_audit_rows(
        questions_path=questions_path,
        gold_path=gold_path,
        chunks_path=chunks_path,
        config_path=config_path,
        bm25_dir=bm25_dir,
        vector_dir=vector_dir,
        run_log_path=run_log_path,
        generate=generate,
    )
    return write_audit_csv(output_path, rows)


@app.command()
def main(
    questions_path: Path = typer.Option(
        REPO_ROOT / "data" / "eval" / "questions.jsonl",
        "--questions-path",
        exists=True,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
    ),
    gold_path: Path = typer.Option(
        REPO_ROOT / "data" / "eval" / "gold_evidence.jsonl",
        "--gold-path",
        exists=True,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
    ),
    chunks_path: Path = typer.Option(
        REPO_ROOT / "data" / "processed" / "chunks.jsonl",
        "--chunks-path",
        exists=True,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
    ),
    config_path: Path = typer.Option(
        REPO_ROOT / "configs" / "default.yaml",
        "--config-path",
        exists=True,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
    ),
    bm25_dir: Path = typer.Option(
        REPO_ROOT / "indexes" / "bm25", "--bm25-dir", dir_okay=True, file_okay=False, resolve_path=True
    ),
    vector_dir: Path = typer.Option(
        REPO_ROOT / "indexes" / "chroma",
        "--vector-dir",
        help="Dense (ChromaDB) index directory.",
        dir_okay=True,
        file_okay=False,
        resolve_path=True,
    ),
    output_path: Path = typer.Option(
        DEFAULT_OUTPUT_PATH, "--output-path", dir_okay=False, file_okay=True, resolve_path=True
    ),
    run_log_path: Path = typer.Option(
        DEFAULT_RUN_LOG_PATH, "--run-log-path", dir_okay=False, file_okay=True, resolve_path=True
    ),
    generate: bool = typer.Option(
        False, "--generate", help="Explicitly enable live LLM-backed rewriting, guard repair, and answer generation."
    ),
) -> None:
    path = generate_audit_sheet(
        questions_path=questions_path,
        gold_path=gold_path,
        chunks_path=chunks_path,
        config_path=config_path,
        bm25_dir=bm25_dir,
        vector_dir=vector_dir,
        output_path=output_path,
        run_log_path=run_log_path,
        generate=generate,
    )
    typer.echo(f"audit_sheet_path: {path}")


if __name__ == "__main__":
    app()
