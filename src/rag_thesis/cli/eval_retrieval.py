"""`rag-thesis eval-retrieval`: command-line wrapper around retrieval evaluation.

Only argument parsing and printing live here; the evaluation itself is
``rag_thesis.evaluation.retrieval.run_retrieval_evaluation``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from rag_thesis.evaluation.retrieval import VALID_RETRIEVERS, run_retrieval_evaluation
from rag_thesis.paths import PROJECT_ROOT as REPO_ROOT

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "reports"

app = typer.Typer(add_completion=False, help="Run deterministic retrieval-only evaluation on all questions.")


@app.command()
def main(
    retriever: str = typer.Option("all", "--retriever"),
    max_qids: int | None = typer.Option(None, "--max_qids", "--max-qids", min=1),
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
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, "--output-dir", dir_okay=True, file_okay=False, resolve_path=True
    ),
    queries_from_log: Path | None = typer.Option(
        None,
        "--queries-from-log",
        help="Replay the queries of this run log (each question's logged rewrites, fused with RRF) "
        "instead of the raw questions.",
        exists=True,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
    ),
) -> None:
    if retriever not in VALID_RETRIEVERS:
        raise typer.BadParameter(f"retriever must be one of: {', '.join(VALID_RETRIEVERS)}", param_hint="--retriever")

    paths = run_retrieval_evaluation(
        retriever=retriever,
        questions_path=questions_path,
        gold_path=gold_path,
        config_path=config_path,
        bm25_dir=bm25_dir,
        vector_dir=vector_dir,
        output_dir=output_dir,
        max_qids=max_qids,
        queries_log_path=queries_from_log,
    )

    typer.echo(f"metrics_path: {paths['metrics_path']}")
    typer.echo(f"details_path: {paths['details_path']}")
    typer.echo(f"summary_path: {paths['summary_path']}")


if __name__ == "__main__":
    app()
