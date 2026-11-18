"""`rag-thesis bench-retrieval`: command-line wrapper around the retrieval latency benchmark.

Only argument parsing and printing live here; the benchmark itself is
``rag_thesis.evaluation.bench.run_retrieval_benchmark``. No API key needed.
"""

from __future__ import annotations

from pathlib import Path

import typer

from rag_thesis.evaluation.bench import COLUMNS, run_retrieval_benchmark
from rag_thesis.evaluation.retrieval import VALID_RETRIEVERS
from rag_thesis.paths import PROJECT_ROOT as REPO_ROOT

app = typer.Typer(add_completion=False, help="Measure retrieval latency per retriever (index load and per query).")


@app.command()
def main(
    retriever: str = typer.Option("all", "--retriever"),
    k: int | None = typer.Option(
        None, "--k", min=1, help="Results per query (default: the largest eval_k_list value)."
    ),
    max_qids: int | None = typer.Option(None, "--max-qids", min=1),
    questions_path: Path = typer.Option(
        REPO_ROOT / "data" / "eval" / "questions.jsonl",
        "--questions-path",
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
        REPO_ROOT / "outputs" / "reports", "--output-dir", dir_okay=True, file_okay=False, resolve_path=True
    ),
) -> None:
    if retriever not in VALID_RETRIEVERS:
        raise typer.BadParameter(f"retriever must be one of: {', '.join(VALID_RETRIEVERS)}", param_hint="--retriever")

    result = run_retrieval_benchmark(
        retriever=retriever,
        questions_path=questions_path,
        config_path=config_path,
        bm25_dir=bm25_dir,
        vector_dir=vector_dir,
        output_dir=output_dir,
        k=k,
        max_qids=max_qids,
    )
    typer.echo("  ".join(COLUMNS))
    for row in result["rows"]:
        typer.echo("  ".join(str(row[column]) for column in COLUMNS))
    typer.echo(f"csv_path: {result['csv_path']}")
    typer.echo(f"summary_path: {result['summary_path']}")


if __name__ == "__main__":
    app()
