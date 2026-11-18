"""`rag-thesis run-batch`: command-line wrapper that runs the RAG workflow over many questions.

Only argument parsing, validation of the mode names and printing live here; the batch logic
(question selection, one shared run context) is ``rag_thesis.pipeline.run_batch.run_rag_batch``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from rag_thesis.paths import PROJECT_ROOT as REPO_ROOT
from rag_thesis.pipeline.run_batch import DEFAULT_RANDOM_SEED, run_rag_batch

VALID_RETRIEVERS = {"bm25", "dense", "hybrid"}
VALID_REWRITES = {"none", "single", "multi"}
VALID_GUARDS = {"off", "on"}
DEFAULT_BATCH_LOG_PATH = REPO_ROOT / "outputs" / "runs" / "run_log_batch.jsonl"

app = typer.Typer(
    add_completion=False,
    help="Run the deterministic RAG workflow over multiple qids using one explicit preloaded context.",
)


@app.command()
def main(
    retriever: str = typer.Option("hybrid", "--retriever"),
    rewrite: str = typer.Option("none", "--rewrite"),
    guard: str = typer.Option("off", "--guard"),
    k: int = typer.Option(10, "--k", min=1),
    first_n: int | None = typer.Option(None, "--first-n", "--first_n", min=1),
    random_n: int | None = typer.Option(None, "--random-n", "--random_n", min=1),
    seed: int = typer.Option(DEFAULT_RANDOM_SEED, "--seed"),
    generate: bool = typer.Option(
        False, "--generate", help="Explicitly enable live LLM-backed rewriting, guard repair, and answer generation."
    ),
    questions_path: Path = typer.Option(
        REPO_ROOT / "data" / "eval" / "questions.jsonl",
        "--questions-path",
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
    bm25_dir: Path = typer.Option(
        REPO_ROOT / "indexes" / "bm25", "--bm25-dir", file_okay=False, dir_okay=True, resolve_path=True
    ),
    vector_dir: Path = typer.Option(
        REPO_ROOT / "indexes" / "chroma",
        "--vector-dir",
        help="Dense (ChromaDB) index directory.",
        file_okay=False,
        dir_okay=True,
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
    log_path: Path = typer.Option(
        DEFAULT_BATCH_LOG_PATH, "--log-path", dir_okay=False, file_okay=True, resolve_path=True
    ),
) -> None:
    if retriever not in VALID_RETRIEVERS:
        raise typer.BadParameter(
            f"retriever must be one of: {', '.join(sorted(VALID_RETRIEVERS))}", param_hint="--retriever"
        )
    if rewrite not in VALID_REWRITES:
        raise typer.BadParameter(f"rewrite must be one of: {', '.join(sorted(VALID_REWRITES))}", param_hint="--rewrite")
    if guard not in VALID_GUARDS:
        raise typer.BadParameter(f"guard must be one of: {', '.join(sorted(VALID_GUARDS))}", param_hint="--guard")
    if first_n is not None and random_n is not None:
        raise typer.BadParameter("Specify only one of --first-n or --random-n.")

    typer.echo("preloading_context: start")
    states = run_rag_batch(
        retriever_mode=retriever,
        rewrite_mode=rewrite,
        guard_mode=guard,
        k=k,
        questions_path=questions_path,
        chunks_path=chunks_path,
        config_path=config_path,
        bm25_dir=bm25_dir,
        vector_dir=vector_dir,
        log_path=log_path,
        first_n=first_n,
        random_n=random_n,
        seed=seed,
        generate=generate,
        progress_callback=lambda index, total, state: typer.echo(
            f"{index}/{total} qid={state['qid']} retrieved={len(state.get('retrieval_results', []))} "
            f"answered={'yes' if state.get('answer_text') else 'no'}"
        ),
    )
    typer.echo("preloading_context: done")
    typer.echo(f"total_qids: {len(states)}")
    typer.echo(f"log_path: {log_path}")


if __name__ == "__main__":
    app()
