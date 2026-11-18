"""`rag-thesis run`: command-line wrapper that runs the RAG workflow for one question.

Only argument parsing, validation of the mode names and printing live here; the workflow itself is
``rag_thesis.pipeline.workflow.run_rag_workflow``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from rag_thesis.paths import PROJECT_ROOT as REPO_ROOT
from rag_thesis.pipeline.workflow import run_rag_workflow

VALID_RETRIEVERS = {"bm25", "dense", "hybrid"}
VALID_REWRITES = {"none", "single", "multi"}
VALID_GUARDS = {"off", "on"}
app = typer.Typer(add_completion=False, help="Run the deterministic LangGraph RAG workflow for one question.")


@app.command()
def main(
    qid: str = typer.Option(..., "--qid"),
    retriever: str = typer.Option("hybrid", "--retriever"),
    rewrite: str = typer.Option("none", "--rewrite"),
    guard: str = typer.Option("on", "--guard"),
    k: int = typer.Option(10, "--k", min=1),
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
    log_path: Path | None = typer.Option(None, "--log-path", dir_okay=False, file_okay=True, resolve_path=True),
) -> None:
    if retriever not in VALID_RETRIEVERS:
        raise typer.BadParameter(
            f"retriever must be one of: {', '.join(sorted(VALID_RETRIEVERS))}", param_hint="--retriever"
        )
    if rewrite not in VALID_REWRITES:
        raise typer.BadParameter(f"rewrite must be one of: {', '.join(sorted(VALID_REWRITES))}", param_hint="--rewrite")
    if guard not in VALID_GUARDS:
        raise typer.BadParameter(f"guard must be one of: {', '.join(sorted(VALID_GUARDS))}", param_hint="--guard")

    workflow_kwargs = {
        "qid": qid,
        "retriever_mode": retriever,
        "rewrite_mode": rewrite,
        "guard_mode": guard,
        "k": k,
        "questions_path": questions_path,
        "chunks_path": chunks_path,
        "config_path": config_path,
        "bm25_dir": bm25_dir,
        "vector_dir": vector_dir,
        "log_path": log_path,
        "allow_llm_calls": generate,
    }
    if generate:
        workflow_kwargs["generation_enabled"] = True

    state = run_rag_workflow(
        **workflow_kwargs,
    )

    typer.echo(f"qid: {state['qid']}")
    typer.echo(f"retriever_mode: {state['retriever_mode']}")
    typer.echo(f"rewrite_mode: {state['rewrite_mode']}")
    typer.echo(f"guard_mode: {state['guard_mode']}")
    typer.echo(f"retrieved_results: {len(state.get('retrieval_results', []))}")
    typer.echo(f"answer_generated: {'yes' if state.get('answer_text') else 'no'}")
    typer.echo(f"log_path: {log_path or (Path('outputs') / 'runs' / 'run_log.jsonl')}")


if __name__ == "__main__":
    app()
