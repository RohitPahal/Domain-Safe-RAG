"""`rag-thesis rewrite-batch`: rewrite a question set into the rewrite cache.

Only argument parsing and printing live here; the work is ``rag_thesis.rewrite.rewriter.rewrite_batch``.
Real LLM calls need ``--allow-llm-calls`` (and the rewrite role's API key); without it the command only
reports which questions are already in the cache.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import typer

from rag_thesis.config import load_config
from rag_thesis.io import load_questions
from rag_thesis.llm.registry import build_chat_model
from rag_thesis.llm.settings import resolve_llm_settings
from rag_thesis.llm.wrappers import CallLog
from rag_thesis.paths import PROJECT_ROOT as REPO_ROOT
from rag_thesis.rewrite.rewriter import QueryRewriter, rewrite_batch

app = typer.Typer(add_completion=False, help="Rewrite a question set into the rewrite cache (e.g. a dev split).")


@app.command()
def main(
    questions_path: Path = typer.Option(
        REPO_ROOT / "data" / "dev" / "questions.jsonl",
        "--questions-path",
        exists=True,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
    ),
    mode: str = typer.Option("multi", "--mode", help="single, multi, structured (R6) or hyde (R8)"),
    max_qids: int | None = typer.Option(None, "--max-qids", min=1),
    config_path: Path = typer.Option(
        REPO_ROOT / "configs" / "default.yaml",
        "--config-path",
        exists=True,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
    ),
    allow_llm_calls: bool = typer.Option(
        False, "--allow-llm-calls", help="Call the rewrite model for questions not in the cache (needs its API key)."
    ),
) -> None:
    if mode not in ("single", "multi", "structured", "hyde"):
        raise typer.BadParameter("mode must be 'single', 'multi', 'structured' or 'hyde'.", param_hint="--mode")
    config = load_config(config_path)
    call_log = CallLog() if allow_llm_calls else None
    chat_model = (
        build_chat_model("rewrite", resolve_llm_settings(config), allow_llm_calls=True, call_log=call_log)
        if allow_llm_calls
        else None
    )
    rows = rewrite_batch(
        load_questions(questions_path, max_qids=max_qids), config=config, rewrite_mode=mode, chat_model=chat_model
    )
    counts = Counter(row["status"] for row in rows)
    typer.echo(f"questions: {len(rows)}")
    typer.echo(f"in_cache: {counts['ok']}")
    typer.echo(f"not_cached: {counts['not_cached']}")
    typer.echo(f"failed: {counts['failed']}")
    typer.echo(f"llm_calls: {len(call_log) if call_log is not None else 0}")
    typer.echo(f"rewrite_cache: {QueryRewriter(config=config).cache_path}")
    for row in [row for row in rows if row["status"] == "failed"][:5]:
        typer.echo(f"failed {row['qid']}: {row['error']}")
    if counts["not_cached"]:
        typer.echo("Run again with --allow-llm-calls (and the rewrite model's API key) to fill the cache.")


if __name__ == "__main__":
    app()
