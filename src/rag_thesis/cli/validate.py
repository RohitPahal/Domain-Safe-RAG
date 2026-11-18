"""`rag-thesis validate`: command-line wrapper that checks the JSONL artifacts.

Only argument parsing, printing and the exit code live here; the checks are in
``rag_thesis.data.validate``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from rag_thesis.data.validate import validate_artifacts
from rag_thesis.paths import PROJECT_ROOT as REPO_ROOT

app = typer.Typer(add_completion=False, help="Validate processed and evaluation JSONL artifacts.")


@app.command()
def main(
    processed_dir: Path = typer.Option(
        REPO_ROOT / "data" / "processed", exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    eval_dir: Path = typer.Option(
        REPO_ROOT / "data" / "eval", exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
) -> None:
    summary = validate_artifacts(processed_dir, eval_dir)

    typer.echo(f"Total records: {summary.total}")
    typer.echo(f"Valid records: {summary.valid}")
    typer.echo(f"Invalid records: {summary.invalid}")

    if summary.errors:
        typer.echo("Errors:", err=True)
        for message in summary.errors[:10]:
            typer.echo(message, err=True)
        raise typer.Exit(code=1)

    typer.echo("Validation passed.")


if __name__ == "__main__":
    app()
