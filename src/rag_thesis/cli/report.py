"""`rag-thesis report`: command-line wrapper around the QA / latency / cost report.

Only argument parsing and printing live here; the report itself is
``rag_thesis.evaluation.report.run_qa_report``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from rag_thesis.evaluation.report import (
    DEFAULT_COMPLETION_PRICE_PER_1K_USD,
    DEFAULT_PROMPT_PRICE_PER_1K_USD,
    run_qa_report,
)
from rag_thesis.paths import PROJECT_ROOT as REPO_ROOT

DEFAULT_RUNS_DIR = REPO_ROOT / "outputs" / "runs"
DEFAULT_REPORTS_DIR = REPO_ROOT / "outputs" / "reports"
DEFAULT_QUESTIONS_PATH = REPO_ROOT / "data" / "eval" / "questions.jsonl"

app = typer.Typer(
    add_completion=False, help="Generate deterministic thesis-final QA, latency, and cost reports from run logs."
)


@app.command()
def main(
    runs_dir: Path = typer.Option(
        DEFAULT_RUNS_DIR, "--runs-dir", exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
    reports_dir: Path = typer.Option(
        DEFAULT_REPORTS_DIR, "--reports-dir", exists=True, dir_okay=True, file_okay=False, resolve_path=True
    ),
    prompt_price_per_1k_usd: float = typer.Option(DEFAULT_PROMPT_PRICE_PER_1K_USD, "--prompt-price-per-1k-usd"),
    completion_price_per_1k_usd: float = typer.Option(
        DEFAULT_COMPLETION_PRICE_PER_1K_USD, "--completion-price-per-1k-usd"
    ),
    questions_path: Path | None = typer.Option(
        None,
        "--questions-path",
        help="Questions with gold answers, to count rewrites that leak an answer "
        "(default: data/eval/questions.jsonl if it exists).",
        exists=True,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
    ),
) -> None:
    if questions_path is None and DEFAULT_QUESTIONS_PATH.is_file():
        questions_path = DEFAULT_QUESTIONS_PATH
    outputs = run_qa_report(
        runs_dir=runs_dir,
        reports_dir=reports_dir,
        prompt_price_per_1k_usd=prompt_price_per_1k_usd,
        completion_price_per_1k_usd=completion_price_per_1k_usd,
        questions_path=questions_path,
    )
    typer.echo(f"qa_metrics_path: {outputs['qa_metrics_path']}")
    typer.echo(f"cost_latency_path: {outputs['cost_latency_path']}")
    typer.echo(f"summary_path: {outputs['summary_path']}")
    typer.echo(f"audit_path: {outputs['audit_path'] or 'none'}")
    typer.echo(f"guard_metrics_path: {outputs['guard_metrics_path'] or 'none'}")


if __name__ == "__main__":
    app()
