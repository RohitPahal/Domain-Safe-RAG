"""`rag-thesis judge`: command-line wrapper around LLM-judge evaluation.

Only argument parsing and printing live here; the evaluation itself is
``rag_thesis.evaluation.judge.evaluate_with_judge``. Judging calls an LLM: it needs
``--allow-llm-calls`` and an API key (without the flag it can only replay cached judgements).
"""

from __future__ import annotations

from pathlib import Path

import typer

from rag_thesis.evaluation.judge import OVERALL_VERDICTS, evaluate_with_judge
from rag_thesis.paths import PROJECT_ROOT as REPO_ROOT

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_CHUNKS_PATH = REPO_ROOT / "data" / "processed" / "chunks.jsonl"
DEFAULT_REPORTS_DIR = REPO_ROOT / "outputs" / "reports"

app = typer.Typer(add_completion=False, help="Evaluate saved RAG run logs with a strict reference-guided LLM judge.")


@app.command()
def main(
    run_log_path: Path = typer.Option(
        ..., "--run-log-path", exists=True, dir_okay=False, file_okay=True, resolve_path=True
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
        DEFAULT_CHUNKS_PATH, "--chunks-path", exists=True, dir_okay=False, file_okay=True, resolve_path=True
    ),
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH, "--config-path", exists=True, dir_okay=False, file_okay=True, resolve_path=True
    ),
    reports_dir: Path = typer.Option(
        DEFAULT_REPORTS_DIR, "--reports-dir", dir_okay=True, file_okay=False, resolve_path=True
    ),
    judge_model: str | None = typer.Option(None, "--judge-model"),
    first_n: int | None = typer.Option(None, "--first-n", min=1),
    output_stem: str | None = typer.Option(None, "--output-stem"),
    allow_llm_calls: bool = typer.Option(
        False, "--allow-llm-calls", help="Explicitly allow real LLM calls to the judge model (costs API usage)."
    ),
) -> None:
    outputs = evaluate_with_judge(
        run_log_path=run_log_path,
        questions_path=questions_path,
        chunks_path=chunks_path,
        config_path=config_path,
        reports_dir=reports_dir,
        judge_model=judge_model,
        first_n=first_n,
        output_stem=output_stem,
        allow_llm_calls=allow_llm_calls,
    )
    summary = outputs["summary"]
    typer.echo(f"judge_model: {outputs['judge_model']} ({outputs['judge_provider']})")
    typer.echo(f"per_question_path: {outputs['per_question_path']}")
    typer.echo(f"summary_path: {outputs['summary_path']}")
    typer.echo(f"total_questions: {summary['total_questions']}")
    for label in OVERALL_VERDICTS:
        count_key = f"{label.lower()}_count"
        rate_key = f"{label.lower()}_rate"
        typer.echo(f"{count_key}: {summary[count_key]}")
        typer.echo(f"{rate_key}: {summary[rate_key]:.6f}")
    typer.echo(f"faithfulness_mean_score: {summary['faithfulness_mean_score']:.6f}")
    typer.echo(f"relevance_mean_score: {summary['relevance_mean_score']:.6f}")
    typer.echo(f"faithful_and_relevant_rate: {summary['faithful_and_relevant_rate']:.6f}")


if __name__ == "__main__":
    app()
