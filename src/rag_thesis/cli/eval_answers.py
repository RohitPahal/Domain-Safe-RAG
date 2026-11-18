"""`rag-thesis eval-answers`: command-line wrapper around answer-quality evaluation.

Only argument parsing and printing live here; the evaluation itself is
``rag_thesis.evaluation.answers.evaluate_answer_quality``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from rag_thesis.evaluation.answers import evaluate_answer_quality
from rag_thesis.paths import PROJECT_ROOT as REPO_ROOT

DEFAULT_QUESTIONS_PATH = REPO_ROOT / "data" / "eval" / "questions.jsonl"
DEFAULT_GOLD_EVIDENCE_PATH = REPO_ROOT / "data" / "eval" / "gold_evidence.jsonl"
DEFAULT_REPORTS_DIR = REPO_ROOT / "outputs" / "reports"

app = typer.Typer(
    add_completion=False, help="Evaluate generated answers against gold answers for an existing RAG run log."
)


@app.command()
def main(
    run_log_path: Path = typer.Option(
        ..., "--run-log-path", exists=True, dir_okay=False, file_okay=True, resolve_path=True
    ),
    questions_path: Path = typer.Option(
        DEFAULT_QUESTIONS_PATH, "--questions-path", exists=True, dir_okay=False, file_okay=True, resolve_path=True
    ),
    gold_evidence_path: Path = typer.Option(
        DEFAULT_GOLD_EVIDENCE_PATH,
        "--gold-evidence-path",
        exists=True,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
    ),
    reports_dir: Path = typer.Option(
        DEFAULT_REPORTS_DIR, "--reports-dir", dir_okay=True, file_okay=False, resolve_path=True
    ),
) -> None:
    outputs = evaluate_answer_quality(
        run_log_path=run_log_path,
        questions_path=questions_path,
        gold_evidence_path=gold_evidence_path,
        reports_dir=reports_dir,
    )
    summary = outputs["summary"]
    typer.echo(f"per_question_path: {outputs['per_question_path']}")
    typer.echo(f"summary_path: {outputs['summary_path']}")
    typer.echo(f"total_questions: {summary['total_questions']}")
    typer.echo(f"answered: {summary['answered']}")
    typer.echo(f"unanswered: {summary['unanswered']}")
    typer.echo(f"answer_rate: {summary['answer_rate']:.6f}")
    typer.echo(f"exact_match_normalized: {summary['exact_match_normalized']:.6f}")
    typer.echo(f"token_f1: {summary['token_f1']:.6f}")
    typer.echo(f"lenient_containment_rate: {summary['lenient_containment_rate']:.6f}")


if __name__ == "__main__":
    app()
