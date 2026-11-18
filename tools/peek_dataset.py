from __future__ import annotations

from pathlib import Path

import typer

from rag_thesis.data.squad import analyze_dataset, load_squad_dataset

app = typer.Typer(add_completion=False, help="Inspect a SQuAD-style dataset without mutating it.")


@app.command()
def main(
    input_path: Path = typer.Option(..., "--in", exists=True, dir_okay=False, file_okay=True, resolve_path=True),
) -> None:
    payload = load_squad_dataset(input_path)
    summary = analyze_dataset(payload)

    typer.echo(f"data_items: {summary['data_items']}")
    typer.echo(f"paragraphs: {summary['paragraphs']}")
    typer.echo(f"total_qas: {summary['total_qas']}")
    typer.echo("first_3_qas:")
    for qid, question in summary["preview_pairs"]:
        typer.echo(f"- {qid}: {question}")

    typer.echo(f"empty_questions: {summary['empty_questions']}")
    typer.echo(f"empty_answers: {summary['empty_answers']}")
    typer.echo(f"invalid_answer_start: {summary['invalid_answer_start']}")
    typer.echo(f"answer_text_mismatch: {summary['answer_text_mismatch']}")
    typer.echo(f"malformed_data_items: {summary['malformed_data_items']}")
    typer.echo(f"malformed_paragraphs: {summary['malformed_paragraphs']}")
    typer.echo(f"malformed_qas: {summary['malformed_qas']}")
    typer.echo(f"malformed_answers: {summary['malformed_answers']}")


if __name__ == "__main__":
    app()
