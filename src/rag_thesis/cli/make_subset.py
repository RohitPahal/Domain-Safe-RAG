"""`rag-thesis make-subset`: command-line wrapper that creates the deterministic QA subset.

Only argument parsing, mode validation and printing live here; loading, selection and saving are
in ``rag_thesis.data.squad``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from rag_thesis.data.squad import (
    BUCKET_ORDER,
    bucket_counts,
    build_subset_payload,
    extract_valid_qa_records,
    load_squad_dataset,
    save_squad_dataset,
    select_qa_records,
)

app = typer.Typer(add_completion=False, help="Create a deterministic QA subset from a SQuAD-style dataset.")
VALID_MODES = {"first", "random", "stratified"}


def _print_bucket_counts(label: str, counts: dict[str, int]) -> None:
    typer.echo(label)
    for bucket in BUCKET_ORDER:
        typer.echo(f"{bucket}: {counts[bucket]}")


@app.command()
def main(
    input_path: Path = typer.Option(..., "--in", exists=True, dir_okay=False, file_okay=True, resolve_path=True),
    output_path: Path = typer.Option(..., "--out", dir_okay=False, file_okay=True, resolve_path=True),
    n_qas: int = typer.Option(..., "--n_qas", min=1),
    mode: str = typer.Option(...),
    seed: int = typer.Option(...),
) -> None:
    if mode not in VALID_MODES:
        raise typer.BadParameter(f"mode must be one of: {', '.join(sorted(VALID_MODES))}", param_hint="--mode")

    payload = load_squad_dataset(input_path)
    valid_records = extract_valid_qa_records(payload)
    before_counts = bucket_counts(valid_records)
    selected_records = select_qa_records(valid_records, n_qas=n_qas, mode=mode, seed=seed)
    after_counts = bucket_counts(selected_records)
    subset_payload = build_subset_payload(payload, selected_records)
    save_squad_dataset(output_path, subset_payload)

    _print_bucket_counts("Before selection:", before_counts)
    _print_bucket_counts("After selection:", after_counts)
    typer.echo(f"valid_qas_available: {len(valid_records)}")
    typer.echo(f"selected_qas: {len(selected_records)}")
    typer.echo(f"output_path: {output_path}")


if __name__ == "__main__":
    app()
