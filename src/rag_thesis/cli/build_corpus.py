"""`rag-thesis build-corpus`: command-line wrapper that builds the corpus artifacts.

Only argument parsing and printing live here; chunking and gold mapping are in
``rag_thesis.data.corpus`` (``build_corpus_artifacts``, ``load_embed_tokenizer``).
"""

from __future__ import annotations

from pathlib import Path

import typer

from rag_thesis.config import load_config
from rag_thesis.data.corpus import build_corpus_artifacts, load_embed_tokenizer
from rag_thesis.paths import PROJECT_ROOT as REPO_ROOT

app = typer.Typer(add_completion=False, help="Build processed corpus artifacts from a SQuAD-style subset dataset.")


@app.command()
def main(
    input_path: Path = typer.Option(..., "--in", exists=True, dir_okay=False, file_okay=True, resolve_path=True),
    config_path: Path = typer.Option(
        REPO_ROOT / "configs" / "default.yaml",
        "--config",
        exists=True,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
    ),
    processed_dir: Path = typer.Option(
        REPO_ROOT / "data" / "processed", file_okay=False, dir_okay=True, resolve_path=True
    ),
    eval_dir: Path = typer.Option(REPO_ROOT / "data" / "eval", file_okay=False, dir_okay=True, resolve_path=True),
    report_dir: Path = typer.Option(
        REPO_ROOT / "outputs" / "reports", file_okay=False, dir_okay=True, resolve_path=True
    ),
) -> None:
    config = load_config(config_path)

    embed_model = config["embed_model"]
    chunk_tokens = int(config["chunk_tokens"])
    chunk_overlap = int(config["chunk_overlap"])
    tokenizer = load_embed_tokenizer(embed_model)

    counts = build_corpus_artifacts(
        input_path=input_path,
        processed_dir=processed_dir,
        eval_dir=eval_dir,
        report_dir=report_dir,
        tokenizer=tokenizer,
        tokenizer_name=embed_model,
        chunk_tokens=chunk_tokens,
        chunk_overlap=chunk_overlap,
    )

    typer.echo(f"documents: {counts['documents']}")
    typer.echo(f"chunks: {counts['chunks']}")
    typer.echo(f"questions: {counts['questions']}")
    typer.echo(f"gold_evidence: {counts['gold_evidence']}")
    typer.echo(f"mapping_failures: {counts['mapping_failures']}")


if __name__ == "__main__":
    app()
