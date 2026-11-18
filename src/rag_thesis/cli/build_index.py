"""`rag-thesis build-index`: build the BM25 and/or dense (ChromaDB) index from chunks.jsonl.

Default locations: `indexes/bm25/` and `indexes/chroma/`.
"""

from __future__ import annotations

from pathlib import Path

import typer

from rag_thesis.config import load_config
from rag_thesis.io import load_chunks
from rag_thesis.paths import PROJECT_ROOT as REPO_ROOT
from rag_thesis.retrieval.bm25 import build_bm25, save_bm25
from rag_thesis.retrieval.chroma_store import LEGACY_HNSWLIB_MARKER, build_chroma_index

app = typer.Typer(add_completion=False, help="Build retrieval indexes (BM25 and/or dense ChromaDB) from chunks.jsonl.")


@app.command()
def main(
    bm25: bool = typer.Option(False, "--bm25", help="Build the BM25 chunk index."),
    dense: bool = typer.Option(False, "--dense", help="Build the dense ChromaDB chunk index."),
    chunks_path: Path = typer.Option(
        REPO_ROOT / "data" / "processed" / "chunks.jsonl",
        "--chunks-path",
        exists=True,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
    ),
    index_dir: Path | None = typer.Option(None, "--index-dir", file_okay=False, dir_okay=True, resolve_path=True),
    config_path: Path = typer.Option(
        REPO_ROOT / "configs" / "default.yaml",
        "--config-path",
        exists=True,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
    ),
) -> None:
    if not bm25 and not dense:
        raise typer.BadParameter("Pass --bm25 or --dense to choose which index to build.", param_hint="--bm25")
    if bm25 and dense and index_dir is not None:
        raise typer.BadParameter(
            "Use one index type per invocation when overriding --index-dir.", param_hint="--index-dir"
        )

    chunks = load_chunks(chunks_path)
    if bm25:
        bm25_index = build_bm25(chunks)
        bm25_index_dir = index_dir or (REPO_ROOT / "indexes" / "bm25")
        save_bm25(bm25_index, bm25_index_dir)
        typer.echo(f"bm25_chunks_indexed: {len(chunks)}")
        typer.echo(f"bm25_index_dir: {bm25_index_dir}")

    if dense:
        dense_index_dir = index_dir or (REPO_ROOT / "indexes" / "chroma")
        # Never write a Chroma database into an old hnswlib index folder (e.g. indexes/vector):
        # the two formats would be mixed in one directory.
        if (dense_index_dir / LEGACY_HNSWLIB_MARKER).exists():
            raise typer.BadParameter(
                f"{dense_index_dir} contains an old hnswlib index; choose an empty or Chroma directory.",
                param_hint="--index-dir",
            )
        config = load_config(config_path)
        build_chroma_index(chunks, config, dense_index_dir)  # writes to disk as it builds
        typer.echo(f"dense_chunks_indexed: {len(chunks)}")
        typer.echo(f"dense_index_dir: {dense_index_dir}")


if __name__ == "__main__":
    app()
