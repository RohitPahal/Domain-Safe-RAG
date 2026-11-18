from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import typer

from rag_thesis.config import load_config
from rag_thesis.io import load_chunks_by_id, load_gold_evidence, load_questions_by_id
from rag_thesis.paths import PROJECT_ROOT as REPO_ROOT
from rag_thesis.retrieval.bm25 import load_bm25
from rag_thesis.retrieval.chroma_store import load_chroma_index

MODE_CHOICES = {"bm25", "dense", "hybrid"}
PREVIEW_LIMIT = 320
app = typer.Typer(add_completion=False, help="Inspect retrieval results from BM25, dense, or hybrid retrieval.")


def _resolve_query(
    query: str | None,
    qid: str | None,
    questions_path: Path,
) -> tuple[str, dict[str, Any] | None]:
    if bool(query) == bool(qid):
        raise typer.BadParameter("Provide exactly one of --query or --qid.")

    if query is not None:
        return query, None

    questions = load_questions_by_id(questions_path)
    question_record = questions.get(qid or "")
    if question_record is None:
        raise typer.BadParameter(f"QID not found: {qid}", param_hint="--qid")
    return str(question_record["question"]), question_record


def _load_backend(mode: str, bm25_dir: Path, vector_dir: Path):
    if mode == "bm25":
        if not bm25_dir.exists():
            raise RuntimeError(f"BM25 index directory is missing: {bm25_dir}")
        try:
            return load_bm25(bm25_dir)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Failed to load the BM25 backend. Ensure rank-bm25 is installed in the active Python environment."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load the BM25 index from {bm25_dir}: {exc}") from exc
    if mode == "dense":
        if not vector_dir.exists():
            raise RuntimeError(f"Dense index directory is missing: {vector_dir}")
        try:
            return load_chroma_index(vector_dir)  # no query cache: always shows the live encoder's results
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Failed to load the dense backend. Ensure sentence-transformers, torch, and chromadb are installed."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load the dense index from {vector_dir}: {exc}") from exc
    raise ValueError(f"Unsupported mode: {mode}")


def _exit_with_error(message: str) -> None:
    typer.secho(f"Error: {message}", err=True, fg=typer.colors.RED)
    raise typer.Exit(code=1)


def _hybrid_retrieve(query: str, k: int, bm25_index: Any, vector_index: Any, rrf_k: int) -> list[dict[str, Any]]:
    candidate_k = max(k, 10)
    bm25_results = bm25_index.retrieve(query, candidate_k)
    dense_results = vector_index.retrieve(query, candidate_k)
    fused: dict[str, dict[str, Any]] = {}

    for source_name, results in (("bm25", bm25_results), ("dense", dense_results)):
        for result in results:
            chunk_id = result["chunk_id"]
            fused_entry = fused.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "doc_id": result["doc_id"],
                    "score": 0.0,
                    "sources": [],
                },
            )
            fused_entry["score"] += 1.0 / (rrf_k + int(result["rank"]))
            if source_name not in fused_entry["sources"]:
                fused_entry["sources"].append(source_name)

    ranked = sorted(
        fused.values(),
        key=lambda item: (-float(item["score"]), item["chunk_id"]),
    )
    final_results: list[dict[str, Any]] = []
    for rank, result in enumerate(ranked[:k], start=1):
        final_results.append(
            {
                "chunk_id": result["chunk_id"],
                "doc_id": result["doc_id"],
                "score": float(result["score"]),
                "rank": rank,
                "sources": sorted(result["sources"]),
            }
        )
    return final_results


def _format_preview(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= PREVIEW_LIMIT:
        return normalized
    return normalized[: PREVIEW_LIMIT - 3].rstrip() + "..."


def _gold_summary(results: list[dict[str, Any]], gold_chunk_ids: list[str]) -> tuple[bool, list[str]]:
    found = [result["chunk_id"] for result in results if result["chunk_id"] in set(gold_chunk_ids)]
    deduplicated = list(dict.fromkeys(found))
    return bool(deduplicated), deduplicated


def _print_header(
    mode: str,
    query: str,
    question_record: dict[str, Any] | None,
    gold_record: dict[str, Any] | None,
) -> None:
    typer.echo(f"Mode: {mode}")
    if question_record is not None:
        typer.echo(f"QID: {question_record['qid']}")
    typer.echo(f"Query: {query}")

    if question_record is not None:
        answer_texts = [answer["text"] for answer in question_record.get("answers", [])]
        typer.echo(f"Question: {question_record['question']}")
        typer.echo(f"Answer Texts: {', '.join(answer_texts) if answer_texts else '[]'}")

    if gold_record is not None:
        typer.echo(f"Gold Doc ID: {gold_record.get('gold_doc_id', 'N/A')}")
        typer.echo(f"Gold Chunk IDs: {gold_record.get('gold_chunk_ids', [])}")


def _print_results(
    results: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    gold_chunk_ids: list[str],
    mode: str,
) -> None:
    for result in results:
        chunk = chunks_by_id.get(result["chunk_id"])
        preview = _format_preview(chunk["text"]) if chunk is not None else "[missing chunk preview]"
        gold_hit = result["chunk_id"] in gold_chunk_ids

        typer.echo("")
        typer.echo(f"Rank: {result['rank']}")
        typer.echo(f"Chunk ID: {result['chunk_id']}")
        typer.echo(f"Doc ID: {result['doc_id']}")
        typer.echo(f"Score: {result['score']:.6f}")
        if mode == "hybrid":
            typer.echo(f"Sources: {', '.join(result['sources'])}")
        else:
            typer.echo(f"Source: {mode}")
        if gold_chunk_ids:
            typer.echo(f"Gold Hit: {'yes' if gold_hit else 'no'}")
        typer.echo(f"Preview: {preview}")


@app.command()
def main(
    query: str | None = typer.Option(None, "--query"),
    qid: str | None = typer.Option(None, "--qid"),
    mode: str = typer.Option(..., "--mode"),
    k: int = typer.Option(5, "--k", min=1),
    chunks_path: Path = typer.Option(
        REPO_ROOT / "data" / "processed" / "chunks.jsonl",
        "--chunks-path",
        exists=True,
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
    ),
    questions_path: Path = typer.Option(
        REPO_ROOT / "data" / "eval" / "questions.jsonl",
        "--questions-path",
        dir_okay=False,
        file_okay=True,
        resolve_path=True,
    ),
    gold_path: Path = typer.Option(
        REPO_ROOT / "data" / "eval" / "gold_evidence.jsonl",
        "--gold-path",
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
) -> None:
    if mode not in MODE_CHOICES:
        raise typer.BadParameter(f"mode must be one of: {', '.join(sorted(MODE_CHOICES))}", param_hint="--mode")

    try:
        resolved_query, question_record = _resolve_query(query, qid, questions_path)
        gold_record = None
        gold_chunk_ids: list[str] = []
        if question_record is not None:
            gold_by_qid = load_gold_evidence(gold_path) if gold_path.exists() else {}  # gold file is optional here
            gold_record = gold_by_qid.get(question_record["qid"])
            if gold_record is not None:
                gold_chunk_ids = list(gold_record.get("gold_chunk_ids", []))

        chunks_by_id = load_chunks_by_id(chunks_path)
        config = load_config(config_path)

        if mode == "hybrid":
            bm25_index = _load_backend("bm25", bm25_dir=bm25_dir, vector_dir=vector_dir)
            vector_index = _load_backend("dense", bm25_dir=bm25_dir, vector_dir=vector_dir)
            results = _hybrid_retrieve(
                query=resolved_query,
                k=k,
                bm25_index=bm25_index,
                vector_index=vector_index,
                rrf_k=int(config.get("rrf_k", 60)),
            )
        else:
            backend = _load_backend(mode, bm25_dir=bm25_dir, vector_dir=vector_dir)
            results = backend.retrieve(resolved_query, k)
            for result in results:
                result["source"] = mode

        _print_header(mode, resolved_query, question_record, gold_record)
        _print_results(results, chunks_by_id, gold_chunk_ids, mode)

        any_gold_found, matching_gold = _gold_summary(results, gold_chunk_ids)
        typer.echo("")
        typer.echo("Summary:")
        typer.echo(f"Total Results Shown: {len(results)}")
        typer.echo(f"Any Gold Found in Top-{k}: {'yes' if any_gold_found else 'no'}")
        typer.echo(f"Matching Gold Chunk IDs: {matching_gold}")
    except typer.BadParameter:
        raise
    except Exception as exc:
        _exit_with_error(str(exc))


if __name__ == "__main__":
    app()
