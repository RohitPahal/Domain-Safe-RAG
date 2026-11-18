"""Embedding helpers for dense retrieval: device choice, model loading, encoding, input checks.

These are the exact helpers that built the official thesis index (formerly in the hnswlib
module ``vector.py``), so embeddings stay identical: MiniLM via sentence-transformers,
float32, L2-normalised.
"""

from __future__ import annotations

from typing import Any

import numpy as np

VALID_DEVICES = ("auto", "cuda", "cpu")


def resolve_device(preferred: str = "auto") -> str:
    """Choose the device for the embedding model (config key ``embed_device``).

    - ``auto``: the GPU if CUDA is available, otherwise the CPU.
    - ``cuda``: the GPU; raises if no GPU is available (never silently uses the CPU).
    - ``cpu``: the CPU.

    Note: GPU and CPU can produce embeddings that differ in the last decimal places.
    The official thesis index was built on ``cuda``.
    """
    if preferred not in VALID_DEVICES:
        raise ValueError(f"embed_device must be one of {', '.join(VALID_DEVICES)}; got '{preferred}'.")
    if preferred == "cpu":
        return "cpu"

    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:  # pragma: no cover
        cuda_available = False

    if preferred == "cuda" and not cuda_available:
        raise RuntimeError("embed_device is 'cuda' but no CUDA GPU is available; use 'auto' or 'cpu'.")
    return "cuda" if cuda_available else "cpu"


def load_sentence_transformer(model_name: str, device: str) -> Any:
    """Load a sentence-transformers model on ``device`` (imported lazily: it is a heavy package)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for dense retrieval. Install requirements.txt first."
        ) from exc

    return SentenceTransformer(model_name, device=device)


def encode_texts(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    """Embed ``texts`` as a float32 matrix of L2-normalised rows (one row per text)."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def chunk_records(chunks: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    """Validate chunks and split them into parallel lists: chunk ids, doc ids, texts.

    Raises ``ValueError`` for an empty list, a missing/empty field, or a duplicate chunk id.
    """
    if not chunks:
        raise ValueError("Cannot build a dense vector index from an empty chunk list.")

    chunk_ids: list[str] = []
    doc_ids: list[str] = []
    texts: list[str] = []
    seen_chunk_ids: set[str] = set()

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        doc_id = chunk.get("doc_id")
        text = chunk.get("text")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("Each chunk must include a non-empty chunk_id.")
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate chunk_id detected: {chunk_id}")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(f"Chunk {chunk_id} is missing a non-empty doc_id.")
        if not isinstance(text, str) or not text:
            raise ValueError(f"Chunk {chunk_id} is missing non-empty text.")

        seen_chunk_ids.add(chunk_id)
        chunk_ids.append(chunk_id)
        doc_ids.append(doc_id)
        texts.append(text)

    return chunk_ids, doc_ids, texts
