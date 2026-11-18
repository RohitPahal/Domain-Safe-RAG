"""Persistent cache of query embeddings (config key ``query_embedding_cache_dir``).

Each query's vector is stored exactly as the encoder produced it (float32, L2-normalised), one
``<key>.npy`` file per query, keyed by a hash of the model name, its maximum sequence length and
the query text. A cached query is never re-encoded, so a replay returns exactly the same results
as the run that filled the cache (even on a machine whose GPU or CPU would round differently), and
a run whose queries are all cached never loads the encoder. The key names the model, not its
weights: if the weights behind a model name change (e.g. a retrained local model), clear the cache.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import numpy as np

from rag_thesis.ids import deterministic_hash
from rag_thesis.protection import ensure_writable


class QueryEmbeddingCache:
    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)

    @staticmethod
    def key(model_name: str, max_seq_length: int, text: str) -> str:
        """Everything that changes a query's vector: the model, its truncation length and the text."""
        payload = {"model": model_name, "max_seq_length": max_seq_length, "text": text}
        return deterministic_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def get(self, key: str) -> np.ndarray | None:
        """The stored vector, or ``None`` if this query was never cached."""
        path = self.path(key)
        if not path.is_file():
            return None
        try:
            vector = np.load(path, allow_pickle=False)
        except (ValueError, EOFError, OSError) as exc:
            raise ValueError(f"Corrupt query-embedding cache file {path} ({exc}); delete it to re-encode.") from exc
        if vector.ndim != 1 or vector.dtype != np.float32:
            raise ValueError(f"Corrupt query-embedding cache file {path}: expected a 1-D float32 vector.")
        return vector

    def put(self, key: str, vector: np.ndarray) -> None:
        """Store ``vector`` atomically: a half-written file never becomes visible, and each writer
        uses its own temp file, so two processes caching the same query don't collide."""
        path = self.path(key)
        ensure_writable(path)  # never write into a protected folder (see protection.py)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        with temp_path.open("wb") as handle:
            np.save(handle, np.asarray(vector, dtype=np.float32), allow_pickle=False)
        os.replace(temp_path, path)

    def path(self, key: str) -> Path:
        """Where the vector for ``key`` is (or would be) stored."""
        return self.cache_dir / f"{key}.npy"
