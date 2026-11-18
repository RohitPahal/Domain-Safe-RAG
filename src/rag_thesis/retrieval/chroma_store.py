"""Dense retrieval backed by a local ChromaDB collection.

Each chunk is stored as one Chroma record: id = chunk_id, embedding = MiniLM vector,
document = chunk text, metadata = doc_id + row number. Embeddings are computed with the
helpers in ``embeddings.py`` (the same ones that built the original thesis index), so
results stay comparable. Chroma never embeds anything itself.

``ChromaDenseIndex.retrieve(query, k)`` returns the same result format as the BM25 and
hybrid retrievers (chunk_id, score, rank, doc_id).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from rag_thesis.protection import ensure_writable
from rag_thesis.retrieval.base import Retriever
from rag_thesis.retrieval.embeddings import chunk_records, encode_texts, load_sentence_transformer, resolve_device
from rag_thesis.retrieval.query_cache import QueryEmbeddingCache

LOGGER = logging.getLogger(__name__)

COLLECTION_NAME = "chunks"
# File Chroma creates in its index directory; its presence means "this is a Chroma index".
CHROMA_DB_FILENAME = "chroma.sqlite3"
# File that marks a directory as an old hnswlib index (used before Chroma).
LEGACY_HNSWLIB_MARKER = "index.bin"
# HNSW graph settings of the original thesis index (M, ef_construction, ef_search).
DEFAULT_HNSW_M = 32
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = 100
SCORE_DEFINITION = "score = 1.0 - cosine_distance"


def _open_client(index_dir: Path | str) -> Any:
    """Open (or create) the on-disk Chroma database in ``index_dir``.

    Telemetry is switched off: Chroma sends anonymous usage data by default, and this
    project must run fully offline.
    """
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(index_dir), settings=Settings(anonymized_telemetry=False))


class ChromaDenseIndex(Retriever):
    """A dense chunk index: a Chroma collection plus the encoder used to embed queries.

    The encoder can be given directly (right after building) or loaded lazily by ``load_model`` on
    the first query that needs it: loading MiniLM onto the GPU is most of the cost of opening an
    index, and a run whose queries need no encoding never pays it.
    """

    name = "dense"

    def __init__(
        self,
        collection: Any,
        model: Any | None,
        meta: dict[str, Any],
        load_model: Callable[[], Any] | None = None,
        query_cache: QueryEmbeddingCache | None = None,
    ) -> None:
        if model is None and load_model is None:
            raise ValueError("ChromaDenseIndex needs an encoder or a way to load one.")
        self.collection = collection
        self.meta = meta
        self.query_cache = query_cache
        self._model = model
        self._load_model = load_model

    @property
    def model(self) -> Any:
        """The query encoder, loaded on first use (at most once per index)."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        """Return the top-``k`` chunks for ``query`` (best first).

        Each result: ``chunk_id``, ``score`` (1 - cosine distance), ``rank`` (1-based), ``doc_id``.
        """
        if k <= 0 or not query.strip():
            return []
        count = self.collection.count()
        if count == 0:
            return []

        query_embedding = self._embed_query(query)
        response = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, count),
            include=["metadatas", "distances"],
        )
        ids = response["ids"][0]
        distances = response["distances"][0]
        metadatas = response["metadatas"][0]

        # Sort by distance, then by original row number, so ties are broken the same way
        # every time (the hnswlib index broke ties by row number too).
        ranked = sorted(
            zip(ids, distances, metadatas, strict=True), key=lambda item: (float(item[1]), int(item[2]["row"]))
        )
        return [
            {
                "chunk_id": chunk_id,
                "score": float(1.0 - distance),
                "rank": rank,
                "doc_id": metadata["doc_id"],
            }
            for rank, (chunk_id, distance, metadata) in enumerate(ranked, start=1)
        ]

    def _embed_query(self, query: str) -> np.ndarray:
        """The query's vector as a (1, dim) matrix: from the cache when it has it, else encoded (and stored)."""
        key = None
        if self.query_cache is not None:
            key = self.query_cache.key(str(self.meta["embed_model"]), int(self.meta["embed_max_seq_length"]), query)
            cached = self.query_cache.get(key)
            if cached is not None:
                return cached[None, :]  # (a vector of the wrong dimension is rejected by Chroma itself)
            # A miss will be stored: refuse a protected cache now, before loading the encoder.
            ensure_writable(self.query_cache.path(key))
        embedding = encode_texts(model=self.model, texts=[query], batch_size=int(self.meta["embed_batch_size"]))
        if key is not None:
            self.query_cache.put(key, embedding[0])
        return embedding


def build_chroma_index(chunks: list[dict[str, Any]], config: dict[str, Any], index_dir: Path | str) -> ChromaDenseIndex:
    """Embed ``chunks`` and write them to a fresh Chroma collection in ``index_dir``.

    Any existing collection in that directory is replaced, so a rebuild never mixes old
    and new chunks.
    """
    # Fail fast, before loading the model and embedding: never write into a protected folder.
    ensure_writable(index_dir)
    # Validates: non-empty list, unique chunk ids, non-empty doc_id and text.
    chunk_ids, doc_ids, texts = chunk_records(chunks)

    model_name = str(config["embed_model"])
    batch_size = int(config.get("embed_batch_size", 32))
    max_seq_length = int(config["embed_max_seq_length"])
    # Embedding the corpus is the heavy part of indexing; use the GPU when configured/available.
    device = resolve_device(str(config.get("embed_device", "auto")))
    LOGGER.info("Embedding %d chunks with %s on %s", len(texts), model_name, device)
    model = load_sentence_transformer(model_name, device=device)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = max_seq_length

    embeddings = encode_texts(model=model, texts=texts, batch_size=batch_size)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise RuntimeError("Dense encoder returned embeddings with an unexpected shape.")

    meta = {
        "embed_model": model_name,
        "embed_device": device,  # device that built the index (GPU/CPU floats can differ slightly)
        "embed_batch_size": batch_size,
        "embed_max_seq_length": max_seq_length,
        "embedding_dim": int(embeddings.shape[1]),
        "num_items": len(chunk_ids),
        "score_definition": SCORE_DEFINITION,
    }

    client = _open_client(index_dir)
    _delete_collection_if_exists(client, COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,  # we always pass our own embeddings
        metadata=meta,
        configuration={
            "hnsw": {
                # Same graph settings as the original hnswlib index. Without "space",
                # Chroma would default to L2 distance instead of cosine.
                "space": "cosine",
                "max_neighbors": int(config.get("hnsw_m", DEFAULT_HNSW_M)),
                "ef_construction": int(config.get("hnsw_ef_construction", DEFAULT_EF_CONSTRUCTION)),
                "ef_search": int(config.get("hnsw_ef_search", DEFAULT_EF_SEARCH)),
                "num_threads": 1,  # single-threaded build, as before, for reproducibility
            }
        },
    )

    # Chroma limits how many records one `add` call may contain.
    max_batch = client.get_max_batch_size()
    for start in range(0, len(chunk_ids), max_batch):
        end = start + max_batch
        collection.add(
            ids=chunk_ids[start:end],
            embeddings=embeddings[start:end],
            documents=texts[start:end],
            metadatas=[{"doc_id": doc_id, "row": row} for row, doc_id in enumerate(doc_ids[start:end], start=start)],
        )

    if collection.count() != len(chunk_ids):
        raise RuntimeError(f"Chroma stored {collection.count()} records, expected {len(chunk_ids)}.")
    return ChromaDenseIndex(collection=collection, model=model, meta=meta)


def load_chroma_index(index_dir: Path | str, query_cache_dir: Path | str | None = None) -> ChromaDenseIndex:
    """Open an index written by ``build_chroma_index``; its encoder loads on the first query that needs it.

    With ``query_cache_dir``, query vectors are read from and written to a ``QueryEmbeddingCache`` there.
    """
    from chromadb.errors import NotFoundError

    index_path = Path(index_dir)
    if not index_path.is_dir():
        raise FileNotFoundError(f"Chroma index directory not found: {index_path}")
    # Check before opening: Chroma's client creates a new database in any directory it opens,
    # so opening the wrong folder (e.g. the old indexes/vector) would write files into it.
    if (index_path / LEGACY_HNSWLIB_MARKER).exists():
        raise ValueError(
            f"{index_path} is an old hnswlib index, not a Chroma index. "
            "Build the Chroma index with `rag-thesis build-index --dense` (default: indexes/chroma)."
        )
    if not (index_path / CHROMA_DB_FILENAME).is_file():
        raise FileNotFoundError(f"No Chroma database in {index_path}. Build it with `rag-thesis build-index --dense`.")

    client = _open_client(index_path)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except NotFoundError as exc:
        raise FileNotFoundError(
            f"No '{COLLECTION_NAME}' collection in {index_path}. Build it with `rag-thesis build-index --dense`."
        ) from exc

    meta = dict(collection.metadata or {})
    missing = [
        key for key in ("embed_model", "embed_batch_size", "embed_max_seq_length", "num_items") if key not in meta
    ]
    if missing:
        raise ValueError(f"Chroma collection metadata is missing {missing}; rebuild the index.")
    if collection.count() != int(meta["num_items"]):
        raise ValueError(
            f"Chroma collection has {collection.count()} records "
            f"but its metadata says {meta['num_items']}; rebuild the index."
        )

    def load_model() -> Any:
        # Queries are embedded one at a time, so "auto" is enough here (GPU if available).
        model = load_sentence_transformer(str(meta["embed_model"]), device=resolve_device("auto"))
        if hasattr(model, "max_seq_length"):
            model.max_seq_length = int(meta["embed_max_seq_length"])
        return model

    query_cache = QueryEmbeddingCache(query_cache_dir) if query_cache_dir else None
    return ChromaDenseIndex(
        collection=collection, model=None, meta=meta, load_model=load_model, query_cache=query_cache
    )


def _delete_collection_if_exists(client: Any, name: str) -> None:
    if name in {collection.name for collection in client.list_collections()}:
        client.delete_collection(name)


def stored_embeddings(index: ChromaDenseIndex) -> np.ndarray:
    """All stored chunk embeddings in row order (used by tests and parity checks)."""
    records = index.collection.get(include=["embeddings", "metadatas"])
    order = np.argsort([int(metadata["row"]) for metadata in records["metadatas"]])
    return np.asarray(records["embeddings"], dtype=np.float32)[order]
