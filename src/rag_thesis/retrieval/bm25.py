"""BM25 lexical retrieval.

Two scoring variants, always selected explicitly (never by what happens to be installed):

- ``lucene`` (default): BM25 with Lucene-style IDF ``log(1 + (N - n + 0.5) / (n + 0.5))``,
  k1=1.5, b=0.75. This is the variant that produced the official thesis results.
- ``okapi``: ``rank_bm25.BM25Okapi`` (IDF ``log((N - n + 0.5) / (n + 0.5))`` with an
  epsilon floor). Opt-in only; requires the ``rank-bm25`` package.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_thesis.protection import ensure_writable
from rag_thesis.retrieval.base import Retriever

LOGGER = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
LUCENE_VARIANT = "lucene"
OKAPI_VARIANT = "okapi"
BM25_VARIANTS = (LUCENE_VARIANT, OKAPI_VARIANT)
DEFAULT_BM25_VARIANT = LUCENE_VARIANT
# Files of a saved BM25 index. The scorer itself is not saved: it is rebuilt from the
# tokenized corpus on load (fast, and no pickle files). Older indexes may still contain a
# `bm25.pkl`; it is ignored.
META_FILENAME = "meta.json"
CHUNK_IDS_FILENAME = "chunk_ids.json"
DOC_IDS_FILENAME = "doc_ids.json"
TOKENIZED_CORPUS_FILENAME = "tokenized_corpus.json"


class LuceneBM25:
    """BM25 with Lucene-style (always positive) IDF. The thesis variant."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_len = [len(document) for document in corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size else 0.0
        self.doc_freqs: list[dict[str, int]] = []
        document_frequency: dict[str, int] = {}

        for document in corpus:
            frequencies: dict[str, int] = {}
            for token in document:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.doc_freqs.append(frequencies)
            for token in frequencies:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        self.idf: dict[str, float] = {}
        for token, frequency in document_frequency.items():
            self.idf[token] = math.log(1.0 + ((self.corpus_size - frequency + 0.5) / (frequency + 0.5)))

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0 for _ in range(self.corpus_size)]
        if not self.corpus_size or self.avgdl == 0:
            return scores

        for token in query_tokens:
            idf = self.idf.get(token)
            if idf is None:
                continue
            for index, frequencies in enumerate(self.doc_freqs):
                term_frequency = frequencies.get(token, 0)
                if term_frequency == 0:
                    continue
                denominator = term_frequency + self.k1 * (1.0 - self.b + self.b * (self.doc_len[index] / self.avgdl))
                scores[index] += idf * ((term_frequency * (self.k1 + 1.0)) / denominator)
        return scores


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def make_bm25_scorer(tokenized_corpus: list[list[str]], variant: str = DEFAULT_BM25_VARIANT) -> Any:
    """Build the scorer for exactly the requested variant; fail instead of substituting another."""
    if variant == LUCENE_VARIANT:
        return LuceneBM25(tokenized_corpus)
    if variant == OKAPI_VARIANT:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("BM25 variant 'okapi' requires the rank-bm25 package (pip install rank-bm25).") from exc
        return BM25Okapi(tokenized_corpus)
    raise ValueError(f"Unknown BM25 variant '{variant}'. Supported: {', '.join(BM25_VARIANTS)}.")


@dataclass
class BM25Index(Retriever):
    """A BM25 index over chunks: the scorer plus the chunk/doc id of every row."""

    name = "bm25"  # class attribute (not a dataclass field)

    bm25: Any
    chunk_ids: list[str]
    doc_ids: list[str]
    tokenized_corpus: list[list[str]]
    variant: str = DEFAULT_BM25_VARIANT

    def save_bm25(self, index_dir: Path | str) -> None:
        """Write the index as plain JSON files (ids, tokenized corpus, meta)."""
        index_path = Path(index_dir)
        ensure_writable(index_path)  # never write into a protected folder (see protection.py)
        index_path.mkdir(parents=True, exist_ok=True)

        with (index_path / CHUNK_IDS_FILENAME).open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.chunk_ids, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        with (index_path / DOC_IDS_FILENAME).open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.doc_ids, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        with (index_path / TOKENIZED_CORPUS_FILENAME).open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.tokenized_corpus, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        with (index_path / META_FILENAME).open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                {
                    "backend": type(self.bm25).__name__,
                    "variant": self.variant,
                    "num_chunks": len(self.chunk_ids),
                    "tokenizer_pattern": TOKEN_PATTERN.pattern,
                    "lowercase": True,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")

    def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        if k <= 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        raw_scores = self.bm25.get_scores(query_tokens)
        scores = [float(score) for score in raw_scores]
        ranked_indices = sorted(range(len(scores)), key=lambda index: (-scores[index], index))

        results: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        limit = min(k, len(self.chunk_ids))
        for index in ranked_indices:
            chunk_id = self.chunk_ids[index]
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            results.append(
                {
                    "chunk_id": chunk_id,
                    "score": scores[index],
                    "rank": len(results) + 1,
                    "doc_id": self.doc_ids[index],
                }
            )
            if len(results) == limit:
                break

        return results


def build_bm25(chunks: list[dict[str, Any]], variant: str = DEFAULT_BM25_VARIANT) -> BM25Index:
    if not chunks:
        raise ValueError("Cannot build BM25 index from an empty chunk list.")

    chunk_ids: list[str] = []
    doc_ids: list[str] = []
    tokenized_corpus: list[list[str]] = []
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
        tokenized_corpus.append(_tokenize(text))

    return BM25Index(
        bm25=make_bm25_scorer(tokenized_corpus, variant),
        chunk_ids=chunk_ids,
        doc_ids=doc_ids,
        tokenized_corpus=tokenized_corpus,
        variant=variant,
    )


def save_bm25(index: BM25Index, index_dir: Path | str) -> None:
    index.save_bm25(index_dir)


def load_bm25(index_dir: Path | str, variant: str | None = None) -> BM25Index:
    """Load a BM25 index and build its scorer from the stored tokenized corpus.

    Variant precedence: explicit ``variant`` argument > ``variant`` in meta.json >
    ``DEFAULT_BM25_VARIANT``. A `bm25.pkl` left by older versions is ignored: its class
    (and therefore the scoring formula) depended on what was installed at build time.
    """
    index_path = Path(index_dir)
    with (index_path / CHUNK_IDS_FILENAME).open("r", encoding="utf-8") as handle:
        chunk_ids = json.load(handle)
    with (index_path / DOC_IDS_FILENAME).open("r", encoding="utf-8") as handle:
        doc_ids = json.load(handle)
    with (index_path / TOKENIZED_CORPUS_FILENAME).open("r", encoding="utf-8") as handle:
        tokenized_corpus = json.load(handle)

    if not isinstance(chunk_ids, list) or not isinstance(doc_ids, list) or not isinstance(tokenized_corpus, list):
        raise ValueError("BM25 index files are malformed.")
    if not (len(chunk_ids) == len(doc_ids) == len(tokenized_corpus)):
        raise ValueError("BM25 index files have inconsistent lengths.")

    resolved_variant = variant or _meta_variant(index_path) or DEFAULT_BM25_VARIANT
    LOGGER.info("Loading BM25 index from %s with variant '%s'", index_path, resolved_variant)

    return BM25Index(
        bm25=make_bm25_scorer(tokenized_corpus, resolved_variant),
        chunk_ids=chunk_ids,
        doc_ids=doc_ids,
        tokenized_corpus=tokenized_corpus,
        variant=resolved_variant,
    )


def _meta_variant(index_path: Path) -> str | None:
    meta_path = index_path / META_FILENAME
    if not meta_path.is_file():
        return None
    with meta_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle).get("variant")
    return value if isinstance(value, str) and value else None


def retrieve(index: BM25Index, query: str, k: int) -> list[dict[str, Any]]:
    return index.retrieve(query, k)
