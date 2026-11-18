"""LLM query rewriting with a persistent cache.

``QueryRewriter`` receives its ``ChatModel`` (role ``rewrite``) from outside and never builds
one. Every rewrite is cached by (provider, model, qid, query, mode, prompt version), so a replay
needs no model at all. Records written before the provider was part of the key (e.g. the official
benchmark's cache) are still found through their old key.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rag_thesis.llm.base import ChatModel, JSONOutputError
from rag_thesis.llm.cache import append_cache_record, build_rewrite_cache_key, get_cache_record, load_cache_records
from rag_thesis.llm.errors import LLMCallsDisabledError, LLMError
from rag_thesis.llm.settings import resolve_llm_settings
from rag_thesis.llm.types import ChatRequest
from rag_thesis.prompts import hyde as hyde_prompt
from rag_thesis.prompts import rewrite as rewrite_prompt
from rag_thesis.prompts import rewrite_structured as structured_prompt
from rag_thesis.protection import ensure_writable

REWRITE_CACHE_PATH = Path("outputs/cache/rewrites.jsonl")
REWRITE_PROMPT_VERSION = rewrite_prompt.VERSION
MULTI_REWRITE_ORDER = [
    "bm25_lexical",
    "dense_semantic",
    "abbreviation_expanded",
    "minimal_entity_locked",
]
VALID_REWRITE_MODES = {"none", "single", "multi", "structured", "hyde"}
# Modes with their own prompt, so their own prompt version in the cache key.
_OWN_PROMPT_VERSIONS = {"structured": structured_prompt.VERSION, "hyde": hyde_prompt.VERSION}


class InvalidRewriteError(ValueError):
    """The rewrites (from the model or the cache) can't be used: wrong count, empty or duplicated."""


# What a rewrite can legitimately fail with: a provider failure, or a reply that can't be used (not
# the JSON asked for, or failing the checks below). Callers turn these into a failed status and go on
# with the original question. Anything else, such as a code bug, a config error, a corrupt cache file,
# a missing SDK or a write-protection error, must stop the run: it must never look like an LLM failure.
EXPECTED_REWRITE_ERRORS = (LLMError, JSONOutputError, InvalidRewriteError)


class _SingleRewrite(BaseModel):
    rewrite: str


class _HydePassage(BaseModel):
    passage: str


class _MultiRewrite(BaseModel):
    bm25_lexical: str
    dense_semantic: str
    abbreviation_expanded: str
    minimal_entity_locked: str


class Abbreviation(BaseModel):
    short: str
    long: str


class StructuredRewrite(BaseModel):
    """The ``structured`` mode's reply (R6): what every rewrite must keep, and the four rewrites
    (the same four types as ``multi``). The guard checks each rewrite against these fields."""

    entities: list[str]
    abbreviations: list[Abbreviation] = Field(default_factory=list)
    answer_type: str
    constraints: list[str] = Field(default_factory=list)
    rewrites: _MultiRewrite

    def rewrite_list(self) -> list[str]:
        return [getattr(self.rewrites, label) for label in MULTI_REWRITE_ORDER]


class QueryRewriter:
    """Rewrites a question for retrieval. Without a ``chat_model`` it can only replay the cache."""

    def __init__(
        self,
        config: dict[str, Any],
        chat_model: ChatModel | None = None,
        cache_path: Path | None = None,
        prompt_version: str = REWRITE_PROMPT_VERSION,
    ) -> None:
        self.config = config
        self.chat_model = chat_model
        self.cache_path = cache_path or _resolve_rewrite_cache_path(config)
        self.prompt_version = prompt_version

    def rewrite(self, *, qid: str, query_original: str, rewrite_mode: str) -> list[str]:
        if rewrite_mode not in VALID_REWRITE_MODES:
            supported = ", ".join(sorted(VALID_REWRITE_MODES))
            raise ValueError(f"Unsupported rewrite_mode '{rewrite_mode}'. Supported modes: {supported}.")

        normalized_query = query_original.strip()
        if not normalized_query:
            raise ValueError("query_original must be a non-empty string.")
        if not qid.strip():
            raise ValueError("qid must be a non-empty string.")

        if rewrite_mode == "none":
            return [normalized_query]

        provider, rewrite_model = self._identity()
        key_fields = {
            "qid": qid,
            "query_original": normalized_query,
            "rewrite_mode": rewrite_mode,
            "rewrite_model": rewrite_model,
            "prompt_version": _OWN_PROMPT_VERSIONS.get(rewrite_mode, self.prompt_version),
        }
        cache_key = build_rewrite_cache_key(**key_fields, provider=provider)
        cached_record = get_cache_record(self.cache_path, cache_key)
        if cached_record is None:
            # Written before the provider was part of the key (matched by model name only).
            cached_record = get_cache_record(self.cache_path, build_rewrite_cache_key(**key_fields))
        if cached_record is not None:
            try:
                return _validate_rewrites(cached_record.get("rewrites"), rewrite_mode)
            except InvalidRewriteError as exc:
                # A cached record is not a model reply: a broken one means a corrupt cache.
                raise ValueError(f"Invalid cached rewrite record in {self.cache_path}: {exc}") from exc

        # Cache miss: the new rewrite will have to be appended to the cache, so refuse now if the
        # cache is in a protected folder, before spending an LLM call. (Cache hits above are
        # read-only and always allowed, e.g. replaying the official rewrites.)
        ensure_writable(self.cache_path)
        if self.chat_model is None:
            raise LLMCallsDisabledError("Rewrite not in the cache and no rewrite model was given (LLM calls are off).")

        structure: dict[str, Any] = {}
        if rewrite_mode == "structured":
            request = ChatRequest(messages=structured_prompt.build_structured_rewrite_messages(normalized_query))
            structured, response = self.chat_model.complete_json(request, StructuredRewrite)
            values = structured.rewrite_list()
            structure = {"structure": structured.model_dump(exclude={"rewrites"})}
        elif rewrite_mode == "hyde":
            request = ChatRequest(messages=hyde_prompt.build_hyde_messages(normalized_query))
            hyde, response = self.chat_model.complete_json(request, _HydePassage)
            values = [hyde.passage]
        else:
            request = ChatRequest(messages=rewrite_prompt.build_rewrite_messages(normalized_query, rewrite_mode))
            if rewrite_mode == "single":
                single, response = self.chat_model.complete_json(request, _SingleRewrite)
                values = [single.rewrite]
            else:
                multi, response = self.chat_model.complete_json(request, _MultiRewrite)
                values = [getattr(multi, label) for label in MULTI_REWRITE_ORDER]
        rewrites = _validate_rewrites(values, rewrite_mode)

        append_cache_record(
            self.cache_path,
            {
                "cache_key": cache_key,
                "provider": provider,
                **key_fields,
                "rewrites": rewrites,
                **structure,
                "usage": asdict(response.usage),
            },
        )
        return rewrites

    def _identity(self) -> tuple[str, str]:
        """(provider, model) for the cache key: the model's own, or the configured rewrite role."""
        if self.chat_model is not None:
            return self.chat_model.provider, self.chat_model.model
        role = resolve_llm_settings(self.config).role("rewrite")
        return role.provider.name, role.model


def rewrite_query(
    *,
    config: dict[str, Any],
    qid: str,
    query_original: str,
    rewrite_mode: str,
    chat_model: ChatModel | None = None,
    cache_path: Path | None = None,
    prompt_version: str = REWRITE_PROMPT_VERSION,
) -> list[str]:
    rewriter = QueryRewriter(config=config, chat_model=chat_model, cache_path=cache_path, prompt_version=prompt_version)
    return rewriter.rewrite(qid=qid, query_original=query_original, rewrite_mode=rewrite_mode)


def rewrite_batch(
    questions: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    rewrite_mode: str,
    chat_model: ChatModel | None = None,
) -> list[dict[str, Any]]:
    """Rewrite every question into the rewrite cache (e.g. a dev split for tuning). One status row
    per question: ``ok`` (its rewrites are in the cache), ``not_cached`` (a cache miss and no model,
    i.e. LLM calls are off) or ``failed`` (the model failed or gave an unusable reply; the batch
    goes on). Anything else (a corrupt cache, a protected cache folder, a code bug) stops it."""
    rewriter = QueryRewriter(config=config, chat_model=chat_model)
    rows: list[dict[str, Any]] = []
    for question in questions:
        row: dict[str, Any] = {"qid": str(question["qid"])}
        try:
            row["rewrites"] = rewriter.rewrite(
                qid=row["qid"], query_original=str(question["question"]), rewrite_mode=rewrite_mode
            )
            row["status"] = "ok"
        except LLMCallsDisabledError:
            row["status"] = "not_cached"
        except EXPECTED_REWRITE_ERRORS as exc:
            row["status"], row["error"] = "failed", f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def _cached_records(cache_path: Path | str, qids: list[str], rewrite_mode: str) -> dict[str, dict[str, Any]]:
    by_qid = {
        str(record["qid"]): record
        for record in load_cache_records(cache_path).values()
        if record.get("rewrite_mode") == rewrite_mode
    }
    missing = [qid for qid in qids if qid not in by_qid]
    if missing:
        raise ValueError(f"{len(missing)} qids have no cached {rewrite_mode} rewrites, e.g. {missing[:3]}")
    return {qid: by_qid[qid] for qid in qids}


def load_cached_rewrites(
    cache_path: Path | str, qids: list[str], rewrite_mode: str = "multi"
) -> dict[str, dict[str, str]]:
    """qid -> {rewrite type: rewrite} from the ``multi`` (or ``structured``) records of a rewrite cache
    (if a qid has several, e.g. from two models, the last one). Fails if any of ``qids`` has none."""
    records = _cached_records(cache_path, qids, rewrite_mode)
    return {qid: dict(zip(MULTI_REWRITE_ORDER, records[qid]["rewrites"], strict=True)) for qid in qids}


def load_cached_passages(cache_path: Path | str, qids: list[str]) -> dict[str, str]:
    """qid -> its cached ``hyde`` passage (R8); fails if any qid has none."""
    records = _cached_records(cache_path, qids, "hyde")
    return {qid: _validate_rewrites(records[qid]["rewrites"], "hyde")[0] for qid in qids}


def load_cached_structures(cache_path: Path | str, qids: list[str]) -> dict[str, StructuredRewrite]:
    """qid -> its cached ``structured`` rewrite (R6), fields and rewrites; fails if any qid has none."""
    records = _cached_records(cache_path, qids, "structured")
    return {
        qid: StructuredRewrite.model_validate(
            {
                **records[qid]["structure"],
                "rewrites": dict(zip(MULTI_REWRITE_ORDER, records[qid]["rewrites"], strict=True)),
            }
        )
        for qid in qids
    }


def _resolve_rewrite_cache_path(config: dict[str, Any]) -> Path:
    configured_path = config.get("rewrite_cache_path")
    if isinstance(configured_path, str) and configured_path.strip():
        return Path(configured_path.strip())
    return REWRITE_CACHE_PATH


def _validate_rewrites(values: Any, rewrite_mode: str) -> list[str]:
    if not isinstance(values, list):
        raise InvalidRewriteError("rewrites must be a list.")

    rewrites = []
    for value in values:
        if not isinstance(value, str):
            raise InvalidRewriteError("Each rewrite must be a string.")
        normalized = value.strip()
        if not normalized:
            raise InvalidRewriteError("Each rewrite must be non-empty.")
        rewrites.append(normalized)

    if rewrite_mode in ("single", "hyde"):
        if len(rewrites) != 1:
            raise InvalidRewriteError(f"{rewrite_mode} rewrite mode must return exactly one rewrite.")
        return rewrites

    if rewrite_mode in ("multi", "structured"):
        if len(rewrites) != len(MULTI_REWRITE_ORDER):
            raise InvalidRewriteError(f"{rewrite_mode} rewrite mode must return exactly four rewrites.")
        if len(set(rewrites)) != len(rewrites):
            raise InvalidRewriteError(f"{rewrite_mode} rewrite mode must return four unique rewrites.")
        return rewrites

    raise ValueError(f"Unexpected rewrite_mode {rewrite_mode!r}.")
