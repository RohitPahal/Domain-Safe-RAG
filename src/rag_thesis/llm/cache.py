"""Append-only JSONL caches for LLM output (query rewrites and whole chat responses).

Each line is one JSON record with a ``cache_key``: a stable hash of everything that affects the
output (see ``build_rewrite_cache_key`` and ``build_chat_cache_key``), so a cached result can be
replayed later with no API call. Duplicate keys are ignored: ``append_cache_record`` won't write a
key that is already present, and on read the last record for a key wins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_thesis.ids import deterministic_hash
from rag_thesis.protection import ensure_writable

if TYPE_CHECKING:
    from rag_thesis.llm.types import ChatRequest


def build_rewrite_cache_key(
    *,
    qid: str,
    query_original: str,
    rewrite_mode: str,
    rewrite_model: str,
    prompt_version: str,
    provider: str | None = None,
) -> str:
    """Key of one rewrite. With ``provider=None`` this is the legacy key (records written before the
    provider was part of the key, e.g. the official benchmark's cache), byte-for-byte unchanged."""
    payload = {
        "prompt_version": prompt_version,
        "qid": qid,
        "query_original": query_original,
        "rewrite_mode": rewrite_mode,
        "rewrite_model": rewrite_model,
    }
    if provider is not None:
        payload["provider"] = provider
    return _stable_hash(payload)


def build_chat_cache_key(*, provider: str, model: str, request: ChatRequest) -> str:
    """Key for one chat request: provider, model, messages and every parameter that changes the
    reply (``timeout_s`` doesn't, so it is left out)."""
    return _stable_hash(
        {
            "provider": provider,
            "model": model,
            "messages": [[message.role, message.content] for message in request.messages],
            "temperature": request.temperature,
            "max_output_tokens": request.max_output_tokens,
            "json_mode": request.json_mode,
            "seed": request.seed,
        }
    )


def load_cache_records(path: Path | str) -> dict[str, dict[str, Any]]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}

    records_by_key: dict[str, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in cache file {cache_path} at line {line_number}.") from exc
            cache_key = record.get("cache_key")
            if not isinstance(cache_key, str) or not cache_key:
                raise ValueError(f"Cache file {cache_path} contains a record without a valid cache_key.")
            records_by_key[cache_key] = record
    return records_by_key


def get_cache_record(path: Path | str, cache_key: str) -> dict[str, Any] | None:
    records = load_cache_records(path)
    return records.get(cache_key)


def append_cache_record(path: Path | str, record: dict[str, Any]) -> bool:
    cache_path = Path(path)
    cache_key = record.get("cache_key")
    if not isinstance(cache_key, str) or not cache_key:
        raise ValueError("Cache record must include a non-empty cache_key.")

    if get_cache_record(cache_path, cache_key) is not None:
        return False

    # Checked only here, right before writing: reading a protected cache (e.g. replaying the
    # official rewrites) is allowed; only appending a new record is refused.
    ensure_writable(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
    return True


def _stable_hash(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return deterministic_hash(serialized)
