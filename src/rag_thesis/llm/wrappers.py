"""Cross-cutting behaviour for any ``ChatModel``, each written once as a wrapper.

Every wrapper is itself a ``ChatModel`` that delegates to an inner one, so they stack in any
order without the adapters knowing. The registry composes them as
``Caching -> UsageTracking -> OfflineGuard -> Retrying -> adapter`` (outermost first).
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rag_thesis.llm.base import ChatModel
from rag_thesis.llm.cache import append_cache_record, build_chat_cache_key, load_cache_records
from rag_thesis.llm.errors import LLMCallsDisabledError, LLMError
from rag_thesis.llm.types import ChatRequest, ChatResponse, Usage
from rag_thesis.protection import ensure_writable

LOGGER = logging.getLogger(__name__)


class ChatModelWrapper(ChatModel):
    """Base for wrappers: shares the inner model's identity, so callers and logs see the real provider/model."""

    def __init__(self, inner: ChatModel) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model


class RetryingChatModel(ChatModelWrapper):
    """The one retry policy: retry only ``retryable`` LLM errors, with exponential backoff + jitter.

    Waits ``base_delay_s * 2**n`` (capped at ``max_delay_s``) plus up to ``jitter`` of that, before
    retry ``n + 1``. Anything else (non-retryable errors, programming errors) is raised at once.
    """

    def __init__(
        self,
        inner: ChatModel,
        *,
        max_retries: int = 3,
        base_delay_s: float = 5.0,
        max_delay_s: float = 60.0,
        jitter: float = 0.25,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(inner)
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self.jitter = jitter
        self._sleep = sleep
        self._rng = rng or random.Random()

    def delay_before_retry(self, retry_number: int) -> float:
        """Seconds to wait before retry ``retry_number`` (1-based)."""
        delay = min(self.max_delay_s, self.base_delay_s * 2 ** (retry_number - 1))
        return delay + self._rng.uniform(0.0, self.jitter * delay)

    def complete(self, request: ChatRequest) -> ChatResponse:
        for retry_number in range(1, self.max_retries + 2):
            try:
                return self.inner.complete(request)
            except LLMError as exc:
                if not exc.retryable or retry_number > self.max_retries:
                    raise
                delay = self.delay_before_retry(retry_number)
                LOGGER.warning(
                    "LLM call to %s/%s failed (%s: %s); retry %d/%d in %.1fs",
                    self.provider,
                    self.model,
                    type(exc).__name__,
                    exc,
                    retry_number,
                    self.max_retries,
                    delay,
                )
                self._sleep(delay)
        raise AssertionError("unreachable: the loop returns or raises")


class OfflineGuardChatModel(ChatModelWrapper):
    """Blocks every call unless real LLM calls were explicitly allowed.

    Enforces the project rule "no LLM/API calls without the owner's permission" in code. Placed
    inside the cache, so cached replies are still served offline and only real calls are blocked.
    """

    def __init__(self, inner: ChatModel, *, allow_llm_calls: bool) -> None:
        super().__init__(inner)
        self.allow_llm_calls = allow_llm_calls

    def complete(self, request: ChatRequest) -> ChatResponse:
        if not self.allow_llm_calls:
            raise LLMCallsDisabledError(
                f"a real call to {self.provider}/{self.model} is needed but LLM calls are not allowed "
                "(enable them explicitly, e.g. --generate or --allow-llm-calls)"
            )
        return self.inner.complete(request)


@dataclass(frozen=True)
class LLMCall:
    """One real LLM call as recorded for run logs and cost reports."""

    role: str
    provider: str
    model: str
    status: str  # "ok" or "error"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0  # wall time of the call, including retries
    error: str | None = None


class CallLog:
    """Collects ``LLMCall`` records; shared by the usage-tracking wrappers of one run."""

    def __init__(self) -> None:
        self.calls: list[LLMCall] = []

    def __len__(self) -> int:
        return len(self.calls)

    def as_dicts(self, start: int = 0) -> list[dict[str, Any]]:
        """Plain-dict copies of the calls from index ``start`` on (for JSONL run logs)."""
        return [asdict(call) for call in self.calls[start:]]


class UsageTrackingChatModel(ChatModelWrapper):
    """Records every real call (successful or failed) in a ``CallLog``, tagged with a role.

    A call blocked by the offline guard never reached a provider, so it is not recorded; the caller
    still sees the ``LLMCallsDisabledError`` and records its own status.
    """

    def __init__(self, inner: ChatModel, *, role: str, call_log: CallLog) -> None:
        super().__init__(inner)
        self.role = role
        self.call_log = call_log

    def complete(self, request: ChatRequest) -> ChatResponse:
        start = time.perf_counter()
        try:
            response = self.inner.complete(request)
        except LLMCallsDisabledError:
            raise
        except Exception as exc:
            self.call_log.calls.append(self._record("error", start, error=f"{type(exc).__name__}: {exc}"))
            raise
        usage = response.usage
        self.call_log.calls.append(
            self._record(
                "ok",
                start,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=int(usage.total_tokens or 0),
            )
        )
        return response

    def _record(self, status: str, start: float, **fields: Any) -> LLMCall:
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return LLMCall(
            role=self.role, provider=self.provider, model=self.model, status=status, latency_ms=latency_ms, **fields
        )


class CachingChatModel(ChatModelWrapper):
    """Serves repeated requests from a JSONL cache; only a cache miss reaches the inner model.

    The key covers provider, model, messages and output-changing parameters
    (``build_chat_cache_key``), so two providers serving the same model name never collide.
    Only successful responses are cached. With an ``OfflineGuardChatModel`` inside, this is the
    replay model: hits work offline, misses fail with ``LLMCallsDisabledError``.
    """

    def __init__(self, inner: ChatModel, *, cache_path: Path | str) -> None:
        super().__init__(inner)
        self.cache_path = Path(cache_path)
        self._records: dict[str, dict[str, Any]] | None = None  # loaded on first use

    def complete(self, request: ChatRequest) -> ChatResponse:
        key = build_chat_cache_key(provider=self.provider, model=self.model, request=request)
        if self._records is None:
            self._records = load_cache_records(self.cache_path)
        cached = self._records.get(key)
        if cached is not None:
            return _response_from_record(cached["response"])

        # A miss will be appended, so refuse a protected cache before spending a real call.
        ensure_writable(self.cache_path)
        response = self.inner.complete(request)
        record = {
            "cache_key": key,
            "provider": self.provider,
            "model": self.model,
            "request": {
                "messages": [{"role": m.role, "content": m.content} for m in request.messages],
                "temperature": request.temperature,
                "max_output_tokens": request.max_output_tokens,
                "json_mode": request.json_mode,
                "seed": request.seed,
            },
            "response": {
                "text": response.text,
                "provider": response.provider,
                "model": response.model,
                "usage": asdict(response.usage),
                "latency_ms": response.latency_ms,
                "finish_reason": response.finish_reason,
            },
        }
        append_cache_record(self.cache_path, record)
        self._records[key] = record
        return response


def _response_from_record(record: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        text=record["text"],
        provider=record["provider"],
        model=record["model"],
        usage=Usage(**record["usage"]),
        latency_ms=record["latency_ms"],
        finish_reason=record["finish_reason"],
    )
