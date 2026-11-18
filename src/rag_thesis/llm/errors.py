"""Normalised LLM errors: every provider adapter raises these, never its SDK's own exceptions.

Callers and the retry wrapper decide only on the error *type* (``retryable``), so a rate limit
from Gemini, OpenAI, Groq or Anthropic is handled the same way. Mapping an SDK exception to one of
these happens only in ``rag_thesis.llm.providers``.
"""

from __future__ import annotations

from typing import ClassVar


class LLMError(RuntimeError):
    """Base class. ``retryable`` says whether the same request may succeed if sent again."""

    retryable: ClassVar[bool] = False

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMRateLimitError(LLMError):
    """Too many requests right now (HTTP 429); waiting helps."""

    retryable = True


class LLMQuotaExceededError(LLMError):
    """A hard quota is used up (e.g. a daily limit); retrying soon won't help."""


class LLMTransientError(LLMError):
    """Temporary failure: server error, overload, timeout or dropped connection."""

    retryable = True


class LLMAuthError(LLMError):
    """Missing or invalid API key, or no permission for this model."""


class LLMBadRequestError(LLMError):
    """The provider rejected the request itself (bad parameters, unknown model, too long)."""


class LLMContentFilterError(LLMError):
    """The provider refused or blocked the output for safety/content reasons."""


class LLMCallsDisabledError(LLMError):
    """A real LLM call was needed but not allowed (``allow_llm_calls`` is off)."""


_AUTH_STATUSES = {401, 403}
_BAD_REQUEST_STATUSES = {400, 404, 413, 422}
_TRANSIENT_STATUSES = {408, 409, 425, 500, 502, 503, 504, 529}


_NETWORK_FAILURE_NAMES = ("Timeout", "ConnectError", "ConnectionError", "RemoteProtocolError")


def is_network_failure(exc: BaseException) -> bool:
    """True for timeouts and dropped connections (httpx / requests / SDK wrappers of them).

    Checked by class name across the exception's ancestry, so no HTTP library has to be imported.
    """
    return any(marker in cls.__name__ for cls in type(exc).__mro__ for marker in _NETWORK_FAILURE_NAMES)


def error_for_status(status_code: int | None, message: str) -> LLMError:
    """Map an HTTP status code to the matching error type (generic HTTP meaning, any provider)."""
    if status_code == 429:
        return LLMRateLimitError(message, status_code=status_code)
    if status_code in _AUTH_STATUSES:
        return LLMAuthError(message, status_code=status_code)
    if status_code in _BAD_REQUEST_STATUSES:
        return LLMBadRequestError(message, status_code=status_code)
    if status_code in _TRANSIENT_STATUSES or (status_code is not None and status_code >= 500):
        return LLMTransientError(message, status_code=status_code)
    return LLMError(message, status_code=status_code)
