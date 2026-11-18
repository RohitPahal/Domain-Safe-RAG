"""Normalised LLM errors: one meaning per error type, whatever the provider."""

from __future__ import annotations

import pytest

from rag_thesis.llm.errors import (
    LLMAuthError,
    LLMBadRequestError,
    LLMCallsDisabledError,
    LLMContentFilterError,
    LLMError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMTransientError,
    error_for_status,
    is_network_failure,
)


def test_only_rate_limits_and_transient_errors_are_retryable():
    assert LLMRateLimitError.retryable and LLMTransientError.retryable
    for error_type in (
        LLMError,
        LLMQuotaExceededError,
        LLMAuthError,
        LLMBadRequestError,
        LLMContentFilterError,
        LLMCallsDisabledError,
    ):
        assert not error_type.retryable, error_type.__name__


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, LLMRateLimitError),
        (401, LLMAuthError),
        (403, LLMAuthError),
        (400, LLMBadRequestError),
        (404, LLMBadRequestError),
        (422, LLMBadRequestError),
        (408, LLMTransientError),
        (500, LLMTransientError),
        (503, LLMTransientError),
        (529, LLMTransientError),  # Anthropic "overloaded"
        (599, LLMTransientError),
        (418, LLMError),
        (None, LLMError),
    ],
)
def test_error_for_status(status, expected):
    error = error_for_status(status, "provider said no")
    assert type(error) is expected
    assert error.status_code == status
    assert str(error) == "provider said no"


class ReadTimeout(Exception):  # same class name as httpx's
    pass


class ConnectError(Exception):  # same class name as httpx's
    pass


@pytest.mark.parametrize("error", [ReadTimeout("x"), ConnectError("x"), TimeoutError("x"), ConnectionResetError("x")])
def test_network_failures_are_recognised_by_class_name(error):
    assert is_network_failure(error)


@pytest.mark.parametrize("error", [ValueError("timeout in the message is not enough"), KeyError("x")])
def test_other_errors_are_not_network_failures(error):
    assert not is_network_failure(error)
