"""Provider-neutral request and response types for every LLM call.

Callers (rewrite, repair, answer, judge) build a ``ChatRequest`` and get a ``ChatResponse`` back,
whatever the provider. Mapping these types to and from a provider's SDK happens only in
``rag_thesis.llm.providers``. All types are immutable and check their own values, so a
malformed request fails when it is built, not inside a provider call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, get_args

Role = Literal["system", "user", "assistant"]
ROLES: frozenset[str] = frozenset(get_args(Role))


@dataclass(frozen=True)
class ChatMessage:
    """One message of a conversation: who says it and what is said."""

    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {sorted(ROLES)}, got {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError(f"content must be a string, got {type(self.content).__name__}")


@dataclass(frozen=True)
class ChatRequest:
    """Everything a provider needs for one call. Defaults are deterministic (``temperature=0``).

    ``json_mode`` asks the provider for JSON output explicitly; it is never inferred from the
    prompt text. ``None`` for ``max_output_tokens``, ``seed`` or ``timeout_s`` means "not set".
    """

    messages: tuple[ChatMessage, ...]
    temperature: float = 0.0
    max_output_tokens: int | None = None
    json_mode: bool = False
    seed: int | None = None
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        # Accept any sequence (e.g. a list) but store a tuple, so the request is truly immutable.
        if not isinstance(self.messages, Sequence) or isinstance(self.messages, str):
            raise TypeError("messages must be a sequence of ChatMessage")
        object.__setattr__(self, "messages", tuple(self.messages))
        if not self.messages:
            raise ValueError("messages must not be empty")
        if not all(isinstance(message, ChatMessage) for message in self.messages):
            raise TypeError("every message must be a ChatMessage")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens must be > 0, got {self.max_output_tokens}")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {self.timeout_s}")


@dataclass(frozen=True)
class Usage:
    """Token counts of one call. ``total_tokens`` defaults to prompt + completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.total_tokens is None:
            object.__setattr__(self, "total_tokens", self.prompt_tokens + self.completion_tokens)
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")


@dataclass(frozen=True)
class ChatResponse:
    """What a provider returned: the text, token usage, who answered, and how long it took."""

    text: str
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(f"text must be a string, got {type(self.text).__name__}")
        if not self.provider or not self.model:
            raise ValueError("provider and model must be non-empty")
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be >= 0, got {self.latency_ms}")
