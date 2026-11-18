"""Offline test double for ``ChatModel``: scripted responses, no network, no SDK."""

from __future__ import annotations

from collections.abc import Iterable

from rag_thesis.llm.base import ChatModel
from rag_thesis.llm.types import ChatRequest, ChatResponse, Usage


class FakeChatModel(ChatModel):
    """Returns scripted replies in order and records every request it receives.

    Each scripted item is a reply text, a full ``ChatResponse``, or an exception to raise.
    Running out of scripted replies fails loudly, so a test can't make an unexpected call.
    """

    def __init__(
        self,
        replies: Iterable[str | ChatResponse | BaseException] = (),
        *,
        provider: str = "fake",
        model: str = "fake-model",
        usage: Usage | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self._replies = list(replies)
        self._usage = usage or Usage()
        self.requests: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.requests.append(request)
        if not self._replies:
            raise AssertionError(f"FakeChatModel has no scripted reply left (call {len(self.requests)})")
        reply = self._replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, ChatResponse):
            return reply
        return ChatResponse(text=reply, provider=self.provider, model=self.model, usage=self._usage)
