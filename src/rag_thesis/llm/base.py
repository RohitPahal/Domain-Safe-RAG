"""The ``ChatModel`` interface: the only way the rest of the code talks to an LLM.

A chat model turns a ``ChatRequest`` into a ``ChatResponse``. Provider adapters
(``rag_thesis.llm.providers``) implement ``complete``; wrappers (retry, cache, offline guard,
usage tracking) implement it by delegating to another ``ChatModel``. JSON output is implemented
once, here, for every provider: ``complete_json`` asks for JSON explicitly, parses the reply
tolerantly and validates it with a Pydantic model.
"""

from __future__ import annotations

import dataclasses
import json
import re
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from rag_thesis.llm.types import ChatRequest, ChatResponse

T = TypeVar("T", bound=BaseModel)

_CODE_FENCE_START = re.compile(r"^```(?:json)?\s*", flags=re.IGNORECASE)
_CODE_FENCE_END = re.compile(r"\s*```$")
_JSON_OBJECT = re.compile(r"\{.*\}", flags=re.DOTALL)


class JSONOutputError(ValueError):
    """The model's reply was not the JSON object that was asked for (never retried)."""

    def __init__(self, message: str, text: str) -> None:
        super().__init__(f"{message}: {text!r}")
        self.text = text


class ChatModel(ABC):
    """One LLM (a provider + model pair), or a wrapper around one."""

    provider: str
    model: str

    @abstractmethod
    def complete(self, request: ChatRequest) -> ChatResponse:
        """Send one request and return the response; raise an ``LLMError`` on failure."""

    def complete_json(self, request: ChatRequest, schema: type[T]) -> tuple[T, ChatResponse]:
        """Ask for a JSON object, parse it and validate it with ``schema``.

        Raises ``JSONOutputError`` if the reply is not a JSON object matching ``schema``.
        """
        response = self.complete(dataclasses.replace(request, json_mode=True))
        payload = extract_json_object(response.text)
        try:
            return schema.model_validate(payload), response
        except ValidationError as exc:
            message = f"reply does not match {schema.__name__} ({exc.error_count()} errors)"
            raise JSONOutputError(message, response.text) from exc


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse the JSON object in a model reply, tolerating code fences and text around it."""
    stripped = text.strip()
    if not stripped:
        raise JSONOutputError("empty reply", text)
    if stripped.startswith("```"):
        stripped = _CODE_FENCE_END.sub("", _CODE_FENCE_START.sub("", stripped)).strip()

    candidate = stripped
    if not (stripped.startswith("{") and stripped.endswith("}")):
        match = _JSON_OBJECT.search(stripped)
        if match is None:
            raise JSONOutputError("no JSON object in reply", text)
        candidate = match.group(0)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise JSONOutputError(f"invalid JSON ({exc.msg})", text) from exc
    if not isinstance(payload, dict):
        raise JSONOutputError("reply is JSON but not an object", text)
    return payload
