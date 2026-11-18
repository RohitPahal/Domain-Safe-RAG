"""Answer generation from retrieved chunks (role ``answer``).

``AnswerGenerator`` receives the answer ``ChatModel`` and never builds one. Each call asks once,
with one window of chunks, using the configured prompt (``answer_prompt_version``: ``answer_v1``,
the thesis prompt, or the hardened ``answer_v2``, which can also refuse). Which windows are asked
(top-1, then top-3 after an empty or
``NOT_FOUND`` reply) is decided by the graph's edges (``pipeline/graph.py``). Failed calls are not
retried here: the model's ``RetryingChatModel`` wrapper is the one retry policy.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from rag_thesis.llm.base import ChatModel
from rag_thesis.llm.errors import LLMError
from rag_thesis.llm.types import ChatRequest
from rag_thesis.prompts import answer as answer_v1
from rag_thesis.prompts import answer_v2
from rag_thesis.prompts.answer import NOT_FOUND_ANSWER

LOGGER = logging.getLogger(__name__)
# A provider failure is the only expected way for an answer call to fail (the reply is plain text,
# so there is nothing to parse). It becomes a failed status; anything else, such as a code bug or a
# write-protection error, stops the run.
EXPECTED_GENERATION_ERRORS = (LLMError,)
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")
ANSWER_PROMPT_VERSIONS = (answer_v1.VERSION, answer_v2.VERSION)
DEFAULT_ANSWER_PROMPT_VERSION = answer_v1.VERSION  # a config without the key is a thesis-era config


def answer_prompt_version(config: dict[str, Any]) -> str:
    """The configured answer prompt (``answer_prompt_version``); the thesis prompt when unset."""
    version = config.get("answer_prompt_version", DEFAULT_ANSWER_PROMPT_VERSION)
    if version not in ANSWER_PROMPT_VERSIONS:
        raise ValueError(f"answer_prompt_version must be one of {', '.join(ANSWER_PROMPT_VERSIONS)}, not {version!r}.")
    return str(version)


@dataclass(frozen=True)
class GenerationOutcome:
    """Result of one answer call; ``status`` is answered / not_found / refused / failed."""

    status: str
    answer_text: str | None = None
    citations: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)  # tokens of this call; {} if it didn't return
    error: str | None = None


class AnswerGenerator:
    def __init__(self, chat_model: ChatModel, prompt_version: str = DEFAULT_ANSWER_PROMPT_VERSION) -> None:
        if prompt_version not in ANSWER_PROMPT_VERSIONS:
            raise ValueError(f"Unknown answer prompt version {prompt_version!r}.")
        self.chat_model = chat_model
        self.prompt_version = prompt_version

    def answer(
        self,
        *,
        qid: str,
        question: str,
        window: list[dict[str, Any]],
        chunks_by_id: dict[str, dict[str, Any]],
    ) -> GenerationOutcome:
        """Ask once with the chunks of ``window``. Chunks missing from the store are left out of the
        prompt (an answer still cites the whole window, as in the thesis runs)."""
        passages = [
            (str(result["chunk_id"]), str(chunks_by_id[result["chunk_id"]]["text"]))
            for result in window
            if result["chunk_id"] in chunks_by_id
        ]
        if not passages:
            return GenerationOutcome(
                status="failed", error="none of the retrieved chunk ids were found in the chunk store"
            )

        if self.prompt_version == answer_v2.VERSION:
            messages = answer_v2.build_answer_messages(question, passages)
        else:
            messages = answer_v1.build_answer_messages(question, [f"[{cid}] {text}" for cid, text in passages])
        request = ChatRequest(messages=messages)
        try:
            response = self.chat_model.complete(request)
        except EXPECTED_GENERATION_ERRORS as exc:
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning(
                "Answer generation failed for qid=%s model=%s/%s: %s",
                qid,
                self.chat_model.provider,
                self.chat_model.model,
                error,
            )
            return GenerationOutcome(status="failed", error=error)

        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": int(response.usage.total_tokens or 0),
        }
        answer_text = postprocess_answer(response.text)
        if self.prompt_version == answer_v2.VERSION and answer_text == answer_v2.REFUSED_ANSWER:
            # The model declined (out of scope or unsafe request): not an answer, and no retry.
            return GenerationOutcome(status="refused", usage=usage)
        if answer_text:
            citations = [result["chunk_id"] for result in window]
            return GenerationOutcome(status="answered", answer_text=answer_text, citations=citations, usage=usage)
        # Empty or NOT_FOUND: an abstention for this window, and its tokens still count.
        return GenerationOutcome(status="not_found", usage=usage)


def add_usage(first: dict[str, int], second: dict[str, int]) -> dict[str, int]:
    """Tokens of two calls together; ``{}`` if neither call returned (as run logs have always shown)."""
    if not first and not second:
        return {}
    return {key: int(first.get(key, 0)) + int(second.get(key, 0)) for key in USAGE_KEYS}


def postprocess_answer(text: str) -> str | None:
    """The minimal answer span from a reply, or ``None`` for empty / ``NOT_FOUND`` replies."""
    if not isinstance(text, str):
        return None

    stripped = text.strip()
    if not stripped:
        return None

    json_match = re.search(r'"exact_answer"\s*:\s*"([^"]+)"', stripped, flags=re.IGNORECASE)
    if json_match:
        answer = json_match.group(1).strip()
    else:
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if not lines:
            return None

        cleaned_lines = [line for line in lines if line not in {"{", "}", "```", "```json", "json"}]
        if not cleaned_lines:
            return None
        answer = cleaned_lines[0]

    answer = re.sub(r"^(answer|final answer)\s*[:\-]\s*", "", answer, flags=re.IGNORECASE).strip()
    answer = answer.strip("`\"' \t")
    answer_norm = answer.rstrip(" .:;!?,").strip()

    if not answer_norm:
        return None
    if answer_norm.upper() == NOT_FOUND_ANSWER:
        return None
    if answer_norm in {"{", "}"}:
        return None

    return answer_norm
