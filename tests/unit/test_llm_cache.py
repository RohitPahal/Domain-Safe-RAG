"""JSONL cache helpers (`llm/cache.py`): deterministic keys, append-only, duplicate-safe."""

import json
from types import SimpleNamespace

from rag_thesis.llm.cache import append_cache_record, build_rewrite_cache_key, get_cache_record, load_cache_records


def test_cache_helpers_are_deterministic_append_safe_and_dedupe_safe(tmp_path) -> None:
    rewrite_cache_path = tmp_path / "rewrites.jsonl"
    rewrite_key = build_rewrite_cache_key(
        qid="q-1",
        query_original="What is aspirin?",
        rewrite_mode="single",
        rewrite_model="llama-3.1-8b-instant",
        prompt_version="rewrite_v1",
    )
    key_inputs = {
        "qid": "q-1",
        "query_original": "What is aspirin?",
        "rewrite_mode": "single",
        "rewrite_model": "llama-3.1-8b-instant",
    }
    # Deterministic: the same inputs always give the same key; changing the prompt version changes it.
    assert build_rewrite_cache_key(**key_inputs, prompt_version="rewrite_v1") == rewrite_key
    assert build_rewrite_cache_key(**key_inputs, prompt_version="rewrite_v2") != rewrite_key

    rewrite_record = {
        "cache_key": rewrite_key,
        "qid": "q-1",
        "rewrites": ["aspirin synonym"],
    }

    assert load_cache_records(rewrite_cache_path) == {}
    assert append_cache_record(rewrite_cache_path, rewrite_record) is True
    assert append_cache_record(rewrite_cache_path, rewrite_record) is False
    assert get_cache_record(rewrite_cache_path, rewrite_key) == rewrite_record

    rewrite_lines = rewrite_cache_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rewrite_lines) == 1
    assert json.loads(rewrite_lines[0])["cache_key"] == rewrite_key


def test_cache_read_is_dedupe_safe_when_duplicate_lines_exist(tmp_path) -> None:
    cache_path = tmp_path / "rewrites.jsonl"
    cache_path.write_text(
        "\n".join(
            [
                '{"cache_key":"same-key","qid":"q-1","rewrites":["first"]}',
                '{"cache_key":"same-key","qid":"q-1","rewrites":["second"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    record = get_cache_record(cache_path, "same-key")

    assert record == {"cache_key": "same-key", "qid": "q-1", "rewrites": ["second"]}


class FakeResponse:
    def __init__(self) -> None:
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[
                        {"type": "output_text", "text": "alpha "},
                        {"type": "output_text", "text": "beta"},
                    ]
                )
            )
        ]
        self.usage = SimpleNamespace(prompt_tokens=11, completion_tokens=5, total_tokens=16)

    def model_dump(self) -> dict:
        return {
            "choices": [{"message": {"content": [{"text": "alpha "}, {"text": "beta"}]}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
        }


class FakeSDKClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self._create,
            )
        )
        self.calls = []

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse()


class FakeGeminiResponse:
    def __init__(self) -> None:
        self.text = '{"rewrite":"aspirin synonym"}'
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=9,
            candidates_token_count=4,
            total_token_count=13,
        )

    def model_dump(self) -> dict:
        return {
            "text": self.text,
            "usage_metadata": {
                "prompt_token_count": 9,
                "candidates_token_count": 4,
                "total_token_count": 13,
            },
        }


class FakeGeminiSDKClient:
    def __init__(self) -> None:
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return FakeGeminiResponse()


class SequentialGeminiSDKClient:
    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, **kwargs):
        self.calls.append(kwargs)
        next_response = self._responses[len(self.calls) - 1]
        if isinstance(next_response, Exception):
            raise next_response
        return next_response
