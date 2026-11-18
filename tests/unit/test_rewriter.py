import json

import pytest

from rag_thesis.llm.base import JSONOutputError
from rag_thesis.llm.cache import build_rewrite_cache_key, load_cache_records
from rag_thesis.llm.errors import LLMCallsDisabledError
from rag_thesis.llm.testing import FakeChatModel
from rag_thesis.llm.types import Usage
from rag_thesis.rewrite.rewriter import MULTI_REWRITE_ORDER, REWRITE_PROMPT_VERSION, QueryRewriter, rewrite_query

QUESTION = "What is the synonym of aspirin?"
MULTI_REPLY = (
    '{"bm25_lexical":"aspirin synonym","dense_semantic":"alternate name for aspirin",'
    '"abbreviation_expanded":"aspirin acetylsalicylic acid synonym",'
    '"minimal_entity_locked":"synonym of aspirin"}'
)


def _config() -> dict:
    return {
        "llm_provider": "groq",
        "llm_api_key_env": "GROQ_API_KEY",
        "llm_api_base": "https://api.groq.com/openai/v1",
        "llm_rewrite_model": "llama-3.1-8b-instant",
        "openai_rewrite_model": "gpt-4.1-mini",
    }


def _model(replies) -> FakeChatModel:
    return FakeChatModel(replies, provider="groq", model="llama-3.1-8b-instant", usage=Usage(10, 5, 15))


def test_none_mode_returns_original_only(tmp_path) -> None:
    rewriter = QueryRewriter(config=_config(), chat_model=_model([]), cache_path=tmp_path / "rewrites.jsonl")
    assert rewriter.rewrite(qid="q-1", query_original=QUESTION, rewrite_mode="none") == [QUESTION]


def test_single_mode_uses_cache_and_returns_one_non_empty_rewrite(tmp_path) -> None:
    model = _model(['{"rewrite":"aspirin synonym"}'])
    cache_path = tmp_path / "rewrites.jsonl"
    rewriter = QueryRewriter(config=_config(), chat_model=model, cache_path=cache_path)

    first = rewriter.rewrite(qid="q-2", query_original=QUESTION, rewrite_mode="single")
    second = rewriter.rewrite(qid="q-2", query_original=QUESTION, rewrite_mode="single")

    assert first == second == ["aspirin synonym"]
    assert len(model.requests) == 1
    assert model.requests[0].json_mode is True  # explicit, not inferred from the prompt
    assert cache_path.exists()


def test_new_records_carry_the_provider_in_key_and_record(tmp_path) -> None:
    cache_path = tmp_path / "rewrites.jsonl"
    QueryRewriter(config=_config(), chat_model=_model([MULTI_REPLY]), cache_path=cache_path).rewrite(
        qid="q-7", query_original=QUESTION, rewrite_mode="multi"
    )
    (record,) = load_cache_records(cache_path).values()
    assert record["cache_key"] == build_rewrite_cache_key(
        qid="q-7",
        query_original=QUESTION,
        rewrite_mode="multi",
        rewrite_model="llama-3.1-8b-instant",
        prompt_version=REWRITE_PROMPT_VERSION,
        provider="groq",
    )
    assert record["provider"] == "groq" and record["rewrite_model"] == "llama-3.1-8b-instant"
    assert record["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    # The same model name on another provider is a different cache entry.
    other = FakeChatModel(
        [MULTI_REPLY.replace("aspirin synonym", "other")], provider="together", model="llama-3.1-8b-instant"
    )
    QueryRewriter(config=_config(), chat_model=other, cache_path=cache_path).rewrite(
        qid="q-7", query_original=QUESTION, rewrite_mode="multi"
    )
    assert len(other.requests) == 1


def test_legacy_records_without_provider_are_still_replayed(tmp_path) -> None:
    cache_path = tmp_path / "rewrites.jsonl"
    legacy_key = build_rewrite_cache_key(
        qid="q-8",
        query_original=QUESTION,
        rewrite_mode="single",
        rewrite_model="llama-3.1-8b-instant",
        prompt_version=REWRITE_PROMPT_VERSION,
    )
    cache_path.write_text(
        json.dumps({"cache_key": legacy_key, "rewrites": ["legacy rewrite"]}) + "\n", encoding="utf-8"
    )
    model = _model([])
    rewriter = QueryRewriter(config=_config(), chat_model=model, cache_path=cache_path)
    assert rewriter.rewrite(qid="q-8", query_original=QUESTION, rewrite_mode="single") == ["legacy rewrite"]
    assert model.requests == []


def test_replay_needs_no_model_at_all_and_a_miss_fails_clearly(tmp_path) -> None:
    cache_path = tmp_path / "rewrites.jsonl"
    QueryRewriter(config=_config(), chat_model=_model([MULTI_REPLY]), cache_path=cache_path).rewrite(
        qid="q-9", query_original=QUESTION, rewrite_mode="multi"
    )
    replay = QueryRewriter(config=_config(), cache_path=cache_path)  # identity from the config's rewrite role
    assert replay.rewrite(qid="q-9", query_original=QUESTION, rewrite_mode="multi")[0] == "aspirin synonym"
    with pytest.raises(LLMCallsDisabledError):
        replay.rewrite(qid="q-10", query_original=QUESTION, rewrite_mode="multi")


def test_multi_mode_returns_exactly_four_unique_rewrites_in_required_order(tmp_path) -> None:
    rewrites = rewrite_query(
        config=_config(),
        qid="q-3",
        query_original=QUESTION,
        rewrite_mode="multi",
        chat_model=_model([MULTI_REPLY]),
        cache_path=tmp_path / "rewrites.jsonl",
    )
    assert rewrites == [
        "aspirin synonym",
        "alternate name for aspirin",
        "aspirin acetylsalicylic acid synonym",
        "synonym of aspirin",
    ]
    assert MULTI_REWRITE_ORDER == ["bm25_lexical", "dense_semantic", "abbreviation_expanded", "minimal_entity_locked"]


def test_multi_mode_accepts_code_fenced_json_and_preserves_required_order(tmp_path) -> None:
    fenced = "```json\n" + MULTI_REPLY.replace("synonym", "alias") + "\n```"
    rewrites = rewrite_query(
        config=_config(),
        qid="q-5",
        query_original=QUESTION,
        rewrite_mode="multi",
        chat_model=_model([fenced]),
        cache_path=tmp_path / "rewrites.jsonl",
    )
    assert rewrites == [
        "aspirin alias",
        "alternate name for aspirin",
        "aspirin acetylsalicylic acid alias",
        "alias of aspirin",
    ]


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        ('{"bm25_lexical": "a", "dense_semantic": "b"}', JSONOutputError),  # missing keys
        (
            '{"bm25_lexical": "a", "dense_semantic": "a", "abbreviation_expanded": "c", "minimal_entity_locked": "d"}',
            ValueError,
        ),
        (
            '{"bm25_lexical": " ", "dense_semantic": "b", "abbreviation_expanded": "c", "minimal_entity_locked": "d"}',
            ValueError,
        ),
        ("not json", JSONOutputError),
    ],
)
def test_invalid_multi_replies_are_rejected_and_not_cached(tmp_path, reply, error) -> None:
    cache_path = tmp_path / "rewrites.jsonl"
    with pytest.raises(error):
        rewrite_query(
            config=_config(),
            qid="q-11",
            query_original=QUESTION,
            rewrite_mode="multi",
            chat_model=_model([reply]),
            cache_path=cache_path,
        )
    assert not cache_path.exists()


def test_same_cache_key_reuses_cached_value(tmp_path) -> None:
    model = _model(['{"rewrite":"aspirin synonym"}', '{"rewrite":"unused second call"}'])
    rewriter = QueryRewriter(config=_config(), chat_model=model, cache_path=tmp_path / "rewrites.jsonl")
    first = rewriter.rewrite(qid="q-4", query_original=QUESTION, rewrite_mode="single")
    second = rewriter.rewrite(qid="q-4", query_original=QUESTION, rewrite_mode="single")
    assert first == second == ["aspirin synonym"]
    assert len(model.requests) == 1


def test_rewriter_uses_configured_benchmark_local_cache_path(tmp_path) -> None:
    benchmark_cache_path = tmp_path / "final_benchmark" / "case_b" / "cache" / "rewrites.jsonl"
    config = _config() | {"rewrite_cache_path": str(benchmark_cache_path)}
    rewriter = QueryRewriter(config=config, chat_model=_model(['{"rewrite":"aspirin synonym"}']))
    assert rewriter.rewrite(qid="q-6", query_original=QUESTION, rewrite_mode="single") == ["aspirin synonym"]
    assert benchmark_cache_path.exists()
