"""Per-role LLM settings: new `llm` block and legacy flat keys resolve the same way."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_thesis.config import load_config
from rag_thesis.llm.settings import CALL_DEFAULTS, ProviderSettings, resolve_llm_settings

OFFICIAL_SNAPSHOT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "final_benchmark"
    / "subset120_caseB_gem25flash_gem25flash_gem25pro_20260416"
    / "config_snapshot.yaml"
)
GEMINI = ProviderSettings(name="gemini", type="gemini", api_key_env="GEMINI_API_KEY")


def _models(settings):
    return {role: (s.provider.name, s.model) for role, s in settings.roles.items()}


# The official snapshot's LLM keys, copied so the test also runs where the results are absent.
THESIS_FLAT_KEYS = {
    "llm_provider": "gemini",
    "llm_api_key_env": "GEMINI_API_KEY",
    "llm_api_base": None,
    "llm_rewrite_model": "gemini-2.5-flash",
    "llm_answer_model": "gemini-2.5-flash",
    "llm_judge_model": "gemini-2.5-pro",
    "openai_rewrite_model": "gpt-4.1-mini",  # legacy fallbacks, unused because llm_* keys are set
    "openai_answer_model": "gpt-4.1-mini",
}


def test_thesis_flat_keys_keep_their_exact_meaning():
    settings = resolve_llm_settings(THESIS_FLAT_KEYS)
    assert _models(settings) == {
        "rewrite": ("gemini", "gemini-2.5-flash"),
        "repair": ("gemini", "gemini-2.5-flash"),  # the old guard repaired with the rewrite model
        "answer": ("gemini", "gemini-2.5-flash"),
        "judge": ("gemini", "gemini-2.5-pro"),
    }
    for role in settings.roles.values():
        assert role.provider == GEMINI
        assert (role.temperature, role.max_output_tokens, role.timeout_s, role.seed) == (0.0, None, None, None)
    assert settings.cache_path is None


@pytest.mark.skipif(not OFFICIAL_SNAPSHOT.is_file(), reason="official benchmark snapshot not present")
def test_the_official_snapshot_loads_unchanged():
    official = load_config(OFFICIAL_SNAPSHOT)
    assert resolve_llm_settings(official) == resolve_llm_settings(THESIS_FLAT_KEYS)


def test_legacy_fallbacks_and_providers():
    settings = resolve_llm_settings(
        {"llm_provider": "Groq", "openai_rewrite_model": "llama-a", "llm_answer_model": "llama-b"}
    )
    assert _models(settings) == {
        "rewrite": ("groq", "llama-a"),
        "repair": ("groq", "llama-a"),
        "answer": ("groq", "llama-b"),
        "judge": ("groq", "llama-b"),  # judge falls back to the answer model, as before
    }
    groq = settings.role("rewrite").provider
    assert (groq.type, groq.api_key_env, groq.base_url) == (
        "openai_compatible",
        "GROQ_API_KEY",
        "https://api.groq.com/openai/v1",
    )
    custom = resolve_llm_settings(
        {"llm_provider": "openai", "llm_answer_model": "m", "llm_api_key_env": "MY_KEY", "llm_api_base": "http://x"}
    )
    assert custom.role("answer").provider.api_key_env == "MY_KEY"
    assert custom.role("answer").provider.base_url == "http://x"


def test_legacy_config_without_provider_fails_instead_of_assuming_openai():
    with pytest.raises(ValueError, match="no default is assumed"):
        resolve_llm_settings({"llm_answer_model": "gpt-x"})  # P6


def test_config_without_any_llm_settings_has_no_roles():
    settings = resolve_llm_settings({"retrieve_k": 10})
    assert settings.roles == {}
    with pytest.raises(ValueError, match="No LLM is configured for the 'answer' role"):
        settings.role("answer")
    with pytest.raises(ValueError, match="Unknown LLM role"):
        settings.role("summarise")


NEW_BLOCK = {
    "llm": {
        "defaults": {"max_output_tokens": 256, "timeout_s": 60},
        "cache_path": "outputs/cache/llm_responses.jsonl",
        "providers": {
            "gemini": {"type": "gemini", "api_key_env": "GEMINI_API_KEY"},
            "local": {"type": "openai_compatible", "api_key_env": None, "base_url": "http://localhost:11434/v1"},
            "claude": {"type": "anthropic", "api_key_env": "ANTHROPIC_API_KEY", "default_max_tokens": 2048},
        },
        "roles": {
            "rewrite": {"provider": "gemini", "model": "gemini-2.5-flash"},
            "answer": {"provider": "local", "model": "llama3", "max_retries": 0},
            "judge": {"provider": "claude", "model": "claude-test", "max_output_tokens": 512, "seed": 1},
        },
    }
}


def test_new_block_gives_each_role_its_own_provider_model_and_parameters():
    settings = resolve_llm_settings(NEW_BLOCK)
    assert _models(settings) == {
        "rewrite": ("gemini", "gemini-2.5-flash"),
        "answer": ("local", "llama3"),
        "judge": ("claude", "claude-test"),
    }
    assert settings.cache_path == "outputs/cache/llm_responses.jsonl"
    rewrite, answer, judge = (settings.role(r) for r in ("rewrite", "answer", "judge"))
    assert (rewrite.max_output_tokens, rewrite.timeout_s, rewrite.max_retries) == (256, 60, 3)  # defaults
    assert answer.max_retries == 0 and answer.provider.api_key_env is None
    assert answer.provider.base_url == "http://localhost:11434/v1"
    assert (judge.max_output_tokens, judge.seed) == (512, 1)  # per-role override
    assert judge.provider.options == {"default_max_tokens": 2048}  # extra adapter arguments


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda b: b["roles"].update(summarise={"provider": "gemini", "model": "m"}), "Unknown LLM role"),
        (lambda b: b["roles"]["rewrite"].update(provider="missing"), "not defined in llm.providers"),
        (lambda b: b["roles"]["rewrite"].pop("model"), "model must be a non-empty string"),
        (lambda b: b["roles"]["rewrite"].update(temprature=0), "Unknown setting"),
        (lambda b: b["providers"]["gemini"].pop("type"), "type must be a non-empty string"),
        (lambda b: b["defaults"].update(max_retries=-1), "max_retries"),
        (lambda b: b.update(extra=1), "Unknown setting"),
    ],
)
def test_new_block_errors_are_clear(change, message):
    import copy

    block = copy.deepcopy(NEW_BLOCK)
    change(block["llm"])
    with pytest.raises(ValueError, match=message):
        resolve_llm_settings(block)


def test_call_defaults_are_deterministic():
    assert CALL_DEFAULTS["temperature"] == 0.0
    assert CALL_DEFAULTS["seed"] is None and CALL_DEFAULTS["max_output_tokens"] is None
