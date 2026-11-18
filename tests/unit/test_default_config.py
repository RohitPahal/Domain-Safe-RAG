"""configs/default.yaml must describe the setup behind the official thesis benchmark, and must
never point into the protected results folder."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_thesis.config import load_config
from rag_thesis.llm.settings import resolve_llm_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
OFFICIAL_SNAPSHOT = (
    REPO_ROOT
    / "outputs"
    / "final_benchmark"
    / "subset120_caseB_gem25flash_gem25flash_gem25pro_20260416"
    / "config_snapshot.yaml"
)
# Documented in default.yaml: settings that differ from the snapshot on purpose.
DIFFERS_ON_PURPOSE = {
    "embed_device",
    "rewrite_cache_path",
    "query_embedding_cache_dir",
    "answer_prompt_version",
    "guardrails",
}
# The snapshot's flat LLM keys; default.yaml expresses them as the `llm:` block instead.
FLAT_LLM_KEYS = {
    "llm_provider",
    "llm_api_key_env",
    "llm_api_base",
    "llm_rewrite_model",
    "llm_answer_model",
    "llm_judge_model",
    "openai_rewrite_model",
    "openai_answer_model",
}


def _load(path: Path) -> dict:
    return load_config(path)  # the same loader the CLI uses


def _strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def test_default_config_never_points_into_the_protected_results():
    for value in _strings(_load(DEFAULT_CONFIG)):  # nested settings too (e.g. llm.cache_path)
        assert "final_benchmark" not in value.replace("\\", "/"), f"{value!r} points into outputs/final_benchmark/"


def test_llm_settings_resolve_to_the_thesis_models():
    settings = resolve_llm_settings(_load(DEFAULT_CONFIG))
    models = {role: (s.provider.name, s.model) for role, s in settings.roles.items()}
    assert models == {
        "rewrite": ("gemini", "gemini-2.5-flash"),
        "repair": ("gemini", "gemini-2.5-flash"),
        "answer": ("gemini", "gemini-2.5-flash"),
        "judge": ("gemini", "gemini-2.5-pro"),
    }
    for role in settings.roles.values():
        assert role.provider.type == "gemini"
        assert role.provider.api_key_env == "GEMINI_API_KEY"  # the variable's name, never a key
        assert role.provider.base_url is None
        assert (role.temperature, role.max_output_tokens, role.timeout_s, role.seed) == (0.0, None, None, None)
    assert settings.cache_path == "outputs/cache/llm_responses.jsonl"


@pytest.mark.skipif(not OFFICIAL_SNAPSHOT.is_file(), reason="official benchmark snapshot not present")
def test_default_config_matches_the_official_snapshot():
    config = _load(DEFAULT_CONFIG)
    official = _load(OFFICIAL_SNAPSHOT)

    mismatches = {
        key: (config[key], official.get(key, "<missing>"))
        for key in config
        if key not in DIFFERS_ON_PURPOSE | {"llm"} and config[key] != official.get(key, "<missing>")
    }
    assert not mismatches, f"default.yaml differs from the official snapshot: {mismatches}"
    assert set(official) - set(config) == FLAT_LLM_KEYS
    # The LLM roles are identical; only the request cache is new (the thesis runs had none).
    assert resolve_llm_settings(config).roles == resolve_llm_settings(official).roles
    assert resolve_llm_settings(official).cache_path is None
