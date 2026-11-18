"""Config loading (rag_thesis/config.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_thesis.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_loads_key_value_settings(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("chunk_tokens: 180\nembed_model: m\neval_k_list: [5, 10]\n", encoding="utf-8")
    assert load_config(path) == {"chunk_tokens": 180, "embed_model": "m", "eval_k_list": [5, 10]}


def test_the_real_default_config_loads():
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert config["embed_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert config["rrf_k"] == 60


@pytest.mark.parametrize(
    "content, error, message",
    [
        (None, FileNotFoundError, "Config file not found"),
        ("", ValueError, "Config file is empty"),
        ("# only a comment\n", ValueError, "Config file is empty"),
        ("key: [unclosed\n", ValueError, "not valid YAML"),
        ("- a\n- b\n", ValueError, "key/value settings at the top level"),
        ("just a string\n", ValueError, "key/value settings at the top level"),
    ],
    ids=["missing", "empty", "comment_only", "invalid_yaml", "list", "scalar"],
)
def test_bad_config_files_fail_early_with_the_file_name(tmp_path, content, error, message):
    path = tmp_path / "bad.yaml"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(error, match=message) as info:
        load_config(path)
    assert "bad.yaml" in str(info.value)


def test_pipeline_still_requires_a_config_or_a_path(tmp_path):
    from rag_thesis.pipeline.workflow import build_rag_run_context

    with pytest.raises(ValueError, match="config or config_path must be provided"):
        build_rag_run_context(
            questions_path=tmp_path / "q.jsonl",
            chunks_path=tmp_path / "c.jsonl",
            retriever_mode="bm25",
        )
