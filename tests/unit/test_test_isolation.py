"""The test-wide safety net in tests/conftest.py is active (checks only; nothing is written)."""

from __future__ import annotations

import pytest

import rag_thesis.rewrite.rewriter as rewriter
from rag_thesis.paths import PROJECT_ROOT
from rag_thesis.protection import ProtectedPathError, ensure_writable


@pytest.mark.parametrize("folder", ["outputs", "data", "indexes"])
def test_real_project_data_is_write_protected_during_tests(folder):
    with pytest.raises(ProtectedPathError):
        ensure_writable(PROJECT_ROOT / folder / "written_by_a_test.txt")


def test_default_rewrite_cache_is_a_temp_file_during_tests(tmp_path):
    assert rewriter.REWRITE_CACHE_PATH == tmp_path / "rewrites.jsonl"
    assert not rewriter.REWRITE_CACHE_PATH.is_relative_to(PROJECT_ROOT)


def test_temp_dirs_stay_writable(tmp_path):
    assert ensure_writable(tmp_path / "ok.txt") == tmp_path / "ok.txt"
