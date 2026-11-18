"""Test-wide safety net: tests never touch the project's real data.

For every test:

1. The default rewrite cache (``outputs/cache/rewrites.jsonl``) is replaced by a file in the
   test's temp dir, so tests neither read nor write the real cache. (A test run on 2026-03-16
   once left two fake records in it.)
2. The project's real data folders (``outputs/``, ``data/``, ``indexes/``) are added to the write
   guard from ``rag_thesis.protection``, so any test that tries to write there fails loudly
   with ``ProtectedPathError`` instead of silently polluting real artifacts.

Tests that need their own guard (tests/unit/test_protection.py) still override it.
"""

from __future__ import annotations

import pytest

import rag_thesis.rewrite.rewriter as rewriter
from rag_thesis import protection
from rag_thesis.paths import PROJECT_ROOT
from rag_thesis.protection import WriteGuard

REAL_DATA_DIRS = ("outputs", "data", "indexes")


@pytest.fixture(autouse=True)
def _isolate_real_project_data(tmp_path, monkeypatch):
    monkeypatch.setattr(rewriter, "REWRITE_CACHE_PATH", tmp_path / "rewrites.jsonl")
    guarded = [*protection.DEFAULT_GUARD.protected_dirs, *(PROJECT_ROOT / name for name in REAL_DATA_DIRS)]
    monkeypatch.setattr(protection, "DEFAULT_GUARD", WriteGuard(guarded))
