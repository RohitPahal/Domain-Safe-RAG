"""Fail loudly if anything under outputs/final_benchmark/ changes.

The official thesis results are gitignored, so this checksum test is the guard.
Skipped where the folder isn't present (e.g. CI).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_DIR = REPO_ROOT / "outputs" / "final_benchmark"
MANIFEST = REPO_ROOT / "tests" / "golden" / "final_benchmark_manifest.json"


@pytest.mark.skipif(not PROTECTED_DIR.is_dir(), reason="official benchmark outputs not present")
def test_official_benchmark_outputs_unchanged() -> None:
    # tests/golden/ is local-only (gitignored): results-derived data is never committed.
    assert MANIFEST.is_file(), (
        f"Missing {MANIFEST}. It is local-only; recreate it with "
        "`python tools/benchmark_manifest.py create` only if outputs/final_benchmark is known to be unchanged."
    )
    result = subprocess.run(
        [
            sys.executable,
            "tools/benchmark_manifest.py",
            "verify",
            "--root",
            str(PROTECTED_DIR),
            "--manifest",
            str(MANIFEST),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
