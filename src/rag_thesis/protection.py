"""Write protection: folders this code must never write into.

The official thesis results in ``outputs/final_benchmark/`` are read-only. Every place in the
package that writes a file or creates a folder first calls :func:`ensure_writable`, so the
rule lives in this one module:

- To protect another folder, add it to ``PROTECTED_DIRS`` (nothing else changes).
- Writers don't know which folders are protected or why; they only ask.

Reading from a protected folder is always allowed (e.g. replaying the official rewrite cache).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from rag_thesis.paths import PROJECT_ROOT

# Folders (relative to the project root) that must never be written to.
PROTECTED_DIRS: tuple[str, ...] = ("outputs/final_benchmark",)


class ProtectedPathError(PermissionError):
    """Raised when code tries to write inside a protected folder."""


def _normalised(path: Path | str) -> Path:
    """Absolute, symlink- and ``..``-free path; case-folded on Windows, where case doesn't matter."""
    return Path(os.path.normcase(Path(path).resolve()))


class WriteGuard:
    """Answers one question: may this path be written? Holds the protected folders."""

    def __init__(self, protected_dirs: Iterable[Path | str]) -> None:
        self._protected = tuple(_normalised(directory) for directory in protected_dirs)

    @property
    def protected_dirs(self) -> tuple[Path, ...]:
        return self._protected

    def is_protected(self, path: Path | str) -> bool:
        """True if ``path`` is a protected folder or anything inside one.

        Compared component by component, so ``outputs/final_benchmark_v2`` is *not* protected.
        """
        target = _normalised(path)
        return any(target == folder or target.is_relative_to(folder) for folder in self._protected)

    def check(self, path: Path | str) -> Path:
        """Return ``path`` unchanged if it may be written; raise ``ProtectedPathError`` otherwise."""
        if self.is_protected(path):
            raise ProtectedPathError(
                f"Refusing to write to {path}: it is inside a protected folder "
                f"({', '.join(str(folder) for folder in self._protected)}). "
                "The official results are read-only; choose a different output path."
            )
        return Path(path)


# The guard used by the whole package (rooted at the project directory).
DEFAULT_GUARD = WriteGuard(PROJECT_ROOT / directory for directory in PROTECTED_DIRS)


def ensure_writable(path: Path | str) -> Path:
    """Call before writing a file or creating a folder. Raises if the target is protected."""
    return DEFAULT_GUARD.check(path)
