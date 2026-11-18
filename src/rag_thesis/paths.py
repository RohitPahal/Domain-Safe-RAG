"""Project root resolution, shared by every module that builds default paths.

Resolution order:
1. ``RAG_THESIS_ROOT`` environment variable, if set.
2. The source checkout this package lives in (``<root>/src/rag_thesis``), detected by
   ``pyproject.toml``. This is the normal case with ``pip install -e .``.
3. The current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_project_root() -> Path:
    env_root = os.environ.get("RAG_THESIS_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    checkout_root = Path(__file__).resolve().parents[2]
    if (checkout_root / "pyproject.toml").is_file():
        return checkout_root
    return Path.cwd().resolve()


PROJECT_ROOT = resolve_project_root()
