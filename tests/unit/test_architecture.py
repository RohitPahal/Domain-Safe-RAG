"""Provider-agnostic LLM layer: provider specifics live only in `llm/providers/`.

Checked on the syntax tree of every module under `src/rag_thesis/`:

- no LLM SDK is imported outside `llm/providers/` (adapters import theirs lazily);
- no code outside `llm/providers/` compares anything with a provider name (no `if provider == ...`);
  which provider a role uses is decided only by config and the registry's adapter table.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "rag_thesis"
PROVIDERS = PACKAGE / "llm" / "providers"
SDK_MODULES = ("google.genai", "openai", "anthropic", "groq", "langchain")
PROVIDER_NAMES = {"gemini", "openai", "openai_compatible", "groq", "anthropic", "ollama", "vllm", "together"}

MODULES = sorted(path for path in PACKAGE.rglob("*.py") if PROVIDERS not in path.parents)


def _imported_modules(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
            names.extend(f"{node.module}.{alias.name}" for alias in node.names)  # from google import genai
    return names


def _provider_comparisons(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            constants = [o.value for o in operands if isinstance(o, ast.Constant) and isinstance(o.value, str)]
            if any(value.lower() in PROVIDER_NAMES for value in constants):
                lines.append(node.lineno)
    return lines


def test_the_scan_covers_the_package():
    assert len(MODULES) > 30 and PACKAGE / "pipeline" / "workflow.py" in MODULES


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(PACKAGE)))
def test_no_sdk_import_or_provider_branch_outside_the_adapters(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    sdk_imports = [name for name in _imported_modules(tree) if name.startswith(SDK_MODULES)]
    assert not sdk_imports, f"SDK import outside llm/providers/: {sdk_imports}"
    assert not _provider_comparisons(tree), f"provider-name comparison at lines {_provider_comparisons(tree)}"
