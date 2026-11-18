"""Single command-line entry point: ``rag-thesis <command>`` (or ``python -m rag_thesis``).

Each command lives in ``cli/<command>.py``, which holds only argument parsing and printing and
calls the library. This file only wires them up.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType

import typer

from rag_thesis.cli import (
    bench_retrieval,
    build_corpus,
    build_index,
    eval_answers,
    eval_retrieval,
    judge,
    make_subset,
    report,
    rewrite_batch,
    run,
    run_batch,
    validate,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Domain-safe query rewriting for biomedical RAG: data, indexing, pipeline and evaluation.",
)

# (command name, module); each module exposes a Typer `app` with one `main` command.
COMMANDS: list[tuple[str, ModuleType]] = [
    ("make-subset", make_subset),
    ("build-corpus", build_corpus),
    ("validate", validate),
    ("build-index", build_index),
    ("run", run),
    ("run-batch", run_batch),
    ("rewrite-batch", rewrite_batch),
    ("eval-retrieval", eval_retrieval),
    ("bench-retrieval", bench_retrieval),
    ("eval-answers", eval_answers),
    ("judge", judge),
    ("report", report),
]


def _register(name: str, module: ModuleType) -> None:
    command: Callable[..., object] = module.main
    app.command(name, help=module.app.info.help)(command)


for _name, _module in COMMANDS:
    _register(_name, _module)


if __name__ == "__main__":
    app()
