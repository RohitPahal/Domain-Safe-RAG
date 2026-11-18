"""The command-line interface must not change by accident.

Every ``rag-thesis`` command's options (names, defaults, types, required, help) are compared
with the snapshot in ``tests/assets/cli_contract.json``. Project paths in defaults are written
as ``<root>`` so the snapshot works on any machine.

After an *intentional* CLI change, regenerate the snapshot and review its diff:

    python tests/unit/test_cli_contract.py --write
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

SNAPSHOT = Path(__file__).resolve().parents[1] / "assets" / "cli_contract.json"


def cli_contract() -> dict:
    """Describe every command and option of the real CLI as plain data."""
    from rag_thesis.cli import app
    from rag_thesis.paths import PROJECT_ROOT

    root = str(PROJECT_ROOT)

    def portable(value):
        text = str(value)
        return text.replace(root, "<root>").replace("\\", "/") if root in text else value

    def describe(param) -> dict:
        # Click's own stable description of an option: names, type (incl. path rules such as
        # exists / file_okay / dir_okay), required, default, help, flag settings, ...
        info = param.to_info_dict()
        info["default"] = portable(info.get("default"))
        return info

    group = typer.main.get_command(app)
    contract = {
        name: {"help": command.help, "params": [describe(param) for param in command.params]}
        for name, command in sorted(group.commands.items())
    }
    return json.loads(json.dumps(contract, default=str))


def test_cli_matches_the_snapshot():
    assert SNAPSHOT.is_file(), "Missing CLI snapshot; create it with: python tests/unit/test_cli_contract.py --write"
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    actual = cli_contract()
    assert sorted(actual) == sorted(expected), "the set of commands changed"
    for name in expected:
        assert actual[name] == expected[name], f"options of `rag-thesis {name}` changed"


if __name__ == "__main__" and "--write" in sys.argv:
    SNAPSHOT.write_text(json.dumps(cli_contract(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {SNAPSHOT}")
