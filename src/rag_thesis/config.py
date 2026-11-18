"""Loading configuration files: one place, one set of rules.

A config is a YAML file with key/value settings at the top level (see ``configs/default.yaml``).
``load_config`` returns it as a plain dict and fails early, with the file name in the message,
when the file is missing, isn't valid YAML, is empty, or isn't a key/value mapping. Without
these checks an empty file would become ``None`` and crash much later, far from the cause.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path | str) -> dict[str, Any]:
    """Read a YAML config file and return its settings as a dict."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        try:
            config = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ValueError(f"Config file is not valid YAML: {config_path}: {exc}") from exc

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain key/value settings at the top level: {config_path}")
    return config
