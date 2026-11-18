"""Protect the official benchmark outputs with a SHA-256 manifest.

`outputs/` is gitignored, so git cannot detect changes to the official results.
This tool records a checksum for every file under the protected folder and
verifies them later. It only reads the protected folder; it never writes there.

Usage:
    python tools/benchmark_manifest.py create   # once, to record the baseline
    python tools/benchmark_manifest.py verify   # after every change; exits 1 on any difference
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROTECTED_DIR = Path("outputs/final_benchmark")
MANIFEST_PATH = Path("tests/golden/final_benchmark_manifest.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scan(root: Path) -> dict[str, dict[str, object]]:
    return {
        path.relative_to(root).as_posix(): {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def create(root: Path, manifest: Path) -> int:
    if manifest.exists():
        print(f"Refusing to overwrite existing manifest {manifest}; delete it deliberately first.")
        return 1
    files = _scan(root)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"root": root.as_posix(), "file_count": len(files), "files": files}
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Recorded {len(files)} files from {root} -> {manifest}")
    return 0


def verify(root: Path, manifest: Path) -> int:
    expected = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    actual = _scan(root)
    missing = sorted(set(expected) - set(actual))
    added = sorted(set(actual) - set(expected))
    changed = sorted(p for p in set(expected) & set(actual) if expected[p]["sha256"] != actual[p]["sha256"])
    for label, paths in (("MISSING", missing), ("ADDED", added), ("CHANGED", changed)):
        for path in paths:
            print(f"{label}: {path}")
    if missing or added or changed:
        print(f"FAIL: {root} differs from {manifest}")
        return 1
    print(f"OK: all {len(expected)} files in {root} are unchanged")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["create", "verify"])
    parser.add_argument("--root", type=Path, default=PROTECTED_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()
    if not args.root.is_dir():
        print(f"Protected folder not found: {args.root}")
        return 1
    return create(args.root, args.manifest) if args.command == "create" else verify(args.root, args.manifest)


if __name__ == "__main__":
    sys.exit(main())
