from __future__ import annotations

import hashlib


def deterministic_hash(text: str) -> str:
    encoded = text.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
