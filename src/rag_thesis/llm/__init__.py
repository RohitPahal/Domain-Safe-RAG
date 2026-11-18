"""Provider-agnostic LLM layer.

``types`` (requests/responses) · ``base`` (``ChatModel``) · ``errors`` · ``wrappers`` (retry, cache,
offline guard, usage tracking) · ``providers`` (SDK adapters) · ``settings`` (per-role config) ·
``registry`` (builds a role's model; the only composition root) · ``cache`` · ``testing`` (fakes).
"""
