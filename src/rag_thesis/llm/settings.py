"""Per-role LLM settings: which provider and model each role uses, read from the config.

Roles: ``rewrite``, ``repair``, ``answer``, ``judge``. Each can use its own provider and model, so
switching provider is a config-only change. New configs use an ``llm`` block::

    llm:
      defaults:  {temperature: 0, max_output_tokens: null, timeout_s: null, max_retries: 3}
      cache_path: outputs/cache/llm_responses.jsonl     # optional request-level cache
      providers:
        gemini: {type: gemini, api_key_env: GEMINI_API_KEY}
        groq:   {type: openai_compatible, api_key_env: GROQ_API_KEY, base_url: https://api.groq.com/openai/v1}
      roles:
        rewrite: {provider: gemini, model: gemini-2.5-flash}
        judge:   {provider: gemini, model: gemini-2.5-pro, max_output_tokens: 512}

Older configs (e.g. the official benchmark's ``config_snapshot.yaml``) use flat ``llm_*`` keys;
``resolve_llm_settings`` converts them to the same settings, keeping their exact old meaning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ROLES: tuple[str, ...] = ("rewrite", "repair", "answer", "judge")
# Per-call settings a role may set (defaults below, overridable in `llm.defaults` or per role).
CALL_DEFAULTS: dict[str, Any] = {
    "temperature": 0.0,
    "max_output_tokens": None,
    "timeout_s": None,
    "seed": None,
    "max_retries": 3,
}
# Provider names the flat legacy keys could use (`llm_provider`), as the old client defined them.
LEGACY_PROVIDERS: dict[str, dict[str, Any]] = {
    "gemini": {"type": "gemini", "api_key_env": "GEMINI_API_KEY", "base_url": None},
    "openai": {"type": "openai_compatible", "api_key_env": "OPENAI_API_KEY", "base_url": None},
    "groq": {"type": "openai_compatible", "api_key_env": "GROQ_API_KEY", "base_url": "https://api.groq.com/openai/v1"},
}


@dataclass(frozen=True)
class ProviderSettings:
    """One configured provider. ``type`` selects the adapter (``llm.registry.ADAPTERS``); ``options``
    are extra adapter arguments (e.g. ``max_tokens_param``)."""

    name: str
    type: str
    api_key_env: str | None = None
    base_url: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoleSettings:
    """Everything needed to build and call the model for one role."""

    role: str
    provider: ProviderSettings
    model: str
    temperature: float = 0.0
    max_output_tokens: int | None = None
    timeout_s: float | None = None
    seed: int | None = None
    max_retries: int = 3


@dataclass(frozen=True)
class LLMSettings:
    roles: dict[str, RoleSettings]
    cache_path: str | None = None

    def role(self, name: str) -> RoleSettings:
        if name not in ROLES:
            raise ValueError(f"Unknown LLM role {name!r}; roles are {', '.join(ROLES)}.")
        if name not in self.roles:
            raise ValueError(f"No LLM is configured for the {name!r} role (set llm.roles.{name} in the config).")
        return self.roles[name]


def resolve_llm_settings(config: dict[str, Any]) -> LLMSettings:
    """Read the LLM settings from a loaded config (new ``llm`` block or legacy flat keys)."""
    if "llm" in config:
        return _from_llm_block(config["llm"])
    return _from_legacy_keys(config)


def _from_llm_block(block: Any) -> LLMSettings:
    if not isinstance(block, dict):
        raise ValueError("llm must be a mapping.")
    _reject_unknown("llm", block, {"defaults", "cache_path", "providers", "roles"})
    defaults = _call_settings("llm.defaults", block.get("defaults") or {}, dict(CALL_DEFAULTS))

    providers: dict[str, ProviderSettings] = {}
    for name, raw in (block.get("providers") or {}).items():
        if not isinstance(raw, dict):
            raise ValueError(f"llm.providers.{name} must be a mapping.")
        options = {k: v for k, v in raw.items() if k not in {"type", "api_key_env", "base_url"}}
        providers[name] = _provider(name, raw.get("type"), raw.get("api_key_env"), raw.get("base_url"), options)

    roles: dict[str, RoleSettings] = {}
    for role, raw in (block.get("roles") or {}).items():
        if role not in ROLES:
            raise ValueError(f"Unknown LLM role llm.roles.{role}; roles are {', '.join(ROLES)}.")
        if not isinstance(raw, dict):
            raise ValueError(f"llm.roles.{role} must be a mapping.")
        _reject_unknown(f"llm.roles.{role}", raw, {"provider", "model", *CALL_DEFAULTS})
        provider_name = raw.get("provider")
        if provider_name not in providers:
            raise ValueError(f"llm.roles.{role}.provider {provider_name!r} is not defined in llm.providers.")
        calls = _call_settings(f"llm.roles.{role}", {k: v for k, v in raw.items() if k in CALL_DEFAULTS}, defaults)
        roles[role] = RoleSettings(role=role, provider=providers[provider_name], model=_model(role, raw), **calls)

    cache_path = block.get("cache_path")
    return LLMSettings(roles=roles, cache_path=str(cache_path) if cache_path else None)


def _from_legacy_keys(config: dict[str, Any]) -> LLMSettings:
    """Flat keys as the thesis code read them (``llm_provider``, ``llm_rewrite_model``, ...)."""
    models = {
        "rewrite": _first(config, "llm_rewrite_model", "openai_rewrite_model"),
        # The old guard repaired rewrites with the rewrite model.
        "repair": _first(config, "llm_rewrite_model", "openai_rewrite_model"),
        "answer": _first(config, "llm_answer_model", "openai_answer_model"),
        # The old judge fell back to the answer model.
        "judge": _first(config, "llm_judge_model", "llm_answer_model", "openai_answer_model"),
    }
    models = {role: model for role, model in models.items() if model}
    if not models:
        return LLMSettings(roles={})

    name = str(config.get("llm_provider") or "").strip().lower()
    if name not in LEGACY_PROVIDERS:
        known = ", ".join(sorted(LEGACY_PROVIDERS))
        raise ValueError(f"llm_provider must be one of {known} (got {name or 'nothing'}); no default is assumed.")
    known_provider = LEGACY_PROVIDERS[name]
    api_key_env = _first(config, "llm_api_key_env") or known_provider["api_key_env"]
    base_url = _first(config, "llm_api_base") or known_provider["base_url"]
    provider = _provider(name, known_provider["type"], api_key_env, base_url, {})
    roles = {
        role: RoleSettings(role=role, provider=provider, model=model, **CALL_DEFAULTS) for role, model in models.items()
    }
    return LLMSettings(roles=roles)


def _provider(name: str, type_: Any, api_key_env: Any, base_url: Any, options: dict[str, Any]) -> ProviderSettings:
    # Which types exist is decided by the registry's adapter table (checked when a model is built).
    if not isinstance(type_, str) or not type_.strip():
        raise ValueError(f"llm.providers.{name}.type must be a non-empty string (e.g. gemini, openai_compatible).")
    return ProviderSettings(
        name=name,
        type=type_,
        api_key_env=str(api_key_env) if api_key_env else None,
        base_url=str(base_url) if base_url else None,
        options=options,
    )


def _call_settings(where: str, raw: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown(where, raw, set(CALL_DEFAULTS))
    merged = {**base, **raw}
    merged["temperature"] = float(merged["temperature"])
    if not isinstance(merged["max_retries"], int) or merged["max_retries"] < 0:
        raise ValueError(f"{where}.max_retries must be a non-negative integer.")
    return merged


def _model(role: str, raw: dict[str, Any]) -> str:
    model = raw.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"llm.roles.{role}.model must be a non-empty string.")
    return model.strip()


def _first(config: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _reject_unknown(where: str, raw: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown setting(s) in {where}: {', '.join(unknown)}.")
