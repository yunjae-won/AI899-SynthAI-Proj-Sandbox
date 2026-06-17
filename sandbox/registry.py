"""Model registry — models are *data*, not code.

Adding a model to the study is a one-line entry in ``sandbox/models.yaml``;
nothing in the Python needs to change. A registry entry maps a short ``key``
(used as ``--model-key`` on the CLI and in run-directory names) to a provider,
a concrete model id, and optionally a ``base_url`` / ``api_key_env`` (for
OpenAI-compatible / vLLM endpoints) and a ``recommended`` sampling block.

For cross-model fairness the study forces a single ``temperature`` on every
model; each vendor's ``recommended`` sampling is recorded but *not* applied by
default (pass ``use_recommended=True`` to opt in).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from .config import LLMConfig

REGISTRY_PATH = Path(__file__).resolve().parent / "models.yaml"


@lru_cache(maxsize=1)
def load_model_registry() -> Dict[str, Dict[str, Any]]:
    """Read ``models.yaml`` → ``{key: entry}``. Cached for the process."""
    import yaml  # local import so non-study code paths don't need pyyaml

    if not REGISTRY_PATH.exists():
        return {}
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("models", {})


def list_model_keys() -> List[str]:
    return sorted(load_model_registry().keys())


def resolve_model_config(
    key: str,
    *,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    use_recommended: bool = False,
) -> LLMConfig:
    """Resolve a registry ``key`` to an :class:`~sandbox.config.LLMConfig`.

    ``temperature`` is forced on every model (fairness). When
    ``use_recommended`` is True, the entry's ``recommended`` block overrides it
    (and feeds extra sampling knobs) — use only for per-model sensitivity checks.
    """
    reg = load_model_registry()
    if key not in reg:
        raise KeyError(f"Unknown model key '{key}'. Available: {list_model_keys()}")
    entry = reg[key]

    extra: Dict[str, Any] = {}
    temp = temperature
    recommended = entry.get("recommended") or {}
    if use_recommended and recommended:
        temp = recommended.get("temperature", temperature)
        # OpenAI-standard knobs go through directly; vendor-only knobs via extra_body.
        for k in ("top_p", "seed"):
            if k in recommended:
                extra[k] = recommended[k]
        extra_body = {
            k: recommended[k]
            for k in ("top_k", "min_p", "repetition_penalty", "presence_penalty")
            if k in recommended
        }
        if extra_body:
            extra["extra_body"] = extra_body

    # Always-on extra_body from the entry (applied regardless of use_recommended).
    # Used e.g. to disable Qwen3.5 "thinking" mode so the model returns short,
    # parseable JSON instead of a long chain-of-thought.
    base_eb = entry.get("extra_body") or {}
    if base_eb:
        eb = dict(extra.get("extra_body") or {})
        eb.update(base_eb)
        extra["extra_body"] = eb

    return LLMConfig(
        provider=entry["provider"],
        model=entry["model"],
        temperature=temp,
        max_tokens=max_tokens or entry.get("max_tokens"),
        base_url=entry.get("base_url"),
        api_key_env=entry.get("api_key_env"),
        extra=extra,
    )
