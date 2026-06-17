"""LLM provider configuration.

The sandbox is deliberately provider-agnostic: any caller that needs an LLM
asks for one via ``get_llm(...)``. Anthropic, OpenAI, and any OpenAI-compatible
endpoint (vLLM, Together, Fireworks, a local server, ...) are supported out of
the box; add more by editing ``_FACTORIES``.

For the persona-measurement study we sweep many models. Rather than hard-code
them here, model definitions live as data in ``sandbox/models.yaml`` and are
resolved via :func:`sandbox.registry.resolve_model_config`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


# Default models per provider — tuned for cheap-but-capable.
DEFAULT_MODELS: Dict[str, str] = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-5.4-nano",
    "openai_compatible": "Qwen/Qwen3.5-4B",
}


@dataclass
class LLMConfig:
    provider: str = "openai"              # "anthropic" | "openai" | "openai_compatible"
    model: Optional[str] = None           # falls back to DEFAULT_MODELS[provider]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    base_url: Optional[str] = None        # for openai_compatible (vLLM/local/proxy)
    api_key_env: Optional[str] = None     # env var holding the key (default per provider)
    extra: Dict[str, Any] = field(default_factory=dict)
    # Free-form, carries seed / top_p / vendor-recommended sampling / future
    # reasoning_effort. Standard knobs (seed, top_p) are forwarded as request
    # params; non-OpenAI sampling (top_k, min_p, repetition_penalty) is sent via
    # extra["extra_body"].


# Sampling keys that OpenAI/Anthropic accept directly as request params.
_PASSTHROUGH_KEYS = ("top_p", "seed")


def _sampling_kwargs(cfg: "LLMConfig") -> Dict[str, Any]:
    """Pull recognized sampling params out of ``cfg.extra``.

    Returns a dict suitable for splatting into the LangChain chat-model
    constructor. ``extra_body`` (vLLM-specific knobs) is passed through verbatim.
    """
    kwargs: Dict[str, Any] = {}
    model_kwargs: Dict[str, Any] = {}
    for k in _PASSTHROUGH_KEYS:
        if k in cfg.extra:
            model_kwargs[k] = cfg.extra[k]
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    if cfg.extra.get("extra_body"):
        kwargs["extra_body"] = cfg.extra["extra_body"]
    return kwargs


def _build_anthropic(cfg: LLMConfig):
    from langchain_anthropic import ChatAnthropic  # type: ignore
    return ChatAnthropic(
        model=cfg.model or DEFAULT_MODELS["anthropic"],
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens or 1024,
    )


def _build_openai(cfg: LLMConfig):
    from langchain_openai import ChatOpenAI  # type: ignore
    return ChatOpenAI(
        model=cfg.model or DEFAULT_MODELS["openai"],
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens or 1024,
        **_sampling_kwargs(cfg),
    )


def _build_openai_compatible(cfg: LLMConfig):
    """Any OpenAI-compatible HTTP endpoint — primarily vLLM-served Qwen.

    ``base_url`` points at the server (e.g. http://localhost:8000/v1). The key
    is read from ``cfg.api_key_env`` (default ``VLLM_API_KEY``) and defaults to
    the literal "EMPTY", which vLLM accepts when started with ``--api-key EMPTY``.
    """
    from langchain_openai import ChatOpenAI  # type: ignore
    base_url = cfg.base_url or os.getenv("OPENAI_BASE_URL")
    if not base_url:
        raise ValueError(
            "provider 'openai_compatible' requires base_url (set it in models.yaml "
            "or via OPENAI_BASE_URL)."
        )
    api_key = os.getenv(cfg.api_key_env or "VLLM_API_KEY", "EMPTY")
    return ChatOpenAI(
        model=cfg.model or DEFAULT_MODELS["openai_compatible"],
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens or 1024,
        base_url=base_url,
        api_key=api_key,
        **_sampling_kwargs(cfg),
    )


_FACTORIES: Dict[str, Callable[[LLMConfig], object]] = {
    "anthropic": _build_anthropic,
    "openai": _build_openai,
    "openai_compatible": _build_openai_compatible,
}


def get_llm(cfg: Optional[LLMConfig] = None):
    """Return a LangChain chat model for the given provider.

    If ``cfg`` is None, reads defaults from env vars:
        LLM_PROVIDER   (default: "anthropic")
        LLM_MODEL      (optional, defaults to DEFAULT_MODELS[provider])
        LLM_TEMPERATURE (optional, default 0.7)
    """
    if cfg is None:
        cfg = LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "anthropic"),
            model=os.getenv("LLM_MODEL") or None,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        )
    if cfg.provider not in _FACTORIES:
        raise ValueError(
            f"Unknown provider '{cfg.provider}'. "
            f"Supported: {list(_FACTORIES.keys())}"
        )
    return _FACTORIES[cfg.provider](cfg)
