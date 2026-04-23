"""LLM provider configuration.

The sandbox is deliberately provider-agnostic: any caller that needs an LLM
asks for one via ``get_llm(...)``. Both Anthropic and OpenAI are supported out
of the box; add more by editing ``_FACTORIES``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional


# Default models per provider — tuned for cheap-but-capable.
DEFAULT_MODELS: Dict[str, str] = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-5.4-nano",
}


@dataclass
class LLMConfig:
    provider: str = "openai"         # "anthropic" | "openai"
    model: Optional[str] = None      # falls back to DEFAULT_MODELS[provider]
    temperature: float = 0.7
    max_tokens: int = None    


def _build_anthropic(cfg: LLMConfig):
    from langchain_anthropic import ChatAnthropic  # type: ignore
    return ChatAnthropic(
        model=cfg.model or DEFAULT_MODELS["anthropic"],
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )


def _build_openai(cfg: LLMConfig):
    from langchain_openai import ChatOpenAI  # type: ignore
    return ChatOpenAI(
        model=cfg.model or DEFAULT_MODELS["openai"],
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )


_FACTORIES: Dict[str, Callable[[LLMConfig], object]] = {
    "anthropic": _build_anthropic,
    "openai": _build_openai,
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
