"""Agent registry with lazy imports.

Importing ``sandbox.agents`` does **not** pull in LangGraph or provider SDKs;
heavy imports happen only when a factory is actually requested via
``get_agent_builder(name)`` or ``AGENT_REGISTRY[name]``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict


def _lazy_full(cfg=None):
    from .full_agent import build_full_agent
    return build_full_agent(cfg)


def _lazy_prompt_only(cfg=None):
    from .baselines import build_prompt_only_agent
    return build_prompt_only_agent(cfg)


def _lazy_memory_only(cfg=None):
    from .baselines import build_memory_only_agent
    return build_memory_only_agent(cfg)


def _lazy_no_desire(cfg=None):
    from .baselines import build_no_desire_agent
    return build_no_desire_agent(cfg)


AGENT_REGISTRY: Dict[str, Callable[..., Any]] = {
    "full": _lazy_full,
    "prompt_only": _lazy_prompt_only,
    "memory_only": _lazy_memory_only,
    "no_desire": _lazy_no_desire,
}


def get_agent_builder(name: str) -> Callable[..., Any]:
    if name not in AGENT_REGISTRY:
        raise KeyError(f"Unknown agent '{name}'. Available: {list(AGENT_REGISTRY)}")
    return AGENT_REGISTRY[name]
