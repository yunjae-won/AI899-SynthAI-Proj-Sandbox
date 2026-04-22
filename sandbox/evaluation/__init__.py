"""Evaluation API. Lazy imports so rule_based works without LLM deps."""

from __future__ import annotations


def score_three_axes(*args, **kwargs):
    from .rule_based import score_three_axes as _f
    return _f(*args, **kwargs)


def judge_persona_consistency(*args, **kwargs):
    from .judge import judge_persona_consistency as _f
    return _f(*args, **kwargs)


def judge_adaptation(*args, **kwargs):
    from .judge import judge_adaptation as _f
    return _f(*args, **kwargs)


def run_swap_test(*args, **kwargs):
    from .swap_test import run_swap_test as _f
    return _f(*args, **kwargs)
