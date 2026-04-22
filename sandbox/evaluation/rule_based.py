"""Rule-based / heuristic 3-axis scoring from ``event.evaluation``.

This is the cheap, deterministic part of the evaluation. It scans the
trajectory for:

  * Hard-loss conditions  : pattern-matched against final state + messages
  * Context-switching     : does reasoning text mention multiple domains?
  * Efficiency            : does an indicator substring appear in any action?

These scores are useful as a first filter; the LLM judges (``judge.py``)
pick up what heuristics miss.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


def _gather_reasoning_text(trajectory: List[Dict[str, Any]]) -> str:
    return " ".join(
        (step.get("action") or {}).get("reasoning", "") for step in trajectory
    )


def _gather_all_text(trajectory: List[Dict[str, Any]]) -> str:
    pieces: List[str] = []
    for step in trajectory:
        act = step.get("action") or {}
        pieces.append(act.get("reasoning", ""))
        pieces.append(str(act.get("name", "")))
        pieces.append(str(act.get("args", "")))
        pieces.append(step.get("observation", ""))
    return " ".join(pieces).lower()


# --------------------------------------------------------------------------- #
# Hard-loss checks
# --------------------------------------------------------------------------- #


_HARD_LOSS_PATTERNS = {
    # event_01
    "assignment_a_missed": [
        # Looks for assignment progress < 0.7 by Wed 23:59, or explicit miss.
        lambda final: any(
            t.get("id") == "assignment_A" and t.get("progress", 0) < 0.7
            for t in final.get("tasks", [])
        ),
    ],
    "quiz_skipped": [
        # Heuristic: quiz_prep progress stays at 0 by end of run.
        lambda final: any(
            t.get("id") == "quiz_prep" and t.get("progress", 0) <= 0.01
            for t in final.get("tasks", [])
        ),
    ],
    "trust_drop": [
        # Relative trust drop >=2 vs. initial.
        lambda final, init=None: False,  # resolved per-run below
    ],
    # event_02
    "three_hour_waste": [
        lambda final: all(t.get("progress", 0) <= 0.01 for t in final.get("tasks", [])),
    ],
}


def _trust_drop_detected(
    initial: List[Dict[str, Any]],
    final: List[Dict[str, Any]],
    threshold: float = 2.0,
) -> bool:
    init_map = {r.get("role") or r.get("name"): r.get("trust", 5) for r in initial}
    for r in final:
        key = r.get("role") or r.get("name")
        if key in init_map and init_map[key] - r.get("trust", 5) >= threshold:
            return True
    return False


def _evaluate_hard_loss(
    event: Dict[str, Any],
    initial_relationships: List[Dict[str, Any]],
    final_state: Dict[str, Any],
) -> Dict[str, Any]:
    conditions = (event.get("evaluation") or {}).get("hard_loss_conditions", [])
    triggered: List[str] = []

    # Per-condition keyword matching. We lift a small set of well-known
    # failure modes and leave the rest as LLM-judge territory.
    tasks = final_state.get("tasks", [])
    lowered = [c.lower() for c in conditions]
    for cond in conditions:
        c = cond.lower()
        if "assignment_a" in c or "assignment a" in c:
            if any(t.get("id") == "assignment_A" and t.get("progress", 0) < 0.7 for t in tasks):
                triggered.append(cond)
        elif "quiz" in c:
            if any(t.get("id") == "quiz_prep" and t.get("progress", 0) <= 0.01 for t in tasks):
                triggered.append(cond)
        elif "trust" in c:
            if _trust_drop_detected(
                initial_relationships, final_state.get("relationships", []),
                threshold=2.0,
            ):
                triggered.append(cond)
        elif "progress 0" in c or "all 3 tasks" in c:
            if all(t.get("progress", 0) <= 0.01 for t in tasks):
                triggered.append(cond)

    return {
        "conditions": conditions,
        "triggered": triggered,
        "any_hard_loss": bool(triggered),
    }


# --------------------------------------------------------------------------- #
# Context-switching & efficiency indicators
# --------------------------------------------------------------------------- #


_DOMAIN_KEYWORDS = {
    "academic": ["assignment", "quiz", "study", "paper", "exam", "과제", "퀴즈"],
    "relationships": ["teammate", "friend", "professor", "message", "팀원", "친구"],
    "health": ["sleep", "rest", "energy", "tired", "수면", "피로"],
    "growth": ["language", "workout", "learn", "practice", "연습"],
    "leisure": ["lunch", "meal", "break", "hang out", "식사"],
}


def _context_switch_score(trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    text = _gather_reasoning_text(trajectory).lower()
    hit_domains = [d for d, kws in _DOMAIN_KEYWORDS.items() if any(k in text for k in kws)]
    cross_mentions = 0
    for step in trajectory:
        r = (step.get("action") or {}).get("reasoning", "").lower()
        domains_in_this_step = sum(
            1 for kws in _DOMAIN_KEYWORDS.values() if any(k in r for k in kws)
        )
        if domains_in_this_step >= 2:
            cross_mentions += 1
    return {
        "domains_mentioned": hit_domains,
        "cross_domain_steps": cross_mentions,
        # Normalized roughly: 1.0 if at least one cross-domain step per 3 turns.
        "score": min(1.0, cross_mentions / max(1.0, len(trajectory) / 3.0)),
    }


def _efficiency_score(event: Dict[str, Any], trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    indicators = (event.get("evaluation") or {}).get("efficiency_indicators", [])
    text = _gather_all_text(trajectory)
    matched: List[str] = []
    for ind in indicators:
        # Pull 2-3 salient tokens per indicator and require any to appear.
        tokens = [t for t in re.split(r"\W+", ind.lower()) if len(t) >= 4]
        if any(tok in text for tok in tokens[:4]):
            matched.append(ind)
    return {
        "indicators": indicators,
        "matched": matched,
        "score": (len(matched) / max(1, len(indicators))) if indicators else 0.0,
    }


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def score_three_axes(
    event: Dict[str, Any],
    final_state: Dict[str, Any],
    trajectory: List[Dict[str, Any]],
    initial_relationships: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Compute hard_loss / context_switching / efficiency for one run."""
    init_rels = initial_relationships or (
        event.get("initial_state", {}).get("relationships", [])
    )
    hl = _evaluate_hard_loss(event, init_rels, final_state)
    cs = _context_switch_score(trajectory)
    ef = _efficiency_score(event, trajectory)

    # Pass/fail per the project brief:
    #   hard_loss >= 1 -> FAIL
    #   hard_loss == 0 AND (cs or ef >= 0.5) -> PASS
    #   both strong -> EXCELLENT
    if hl["any_hard_loss"]:
        verdict = "fail"
    elif cs["score"] >= 0.5 and ef["score"] >= 0.5:
        verdict = "excellent"
    elif cs["score"] >= 0.5 or ef["score"] >= 0.5:
        verdict = "pass"
    else:
        verdict = "weak_pass"

    return {
        "hard_loss": hl,
        "context_switching": cs,
        "efficiency": ef,
        "verdict": verdict,
    }
