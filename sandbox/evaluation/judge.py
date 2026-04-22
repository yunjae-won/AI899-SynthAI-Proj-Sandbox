"""LLM-as-judge evaluators.

We expose two judges:

``judge_persona_consistency``
    Takes a full trajectory (goals + actions) and a persona spec, and asks a
    judge model to score how well the trajectory stays in-character. This is
    the "does it feel like this person?" question.

``judge_adaptation``
    Compares pre-event vs. post-event goal/action segments and asks the judge
    whether the agent adapted *reasonably* to the new constraint or event.
    "Reasonable" is persona-conditioned: a Tired Achiever stopping mid-task to
    respond to a crisis is adaptation; dropping everything to hang out with
    friends is not.

Both return structured JSON dicts. All LLM calls are single-shot, stateless.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..config import LLMConfig, get_llm
from ..agents._llm_utils import call_llm, parse_json


JUDGE_SYSTEM = """You are an impartial evaluator of synthetic-person agents.
You DO NOT role-play the persona; you are an outside evaluator. Score
strictly according to the rubric and return valid JSON only."""


PERSONA_CONSISTENCY_PROMPT = """Below is (1) a persona specification and
(2) a trajectory of goals/actions/reasoning the agent produced in a single
event. Score how consistently the trajectory reflects this persona across
five dimensions. For each, return an integer 1-5 (1=contradicts persona,
5=strongly expresses persona) and a one-line justification.

Dimensions:
  - value_alignment        (matches value_priorities weighting)
  - communication_style    (assertiveness / directness / conflict_style)
  - decision_tendencies    (planning_horizon / risk / procrastination / sleep)
  - goal_coherence         (goals at t+1 evolve plausibly from goals at t,
                            don't snap to a different personality)
  - weakness_expression    (the persona's known weaknesses show up --
                            a flawless Tired Achiever is SUSPICIOUS)

PERSONA:
{persona}

TRAJECTORY (goals + action + reasoning per turn):
{trajectory_view}

Return JSON:
{{
  "scores": {{
    "value_alignment": {{"score": 1, "note": "..."}},
    "communication_style": {{"score": 1, "note": "..."}},
    "decision_tendencies": {{"score": 1, "note": "..."}},
    "goal_coherence": {{"score": 1, "note": "..."}},
    "weakness_expression": {{"score": 1, "note": "..."}}
  }},
  "overall": {{"score": 1, "summary": "..."}}
}}"""


ADAPTATION_PROMPT = """You are judging whether an agent ADAPTED appropriately
after a new event or constraint was introduced mid-episode. Compare the
goals/actions BEFORE the injected event vs. AFTER.

A good adaptation:
  - Acknowledges the new constraint explicitly in reasoning
  - Revises goals rather than continuing the old plan unchanged
  - Stays in-character (doesn't suddenly become a different persona)
  - Avoids extreme reactions disproportionate to the constraint

A bad adaptation:
  - Ignores the event
  - Overreacts (drops all prior goals for a minor update)
  - Abandons persona to "solve" the problem

PERSONA:
{persona}

INJECTED EVENT (the new constraint):
{injected_event}

BEFORE ({n_before} turns):
{before}

AFTER ({n_after} turns):
{after}

Return JSON:
{{
  "acknowledged": true/false,
  "revised_goals": true/false,
  "in_character": true/false,
  "proportionate": true/false,
  "score": 1-5,
  "summary": "..."
}}"""


# --------------------------------------------------------------------------- #
# Trajectory view helpers
# --------------------------------------------------------------------------- #


def _trajectory_view(trajectory: List[Dict[str, Any]], max_turns: Optional[int] = None) -> str:
    if max_turns is not None:
        trajectory = trajectory[:max_turns]
    lines: List[str] = []
    for step in trajectory:
        act = step.get("action") or {}
        snap = step.get("state_snapshot", {}) or {}
        agent = snap.get("agent", {})
        goals = agent.get("goals", [])
        lines.append(
            f"turn {step.get('turn')}: "
            f"goals={json.dumps(goals, ensure_ascii=False)} "
            f"action={json.dumps({'name': act.get('name'), 'args': act.get('args')}, ensure_ascii=False)} "
            f"reasoning={act.get('reasoning', '')!r} "
            f"observation={step.get('observation', '')!r}"
        )
    return "\n".join(lines) or "(empty)"


# --------------------------------------------------------------------------- #
# Public judges
# --------------------------------------------------------------------------- #


def judge_persona_consistency(
    persona: Dict[str, Any],
    trajectory: List[Dict[str, Any]],
    llm_config: Optional[LLMConfig] = None,
) -> Dict[str, Any]:
    llm = get_llm(llm_config)
    user = PERSONA_CONSISTENCY_PROMPT.format(
        persona=json.dumps(persona, ensure_ascii=False, indent=2),
        trajectory_view=_trajectory_view(trajectory),
    )
    raw = call_llm(llm, JUDGE_SYSTEM, user)
    return parse_json(raw, fallback={"raw": raw, "error": "parse_failed"})


def judge_adaptation(
    persona: Dict[str, Any],
    trajectory: List[Dict[str, Any]],
    split_turn: int,
    injected_event: Dict[str, Any],
    llm_config: Optional[LLMConfig] = None,
) -> Dict[str, Any]:
    """Split trajectory at ``split_turn`` and judge pre/post adaptation."""
    before = [s for s in trajectory if s.get("turn", 0) < split_turn]
    after = [s for s in trajectory if s.get("turn", 0) >= split_turn]
    llm = get_llm(llm_config)
    user = ADAPTATION_PROMPT.format(
        persona=json.dumps(persona, ensure_ascii=False, indent=2),
        injected_event=json.dumps(injected_event, ensure_ascii=False, indent=2),
        n_before=len(before),
        n_after=len(after),
        before=_trajectory_view(before),
        after=_trajectory_view(after),
    )
    raw = call_llm(llm, JUDGE_SYSTEM, user)
    return parse_json(raw, fallback={"raw": raw, "error": "parse_failed"})
