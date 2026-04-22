"""Persona-swap consistency test.

The idea: if persona P1's trajectory really reflects P1 (and not P2), then a
blind LLM judge, shown ONLY the goal+action sequence, should pick P1 as the
better-matching persona more often than P2. This is a relative signal --
useful because we have no "golden" trajectories.

We run pairwise contests:

    run_i = trajectory produced by (persona_i, event, agent)
    For each pair (i, j) with i != j:
        judge is shown (run_i) and asked: is this P_i or P_j?
    Score = accuracy across all such pairs.

A well-in-character agent produces trajectories that are distinguishable by
their persona; a flattened / generic agent does not.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Tuple

from ..config import LLMConfig, get_llm
from ..agents._llm_utils import call_llm, parse_json
from .judge import _trajectory_view, JUDGE_SYSTEM


SWAP_PROMPT = """You will see one trajectory of goals, actions, and reasoning
from a synthetic-person agent. You are also given two candidate personas,
labeled A and B. Your job is to decide which persona MORE LIKELY produced
this trajectory, based on value priorities, communication style, and
decision tendencies.

PERSONA A:
{persona_a}

PERSONA B:
{persona_b}

TRAJECTORY:
{trajectory_view}

Return JSON:
{{
  "choice": "A" or "B",
  "confidence": 1-5,
  "justification": "one-line reason pointing to specific persona fields"
}}"""


def _one_match(
    llm,
    trajectory: List[Dict[str, Any]],
    persona_true: Dict[str, Any],
    persona_distractor: Dict[str, Any],
) -> Dict[str, Any]:
    # Randomize A/B so the judge can't cheat by position.
    if random.random() < 0.5:
        a, b, true_label = persona_true, persona_distractor, "A"
    else:
        a, b, true_label = persona_distractor, persona_true, "B"

    user = SWAP_PROMPT.format(
        persona_a=json.dumps(a, ensure_ascii=False, indent=2),
        persona_b=json.dumps(b, ensure_ascii=False, indent=2),
        trajectory_view=_trajectory_view(trajectory, max_turns=12),
    )
    raw = call_llm(llm, JUDGE_SYSTEM, user)
    out = parse_json(raw, fallback={"choice": "?", "confidence": 0, "justification": raw[:200]})
    out["true_label"] = true_label
    out["correct"] = (out.get("choice") == true_label)
    return out


def run_swap_test(
    runs: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]],
    llm_config: Optional[LLMConfig] = None,
    trials_per_pair: int = 1,
    seed: Optional[int] = 0,
) -> Dict[str, Any]:
    """Run pairwise swap tests.

    ``runs`` is a list of ``(persona_dict, trajectory)`` pairs -- one entry
    per (persona × event × agent) episode you want to compare. The usual
    setup is: same agent, same event, all 3 personas. The judge is then
    asked, for each trajectory, to pick the true persona against each of
    the others.
    """
    if seed is not None:
        random.seed(seed)
    llm = get_llm(llm_config)

    matches: List[Dict[str, Any]] = []
    for i, (persona_true, traj) in enumerate(runs):
        for j, (persona_other, _) in enumerate(runs):
            if i == j:
                continue
            for _ in range(trials_per_pair):
                result = _one_match(llm, traj, persona_true, persona_other)
                result["true_persona_id"] = persona_true.get("id")
                result["distractor_persona_id"] = persona_other.get("id")
                matches.append(result)

    n = len(matches)
    correct = sum(1 for m in matches if m.get("correct"))
    return {
        "n_matches": n,
        "accuracy": (correct / n) if n else 0.0,
        "matches": matches,
    }
