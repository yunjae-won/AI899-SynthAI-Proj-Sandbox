"""Deterministic, LLM-free persona-measurement metrics.

These are the *primary* metrics for the re-scoped study (the LLM judge and the
swap test are now secondary). Everything here reads only the agent's chosen
actions from a trajectory — never the world engine's effects — because that is
where the persona lives.

Three things are measured:

  1. **Behavioural divergence** — how different is the action distribution
     across personas? Computed as Jensen-Shannon divergence (base-2, in [0,1])
     between pooled action distributions of the two poles of a persona axis.
  2. **Directional fidelity** — does the agent take the *pre-registered*
     action for its pole on the axis-sensitive event? (`diagnostic_hit_rate`)
  3. **Parse-fail rate** — fraction of turns where the model failed to emit
     parseable JSON and fell back. A confound covariate: a model that "looks
     flat" may simply be failing to produce actions.

The headline artifact is the **axis × event matrix**: divergence should be
HIGH on the event sensitive to that axis and near the floor elsewhere (and on
the orthogonal control). That contrast is both the persona-effect estimate and
the metric's own discriminant-validity check.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

# Persona axis -> (pole_a, pole_b). Order is arbitrary; divergence is symmetric.
AXES: Dict[str, tuple] = {
    "value": ("achievement", "affiliation"),
    "conflict": ("avoidant", "assertive"),
    "time": ("impulsive", "deliberate"),
}

# Normalize action-name synonyms the engine treats as equivalent.
_NAME_SYNONYMS = {
    "work": "work_on", "contact": "message", "join": "attend",
    "reschedule": "defer", "move": "move_to", "boundary": "propose_boundary",
    "rest": "sleep", "noop": "wait", "pass": "wait",
}
_LABEL_ARG_KEYS = ("task", "task_id", "event", "target", "to", "role", "location")


def canonical_action(action: Dict[str, Any]) -> str:
    """Reduce a free-form action to a low-cardinality ``name`` or ``name:label``.

    e.g. ``{"name": "work_on(task, hours)", "args": {"task": "future_project"}}``
    -> ``"work_on:future_project"``.
    """
    name = str(action.get("name") or "").strip().lower()
    name = re.split(r"[(\s]", name, 1)[0] if name else "wait"
    name = _NAME_SYNONYMS.get(name, name)

    args = action.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    label = ""
    for k in _LABEL_ARG_KEYS:
        v = args.get(k)
        if v:
            label = str(v).strip().lower()
            break
    return f"{name}:{label}" if label else name


def _matches(canon: str, spec: str) -> bool:
    """A canonical action matches a diagnostic spec.

    Spec ``"work_on"`` matches any ``work_on:*``; spec ``"work_on:future"``
    matches if the canonical label *contains* ``"future"``.
    """
    spec = spec.strip().lower()
    sname, _, slabel = spec.partition(":")
    cname, _, clabel = canon.partition(":")
    if cname != _NAME_SYNONYMS.get(sname, sname):
        return False
    return (not slabel) or (slabel in clabel)


# --------------------------------------------------------------------------- #
# Distributions + divergence
# --------------------------------------------------------------------------- #


def action_counts(trajectory: List[Dict[str, Any]]) -> Counter:
    return Counter(canonical_action(step.get("action", {})) for step in trajectory)


def action_distribution(trajectory: List[Dict[str, Any]]) -> Dict[str, float]:
    counts = action_counts(trajectory)
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def js_divergence(counts_a, counts_b) -> Optional[float]:
    """Jensen-Shannon divergence (base 2, in [0,1]) between two count maps.

    Returns None if either side has no observations (nothing to compare).
    """
    a, b = Counter(counts_a), Counter(counts_b)
    if not sum(a.values()) or not sum(b.values()):
        return None
    keys = set(a) | set(b)
    ta, tb = sum(a.values()), sum(b.values())
    jsd = 0.0
    for k in keys:
        p = a.get(k, 0) / ta
        q = b.get(k, 0) / tb
        m = 0.5 * (p + q)
        if p > 0:
            jsd += 0.5 * p * math.log2(p / m)
        if q > 0:
            jsd += 0.5 * q * math.log2(q / m)
    return max(0.0, min(1.0, jsd))


def pole_divergence(records: List[Dict[str, Any]], axis: str, event_id: str) -> Optional[float]:
    """JSD between the two poles of ``axis`` on a single event.

    Pools canonical actions of every record at pole_a vs pole_b (the other two
    axes vary within each pool and are thereby marginalized).
    """
    pole_a, pole_b = AXES[axis]
    ca, cb = Counter(), Counter()
    for r in records:
        if r.get("event_id") != event_id:
            continue
        pole = (r.get("axes") or {}).get(axis)
        if pole == pole_a:
            ca.update(action_counts(r["trajectory"]))
        elif pole == pole_b:
            cb.update(action_counts(r["trajectory"]))
    return js_divergence(ca, cb)


# --------------------------------------------------------------------------- #
# Directional fidelity + parse-fail covariate
# --------------------------------------------------------------------------- #


def parse_fail_rate(trajectory: List[Dict[str, Any]]) -> float:
    if not trajectory:
        return 0.0
    fails = sum(1 for s in trajectory if (s.get("action") or {}).get("parse_failed"))
    return fails / len(trajectory)


def diagnostic_hit_rate(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fraction of turns whose action matches the pre-registered action for
    this persona's pole on the event's sensitive axis.

    Returns None for the orthogonal control (no sensitive axis / no diagnostics).
    """
    axis = record.get("sensitive_axis")
    if not axis or axis == "none":
        return None
    pole = (record.get("axes") or {}).get(axis)
    specs = (record.get("diagnostic_actions") or {}).get(pole)
    if not specs:
        return None
    traj = record["trajectory"]
    hits = sum(
        1 for s in traj
        if any(_matches(canonical_action(s.get("action", {})), spec) for spec in specs)
    )
    total = len(traj) or 1
    return {"pole": pole, "hits": hits, "total": len(traj), "rate": hits / total}


# --------------------------------------------------------------------------- #
# Aggregations
# --------------------------------------------------------------------------- #


def build_axis_event_matrix(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    """``{axis: {event_id: jsd}}`` — the headline diagnostic matrix.

    Pass records for a *single* agent (filter by agent upstream); the diagonal
    (axis-sensitive event) should dominate its row.
    """
    event_ids = sorted({r["event_id"] for r in records})
    return {
        axis: {ev: pole_divergence(records, axis, ev) for ev in event_ids}
        for axis in AXES
    }


def persona_effect_contrast(
    matrix: Dict[str, Dict[str, Optional[float]]],
    event_meta: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """For each axis: divergence on its sensitive event minus the orthogonal floor.

    ``event_meta`` maps event_id -> sensitive_axis. A large positive contrast is
    the persona effect; a contrast near zero means persona is not driving behaviour.
    """
    orth = [e for e, ax in event_meta.items() if ax in (None, "none")]
    out: Dict[str, Dict[str, Any]] = {}
    for axis in AXES:
        # Average over ALL events sensitive to this axis (supports k>=2 per axis).
        sensitive = [e for e, ax in event_meta.items() if ax == axis]
        s_vals = [matrix.get(axis, {}).get(e) for e in sensitive]
        s_vals = [v for v in s_vals if v is not None]
        s_jsd = (sum(s_vals) / len(s_vals)) if s_vals else None
        o_id = orth[0] if orth else None
        o_jsd = matrix.get(axis, {}).get(o_id) if o_id else None
        contrast = (s_jsd - o_jsd) if (s_jsd is not None and o_jsd is not None) else None
        out[axis] = {
            "sensitive_events": sensitive, "sensitive_jsd": s_jsd,
            "orthogonal_event": o_id, "orthogonal_jsd": o_jsd,
            "contrast": contrast,
        }
    return out


def summarize_agent(
    records: List[Dict[str, Any]],
    event_meta: Dict[str, str],
) -> Dict[str, Any]:
    """One agent's full measurement bundle from its run records."""
    matrix = build_axis_event_matrix(records)
    diagnostics: List[Dict[str, Any]] = []
    for r in records:
        hr = diagnostic_hit_rate(r)
        diagnostics.append({
            "persona_id": r["persona_id"],
            "event_id": r["event_id"],
            "hit_rate": hr,
            "parse_fail_rate": parse_fail_rate(r["trajectory"]),
        })
    mean_parse_fail = (
        sum(d["parse_fail_rate"] for d in diagnostics) / len(diagnostics)
        if diagnostics else 0.0
    )
    return {
        "axis_event_matrix": matrix,
        "persona_effect": persona_effect_contrast(matrix, event_meta),
        "diagnostics": diagnostics,
        "mean_parse_fail_rate": mean_parse_fail,
    }
