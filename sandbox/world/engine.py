"""Turn-based simulation driver.

The engine doesn't know anything about LangGraph — it calls an ``agent_step``
callable that takes a ``SimState`` and returns an action dict. Between turns,
the engine advances time, fires scheduled events from ``pending_events``, and
applies a minimal rule-based effect for each action so that ``tasks`` /
``agent`` / ``relationships`` drift in plausible directions.

Design choice (per the project brief): mechanics stay intentionally light.
The point is to *observe* the agent's goal evolution, not to simulate a
realistic undergrad in full fidelity.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .loader import build_initial_sim_state


# --------------------------------------------------------------------------- #
# Time handling
# --------------------------------------------------------------------------- #

_DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _parse_time(label: str) -> Optional[int]:
    """Parse 'Wed 14:00' → absolute minute count across the week.

    Returns None if the label doesn't match. Used to order events and
    progress time; nothing depends on precise wall-clock semantics.
    """
    m = re.match(r"(\w{3})\s+(\d{1,2}):(\d{2})", label)
    if not m:
        return None
    day, hh, mm = m.groups()
    if day not in _DAYS:
        return None
    return (_DAYS.index(day) * 24 * 60) + int(hh) * 60 + int(mm)


def _format_time(minutes: int) -> str:
    day_idx = (minutes // (24 * 60)) % 7
    rem = minutes % (24 * 60)
    return f"{_DAYS[day_idx]} {rem // 60:02d}:{rem % 60:02d}"


def _advance_time(label: str, hours: float) -> str:
    base = _parse_time(label)
    if base is None:
        return label
    return _format_time(base + int(round(hours * 60)))


# --------------------------------------------------------------------------- #
# Action effects (rule-based, deliberately simple)
# --------------------------------------------------------------------------- #


def _find_task(tasks: List[Dict[str, Any]], task_id: str) -> Optional[Dict[str, Any]]:
    for t in tasks:
        if t.get("id") == task_id or t.get("label", "").lower().startswith(task_id.lower()):
            return t
    return None


def _find_relationship(
    relationships: List[Dict[str, Any]], role_or_name: str
) -> Optional[Dict[str, Any]]:
    key = role_or_name.lower()
    for r in relationships:
        if r.get("role", "").lower() == key or r.get("name", "").lower() == key:
            return r
    return None


def _apply_action(state: Dict[str, Any], action: Dict[str, Any]) -> str:
    """Mutate ``state`` in-place given an action. Returns a short observation."""
    name = (action.get("name") or "").lower()
    args = action.get("args", {}) or {}
    agent = state["agent"]
    obs_lines: List[str] = []

    # Normalize common shape: args may arrive as a dict or a free-form string.
    if isinstance(args, str):
        args = {"raw": args}

    if "work_on" in name or name == "work":
        task_ref = args.get("task") or args.get("task_id") or ""
        hours = float(args.get("hours", 1))
        task = _find_task(state["tasks"], task_ref)
        if task is not None:
            # Progress scales with current focus (energy proxy).
            focus = max(0.2, min(1.0, agent.get("energy", 5) / 10.0))
            delta = 0.15 * hours * focus
            task["progress"] = min(1.0, task.get("progress", 0.0) + delta)
            agent["energy"] = max(0.0, agent.get("energy", 5) - 0.6 * hours)
            agent["stress"] = max(0.0, agent.get("stress", 3) - 0.1)
            state["current_time"] = _advance_time(state["current_time"], hours)
            obs_lines.append(
                f"Worked on {task['label']} for {hours}h; progress={task['progress']:.2f}."
            )
        else:
            obs_lines.append(f"No matching task '{task_ref}'.")

    elif name.startswith("sleep") or name == "rest":
        hours = float(args.get("hours", 1))
        agent["energy"] = min(10.0, agent.get("energy", 5) + 1.5 * hours)
        agent["sleep_debt_hours"] = max(0.0, agent.get("sleep_debt_hours", 0) - hours)
        agent["stress"] = max(0.0, agent.get("stress", 3) - 0.5)
        state["current_time"] = _advance_time(state["current_time"], hours)
        obs_lines.append(f"Slept {hours}h; energy={agent['energy']:.1f}.")

    elif name.startswith("message") or name == "contact":
        target = args.get("role") or args.get("target") or args.get("to") or ""
        content = args.get("content", "")
        rel = _find_relationship(state["relationships"], target)
        if rel is not None:
            # Proactive messaging nudges trust up slightly.
            rel["trust"] = min(10.0, rel.get("trust", 5) + 0.2)
            obs_lines.append(
                f"Messaged {rel.get('name') or rel.get('role')}: '{content[:60]}'."
            )
        else:
            obs_lines.append(f"Tried to contact unknown role '{target}'.")
        state["current_time"] = _advance_time(state["current_time"], 0.25)

    elif name.startswith("defer") or name == "reschedule":
        task_ref = args.get("task") or args.get("task_id") or ""
        task = _find_task(state["tasks"], task_ref)
        new_deadline = args.get("new_deadline", "")
        if task is not None:
            task["deadline"] = new_deadline or task.get("deadline")
            task["hardness"] = "soft"  # negotiated => softer
            obs_lines.append(f"Requested reschedule of {task['label']} → {new_deadline}.")
        state["current_time"] = _advance_time(state["current_time"], 0.25)

    elif name.startswith("attend") or name == "join":
        what = args.get("event") or args.get("target") or "event"
        # Attending something plausibly reveals a hidden state; engine doesn't
        # literally flip the variable, but we surface that in the observation
        # so the agent can reason that it "learned something".
        obs_lines.append(f"Attended {what}. (May reveal hidden information.)")
        state["current_time"] = _advance_time(state["current_time"], 1.0)
        agent["energy"] = max(0.0, agent.get("energy", 5) - 0.3)

    elif name.startswith("move_to") or name == "move":
        loc = args.get("location", "")
        state["location"] = loc or state["location"]
        state["current_time"] = _advance_time(state["current_time"], 0.25)
        obs_lines.append(f"Moved to {state['location']}.")

    elif name.startswith("propose_boundary") or name == "boundary":
        duration = args.get("duration_minutes", 30)
        obs_lines.append(f"Proposed a {duration}-minute bounded participation.")
        state["current_time"] = _advance_time(state["current_time"], 0.25)

    elif name in ("wait", "noop", "pass"):
        state["current_time"] = _advance_time(state["current_time"], 0.5)
        obs_lines.append("Waited.")

    else:
        # Free-form / unknown action: treat as narrative step and burn 30m.
        raw = args.get("raw") or action.get("description") or name
        state["current_time"] = _advance_time(state["current_time"], 0.5)
        obs_lines.append(f"Performed free-form action: {raw}.")

    return " ".join(obs_lines) or "(no observable change)"


# --------------------------------------------------------------------------- #
# Event queue firing
# --------------------------------------------------------------------------- #


def _fire_due_events(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Remove any scheduled events whose timestamp <= current_time; return them."""
    now = _parse_time(state["current_time"])
    if now is None:
        return []
    due: List[Dict[str, Any]] = []
    remaining: List[Dict[str, Any]] = []
    for ev in state.get("pending_events", []):
        ts = _parse_time(ev.get("timestamp", ""))
        if ts is not None and ts <= now:
            due.append(ev)
        else:
            remaining.append(ev)
    state["pending_events"] = remaining
    return due


# --------------------------------------------------------------------------- #
# Top-level run
# --------------------------------------------------------------------------- #


def run_simulation(
    *,
    persona: Dict[str, Any],
    event: Dict[str, Any],
    agent_step: Callable[[Dict[str, Any]], Dict[str, Any]],
    max_turns: int = 10,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run one (persona × event × agent) episode to completion.

    ``agent_step`` receives the live ``SimState`` and must return a dict with
    at least ``{"action": {"name": str, "args": dict, "reasoning": str}}``.
    Any other keys it returns (e.g. ``goals``, ``desires``, ``memory``) are
    merged back into ``state["agent"]`` so that graph-internal bookkeeping
    (memory etc.) persists across turns.

    Returns the final ``SimState``.
    """
    state = build_initial_sim_state(persona, event, max_turns=max_turns)

    while not state["finished"] and state["turn"] < state["max_turns"]:
        # 1. Deliver any due scheduled events as observations.
        new_events = _fire_due_events(state)
        for ev in new_events:
            state["new_observations"].append({
                "timestamp": ev.get("timestamp", state["current_time"]),
                "channel": ev.get("type", "scheduled_event"),
                "content": ev.get("content", ""),
            })

        # 2. Ask the agent for a decision.
        step_result = agent_step(copy.deepcopy(state))
        action = step_result.get("action") or {"name": "wait", "args": {}, "reasoning": ""}

        # Merge agent-internal updates back into state.
        for k in ("memory", "desires", "goals", "last_reasoning"):
            if k in step_result:
                state["agent"][k] = step_result[k]

        # 3. Snapshot pre-action state, apply action, log trajectory.
        snapshot = {
            "current_time": state["current_time"],
            "location": state["location"],
            "agent": {
                k: state["agent"].get(k)
                for k in ("energy", "sleep_debt_hours", "stress", "goals", "desires")
            },
            "tasks": copy.deepcopy(state["tasks"]),
            "relationships": copy.deepcopy(state["relationships"]),
        }
        observation = _apply_action(state, action)
        state["trajectory"].append({
            "turn": state["turn"],
            "timestamp": state["current_time"],
            "state_snapshot": snapshot,
            "action": action,
            "observation": observation,
        })

        # 4. Consume observations — they've been "seen" this turn.
        state["new_observations"] = []
        state["turn"] += 1

        # 5. Termination heuristics.
        all_done = all(t.get("progress", 0) >= 0.99 for t in state["tasks"])
        if all_done:
            state["finished"] = True

    # Write trajectory log if requested.
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "persona_id": persona.get("id"),
                    "event_id": event.get("id"),
                    "final_state": {
                        "current_time": state["current_time"],
                        "tasks": state["tasks"],
                        "relationships": state["relationships"],
                        "agent": state["agent"],
                    },
                    "trajectory": state["trajectory"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    return state
