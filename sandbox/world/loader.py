"""Load personas and events from the ``src/`` library."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Resolve relative to the repo root (parent of sandbox/).
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
PERSONA_DIR = SRC_DIR / "personas"
EVENT_DIR = SRC_DIR / "events"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_personas() -> List[str]:
    return sorted(p.stem for p in PERSONA_DIR.glob("*.json"))


def list_events() -> List[str]:
    return sorted(p.stem for p in EVENT_DIR.glob("*.json"))


def load_persona(persona_id: str) -> Dict[str, Any]:
    """Load a persona by id (with or without .json)."""
    persona_id = persona_id.removesuffix(".json")
    path = PERSONA_DIR / f"{persona_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Persona '{persona_id}' not found. Available: {list_personas()}"
        )
    return _load_json(path)


def load_event(event_id: str) -> Dict[str, Any]:
    event_id = event_id.removesuffix(".json")
    path = EVENT_DIR / f"{event_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Event '{event_id}' not found. Available: {list_events()}"
        )
    return _load_json(path)


def build_initial_sim_state(
    persona: Dict[str, Any],
    event: Dict[str, Any],
    max_turns: int = 12,
) -> Dict[str, Any]:
    """Assemble a fresh ``SimState`` dict from a persona + event pair."""
    init = event.get("initial_state", {})
    world = init.get("world", {})
    agent_init = init.get("agent", {})
    baseline = persona.get("baseline_state", {})

    # Agent state merges persona baseline with event-specific overrides.
    agent: Dict[str, Any] = {
        "energy": agent_init.get("energy", baseline.get("energy", 6)),
        "sleep_debt_hours": agent_init.get(
            "sleep_debt_hours", baseline.get("sleep_debt_hours", 0)
        ),
        "stress": agent_init.get("stress", baseline.get("stress", 3)),
        "memory": [],
        "desires": [],
        "goals": [],
        "last_reasoning": "",
    }

    return {
        "persona": persona,
        "event": event,
        "current_time": world.get("current_time", "T+0"),
        "location": world.get("location", "unknown"),
        "tasks": list(init.get("tasks", [])),
        "relationships": list(init.get("relationships", [])),
        "pending_events": list(event.get("events_queue", [])),
        "agent": agent,
        "new_observations": list(event.get("visible_information", [])),
        "turn": 0,
        "max_turns": max_turns,
        "trajectory": [],
        "finished": False,
    }
