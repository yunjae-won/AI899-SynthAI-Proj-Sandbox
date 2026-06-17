"""Load personas and events.

Stimuli come from two roots that are merged transparently:

  1. the upstream ``src/`` submodule (a teammate's design-library repo), and
  2. a sandbox-local ``sandbox/library/`` directory authored in *this* repo.

The local library is searched FIRST, so a local file may override an upstream
id of the same name; otherwise the two simply union. This lets us grow the
persona/event set (e.g. the controlled factorial for the measurement study)
without modifying the upstream submodule.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Resolve relative to the repo root (parent of sandbox/).
REPO_ROOT = Path(__file__).resolve().parents[2]
SANDBOX_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
LIBRARY_DIR = SANDBOX_DIR / "library"

# Local library takes precedence over upstream src/.
PERSONA_DIRS: List[Path] = [LIBRARY_DIR / "personas", SRC_DIR / "personas"]
EVENT_DIRS: List[Path] = [LIBRARY_DIR / "events", SRC_DIR / "events"]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _list_stems(dirs: List[Path]) -> List[str]:
    stems: set[str] = set()
    for d in dirs:
        if d.exists():
            stems.update(p.stem for p in d.glob("*.json"))
    return sorted(stems)


def _find(dirs: List[Path], stem: str) -> Optional[Path]:
    for d in dirs:
        path = d / f"{stem}.json"
        if path.exists():
            return path
    return None


def list_personas() -> List[str]:
    return _list_stems(PERSONA_DIRS)


def list_events() -> List[str]:
    return _list_stems(EVENT_DIRS)


def load_persona(persona_id: str) -> Dict[str, Any]:
    """Load a persona by id (with or without .json)."""
    persona_id = persona_id.removesuffix(".json")
    path = _find(PERSONA_DIRS, persona_id)
    if path is None:
        raise FileNotFoundError(
            f"Persona '{persona_id}' not found. Available: {list_personas()}"
        )
    return _load_json(path)


def load_event(event_id: str) -> Dict[str, Any]:
    event_id = event_id.removesuffix(".json")
    path = _find(EVENT_DIRS, event_id)
    if path is None:
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
        # deepcopy so the engine's in-place mutations (task progress, NPC trust,
        # consumed queue items) never leak across episodes that share the same
        # loaded event object — fixes cross-episode state contamination + the
        # concurrency race where parallel episodes mutated the same dicts.
        "tasks": copy.deepcopy(init.get("tasks", [])),
        "relationships": copy.deepcopy(init.get("relationships", [])),
        "pending_events": copy.deepcopy(event.get("events_queue", [])),
        "agent": agent,
        "new_observations": copy.deepcopy(event.get("visible_information", [])),
        "turn": 0,
        "max_turns": max_turns,
        "trajectory": [],
        "finished": False,
    }
