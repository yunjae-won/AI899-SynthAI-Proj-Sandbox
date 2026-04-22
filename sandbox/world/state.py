"""State schemas shared across the sandbox.

We use ``TypedDict`` (not Pydantic) because LangGraph's ``StateGraph`` merges
dict updates natively and we want contributors to be able to ``state["..."]``
without ceremony. The runtime guarantees are loose; the field names and types
below are the contract.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class Task(TypedDict, total=False):
    id: str
    label: str
    deadline: str
    progress: float
    hardness: str  # "hard" | "soft"


class Relationship(TypedDict, total=False):
    role: str
    name: str
    trust: float
    closeness: float


class Observation(TypedDict, total=False):
    timestamp: str
    channel: str
    content: str


class AgentState(TypedDict, total=False):
    """Dynamic per-turn agent internals."""

    # Physical / psychological state
    energy: float
    sleep_debt_hours: float
    stress: float

    # Cognitive layers. These are what the four modules read/write.
    memory: List[Dict[str, Any]]      # reflections: [{turn, summary}]
    desires: List[str]                # short strings e.g. "finish assignment A"
    goals: List[Dict[str, Any]]       # [{id, description, priority, status}]
    last_reasoning: str               # free-form trace of last decision


class TrajectoryStep(TypedDict, total=False):
    """One (timestamp, state_snapshot, action, observation) tuple."""

    turn: int
    timestamp: str
    state_snapshot: Dict[str, Any]
    action: Dict[str, Any]      # {name, args, reasoning}
    observation: str            # post-action world feedback


class SimState(TypedDict, total=False):
    """The full graph state passed between LangGraph nodes.

    Keep this flat — LangGraph merges top-level keys, not nested dicts.
    """

    # Static (doesn't change during run)
    persona: Dict[str, Any]
    event: Dict[str, Any]

    # World
    current_time: str                 # "Wed 14:00"-style label
    location: str
    tasks: List[Task]
    relationships: List[Relationship]
    pending_events: List[Dict[str, Any]]   # remaining events_queue

    # Agent
    agent: AgentState
    new_observations: List[Observation]    # unread since last perceive

    # Bookkeeping
    turn: int
    max_turns: int
    trajectory: List[TrajectoryStep]
    finished: bool
