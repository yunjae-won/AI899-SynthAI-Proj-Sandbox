"""The full "synthetic person" agent, built as a LangGraph ``StateGraph``.

Topology (linear per turn; the engine re-invokes the graph each turn):

    perceive --> reflect --> update_desires --> generate_goals --> plan_action

Each node reads/writes a small slice of the state. The engine layer
(``world.engine.run_simulation``) wraps this as a single "agent_step"
callable and handles world time, events, trajectory logging.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from ..config import LLMConfig, get_llm
from . import prompts
from ._llm_utils import call_llm, parse_json


# --------------------------------------------------------------------------- #
# Graph state (per-turn working buffer, NOT the world SimState)
# --------------------------------------------------------------------------- #


class TurnState(TypedDict, total=False):
    # Input from the engine
    persona: Dict[str, Any]
    event: Dict[str, Any]
    current_time: str
    location: str
    agent: Dict[str, Any]
    tasks: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    new_observations: List[Dict[str, Any]]

    # Node outputs
    reflection: str
    desires: List[str]
    goals: List[Dict[str, Any]]
    action: Dict[str, Any]


# --------------------------------------------------------------------------- #
# Node factories
# --------------------------------------------------------------------------- #


def _fmt(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _obs_lines(observations: List[Dict[str, Any]]) -> str:
    if not observations:
        return "(no new observations)"
    return "\n".join(
        f"- [{o.get('timestamp', '?')}] ({o.get('channel', '?')}) {o.get('content', '')}"
        for o in observations
    )


def _make_reflect(llm) -> Callable[[TurnState], Dict[str, Any]]:
    def reflect(state: TurnState) -> Dict[str, Any]:
        agent = state.get("agent", {})
        memory = agent.get("memory", []) or []
        recent_mem = memory[-3:]
        sys_p = prompts.persona_system(state["persona"])
        user_p = prompts.REFLECT_PROMPT.format(
            observations=_obs_lines(state.get("new_observations", [])),
            energy=agent.get("energy", "?"),
            sleep_debt=agent.get("sleep_debt_hours", "?"),
            stress=agent.get("stress", "?"),
            memory=_fmt(recent_mem),
        )
        raw = call_llm(llm, sys_p, user_p)
        out = parse_json(raw, fallback={"reflection": raw[:200]})
        reflection = out.get("reflection", "").strip()
        updated_memory = list(memory) + [{
            "turn_time": state.get("current_time"),
            "summary": reflection,
        }]
        # Keep memory bounded to avoid runaway context.
        updated_memory = updated_memory[-15:]
        new_agent = dict(agent)
        new_agent["memory"] = updated_memory
        return {"agent": new_agent, "reflection": reflection}

    return reflect


def _make_update_desires(llm) -> Callable[[TurnState], Dict[str, Any]]:
    def update_desires(state: TurnState) -> Dict[str, Any]:
        agent = state.get("agent", {})
        sys_p = prompts.persona_system(state["persona"])
        user_p = prompts.DESIRE_PROMPT.format(
            memory=_fmt((agent.get("memory") or [])[-5:]),
            agent_state=_fmt({
                "energy": agent.get("energy"),
                "sleep_debt_hours": agent.get("sleep_debt_hours"),
                "stress": agent.get("stress"),
            }),
            tasks=_fmt(state.get("tasks", [])),
            relationships=_fmt(state.get("relationships", [])),
        )
        raw = call_llm(llm, sys_p, user_p)
        out = parse_json(raw, fallback={"desires": []})
        desires = out.get("desires", [])
        if not isinstance(desires, list):
            desires = [str(desires)]
        new_agent = dict(agent)
        new_agent["desires"] = desires
        return {"agent": new_agent, "desires": desires}

    return update_desires


def _make_generate_goals(llm) -> Callable[[TurnState], Dict[str, Any]]:
    def generate_goals(state: TurnState) -> Dict[str, Any]:
        agent = state.get("agent", {})
        sys_p = prompts.persona_system(state["persona"])
        user_p = prompts.GOAL_PROMPT.format(
            prior_goals=_fmt(agent.get("goals", [])),
            desires=_fmt(agent.get("desires", [])),
            observations=_obs_lines(state.get("new_observations", [])),
            tasks=_fmt(state.get("tasks", [])),
        )
        raw = call_llm(llm, sys_p, user_p)
        out = parse_json(raw, fallback={"goals": agent.get("goals", [])})
        goals = out.get("goals", [])
        if not isinstance(goals, list):
            goals = []
        new_agent = dict(agent)
        new_agent["goals"] = goals
        return {"agent": new_agent, "goals": goals}

    return generate_goals


def _make_plan_action(llm) -> Callable[[TurnState], Dict[str, Any]]:
    def plan_action(state: TurnState) -> Dict[str, Any]:
        agent = state.get("agent", {})
        event = state.get("event", {})
        sys_p = prompts.persona_system(state["persona"])
        user_p = prompts.ACTION_PROMPT.format(
            goals=_fmt(agent.get("goals", [])),
            available_actions=_fmt(event.get("available_actions", [])),
            current_time=state.get("current_time", "?"),
            location=state.get("location", "?"),
            energy=agent.get("energy", "?"),
            sleep_debt=agent.get("sleep_debt_hours", "?"),
            stress=agent.get("stress", "?"),
            tasks=_fmt(state.get("tasks", [])),
            relationships=_fmt(state.get("relationships", [])),
            observations=_obs_lines(state.get("new_observations", [])),
        )
        raw = call_llm(llm, sys_p, user_p)
        out = parse_json(raw, fallback={
            "action": {"name": "wait", "args": {}, "reasoning": raw[:200]}
        })
        action = out.get("action", {})
        if not isinstance(action, dict):
            action = {"name": "wait", "args": {}, "reasoning": str(action)}
        action.setdefault("name", "wait")
        action.setdefault("args", {})
        action.setdefault("reasoning", "")
        new_agent = dict(agent)
        new_agent["last_reasoning"] = action.get("reasoning", "")
        return {"agent": new_agent, "action": action}

    return plan_action


# --------------------------------------------------------------------------- #
# Public builder
# --------------------------------------------------------------------------- #


def build_full_agent(llm_config: Optional[LLMConfig] = None) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Return an ``agent_step(sim_state) -> update_dict`` callable.

    Internally a compiled LangGraph graph; exposed as a plain callable so the
    engine doesn't need to know about LangGraph.
    """
    llm = get_llm(llm_config)

    g = StateGraph(TurnState)
    g.add_node("reflect", _make_reflect(llm))
    g.add_node("update_desires", _make_update_desires(llm))
    g.add_node("generate_goals", _make_generate_goals(llm))
    g.add_node("plan_action", _make_plan_action(llm))

    g.set_entry_point("reflect")
    g.add_edge("reflect", "update_desires")
    g.add_edge("update_desires", "generate_goals")
    g.add_edge("generate_goals", "plan_action")
    g.add_edge("plan_action", END)

    compiled = g.compile()

    def step(sim_state: Dict[str, Any]) -> Dict[str, Any]:
        # Pass the relevant slice into the graph.
        turn_in: TurnState = {
            "persona": sim_state["persona"],
            "event": sim_state["event"],
            "current_time": sim_state.get("current_time", ""),
            "location": sim_state.get("location", ""),
            "agent": dict(sim_state.get("agent", {})),
            "tasks": sim_state.get("tasks", []),
            "relationships": sim_state.get("relationships", []),
            "new_observations": sim_state.get("new_observations", []),
        }
        result = compiled.invoke(turn_in)
        agent = result.get("agent", {})
        return {
            "action": result.get("action", {"name": "wait", "args": {}, "reasoning": ""}),
            "memory": agent.get("memory", []),
            "desires": agent.get("desires", []),
            "goals": agent.get("goals", []),
            "last_reasoning": agent.get("last_reasoning", ""),
        }

    return step
