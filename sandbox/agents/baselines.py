"""Reactive baselines for comparison with the full synthetic-person agent.

All three share the same ``agent_step`` interface as ``build_full_agent`` so
the runner treats them interchangeably. They differ only in which cognitive
modules they include:

    prompt_only : observe -> act                    (no memory, no desire, no goals)
    memory_only : observe -> memory -> act          (no desires, no explicit goals)
    no_desire   : observe -> memory -> goals -> act (skips desire module;
                                                      goals come from tasks directly)
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from ..config import LLMConfig, get_llm
from . import prompts
from ._llm_utils import call_llm, parse_json


def _fmt(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _obs_lines(observations: List[Dict[str, Any]]) -> str:
    if not observations:
        return "(no new observations)"
    return "\n".join(
        f"- [{o.get('timestamp', '?')}] ({o.get('channel', '?')}) {o.get('content', '')}"
        for o in observations
    )


# --------------------------------------------------------------------------- #
# prompt_only — purely reactive
# --------------------------------------------------------------------------- #


def build_prompt_only_agent(llm_config: Optional[LLMConfig] = None):
    llm = get_llm(llm_config)

    def step(sim_state: Dict[str, Any]) -> Dict[str, Any]:
        agent = sim_state.get("agent", {})
        event = sim_state.get("event", {})
        sys_p = prompts.persona_system(sim_state["persona"])
        user_p = prompts.REACTIVE_ACTION_PROMPT.format(
            current_time=sim_state.get("current_time"),
            location=sim_state.get("location"),
            energy=agent.get("energy"),
            sleep_debt=agent.get("sleep_debt_hours"),
            stress=agent.get("stress"),
            tasks=_fmt(sim_state.get("tasks", [])),
            relationships=_fmt(sim_state.get("relationships", [])),
            observations=_obs_lines(sim_state.get("new_observations", [])),
            available_actions=_fmt(event.get("available_actions", [])),
        )
        raw = call_llm(llm, sys_p, user_p)
        out = parse_json(raw, fallback={"action": {"name": "wait", "args": {}, "reasoning": raw[:200]}})
        action = out.get("action", {"name": "wait", "args": {}, "reasoning": ""})
        return {"action": action, "last_reasoning": action.get("reasoning", "")}

    return step


# --------------------------------------------------------------------------- #
# memory_only — observe → reflect → act (no desires/goals)
# --------------------------------------------------------------------------- #


def build_memory_only_agent(llm_config: Optional[LLMConfig] = None):
    llm = get_llm(llm_config)

    def step(sim_state: Dict[str, Any]) -> Dict[str, Any]:
        agent = dict(sim_state.get("agent", {}))
        event = sim_state.get("event", {})
        sys_p = prompts.persona_system(sim_state["persona"])

        # Reflect.
        memory = agent.get("memory", []) or []
        reflect_prompt = prompts.REFLECT_PROMPT.format(
            observations=_obs_lines(sim_state.get("new_observations", [])),
            energy=agent.get("energy"),
            sleep_debt=agent.get("sleep_debt_hours"),
            stress=agent.get("stress"),
            memory=_fmt(memory[-3:]),
        )
        reflection = parse_json(call_llm(llm, sys_p, reflect_prompt),
                                fallback={"reflection": ""}).get("reflection", "")
        memory = (memory + [{"turn_time": sim_state.get("current_time"),
                             "summary": reflection}])[-15:]

        # Act (reactive, but with memory visible).
        user_p = prompts.REACTIVE_ACTION_PROMPT.format(
            current_time=sim_state.get("current_time"),
            location=sim_state.get("location"),
            energy=agent.get("energy"),
            sleep_debt=agent.get("sleep_debt_hours"),
            stress=agent.get("stress"),
            tasks=_fmt(sim_state.get("tasks", [])),
            relationships=_fmt(sim_state.get("relationships", [])),
            observations=(
                _obs_lines(sim_state.get("new_observations", []))
                + "\n\nYour recent reflections:\n"
                + _fmt(memory[-3:])
            ),
            available_actions=_fmt(event.get("available_actions", [])),
        )
        out = parse_json(call_llm(llm, sys_p, user_p),
                         fallback={"action": {"name": "wait", "args": {}, "reasoning": ""}})
        action = out.get("action", {"name": "wait", "args": {}, "reasoning": ""})
        return {"action": action, "memory": memory, "last_reasoning": action.get("reasoning", "")}

    return step


# --------------------------------------------------------------------------- #
# no_desire — memory + goals, but goals are derived from tasks, not desires
# --------------------------------------------------------------------------- #


NO_DESIRE_GOAL_PROMPT = """You need a SHORT ordered goal list for the next
few hours. Derive goals directly from the pending tasks and deadlines --
do NOT introspect your internal desires. Output JSON same shape as before.

TASKS: {tasks}
PRIOR GOALS: {prior_goals}

Return JSON: {{"goals": [{{"id":"g1","description":"...","priority":"high","status":"active","rationale":"..."}}]}}"""


def build_no_desire_agent(llm_config: Optional[LLMConfig] = None):
    llm = get_llm(llm_config)

    def step(sim_state: Dict[str, Any]) -> Dict[str, Any]:
        agent = dict(sim_state.get("agent", {}))
        event = sim_state.get("event", {})
        sys_p = prompts.persona_system(sim_state["persona"])

        # Reflect.
        memory = agent.get("memory", []) or []
        reflection = parse_json(
            call_llm(llm, sys_p, prompts.REFLECT_PROMPT.format(
                observations=_obs_lines(sim_state.get("new_observations", [])),
                energy=agent.get("energy"),
                sleep_debt=agent.get("sleep_debt_hours"),
                stress=agent.get("stress"),
                memory=_fmt(memory[-3:]),
            )),
            fallback={"reflection": ""},
        ).get("reflection", "")
        memory = (memory + [{"turn_time": sim_state.get("current_time"),
                             "summary": reflection}])[-15:]

        # Goals from tasks.
        goals = parse_json(
            call_llm(llm, sys_p, NO_DESIRE_GOAL_PROMPT.format(
                tasks=_fmt(sim_state.get("tasks", [])),
                prior_goals=_fmt(agent.get("goals", [])),
            )),
            fallback={"goals": agent.get("goals", [])},
        ).get("goals", [])

        # Act.
        user_p = prompts.ACTION_PROMPT.format(
            goals=_fmt(goals),
            available_actions=_fmt(event.get("available_actions", [])),
            current_time=sim_state.get("current_time"),
            location=sim_state.get("location"),
            energy=agent.get("energy"),
            sleep_debt=agent.get("sleep_debt_hours"),
            stress=agent.get("stress"),
            tasks=_fmt(sim_state.get("tasks", [])),
            relationships=_fmt(sim_state.get("relationships", [])),
            observations=_obs_lines(sim_state.get("new_observations", [])),
        )
        action = parse_json(call_llm(llm, sys_p, user_p),
                            fallback={"action": {"name": "wait", "args": {}, "reasoning": ""}}).get("action", {})
        return {
            "action": action,
            "memory": memory,
            "goals": goals,
            "desires": [],  # explicitly empty by design
            "last_reasoning": action.get("reasoning", ""),
        }

    return step
