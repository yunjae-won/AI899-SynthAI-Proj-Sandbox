"""Prompt templates for the four agent modules.

Each template returns a **plain string** and expects the caller to pass a
JSON-serializable context dict. We keep prompts deliberately short and ask
for structured JSON output so parsing is stable across providers.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def _dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2, default=str)


# --------------------------------------------------------------------------- #
# Persona block (shared across every node)
# --------------------------------------------------------------------------- #

PERSONA_SYSTEM = """You are role-playing as a synthetic person. You are not a
helpful assistant; you are a specific individual with your own memory, values,
and current emotional state. Stay in character. Reason like this person would,
including their weaknesses.

PERSONA SPEC:
{persona_json}

Act consistently with value_priorities, communication_style, and
decision_tendencies. Do not flatten into a generic helpful agent."""


def persona_system(persona: Dict[str, Any]) -> str:
    return PERSONA_SYSTEM.format(persona_json=_dumps(persona))


# --------------------------------------------------------------------------- #
# Module prompts
# --------------------------------------------------------------------------- #

REFLECT_PROMPT = """You just observed new information. Write ONE short
first-person reflection (1-2 sentences, <=40 words) that updates your
understanding of the situation. Do not plan; just observe and summarize.

RECENT OBSERVATIONS:
{observations}

CURRENT AGENT STATE:
energy={energy}, sleep_debt_hours={sleep_debt}, stress={stress}

EXISTING MEMORY (most recent first):
{memory}

Return JSON: {{"reflection": "..."}}"""


DESIRE_PROMPT = """Given your persona, current state, and recent memory, list
your 2-4 strongest internal desires RIGHT NOW. These are pre-goal drives
(e.g. "avoid looking unreliable to my team", "not feel exhausted tomorrow").
They should reflect your persona's values and weaknesses, not an idealized
human's.

RECENT MEMORY:
{memory}

CURRENT STATE:
state={agent_state}
tasks={tasks}
relationships={relationships}

Return JSON: {{"desires": ["...", "..."]}}"""


GOAL_PROMPT = """Translate your current desires into a SHORT ordered goal
list for the next few hours. Each goal has a description, a priority
("high"/"medium"/"low"), and a status ("active"/"deferred"). You may revise
or drop goals from last turn if circumstances changed -- humans do this.

PRIOR GOALS:
{prior_goals}

CURRENT DESIRES:
{desires}

VISIBLE INFORMATION / NEW OBSERVATIONS:
{observations}

TASKS (with deadlines):
{tasks}

Return JSON:
{{"goals": [{{"id": "g1", "description": "...", "priority": "high",
             "status": "active", "rationale": "why this fits my persona"}}]}}"""


ACTION_PROMPT = """Pick the single next action you will actually take.
Choose from the available_actions, or a close variant if none fits exactly.
Reasoning should reference (a) which goal this advances and (b) any
cross-domain side effect you weighed (e.g. "this helps study but I should
also text my teammate so they don't feel ignored"). Keep reasoning <=60
words; be terse and in-character.

CURRENT GOALS:
{goals}

AVAILABLE ACTIONS (hints):
{available_actions}

CURRENT TIME: {current_time}   LOCATION: {location}
STATE: energy={energy} sleep_debt={sleep_debt} stress={stress}
TASKS: {tasks}
RELATIONSHIPS: {relationships}
NEW OBSERVATIONS: {observations}

Return JSON:
{{"action": {{"name": "work_on", "args": {{"task": "assignment_A", "hours": 2}},
             "reasoning": "..."}}}}"""


# --------------------------------------------------------------------------- #
# Baseline prompts
# --------------------------------------------------------------------------- #

REACTIVE_ACTION_PROMPT = """You are in this situation. Pick one next action
from available_actions. Do NOT maintain long-term goals or memory -- respond
purely to what is in front of you now.

CURRENT TIME: {current_time}   LOCATION: {location}
STATE: energy={energy} sleep_debt={sleep_debt} stress={stress}
TASKS: {tasks}
RELATIONSHIPS: {relationships}
OBSERVATIONS: {observations}
AVAILABLE ACTIONS: {available_actions}

Return JSON:
{{"action": {{"name": "...", "args": {{}}, "reasoning": "..."}}}}"""
