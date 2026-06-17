"""Turn-based simulation driver.

The engine calls an ``agent_step`` callable that takes a ``SimState`` and returns
an action dict. Between turns the engine advances time, fires scheduled events,
applies rule-based effects for each action, and — crucially — feeds the
*consequences* of the agent's action back as observations on the **following**
turn, so the perception -> action loop is actually closed and the world is
responsive instead of mute.

Mechanics stay light-weight and deterministic (stdlib only), but they are now
*consequential* rather than decorative:

  * **Hidden state is enacted.** When the agent takes an action listed in an
    event's ``hidden_state[*].discoverable_by``, the engine reveals the insight
    as an observation and applies its mechanical effect (a numeric early-
    investment bonus on the affected task, or an interpersonal renegotiability
    that the conflict actions can then cash in).
  * **Interpersonal actions matter.** ``message`` is content-aware (assertive vs
    accommodating vs neutral, EN + KO) and ``defer`` / ``propose_boundary``
    genuinely renegotiate an *imposed* task or resolve a roommate conflict —
    reassigning/ bounding the work and moving NPC trust accordingly, instead of
    just printing a line.
  * **NPCs react to the agent.** Queued NPC messages are suppressed or turned
    into acknowledgements once the agent has addressed that thread, and internal
    "time is slipping" / "bedtime" events carry their declared consequence
    (rising crunch, lost sleep) instead of firing blind.
  * **Deadlines bite.** A hard deadline that passes with the task unfinished
    produces a real consequence (stress, a missed-submission flag); finishing
    the one dominant task resolves the episode.

The point is still to *observe* the agent's persona-driven choices — not to
simulate an undergrad in full fidelity — but the environment now pushes back
enough that multi-turn behaviour reflects disposition rather than decaying into
repetition. The measurement layer reads the agent's *chosen* action, so none of
this changes how fidelity/divergence are scored; it only makes the trajectory
the agent produces a faithful response to a world that answers back.
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
_MIN_PER_DAY = 24 * 60


def _parse_time(label: str) -> Optional[int]:
    """Parse 'Wed 14:00' -> absolute minute count across the week.

    Returns None if the label doesn't match (e.g. "ongoing"). Used to order
    events and progress time; nothing depends on precise wall-clock semantics.
    """
    if not isinstance(label, str):
        return None
    m = re.match(r"(\w{3})\s+(\d{1,2}):(\d{2})", label)
    if not m:
        return None
    day, hh, mm = m.groups()
    if day not in _DAYS:
        return None
    return (_DAYS.index(day) * _MIN_PER_DAY) + int(hh) * 60 + int(mm)


def _format_time(minutes: int) -> str:
    day_idx = (minutes // _MIN_PER_DAY) % 7
    rem = minutes % _MIN_PER_DAY
    return f"{_DAYS[day_idx]} {rem // 60:02d}:{rem % 60:02d}"


def _advance_time(label: str, hours: float) -> str:
    base = _parse_time(label)
    if base is None:
        return label
    return _format_time(base + int(round(hours * 60)))


def _day_of(label: str) -> Optional[int]:
    base = _parse_time(label)
    return None if base is None else base // _MIN_PER_DAY


# --------------------------------------------------------------------------- #
# Lookup helpers
# --------------------------------------------------------------------------- #


def _find_task(tasks: List[Dict[str, Any]], task_id: str) -> Optional[Dict[str, Any]]:
    if not task_id:
        return None
    ref = task_id.lower()
    for t in tasks:
        if t.get("id", "").lower() == ref:
            return t
    for t in tasks:
        label = t.get("label", "").lower()
        if label.startswith(ref) or ref in label or t.get("id", "").lower() in ref:
            return t
    return None


def _find_relationship(
    relationships: List[Dict[str, Any]], role_or_name: str
) -> Optional[Dict[str, Any]]:
    if not role_or_name:
        return relationships[0] if len(relationships) == 1 else None
    key = role_or_name.lower()
    for r in relationships:
        if r.get("role", "").lower() == key or r.get("name", "").lower() == key:
            return r
    for r in relationships:
        if key in r.get("role", "").lower() or key in r.get("name", "").lower():
            return r
    # A single-NPC scene: an unmatched target almost always means "that person".
    return relationships[0] if len(relationships) == 1 else None


def _obs(state: Dict[str, Any], channel: str, content: str) -> Dict[str, Any]:
    return {"timestamp": state["current_time"], "channel": channel, "content": content}


# --------------------------------------------------------------------------- #
# Message tone (EN + KO keyword heuristic)
# --------------------------------------------------------------------------- #

_ASSERTIVE_CUES = (
    "boundary", "can't take", "cannot take", "won't be able", "can not",
    "your part", "your share", "your half", "split", "share it", "renegotiate",
    "not fair", "unfair", "fairly", "i'll do my", "only do", "by yourself",
    "do your own", "push back", "decline", "i can't", "respectfully", "limit",
    "please clean", "please wash", "laundry", "shower", "tidy", "the smell",
    "the odor", "거절", "안 돼", "안돼", "못 해", "못해", "네 파트", "네 몫",
    "각자", "나눠", "나눠서", "공평", "치워", "빨래", "씻", "정리", "지켜",
    "곤란", "어려울", "선은",
)
_ACCOMMODATE_CUES = (
    "sure", "no problem", "i'll take", "i can do", "i got it", "of course",
    "happy to", "don't worry", "leave it to me", "i'll handle", "i'll cover",
    "no worries", "ok i", "okay i", "fine i'll", "알겠", "내가 할게",
    "해줄게", "괜찮아", "물론", "걱정 마", "걱정마", "내가 다", "맡겨",
)


def _classify_tone(text: str) -> str:
    t = (text or "").lower()
    a = sum(cue in t for cue in _ASSERTIVE_CUES)
    b = sum(cue in t for cue in _ACCOMMODATE_CUES)
    if a > b:
        return "assertive"
    if b > a:
        return "accommodate"
    return "neutral"


# --------------------------------------------------------------------------- #
# Hidden-state discovery + enactment
# --------------------------------------------------------------------------- #
#
# Hidden state is declared in the event JSON as a list of
#   {key, value, description, affects, discoverable_by:[tokens]}
# entries. The engine reveals an entry (once) when the agent takes an action
# whose discovery tokens intersect ``discoverable_by``, then applies a generic
# mechanical effect inferred from ``value``/``affects``. Nothing here is
# event-specific: numeric entries become a per-hour task bonus; boolean-true
# entries that name a relationship/trust become "renegotiable", which the
# interpersonal action handlers cash in.


# engine-action -> the discovery tokens it counts as.
_DISCOVERY_TOKENS = {
    "work_on": {"work_on", "reflect", "self_state_check"},
    "sleep": {"reflect", "self_state_check"},
    "wait": {"reflect", "self_state_check", "reread_portal"},
    "message": {"message"},
    "propose_boundary": {"propose_boundary"},
    "defer": {"defer"},
    "move_to": {"move_to", "reflect"},
    "attend": {"attend"},
}


def _is_numeric(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _discover(
    state: Dict[str, Any],
    engine_action: str,
    *,
    task: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Reveal (once) hidden_state entries discoverable by ``engine_action``.

    Applies the generic mechanical effect of each newly-revealed entry and
    returns observation dicts narrating the realisation. ``task`` scopes numeric
    "investment" insights so they only surface while engaging the task they act
    on (you learn early-start pays off by actually starting).
    """
    flags = state["flags"]
    discovered: set = flags.setdefault("_discovered", set())
    tokens = _DISCOVERY_TOKENS.get(engine_action, set())
    out: List[Dict[str, Any]] = []
    for entry in state.get("event", {}).get("hidden_state", []) or []:
        key = entry.get("key")
        if not key or key in discovered:
            continue
        by = {str(d).lower() for d in entry.get("discoverable_by", []) or []}
        if not (by & tokens):
            continue
        affects = (entry.get("affects") or "").lower()
        value = entry.get("value")

        if _is_numeric(value):
            # Numeric insight: only surfaces while working the affected task.
            if task is None or task.get("id", "").lower() not in affects:
                continue
            discovered.add(key)
            flags[f"bonus:{task['id']}"] = float(value)
            out.append(_obs(state, "insight", f"(you realize) {entry.get('description', '')}"))
        else:
            discovered.add(key)
            # Boolean-true entries that concern a relationship/trust become a
            # renegotiability signal the conflict actions can act on.
            if value is True and ("trust" in affects or _affects_relationship(state, affects)):
                flags["renegotiable"] = True
            out.append(_obs(state, "insight", f"(you realize) {entry.get('description', '')}"))
    return out


def _affects_relationship(state: Dict[str, Any], affects: str) -> bool:
    for r in state.get("relationships", []):
        if r.get("name", "").lower() in affects or r.get("role", "").lower() in affects:
            return True
    return False


def _renegotiation_allowed(state: Dict[str, Any]) -> bool:
    """True if a hidden_state entry sanctions a graceful boundary/renegotiation.

    Default-True when the event declares no relevant hidden state, so a sensible
    boundary is never silently punished; an event can suppress this by declaring
    a falsy interpersonal entry.
    """
    if state["flags"].get("renegotiable"):
        return True
    entries = state.get("event", {}).get("hidden_state", []) or []
    interpersonal = [
        e for e in entries
        if "trust" in (e.get("affects") or "").lower()
        or _affects_relationship(state, (e.get("affects") or "").lower())
    ]
    if not interpersonal:
        return True
    return any(e.get("value") is True for e in interpersonal)


# --------------------------------------------------------------------------- #
# Task helpers
# --------------------------------------------------------------------------- #


def _is_imposed(task: Dict[str, Any], state: Dict[str, Any]) -> bool:
    """A task the agent did not originate — dumped by an NPC or an external ask."""
    if task.get("imposed") or task.get("owner"):
        return True
    tid = task.get("id", "").lower()
    for e in state.get("event", {}).get("hidden_state", []) or []:
        aff = (e.get("affects") or "").lower()
        if tid and tid in aff and ("trust" in aff or _affects_relationship(state, aff)):
            return True
    return False


def _due_today(state: Dict[str, Any], task: Dict[str, Any]) -> bool:
    d = _day_of(task.get("deadline", ""))
    return d is not None and d == state["flags"].get("start_day")


def _optional_task_ids(state: Dict[str, Any]) -> set:
    """Task ids that are merely *opportunities* (an ``attend:<id>`` diagnostic),
    not obligations — declining them is a legitimate terminal choice, so they
    must not keep the episode alive as 'pressing unfinished work'."""
    out: set = set()
    for acts in (state.get("event", {}).get("diagnostic_actions") or {}).values():
        for a in acts or []:
            if isinstance(a, str) and ":" in a:
                nm, arg = a.split(":", 1)
                if nm.strip() == "attend":
                    out.add(arg.strip())
    return out


# --------------------------------------------------------------------------- #
# Action effects
# --------------------------------------------------------------------------- #


def _apply_action(state: Dict[str, Any], action: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Mutate ``state`` in-place for ``action``; return observation dicts.

    The returned observations are what the agent will perceive on the *next*
    turn, closing the loop.
    """
    name = (action.get("name") or "").lower()
    args = action.get("args", {}) or {}
    if isinstance(args, str):
        args = {"raw": args}
    agent = state["agent"]
    flags = state["flags"]
    obs: List[Dict[str, Any]] = []

    # ---- work_on ---------------------------------------------------------- #
    if "work_on" in name or name == "work":
        task_ref = args.get("task") or args.get("task_id") or args.get("target") or ""
        hours = _as_hours(args.get("hours", 1))
        task = _find_task(state["tasks"], task_ref)
        if task is None:
            obs.append(_obs(state, "result", f"No matching task '{task_ref}' to work on."))
        else:
            focus = max(0.2, min(1.0, agent.get("energy", 5) / 10.0))
            bonus = flags.get(f"bonus:{task.get('id')}", 0.0)
            delta = 0.15 * hours * focus * (1.0 + bonus)
            before = task.get("progress", 0.0)
            task["progress"] = min(1.0, before + delta)
            agent["energy"] = max(0.0, agent.get("energy", 5) - 0.6 * hours)
            agent["stress"] = max(0.0, agent.get("stress", 3) - 0.1 - (0.2 if bonus else 0))
            state["current_time"] = _advance_time(state["current_time"], hours)
            if _is_imposed(task, state):
                # Silently absorbing dumped work is the avoidant outcome.
                flags["absorbed_imposed"] = True
                obs.append(_obs(
                    state, "result",
                    f"You took on {task['label']} yourself ({before:.2f} -> "
                    f"{task['progress']:.2f}); your own load is now heavier."))
            else:
                tail = " It's essentially done." if task["progress"] >= 0.99 else ""
                obs.append(_obs(
                    state, "result",
                    f"Worked {hours:g}h on {task['label']}: progress "
                    f"{before:.2f} -> {task['progress']:.2f}.{tail}"))
            obs += _discover(state, "work_on", task=task)
            obs += _deadline_pressure(state, task)

    # ---- sleep ------------------------------------------------------------ #
    elif name.startswith("sleep") or name == "rest" or name == "nap":
        hours = _as_hours(args.get("hours", 6))
        room_penalty = flags.get("room_unresolved_sleep_penalty") and not flags.get("interpersonal_resolved")
        recover = hours * (0.4 if room_penalty else 1.0)
        agent["energy"] = min(10.0, agent.get("energy", 5) + 1.2 * (recover / max(hours, 1)) * hours)
        agent["sleep_debt_hours"] = max(0.0, agent.get("sleep_debt_hours", 0) - recover)
        agent["stress"] = max(0.0, agent.get("stress", 3) - 0.4)
        state["current_time"] = _advance_time(state["current_time"], hours)
        if room_penalty:
            obs.append(_obs(state, "result",
                            f"Slept {hours:g}h but the room still smells — restless, "
                            f"only partial recovery (sleep_debt={agent['sleep_debt_hours']:.1f})."))
        else:
            obs.append(_obs(state, "result",
                            f"Slept {hours:g}h; energy={agent['energy']:.1f}, "
                            f"sleep_debt={agent['sleep_debt_hours']:.1f}."))
        obs += _discover(state, "sleep")

    # ---- message ---------------------------------------------------------- #
    elif name.startswith("message") or name == "contact" or name == "reply":
        target = args.get("role") or args.get("target") or args.get("to") or ""
        content = str(args.get("content", "") or args.get("raw", ""))
        rel = _find_relationship(state["relationships"], target)
        state["current_time"] = _advance_time(state["current_time"], 0.25)
        if rel is None:
            obs.append(_obs(state, "result", f"No one named '{target}' to message."))
        else:
            tone = _classify_tone(content)
            obs += _discover(state, "message")
            obs += _handle_interpersonal(state, rel, tone, via="message", content=content)

    # ---- defer / reschedule ---------------------------------------------- #
    elif name.startswith("defer") or name == "reschedule":
        task_ref = args.get("task") or args.get("task_id") or ""
        task = _find_task(state["tasks"], task_ref)
        new_deadline = args.get("new_deadline", "")
        state["current_time"] = _advance_time(state["current_time"], 0.25)
        if task is None:
            obs.append(_obs(state, "result", "Nothing specific to defer."))
        elif _is_imposed(task, state):
            rel = _relationship_for_task(state, task)
            obs += _discover(state, "defer")
            obs += _handle_interpersonal(state, rel, "assertive", via="defer", task=task)
        else:
            task["deadline"] = new_deadline or task.get("deadline")
            task["hardness"] = "soft"
            flags["deferred_own_work"] = True
            obs.append(_obs(state, "result",
                            f"Pushed {task['label']} back to {task.get('deadline')}; "
                            f"you'll have to make the time up later."))
            obs += _discover(state, "defer")

    # ---- attend / join --------------------------------------------------- #
    elif name.startswith("attend") or name == "join" or name == "go_out":
        what = str(args.get("event") or args.get("target") or args.get("raw") or "it")
        state["current_time"] = _advance_time(state["current_time"], 1.5)
        agent["energy"] = max(0.0, agent.get("energy", 5) - 0.3)
        low = what.lower()
        if any(k in low for k in ("leisure", "fun", "relax", "game", "stream", "hobby", "nap")):
            # Immediate gratification; the deferred task quietly slips.
            agent["stress"] = max(0.0, agent.get("stress", 3) - 0.6)
            flags["took_immediate_payoff"] = True
            obs.append(_obs(state, "result",
                            f"Took the immediate option ({what}). Felt good; "
                            f"the afternoon block is gone, though."))
            obs += _slipping_deadline_note(state)
        else:
            # Social attendance nurtures the friend group.
            mark = False
            for r in state["relationships"]:
                role = r.get("role", "").lower()
                if "friend" in role or "group" in role or "cohort" in role or "club" in role:
                    r["closeness"] = min(10.0, r.get("closeness", 5) + 0.6)
                    r["trust"] = min(10.0, r.get("trust", 5) + 0.2)
                    mark = True
            sc = _find_task(state["tasks"], what) or _find_task(state["tasks"], "social")
            if sc is not None:
                sc["progress"] = 1.0
                sc["resolved"] = True
            flags["addressed:friend"] = True
            obs.append(_obs(state, "result",
                            f"Joined {what}." + (" The group's glad you came." if mark else "")))
        obs += _discover(state, "attend")

    # ---- move_to ---------------------------------------------------------- #
    elif name.startswith("move_to") or name == "move" or name == "relocate":
        loc = str(args.get("location", "") or args.get("target", "") or args.get("raw", ""))
        state["location"] = loc or state["location"]
        state["current_time"] = _advance_time(state["current_time"], 0.25)
        quiet = any(k in loc.lower() for k in ("library", "quiet", "study", "cafe", "reading"))
        if quiet:
            flags["focus_restored"] = True
        # Relocating routes around an interpersonal problem without resolving it.
        if _open_interpersonal(state):
            flags["room_unresolved_sleep_penalty"] = True
        obs.append(_obs(state, "result",
                        f"Moved to {state['location']}." +
                        (" Quiet here — focus restored for the work session." if quiet else "")))
        obs += _discover(state, "move_to")

    # ---- propose_boundary ------------------------------------------------- #
    elif name.startswith("propose_boundary") or name == "boundary" or name == "set_boundary":
        duration = args.get("duration_minutes", 30)
        state["current_time"] = _advance_time(state["current_time"], 0.25)
        rel = _relationship_for_task(state, _imposed_or_interpersonal_task(state))
        obs += _discover(state, "propose_boundary")
        obs += _handle_interpersonal(
            state, rel, "assertive", via="propose_boundary",
            task=_imposed_or_interpersonal_task(state), duration=duration)

    # ---- wait / noop ------------------------------------------------------ #
    elif name in ("wait", "noop", "pass", "idle", "do_nothing"):
        state["current_time"] = _advance_time(state["current_time"], 0.5)
        flags["idled"] = True
        obs.append(_obs(state, "result", "You let the moment pass; nothing changes."))
        obs += _slipping_deadline_note(state)
        obs += _discover(state, "wait")

    # ---- free-form -------------------------------------------------------- #
    else:
        raw = args.get("raw") or action.get("description") or name
        state["current_time"] = _advance_time(state["current_time"], 0.5)
        obs.append(_obs(state, "result", f"You did: {raw}."))

    if not obs:
        obs.append(_obs(state, "result", "(no observable change)"))
    return obs


def _as_hours(v: Any) -> float:
    try:
        h = float(v)
    except (TypeError, ValueError):
        return 1.0
    return max(0.25, min(8.0, h))


# --------------------------------------------------------------------------- #
# Interpersonal resolution
# --------------------------------------------------------------------------- #


def _relationship_for_task(
    state: Dict[str, Any], task: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    rels = state.get("relationships", [])
    if task is not None:
        owner = (task.get("owner") or "").lower()
        if owner:
            r = _find_relationship(rels, owner)
            if r:
                return r
        tid = task.get("id", "").lower()
        for e in state.get("event", {}).get("hidden_state", []) or []:
            aff = (e.get("affects") or "").lower()
            if tid and tid in aff:
                for r in rels:
                    if r.get("name", "").lower() in aff or r.get("role", "").lower() in aff:
                        return r
    return rels[0] if len(rels) == 1 else (rels[0] if rels else None)


def _imposed_or_interpersonal_task(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for t in state.get("tasks", []):
        if not t.get("resolved") and _is_imposed(t, state):
            return t
    return None


def _open_interpersonal(state: Dict[str, Any]) -> bool:
    if state["flags"].get("interpersonal_resolved"):
        return False
    return _imposed_or_interpersonal_task(state) is not None


def _npc(rel: Optional[Dict[str, Any]]) -> str:
    if not rel:
        return "they"
    return rel.get("name") or rel.get("role") or "they"


def _handle_interpersonal(
    state: Dict[str, Any],
    rel: Optional[Dict[str, Any]],
    tone: str,
    *,
    via: str,
    task: Optional[Dict[str, Any]] = None,
    content: str = "",
    duration: Any = None,
) -> List[Dict[str, Any]]:
    """Resolve (or fail to resolve) an interpersonal thread and move NPC trust.

    Generic across the conflict events: an assertive boundary on an imposed /
    interpersonal task renegotiates it (and is accepted when the hidden state
    allows), an accommodating reply absorbs it, a neutral note nudges trust.
    """
    flags = state["flags"]
    obs: List[Dict[str, Any]] = []
    if rel is None:
        return [_obs(state, "result", "Message sent, but no one in particular is involved.")]

    name = _npc(rel)
    target_task = task if task is not None else _imposed_or_interpersonal_task(state)
    if via in ("message", "defer", "propose_boundary"):
        flags[f"addressed:{rel.get('role', '').lower()}"] = True
        flags["addressed_npc"] = True

    # Assertive boundary / renegotiation on a real interpersonal thread.
    if tone == "assertive" and target_task is not None:
        if _renegotiation_allowed(state):
            rel["trust"] = min(10.0, rel.get("trust", 5) + 0.3)
            target_task["resolved"] = True
            flags["interpersonal_resolved"] = True
            flags["room_unresolved_sleep_penalty"] = False
            if via == "propose_boundary" and target_task.get("imposed"):
                target_task["status"] = f"bounded help agreed (~{duration} min)"
                msg = f"{name}: 그 정도면 충분해, 고마워! 나머지는 내가 할게."
            elif target_task.get("imposed") or target_task.get("owner"):
                target_task["status"] = f"returned to {name} (agreed)"
                msg = f"{name}: 알겠어, 내가 어떻게든 해볼게. 미안, 부탁해서."
            else:
                target_task["status"] = "agreement reached"
                target_task["progress"] = max(target_task.get("progress", 0.0), 0.6)
                msg = f"{name}: 아 그랬구나, 몰랐어. 미안 — 바로 신경쓸게. 말해줘서 고마워."
            obs.append(_obs(state, "result",
                            f"You raised it directly with {name}; it lands well, no hard feelings."))
            obs.append(_obs(state, _channel_for(rel), msg))
        else:
            rel["trust"] = max(0.0, rel.get("trust", 5) - 0.4)
            obs.append(_obs(state, _channel_for(rel),
                            f"{name}: ... 알겠어. (살짝 서운한 기색)"))
        return obs

    # Accommodating reply: agent absorbs the ask.
    if tone == "accommodate":
        rel["trust"] = min(10.0, rel.get("trust", 5) + 0.3)
        rel["closeness"] = min(10.0, rel.get("closeness", 5) + 0.2)
        if target_task is not None and (target_task.get("imposed") or target_task.get("owner")):
            flags["absorbed_imposed"] = True
            obs.append(_obs(state, _channel_for(rel),
                            f"{name}: 진짜 고마워!! ㅠㅠ 나중에 갚을게."))
            obs.append(_obs(state, "result",
                            "You agreed to take it on — your own deadline just got tighter."))
        else:
            flags["addressed:friend"] = True
            obs.append(_obs(state, _channel_for(rel), f"{name}: 좋아, 고마워!"))
        return obs

    # Neutral note.
    rel["trust"] = min(10.0, rel.get("trust", 5) + 0.1)
    snippet = (content[:50] + "…") if len(content) > 50 else content
    obs.append(_obs(state, "result",
                    f"Messaged {name}" + (f": '{snippet}'." if snippet else ".")))
    if target_task is not None:
        obs.append(_obs(state, _channel_for(rel), f"{name}: ...그래서 어떻게 할까?"))
    return obs


def _channel_for(rel: Dict[str, Any]) -> str:
    role = (rel.get("role") or "").lower()
    if "teammate" in role or "classmate" in role:
        return "team_chat"
    if "roommate" in role:
        return "roommate_chat"
    if "friend" in role or "group" in role or "cohort" in role:
        return "friend_chat"
    if "prof" in role:
        return "email"
    return "chat"


# --------------------------------------------------------------------------- #
# Deadlines & time pressure
# --------------------------------------------------------------------------- #


def _deadline_pressure(state: Dict[str, Any], task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A one-time nudge when a task is close to its deadline but unfinished."""
    dl = _parse_time(task.get("deadline", ""))
    now = _parse_time(state["current_time"])
    if dl is None or now is None:
        return []
    remaining = dl - now
    key = f"warned_dl:{task.get('id')}"
    if 0 < remaining <= 60 and task.get("progress", 0) < 0.99 and not state["flags"].get(key):
        state["flags"][key] = True
        return [_obs(state, "clock",
                     f"{task['label']} is due in under an hour and not finished yet.")]
    return []


def _slipping_deadline_note(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """When the agent idles/leisures, surface any near-term task slipping away."""
    now = _parse_time(state["current_time"])
    if now is None:
        return []
    for t in state["tasks"]:
        if t.get("resolved") or t.get("progress", 0) >= 0.99:
            continue
        dl = _parse_time(t.get("deadline", ""))
        if dl is None:
            continue
        if 0 < dl - now <= 180:
            key = f"slip:{t.get('id')}"
            if not state["flags"].get(key):
                state["flags"][key] = True
                return [_obs(state, "clock",
                             f"Time is passing — {t['label']} still needs work before "
                             f"{t.get('deadline')}.")]
    return []


def _enforce_deadlines(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Apply consequences for deadlines that have now passed."""
    now = _parse_time(state["current_time"])
    if now is None:
        return []
    optional = _optional_task_ids(state)
    obs: List[Dict[str, Any]] = []
    for t in state["tasks"]:
        dl = _parse_time(t.get("deadline", ""))
        if dl is None or now < dl:
            continue
        tid = t.get("id")
        seen = f"dl_passed:{tid}"
        if state["flags"].get(seen):
            continue
        state["flags"][seen] = True
        if t.get("resolved"):
            continue
        if t.get("progress", 0) >= 0.99:
            if t.get("hardness") == "hard":
                obs.append(_obs(state, "result", f"Submitted: {t['label']}. Done in time."))
        elif tid in optional:
            # An opportunity the agent chose not to take — its window simply
            # lapses; declining is a valid choice, not a missed obligation.
            obs.append(_obs(state, "result",
                            f"The window for {t['label']} has passed; you let it go by."))
        else:
            penalty = 1.5 if t.get("hardness") == "hard" else 0.6
            state["agent"]["stress"] = min(10.0, state["agent"].get("stress", 3) + penalty)
            state["flags"][f"missed:{tid}"] = True
            obs.append(_obs(state, "result",
                            f"The deadline for {t['label']} passed at {t.get('deadline')} "
                            f"with it unfinished (progress {t.get('progress', 0):.2f})."))
    return obs


# --------------------------------------------------------------------------- #
# Event queue firing (behaviour-conditioned)
# --------------------------------------------------------------------------- #

_NAG_MARKERS = ("답 줘", "부탁", "last call", "you in", "answer", "?", "줘~", "올래")


def _fire_due_events(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fire scheduled events whose timestamp <= now, conditioned on behaviour.

    NPC nags are dropped/acknowledged once the agent has addressed that thread;
    internal "time slipping"/"bedtime" beats carry their declared consequence.
    """
    now = _parse_time(state["current_time"])
    if now is None:
        return []
    flags = state["flags"]
    fired: List[Dict[str, Any]] = []
    remaining: List[Dict[str, Any]] = []
    for ev in state.get("pending_events", []):
        ts = _parse_time(ev.get("timestamp", ""))
        if ts is None or ts > now:
            remaining.append(ev)
            continue
        etype = (ev.get("type") or "scheduled_event").lower()
        content = ev.get("content", "")

        if etype == "internal":
            low = content.lower()
            # A looming-deadline beat: only bites if the task is still neglected.
            neglected = any(
                (not t.get("resolved")) and t.get("progress", 0) < 0.3
                and _parse_time(t.get("deadline", "")) is not None
                for t in state["tasks"]
            )
            if "bed" in low or "sleep" in low:
                if _open_interpersonal(state) and not flags.get("interpersonal_resolved"):
                    state["agent"]["sleep_debt_hours"] = state["agent"].get("sleep_debt_hours", 0) + 2
                    state["agent"]["stress"] = min(10.0, state["agent"].get("stress", 3) + 0.5)
                    fired.append(_obs(state, "internal",
                                      content + " (The room was never sorted — poor sleep again.)"))
                else:
                    fired.append(_obs(state, "internal", "Wound down for the night; rested well."))
            elif neglected:
                state["agent"]["stress"] = min(10.0, state["agent"].get("stress", 3) + 0.4)
                fired.append(_obs(state, "internal", content))
            # if not neglected, the beat simply doesn't fire (agent is on top of it).
            continue

        # NPC chat / reminder events.
        is_nag = any(m in content for m in _NAG_MARKERS)
        if is_nag and flags.get("addressed_npc"):
            continue  # already replied — the NPC isn't still pestering.
        fired.append(_obs(state, etype, content))
    state["pending_events"] = remaining
    return fired


# --------------------------------------------------------------------------- #
# Termination
# --------------------------------------------------------------------------- #


def _episode_resolved(state: Dict[str, Any], last_action: str) -> bool:
    tasks = state["tasks"]
    pending = state.get("pending_events") or []
    # The one dominant hard deadline is handled -> the scene is over.
    hard = [t for t in tasks if t.get("hardness") == "hard"]
    if hard and all((t.get("progress", 0) >= 0.99 or t.get("resolved")) for t in hard) and not pending:
        return True
    # The agent voluntarily idles and nothing is pressing.
    if last_action in ("wait", "noop", "pass", "idle", "do_nothing", "sleep") and not pending:
        optional = _optional_task_ids(state)
        pressing = any(
            (not t.get("resolved")) and t.get("progress", 0) < 0.99
            and _due_today(state, t) and t.get("id") not in optional
            for t in tasks
        )
        if not pressing and not _open_interpersonal(state):
            return True
    return False


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
    """Run one (persona x event x agent) episode to completion.

    ``agent_step`` receives the live ``SimState`` and must return a dict with at
    least ``{"action": {"name": str, "args": dict, "reasoning": str}}``. Other
    keys it returns (``goals``, ``desires``, ``memory``, ``last_reasoning``) are
    merged back into ``state["agent"]`` so graph-internal bookkeeping persists.

    Returns the final ``SimState``.
    """
    state = build_initial_sim_state(persona, event, max_turns=max_turns)
    # Engine-private bookkeeping (never serialised into the agent's prompt slice,
    # which only reads persona/event/time/location/agent/tasks/relationships/
    # new_observations). Keeps mechanical flags out of the model's view.
    state["flags"] = {"start_day": _day_of(state["current_time"]), "_discovered": set()}
    state["_history"] = []

    while not state["finished"] and state["turn"] < state["max_turns"]:
        # 1. Deliver any due scheduled events (behaviour-conditioned) on top of
        #    the consequences carried over from last turn.
        state["new_observations"] = (state.get("new_observations") or []) + _fire_due_events(state)

        # 2. Ask the agent for a decision (it perceives new_observations).
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
        consequences = _apply_action(state, action)
        consequences += _enforce_deadlines(state)
        observation = " ".join(o["content"] for o in consequences) or "(no observable change)"
        state["trajectory"].append({
            "turn": state["turn"],
            "timestamp": state["current_time"],
            "state_snapshot": snapshot,
            "action": action,
            "observation": observation,
        })
        state["_history"].append((action.get("name") or "").lower())

        # 4. Carry this turn's consequences forward as next turn's perception.
        state["new_observations"] = consequences
        state["turn"] += 1

        # 5. Termination.
        if _episode_resolved(state, (action.get("name") or "").lower()):
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
