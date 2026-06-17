#!/usr/bin/env python3
"""Build the interactive Synthetic-Person demo.

The generated artifact is a single-screen chat/workbench interface driven by
presenter clicks on workflow modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEMO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_ROOT.parent
RUNS = PROJECT_ROOT / "sandbox" / "runs"
LIB = PROJECT_ROOT / "sandbox" / "library"
OUT = DEMO_ROOT / "synthetic_person_demo.html"

MODEL_KEY = "qwen3.5-4b"
MODEL_LABEL = "Qwen3.5-4B"
EVENT_ID = "event_value_clash"
ACH_PID = "persona_ach_asr_imp"
AFF_PID = "persona_aff_asr_imp"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def persona(pid: str) -> dict[str, Any]:
    return read_json(LIB / "personas" / f"{pid}.json")


def event(eid: str) -> dict[str, Any]:
    return read_json(LIB / "events" / f"{eid}.json")


def trajectory(agent: str, pid: str, seed: int) -> dict[str, Any]:
    return read_json(
        RUNS
        / f"study__{MODEL_KEY}"
        / agent
        / f"{pid}__{EVENT_ID}"
        / f"seed{seed}"
        / "trajectory.json"
    )


def first_turn(agent: str, pid: str, seed: int) -> dict[str, Any]:
    tj = trajectory(agent, pid, seed)
    step = tj["trajectory"][0]
    state = step["state_snapshot"]
    next_state = tj["trajectory"][1]["state_snapshot"] if len(tj["trajectory"]) > 1 else tj["final_state"]
    return {
        "action": step.get("action", {}),
        "observation": step.get("observation", ""),
        "agent": state.get("agent", {}),
        "tasks": state.get("tasks", []),
        "relationships": state.get("relationships", []),
        "nextTasks": next_state.get("tasks", []),
        "nextRelationships": next_state.get("relationships", []),
    }


def action_label(action: dict[str, Any]) -> str:
    args = action.get("args") or {}
    label = args.get("task") or args.get("event") or args.get("role") or ""
    return f"{action.get('name', 'wait')}({label})" if label else f"{action.get('name', 'wait')}()"


def axis_effects() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, label in [
        ("qwen3.5-4b", "4B"),
        ("qwen3.5-9b", "9B"),
        ("qwen3.5-35b-a3b-int4", "35B-A3B"),
    ]:
        m = read_json(RUNS / f"study__{key}" / "measurement.json")
        full = m["agents"]["full"]["persona_effect"]
        neutral = m["agents"]["no_desire"]["persona_effect"]
        rows.append(
            {
                "model": label,
                "value": full["value"]["sensitive_jsd"] - neutral["value"]["sensitive_jsd"],
                "conflict": full["conflict"]["sensitive_jsd"] - neutral["conflict"]["sensitive_jsd"],
                "time": full["time"]["sensitive_jsd"] - neutral["time"]["sensitive_jsd"],
                "parseFail": m["agents"]["full"]["mean_parse_fail_rate"],
            }
        )
    return rows


def build_data() -> dict[str, Any]:
    ev = event(EVENT_ID)
    ach = persona(ACH_PID)
    aff = persona(AFF_PID)
    init = ev["initial_state"]
    measurement = read_json(RUNS / f"study__{MODEL_KEY}" / "measurement.json")
    full = measurement["agents"]["full"]
    neutral = measurement["agents"]["no_desire"]
    aff_turn = first_turn("full", AFF_PID, 1)
    ach_turn = first_turn("full", ACH_PID, 0)
    neutral_turn = first_turn("no_desire", ACH_PID, 0)

    return {
        "model": MODEL_LABEL,
        "event": {
            "id": ev["id"],
            "title": "Study block vs lunch invite",
            "visible": ev.get("visible_information", [])[:2],
            "tasks": init.get("tasks", []),
            "relationships": init.get("relationships", []),
        },
        "personas": {
            "achievement": {
                "id": ach["id"],
                "axes": ach["axes"],
                "weights": ach["value_priorities"],
                "turn": ach_turn,
                "actionText": action_label(ach_turn["action"]),
            },
            "affiliation": {
                "id": aff["id"],
                "axes": aff["axes"],
                "weights": aff["value_priorities"],
                "style": aff["communication_style"],
                "time": aff["decision_tendencies"],
                "turn": aff_turn,
                "actionText": action_label(aff_turn["action"]),
            },
            "neutral": {
                "id": "no_desire",
                "turn": neutral_turn,
                "actionText": action_label(neutral_turn["action"]),
            },
        },
        "metrics": {
            "axisEffects": axis_effects(),
            "valueEventFull": full["axis_event_matrix"]["value"][EVENT_ID],
            "valueEventNeutral": neutral["axis_event_matrix"]["value"][EVENT_ID],
            "parseFail": full["mean_parse_fail_rate"],
        },
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synthetic Person Workbench</title>
<style>
:root {
  --bg: #eef2f3;
  --ink: #11211f;
  --muted: #61716e;
  --panel: #ffffff;
  --line: #d2dcda;
  --soft: #f7faf9;
  --teal: #0f8b7d;
  --teal-soft: #def5ef;
  --blue: #2e5eaa;
  --blue-soft: #e4edfb;
  --coral: #d95d39;
  --coral-soft: #fde5dc;
  --amber: #d98b00;
  --amber-soft: #fff1cf;
  --green: #23845f;
  --green-soft: #e2f3eb;
  --shadow: 0 16px 34px rgba(17, 33, 31, 0.13);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  height: 100vh;
  overflow: hidden;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
button { font: inherit; }
.app {
  height: 100vh;
  display: grid;
  grid-template-rows: 42px 1fr;
}
.top {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: 12px;
  padding: 5px 12px;
  background: rgba(255, 255, 255, 0.96);
  border-bottom: 1px solid var(--line);
}
.brand {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 18px;
  font-weight: 950;
  letter-spacing: 0;
}
.mark {
  width: 24px;
  height: 24px;
  border-radius: 7px;
  background: linear-gradient(135deg, var(--teal), var(--blue));
}
.run-meta {
  display: flex;
  gap: 7px;
  align-items: center;
  min-width: 0;
}
.pill {
  border: 1px solid var(--line);
  background: var(--soft);
  border-radius: 999px;
  padding: 4px 8px;
  color: var(--muted);
  font-size: 14px;
  font-weight: 850;
  white-space: nowrap;
}
.pill.hot { background: var(--teal-soft); color: var(--teal); border-color: #9bd8ce; }
.undo {
  width: 28px;
  height: 28px;
  border: 0;
  background: transparent;
  color: rgba(17, 33, 31, 0.38);
  font-size: 20px;
  cursor: pointer;
}
.undo:hover { color: var(--ink); }
.workspace {
  min-height: 0;
  padding: 8px;
  display: grid;
  grid-template-columns: minmax(280px, 0.52fr) minmax(320px, 0.58fr) minmax(620px, 1.48fr);
  gap: 8px;
}
.pane {
  min-height: 0;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: 0 10px 24px rgba(17, 33, 31, 0.1);
  overflow: hidden;
}
.pane-head {
  height: 42px;
  padding: 6px 14px;
  border-bottom: 1px solid var(--line);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.pane-title {
  font-size: 22px;
  font-weight: 950;
  line-height: 1;
}
.pane-sub {
  color: var(--muted);
  font-size: 13px;
  font-weight: 750;
  display: none;
}
.chat {
  height: calc(100% - 42px);
  overflow: hidden;
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  justify-content: stretch;
}
.msg {
  max-width: 96%;
  border-radius: 9px;
  padding: 12px 13px;
  border: 1px solid var(--line);
  background: var(--soft);
  animation: rise 220ms ease both;
  flex: 1 1 0;
  min-height: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
}
.msg.agent { margin-left: auto; background: var(--blue-soft); border-color: #b9c9e7; }
.msg.system { background: var(--teal-soft); border-color: #9bd8ce; }
.msg.world { background: var(--green-soft); border-color: #a7d8c4; }
.msg.eval { background: var(--amber-soft); border-color: #e8c56f; }
.sender {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 15px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--muted);
  font-weight: 950;
  margin-bottom: 6px;
}
.bubble-title {
  font-size: 24px;
  line-height: 1.15;
  font-weight: 950;
  margin-bottom: 7px;
}
.bubble-text {
  font-size: 21px;
  line-height: 1.25;
  color: #21302e;
  display: -webkit-box;
  -webkit-line-clamp: 4;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 9px;
}
.chip {
  border-radius: 999px;
  padding: 6px 9px;
  background: rgba(255,255,255,0.72);
  border: 1px solid var(--line);
  font-size: 15px;
  font-weight: 900;
}
.workbench {
  height: calc(100% - 42px);
  display: grid;
  grid-template-rows: auto auto 1fr auto;
  gap: 9px;
  padding: 8px;
}
.persona-strip {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--soft);
  padding: 10px;
}
.persona-name {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: center;
  font-size: 23px;
  font-weight: 950;
}
.axis-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 7px;
  margin-top: 8px;
}
.axis {
  border: 1px solid var(--line);
  background: white;
  border-radius: 7px;
  padding: 8px;
  min-width: 0;
}
.axis b {
  display: block;
  color: var(--muted);
  font-size: 15px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 3px;
}
.axis span {
  display: block;
  font-size: 20px;
  font-weight: 950;
  overflow-wrap: anywhere;
}
.axis.value { background: var(--coral-soft); border-color: #f0b49f; }
.task-panel {
  display: grid;
  gap: 8px;
}
.task {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px;
  background: var(--soft);
}
.task-top {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 8px;
  margin-bottom: 8px;
}
.task-title { font-size: 20px; font-weight: 950; }
.task-deadline { color: var(--muted); font-size: 16px; font-weight: 900; }
.bar {
  height: 18px;
  border-radius: 999px;
  overflow: hidden;
  background: #e7eeee;
}
.bar span {
  display: block;
  height: 100%;
  width: var(--w);
  background: var(--c);
  border-radius: inherit;
  transition: width 280ms ease;
}
.action-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.action-btn {
  border: 1px solid var(--line);
  background: white;
  border-radius: 8px;
  padding: 13px;
  text-align: left;
  min-height: 78px;
  font-weight: 950;
  font-size: 23px;
  color: var(--ink);
}
.action-btn small {
  display: block;
  color: var(--muted);
  font-size: 15px;
  margin-top: 4px;
  font-weight: 800;
}
.action-btn.selected {
  border-color: var(--coral);
  background: var(--coral-soft);
  outline: 3px solid rgba(217, 93, 57, 0.22);
}
.runner {
  border-top: 1px solid var(--line);
  padding-top: 8px;
}
.run-button {
  width: 100%;
  min-height: 54px;
  border: 0;
  border-radius: 8px;
  background: var(--teal);
  color: white;
  font-size: 24px;
  font-weight: 950;
  cursor: pointer;
  box-shadow: 0 12px 24px rgba(15, 139, 125, 0.24);
}
.run-button:hover { filter: brightness(0.96); outline: 5px solid rgba(255, 194, 64, 0.8); }
.run-button.done { background: var(--green); }
.pipeline {
  height: calc(100% - 42px);
  display: grid;
  grid-template-rows: auto 1fr;
  min-height: 0;
}
.modules {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 7px;
  padding: 8px;
  border-bottom: 1px solid var(--line);
}
.module {
  border: 1px solid var(--line);
  background: var(--soft);
  border-radius: 8px;
  min-height: 54px;
  padding: 8px 10px;
  text-align: left;
  cursor: default;
}
.module.current {
  border-color: var(--teal);
  background: var(--teal-soft);
  cursor: pointer;
  animation: pulse 1100ms ease infinite;
}
.module.done {
  border-color: #9ed1bd;
  background: var(--green-soft);
}
.module b {
  display: block;
  font-size: 22px;
  margin-bottom: 0;
}
.module span {
  display: none;
  color: var(--muted);
  font-size: 12px;
  font-weight: 800;
}
.inspector {
  min-height: 0;
  padding: 10px;
  display: grid;
  grid-template-rows: auto 1fr auto;
  gap: 9px;
}
.packet-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.packet-title h2 {
  margin: 0;
  font-size: 31px;
}
.packet-title span {
  color: var(--muted);
  font-size: 17px;
  font-weight: 850;
}
.prompt {
  min-height: 0;
  overflow: auto;
  border-radius: 8px;
  background: #0d1a18;
  color: #f4fbf8;
  padding: 16px;
  font-size: 24px;
  line-height: 1.22;
  white-space: pre-wrap;
  scrollbar-width: none;
}
.prompt::-webkit-scrollbar { width: 0; height: 0; }
.packet-block {
  display: block;
  margin-bottom: 10px;
  padding: 12px 13px;
  border-left: 7px solid rgba(157, 178, 173, 0.45);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.045);
}
.packet-block.active {
  border-color: #91f0d8;
  background: rgba(145, 240, 216, 0.13);
}
.prompt .dim { color: #9db2ad; }
.prompt .hot { color: #91f0d8; font-weight: 950; }
.prompt .coral { color: #ffb49a; font-weight: 950; }
.prompt .blue { color: #adc8ff; font-weight: 950; }
.prompt .big { font-size: 38px; line-height: 1.08; font-weight: 950; }
.prompt .metric-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-top: 10px;
}
.prompt .metric-cell {
  display: block;
  min-height: 112px;
  border-radius: 8px;
  padding: 14px;
  background: rgba(255,255,255,0.065);
  border: 1px solid rgba(157,178,173,0.25);
}
.result-strip {
  display: none;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
.result-card {
  border-radius: 10px;
  border: 1px solid var(--line);
  padding: 11px;
  background: var(--soft);
}
.result-card b {
  display: block;
  font-size: 18px;
}
.result-card span {
  display: block;
  color: var(--muted);
  font-size: 15px;
  margin-top: 3px;
  font-weight: 800;
}
.compare {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  margin-top: 10px;
}
.compare-card {
  border-radius: 9px;
  border: 1px solid var(--line);
  padding: 10px;
  background: white;
}
.compare-card b {
  display: block;
  font-size: 18px;
}
.compare-card span {
  font-size: 16px;
  color: var(--muted);
  font-weight: 850;
}
.metric-row {
  display: grid;
  grid-template-columns: 105px 1fr 70px;
  align-items: center;
  gap: 9px;
  margin-top: 9px;
}
.metric-row label { font-size: 18px; font-weight: 950; color: var(--muted); }
.metric-row strong { font-size: 22px; text-align: right; }
.msg,
.task,
.persona-strip,
.axis,
.action-btn,
.module,
.packet-block,
.compare-card,
.metric-row,
.prompt,
.pane-head {
  transition: none;
}
.msg:hover,
.task:hover,
.persona-strip:hover,
.axis:hover,
.action-btn:hover,
.module:hover,
.packet-block:hover,
.compare-card:hover,
.metric-row:hover,
.prompt:hover,
.pane-head:hover {
  outline: 5px solid rgba(255, 196, 57, 0.9);
  outline-offset: 2px;
  box-shadow: 0 0 0 8px rgba(255, 196, 57, 0.22), 0 12px 28px rgba(17, 33, 31, 0.18);
  border-color: #d99800;
  z-index: 5;
}
.action-btn:hover,
.module.current:hover,
.run-button:hover {
  transform: translateY(-1px);
}
@keyframes rise {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(15, 139, 125, 0.22); }
  50% { box-shadow: 0 0 0 5px rgba(15, 139, 125, 0.12); }
}
@media (max-width: 1200px) {
  .workspace { grid-template-columns: 1fr; overflow: auto; }
  body { overflow: auto; }
  .app { height: auto; min-height: 100vh; }
}
</style>
</head>
<body>
<div class="app">
  <header class="top">
    <div class="brand"><span class="mark"></span><span>Synthetic Person Workbench</span></div>
    <div class="run-meta">
      <span class="pill hot">Qwen3.5-4B</span>
      <span class="pill">value-sensitive task</span>
      <span class="pill" id="runState">ready</span>
    </div>
    <button class="undo" id="undo" title="Undo last step" type="button">↶</button>
  </header>
  <main class="workspace">
    <section class="pane">
      <div class="pane-head">
        <div>
          <div class="pane-title">Trace</div>
          <div class="pane-sub">event_value_clash</div>
        </div>
      </div>
      <div class="chat" id="chat"></div>
    </section>

    <section class="pane">
      <div class="pane-head">
        <div>
          <div class="pane-title">Task + action</div>
          <div class="pane-sub">task state + actions</div>
        </div>
      </div>
      <div class="workbench">
        <div class="persona-strip" id="personaStrip"></div>
        <div class="task-panel" id="taskPanel"></div>
        <div class="action-grid" id="actions"></div>
        <div class="runner"><button class="run-button" id="runStep" type="button"></button></div>
      </div>
    </section>

    <section class="pane">
      <div class="pane-head">
        <div>
          <div class="pane-title">LLM prompt</div>
          <div class="pane-sub">prompt packet</div>
        </div>
      </div>
      <div class="pipeline">
        <div class="modules" id="modules"></div>
        <div class="inspector">
          <div class="packet-title">
            <h2 id="packetTitle"></h2>
            <span id="packetStatus"></span>
          </div>
          <pre class="prompt" id="prompt"></pre>
          <div id="inspectorFoot"></div>
        </div>
      </div>
    </section>
  </main>
</div>

<script>
const DATA = __DATA__;
let step = 0;

const STEPS = [
  { label: "Load persona", module: "Persona", state: "ready" },
  { label: "Inject task", module: "World", state: "task loaded" },
  { label: "Build prompt", module: "Prompt", state: "prompt packet" },
  { label: "Reflect", module: "Reflect", state: "reflection" },
  { label: "Update desires", module: "Desires", state: "desires" },
  { label: "Generate goals", module: "Goals", state: "goals" },
  { label: "Plan action", module: "Action", state: "action chosen" },
  { label: "Apply world", module: "World", state: "world updated" },
  { label: "Measure effect", module: "Measure", state: "finding" }
];

const MODULES = [
  ["Persona", "fields"],
  ["Prompt", "packet"],
  ["Reflect", "summary"],
  ["Desires", "drives"],
  ["Goals", "ranked"],
  ["Action", "JSON"],
  ["World", "update"],
  ["Measure", "controls"]
];

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
}

function pct(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function short(text, max = 150) {
  const s = String(text || "");
  return s.length > max ? `${s.slice(0, max - 3)}...` : s;
}

function firstSentence(text, max = 150) {
  const s = String(text || "").split(/(?<=[.!?])\s+/)[0] || "";
  return short(s, max);
}

function currentLabel() {
  if (step >= STEPS.length) return "Reset run";
  return STEPS[step].label;
}

function actionText(action) {
  const args = action.args || {};
  const label = args.task || args.event || args.role || "";
  return `${action.name || "wait"}${label ? "(" + label + ")" : "()"}`;
}

function getTask(tasks, id) {
  return (tasks || []).find(t => t.id === id) || {};
}

function currentTasks() {
  const t = DATA.personas.affiliation.turn;
  return step >= 8 ? t.nextTasks : t.tasks;
}

function relationshipCloseness() {
  const t = DATA.personas.affiliation.turn;
  const rel = (step >= 8 ? t.nextRelationships : t.relationships)[0] || {};
  return Number(rel.closeness || 0);
}

function taskProgress(id) {
  const t = getTask(currentTasks(), id);
  return Number(t.progress || 0);
}

function renderPersona() {
  const p = DATA.personas.affiliation;
  const visible = step >= 1;
  document.getElementById("personaStrip").innerHTML = visible ? `
    <div class="persona-name"><span>Affiliation persona</span></div>
    <div class="axis-row">
      <div class="axis value"><b>value</b><span>${esc(p.axes.value)}</span></div>
      <div class="axis"><b>conflict</b><span>${esc(p.axes.conflict)}</span></div>
      <div class="axis"><b>time</b><span>${esc(p.axes.time)}</span></div>
    </div>
  ` : `
    <div class="persona-name"><span>Persona not loaded</span></div>
    <div class="axis-row">
      <div class="axis"><b>value</b><span>pending</span></div>
      <div class="axis"><b>conflict</b><span>pending</span></div>
      <div class="axis"><b>time</b><span>pending</span></div>
    </div>
  `;
}

function renderTasks() {
  const tasks = currentTasks();
  const assignment = getTask(tasks, "assignment_value");
  const reading = getTask(tasks, "reading_value");
  const close = relationshipCloseness();
  document.getElementById("taskPanel").innerHTML = `
    <div class="task">
      <div class="task-top"><span class="task-title">Assignment analysis</span><span class="task-deadline">${esc(assignment.deadline || "Mon 18:00")}</span></div>
      <div class="bar"><span style="--w:${pct(taskProgress("assignment_value"))};--c:var(--blue)"></span></div>
    </div>
    <div class="task">
      <div class="task-top"><span class="task-title">Assigned reading</span><span class="task-deadline">${esc(reading.deadline || "Mon 20:00")}</span></div>
      <div class="bar"><span style="--w:${pct(taskProgress("reading_value"))};--c:var(--teal)"></span></div>
    </div>
    <div class="task">
      <div class="task-top"><span class="task-title">cohort_3 closeness</span><span class="task-deadline">${close.toFixed(1)} / 10</span></div>
      <div class="bar"><span style="--w:${Math.round(close * 10)}%;--c:var(--coral)"></span></div>
    </div>
  `;
}

function renderActions() {
  const selected = step >= 7 ? "attend" : "";
  const actions = [
    ["work_on", "assignment", "study now"],
    ["attend", "social_value", "join lunch"],
    ["message", "cohort_3", "text group"],
    ["boundary", "30 min", "compromise"],
    ["wait", "none", "do nothing"]
  ];
  document.getElementById("actions").innerHTML = actions.map(([name, label, sub]) => `
    <button class="action-btn ${selected === name ? "selected" : ""}" type="button">
      ${esc(name)} <small>${esc(label)} · ${esc(sub)}</small>
    </button>
  `).join("");
}

function messageHtml(kind, sender, title, text, chips = []) {
  return `
    <article class="msg ${kind}">
      <div class="sender">${esc(sender)}</div>
      <div class="bubble-title">${esc(title)}</div>
      <div class="bubble-text">${text}</div>
      ${chips.length ? `<div class="chips">${chips.map(c => `<span class="chip">${esc(c)}</span>`).join("")}</div>` : ""}
    </article>`;
}

function renderChat() {
  const p = DATA.personas.affiliation;
  const t = p.turn;
  const ev = DATA.event;
  const messages = [];
  messages.push(messageHtml("system", "Run", "event_value_clash", "Study block vs lunch invite", ["same state", "same actions"]));
  if (step >= 1) {
    messages.push(messageHtml("system", "Persona", "persona_aff_asr_imp",
      `value=<b>affiliation</b> · conflict=<b>assertive</b> · time=<b>impulsive</b>`, [
        `relationships ${pct(p.weights.relationships)}`,
        `academic ${pct(p.weights.academic)}`
      ]));
  }
  if (step >= 2) {
    messages.push(messageHtml("world", "World", "observation",
      `plan_note: assignment analysis, then reading<br>friend_chat: cohort_3 lunch invite`));
  }
  if (step >= 3) {
    messages.push(messageHtml("system", "Prompt", "packet ready",
      "SYSTEM + USER + STATE"));
  }
  if (step >= 4) {
    messages.push(messageHtml("agent", "Reflect", "output",
      "Lunch is happening now while the study plan is unfinished."));
  }
  if (step >= 5) {
    messages.push(messageHtml("agent", "Desires", "output",
      esc(short(t.agent.desires[0], 74)) + "<br>" + esc(short(t.agent.desires[1], 74))));
  }
  if (step >= 6) {
    messages.push(messageHtml("agent", "Goals", "top goal",
      esc(t.agent.goals[0].description), [t.agent.goals[0].priority, t.agent.goals[0].status]));
  }
  if (step >= 7) {
    messages.push(messageHtml("agent", "Action", actionText(t.action),
      esc(firstSentence(t.action.reasoning, 120))));
  }
  if (step >= 8) {
    messages.push(messageHtml("world", "World", "state update",
      esc(t.observation), ["closeness 7.0 -> 7.6"]));
  }
  if (step >= 9) {
    messages.push(messageHtml("eval", "Measure", "finding",
      `achievement: <b>${esc(DATA.personas.achievement.actionText)}</b><br>neutral: <b>${esc(DATA.personas.neutral.actionText)}</b><br>value effect: <b>+0.39</b>`));
  }
  document.getElementById("chat").innerHTML = messages.slice(-4).join("");
}

function moduleState(name) {
  const order = ["Persona", "World", "Prompt", "Reflect", "Desires", "Goals", "Action", "World", "Measure"];
  const lastDone = Math.min(step - 1, order.length - 1);
  const current = step < order.length ? order[step] : "";
  if (name === current) return "current";
  if (order.slice(0, lastDone + 1).includes(name)) return "done";
  return "";
}

function renderModules() {
  document.getElementById("modules").innerHTML = MODULES.map(([name, sub]) => `
    <button class="module ${moduleState(name)}" type="button" data-module="${esc(name)}">
      <b>${esc(name)}</b><span>${esc(sub)}</span>
    </button>
  `).join("");
  document.querySelectorAll(".module.current").forEach(btn => {
    btn.addEventListener("click", advance);
  });
}

function promptText() {
  const p = DATA.personas.affiliation;
  const t = p.turn;
  const ev = DATA.event;
  const blocks = [];
  const block = (at, label, body) => `<span class="packet-block ${step === at ? "active" : ""}"><span class="dim">${label}</span>\n${body}</span>`;
  if (step >= 1) {
    blocks.push(block(1, "SYSTEM / PERSONA", `
Role-play as a synthetic person.
value=<span class="coral">affiliation</span>
academic=<span class="blue">0.15</span>, relationships=<span class="coral">0.50</span>
conflict=assertive, time=impulsive`));
  }
  if (step >= 2) {
    blocks.push(block(2, "USER / OBSERVATIONS", `
plan_note: assignment analysis, then reading
friend_chat: cohort_3 lunch invite
event_action: attend(social_value)
state: energy=6, stress=4, sleep_debt=2`));
  }
  if (step >= 3) {
    blocks.push(block(3, "PROMPT CALL", `
messages=[system_persona, user_observation, state_snapshot]
next=reflect`));
  }
  if (step >= 4) {
    blocks.push(block(4, "REFLECT OUTPUT", `
"Lunch is happening now while study is unfinished."`));
  }
  if (step >= 5) {
    blocks.push(block(5, "DESIRE OUTPUT", `
<span class="coral">"${esc(short(t.agent.desires[0], 112))}"</span>`));
  }
  if (step >= 6) {
    blocks.push(block(6, "GOAL OUTPUT", `
<span class="coral">"${esc(short(t.agent.goals[0].description, 132))}"</span>`));
  }
  if (step >= 7) {
    blocks.push(block(7, "ACTION PROMPT", `
actions=[work_on, attend, message, boundary, wait]
json={"name":"...", "args":{}, "reasoning":"..."}
<span class="dim">MODEL ACTION</span> <span class="hot">${esc(actionText(t.action))}</span>
reason: relationships 0.50 > academic 0.15`));
  }
  if (!blocks.length) {
    return `<span class="dim">Prompt packet</span>\n...`;
  }
  return blocks.join("\n\n");
}

function renderInspectorFoot() {
  const m = DATA.metrics;
  if (step < 9) {
    const completedModule = step > 0 ? STEPS[step - 1].module : "pending";
    document.getElementById("inspectorFoot").innerHTML = `
      <div class="result-strip">
        <div class="result-card"><b>last module</b><span>${esc(completedModule)}</span></div>
        <div class="result-card"><b>agent</b><span>full persona</span></div>
      </div>`;
    return;
  }
  const row = m.axisEffects[0];
  const max = 0.42;
  document.getElementById("inspectorFoot").innerHTML = `
    <div class="compare">
      <div class="compare-card"><b>achievement</b><span>${esc(DATA.personas.achievement.actionText)}</span></div>
      <div class="compare-card"><b>affiliation</b><span>${esc(DATA.personas.affiliation.actionText)}</span></div>
      <div class="compare-card"><b>neutral</b><span>${esc(DATA.personas.neutral.actionText)}</span></div>
    </div>
    ${[
      ["value", row.value, "var(--teal)"],
      ["conflict", row.conflict, "var(--coral)"],
      ["time", row.time, "var(--blue)"],
    ].map(([label, value, color]) => `
      <div class="metric-row">
        <label>${label}</label>
        <div class="bar"><span style="--w:${Math.round(value / max * 100)}%;--c:${color}"></span></div>
        <strong>+${Number(value).toFixed(2)}</strong>
      </div>
    `).join("")}`;
}

function renderInspector() {
  const title = step >= 9 ? "Measured persona effect" : "LLM input packet";
  document.getElementById("packetTitle").textContent = title;
  document.getElementById("packetStatus").textContent = step >= STEPS.length ? "complete" : `${step}/${STEPS.length} components`;
  const promptEl = document.getElementById("prompt");
  promptEl.innerHTML = step >= 9
    ? `<span class="dim">MEASUREMENT</span>
<span class="big">value axis: <span class="hot">+${DATA.metrics.axisEffects[0].value.toFixed(2)}</span></span>

<span class="metric-grid"><span class="metric-cell"><span class="dim">FULL PERSONA</span>
value-task JSD
<span class="hot">${DATA.metrics.valueEventFull.toFixed(2)}</span></span><span class="metric-cell"><span class="dim">NEUTRAL FLOOR</span>
value-task JSD
<span class="hot">${DATA.metrics.valueEventNeutral.toFixed(2)}</span></span><span class="metric-cell"><span class="dim">COUNTERFACTUAL</span>
achievement: work_on
affiliation: attend</span><span class="metric-cell"><span class="dim">PARSE</span>
fail_rate
${DATA.metrics.parseFail.toFixed(2)}</span></span>`
    : promptText();
  promptEl.scrollTop = promptEl.scrollHeight;
  renderInspectorFoot();
}

function renderRunner() {
  const btn = document.getElementById("runStep");
  btn.textContent = step >= STEPS.length ? "Reset run" : `Execute ${STEPS[step].module}`;
  btn.className = `run-button ${step >= STEPS.length ? "done" : ""}`;
  document.getElementById("runState").textContent = step >= STEPS.length ? "complete" : (step === 0 ? "ready" : STEPS[step - 1].state);
}

function render() {
  renderPersona();
  renderTasks();
  renderActions();
  renderChat();
  renderModules();
  renderInspector();
  renderRunner();
}

function advance() {
  if (step >= STEPS.length) step = 0;
  else step += 1;
  render();
}

function undo() {
  step = Math.max(0, step - 1);
  render();
}

document.getElementById("runStep").addEventListener("click", advance);
document.getElementById("undo").addEventListener("click", undo);
window.addEventListener("keydown", event => {
  if (event.key === " " || event.key === "Enter") advance();
  if (event.key === "ArrowLeft") undo();
  if (event.key.toLowerCase() === "r") { step = 0; render(); }
});
render();
</script>
</body>
</html>
"""


def main() -> None:
    data = json.dumps(build_data(), ensure_ascii=True, separators=(",", ":"))
    OUT.write_text(HTML.replace("__DATA__", data), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
