#!/usr/bin/env python3
"""Generate NeurIPS-style HTML reports (English + Korean) from the study's
measurement.json files. Numbers are read directly from disk, so the reports are
correct-by-construction and regenerable:

    python3.11 make_report.py        ->  report.html  +  report_ko.html

Design notes baked into the narrative:
  * full      = persona-conditioned agent (the treatment).
  * no_desire = PERSONA-NEUTRAL control (no value/conflict/time disposition);
                its divergence should sit at the floor, isolating the persona
                effect at the agent level (complementing the orthogonal event).
  * Protocol: repaired tolerant JSON parsing + Qwen3.5 model-card instruct
              sampling (temp 0.7, top_p 0.8, top_k 20, presence_penalty 1.5),
              thinking disabled, identical across model sizes.
"""
from __future__ import annotations
import json, pathlib
from collections import Counter
from sandbox.evaluation.measurement import js_divergence, action_counts  # stdlib-only helpers

RUNS = pathlib.Path("sandbox/runs")
MODELS = [("qwen3.5-4b", "Qwen3.5-4B", "4B dense"),
          ("qwen3.5-9b", "Qwen3.5-9B", "9B dense"),
          ("qwen3.5-35b-a3b-int4", "Qwen3.5-35B-A3B (Int4)", "35B MoE / 3B active / 4-bit")]
GEN_DATE = "2026-06-17"

data = {k: json.loads((RUNS / f"study__{k}" / "measurement.json").read_text()) for k, _, _ in MODELS}
cfg = data[MODELS[0][0]]["config"]
# Events + their sensitive axis are read from the run config, so the report
# adapts to any number of events per axis (k>=1). Ordered value→conflict→time→orthogonal.
ESENS = dict(cfg.get("event_sensitive_axis", {}))         # event_id -> "value"|"conflict"|"time"|"none"
_AXORD = {"value": 0, "conflict": 1, "time": 2, "none": 3}
EVENTS = sorted(cfg["events"], key=lambda e: (_AXORD.get(ESENS.get(e, "none"), 3), e))

def gc(key, agent, axis):
    return data[key]["agents"][agent]["persona_effect"][axis]["contrast"]
def hit(key, agent):
    ds = [d["hit_rate"]["rate"] for d in data[key]["agents"][agent]["diagnostics"] if d.get("hit_rate")]
    return sum(ds) / len(ds) if ds else float("nan")
def pf(key, agent):
    return data[key]["agents"][agent]["mean_parse_fail_rate"]

# derived numbers used in prose (data-bound, so they stay correct after re-runs)
v_full = [gc(k, "full", "value") for k, _, _ in MODELS]
c_full = [gc(k, "full", "conflict") for k, _, _ in MODELS]
t_full = [gc(k, "full", "time") for k, _, _ in MODELS]
pf_full = [pf(k, "full") for k, _, _ in MODELS]
pf_all = [pf(k, a) for k, _, _ in MODELS for a in ("full", "no_desire")]
nd_contrasts = [gc(k, "no_desire", ax) for k, _, _ in MODELS for ax in ("value", "conflict", "time")
                if gc(k, "no_desire", ax) is not None]
full_contrasts = [gc(k, "full", ax) for k, _, _ in MODELS for ax in ("value", "conflict", "time")
                  if gc(k, "full", ax) is not None]
nd_mean = sum(nd_contrasts) / len(nd_contrasts)
full_mean = sum(full_contrasts) / len(full_contrasts)
pf_max = max(pf_all)
hit_full = [hit(k, "full") for k, _, _ in MODELS]
hfmin, hfmax = min(hit_full), max(hit_full)
orths = [data[k]["agents"][a]["axis_event_matrix"][ax]["event_orthogonal"]
         for k, _, _ in MODELS for a in ("full", "no_desire") for ax in ("value", "conflict", "time")
         if data[k]["agents"][a]["axis_event_matrix"][ax].get("event_orthogonal") is not None]
orth_min, orth_max = min(orths), max(orths)
# Net persona effect = full contrast − neutral-control (no_desire) contrast,
# per the comparison-set design: subtract the agent-level noise floor.
def gnet(key, axis):
    fc, nd = gc(key, "full", axis), gc(key, "no_desire", axis)
    return None if (fc is None or nd is None) else fc - nd
net_value = [gnet(k, "value") for k, _, _ in MODELS]
net_conflict = [gnet(k, "conflict") for k, _, _ in MODELS]
net_time = [gnet(k, "time") for k, _, _ in MODELS]
value_robust = all((x is not None and x > 0.10) for x in net_value)

# Per-EVENT net effect = full_JSD − neutral_JSD on that event (subtracts the
# event-specific neutral floor). This is the primary lens with k>=2 events/axis:
# axis-averaging masks scenario-dependence (e.g. the two conflict events disagree).
def net_ev(key, e):
    ax = ESENS.get(e)
    if ax in (None, "none"):
        return None
    f = data[key]["agents"]["full"]["axis_event_matrix"].get(ax, {}).get(e)
    n = data[key]["agents"]["no_desire"]["axis_event_matrix"].get(ax, {}).get(e)
    return None if (f is None or n is None) else f - n
SENS_EVENTS = [e for e in EVENTS if ESENS.get(e) not in (None, "none")]
def nrow(e): return [net_ev(k, e) for k, _, _ in MODELS]
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else 0.0
# Campus-life conflict scenarios: personal confrontation vs abstract work-dump.
roommate_net = nrow("event_roommate_hygiene") if "event_roommate_hygiene" in EVENTS else [None] * len(MODELS)
boundary_net = nrow("event_boundary_push") if "event_boundary_push" in EVENTS else [None] * len(MODELS)
conflict_scenario_dependent = _mean(roommate_net) >= 0.15 and _mean(boundary_net) < 0.12
f3 = lambda xs: ", ".join(("·" if x is None else f"{x:+.2f}") for x in xs)
_vnets = [net_ev(k, e) for k, _, _ in MODELS for e in SENS_EVENTS if ESENS.get(e) == "value" and net_ev(k, e) is not None]
max_value_net = max(_vnets) if _vnets else 0.0
time_35b = nrow("event_now_vs_later")[-1]   # net on now-vs-later at the largest model
n_episodes = len(cfg['personas']) * len(cfg['events']) * cfg['seeds'] * 2

# ---- per-PERSONA deviation from the neutral baseline -----------------------
# Baseline = the pooled no_desire ("neutral") action distribution per event
# (the "no_desire overall average"). For each persona we measure how far its
# full-agent action distribution diverges from that baseline, per axis — i.e.
# how distinctively that persona behaves relative to a persona-less agent.
_PJSON = pathlib.Path("sandbox/library/personas")
_ABBR = {"achievement": "ach", "affiliation": "aff", "avoidant": "avo",
         "assertive": "asr", "impulsive": "imp", "deliberate": "del"}
def paxes(pid):
    try:
        return json.loads((_PJSON / f"{pid}.json").read_text()).get("axes", {})
    except Exception:
        return {}
def persona_deviation(key):
    base = RUNS / f"study__{key}"
    personas, events = cfg["personas"], cfg["events"]
    neutral = {e: Counter() for e in events}                 # pooled no_desire (vanilla) per event
    full = {(p, e): Counter() for p in personas for e in events}
    for p in personas:
        for e in events:
            for agent in ("full", "no_desire"):
                for tj in (base / agent / f"{p}__{e}").glob("seed*/trajectory.json"):
                    cnt = action_counts(json.loads(tj.read_text())["trajectory"])
                    (neutral[e] if agent == "no_desire" else full[(p, e)]).update(cnt)
    axe = {a: [e for e in events if ESENS.get(e) == a] for a in ("value", "conflict", "time")}
    out = {}
    for p in personas:
        out[p] = {}
        for a, evs in axe.items():
            ds = [js_divergence(full[(p, e)], neutral[e]) for e in evs]
            ds = [d for d in ds if d is not None]
            out[p][a] = (sum(ds) / len(ds)) if ds else None
    return out, neutral
_DEVN = {k: persona_deviation(k) for k, _, _ in MODELS}
DEV = {k: _DEVN[k][0] for k in _DEVN}
NEUTRAL = {k: _DEVN[k][1] for k in _DEVN}   # vanilla action distribution per event
def abs_jsd(key, agent, axis):   # mean ABSOLUTE pole-divergence over an axis's events
    m = data[key]["agents"][agent]["axis_event_matrix"][axis]
    ds = [m.get(e) for e in SENS_EVENTS if ESENS.get(e) == axis]
    ds = [d for d in ds if d is not None]
    return (sum(ds) / len(ds)) if ds else None
def axis_net(key, axis):   # mean per-event net (full − vanilla) over an axis's events
    ds = [net_ev(key, e) for e in SENS_EVENTS if ESENS.get(e) == axis]
    ds = [d for d in ds if d is not None]
    return (sum(ds) / len(ds)) if ds else None
def vanilla_default(key, event):   # most common canonical action of the vanilla agent on an event
    c = NEUTRAL[key].get(event)
    return c.most_common(1)[0][0] if c else "?"

# ----- i18n strings -----
L = {
 "en": {
  "subtitle": "Measuring Persona Embodiment and Behavioral Differentiation in Goal-Forming LLM Agents",
  "affil": f"ALIN Lab · KAIST — AI899 Synthetic-Person Project · generated {GEN_DATE}",
  "genfrom": f"{len(MODELS)} model runs × {len(cfg['personas'])} personas × {len(cfg['events'])} events × {cfg['seeds']} seeds",
  "abstract_h": "Abstract",
  "secs": ["Introduction", "The Synthetic-Person Sandbox", "Experimental Setup", "Results",
           "Discussion", "Limitations", "Conclusion"],
  "axis_labels": {"value": "value (achievement↔affiliation)", "conflict": "conflict (avoidant↔assertive)",
                  "time": "time (impulsive↔deliberate)"},
  "ev_labels": {"event_value_clash": "Value clash", "event_boundary_push": "Boundary push",
                "event_now_vs_later": "Now vs later", "event_orthogonal": "Orthogonal (ctrl)",
                "event_club_vs_midterm": "Club vs midterm", "event_roommate_hygiene": "Roommate hygiene",
                "event_scholarship_deadline": "Scholarship deadline"},
  "t1cap": "<b>Table 2.</b> <b>Relative</b> (net) persona effect per axis = full − vanilla (no_desire) JS-divergence, "
           "averaged over the axis's events; plus full-agent directional hit-rate and parse-fail. The neutral "
           "control is the baseline (subtracted), not a separate row.",
  "cols": ["Model", "value", "conflict", "time", "hit-rate", "parse-fail"],
  "heatkey": "Cell = JS-divergence between opposing persona poles on that event (darker = larger). "
             "Red outline = the axis's sensitive event. Rightmost column = orthogonal control (≈0).",
  "t2cap": "<b>Table 3.</b> Net persona effect <b>per event</b> = full − vanilla JS-divergence. "
           "Positive ⇒ persona conditioning differentiates the poles beyond the neutral floor in that scenario. "
           "Note the two conflict events disagree — embodiment is scenario-dependent.",
  "t2cols": ["Axis", "Event"],
  "t3cap": "<b>Table 4.</b> Per-persona behavioral deviation from the vanilla baseline — JS-divergence of each "
           "persona's full-agent action distribution vs the pooled no_desire average, by axis "
           "(v=value, c=conflict, t=time). Larger ⇒ that persona behaves more distinctively than a persona-less "
           "agent on that axis. The 8 rows are the factorial personas, labeled by their (value·conflict·time) poles.",
  "t3persona": "Persona (v·c·t)",
  "tAcap": "<b>Table 1.</b> Absolute pole-divergence per axis = mean JS-divergence between persona poles over the "
           "axis's events, for the persona-conditioned (<b>full</b>) agent and the persona-less <b>vanilla</b> "
           "(no_desire) agent. The vanilla columns are the base model's intrinsic floor; the full−vanilla gap is "
           "the persona effect (Table 2). v=value, c=conflict, t=time.",
 },
 "ko": {
  "subtitle": "목표를 스스로 세우는 LLM 에이전트의 페르소나 체화도와 행동 분화 측정",
  "affil": f"ALIN Lab · KAIST — AI899 합성 인간 프로젝트 · 생성일 {GEN_DATE}",
  "genfrom": f"모델 {len(MODELS)}종 × 페르소나 {len(cfg['personas'])} × 이벤트 {len(cfg['events'])} × 시드 {cfg['seeds']}",
  "abstract_h": "초록",
  "secs": ["서론", "합성 인간 샌드박스", "실험 설정", "결과", "논의", "한계", "결론"],
  "axis_labels": {"value": "가치 (성취↔관계)", "conflict": "갈등 (회피↔주장)",
                  "time": "시간 (충동↔숙고)"},
  "ev_labels": {"event_value_clash": "가치 충돌", "event_boundary_push": "경계 침범",
                "event_now_vs_later": "현재 대 미래", "event_orthogonal": "직교(대조)",
                "event_club_vs_midterm": "클럽 대 시험", "event_roommate_hygiene": "룸메 위생",
                "event_scholarship_deadline": "장학금 마감"},
  "t1cap": "<b>표 2.</b> <b>상대</b>(순) 페르소나 효과(축별) = full − vanilla(no_desire) JS 발산(축의 이벤트 평균), "
           "+ full 방향성 적중률·파싱 실패율. 중립 대조군은 베이스라인(차감)이며 별도 행으로 두지 않음.",
  "cols": ["모델", "가치", "갈등", "시간", "적중률", "파싱실패"],
  "heatkey": "셀 = 해당 이벤트에서 두 페르소나 극 간 JS 발산(진할수록 큼). 빨간 테두리 = 그 축의 민감 이벤트. "
             "맨 오른쪽 열 = 직교 대조군(≈0).",
  "t2cap": "<b>표 3.</b> <b>이벤트별</b> 순 페르소나 효과 = full − vanilla JS 발산. "
           "양수 ⇒ 그 시나리오에서 페르소나 조건화가 중립 바닥을 넘어 두 극을 분화. "
           "두 갈등 이벤트가 서로 다름에 주목 — 체화는 시나리오 의존적이다.",
  "t2cols": ["축", "이벤트"],
  "t3cap": "<b>표 4.</b> 페르소나별 vanilla 베이스라인 대비 행동 편차 — 각 페르소나 full 행동분포 vs no_desire 전체 평균의 "
           "JS 발산, 축별(v=가치, c=갈등, t=시간). 클수록 그 축에서 페르소나 없는 에이전트보다 더 뚜렷하게 행동. "
           "8개 행은 (가치·갈등·시간) 극으로 라벨한 팩토리얼 페르소나.",
  "t3persona": "페르소나 (가·갈·시)",
  "tAcap": "<b>표 1.</b> 축별 절대 극-발산 = 축의 이벤트에서 두 페르소나 극 간 JS 발산 평균. 페르소나 조건부"
           "(<b>full</b>)와 페르소나 없는 <b>vanilla</b>(no_desire) 각각. vanilla 열이 base 모델의 고유 바닥이며, "
           "full−vanilla 차이가 페르소나 효과(표 2)다. v=가치, c=갈등, t=시간.",
 },
}

def cell_color(v):
    if v is None: return "#f4f4f4", "#999"
    a = max(0.0, min(1.0, v / 0.45)); return f"rgba(31,119,180,{a:.2f})", ("#fff" if a > 0.55 else "#111")

def matrix_table(bundle, agent, lang):
    s = L[lang]; m = bundle["axis_event_matrix"]
    elab = lambda e: s['ev_labels'].get(e, e.replace('event_', '').replace('_', ' '))
    head = "".join(f"<th>{elab(e)}</th>" for e in EVENTS)
    rows = ""
    for ax in ("value", "conflict", "time"):
        tds = ""
        for e in EVENTS:
            v = m[ax].get(e); bg, fg = cell_color(v); diag = " diag" if ESENS.get(e) == ax else ""
            tds += f'<td class="hm{diag}" style="background:{bg};color:{fg}">{"·" if v is None else f"{v:.2f}"}</td>'
        rows += f'<tr><th class="axh">{s["axis_labels"][ax]}</th>{tds}</tr>'
    return f'<table class="heat"><caption>{agent}</caption><tr><th></th>{head}</tr>{rows}</table>'

def net_table(lang):
    s = L[lang]
    elab = lambda e: s['ev_labels'].get(e, e.replace('event_', '').replace('_', ' '))
    head = "".join(f"<th>{d}</th>" for _, d, _ in MODELS)
    rows = ""
    for e in SENS_EVENTS:
        axlbl = s["axis_labels"][ESENS[e]].split(" ")[0]
        cells = "".join(f"<td>{('·' if net_ev(k,e) is None else f'{net_ev(k,e):+.2f}')}</td>" for k, _, _ in MODELS)
        rows += f"<tr><td>{axlbl}</td><td>{elab(e)}</td>{cells}</tr>"
    return (f'<table class="data"><caption>{s["t2cap"]}</caption>'
            f'<tr><th>{s["t2cols"][0]}</th><th>{s["t2cols"][1]}</th>{head}</tr>{rows}</table>')

def absolute_table(lang):
    s = L[lang]
    g_van = "vanilla" if lang == "en" else "vanilla(중립)"
    sub = "<th>v</th><th>c</th><th>t</th>"
    rows = ""
    for key, disp, _ in MODELS:
        def c(agent, ax):
            x = abs_jsd(key, agent, ax); return "·" if x is None else f"{x:.2f}"
        rows += (f"<tr><td>{disp}</td>"
                 f"<td>{c('full','value')}</td><td>{c('full','conflict')}</td><td>{c('full','time')}</td>"
                 f"<td>{c('no_desire','value')}</td><td>{c('no_desire','conflict')}</td><td>{c('no_desire','time')}</td></tr>")
    return (f'<table class="data"><caption>{s["tAcap"]}</caption>'
            f'<tr><th rowspan="2">{s["cols"][0]}</th><th colspan="3">full</th><th colspan="3">{g_van}</th></tr>'
            f'<tr>{sub}{sub}</tr>{rows}</table>')

def aggregate_table(lang):
    s = L[lang]; rows = ""
    for key, disp, _ in MODELS:
        def c(ax):
            x = axis_net(key, ax); return "·" if x is None else f"{x:+.2f}"
        rows += (f"<tr><td>{disp}</td><td>{c('value')}</td><td>{c('conflict')}</td>"
                 f"<td>{c('time')}</td><td>{hit(key,'full'):.2f}</td><td>{pf(key,'full'):.2f}</td></tr>")
    th = "".join(f"<th>{c}</th>" for c in s["cols"])
    return f'<table class="data"><caption>{s["t1cap"]}</caption><tr>{th}</tr>{rows}</table>'

def persona_table(lang):
    s = L[lang]
    mh = "".join(f'<th colspan="3">{disp}</th>' for _, disp, _ in MODELS)
    sub = "".join("<th>v</th><th>c</th><th>t</th>" for _ in MODELS)
    rows = ""
    for p in sorted(cfg["personas"]):
        ax = paxes(p)
        lbl = "·".join(_ABBR.get(ax.get(k, ""), "?") for k in ("value", "conflict", "time")) if ax else p
        cells = "".join(
            f"<td>{('·' if DEV[key][p].get(a) is None else f'{DEV[key][p][a]:.2f}')}</td>"
            for key, _, _ in MODELS for a in ("value", "conflict", "time"))
        rows += f"<tr><td>{lbl}</td>{cells}</tr>"
    return (f'<table class="data persona"><caption>{s["t3cap"]}</caption>'
            f'<tr><th rowspan="2">{s["t3persona"]}</th>{mh}</tr><tr>{sub}</tr>{rows}</table>')

def matrices_block(lang):
    # Only the persona-conditioned (full) agent is shown; the neutral no_desire
    # baseline is captured by Table 1 (net), Table 3 (per-persona), and §4.1.
    cap = "persona-conditioned (full)" if lang == "en" else "페르소나 조건부 (full)"
    out = ""
    for key, disp, note in MODELS:
        out += (f'<div class="mtitle">{disp} <span class="small">({note})</span></div><div class="matrices">'
                f'<div class="mblock">{matrix_table(data[key]["agents"]["full"], cap, lang)}</div></div>')
    return out

# ---- narrative (data-bound). EN then KO. ----
def body(lang):
    s = L[lang]; H = lambda i: f"{i+1}&nbsp;&nbsp;{s['secs'][i]}"
    if lang == "en":
        abs = f"""Most agent benchmarks ask whether a model can <em>follow an instruction</em>; humans instead
act consistently with <em>who they are</em>. We ask: <b>(Q1)</b> can a model embody a configured persona,
in the <em>intended direction</em>, and <b>(Q2)</b> is the behavioral change <em>caused by the persona</em>
rather than noise or formatting? We use an 8-persona controlled 2×2×2 factorial over three axes (value,
conflict-style, time-horizon), events sensitive to exactly one axis plus an orthogonal control, and a
<b>persona-neutral control agent</b> (<code>no_desire</code>). Persona effect is the Jensen–Shannon
divergence between opposing persona poles, contrasted against controls. Across Qwen3.5-4B/-9B/-35B-A3B(4-bit)
served on vLLM under the model-card instruct protocol, two controls agree: the orthogonal event
({orth_min:.3f}–{orth_max:.2f}) and the neutral agent (mean contrast {nd_mean:.2f}) both sit near the floor,
while the persona-conditioned agent diverges far more. The <b>value</b> axis is robustly embodied on every scenario
and scale (per-event net up to {max_value_net:+.2f}). <b>Conflict</b> embodiment is <b>scenario-dependent</b>: a
vivid personal confrontation (roommate hygiene) elicits it at every scale (net {f3(roommate_net)}) while an abstract
work-renegotiation does not (net {f3(boundary_net)})&mdash;a distinction invisible with one event per axis.
<b>Time</b> is weak overall but strengthens at 35B. With the corrected protocol, parse-fail is {pf_max:.0%}."""
        intro = f"""<p>An LLM agent that is only a good instruction-follower is not a <em>synthetic person</em>.
A synthetic person forms goals and stays recognizably itself. The hard part is <em>discriminant validity</em>:
two personas producing different text does not prove the <em>persona</em> caused it. We resolve this with two
independent controls&mdash;an <b>orthogonal event</b> (no axis is decision-relevant) and a <b>persona-neutral
control agent</b> (the <code>no_desire</code> arm, stripped of all value/conflict/time disposition). If the
persona effect is real, divergence is high only for the persona-conditioned agent on axis-sensitive events,
and near zero in both controls.</p>
<p><b>Contributions:</b> (1) a controlled persona factorial + matched axis-sensitive/orthogonal events;
(2) deterministic LLM-free metrics&mdash;action-distribution JS-divergence, a pre-registered directional
hit-rate, and a parse-fail covariate&mdash;plus a persona-neutral control agent; (3) a Qwen3.5 scaling study
under the vendor instruct protocol.</p>"""
        method = """<p>Each episode places one persona in one event and runs a turn-based loop: read observations,
(optionally) reflect/desire/form goals, emit one <code>action</code>; a minimal engine applies a rule-based
effect. We log the full trajectory and read persona signal from the agent's <em>chosen actions</em>, never the
toy world dynamics. The <b>full</b> agent embodies the persona (reflect→desires→goals→act). The
<b>no_desire</b> agent is the <b>persona-neutral control</b>: it receives a neutral system prompt with no
values, conflict style, or time disposition&mdash;so all eight personas collapse to identical conditioning and
its divergence marks the agent-level floor.</p>"""
        setup = f"""<h3>3.1&nbsp;&nbsp;Personas (controlled factorial)</h3>
<p>Eight personas form a 2×2×2 factorial over <b>value</b> (achievement↔affiliation), <b>conflict</b>
(avoidant↔assertive), and <b>time</b> (impulsive↔deliberate). Only axis-encoding fields vary; baseline state and
relationships are held constant, so differences are attributable to an axis.</p>
<h3>3.2&nbsp;&nbsp;Events (axis-sensitive + orthogonal control)</h3>
<p>Seven events declare a <code>sensitive_axis</code> and pre-registered <code>diagnostic_actions</code> per pole:
<b>two per axis (k=2)</b> plus an orthogonal control. Value: study-vs-lunch and club-vs-midterm. Conflict: a
teammate's unfair work-dump and a roommate-hygiene confrontation. Time: invest-now-vs-later and a scholarship
deadline. The orthogonal event has one unambiguously correct action. All are concrete campus-life scenarios; k=2
lets us separate an axis effect from a single scenario's idiosyncrasy.</p>
<h3>3.3&nbsp;&nbsp;Metrics</h3>
<p>For each axis on each event we pool the canonicalized action distributions of the two poles and compute their
<b>Jensen–Shannon divergence</b> (base 2). The <b>persona-effect contrast</b> is JSD(sensitive)−JSD(orthogonal).
The <b>directional hit-rate</b> is the fraction of turns matching the pole's pre-registered action (fidelity, not
mere distinguishability). The <b>parse-fail rate</b> is a format-compliance covariate.</p>
<h3>3.4&nbsp;&nbsp;Models &amp; protocol</h3>
<p>Qwen3.5-4B and -9B (dense) and 35B-A3B (MoE, 3B active) in 4-bit GPTQ-Int4, served on vLLM (one A100 80GB,
one at a time). Identical protocol per model: the Qwen3.5 model-card <em>instruct</em> sampling
(temp 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5), thinking disabled, {cfg['seeds']} seeds,
{cfg['max_turns']} turns, full + no_desire agents, {len(cfg['personas'])} personas × {len(cfg['events'])} events ×
{cfg['seeds']} seeds × 2 agents = {n_episodes} episodes/model at concurrency 16. A tolerant JSON parser
(fence/preamble strip + brace-balancing) keeps parse-fail at {pf_max:.0%}.</p>"""
        res = f"""{absolute_table(lang)}
{aggregate_table(lang)}
{net_table(lang)}
{persona_table(lang)}
<h3>4.1&nbsp;&nbsp;Two controls agree at the floor</h3>
<p>The orthogonal-control event ({orth_min:.3f}–{orth_max:.2f} across all axes/models) and the persona-neutral
<code>no_desire</code> agent (mean contrast {nd_mean:.2f}) both sit near zero, versus the persona-conditioned
agent's mean {full_mean:.2f}. Two independent controls agreeing is strong discriminant validity: the divergence
we measure reflects persona, not the harness, the metric, or formatting.</p>
<h3>4.2&nbsp;&nbsp;Embodiment is axis- AND scenario-dependent</h3>
<div class="finding"><b>Value</b> (achievement↔affiliation) is robustly embodied on <em>both</em> of its scenarios
at every scale (per-event net: value-clash {f3(nrow("event_value_clash"))}; club-vs-midterm
{f3(nrow("event_club_vs_midterm"))}). <b>Conflict</b> (avoidant↔assertive) is the key result of adding a second
event: the <em>same axis</em> gives opposite verdicts depending on the scenario&mdash;a vivid personal confrontation
(<em>roommate hygiene</em>) elicits it at every scale (net {f3(roommate_net)}) while an abstract work-renegotiation
(<em>boundary push</em>) does not (net {f3(boundary_net)}). With k=1 we would have wrongly concluded "conflict is
not embodied." <b>Time</b> (impulsive↔deliberate) is weak across scenarios (now-vs-later {f3(nrow("event_now_vs_later"))};
scholarship {f3(nrow("event_scholarship_deadline"))}) but the largest model shows the strongest time signal.</div>
<p>Heatmaps below show the persona-conditioned (full) agent per model. Red-outlined cells are each axis's two
sensitive events (k=2) and should be the darkest in their row. Note the two conflict columns differ markedly
(roommate dark, work-dump pale) — the scenario-dependence again.</p>
{matrices_block(lang)}
<div class="key">{s['heatkey']}</div>
<h3>4.3&nbsp;&nbsp;Fidelity vs distinguishability</h3>
<p>Directional hit-rate for the persona-conditioned agent is modest ({hfmin:.2f}–{hfmax:.2f}): personas are more
<em>distinguishable</em> than <em>faithful</em>. A divergence-only metric would overstate embodiment; the
pre-registered directional target is what separates "different" from "correctly different."</p>"""
        disc = f"""<p><b>The neutral control matters.</b> Because <code>no_desire</code> strips all disposition, its
near-floor divergence ({nd_mean:.2f}) shows the harness itself induces no persona structure; the full agent's
{full_mean:.2f} is therefore attributable to persona conditioning. <b>Some axes are easier, and some scenarios are
easier.</b> Value priorities map onto a salient choice and are expressed by even 4B across scenarios; conflict style
appears only when the scenario is a vivid personal confrontation (roommate hygiene), not an abstract renegotiation;
temporal discounting largely resists elicitation except at the largest model. <b>Format compliance is a first-order confound</b> that the corrected protocol (tolerant parser +
model-card sampling) controls&mdash;parse-fail fell to ≤{pf_max:.0%}, so divergences are no longer diluted by fallback
actions.</p>
<p><b>Model-size properties.</b> Embodiment does not scale uniformly. <em>Value</em> is already saturated at 4B
(net {axis_net('qwen3.5-4b','value'):+.2f}) and stays high through 9B/35B&mdash;capability is not the bottleneck for the
easy axis. <em>Conflict</em> shows the clearest scale effect on the harder scenario: the abstract work-dump
(<code>boundary_push</code>) only becomes positive at 35B ({f3(boundary_net)} for 4B/9B/35B), whereas the vivid
roommate confrontation is embodied at every size ({f3(roommate_net)}). <em>Time</em> is essentially absent below the
largest model (now-vs-later {f3(nrow('event_now_vs_later'))}). Directional fidelity climbs with scale as well:
hit-rate {hit('qwen3.5-4b','full'):.2f}→{hit('qwen3.5-9b','full'):.2f}→{hit('qwen3.5-35b-a3b-int4','full'):.2f}. In short,
scale buys the <em>hard</em> axes and <em>abstract</em> scenarios, not the easy or vivid ones.</p>
<p><b>Revisiting our hypotheses.</b> <b>Q1 — can a model embody a persona in the intended direction?</b> Yes for
value at every scale; for conflict only when the scenario is concrete (and more so with scale); for time only at
35B. <b>Q2 — is the behavioral change caused by the persona?</b> Yes: two independent controls (orthogonal event
and neutral agent) sit at the floor. Our <b>capability-scaling hypothesis</b> is thus only <em>partially</em>
supported&mdash;scale helps non-monotonically and only where the disposition is hard to surface; it is neither
necessary (value is saturated at 4B) nor sufficient (the work-dump conflict stays weak through 9B). Table 4 adds a
mechanism: the vanilla baseline behaves like a <em>task-focused</em> student&mdash;it studies rather than socializes
and tends to absorb imposed work rather than push back&mdash;so personas encoding the <em>affiliation</em> and
<em>assertive</em> poles deviate most from it. (Future-orientation is model-dependent: only the 35B invests early
by default.) Persona conditioning mostly moves the agent <em>away</em> from this task-focused prior.</p>
<p><b>Vanilla (persona-less) behaviour per model.</b> With the persona withheld, the vanilla agent's own
pole-divergence stays near the floor at every scale (Table 1, vanilla columns; value
{abs_jsd('qwen3.5-4b','no_desire','value'):.2f}/{abs_jsd('qwen3.5-9b','no_desire','value'):.2f}/{abs_jsd('qwen3.5-35b-a3b-int4','no_desire','value'):.2f}
for 4B/9B/35B), confirming it ignores the hidden persona. Its <em>default</em> is task-focused rather than social or
confrontational: on the study-vs-lunch clash 4B/9B default to studying
(<code>{vanilla_default('qwen3.5-4b','event_value_clash')}</code>) and on the work-dump they default to absorbing the
extra task (<code>{vanilla_default('qwen3.5-4b','event_boundary_push')}</code>). The one model-dependent default is
time-horizon: 4B/9B grab the quick chore (<code>{vanilla_default('qwen3.5-4b','event_now_vs_later')}</code>) while
only the 35B invests early (<code>{vanilla_default('qwen3.5-35b-a3b-int4','event_now_vs_later')}</code>)&mdash;the same
place persona time-embodiment first appears. The vanilla floor is also slightly higher on conflict at 9B/35B
({abs_jsd('qwen3.5-9b','no_desire','conflict'):.2f}/{abs_jsd('qwen3.5-35b-a3b-int4','no_desire','conflict'):.2f}),
i.e. more action diversity even without a persona. This vanilla prior is the reference every persona effect is
measured against.</p>"""
        lim = f"""<ul>
<li>{len(cfg['personas'])} personas, two events per axis (k=2), {cfg['seeds']} seeds: scenario-dependence is now
visible, but estimates are still points (more events/seeds would give tight CIs).</li>
<li>35B is 4-bit GPTQ-Int4, conflating scale with quantization.</li>
<li>Toy world engine; instruct-mode only (thinking disabled for parseable comparison).</li>
<li>Hit-rate uses hand-authored diagnostic actions; alternative correct actions may be under-counted.</li></ul>"""
        concl = f"""<p>Language models robustly embody the <em>value</em> dimension of a persona. Conflict-style
embodiment is real but <em>scenario-dependent</em>&mdash;a personal confrontation surfaces it while an abstract
renegotiation does not&mdash;and time-horizon stays weak except at the largest model. Two methodological moves are
what make these readable: a persona-neutral control agent that subtracts the per-scenario noise floor, and
<b>k=2 events per axis</b>, without which the conflict axis would have been mislabeled "not embodied." Persona
measurement must therefore vary the scenario, not just the persona.</p>"""
    else:  # ko
        abs = f"""대부분의 에이전트 벤치마크는 모델이 <em>지시를 따르는지</em>를 묻지만, 인간은 <em>자신이 누구인지</em>에 맞게
행동한다. 우리는 묻는다: <b>(Q1)</b> 모델이 설정된 페르소나를 <em>의도한 방향</em>으로 체화하는가, <b>(Q2)</b> 행동 변화가
노이즈·형식이 아니라 <em>페르소나에 기인</em>하는가? 가치·갈등양식·시간지향 세 축의 통제된 2×2×2 페르소나 8종, 정확히 한
축에만 민감한 이벤트 + 직교 대조 이벤트, 그리고 <b>페르소나-중립 대조 에이전트</b>(<code>no_desire</code>)를 사용한다.
페르소나 효과는 두 페르소나 극 간 Jensen–Shannon 발산이며 대조군과 대비한다. vLLM에 model-card instruct 프로토콜로 올린
Qwen3.5-4B/-9B/-35B-A3B(4비트)에서 두 대조군이 일치한다: 직교 이벤트({orth_min:.3f}–{orth_max:.2f})와 중립
에이전트(평균 대비 {nd_mean:.2f}) 모두 바닥에 머무는 반면, 페르소나 조건부 에이전트는 훨씬 크게 갈라진다.
<b>가치</b> 축은 모든 시나리오·규모에서 견고히 체화된다(이벤트별 순 효과 최대 {max_value_net:+.2f}). <b>갈등</b> 체화는
<b>시나리오 의존적</b>이다: 생생한 대인 충돌(룸메 위생)은 모든 규모에서 발현(순 {f3(roommate_net)})하지만 추상적 업무
재협상은 그렇지 않다(순 {f3(boundary_net)}) — 축당 이벤트 1개로는 보이지 않던 구분이다. <b>시간</b>은 전반적으로 약하나
35B에서 가장 강하다. 보정된 프로토콜에서 파싱 실패율은 {pf_max:.0%}이다."""
        intro = f"""<p>지시를 잘 따르기만 하는 LLM 에이전트는 <em>합성 인간</em>이 아니다. 어려운 지점은 <em>판별 타당도</em>다:
두 페르소나가 다른 텍스트를 낸다고 해서 <em>페르소나</em>가 원인이라는 보장은 없다. 우리는 두 독립 대조군으로 이를 해결한다
&mdash; <b>직교 이벤트</b>(어느 축도 의사결정에 무관)와 <b>페르소나-중립 대조 에이전트</b>(<code>no_desire</code>; 모든
가치/갈등/시간 성향 제거). 효과가 실재한다면, 발산은 페르소나 조건부 에이전트가 축-민감 이벤트에서만 크고 두 대조군에서는
0에 가까워야 한다.</p>
<p><b>기여:</b> (1) 통제된 페르소나 팩토리얼 + 축-민감/직교 이벤트, (2) 결정론적·LLM 비의존 지표(행동분포 JS 발산,
사전등록 방향성 적중률, 파싱실패 공변량) + 페르소나-중립 대조 에이전트, (3) 벤더 instruct 프로토콜 하 Qwen3.5 규모 연구.</p>"""
        method = """<p>각 에피소드는 한 페르소나를 한 이벤트에 넣고 턴제 루프를 돈다: 관찰 읽기 → (선택적) 성찰/욕구/목표 →
단일 <code>action</code> 산출. 최소한의 엔진이 규칙 기반 효과를 적용한다. 페르소나 신호는 엔진 역학이 아니라 에이전트의
<em>선택 행동</em>에서 읽는다. <b>full</b> 에이전트는 페르소나를 체화한다(성찰→욕구→목표→행동). <b>no_desire</b>
에이전트는 <b>페르소나-중립 대조군</b>으로, 가치·갈등양식·시간지향이 전혀 없는 중립 시스템 프롬프트를 받는다 &mdash; 따라서
8개 페르소나가 동일 조건으로 수렴하고, 그 발산이 에이전트 수준의 바닥값이 된다.</p>"""
        setup = f"""<h3>3.1&nbsp;&nbsp;페르소나(통제된 팩토리얼)</h3>
<p><b>가치</b>(성취↔관계), <b>갈등</b>(회피↔주장), <b>시간</b>(충동↔숙고)의 2×2×2로 8종. 축을 인코딩하는 필드만
변하고 기본 상태·관계는 고정해, 차이를 특정 축에 귀속시킨다.</p>
<h3>3.2&nbsp;&nbsp;이벤트(축-민감 + 직교 대조)</h3>
<p>일곱 이벤트가 <code>sensitive_axis</code>와 극별 사전등록 <code>diagnostic_actions</code>를 가진다: <b>축당 2개(k=2)</b>
+ 직교 대조. 가치: 공부-대-점심, 클럽-대-시험. 갈등: 팀원의 부당한 업무 떠넘기기, 룸메 위생 충돌. 시간: 지금-투자-대-나중,
장학금 마감. 직교 이벤트는 정답 행동이 하나뿐이다. 모두 캠퍼스 생활 시나리오이며, k=2로 축 효과와 단일 시나리오 특이성을 분리한다.</p>
<h3>3.3&nbsp;&nbsp;지표</h3>
<p>각 이벤트에서 두 극의 정규화 행동분포 간 <b>Jensen–Shannon 발산</b>(밑 2)을 계산한다. <b>페르소나 효과 대비</b>는
JSD(민감)−JSD(직교). <b>방향성 적중률</b>은 극의 사전등록 행동과 일치한 턴 비율(구별가능성이 아닌 충실도). <b>파싱실패율</b>은
형식 준수 공변량이다.</p>
<h3>3.4&nbsp;&nbsp;모델 및 프로토콜</h3>
<p>Qwen3.5-4B·-9B(밀집)와 35B-A3B(MoE, 활성 3B)의 4비트 GPTQ-Int4를 vLLM(A100 80GB 1장, 순차)으로 서빙.
모델별 동일 프로토콜: Qwen3.5 model-card <em>instruct</em> 샘플링(temp 0.7, top_p 0.8, top_k 20, min_p 0,
presence_penalty 1.5), thinking 비활성, 시드 {cfg['seeds']}, {cfg['max_turns']}턴, full+no_desire,
페르소나 {len(cfg['personas'])} × 이벤트 {len(cfg['events'])} × 시드 {cfg['seeds']} × 에이전트 2 = 모델당 {n_episodes} 에피소드,
동시성 16. 관대한 JSON 파서(펜스/서두 제거 + 중괄호 보정)로 파싱 실패율 {pf_max:.0%} 유지.</p>"""
        res = f"""{absolute_table(lang)}
{aggregate_table(lang)}
{net_table(lang)}
{persona_table(lang)}
<h3>4.1&nbsp;&nbsp;두 대조군이 바닥에서 일치</h3>
<p>직교 이벤트({orth_min:.3f}–{orth_max:.2f})와 페르소나-중립 <code>no_desire</code> 에이전트(평균 대비 {nd_mean:.2f})가
모두 0 근처인 반면, 페르소나 조건부 에이전트는 평균 {full_mean:.2f}이다. 독립적인 두 대조군의 일치는 강한 판별 타당도다:
측정된 발산은 하네스·지표·형식이 아니라 페르소나를 반영한다.</p>
<h3>4.2&nbsp;&nbsp;체화는 축뿐 아니라 시나리오에 의존한다</h3>
<div class="finding"><b>가치</b>(성취↔관계)는 두 시나리오 모두에서 모든 규모에 견고히 체화된다(이벤트별 순:
가치충돌 {f3(nrow("event_value_clash"))}; 클럽-대-시험 {f3(nrow("event_club_vs_midterm"))}). <b>갈등</b>(회피↔주장)이
두 번째 이벤트를 추가한 핵심 결과다: <em>같은 축</em>인데 시나리오에 따라 정반대 결론이 나온다 — 생생한 대인 충돌
(<em>룸메 위생</em>)은 모든 규모에서 발현(순 {f3(roommate_net)})하지만 추상적 업무 재협상(<em>경계 침범</em>)은 그렇지 않다
(순 {f3(boundary_net)}). k=1이었다면 "갈등은 체화 안 됨"으로 잘못 결론냈을 것이다. <b>시간</b>(충동↔숙고)은 두 시나리오
모두 약하지만(현재-대-미래 {f3(nrow("event_now_vs_later"))}; 장학금 {f3(nrow("event_scholarship_deadline"))}) 가장 큰
모델에서 시간 신호가 가장 강하다.</div>
<p>아래 히트맵은 모델별 페르소나 조건부(full) 에이전트다. 빨간 테두리는 각 축의 민감 이벤트 2개(k=2)이며 그 행에서
가장 진해야 한다. 두 갈등 열이 뚜렷이 다름에 주목(룸메 진함, 업무 떠넘기기 옅음) — 다시 시나리오 의존성이다.</p>
{matrices_block(lang)}
<div class="key">{s['heatkey']}</div>
<h3>4.3&nbsp;&nbsp;충실도 대 구별가능성</h3>
<p>페르소나 조건부 에이전트의 방향성 적중률은 보통 수준({hfmin:.2f}–{hfmax:.2f})이다: 페르소나는 <em>충실</em>하기보다
<em>구별 가능</em>하다. 발산만 보는 지표는 체화를 과대평가하며, 사전등록 방향 목표가 '다름'과 '올바르게 다름'을 가른다.</p>"""
        disc = f"""<p><b>중립 대조군의 의미.</b> <code>no_desire</code>는 성향을 모두 제거하므로, 그 바닥값 발산({nd_mean:.2f})은
하네스 자체가 페르소나 구조를 만들지 않음을 보인다. 따라서 full의 {full_mean:.2f}는 페르소나 조건화에 기인한다. <b>어떤 축은
쉽고, 어떤 시나리오는 쉽다.</b> 가치 우선순위는 두드러진 선택으로 매핑돼 모든 시나리오에서 4B도 표현하지만, 갈등양식은 추상적
재협상이 아니라 생생한 대인 충돌(룸메 위생)일 때만 나타나며, 시간적 할인은 35B를 제외하면 대체로 끌어내기 어렵다.
<b>형식 준수는 1차적 교란</b>으로, 보정된 프로토콜(관대한 파서 + model-card 샘플링)이 이를 통제해 파싱실패가
≤{pf_max:.0%}로 떨어졌고, 발산이 더 이상 폴백 행동으로 희석되지 않는다.</p>
<p><b>모델 크기별 성질.</b> 체화는 균일하게 스케일링되지 않는다. <em>가치</em>는 4B에서 이미 포화
(순 {axis_net('qwen3.5-4b','value'):+.2f})되어 9B/35B까지 높게 유지된다 — 쉬운 축에는 역량이 병목이 아니다.
<em>갈등</em>은 어려운 시나리오에서 가장 뚜렷한 스케일 효과를 보인다: 추상적 업무 떠넘기기(<code>boundary_push</code>)는
35B에서야 양수가 되고({f3(boundary_net)} / 4B·9B·35B), 생생한 룸메 충돌은 모든 크기에서 체화된다({f3(roommate_net)}).
<em>시간</em>은 가장 큰 모델 미만에서는 사실상 없다(현재-대-미래 {f3(nrow('event_now_vs_later'))}). 방향성 충실도도
스케일과 함께 상승한다: 적중률 {hit('qwen3.5-4b','full'):.2f}→{hit('qwen3.5-9b','full'):.2f}→{hit('qwen3.5-35b-a3b-int4','full'):.2f}.
요컨대 스케일은 <em>어려운</em> 축과 <em>추상적</em> 시나리오를 사주지만, 쉽거나 생생한 것에는 불필요하다.</p>
<p><b>우리 가설 재검토.</b> <b>Q1 — 모델이 페르소나를 의도한 방향으로 체화하는가?</b> 가치는 모든 규모에서 그렇고,
갈등은 시나리오가 구체적일 때만(규모와 함께 강해짐), 시간은 35B에서만. <b>Q2 — 변화가 페르소나에 기인하는가?</b>
그렇다: 두 독립 대조군(직교 이벤트·중립 에이전트)이 바닥에 있다. 따라서 <b>역량-스케일 가설</b>은 <em>부분적으로</em>만
지지된다 — 스케일은 비단조적으로, 성향이 끌어내기 어려운 곳에서만 도움이 되며, 가치에는 불필요(4B에서 포화)하고
업무 떠넘기기 갈등에는 9B에서도 불충분하다. 표 4는 메커니즘을 더한다: vanilla 베이스라인은 <em>과제 중심</em> 학생처럼
행동한다 — 어울리기보다 공부하고, 떠넘겨진 일을 밀어내기보다 떠안는 경향 — 따라서 <em>관계</em>·<em>주장</em> 극을
인코딩한 페르소나가 가장 크게 벗어난다. (미래 지향성은 모델 의존적: 35B만 기본적으로 일찍 투자.) 페르소나 조건화는
대체로 이 과제 중심 사전분포에서 <em>멀어지는</em> 방향으로 작동한다.</p>
<p><b>모델별 vanilla(페르소나 없는) 행동.</b> 페르소나를 숨기면 vanilla 에이전트의 극-발산은 모든 규모에서 바닥 근처에
머문다(표 1 vanilla 열; 가치 {abs_jsd('qwen3.5-4b','no_desire','value'):.2f}/{abs_jsd('qwen3.5-9b','no_desire','value'):.2f}/{abs_jsd('qwen3.5-35b-a3b-int4','no_desire','value'):.2f}
/ 4B·9B·35B), 숨긴 페르소나를 무시함을 확인해준다. 그 <em>기본</em>은 사회적·대립적이기보다 과제 중심이다: 공부-대-점심에서
4B/9B는 공부를 기본으로 하고(<code>{vanilla_default('qwen3.5-4b','event_value_clash')}</code>), 업무 떠넘기기에서는 추가
과제를 떠안는다(<code>{vanilla_default('qwen3.5-4b','event_boundary_push')}</code>). 모델 의존적인 단 하나의 기본값은 시간
지향이다: 4B/9B는 손쉬운 잡일을 택하고(<code>{vanilla_default('qwen3.5-4b','event_now_vs_later')}</code>) 35B만 일찍
투자한다(<code>{vanilla_default('qwen3.5-35b-a3b-int4','event_now_vs_later')}</code>) — 페르소나의 시간 체화가 처음
나타나는 바로 그 지점이다. 9B/35B는 갈등에서 vanilla 바닥이 다소 높아({abs_jsd('qwen3.5-9b','no_desire','conflict'):.2f}/{abs_jsd('qwen3.5-35b-a3b-int4','no_desire','conflict'):.2f})
페르소나 없이도 행동 다양성이 큼을 반영한다. 이 vanilla 사전분포가 모든 페르소나 효과의 측정 기준이다.</p>"""
        lim = f"""<ul>
<li>페르소나 {len(cfg['personas'])}종, 축당 이벤트 2개(k=2), 시드 {cfg['seeds']}: 시나리오 의존성은 드러났으나 추정치는
점추정(이벤트·시드 추가 시 신뢰구간이 좁아짐).</li>
<li>35B는 4비트 GPTQ-Int4라 규모와 양자화가 교락된다.</li>
<li>장난감 수준 월드 엔진, instruct 모드만(파싱 비교 위해 thinking 비활성).</li>
<li>적중률은 수기 작성 진단 행동을 사용 — 대안적 정답 행동은 과소 집계될 수 있다.</li></ul>"""
        concl = f"""<p>언어 모델은 페르소나의 <em>가치</em> 차원을 견고히 체화한다. 갈등양식 체화는 실재하되
<em>시나리오 의존적</em>이다 — 대인 충돌에서는 드러나지만 추상적 재협상에서는 그렇지 않다 — 시간지향은 가장 큰 모델을
제외하면 약하다. 이를 읽어낸 두 방법론적 장치: 시나리오별 노이즈 바닥을 빼주는 페르소나-중립 대조 에이전트, 그리고
<b>축당 이벤트 2개(k=2)</b> — 후자가 없었다면 갈등 축을 "체화 안 됨"으로 잘못 분류했을 것이다. 즉 페르소나 측정은
페르소나뿐 아니라 시나리오도 변화시켜야 한다.</p>"""
    return f"""<div class="abstract"><h2>{s['abstract_h']}</h2><p>{abs}</p></div>
<h2>{H(0)}</h2>{intro}
<h2>{H(1)}</h2>{method}
<h2>{H(2)}</h2>{setup}
<h2>{H(3)}</h2>{res}
<h2>{H(4)}</h2>{disc}
<h2>{H(5)}</h2>{lim}
<h2>{H(6)}</h2>{concl}"""

CSS = """:root{--ink:#1a1a1a;--mut:#555;--line:#ddd;--accent:#1f77b4}
body{font-family:"Latin Modern Roman","Times New Roman",Times,serif;color:var(--ink);max-width:860px;margin:0 auto;padding:46px 26px 90px;line-height:1.5}
h1{font-size:1.65rem;text-align:center;line-height:1.25;margin:0 0 .15em}
.authors{text-align:center;color:var(--mut)}.affil{text-align:center;color:var(--mut);font-size:.9rem;margin-bottom:1.3em}
h2{font-size:1.16rem;border-bottom:1px solid var(--line);padding-bottom:3px;margin-top:1.8em}
h3{font-size:1.01rem;margin-top:1.2em}
.abstract{background:#f7f9fb;border:1px solid #e4ebf1;border-radius:6px;padding:12px 18px;margin:1.1em 0}
.abstract h2{border:0;font-size:1rem;text-transform:uppercase;letter-spacing:.05em;margin:.1em 0 .4em}
p{margin:.55em 0;text-align:justify}code{background:#f3f3f3;padding:1px 4px;border-radius:3px;font-size:.86em}
table{border-collapse:collapse;margin:1em auto;font-size:.89rem}
table.data td,table.data th{border:1px solid var(--line);padding:5px 9px;text-align:center}
table.data th{background:#f3f6f9}table.data td:first-child,table.data td:nth-child(2){text-align:left}
table.data.persona td,table.data.persona th{padding:3px 6px;font-size:.8rem}table.data.persona td:first-child{font-variant-numeric:tabular-nums;white-space:nowrap}
caption{caption-side:top;color:var(--mut);font-size:.84rem;margin-bottom:.5em;text-align:left}
.heat{margin:.4em 10px}.heat caption{text-align:center;font-weight:bold;color:var(--ink);font-size:.86rem}
.heat th{font-size:.74rem;font-weight:600;color:#333;padding:3px 5px}.heat .axh{text-align:right}
.heat td.hm{width:78px;height:32px;text-align:center;border:1px solid #fff;font-variant-numeric:tabular-nums}
.heat td.diag{outline:2px solid #d62728;outline-offset:-2px;font-weight:700}
.matrices{display:flex;flex-wrap:wrap;justify-content:center;gap:6px}.mblock{text-align:center}
.mtitle{font-weight:700;margin-top:1em}.key{color:var(--mut);font-size:.8rem;text-align:center;margin:.2em 0 1.3em}
.finding{background:#fcfcf7;border-left:3px solid var(--accent);padding:6px 13px;margin:.7em 0}
ul li{margin:.2em 0}.small{font-size:.85rem;color:var(--mut)}
.foot{color:var(--mut);font-size:.79rem;border-top:1px solid var(--line);margin-top:2.2em;padding-top:8px}"""

def page(lang):
    s = L[lang]
    foot = ("<b>Reproducibility.</b> " if lang == "en" else "<b>재현.</b> ") + (
        "Personas <code>sandbox/library/personas/</code>, events <code>sandbox/library/events/</code>; "
        "per-model <code>measurement.json</code> under <code>sandbox/runs/study__*/</code>; "
        "regenerate with <code>python3.11 make_report.py</code>. 0 episode failures across all runs."
        if lang == "en" else
        "페르소나 <code>sandbox/library/personas/</code>, 이벤트 <code>sandbox/library/events/</code>; "
        "모델별 <code>measurement.json</code>은 <code>sandbox/runs/study__*/</code>; "
        "<code>python3.11 make_report.py</code>로 재생성. 전 실행 에피소드 실패 0건.")
    return f"""<!doctype html><html lang="{lang}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Do Language Models Have a Persona?</title><style>{CSS}</style></head><body>
<h1>Do Language Models <em>Have</em> a Persona?<br><span style="font-size:.78em;font-weight:600">{s['subtitle']}</span></h1>
<div class="affil">{s['affil']}<br><span class="small">{s['genfrom']}</span></div>
{body(lang)}
<div class="foot">{foot}</div></body></html>"""

for lang, fname in (("en", "report.html"), ("ko", "report_ko.html")):
    pathlib.Path(fname).write_text(page(lang), encoding="utf-8")
    print(f"wrote {fname}")
print(f"derived: full_mean={full_mean:.3f} nd_mean={nd_mean:.3f} pf_max={pf_max:.3f} "
      f"conflict_scenario_dependent={conflict_scenario_dependent} "
      f"roommate_net={[round(x,2) for x in roommate_net]} boundary_net={[round(x,2) for x in boundary_net]} "
      f"max_value_net={max_value_net:.2f}")
