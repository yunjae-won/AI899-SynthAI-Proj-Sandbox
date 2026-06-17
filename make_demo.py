#!/usr/bin/env python3
"""Build a self-contained HTML demo that replays study trajectories as a
multi-turn conversation (env = user, LLM agent = assistant), revealed one step
at a time. For the class-project live demo.

    python3.11 make_demo.py   ->   demo.html   (open in any browser)
"""
from __future__ import annotations
import json, pathlib, html

RUNS_DIR = pathlib.Path("sandbox/runs")
LIB = pathlib.Path("sandbox/library")

# Curated story: the conflict work-dump as the headline showcase of the
# *consequential* engine (assertive renegotiates → the dumped task is actually
# returned to its owner; avoidant absorbs it), then the roommate confrontation,
# value, time, and a 4B→9B→35B scale comparison on the same work-dump.
RUNS = [
    # Conflict · work-dump — the clearest "world answers back" contrast:
    ("qwen3.5-9b", "full", "persona_ach_asr_del", "event_boundary_push", "Assertive → defers the dump back to the teammate", "Conflict · work-dump (9B)"),
    ("qwen3.5-9b", "full", "persona_ach_avo_imp", "event_boundary_push", "Avoidant → silently absorbs the dump", "Conflict · work-dump (9B)"),
    ("qwen3.5-9b", "no_desire", "persona_ach_avo_imp", "event_boundary_push", "Vanilla baseline (no persona)", "Conflict · work-dump (9B)"),
    # Conflict · roommate confrontation — same axis, different scenario:
    ("qwen3.5-9b", "full", "persona_aff_asr_del", "event_roommate_hygiene", "Assertive → proposes a cleanliness pact", "Conflict · roommate (9B)"),
    ("qwen3.5-9b", "full", "persona_ach_avo_del", "event_roommate_hygiene", "Avoidant → relocates to the library", "Conflict · roommate (9B)"),
    # Value · study vs lunch — opposite value poles:
    ("qwen3.5-9b", "full", "persona_ach_asr_del", "event_value_clash", "Achievement → keeps studying", "Value · study vs lunch (9B)"),
    ("qwen3.5-9b", "full", "persona_aff_avo_del", "event_value_clash", "Affiliation → joins the lunch", "Value · study vs lunch (9B)"),
    # Time · now vs later — at 35B, where time embodiment appears:
    ("qwen3.5-35b-a3b-int4", "full", "persona_ach_asr_del", "event_now_vs_later", "Deliberate → invests in the project", "Time · now vs later (35B)"),
    ("qwen3.5-35b-a3b-int4", "full", "persona_aff_avo_imp", "event_now_vs_later", "Impulsive → takes leisure now", "Time · now vs later (35B)"),
    # Scale — the work-dump renegotiation holds at every size:
    ("qwen3.5-4b", "full", "persona_ach_asr_del", "event_boundary_push", "Assertive @ 4B", "Scale · assertive on the work-dump"),
    ("qwen3.5-9b", "full", "persona_ach_asr_del", "event_boundary_push", "Assertive @ 9B", "Scale · assertive on the work-dump"),
    ("qwen3.5-35b-a3b-int4", "full", "persona_ach_asr_del", "event_boundary_push", "Assertive @ 35B", "Scale · assertive on the work-dump"),
]
MODEL_DISP = {"qwen3.5-9b": "Qwen3.5-9B", "qwen3.5-4b": "Qwen3.5-4B",
              "qwen3.5-35b-a3b-int4": "Qwen3.5-35B-A3B-Int4"}


def _persona(pid):
    for d in (LIB / "personas", pathlib.Path("src/personas")):
        p = d / f"{pid}.json"
        if p.exists():
            return json.loads(p.read_text())
    return {"id": pid, "name": pid}


def _event(eid):
    for d in (LIB / "events", pathlib.Path("src/events")):
        p = d / f"{eid}.json"
        if p.exists():
            return json.loads(p.read_text())
    return {"id": eid, "title": eid}


def env_line(act, obs):
    """The environment bubble shows the world's response to the action.

    The engine now emits rich, untruncated observations (NPC replies, task
    progress, revealed insights, deadline beats) and feeds them forward, so we
    surface them directly — the agent's own words already appear in the agent
    bubble via the action args. Fall back to a reconstructed message echo only
    if the engine produced no observation for a message turn."""
    if obs:
        return obs
    name = (act.get("name") or "").lower()
    args = act.get("args") or {}
    if isinstance(args, dict) and args.get("content") and ("message" in name or name == "contact"):
        tgt = args.get("role") or args.get("target") or args.get("to") or "someone"
        return f"\U0001f4e9 Messaged {tgt}: “{args['content']}”"
    return obs or "(no observable change)"


def _state(agent, tasks, rels, time, loc):
    return {
        "time": time, "loc": loc,
        "energy": agent.get("energy"), "sleep": agent.get("sleep_debt_hours"), "stress": agent.get("stress"),
        "tasks": [{"label": t.get("label", t.get("id")), "progress": round(float(t.get("progress", 0) or 0), 2),
                   "deadline": t.get("deadline", "")} for t in (tasks or [])],
        "rels": [{"name": r.get("name", r.get("role")), "trust": r.get("trust")} for r in (rels or [])],
    }


def build_run(key, agent, pid, eid, label, group):
    tj = RUNS_DIR / f"study__{key}" / agent / f"{pid}__{eid}" / "seed0" / "trajectory.json"
    if not tj.exists():
        return None
    traj = json.loads(tj.read_text())
    steps = traj["trajectory"]
    if not steps:
        return None
    persona, event = _persona(pid), _event(eid)
    init = event.get("initial_state", {})
    world = init.get("world", {})
    a0 = {**persona.get("baseline_state", {}), **init.get("agent", {})}
    intro = _state(a0, init.get("tasks", []), init.get("relationships", []),
                   world.get("current_time", "?"), world.get("location", "?"))
    intro["visible"] = [{"channel": v.get("channel", ""), "content": v.get("content", "")}
                        for v in event.get("visible_information", [])]
    # snapshots: per step + final, for the state panel
    snaps = []
    for s in steps:
        ss = s["state_snapshot"]; a = ss.get("agent", {})
        snaps.append(_state(a, ss.get("tasks"), ss.get("relationships"),
                            ss.get("current_time", s.get("timestamp")), ss.get("location", "")))
    fin = traj.get("final_state", {})
    final_state = _state(fin.get("agent", {}), fin.get("tasks"), fin.get("relationships"),
                         fin.get("current_time", ""), "")
    turns = []
    for i, s in enumerate(steps):
        ss = s["state_snapshot"]; a = ss.get("agent", {})
        act = s.get("action", {}) or {}
        turns.append({
            "turn": s.get("turn", i),
            "pre": snaps[i],
            "post": snaps[i + 1] if i + 1 < len(snaps) else final_state,
            "goals": a.get("goals") or [],
            "desires": a.get("desires") or [],
            "action": {"name": act.get("name", "?"), "args": act.get("args", {}),
                       "reasoning": act.get("reasoning", ""), "parse_failed": bool(act.get("parse_failed"))},
            "observation": env_line(act, s.get("observation", "")),
        })
    ax = persona.get("axes")
    return {
        "id": f"{key}__{agent}__{pid}__{eid}", "label": label, "group": group,
        "model": MODEL_DISP.get(key, key), "agent": agent,
        "persona": {"name": persona.get("name", pid),
                    "axes": ax if agent != "no_desire" else None,
                    "hidden": agent == "no_desire"},
        "event": {"title": event.get("title", eid), "summary": event.get("summary", ""),
                  "sensitive_axis": event.get("sensitive_axis", "")},
        "intro": intro, "turns": turns,
    }


runs = [r for r in (build_run(*spec) for spec in RUNS) if r]
DATA = json.dumps(runs, ensure_ascii=False)

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synthetic Person — Trajectory Replay</title>
<style>
 :root{--bg:#0f1419;--panel:#171d26;--line:#2a3340;--ink:#e6edf3;--mut:#8b98a8;--env:#222b36;--agent:#1c3a5e;--accent:#4ea1ff;--good:#3fb950;--warn:#d29922;--bad:#f85149}
 *{box-sizing:border-box}body{margin:0;font-family:-apple-system,"Segoe UI",Roboto,"Noto Sans KR",sans-serif;background:var(--bg);color:var(--ink)}
 header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:center;flex-wrap:wrap;position:sticky;top:0;background:var(--bg);z-index:5}
 header h1{font-size:1.05rem;margin:0;font-weight:700}
 select{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:7px 10px;font-size:.92rem;max-width:420px}
 .wrap{display:grid;grid-template-columns:1fr 320px;gap:0;height:calc(100vh - 60px)}
 .chat{overflow-y:auto;padding:22px 26px 120px}
 .side{border-left:1px solid var(--line);padding:18px 18px 40px;overflow-y:auto;background:var(--panel)}
 .scene{color:var(--mut);font-size:.86rem;margin:0 0 16px;line-height:1.5}
 .scene b{color:var(--ink)}
 .msg{max-width:78%;margin:14px 0;padding:12px 15px;border-radius:14px;line-height:1.5;animation:f .35s ease}
 @keyframes f{from{opacity:0;transform:translateY(8px)}to{opacity:1}}
 .env{background:var(--env);border:1px solid var(--line);border-bottom-left-radius:4px}
 .agent{background:var(--agent);border:1px solid #2d5a8c;border-bottom-right-radius:4px;margin-left:auto}
 .role{font-size:.72rem;letter-spacing:.04em;text-transform:uppercase;color:var(--mut);margin-bottom:6px;font-weight:700}
 .agent .role{color:#9ecbff}
 .think{font-size:.84rem;color:#c9d6e4;background:#0c1c2e;border-left:3px solid var(--accent);padding:6px 10px;border-radius:6px;margin:6px 0}
 .think .lab{color:var(--accent);font-weight:700}
 .goal{font-size:.82rem;color:#c9d6e4;margin:3px 0;padding-left:14px;text-indent:-10px}
 .action{font-size:1rem;margin:8px 0 4px}
 .action .name{font-weight:800;color:#7ee787;font-family:ui-monospace,monospace}
 .action .args{color:var(--mut);font-family:ui-monospace,monospace;font-size:.85rem}
 .reason{color:#dbe4ee}
 .obs{font-family:ui-monospace,monospace;font-size:.85rem;color:#cdd9e5}
 .chip{display:inline-block;font-size:.7rem;padding:1px 7px;border-radius:10px;margin-left:6px;background:var(--bad);color:#fff;font-weight:700}
 .delta{color:var(--mut);font-size:.78rem;margin-top:6px}
 .side h3{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:var(--mut);margin:18px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
 .side h3:first-child{margin-top:0}
 .pax{display:flex;gap:6px;flex-wrap:wrap}.pax span{font-size:.75rem;background:#243; color:#9be39b;border:1px solid #2f5;border-radius:8px;padding:2px 8px}
 .pax .hidden{background:#332;color:#e3c99b;border-color:#5a4}
 .bar{height:8px;background:#0c1219;border-radius:6px;overflow:hidden;margin:3px 0 9px}
 .bar>i{display:block;height:100%;border-radius:6px;transition:width .4s}
 .row{display:flex;justify-content:space-between;font-size:.82rem;margin-top:6px}.row b{font-weight:600}
 .tasklbl{font-size:.78rem;color:#cdd9e5;margin-top:8px}
 footer{position:fixed;bottom:0;left:0;right:320px;background:#0c1117ee;border-top:1px solid var(--line);padding:10px 16px;display:flex;gap:10px;align-items:center;backdrop-filter:blur(6px)}
 button{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px 13px;font-size:.9rem;cursor:pointer}
 button:hover{border-color:var(--accent)}button:disabled{opacity:.4;cursor:default}
 button.play{background:var(--accent);color:#06243f;font-weight:700;border:0}
 .count{color:var(--mut);font-size:.85rem;margin-left:auto}
 input[type=range]{width:120px}
</style></head><body>
<header>
 <h1>🧪 Synthetic Person — Trajectory Replay</h1>
 <select id="run"></select>
 <span class="count" id="meta"></span>
</header>
<div class="wrap">
 <div class="chat" id="chat"></div>
 <div class="side" id="side"></div>
</div>
<footer>
 <button id="restart">⏮ Restart</button>
 <button id="prev">◀ Prev</button>
 <button id="play" class="play">▶ Play</button>
 <button id="next">Next ▶</button>
 <label style="color:var(--mut);font-size:.8rem">speed <input type="range" id="speed" min="400" max="2600" value="1300" step="100"></label>
 <span class="count" id="count"></span>
</footer>
<script>
const DATA = __DATA__;
let R=null, msgs=[], ptr=0, timer=null;
const $=id=>document.getElementById(id);
const esc=s=>(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const argstr=a=>{if(!a||typeof a!=="object")return"";return Object.entries(a).map(([k,v])=>`${k}=${typeof v==="object"?JSON.stringify(v):v}`).join(", ");};

function buildMsgs(r){
 const m=[];
 const vis=(r.intro.visible||[]).map(v=>`<div class="obs">[${esc(v.channel)}] ${esc(v.content)}</div>`).join("");
 m.push({role:"env",st:r.intro,html:`<div class="role">🌍 Environment — setup</div>
   <b>${esc(r.event.title)}</b><div class="delta">${esc(r.event.summary)}</div>
   <div style="margin-top:8px">${vis}</div>`});
 for(const t of r.turns){
   let inner=`<div class="role">🤖 ${esc(r.model)} · ${r.agent} agent — turn ${t.turn}</div>`;
   if(t.desires&&t.desires.length) inner+=`<div class="think"><span class="lab">desires</span> ${esc(t.desires.join(" · "))}</div>`;
   if(t.goals&&t.goals.length){inner+=`<div class="think"><span class="lab">goals</span>`+t.goals.slice(0,3).map(g=>`<div class="goal">• ${esc(g.description||g.id||"")} <span style="color:#8b98a8">(${esc(g.priority||"")})</span></div>`).join("")+`</div>`;}
   inner+=`<div class="action">▶ <span class="name">${esc(t.action.name)}</span><span class="args">(${esc(argstr(t.action.args))})</span>${t.action.parse_failed?'<span class="chip">parse-fail→wait</span>':''}</div>`;
   if(t.action.reasoning) inner+=`<div class="reason">${esc(t.action.reasoning)}</div>`;
   m.push({role:"agent",st:t.pre,html:inner});
   m.push({role:"env",st:t.post,html:`<div class="role">🌍 Environment</div><div class="obs">${esc(t.observation)}</div><div class="delta">⏱ ${esc(t.post.time)}</div>`});
 }
 return m;
}
function barColor(v,max,inv){const f=v/max;const c=inv?(1-f):f;return c>.6?'var(--good)':c>.3?'var(--warn)':'var(--bad)';}
function renderSide(st){
 const ax=R.persona.axes;
 let axhtml = R.persona.hidden
   ? `<span class="hidden">no persona (vanilla)</span>`
   : (ax?Object.entries(ax).map(([k,v])=>`<span>${k}: ${esc(v)}</span>`).join(""):"—");
 const en=st.energy??0, sl=st.sleep??0, sr=st.stress??0;
 let tasks=(st.tasks||[]).map(t=>`<div class="tasklbl">${esc(t.label)} — ${Math.round(t.progress*100)}%</div>
   <div class="bar"><i style="width:${Math.round(t.progress*100)}%;background:var(--accent)"></i></div>`).join("");
 let rels=(st.rels||[]).map(r=>`<div class="row"><span>${esc(r.name)}</span><b>trust ${r.trust}</b></div>`).join("");
 $("side").innerHTML=`
  <h3>Persona</h3><b>${esc(R.persona.name)}</b><div class="pax" style="margin-top:8px">${axhtml}</div>
  <h3>Clock</h3><div class="row"><span>time</span><b>${esc(st.time||"?")}</b></div>
  <h3>State</h3>
   <div class="row"><span>energy</span><b>${en}</b></div><div class="bar"><i style="width:${en*10}%;background:${barColor(en,10)}"></i></div>
   <div class="row"><span>sleep debt (h)</span><b>${sl}</b></div><div class="bar"><i style="width:${Math.min(sl*10,100)}%;background:${barColor(sl,10,true)}"></i></div>
   <div class="row"><span>stress</span><b>${sr}</b></div><div class="bar"><i style="width:${sr*10}%;background:${barColor(sr,10,true)}"></i></div>
  <h3>Tasks</h3>${tasks||'<span style="color:#8b98a8">—</span>'}
  <h3>Relationships</h3>${rels||'<span style="color:#8b98a8">—</span>'}`;
}
function render(){
 const c=$("chat"); c.innerHTML="";
 for(let i=0;i<=ptr&&i<msgs.length;i++){const m=msgs[i];const d=document.createElement("div");d.className="msg "+m.role;d.innerHTML=m.html;c.appendChild(d);}
 const last=msgs[Math.min(ptr,msgs.length-1)]; if(last) renderSide(last.st);
 c.scrollTop=c.scrollHeight;
 $("count").textContent=`step ${ptr+1} / ${msgs.length}`;
 $("prev").disabled=ptr<=0; $("next").disabled=ptr>=msgs.length-1;
}
function loadRun(idx){stop();R=DATA[idx];msgs=buildMsgs(R);ptr=0;
 $("meta").textContent=`${R.event.title}${R.event.sensitive_axis?'  ·  sensitive axis: '+R.event.sensitive_axis:''}`;
 render();}
function step(d){ptr=Math.max(0,Math.min(msgs.length-1,ptr+d));render();if(ptr>=msgs.length-1)stop();}
function play(){if(timer){stop();return;}$("play").textContent="⏸ Pause";timer=setInterval(()=>step(1),+$("speed").value);}
function stop(){if(timer)clearInterval(timer);timer=null;$("play").textContent="▶ Play";}
// build grouped dropdown
const sel=$("run");let lastG=null,og=null;
DATA.forEach((r,i)=>{if(r.group!==lastG){og=document.createElement("optgroup");og.label=r.group;sel.appendChild(og);lastG=r.group;}
 const o=document.createElement("option");o.value=i;o.textContent=r.label;og.appendChild(o);});
sel.onchange=()=>loadRun(+sel.value);
$("next").onclick=()=>step(1);$("prev").onclick=()=>step(-1);$("restart").onclick=()=>{stop();ptr=0;render();};$("play").onclick=play;
document.onkeydown=e=>{if(e.key==="ArrowRight"){step(1);e.preventDefault();}if(e.key==="ArrowLeft")step(-1);if(e.key===" "){play();e.preventDefault();}};
loadRun(0);
</script></body></html>"""

out = pathlib.Path("demo.html")
out.write_text(PAGE.replace("__DATA__", DATA), encoding="utf-8")
print(f"wrote {out} with {len(runs)} runs:")
for r in runs:
    print(f"  [{r['group']}] {r['label']} — {len(r['turns'])} turns")
