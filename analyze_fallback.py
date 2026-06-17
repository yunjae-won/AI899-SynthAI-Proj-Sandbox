#!/usr/bin/env python3
"""Diagnose WHY action-parse fallbacks happen: run real full-agent episodes
against the live 9B server, but wrap call_llm to capture every raw output +
finish_reason + token count, then classify each failure as truncation
(finish_reason=length / unbalanced braces) vs malformed-but-complete JSON
(finish_reason=stop but json.loads errors, e.g. unescaped quotes/newlines)."""
from __future__ import annotations
import json, collections
from langchain_core.messages import SystemMessage, HumanMessage
import sandbox.agents._llm_utils as U
import sandbox.agents.full_agent as FA
from sandbox.registry import resolve_model_config
from sandbox.config import get_llm
from sandbox.world import load_persona, load_event, run_simulation
from sandbox.agents import AGENT_REGISTRY

cfg = resolve_model_config("qwen3.5-9b", temperature=0.7)
print("cfg.max_tokens:", cfg.max_tokens, "extra:", cfg.extra)

records = []
def cap_call_llm(llm, system, user):
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        content = "".join(b["text"] if isinstance(b, dict) and "text" in b else str(b) for b in content)
    content = str(content)
    meta = getattr(resp, "response_metadata", {}) or {}
    usage = getattr(resp, "usage_metadata", None) or {}
    records.append({
        "finish_reason": meta.get("finish_reason") or meta.get("stop_reason"),
        "out_tokens": (usage or {}).get("output_tokens"),
        "len": len(content),
        "raw": content,
    })
    return content

U.call_llm = cap_call_llm
FA.call_llm = cap_call_llm   # full_agent imported the name directly

combos = [
    ("persona_ach_avo_imp", "event_value_clash"),
    ("persona_aff_asr_del", "event_boundary_push"),
    ("persona_ach_asr_imp", "event_now_vs_later"),
    ("persona_aff_avo_del", "event_orthogonal"),
    ("persona_ach_avo_del", "event_boundary_push"),
    ("persona_aff_asr_imp", "event_value_clash"),
    ("persona_ach_asr_del", "event_now_vs_later"),
    ("persona_aff_avo_imp", "event_value_clash"),
]
import concurrent.futures
def _run_one(pe):
    pid, eid = pe
    step = AGENT_REGISTRY["full"](cfg)
    run_simulation(persona=load_persona(pid), event=load_event(eid), agent_step=step, max_turns=7, log_path=None)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(_run_one, combos))

def is_fail(raw):
    return U.parse_json(raw, fallback={"__fail__": True}).get("__fail__") is True

fails = [r for r in records if is_fail(r["raw"])]
n = len(records)
print(f"\nTOTAL llm calls={n}  parse-fail={len(fails)} ({len(fails)/n:.0%})")
print("finish_reason (ALL):", dict(collections.Counter(r["finish_reason"] for r in records)))
print("finish_reason (FAILS):", dict(collections.Counter(r["finish_reason"] for r in fails)))
print("out_tokens (FAILS) sample:", sorted([r["out_tokens"] for r in fails if r["out_tokens"]], reverse=True)[:12])
print("max out_tokens overall:", max((r["out_tokens"] or 0) for r in records))

trunc = [r for r in fails if r["finish_reason"] == "length"]
other = [r for r in fails if r["finish_reason"] != "length"]
print(f"\n=> TRUNCATED (finish_reason=length): {len(trunc)}/{len(fails)}")
print(f"=> COMPLETE-but-unparseable (finish_reason!=length): {len(other)}/{len(fails)}")
print("\n--- why do the COMPLETE-but-unparseable ones fail json.loads? ---")
for r in other[:8]:
    cleaned = U._strip_fences(r["raw"])
    try:
        json.loads(cleaned); err = "loads-OK-on-cleaned (first_json_object issue?)"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    print(f"  fr={r['finish_reason']} len={r['len']} tok={r['out_tokens']} | {err}")
    print(f"     tail: ...{r['raw'][-90:]!r}")
print("\n--- a couple TRUNCATED tails (should end mid-token, no closing brace) ---")
for r in trunc[:4]:
    print(f"  tok={r['out_tokens']} len={r['len']} | tail: ...{r['raw'][-90:]!r}")

# ---- candidate robust parser: strip fences/preamble + brace-balance repair ----
import re as _re
def robust_extract(s):
    s = s.strip()
    s = _re.sub(r"^```(?:json)?", "", s).strip()
    s = _re.sub(r"```$", "", s).strip()
    i = s.find("{")
    if i < 0: return None
    s = s[i:]
    depth = 0; in_str = False; esc = False; end = None
    for j, ch in enumerate(s):
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0: end = j + 1; break
    if end is not None:
        return s[:end]
    if in_str: s += '"'
    return s + ("}" * depth)

import json as _json
rec_ok = 0
for r in fails:
    cand = robust_extract(r["raw"])
    try:
        d = _json.loads(cand)
        if isinstance(d, dict) and "action" in d: rec_ok += 1
    except Exception:
        pass
print(f"\n*** ROBUST PARSER RECOVERY: {rec_ok}/{len(fails)} previously-failed outputs now parse to a valid action ***")
