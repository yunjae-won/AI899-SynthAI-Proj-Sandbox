"""CLI entry point for the sandbox.

Subcommands:
    single    one agent × persona × event episode (+ optional --evaluate)
    matrix    agent × all personas × one event + persona-swap test (legacy)
    study     the persona-measurement grid: factorial personas × axis events ×
              {full,no_desire} × seeds, for ONE model, with deterministic metrics
    aggregate roll up every runs/study__*/measurement.json into a model × axis table
    list      available personas / events / agents / model-keys

Examples:
    python -m sandbox.run study --model-key qwen3.5-9b --seeds 3
    python -m sandbox.run aggregate
    python -m sandbox.run single --model-key gpt-nano \
        --persona persona_ach_avo_imp --event event_value_clash --max-turns 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from .config import LLMConfig
from .world import load_persona, load_event, run_simulation, list_personas, list_events
from .agents import AGENT_REGISTRY
from .evaluation import (
    score_three_axes,
    judge_persona_consistency,
    run_swap_test,
    summarize_agent,
)


RUNS_DIR = Path(__file__).resolve().parent / "runs"

# Defaults for the measurement study.
STUDY_AGENTS = ["full", "no_desire"]   # desire on / off — the retained ablation


def _safe(name: str) -> str:
    """Filesystem-safe fragment (model ids may contain '/')."""
    return str(name).replace("/", "-")


def _build_llm_config(args) -> LLMConfig:
    """Resolve a model. ``--model-key`` (registry) takes precedence over the
    ad-hoc ``--provider/--model`` path. ``--temperature`` is always honored."""
    model_key = getattr(args, "model_key", None)
    if model_key:
        from .registry import resolve_model_config
        return resolve_model_config(
            model_key,
            temperature=args.temperature,
            use_recommended=getattr(args, "use_recommended", False),
        )
    return LLMConfig(provider=args.provider, model=args.model, temperature=args.temperature)


def _factorial_personas() -> List[str]:
    """Persona ids that carry an ``axes`` tag (the controlled factorial)."""
    return [p for p in list_personas() if load_persona(p).get("axes")]


def _axis_events() -> List[str]:
    """Event ids that carry a ``sensitive_axis`` tag (the study events)."""
    return [e for e in list_events() if load_event(e).get("sensitive_axis") is not None]


def _make_record(agent: str, persona: Dict, event: Dict, trajectory: List[Dict]) -> Dict[str, Any]:
    return {
        "agent": agent,
        "persona_id": persona["id"],
        "axes": persona.get("axes", {}),
        "event_id": event["id"],
        "sensitive_axis": event.get("sensitive_axis", "none"),
        "diagnostic_actions": event.get("diagnostic_actions", {}),
        "trajectory": trajectory,
    }


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #


def cmd_single(args) -> int:
    persona = load_persona(args.persona)
    event = load_event(args.event)

    if args.agent not in AGENT_REGISTRY:
        print(f"Unknown agent '{args.agent}'. Choose from: {list(AGENT_REGISTRY)}")
        return 2

    cfg = _build_llm_config(args)
    agent_step = AGENT_REGISTRY[args.agent](cfg)

    out_dir = RUNS_DIR / f"{args.agent}__{persona['id']}__{event['id']}__{_safe(cfg.model)}"
    log_path = out_dir / "trajectory.json"
    print(f"[run] {args.agent} × {persona['id']} × {event['id']}  →  {log_path}")

    final = run_simulation(
        persona=persona,
        event=event,
        agent_step=agent_step,
        max_turns=args.max_turns,
        log_path=log_path,
    )

    if args.evaluate:
        trajectory = final["trajectory"]
        axes = score_three_axes(
            event=event,
            final_state={
                "tasks": final["tasks"],
                "relationships": final["relationships"],
                "agent": final["agent"],
            },
            trajectory=trajectory,
            initial_relationships=event.get("initial_state", {}).get("relationships", []),
        )
        consistency = judge_persona_consistency(persona, trajectory, cfg)
        evaluation = {"three_axes": axes, "persona_consistency": consistency}
        (out_dir / "evaluation.json").write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(evaluation, ensure_ascii=False, indent=2))
    return 0


def cmd_matrix(args) -> int:
    """Run `agent × all personas × event` then cross-persona swap test."""
    event = load_event(args.event)
    persona_ids = args.personas or list_personas()
    cfg = _build_llm_config(args)
    agent_step_factory = AGENT_REGISTRY[args.agent]

    results: List[Dict[str, Any]] = []
    swap_input = []  # list of (persona_dict, trajectory)
    base = RUNS_DIR / f"matrix__{args.agent}__{event['id']}__{_safe(cfg.model)}"
    for pid in persona_ids:
        persona = load_persona(pid)
        agent_step = agent_step_factory(cfg)
        out_dir = base / pid
        log_path = out_dir / "trajectory.json"
        print(f"[matrix] {args.agent} × {pid} × {event['id']}  →  {log_path}")
        final = run_simulation(
            persona=persona,
            event=event,
            agent_step=agent_step,
            max_turns=args.max_turns,
            log_path=log_path,
        )
        trajectory = final["trajectory"]
        axes = score_three_axes(
            event=event,
            final_state={
                "tasks": final["tasks"],
                "relationships": final["relationships"],
                "agent": final["agent"],
            },
            trajectory=trajectory,
            initial_relationships=event.get("initial_state", {}).get("relationships", []),
        )
        consistency = judge_persona_consistency(persona, trajectory, cfg)
        (out_dir / "evaluation.json").write_text(
            json.dumps({"three_axes": axes, "persona_consistency": consistency},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        results.append({"persona_id": pid, "three_axes": axes,
                        "consistency_overall": consistency.get("overall", {})})
        swap_input.append((persona, trajectory))

    swap = run_swap_test(swap_input, cfg, trials_per_pair=args.swap_trials)
    summary_path = base / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "event_id": event["id"],
        "agent": args.agent,
        "per_persona": results,
        "swap_test": {"n_matches": swap["n_matches"], "accuracy": swap["accuracy"]},
    }
    summary_path.write_text(json.dumps({**summary, "swap_detail": swap["matches"]},
                                       ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_study(args) -> int:
    """The persona-measurement grid for ONE model.

    factorial personas × axis events × {full,no_desire} × seeds, scored with the
    deterministic measurement metrics (axis×event divergence matrix, directional
    hit-rate, parse-fail covariate). LLM judge / swap test only if asked.
    """
    cfg = _build_llm_config(args)
    if not getattr(args, "model_key", None):
        print("study requires --model-key (so runs and the registry stay aligned).")
        return 2

    personas = args.personas or _factorial_personas()
    events = args.events or _axis_events()
    agents = args.agents or STUDY_AGENTS
    event_meta = {eid: load_event(eid).get("sensitive_axis", "none") for eid in events}

    base = RUNS_DIR / f"study__{args.model_key}"
    records_by_agent: Dict[str, List[Dict]] = {a: [] for a in agents}

    agents = [a for a in agents if a in AGENT_REGISTRY]
    personas_obj = {pid: load_persona(pid) for pid in personas}
    events_obj = {eid: load_event(eid) for eid in events}

    # One job per (agent, persona, event, seed). Episodes are independent, so we
    # run them through a thread pool: each agent_step does blocking HTTP calls to
    # vLLM, which then BATCHES the concurrent requests — the whole point of
    # --concurrency. (Sequential runs waste vLLM's batching entirely.)
    jobs = [(agent, pid, eid, seed)
            for agent in agents for pid in personas
            for eid in events for seed in range(args.seeds)]
    total = len(jobs)
    conc = max(1, args.concurrency)
    print(f"[study] model={args.model_key} agents={agents} "
          f"personas={len(personas)} events={len(events)} seeds={args.seeds} "
          f"→ {total} episodes, concurrency={conc}", flush=True)

    import concurrent.futures
    import threading
    import time
    lock = threading.Lock()
    progress = {"done": 0, "fail": 0}
    t0 = time.time()

    def _run_job(job):
        agent, pid, eid, seed = job
        persona, event = personas_obj[pid], events_obj[eid]
        log_path = base / agent / f"{pid}__{eid}" / f"seed{seed}" / "trajectory.json"
        try:
            agent_step = AGENT_REGISTRY[agent](cfg)  # fresh agent + LLM client per episode
            final = run_simulation(persona=persona, event=event, agent_step=agent_step,
                                   max_turns=args.max_turns, log_path=log_path)
            rec = _make_record(agent, persona, event, final["trajectory"])
        except Exception as exc:  # one bad episode must not kill the whole study
            rec = None
            with lock:
                progress["fail"] += 1
            print(f"  !! FAILED {agent} {pid}×{eid} s{seed}: {type(exc).__name__}: {str(exc)[:160]}",
                  flush=True)
        with lock:
            progress["done"] += 1
            n = progress["done"]
            if n == 1 or n % 10 == 0 or n == total:
                el = time.time() - t0
                rate = n / el if el > 0 else 0.0
                eta_min = ((total - n) / rate / 60) if rate > 0 else 0.0
                print(f"  [{n}/{total}] {agent} {pid}×{eid} s{seed} | "
                      f"{rate*60:.0f} ep/min | ETA {eta_min:.1f} min | fails={progress['fail']}",
                      flush=True)
        return agent, rec

    if conc == 1:
        results = [_run_job(j) for j in jobs]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
            results = list(ex.map(_run_job, jobs))

    for agent, rec in results:
        if rec is not None:
            records_by_agent[agent].append(rec)
    el = time.time() - t0
    print(f"[study] {total - progress['fail']}/{total} episodes ok in {el/60:.1f} min "
          f"({progress['fail']} failed)", flush=True)

    # Deterministic measurement per agent.
    agents_out: Dict[str, Any] = {}
    for agent, records in records_by_agent.items():
        bundle = summarize_agent(records, event_meta)
        if args.judge:
            bundle["judge"] = _study_judge(records, cfg)
        if args.swap_trials > 0:
            bundle["swap_by_event"] = _study_swap(records, events, cfg, args.swap_trials)
        agents_out[agent] = bundle

    measurement = {
        "model_key": args.model_key,
        "model_config": {
            "provider": cfg.provider, "model": cfg.model,
            "temperature": cfg.temperature, "base_url": cfg.base_url,
        },
        "config": {
            "personas": personas, "events": events, "agents": agents,
            "seeds": args.seeds, "max_turns": args.max_turns,
            "event_sensitive_axis": event_meta,
        },
        "agents": agents_out,
    }
    base.mkdir(parents=True, exist_ok=True)
    (base / "measurement.json").write_text(
        json.dumps(measurement, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_study_summary(measurement)
    print(f"\n[study] wrote {base / 'measurement.json'}")
    return 0


def _study_judge(records: List[Dict], cfg) -> List[Dict[str, Any]]:
    """Optional: persona-consistency overall on each record (cost ~1 call/record)."""
    out = []
    for r in records:
        persona = load_persona(r["persona_id"])
        c = judge_persona_consistency(persona, r["trajectory"], cfg)
        out.append({"persona_id": r["persona_id"], "event_id": r["event_id"],
                    "overall": c.get("overall", {})})
    return out


def _study_swap(records: List[Dict], events: List[str], cfg, trials: int) -> Dict[str, Any]:
    """Optional: per-event blind persona-swap test (seed-0 trajectory per persona)."""
    out: Dict[str, Any] = {}
    for eid in events:
        seen = {}
        for r in records:
            if r["event_id"] == eid and r["persona_id"] not in seen:
                seen[r["persona_id"]] = r["trajectory"]
        swap_input = [(load_persona(pid), traj) for pid, traj in seen.items()]
        if len(swap_input) >= 2:
            swap = run_swap_test(swap_input, cfg, trials_per_pair=trials)
            out[eid] = {"accuracy": swap["accuracy"], "n_matches": swap["n_matches"]}
    return out


def _fmt_jsd(v) -> str:
    return "  .  " if v is None else f"{v:.2f}"


def _print_study_summary(measurement: Dict[str, Any]) -> None:
    print(f"\n=== STUDY {measurement['model_key']} "
          f"({measurement['model_config']['model']}) ===")
    meta = measurement["config"]["event_sensitive_axis"]
    events = measurement["config"]["events"]
    for agent, bundle in measurement["agents"].items():
        print(f"\n[{agent}] axis × event divergence (diagonal should dominate):")
        header = "  axis\\event   " + " ".join(f"{e[:14]:>14}" for e in events)
        print(header)
        for axis, row in bundle["axis_event_matrix"].items():
            cells = " ".join(f"{_fmt_jsd(row.get(e)):>14}" for e in events)
            print(f"  {axis:<12} {cells}")
        print("  persona-effect contrast (sensitive − orthogonal):")
        for axis, pe in bundle["persona_effect"].items():
            print(f"    {axis:<10} {_fmt_jsd(pe['contrast'])}  "
                  f"(sensitive={pe['sensitive_event']})")
        print(f"  mean parse-fail rate: {bundle['mean_parse_fail_rate']:.2f}")


def cmd_aggregate(args) -> int:
    """Roll up every runs/study__*/measurement.json into a model × axis table."""
    files = sorted(RUNS_DIR.glob("study__*/measurement.json"))
    if not files:
        print("No study runs found under runs/study__*/. Run `study` first.")
        return 1

    rows: List[Dict[str, Any]] = []
    for f in files:
        m = json.loads(f.read_text(encoding="utf-8"))
        for agent, bundle in m["agents"].items():
            pe = bundle["persona_effect"]
            hit_rates = [d["hit_rate"]["rate"] for d in bundle["diagnostics"]
                         if d.get("hit_rate")]
            rows.append({
                "model_key": m["model_key"],
                "model": m["model_config"]["model"],
                "agent": agent,
                "contrast_value": pe["value"]["contrast"],
                "contrast_conflict": pe["conflict"]["contrast"],
                "contrast_time": pe["time"]["contrast"],
                "mean_hit_rate": (sum(hit_rates) / len(hit_rates)) if hit_rates else None,
                "mean_parse_fail": bundle["mean_parse_fail_rate"],
            })

    (RUNS_DIR / "aggregate.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("model × axis persona-effect (contrast = sensitive − orthogonal JSD)\n")
    print(f"{'model_key':<16}{'agent':<11}{'value':>7}{'conflict':>9}{'time':>7}"
          f"{'hit':>7}{'pfail':>7}")
    for r in rows:
        print(f"{r['model_key']:<16}{r['agent']:<11}"
              f"{_fmt_jsd(r['contrast_value']):>7}{_fmt_jsd(r['contrast_conflict']):>9}"
              f"{_fmt_jsd(r['contrast_time']):>7}"
              f"{(_fmt_jsd(r['mean_hit_rate'])):>7}{r['mean_parse_fail']:>7.2f}")
    print(f"\n[aggregate] wrote {RUNS_DIR / 'aggregate.json'}")
    return 0


def cmd_list(args) -> int:
    factorial = set(_factorial_personas())
    print("personas:")
    for p in list_personas():
        tag = "  [factorial]" if p in factorial else ""
        print(f"  {p}{tag}")
    print("events:")
    for e in list_events():
        sax = load_event(e).get("sensitive_axis")
        tag = f"  [sensitive_axis={sax}]" if sax is not None else ""
        print(f"  {e}{tag}")
    print("agents:")
    for a in AGENT_REGISTRY:
        print(f"  {a}")
    print("model-keys:")
    try:
        from .registry import list_model_keys
        for k in list_model_keys():
            print(f"  {k}")
    except Exception as exc:  # pragma: no cover
        print(f"  (could not read models.yaml: {exc})")
    return 0


# --------------------------------------------------------------------------- #
# Argparse
# --------------------------------------------------------------------------- #


def _add_llm_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model-key", default=None,
                   help="Registry key from models.yaml (preferred). Overrides --provider/--model.")
    p.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "anthropic"),
                   choices=["anthropic", "openai", "openai_compatible"])
    p.add_argument("--model", default=None,
                   help="Model id; falls back to config.DEFAULT_MODELS[provider].")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--use-recommended", action="store_true",
                   help="Use the model's vendor-recommended sampling (sensitivity checks only).")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sandbox.run")
    sub = parser.add_subparsers(dest="cmd")

    p_single = sub.add_parser("single", help="Run one agent × persona × event episode.")
    p_single.add_argument("--persona", required=True)
    p_single.add_argument("--event", required=True)
    p_single.add_argument("--agent", default="full", choices=list(AGENT_REGISTRY))
    p_single.add_argument("--max-turns", type=int, default=10)
    p_single.add_argument("--evaluate", action="store_true",
                          help="Run rule-based + LLM judge evaluations after the episode.")
    _add_llm_flags(p_single)
    p_single.set_defaults(func=cmd_single)

    p_matrix = sub.add_parser("matrix",
                              help="Run agent × all personas × one event + swap test.")
    p_matrix.add_argument("--event", required=True)
    p_matrix.add_argument("--agent", default="full", choices=list(AGENT_REGISTRY))
    p_matrix.add_argument("--personas", nargs="*", default=None,
                          help="Optional subset of persona ids. Defaults to all.")
    p_matrix.add_argument("--max-turns", type=int, default=10)
    p_matrix.add_argument("--swap-trials", type=int, default=1)
    _add_llm_flags(p_matrix)
    p_matrix.set_defaults(func=cmd_matrix)

    p_study = sub.add_parser("study",
                             help="Persona-measurement grid for one model (the new default).")
    p_study.add_argument("--personas", nargs="*", default=None,
                         help="Defaults to the factorial personas (those with an 'axes' tag).")
    p_study.add_argument("--events", nargs="*", default=None,
                         help="Defaults to events with a 'sensitive_axis' tag.")
    p_study.add_argument("--agents", nargs="*", default=None,
                         help=f"Defaults to {STUDY_AGENTS} (desire on/off).")
    p_study.add_argument("--seeds", type=int, default=3)
    p_study.add_argument("--max-turns", type=int, default=12)
    p_study.add_argument("--concurrency", type=int, default=16,
                         help="Episodes run in parallel (vLLM batches the requests). 1 = sequential.")
    p_study.add_argument("--judge", action="store_true",
                         help="Also run the LLM persona-consistency judge (slow).")
    p_study.add_argument("--swap-trials", type=int, default=0,
                         help=">0 also runs the per-event persona-swap test.")
    _add_llm_flags(p_study)
    p_study.set_defaults(func=cmd_study)

    p_agg = sub.add_parser("aggregate", help="Roll up all study runs into a model × axis table.")
    p_agg.set_defaults(func=cmd_aggregate)

    p_list = sub.add_parser("list", help="List personas / events / agents / model-keys.")
    p_list.set_defaults(func=cmd_list)

    # Back-compat: bare flags (no subcommand) -> treat as "single".
    args, unknown = parser.parse_known_args(argv)
    if args.cmd is None:
        argv2 = ["single"] + (argv if argv is not None else sys.argv[1:])
        args = parser.parse_args(argv2)

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
