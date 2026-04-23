"""CLI entry point for the sandbox.

Typical use:

    # Single run
    python -m sandbox.run \
        --persona persona_02_tired_achiever \
        --event   event_01_deadline_compression \
        --agent   full \
        --provider anthropic \
        --max-turns 10

    # Run the evaluation matrix for one event (all 3 personas × full agent)
    # and produce the persona-swap test + LLM judge scores.
    python -m sandbox.run matrix --event event_01_deadline_compression
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
)


RUNS_DIR = Path(__file__).resolve().parent / "runs"


def _build_llm_config(args) -> LLMConfig:
    return LLMConfig(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )


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

    out_dir = RUNS_DIR / f"{args.agent}__{persona['id']}__{event['id']}__{cfg.model}"
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
    for pid in persona_ids:
        persona = load_persona(pid)
        agent_step = agent_step_factory(cfg)
        out_dir = RUNS_DIR / f"matrix__{args.agent}__{event['id']}__{cfg.model}" / pid
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

    # Swap test across all persona pairs.
    swap = run_swap_test(swap_input, cfg, trials_per_pair=args.swap_trials)

    summary_path = RUNS_DIR / f"matrix__{args.agent}__{event['id']}__{cfg.model}" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "event_id": event["id"],
        "agent": args.agent,
        "per_persona": results,
        "swap_test": {
            "n_matches": swap["n_matches"],
            "accuracy": swap["accuracy"],
        },
    }
    summary_path.write_text(json.dumps({**summary, "swap_detail": swap["matches"]},
                                       ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_list(args) -> int:
    print("personas:")
    for p in list_personas():
        print(f"  {p}")
    print("events:")
    for e in list_events():
        print(f"  {e}")
    print("agents:")
    for a in AGENT_REGISTRY:
        print(f"  {a}")
    return 0


# --------------------------------------------------------------------------- #
# Argparse
# --------------------------------------------------------------------------- #


def _add_llm_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "anthropic"),
                   choices=["anthropic", "openai"])
    p.add_argument("--model", default=None,
                   help="Model id; falls back to config.DEFAULT_MODELS[provider].")
    p.add_argument("--temperature", type=float, default=0.7)


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

    p_list = sub.add_parser("list", help="List available personas / events / agents.")
    p_list.set_defaults(func=cmd_list)

    # Back-compat: allow bare flags (no subcommand) -> treat as "single".
    args, unknown = parser.parse_known_args(argv)
    if args.cmd is None:
        # Re-parse as "single" with the original argv.
        argv2 = ["single"] + (argv if argv is not None else sys.argv[1:])
        args = parser.parse_args(argv2)

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
