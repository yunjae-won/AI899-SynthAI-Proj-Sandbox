# Midterm Report Plan — Synthetic Person

This document is the writing and experiment plan for the midterm report.
Before we discuss sections, we need a single sharp narrative, because every
section should be serving the same argument.

## 0. Project narrative (the one-sentence version)

> **Persona-grounded agents face a tradeoff between *persona fidelity* and
> *adaptive intelligence*, and the four-module topology
> (memory → desires → goals → action) is our attempt to navigate that
> tradeoff.**

Read this aloud before every writing session. If a sentence in the report
doesn't serve this argument, cut it.

Why this framing works for a course project:

- It's **falsifiable**: we can show the tradeoff empirically by putting
  reactive and fully-modular agents on the same persona × event grid and
  reading off two numbers (persona-consistency judge, hard-loss rate).
- It's **novel enough**: "persona fidelity vs. adaptive planning" hasn't
  been systematically studied. Generative Agents (Park et al. 2023) show
  personas *emerge*; AgentBench-style work shows agents *plan*. Nobody
  has pitted these two success criteria against each other.
- It's **bounded**: we don't need to build a realistic human. We need
  three personas, two events, and four agent variants — all of which
  exist. Scope is finite.

The rest of this plan is how each report section serves that narrative.

## 1. Introduction (≈ 0.5–1 page)

**Hook.** Most LLM agents wait for instructions. Humans don't — they hold
goals under uncertainty, revise them when the world pushes back, and stay
recognizably themselves across the process. The course project asks
whether LLM agents can do the same, and what it costs them when they
try.

**Problem.** If an agent's values come from a persona, and the persona
has known weaknesses (a tired over-achiever who undervalues sleep),
then "behave like this person" and "avoid irreversible losses" pull in
opposite directions. An agent that follows the persona commits its
mistakes; an agent that overrides the persona stops being that person.
This is the central tension we want to study.

**Research question.** *Can a four-module agent topology (memory,
desires, goals, action) produce trajectories that are simultaneously
(a) recognizably in-character, (b) adaptive when constraints shift, and
(c) defensive against irreversible losses? How do each of the three
properties trade off when modules are ablated?*

**Contributions.** The midterm report claims four:

1. A runnable sandbox (LangGraph) with a clean persona/event schema and
   plug-in agent interface. 3 personas, 2 events, 4 agent variants.
2. An evaluation suite with three independent lenses: rule-based
   three-axis scoring, LLM-as-judge persona consistency, and a
   cross-persona swap test for distinguishability.
3. An ablation study isolating the contribution of each module
   (memory, desires, goals) to each evaluation lens.
4. Preliminary qualitative findings from the persona × event × agent
   matrix that motivate the final-report experiments.

**Keep it short.** The introduction shouldn't redo the methodology.
One page, then move on.

## 2. Related Work (≈ 1 page)

Group the citations into three buckets. Don't write a list; write three
paragraphs, one per bucket, each ending with how our project is
different.

**Agentic frameworks and memory.** ReAct (Yao et al. 2023) showed
reasoning + acting loops. Reflexion (Shinn et al. 2023) added a verbal
critique-and-retry. Voyager (Wang et al. 2023) is the long-horizon,
open-ended pick. Generative Agents (Park et al. 2023) is the closest
relative — a town of LLM characters with memory streams and reflection.
*Difference:* Generative Agents optimizes for emergent social behavior;
we optimize for evaluable single-agent goal evolution, and we actually
grade persona fidelity with a judge.

**Persona / role simulation.** Character.ai-style role-play, PersonaChat
(Zhang et al. 2018), role-aware instruction tuning, recent work on
persona drift in multi-turn dialogue. *Difference:* these measure
persona in linguistic surface features (word choice, tone). We measure
it in *decision-making surface* — goals, priority orderings, tradeoffs.

**LLM-as-judge and long-horizon evaluation.** JudgeLM, MT-Bench,
AgentBench, WebArena, SWE-bench. *Difference:* these judge task
completion. We judge character-consistency, which is novel at the
benchmark level.

Cite 8–12 papers total. Don't name-drop — each citation should do
work in one of those "Difference" sentences.

## 3. Background / Motivation (≈ 0.5–1 page)

Lay the two conceptual foundations the reader needs before the
methodology hits:

**Hard vs. soft losses.** Borrowed from the original analysis doc
(`src/analysis/analysis.md`): some costs are irreversible (a missed
exam, a broken trust) and some are negotiable (a rescheduled meeting,
a skipped workout). Humans navigate life by ruthlessly protecting the
hard losses and spending the soft ones. An agent that treats them as
interchangeable will either pay hard losses that should have been
defended, or over-protect everything and freeze. We evaluate agents in
part on this distinction.

**Personas as distributions over failure modes, not identities.** A
persona is not a writing style. In our formulation it's a vector over
value priorities, decision tendencies, and relationship trust — which
induces a *predictable set of failure modes* under specific event
pressures. Tired Achiever + deadline compression predictably pulls
toward all-nighter-then-quiz-fails. Social Explorer + free period
predictably pulls toward over-joining. A competent agent needs to
*feel* the persona's pull and still protect the hard loss. That's the
tradeoff.

One good diagram here (figure 1): the persona × event matrix, with
cells color-coded for expected failure mode, copied from analysis.md.

## 4. Methodology (≈ 1.5–2 pages)

Organize this section into four subsections.

**4.1 Sandbox and schema.** Briefly describe the `src/` design: a
persona JSON with value_priorities, communication_style,
decision_tendencies, and baseline_state; an event JSON with
initial_state, visible_information, hidden_state, events_queue, and an
evaluation block. One sentence on the `.md` + `.json` pairing
discipline. One paragraph on the world engine — turn-based, advances
time on each action, fires `events_queue` items when their timestamp
is due, logs `(timestamp, state, action, observation)` trajectories.
Do not reproduce the full schema in the body; put one trimmed JSON
example in an appendix.

**4.2 Four-module agent (figure 2).** Diagram + prose for
`perceive → reflect → update_desires → generate_goals → plan_action`.
Three points to hit:

- Each node is one LLM call with the persona as system prompt, so the
  persona is in-the-loop at every decision point (not just once at
  initialization).
- Memory is bounded (15 most recent reflections) — not because that's
  optimal but because it's a pragmatic choice that our evaluation is
  blind to.
- Goals are allowed to change every turn; this is deliberate, because
  revising goals under new information is exactly what we want to
  observe.

**4.3 Baselines (ablation design).** Motivate the three baselines as
clean module-removals:

| Variant        | Memory | Desires | Goals | Interpretation                  |
|----------------|--------|---------|-------|---------------------------------|
| `full`         | yes    | yes     | yes   | Reference topology              |
| `no_desire`    | yes    | —       | yes   | Goals from tasks, not desires   |
| `memory_only`  | yes    | —       | —     | Reflective but not goal-driven  |
| `prompt_only`  | —      | —       | —     | Purely reactive lower bound     |

This is not four arbitrary agents — it's a clean subtractive ablation.
State this explicitly.

**4.4 Evaluation design.** Four metrics, each with a one-paragraph
rationale:

- **Three-axis rule-based score** (hard_loss, context_switching,
  efficiency) — cheap, deterministic, grounded in the event's own
  `evaluation` block.
- **Persona-consistency judge** — LLM judge rates five 1–5 dimensions
  (value alignment, communication style, decision tendencies, goal
  coherence, weakness expression). `weakness_expression` is there on
  purpose: a Tired Achiever who never behaves like one has been
  flattened into a generic planner, and we want that to *lower* the
  score, not raise it.
- **Adaptation judge** — compares pre- and post-perturbation slices of
  a trajectory. The perturbation is the event's own `events_queue`
  item (teammate message, outlet availability, etc.).
- **Persona swap test** — blind pairwise judge: "was this trajectory
  produced by persona A or persona B?" Accuracy above chance ⇒
  trajectories are persona-distinguishable. Chance-level ⇒ persona
  has collapsed.

State the three hypotheses that the experiments will test:

> **H1.** Removing the desire module lowers persona-consistency more
> than it lowers three-axis verdict — i.e., you can still plan without
> desires, but you stop being a person.

> **H2.** Removing memory hurts adaptation judgments more than any
> other metric — reactive agents fail specifically when the world
> changes mid-episode.

> **H3.** Swap-test accuracy correlates positively with
> persona-consistency judge score. If it doesn't, one of the two
> metrics is broken.

## 5. Progress (≈ 1–1.5 pages)

This is the honest section. Write it last, but plan it early so you
know what to run.

**5.1 Implementation complete.** List with intentionally small claims:

- Sandbox implemented in LangGraph (`sandbox/`), 4 agent variants, 4
  evaluators, CLI with single-run and matrix-run modes, provider-
  configurable (Anthropic, OpenAI).
- 3 personas and 2 events ingested from the design team's `src/`
  without modification.
- Rule-based evaluator reproduces the pass/fail verdict from the
  original analysis document (verified by running hand-crafted
  "good run" and "bad run" trajectories end-to-end).

**5.2 Preliminary experiments (see section 6 below for the plan).**
Show one table and one qualitative example.

- **Table 1**: The 3 × 2 × 4 = 24 runs of the main matrix, one row per
  run, columns for three-axis verdict, consistency score (overall),
  hard_loss triggered (y/n), swap-test contribution when available.
  If budget only allows 1 seed, say so.
- **Figure 3 (qualitative)**: Pick the single most illustrative
  trajectory — probably Tired Achiever × event_01 × `full` vs.
  `prompt_only`. Show the first 4 turns of each side-by-side, with
  reasoning traces. This is where the reader's understanding clicks.

**5.3 Known gaps / planned for final report.**

- **More events.** `src/analysis/analysis.md` already identifies four
  uncovered axes (information_asymmetry, value_conflict,
  long_vs_short_term, recovery_after_mistake). We plan to add one of
  each by the final report.
- **Seed variance.** Midterm runs use a single seed per cell; final
  report runs 3 seeds and reports mean ± std on all four metrics.
- **Persona drift metric.** A proposed addition: run the
  consistency judge separately on the first-half and second-half of a
  trajectory. If consistency degrades, the agent is drifting. Easy to
  add (one file in `evaluation/`) and publishable.
- **Human upper bound (stretch).** Have two team members play the
  same episode manually, run all four evaluators on their trajectories.
  Shows what "excellent" looks like.

## 6. Experiment plan (for the team, not the report)

This is a separate deliverable, not part of the write-up. It's the
set of runs that produces every number in the report.

### 6.1 Main matrix (required for midterm)

```
personas (3) × events (2) × agents (4) × seeds (1) = 24 runs
```

Each run is 10 turns. With the four-module agent, that's
~4 LLM calls/turn × 10 turns = 40 calls/run × 24 runs ≈ 1000 LLM calls
for the main agent. Baselines are cheaper (1–3 calls/turn). Rough
total: 1500–2000 LLM calls. On Claude Sonnet 4.5 this is well under
$20; on GPT-4o-mini it's under $5. The matrix should finish in under
an hour wall-clock.

Use the existing CLI:

```bash
for agent in full no_desire memory_only prompt_only; do
  for event in event_01_deadline_compression event_02_campus_routine; do
    python -m sandbox.run matrix \
      --agent "$agent" --event "$event" \
      --max-turns 10 --swap-trials 2
  done
done
```

This writes to `sandbox/runs/matrix__<agent>__<event>/` with one
`summary.json` per cell. The per-persona three-axis verdict,
consistency score, and swap-test accuracy all live there.

### 6.2 Adaptation experiment (required for midterm)

For every run from 6.1, find the turn index when the first
`events_queue` item fired (e.g., Wed 16:30 team_chat in event_01) and
run `judge_adaptation` with that as `split_turn`. Save results into
the same run directory.

This isn't automated in the CLI yet — one small script (~30 lines)
loads the trajectory, finds the split turn, calls the judge. Put it
in `sandbox/scripts/run_adaptation.py` as a team follow-up task.

### 6.3 Aggregate reporting (required for midterm)

One post-processing script that reads `sandbox/runs/matrix__*/` and
produces:

- Table 1: the 24-row main table (described above).
- Table 2: 4×4 metric correlation matrix
  (three-axis verdict, consistency, adaptation, swap accuracy)
  across all 24 runs.
- Figure 4: bar chart, one bar per agent variant, height = persona
  consistency, split by hatching into risk-pair (Tired × event_01,
  Social × event_02) vs. safe pairings.

All three artifacts plug into the report.

### 6.4 Qualitative deep-dive (required for midterm)

Pick **exactly one** (agent_variant × persona × event) cell whose
trajectory best illustrates the persona-fidelity tradeoff. Our current
guess: Tired Achiever × event_01 × `full` vs. `prompt_only`. Screenshot
the first 4 turns of each. Annotate 2–3 sentences per turn in the
report describing *what the persona was pulling the agent toward* and
*what the agent did instead.*

Don't try to include more than one deep-dive in the midterm. One
well-annotated example beats three rushed ones.

### 6.5 Deferred to final report

- Adding 1–2 new events covering the uncovered axes
  (`information_asymmetry`, `value_conflict`).
- Seed variance (3 seeds → mean ± std).
- Persona-drift metric (new evaluator).
- Human upper-bound baseline (stretch).
- Cross-provider comparison (Sonnet vs. GPT-4o) — cheap to run, useful
  to include for robustness.

## 7. Timeline (4 weeks until midterm, adjust to your course)

| Week | Deliverable                                                    |
|------|----------------------------------------------------------------|
| 1    | Main matrix run (6.1) completes. Raw JSON in `sandbox/runs/`.  |
| 1    | Introduction + Background drafted (can be written from plan)   |
| 2    | Adaptation script (6.2) and aggregate script (6.3) done.       |
| 2    | Methodology section drafted with final figures.                |
| 3    | Qualitative deep-dive (6.4) written. Related Work completed.   |
| 3    | Progress section drafted with real numbers.                    |
| 4    | Integration pass, figure polish, references, final proofread.  |

This assumes 3–4 people. Parallelize: one person runs the experiment
loop while another drafts intro + related work; a third writes
methodology; a fourth does the qualitative deep-dive. Integration
and polish are single-threaded, budget the last week for them.

## 8. Writing style recommendations

A few stylistic choices that will help the report read like a paper
instead of a project dump:

- **No bullet lists in the report body**, except for tables of
  variants / results. Prose forces you to connect the argument.
  (This document is a plan, so it allows itself bullets; the report
  should not.)
- **Each figure must have a caption that states the finding**, not
  just what's being shown. "Figure 3. Persona-consistency scores
  collapse when the desire module is removed" is a caption. "Figure 3.
  Persona-consistency scores by agent variant" is not.
- **Results section should start with the strongest single claim**,
  stated in one sentence, before any numbers. Then the numbers
  support the claim. Reversing the order buries the lede.
- **Negative results are fine and interesting**. If H3 (swap accuracy
  ↔ consistency correlation) doesn't hold, that's a finding. Don't
  hide it.
- **Appendices are cheap**. Put the full prompt templates, the 24-row
  table, the full persona JSONs there. The body stays lean.

## 9. Risks and mitigations

Three things most likely to derail the midterm, and what to do about
each if they happen:

- **Judge variance.** The consistency judge will sometimes disagree
  with itself on reruns. Mitigation: run each judge call twice, report
  mean. If variance is high, that itself is a finding worth noting.
- **Agents collapse to generic planners in practice.** The full agent
  might produce trajectories nearly identical to `prompt_only` because
  the LLM over-weights the task description. Mitigation: strengthen
  the persona system prompt (already a file: `agents/prompts.py`); if
  that doesn't help, reporting the collapse *is* the finding.
- **Rule-based hard-loss thresholds are too strict.** Early runs
  showed even "good" runs tripping `assignment_A < 0.7` because the
  world engine's `work_on` gain per hour is small. Mitigation: tune
  the gain in `world/engine.py::_apply_action` so a reasonable
  trajectory passes, or relax the threshold. Either way, do it once
  and freeze.

## 10. What to push on the sandbox before running the matrix

Small fixes that would pay back in report quality:

1. Add a `persona_drift.py` evaluator (50 lines) that splits a
   trajectory in half and reports consistency-judge scores on each
   half. This gives you a figure for free.
2. Add a small `scripts/aggregate.py` that reads
   `runs/matrix__*/summary.json` and produces the Table 1 CSV. Saves
   a full day of hand-aggregation at report time.
3. Sanity-check `world/engine.py::_apply_action`: can a reasonable
   hand-coded trajectory pass all three hard-loss checks on both
   events? If not, adjust the gains (or relax the threshold) before
   running the full matrix.

Do these three before the run in 6.1 and you won't regret it.

---

If this framing lands, the next concrete step is: decide as a team
whether H1/H2/H3 are the right three hypotheses, then freeze the
experiment plan in 6.1 and kick off the run. Everything else in the
report can be written in parallel from this document.
