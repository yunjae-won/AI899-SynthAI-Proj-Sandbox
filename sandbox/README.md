# Synthetic Person Sandbox

This is the runnable sandbox for the **Synthetic Person** project.

> **Re-scope (current focus): a persona *measurement* study.** Earlier the
> question was whether persona-conditioned cognitive scaffolding helps agents
> *perform* better. The project has shifted to two measurement questions —
> **(Q1) can models actually embody a persona, in the intended direction?** and
> **(Q2) how does behaviour change across personas, and is that change caused by
> persona rather than noise?** See [**Measurement study**](#measurement-study-current-focus)
> below. The original engine/agents/evaluators are unchanged and still documented.

The sandbox is implemented in **LangGraph**. Stimuli come from two merged
roots: the upstream [`../src/`](../src) library (a teammate's repo, left
untouched) and a sandbox-local `library/` authored in this repo — so we can
grow the persona/event set without modifying the submodule.

## What's inside

```
sandbox/
├── README.md            ← you are here
├── requirements.txt
├── config.py            ← LLM provider selection (anthropic | openai | openai_compatible)
├── models.yaml          ← model registry: map a --model-key to a provider/model
├── registry.py          ← resolve a model-key → LLMConfig
├── run.py               ← CLI: `single`, `matrix`, `study`, `aggregate`, `list`
├── world/
│   ├── state.py         ← TypedDict schemas (SimState, AgentState, ...)
│   ├── loader.py        ← load personas/events (merges ../src/ + ./library/)
│   └── engine.py        ← turn-based simulator with event-queue injection
├── library/             ← sandbox-local stimuli (this repo)
│   ├── personas/        ← 8 controlled-factorial personas (value × conflict × time)
│   └── events/          ← 3 axis-sensitive events + 1 orthogonal control
├── agents/
│   ├── full_agent.py    ← LangGraph graph: reflect → desires → goals → action
│   ├── baselines.py     ← prompt_only / memory_only / no_desire baselines
│   ├── prompts.py       ← every LLM prompt template lives here
│   └── _llm_utils.py    ← robust JSON parsing, single-shot call helper
├── evaluation/
│   ├── measurement.py   ← PRIMARY: action-JSD, diagnostic hit-rate, parse-fail, axis×event matrix
│   ├── rule_based.py    ← deterministic 3-axis scoring (hard_loss / context / efficiency)
│   ├── judge.py         ← LLM-as-judge: persona consistency + adaptation (secondary)
│   └── swap_test.py     ← pairwise persona-swap consistency test (secondary)
└── runs/                ← trajectory + evaluation/measurement JSON outputs

../scripts/serve_vllm.sh ← serve a HF model with vLLM (OpenAI-compatible) for local Qwen
../run_study.sh          ← sweep the study across model-keys, then aggregate
```

## Measurement study (current focus)

### Personas — a controlled 2×2×2 factorial (`library/personas/`)

Eight personas span three **simple binary axes**; only the axis-relevant fields
vary, everything else (baseline state, relationships) is held constant, so a
behavioural difference is attributable to an axis rather than a confound. Each
persona carries an `axes` tag:

| Axis | Poles | Fields it sets |
|---|---|---|
| **value** | `achievement` ↔ `affiliation` | `value_priorities` (academic vs relationships) |
| **conflict** | `avoidant` ↔ `assertive` | `communication_style` (conflict_style, assertiveness) |
| **time** | `impulsive` ↔ `deliberate` | `decision_tendencies` (planning_horizon, procrastination, sleep_discipline) |

Names: `persona_{ach|aff}_{avo|asr}_{imp|del}`.

### Events — axis-sensitive + an orthogonal control (`library/events/`)

Each event carries `sensitive_axis` and pre-registered `diagnostic_actions`
(the expected action per pole, for **directional fidelity**):

| Event | sensitive_axis | Tension |
|---|---|---|
| `event_value_clash` | value | study block vs spontaneous lunch |
| `event_boundary_push` | conflict | a teammate dumps work on you last-minute |
| `event_now_vs_later` | time | invest early vs take the easy/now option |
| `event_orthogonal` | none | one obviously-correct urgent submission (control floor) |

An axis-A-sensitive event is automatically the **negative control** for axes B
and C. The headline artifact is the **axis × event divergence matrix**: the
diagonal (axis on its sensitive event) should dominate; off-diagonal and the
orthogonal event should sit near the floor. That contrast is both the
persona-effect estimate *and* the metric's discriminant-validity check.

### Metrics (`evaluation/measurement.py`, deterministic, no LLM)

- **action-distribution JS divergence** between persona poles → the matrix.
- **`persona_effect`** = `divergence(sensitive) − divergence(orthogonal)` per axis.
- **`diagnostic_hit_rate`** — did the agent take its pole's pre-registered action? (Q1 direction)
- **`parse_fail_rate`** — fraction of turns the model failed to emit JSON (a confound covariate).

The LLM judge and swap test still exist but are now *secondary* (`--judge`, `--swap-trials`).

### Models — registry + local Qwen via vLLM

Models are **data**, not code: add one entry to [`models.yaml`](models.yaml) and
reference it with `--model-key`. Providers: `openai`, `anthropic`, and
`openai_compatible` (any OpenAI-compatible endpoint — vLLM-served Qwen, local
servers, proxies). For fair comparison the study **forces one temperature** on
every model; each vendor's `recommended` sampling is recorded but unused unless
you pass `--use-recommended`.

The default registry includes hosted size ladders (`gpt-nano`/`gpt-mini`,
`claude-haiku`/`claude-sonnet`) and the dense **Qwen3.5** ladder
(`qwen3.5-4b` / `-9b` / `-27b`; HF, Feb 2026 — see
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B),
[Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B),
[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)).

Serve a Qwen model (separate GPU env; vLLM is **not** a sandbox dependency):

```bash
pip install -r ../requirements-serve.txt           # in a separate venv
../scripts/serve_vllm.sh Qwen/Qwen3.5-9B 8001      # port matches models.yaml
```

### Running the study

```bash
pip install -r sandbox/requirements.txt
export OPENAI_API_KEY=...        # and/or ANTHROPIC_API_KEY, or serve Qwen above

# one model: factorial personas × 4 events × {full, no_desire} × seeds
python -m sandbox.run study --model-key qwen3.5-9b --seeds 3

# sweep several models, then build the model × axis table
SEEDS=5 ../run_study.sh qwen3.5-4b qwen3.5-9b gpt-mini claude-sonnet
python -m sandbox.run aggregate
```

Outputs:

```
runs/study__<model_key>/
  <agent>/<persona_id>__<event_id>/seed<n>/trajectory.json
  measurement.json        ← axis×event matrix, persona_effect, hit-rate, parse-fail
runs/aggregate.json       ← model × axis contrasts across all study runs
```

**Validation gate:** on the pilot, confirm the matrix diagonal dominates and the
orthogonal event is at the floor *before* scaling to API models. If it doesn't,
the events need fixing, not the models.

## The question the sandbox is built to answer

Given an abstract goal and a persona, an agent takes turns. Each turn it
reads new observations, reflects on them, updates internal desires, revises
its goal list, and picks one action. The world engine applies the action,
advances time, and fires any scheduled events from the event's
`events_queue`. We record the full `(timestamp, state, action, observation)`
trajectory.

We then ask three evaluation questions about that trajectory:

1. **Personality fidelity** — does the sequence of goals and actions look
   like the persona we configured? We score this two ways: an LLM judge
   rates five dimensions (value alignment, communication style, decision
   tendencies, goal coherence, weakness expression), and a persona-swap
   test asks a blind judge to pick the true persona given only the
   trajectory. If the same agent's trajectories all look identical
   regardless of persona, the swap test's accuracy collapses to chance.
2. **Adaptation under constraint** — when a mid-episode event changes the
   situation (a teammate messages at 19:00, outlets are occupied, a quiz
   is posted), does the agent revise its goals in a way that's both
   proportionate *and* in-character? An LLM judge compares the pre-event
   and post-event trajectory slices.
3. **Original three-axis scoring** — the hard-loss / context-switching /
   efficiency rubric from `../src/analysis/analysis.md`, computed from
   trajectory heuristics plus the event's declared
   `hard_loss_conditions` / `context_switching_indicators` /
   `efficiency_indicators`.

The first two are what's new here; the third is retained so everything the
analysis team already designed still works.

## Running the sandbox

Install once:

```bash
pip install -r sandbox/requirements.txt
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY
```

List what's available:

```bash
python -m sandbox.run list
```

Run one episode with evaluation (single agent × persona × event):

```bash
python -m sandbox.run single \
    --agent full \
    --persona persona_02_tired_achiever \
    --event event_01_deadline_compression \
    --max-turns 10 \
    --evaluate
```

Outputs go to `sandbox/runs/full__persona_02_tired_achiever__event_01_deadline_compression/`:
the full trajectory and, if `--evaluate` was passed, the combined
evaluation JSON (three-axis + persona-consistency judge).

Run the full evaluation matrix for one event (all personas × one agent)
with the persona-swap consistency test:

```bash
python -m sandbox.run matrix \
    --agent full \
    --event event_01_deadline_compression \
    --swap-trials 2
```

Switch providers per invocation:

```bash
python -m sandbox.run single \
    --provider openai --model gpt-4o-mini \
    --persona persona_01_balanced_sophomore --event event_02_campus_routine \
    --agent full --evaluate
```

## The four-module agent

The full agent is a `StateGraph` with one linear sweep per turn:

**perceive → reflect → update_desires → generate_goals → plan_action**

Each node reads a slice of `SimState`, runs one LLM call with a persona
system prompt, and writes back into the agent's cognitive fields. The
state carried across turns is small: `memory` (recent reflections,
bounded to 15 entries), `desires` (a handful of short strings), and
`goals` (an ordered list with priority/status/rationale). `last_reasoning`
holds the free-form trace of the last action, which the evaluators read.

The prompts are intentionally short. The goal is that when you look at a
turn's reasoning you can tell which module produced it, and the persona
shines through rather than being buried in instruction.

## Baselines

Three reactive variants share the same `agent_step(sim_state) → dict`
interface and plug into the same engine, so swapping `--agent full` for
`--agent prompt_only` is the only change:

- `prompt_only` — one LLM call per turn: observe, then act. No memory,
  no desires, no goals. This is the lower bound.
- `memory_only` — reflects each turn and reads its own memory when acting,
  but never forms explicit desires or goals.
- `no_desire` — has memory *and* goals, but goals are derived directly
  from pending tasks instead of from introspected desires. Useful for
  isolating whether the desire module is doing anything, versus just
  giving the agent a scratchpad.

Ablating each module lets us ask questions like *"does removing desires
collapse persona fidelity?"* or *"does memory alone buy us adaptation?"*
— answers go in the paper.

## Evaluation design

**Rule-based (cheap, deterministic).** `evaluation/rule_based.py`
computes the three axes from the trajectory and the event's
`evaluation` block. Hard-loss checks look at final task progress and
trust deltas; context-switching counts turns whose `reasoning` mentions
multiple domains; efficiency matches salient tokens from each indicator
against the full action/reasoning text. The verdict follows the project
brief's pass/fail rule: any hard loss → fail; no hard loss and both
context ≥ 0.5 and efficiency ≥ 0.5 → excellent; either → pass.

**Persona-consistency judge.** `evaluation/judge.py::judge_persona_consistency`
shows the judge the full persona spec plus a compact trajectory view
(goals, action, reasoning, observation per turn) and scores five
dimensions 1–5. The `weakness_expression` dimension is there on purpose:
a persona that *never* exhibits its known weakness is a sign the agent
flattened into a generic helpful assistant.

**Adaptation judge.** `judge_adaptation(trajectory, split_turn, injected_event)`
splits the trajectory at the turn when a new event fires and asks the
judge whether the agent (a) acknowledged the event, (b) revised goals,
(c) stayed in character, and (d) reacted proportionately. Use this when
you inject a perturbation mid-run; `events_queue` in each event JSON is
already the perturbation schedule.

**Swap test.** `evaluation/swap_test.py::run_swap_test` takes N runs
`[(persona_1, trajectory_1), (persona_2, trajectory_2), ...]` and, for
every ordered pair (i, j) with i ≠ j, asks a blind judge: "was this
trajectory produced by persona i or persona j?" Position is randomized
per trial to prevent A/B bias. The reported `accuracy` is the fraction
of pairs the judge got right. High accuracy means the personas produce
distinguishable behavior; chance-level means the agent is collapsing
personas.

Together these give a per-run score (three-axis + consistency judge),
a per-event score (adaptation judge on trajectory segments), and a
per-agent-over-personas score (swap-test accuracy). That's the full
evaluation fabric.

## The trajectory format

Every run produces `runs/<tag>/trajectory.json` with this shape:

```jsonc
{
  "persona_id": "...",
  "event_id": "...",
  "final_state": { "current_time": "...", "tasks": [...], "relationships": [...], "agent": {...} },
  "trajectory": [
    {
      "turn": 0,
      "timestamp": "Wed 14:00",
      "state_snapshot": { "agent": {...}, "tasks": [...], "relationships": [...] },
      "action": { "name": "work_on", "args": {...}, "reasoning": "..." },
      "observation": "Worked on Assignment A for 2h; progress=0.65."
    }
  ]
}
```

This is the format the evaluators consume. Any new evaluator you write
should take `(trajectory, event)` and return a JSON-serializable result.

## Extending the sandbox

**Add a new persona or event.** Drop a matching `.json` + `.md` pair into
`../src/personas/` or `../src/events/` using the existing schemas. The
loader picks them up automatically — no sandbox changes needed.

**Add a new agent.** Implement a `build_<name>_agent(llm_config) →
step(sim_state) → dict` factory (the dict must at least contain `action`;
optional `goals`, `desires`, `memory`, `last_reasoning` are merged back
into agent state). Register it in `sandbox/agents/__init__.py::AGENT_REGISTRY`.
The engine doesn't care whether the agent is a LangGraph graph, a single
call, or a hand-coded policy.

**Add a new action primitive.** Extend `_apply_action` in
`world/engine.py` to handle the new name and whatever effect it should
have on state. Add it to the event's `available_actions` block so agents
know it exists. Keep effects small and explainable; the world engine is
not meant to be a physics simulator.

**Add a new evaluator.** Add a module under `evaluation/`, export a
function from `evaluation/__init__.py` (lazily — so rule-based scoring
still works without LLM deps installed), and call it from `run.py`.

## Design notes for reviewers

Why TypedDict instead of Pydantic for `SimState`: LangGraph's `StateGraph`
merges dict updates natively and we lean heavily on that. Pydantic
validation would force converting back-and-forth every node and the
runtime guarantees aren't worth the friction at this scale.

Why the world engine is simple: the project's research question is
*goal evolution and persona fidelity*, not realistic physics. A richer
world (energy math, NPC probability models) is easy to add later, but
putting it in v1 would invite over-fitting the agent to the engine
instead of to the persona.

Why prompts in a single file: every prompt is under 40 lines; keeping
them together makes the "what does the agent actually see?" question
answerable by reading one file.

Why lazy imports in `sandbox/agents/__init__.py` and
`sandbox/evaluation/__init__.py`: someone grading rule-based runs
shouldn't have to install langgraph, and someone iterating on agents
shouldn't have to install the judge's dependencies. Every top-level
import is cheap.
