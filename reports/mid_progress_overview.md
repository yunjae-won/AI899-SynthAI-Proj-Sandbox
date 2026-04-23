# Midterm Report — Synthetic-Person Sandbox: Goal-Evolving LLM Agents Under Persona Pressure

## 1. Motivation & Research Question

Most LLM “agent” benchmarks judge whether a model can follow a given instruction. Humans do not live under instructions — they *form their own goals*, revise them when circumstances change, and stay recognizably themselves while doing it. A synthetic person is only useful if it does the same.

We therefore ask three concrete sub-questions:

1. **Personality fidelity.** Do the goals and actions an agent produces over an episode actually look like the persona we configured, or does the agent collapse into a generic “helpful” default?
2. **Adaptation under constraint.** When a mid-episode event changes the situation (a teammate pings, outlets are occupied, a quiz is posted), does the agent revise its goals proportionately *and* in-character?
3. **Module sufficiency.** Which cognitive scaffolding is actually required — memory, desires, explicit goals, or any combination — and when do they start hurting rather than helping?

## 2. Experimental Setup

### 2.1 Stimuli (from `src/`, fixed upstream)

- **Personas (3):**
  - `persona_01_balanced_sophomore` (baseline)
  - `persona_02_tired_achiever` (academic 0.55, sleep_discipline 2, avoidant)
  - `persona_03_social_explorer` (relationships 0.40, accommodating, short planning horizon)

  Designed as *polar opposites* around a neutral baseline.

- **Events (2):**
  - `event_01_deadline_compression`
    - Assignment A is half-done and due Wednesday 23:59
    - 19:00 team meeting
    - Surprise Thursday 9 AM quiz announced mid-episode
    - Hard losses: assignment progress < 0.7, quiz skipped, or teammate/prof trust dropping ≥ 2
  - `event_02_campus_routine`
    - A 3-hour afternoon window with language study, workout, and lunch
    - Interrupted by outlet unavailability and a spontaneous cohort lunch invite
    - Hard losses: all 3 tasks at progress 0, friend trust collapse, or over-joining that causes a later commitment miss

### 2.2 Agents (`sandbox/agents/`)

Four variants share the same `agent_step(sim_state) → dict` interface:

| Agent | Per-turn flow | Ablation purpose |
|---|---|---|
| **full** | reflect → desires → goals → act | all four modules |
| **no_desire** | reflect → goals-from-tasks → act | isolate desire module |
| **memory_only** | reflect → act | isolate goals + desires |
| **prompt_only** | act (reactive) | lower bound |

All four use the same persona system prompt (`agents/prompts.py`) and the same world engine (`world/engine.py`).

The engine is deliberately simple: actions mutate tasks, energy, and stress by fixed rules, and a scheduled `events_queue` delivers perturbations mid-episode.

### 2.3 Models & Matrix

Two OpenAI models were swept: **gpt-5.4-nano** and **gpt-5.4-mini**.

Runs actually stored under `sandbox/runs/`:

- **Model comparison** (full agent, both events): nano vs. mini
- **Ablation** (mini only, both events): prompt_only, memory_only, no_desire, full — 3 personas each

Each matrix run = 3 personas × 10 turns × 2 swap-trials per pair → 12 pairwise swap judgements per matrix.

### 2.4 Evaluators (`sandbox/evaluation/`)

- **Rule-based three-axis** (deterministic, from `analysis/analysis.md`)
  - `hard_loss` — events-declared failure list, pattern-matched on final state
  - `context_switching` — cross-domain reasoning per turn
  - `efficiency` — indicator substring match

  Verdict:
  - `fail` if any hard loss
  - `excellent` if no hard loss + both axes ≥ 0.5
  - `pass` otherwise

- **LLM judge — persona consistency**
  - Scores 1–5 on five axes:
    - `value_alignment`
    - `communication_style`
    - `decision_tendencies`
    - `goal_coherence`
    - `weakness_expression`

  The “weakness” axis is deliberate: a flawless Tired Achiever is *suspicious*.

- **LLM judge — persona swap test**
  - Blind pairwise: “Given this trajectory, did persona A or B produce it?”
  - A/B order randomized
  - Chance = 0.50
  - 1.0 = fully distinguishable

## 3. Numerical Results

### 3.1 Verdicts (rule-based three-axis)

**Event 01 — deadline compression (`gpt-5.4-mini`):**

| Agent | P01 balanced | P02 tired | P03 social |
|---|---|---|---|
| **full** | **fail** (assignment 0.59 + quiz skipped) | **fail** (quiz skipped) | **fail** (quiz skipped) |
| no_desire | fail (quiz skipped) | **excellent** | **excellent** |
| memory_only | fail (assignment + quiz) | **excellent** | **excellent** |
| prompt_only | fail (quiz skipped) | **excellent** | **excellent** |

**Event 02 — campus routine (`gpt-5.4-mini`):**

- All 12 runs pass.
- All but one are **excellent**.
- `full × P03` on nano is the sole `pass`.

**Full agent — nano vs. mini on Event 01:**

- No difference in pass/fail: all 3 personas fail on both models.
- Mini is marginally more efficient (100% indicator match on P01 and P03), but that does not prevent the hard loss.

### 3.2 LLM-judge persona consistency (1–5, overall)

| Agent (mini, E01) | P01 | P02 | P03 |
|---|---|---|---|
| full | 4 | 4 | 3 |
| no_desire | 4 | **5** | 3 |
| memory_only | 4 | 4 | 3 |
| prompt_only | 4 | 4 | 3 |

| Agent (mini, E02) | P01 | P02 | P03 |
|---|---|---|---|
| full | 4 | 4 | 3 |
| no_desire | 3 | 3 | **2** |
| memory_only | **5** | 4 | 4 |
| prompt_only | 4 | 4 | 3 |

Pattern: scores cluster in the 3–4 range. `P03 social_explorer` is consistently the hardest persona to keep in character; judges flag that it “under-expresses the overcommitting weakness” in nearly every run.

### 3.3 Persona swap-test accuracy

| Matrix | Accuracy (chance = 0.50) |
|---|---|
| full · E01 · mini | 0.58 |
| full · E01 · nano | 0.67 |
| full · E02 · mini | **1.00** |
| full · E02 · nano | 0.58 |
| memory_only · E01 · mini | 0.50 |
| memory_only · E02 · mini | 0.67 |
| no_desire · E01 · mini | 0.58 |
| no_desire · E02 · mini | 0.58 |
| prompt_only · E01 · mini | 0.50 |
| prompt_only · E02 · mini | 0.58 |

Most cells hover at 0.50–0.67 — the personas are weakly, not strongly, distinguishable from trajectories alone.

The lone 1.00 (`full · E02 · mini`) is encouraging: it shows distinguishability is *achievable* when the event surfaces persona-diagnostic choices (lunch vs. study vs. workout), but it does not generalize across runs.

## 4. Qualitative Analysis (from raw trajectories)

### 4.1 The full agent’s perseveration failure

The single clearest finding is behavioral: on the **hard** event (`E01`), the full agent’s extra cognitive machinery *hurts* rather than helps.

Two concrete failure modes:

#### (a) Message-loop on P01 (balanced)

`runs/matrix__full__event_01_deadline_compression__gpt-5.4-mini/persona_01_balanced_sophomore` shows turns 2–9 all sending near-identical “Materials are basically ready” messages to `teammate_A`.

Each reflection re-seeds the same “reassure teammate” desire → identical goal → identical action.

The assignment stays at 0.59, quiz prep never starts, and the 10-turn budget expires. Both hard losses trigger.

#### (b) Rumination spiral on P02 (tired achiever)

The memory log becomes a record of exhaustion:

> "Nothing new changed, but the pressure still feels stacked..."
>
> "Nothing new is happening, but I'm still stuck..."

Each new reflection echoes the last.

The agent ends with assignment at 0.88 (just below the 0.7 threshold it safely cleared — but then burned all remaining turns on sub-0.25-hour tweaks) and quiz prep at 0.0.

Contrast the `prompt_only` baseline on the same persona (`runs/matrix__prompt_only__event_01_deadline_compression__gpt-5.4-mini/persona_02_tired_achiever`):

- Turns 0–6 drive assignment A to 1.00
- Turn 7 sleeps 4 h
- Turns 8–9 reach quiz progress 0.43

No hard loss.

### 4.2 Why the baselines win on the hard event

The reactive baselines do not have desires or goals to re-evoke each turn. They see the current world state and choose the most urgent action directly.

On a time-pressured event with hard deadlines, this is strictly better than the full agent’s multi-step reasoning, which tends to overweight *psychological* goals (“don’t look unreliable”) over *state* goals (“finish the deliverable”).

### 4.3 What the full agent gets right (when the event allows)

On `E02` (the non-deadline social-routine event), the full agent’s goal stack helps.

The `P03 social_explorer` trajectory shows explicit boundaries (“30-minute bounded participation”, “join lunch with a soft cap”) that the swap-test judges identify as persona-diagnostic.

This produced the 1.00 swap accuracy. Desires + goals become an asset when the event rewards *composing* multiple soft constraints.

### 4.4 Persona collapse patterns

Across all matrices, judges consistently note that `P03 (social_explorer)` is “too disciplined” and “under-expresses over-joining.”

The persona’s stated weakness — saying yes too easily — rarely manifests. Instead, the agent defaults to bounded, self-protective participation.

This is the single largest persona-fidelity gap in the data.

## 5. Headline Findings

1. **More cognitive structure is not monotonically better.** On hard deadline-pressured events, the full agent fails all three personas while simpler baselines pass two of three. The extra modules create perseveration and rumination loops.
2. **Modules pay off on non-adversarial events.** On `E02`, the full agent’s explicit goals + desires drive the only 1.00 swap-test accuracy in the dataset.
3. **Personas are detectable but not strongly.** Most matrices sit between 0.50 and 0.67 swap accuracy. Agents do inherit persona flavor, but they partially collapse toward a common “diligent-student” default under pressure.
4. **Weakness expression is the weakest axis.** The persona the engine finds hardest to preserve is `P03`: social impulsivity is systematically filtered out.
5. **Model scale (nano vs. mini) barely moves the hard-loss outcome.** The bottleneck is the agent architecture, not raw capability.

## 6. Limitations

- Single seed per matrix; `swap-trials = 2` gives 12 judge calls per matrix but no true variance estimate.
- `max_turns = 10` is aggressive for `E01`; a longer horizon might let the full agent recover.
- The rule-based hard-loss check is a keyword match, which is why `full × P01 × E01` is marked `fail` on assignment *and* quiz even though the root cause is the same loop.
- Adaptation judge (`judge_adaptation`) is implemented but not run in the current matrix; it is a natural next step.

## 7. Next Steps

1. **Break the perseveration.**
   - Cheapest fix: when the goal list is unchanged from the previous turn, force the action module to penalize the previously taken action.
   - More principled fix: have `generate_goals` accept a “last action was X, did it advance any goal?” signal.
2. **Run `judge_adaptation`** on the scheduled-event split (quiz posting at Wednesday 14:00 in `E01`, outlets-full at Tuesday 13:15 in `E02`) to get adaptation scores, not just end-state scores.
3. **Persona-specific prompts for weakness expression.** The flattening of `P03` suggests a system-prompt layer that explicitly asks the agent to surface its known weakness in at least one turn.
4. **Scale the matrix** to more seeds, longer horizons, and a stronger frontier model as a capability ceiling.
5. **Richer swap-test.** Currently, there is one trajectory per persona per matrix; averaging across multiple runs per persona would tighten the accuracy estimate.

