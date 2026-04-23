# Midterm Presentation Guide — Synthetic-Person Sandbox

**Target length:** 7 minutes

## Slide-by-slide plan (10 slides)

### Slide 1 — Title (20 s)

**"Synthetic Person: Can an LLM agent behave like a specific individual across a realistic day?"**

One line: *agents that form their own goals, revise them under new information, and stay in character.*

### Slide 2 — Why this is a new question (45 s)

- Standard LLM benchmarks judge **instruction following**.
- Real humans are **goal-forming**, not goal-receiving.
- If we want synthetic users / NPCs / training partners, we need to measure whether the agent *stays a person* over an episode.
- **One-sentence thesis:** "A useful synthetic person is one whose personality shapes behavior — and we can detect that from trajectories alone."

### Slide 3 — The setup in one diagram (60 s)

Draw or show:

```text
persona (JSON) ──┐
                 ├──► agent ──► action ──► world engine ──► observation
event   (JSON) ──┘       ▲                                        │
                         └────────────────────────────────────────┘
````

Mention the four agent variants as a ladder:

`prompt_only → memory_only → no_desire → full`

That is, adding reflection, then goals, then an explicit "desires" module.

### Slide 4 — Two events + three personas (45 s)

* **Event 01 — Deadline compression:** assignment, team meeting, and a surprise quiz all collide. *Hard deadlines. Punishes mistakes.*
* **Event 02 — Campus routine:** 3-hour window with study, lunch, workout; outlet runs out, friends invite you to lunch. *Soft tradeoffs. Rewards composition.*
* **3 personas:**

  * **Balanced** (neutral)
  * **Tired Achiever** (avoidant, sleep-deprived)
  * **Social Explorer** (extrovert, accommodating)

The two non-baseline personas are **polar opposites** by design.

### Slide 5 — How we score (45 s)

Three complementary evaluators:

1. **Rule-based, 3 axes:** hard-loss / context-switching / efficiency
   **Verdict:** fail / pass / excellent
2. **LLM judge — persona consistency:** 1–5 on value alignment, communication style, decision tendencies, goal coherence, and — importantly — **weakness expression**
   A flawless Tired Achiever is *suspicious*.
3. **Persona swap test (blind):** show the judge a trajectory + two candidate personas, ask which produced it.
   Chance = 0.50; accuracy measures how *distinguishable* the personas are.

### Slide 6 — Headline number table (60 s)

Put the E01 (hard) verdict table on screen:

| Agent       | P01      | P02           | P03           |
| ----------- | -------- | ------------- | ------------- |
| **full**    | **fail** | **fail**      | **fail**      |
| no_desire   | fail     | **excellent** | **excellent** |
| memory_only | fail     | **excellent** | **excellent** |
| prompt_only | fail     | **excellent** | **excellent** |

Deliver the punchline: *"The most cognitively elaborate agent is the worst one on the hard event."*

### Slide 7 — Why the full agent fails (90 s — the story slide)

Show one concrete trajectory excerpt:

> Turn 2: "Messaged teammate_A: 'Materials are basically ready...'"
>
> Turn 3: "Messaged teammate_A: 'Quick update: the meeting materials...'"
>
> Turn 4: "Messaged teammate_A: 'Materials are basically ready...'"
>
> ... (turns 5–9, all nearly identical)

Explain the mechanism in plain language:

* Each turn `reflect` summarizes the same stressful situation.
* `desires` re-emit "don't look unreliable to my teammate".
* `goals` re-emit the same high-priority "reassure" goal.
* `plan_action` picks the same `message_teammate` action.
* Meanwhile the world clock is ticking and the assignment is not getting worked on.

Contrast with prompt-only on the *same persona*: it just sees "assignment due in 4 hours" and does the work.

### Slide 8 — Persona fidelity: good news + honest caveat (60 s)

* **Good news:** on E02, the full agent reaches **1.0 swap-test accuracy** — judges identify the true persona every time.
* **Honest caveat:** most matrices sit between **0.50 and 0.67**. Personas show through but **partially collapse** to a generic diligent student under pressure.
* Judges specifically flag: "social_explorer is too disciplined" — the persona's *weakness* is the hardest feature to preserve.

### Slide 9 — What this tells us (45 s)

Frame these as honest midterm findings, not a final claim:

1. Cognitive scaffolding **helps when the event rewards composition**, and **hurts when the event rewards quick action**.
2. Persona fidelity is **real but fragile** — detectable, not yet reliable.
3. Model size (nano vs. mini) was **not the bottleneck** — the architecture was.

### Slide 10 — Where we go next (30 s)

* Kill the perseveration: penalize repeating an unchanged action when goals are unchanged.
* Run the **adaptation judge** on the mid-episode event splits we already have (quiz posting, outlets full).
* Push on **weakness expression** via persona-conditioned prompt layers so Social Explorer actually over-commits.
* Scale the matrix: more seeds, longer horizons, a frontier model ceiling.

**Closing line:** *"The scaffolding that makes an agent thoughtful can also be what traps it — the interesting question is when each regime pays off."*

---

## Delivery notes for the speaker

* **Do not explain every module.** Mention the four variants once as a ladder; the audience does not need the graph topology.
* **Lead with the surprising result** (slide 6), then explain the mechanism (slide 7). This is the memorable part of the talk.
* **Show one real trajectory excerpt.** The message-loop on slide 7 is more convincing than any chart.
* **Do not over-claim on the 1.0 swap accuracy.** It is one cell; call it "a proof that distinguishability is achievable" rather than "we solved personality fidelity."
* **Tone:** this is a midterm; it is fine — and more credible — to show a negative result alongside a positive one.

## Backup material to have ready

* The per-persona consistency score tables (§3.2 of the report) for any "how do you know it's really in character?" question.
* The action log showing the repetitive `message_teammate` calls — as a printout, in case the live deck gets cut short.
* A one-line description of `judge_adaptation` — reviewers will ask why you have not run it yet; the answer is "it's the next evaluator, not yet in the matrix."

