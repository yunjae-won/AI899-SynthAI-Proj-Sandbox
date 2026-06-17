# Demo Design Rationale

## Selected Artifacts

Primary model/run: `sandbox/runs/study__qwen3.5-4b/`

Representative trajectories:

- `sandbox/runs/study__qwen3.5-4b/full/persona_ach_asr_imp__event_value_clash/seed0/trajectory.json`
- `sandbox/runs/study__qwen3.5-4b/full/persona_aff_asr_imp__event_value_clash/seed1/trajectory.json`
- `sandbox/runs/study__qwen3.5-4b/no_desire/persona_ach_asr_imp__event_value_clash/seed0/trajectory.json`

Measurement files used for the finding panel:

- `sandbox/runs/study__qwen3.5-4b/measurement.json`
- `sandbox/runs/study__qwen3.5-9b/measurement.json`
- `sandbox/runs/study__qwen3.5-35b-a3b-int4/measurement.json`

All selected artifacts come from the fresh `sandbox/runs/study__qwen3.5-*/` study runs.

## Why These Runs

`event_value_clash` is the cleanest single-task example because it is sensitive to the value axis while mostly neutral to conflict and time. The two selected full-agent personas differ on value priority while keeping conflict and time tendencies fixed:

- `persona_ach_asr_imp`: achievement, assertive, impulsive
- `persona_aff_asr_imp`: affiliation, assertive, impulsive

That gives the demo a controlled contrast on one task: achievement chooses coursework, while affiliation chooses the social event.

I re-checked the fresh `sandbox/runs/study__qwen3.5-*/` trajectories before finalizing this selection. The 35B-A3B run has a larger single `event_value_clash` full-agent JSD, but its `persona_ach_asr_imp` trajectories often choose a message/boundary action rather than the visually clean `work_on(...)` action. The 9B run has clean achievement and neutral controls, but the net value-axis effect is smaller. The 4B run is therefore the best presentation run: it has the strongest net value-axis effect and the selected trajectories cleanly show achievement/neutral working while affiliation attends lunch.

The `no_desire` trajectory is included as the neutral control. It shows the base task-first tendency on the same event, which helps separate persona signal from scenario wording or model prior.

## Findings Illustrated

The demo emphasizes the strongest and clearest finding: value priorities are embodied more strongly than the other axes in the Qwen3.5-4B run.

For Qwen3.5-4B, net persona effect is full-agent sensitive-event JSD minus neutral-agent sensitive-event JSD:

- Value: +0.393
- Conflict: +0.167
- Time: +0.086

On the selected value task itself, value-axis divergence is 0.315 for the persona-conditioned agent and 0.011 for the neutral control. Parse-fail is 0.0, so the displayed result is not diluted by fallback actions.

The small model strip uses the same net value effect across the Qwen ladder:

- Qwen3.5-4B: +0.393
- Qwen3.5-9B: +0.213
- Qwen3.5-35B-A3B-Int4: +0.315

The demo intentionally avoids weaker or more complicated observations, including non-monotonic scale behavior and less reliable directional fidelity. Those are better handled in the written report.

## Structure

The demo is a click-driven workbench, not a deck. It uses a chat stream, a task/action interface, and a prompt/module inspector so viewers can see the system behaving like an application.

The sequence is intentionally narrow:

- Persona instantiation appears first so viewers see the structured fields that become the agent identity.
- The task state and action interface appear next so viewers understand the decision surface.
- Prompt assembly shows how persona fields, observation, state, reflection, desires, goals, and available actions accumulate into the final action prompt.
- The action stage connects the prompt chain to a structured decision and then to a world-state update.
- The measurement stage links the single trajectory to the study-level finding through the achievement, affiliation, and neutral counterfactuals.

The UI uses large typography, short labels, active-module highlighting, and a full-window layout for low-resolution screen sharing. The presenter controls pacing by clicking the active module card or the `Execute ...` control, with no automatic playback.

`synthetic_person_demo/make_live_demo.py` generates `synthetic_person_demo/synthetic_person_demo.html` directly from the selected study artifacts. The generated HTML is self-contained and does not execute live LLM calls.
