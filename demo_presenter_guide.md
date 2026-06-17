# Synthetic-Person Demo Presenter Guide

## Setup

Open `synthetic_person_demo.html` in a wide browser window or fullscreen mode. The demo advances only when the presenter clicks the active module card or the large `Execute ...` control. Space and Enter also advance; the small curved arrow in the top-right corner reverses one step if needed.

Target runtime is 45-55 seconds. The presenter should explain the components verbally; the page is intentionally a visual stage with labels, prompt fragments, module highlights, task state, action buttons, and chat-style runtime output.

## Presentation Flow

### 0. Ready State, 3 seconds

Point to the three-pane layout:

- Left: live trajectory stream.
- Middle: task state and action interface.
- Right: prompt/module inspector.

Say that this is one representative task from the study: a study block interrupted by a lunch invite.

### 1. Execute Persona, 5 seconds

Click `Execute Persona`.

Explain that a persona is structured data, not a nickname. In this comparison, conflict style and time tendency are held fixed, while the value axis changes. For the visible persona, relationships are weighted high and academic work is weighted low.

Key line: "This is the controlled intervention: the same world, but a different value priority."

### 2. Execute World, 5 seconds

Click `Execute World`.

Explain the event: the agent has an assignment and reading to do, then receives a lunch invitation from `cohort_3`. The middle pane shows the task/action interface that the final decision must use.

Key line: "The model sees the same state and the same actions no matter which persona is loaded."

### 3. Execute Prompt, 6 seconds

Click `Execute Prompt`.

Use the right pane. Explain that the system now has the persona fields, the observation, and the state snapshot in one packet. Do not read every line; point out the highlighted prompt block and the module card.

Key line: "The prompt is assembled from explicit fields rather than freeform storytelling."

### 4. Execute Reflect, 5 seconds

Click `Execute Reflect`.

Explain that the first internal call summarizes what matters in the current situation. This keeps the following desire and goal steps grounded in the same observation.

Key line: "Reflection is the bridge from raw event to internal state."

### 5. Execute Desires, 6 seconds

Click `Execute Desires`.

Explain that the affiliation persona turns the lunch invite into immediate social drives. The prompt inspector shows the desire output that will be fed into the next stage.

Key line: "The persona does not choose an action yet; it first changes the agent's drives."

### 6. Execute Goals, 5 seconds

Click `Execute Goals`.

Explain that desires are converted into ranked goals. The top goal becomes leaving the study block to join `cohort_3`.

Key line: "The final decision is now being shaped by an ordered goal, not just by the event text."

### 7. Execute Action, 7 seconds

Click `Execute Action`.

Explain that the final action prompt combines the top goal with the available action list, then requires one structured JSON decision. Call out the selected `attend(social_value)` button and the action appearing in the chat stream.

Key line: "Same action interface, different persona-conditioned decision."

### 8. Execute World, 4 seconds

Click `Execute World`.

Explain that the world applies the action. The relationship score changes from 7.0 to 7.6 while assignment progress remains at 40%.

Key line: "The chosen action becomes a state transition."

### 9. Execute Measure, 7 seconds

Click `Execute Measure`.

Explain the counterfactual:

- Achievement persona chooses `work_on(assignment_value)`.
- Affiliation persona chooses `attend(social_value)`.
- Neutral control chooses `work_on(assignment_value)`.

Then call out the finding: for Qwen3.5-4B, the value axis has the largest net persona effect in this study slice, about +0.39 after subtracting the neutral floor.

Key line: "The framework measures whether persona fields actually change actions, rather than only producing plausible text."

## Timing

Suggested pacing:

- Ready + persona + world: 13 seconds.
- Prompt formation through goals: 22 seconds.
- Action, world update, measurement: 20 seconds.

Total: about 55 seconds. To fit closer to 45 seconds, shorten the verbal explanation for reflection and desires and spend the most time on the action prompt and counterfactual.
