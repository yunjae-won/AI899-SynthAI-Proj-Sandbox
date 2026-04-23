#!/bin/bash

source .venv/bin/activate

## Compare performance of gpt-5.4-mini and gpt-5.4-nano for both events and and full agent configurations.
# Results in runs/matrix__full__event_01_deadline_compression__gpt-5.4-nano
python -m sandbox.run matrix  --event   event_01_deadline_compression --agent full --provider  openai --max-turns 10 --swap-trials 2 
# Results in runs/matrix__full__event_01_deadline_compression__gpt-5.4-mini
python -m sandbox.run matrix  --event   event_01_deadline_compression --agent full --provider  openai --max-turns 10 --swap-trials 2 --model gpt-5.4-mini
# Results in runs/matrix__full__event_02_campus_routine__gpt-5.4-nano
python -m sandbox.run matrix  --event   event_02_campus_routine --agent full --provider  openai --max-turns 10 --swap-trials 2 
# Results in runs/matrix__full__event_02_campus_routine__gpt-5.4-mini
python -m sandbox.run matrix  --event   event_02_campus_routine --agent full --provider  openai --max-turns 10 --swap-trials 2 --model gpt-5.4-mini


## Ablate performance of agent configurations for gpt-5.4-mini on both events.
# Results in runs/matrix__prompt_only__event_01_deadline_compression__gpt-5.4-mini
python -m sandbox.run matrix  --event   event_01_deadline_compression --agent prompt_only --provider  openai --max-turns 10 --swap-trials 2 --model gpt-5.4-mini
# Results in runs/matrix__memory_only__event_01_deadline_compression__gpt-5.4-mini
python -m sandbox.run matrix  --event   event_01_deadline_compression --agent memory_only --provider  openai --max-turns 10 --swap-trials 2 --model gpt-5.4-mini
# Results in runs/matrix__no_desire__event_01_deadline_compression__gpt-5.4-mini
python -m sandbox.run matrix  --event   event_01_deadline_compression --agent no_desire --provider  openai --max-turns 10 --swap-trials 2 --model gpt-5.4-mini
# Results in runs/matrix__no_desire__event_01_deadline_compression__gpt-5.4-mini
python -m sandbox.run matrix  --event   event_02_campus_routine --agent prompt_only --provider  openai --max-turns 10 --swap-trials 2 --model gpt-5.4-mini
# Results in runs/matrix__no_desire__event_01_deadline_compression__gpt-5.4-mini
python -m sandbox.run matrix  --event   event_02_campus_routine --agent memory_only --provider  openai --max-turns 10 --swap-trials 2 --model gpt-5.4-mini
# Results in runs/matrix__no_desire__event_01_deadline_compression__gpt-5.4-mini
python -m sandbox.run matrix  --event   event_02_campus_routine --agent no_desire --provider  openai --max-turns 10 --swap-trials 2 --model gpt-5.4-mini