# Study run logs (final 3-model × 7-event run)

- **`driver_run.log`** — the `run_all_qwen.sh` driver log: per-episode progress +
  ETA, and the printed **final-result tables** (`=== STUDY <model> ===` axis×event
  divergence matrices and the model×axis aggregate). Final results in log form.
- **`vllm_4b.log` / `vllm_9b.log` / `vllm_35b_int4.log`** — the vLLM OpenAI-server
  logs for each model serve (startup, KV-cache, throughput stats).

The full per-turn **environment ↔ agent interaction** for every episode
(`state_snapshot` → `action` → `observation`, plus `final_state`) is stored as
structured JSON, not here:

```
../sandbox/runs/study__<model>/<agent>/<persona>__<event>/seed<n>/trajectory.json
```

Aggregated final metrics:
`../sandbox/runs/study__*/measurement.json` and `../sandbox/runs/aggregate.json`.
A human-readable replay of these interactions is in `../demo.html`.
