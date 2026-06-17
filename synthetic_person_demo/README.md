# Synthetic Person Demo

This directory contains the standalone live demo for the persona-measurement study.

## Files

- `synthetic_person_demo.html`: the self-contained browser demo for presentation.
- `make_live_demo.py`: regenerates the HTML from the fresh study artifacts under `../sandbox/runs/study__qwen3.5-*/`.
- `demo_presenter_guide.md`: presenter-only click flow, timing, and narration notes.
- `demo_design_rationale.md`: selected runs, findings, and design rationale.

## Open The Demo

Open `synthetic_person_demo.html` directly in a browser, or serve the repository root and navigate to:

```text
synthetic_person_demo/synthetic_person_demo.html
```

The demo is click-driven. It does not call an LLM or require network access during presentation.

## Regenerate

From the repository root:

```bash
python3 synthetic_person_demo/make_live_demo.py
```

The generator reads project data from `sandbox/library/` and fresh Qwen3.5 results from `sandbox/runs/study__qwen3.5-*/`, then rewrites `synthetic_person_demo/synthetic_person_demo.html`.
