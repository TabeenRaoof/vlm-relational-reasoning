# Pilot-scale (n=300) analysis input directory

`analyze_pilot.py`'s loader (`load_all_results`) concatenates every `*.csv`
directly in whatever `--results-dir` it's pointed at and dedups per
`(model_name, item_id)`. `results/` root now also holds the n=2000
confirmatory CSVs for `qwen2.5vl:3b`, `qwen2.5vl:7b`, and
`jetson-qwen2.5vl:3b-q4_K_M` — each sharing a `model_name` with its pilot-scale
counterpart. Pointing the loader at `results/` root would silently merge the
two generations (n=2000 rows win on dedup), corrupting the pilot analysis
with a mixed n across models.

This directory is a **copy-only, non-destructive isolation** of the six
pilot-scale (n=300, or n=100 for the inverted-prompt ablation) CSVs, so the
pilot analysis can be regenerated with `--results-dir results/pilot_n300`
without touching the frozen n=2000 files (which are path-referenced by
`preregistration/PREREGISTRATION_H3_n2000.md`, `docs/RUN_PLAN_H3.md`, and
`scripts/h3_equivalence_test.py`, and therefore must not move).

Regenerate with:
```bash
python scripts/analyze_pilot.py --results-dir results/pilot_n300 \
    --output-dir results/analysis_full \
    --model-order "jetson-qwen2.5vl:3b-q4_K_M,qwen2.5vl:3b,qwen2.5vl:7b,gemini-2.5-flash"
```

**These files are copies, not the source of truth.** If a pilot CSV is ever
re-collected, re-copy it here — don't hand-edit anything in this directory.

Source files (all from `results/` root):
- `mac_qwen25vl_3b.csv`, `mac_qwen25vl_7b.csv`
- `jetson_qwen25vl_3b_q4.csv`
- `gemini_2_5_flash.csv`, `gemini_2_5_flash_thinking.csv`, `gemini_2_5_flash_inverted_prompt.csv`
