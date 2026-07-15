# Instructions for Claude Code — n=2000 Confirmatory Study

You (Claude Code) are running on the user's Mac, in their `vlm-relational-reasoning` repo. This
file, plus the referenced docs, is your full brief. Read `docs/RUN_PLAN.md` and
`preregistration/PREREGISTRATION.md` fully before doing anything.

## Context you need (the short version)

The user ran a pilot study (n=300) on compute-sensitivity of relational reasoning
in vision-language models. It found real projective-spatial degradation at lower
compute, but the key **interaction** test (does projective degrade *more* than
containment?) was underpowered (p=0.12, only 35% power). A power analysis showed
n=2000 gives ~98% power. The user, with advisor approval, is scaling up to a
pre-registered n=2000 confirmatory study.

Full background: `scripts/POST_SUBMISSION_WORK_REPORT.md` (if present) and
`docs/PIPELINE_NOTES.md`.

## The single most important rule

**This is a PRE-REGISTERED study. The analysis plan must be frozen (committed to
git) BEFORE any n=2000 data is collected or viewed.** Do not run any model over
the new eval set, and do not compute any new result, until Phase 0's gates are
cleared and the pre-registration is committed. The scientific validity of the
whole exercise depends on this ordering. If you're ever unsure whether an action
would "peek" at confirmatory results before the freeze, STOP and ask the user.

## Work in gated phases (from docs/RUN_PLAN.md) — do NOT chain past a gate

- **Phase 0:** run `scripts/check_vsr_ceiling.py`, help the user finalize and
  commit `preregistration/PREREGISTRATION.md`. **Stop at each 🔴 GATE and get
  explicit user confirmation before proceeding.** Do not fill in the user's
  scientific choices (target n, correction method) yourself — surface them and
  let the user decide.
- **Phase 1:** curate the n=2000 set, verify nesting against the pilot, fetch
  images. Stop at the nesting gate.
- **Phase 2:** run each model. The Jetson run is long and must follow
  `docs/jetson_setup.md` (chunk + reboot). Confirm with the user before kicking
  off the multi-hour Jetson run and before spending money on Gemini API calls.
- **Phase 3:** make the two pre-registered code fixes (see docs/PIPELINE_NOTES.md),
  then run the frozen analysis + interaction_test.py.

## Code changes you are authorized to make (and only these, before analysis)

1. Add `--model-order` to `analyze_pilot.py` (FIX 1 in PIPELINE_NOTES.md).
2. Add multiple-comparisons correction to `analyze_pilot.py` (FIX 2). The
   confirmatory-family allowlist must be hard-coded to match PREREGISTRATION.md §3
   exactly — show the user the allowlist and get confirmation it matches before
   running.
`scripts/interaction_test.py` and `scripts/check_vsr_ceiling.py` are provided; use
them, don't rewrite them unless they don't run in this environment (then fix
minimally and tell the user what you changed).

## Guardrails

- **Never commit `.env`** or any API key. Verify `git log --all --full-history --
  .env` is empty before any push. If it's not, tell the user immediately (history
  rewrite needed before public release).
- **One authoritative CSV per model_name** in `results/` at analysis time; move
  pilot/intermediate files to a subdir (see gotcha #1 in PIPELINE_NOTES.md).
- **Do not delete the user's data.** Preserve pilot files; you're extending, not
  replacing.
- **Show plans before executing** multi-step or expensive operations, matching how
  the user has worked throughout this project.
- Report deviations from the pre-registration honestly; they go in the paper's
  "Deviations" subsection.

## Definition of done for this handoff

- Pre-registration committed (hash recorded in §9) BEFORE data collection.
- n=2000 eval set built and nesting-verified.
- All four models run over it; outputs in `results/` with unique names.
- Two code fixes made and confirmed against the pilot sanity numbers.
- Frozen analysis + interaction test run; results written to
  `results/analysis_n2000/`.
- Whatever the interaction result is, it's reported per PREREGISTRATION.md §7.
