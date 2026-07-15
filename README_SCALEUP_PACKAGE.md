# n=2000 Scale-Up Package — README

This package contains everything needed to run the pre-registered n=2000
confirmatory study. **These are ADDITIVE files** — copy them into your existing
`cs587-pilot` repo; they do not replace anything you already have. Review each,
then `git add` them yourself so every change is version-controlled and you
understand it before it's public.

## What's here and where it goes in your repo

```
preregistration/PREREGISTRATION.md   -> preregistration/   (NEW dir)
scripts/check_vsr_ceiling.py         -> scripts/           (GATE script, run first)
scripts/interaction_test.py          -> scripts/           (primary confirmatory test)
docs/RUN_PLAN.md                     -> docs/              (the ordered, gated plan)
docs/PIPELINE_NOTES.md               -> docs/              (2 required code fixes + gotchas)
CLAUDE_CODE_INSTRUCTIONS.md          -> repo root          (brief for Claude Code)
README_SCALEUP_PACKAGE.md            -> repo root (this file)
```

## The order to do things (short version — full detail in docs/RUN_PLAN.md)

1. **`python scripts/check_vsr_ceiling.py`** — does VSR even have 2000 balanced
   items in your two categories? This gates everything. If it caps lower, decide
   (pre-register the achievable n, or add a dataset) before going further.
2. **Finalize + commit `preregistration/PREREGISTRATION.md`** — fill every
   `<FILL>`, record the git commit hash in §9. THIS MUST HAPPEN BEFORE YOU COLLECT
   OR LOOK AT ANY n=2000 DATA. That ordering is the entire point — it's what makes
   the confirmatory result honest.
3. **Curate + verify nesting** (`curate_eval_set.py` at the new n, then the
   nesting check).
4. **Run all four models** over the frozen set (Mac 3B, Mac 7B, Gemini x3 arms,
   Jetson 3B-Q4). Jetson is the slow, chunked, reboot-between part.
5. **Make the two pre-registered code fixes** to `analyze_pilot.py`, then run the
   frozen analysis + `interaction_test.py`.
6. **Report whatever you find**, per PREREGISTRATION.md §7 — a null interaction is
   an honest, reportable outcome, not a failure.

## Why this is structured as a pre-registration

You asked to avoid the weakness that "pre-specification is strongest when
genuinely done before analysis." The fix isn't code — it's sequence. Because the
n=2000 data does not exist yet, a pre-registration written and committed now is
genuinely before-the-fact for the confirmatory study. The pilot (n=300) is framed
as exploratory / hypothesis-generating; n=2000 is the confirmatory test. That is
the standard, defensible way to do this, and it's what makes the result
bulletproof. The one hard rule: **commit the pre-reg before touching new data.**

## What is NOT in this package (still yours to do, later)

- The public-repo polish (top-level README for the whole project, requirements.txt
  audit, LICENSE, .gitignore review, secrets-history check) — separate task, do it
  before going public.
- The paper write-up itself.
- Any decision that's yours to make: target n if VSR caps below 2000, the exact
  correction method, secondary/exploratory questions. The pre-reg has `<FILL>`
  markers for each — they're your calls, not mine.
