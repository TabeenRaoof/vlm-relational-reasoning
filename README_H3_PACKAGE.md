# H3 n=2000 Study Package — README

Additive files for the Jetson H3 edge-equivalence study (n=2000), plus the
standalone planning doc for the deferred high-powered interaction study. Copy into
your `vlm-relational-reasoning` repo; review each before `git add`. Nothing here
replaces existing files.

## What's here and where it goes

```
preregistration/PREREGISTRATION_H3_n2000.md   -> preregistration/
scripts/h3_equivalence_test.py                -> scripts/   (verified on n=300 data)
docs/RUN_PLAN_H3.md                           -> docs/
docs/jetson_setup.md                          -> docs/      (consolidated Jetson ops)
CLAUDE_CODE_INSTRUCTIONS_H3.md                -> repo root   (brief for Cursor/Claude Code)
FUTURE_STUDY_interaction_high_power.md        -> docs/ or a planning/ dir (NOT active)
README_H3_PACKAGE.md                          -> repo root (this file)
```

## Two separate things in this package — don't conflate them

1. **The H3 n=2000 study (ACTIVE).** Everything except the FUTURE_STUDY file. This
   is ready to run now: it re-tests edge-hardware equivalence at n=2000 with a
   pre-registered TOST (±3pp margin). VSR-only, no new dataset, no API cost, no
   sequential design. The only real cost is the Jetson chunked-reboot marathon.

2. **The deferred high-powered interaction study (PLANNING ONLY).**
   `FUTURE_STUDY_interaction_high_power.md` is a design brief, not a pre-reg.
   It needs a second dataset (VSR caps at ~3000; the interaction needs ~8000) and
   has open questions to resolve before it becomes real. Do not start it now.

## H3 study — the workflow (full detail in docs/RUN_PLAN_H3.md)

1. **Freeze:** finalize + commit `PREREGISTRATION_H3_n2000.md` BEFORE any Jetson
   n=2000 data. (You've already seen n=300; committing the design/margin first is
   what keeps the confirmatory re-test honest.)
2. **Prep Jetson** (over SSH): reachability, copy `eval_set_n2000.csv` + images,
   re-verify the weight SHA-256 matches the Mac (this is what makes it a
   hardware-only test).
3. **Run** the Jetson in ~100-item chunks with a reboot before each (a single long
   run gets OOM-killed — see docs/jetson_setup.md §5). ~20 chunks.
4. **Analyze:** pull the file, run `h3_equivalence_test.py --margin 3.0`. Report
   whichever of the three §7 outcomes occurred. Do NOT collect more to force
   equivalence.

## Key facts baked into the design (so you don't have to rederive them)

- Margin **±3pp**, pre-specified (your call, professor-approved).
- Power at n=2000: **~99%** to declare equivalence if the true difference is ~1pp
  (the n=300 estimate); ~68% if it's as large as 2pp (§7 covers that case).
- On the n=300 data, the ±3pp TOST *fails* to declare equivalence (90% CI
  [−0.99,+3.02]pp — pokes just past +3pp). This is exactly why n=2000 is worth
  running: the ~2.6× tighter interval should pull both ends inside ±3pp.
- This study makes **no API calls** and does **not** involve Gemini or the 7B
  model — it's purely the 3B on two hardware platforms.

## Built for Cursor / Claude Code

`CLAUDE_CODE_INSTRUCTIONS_H3.md` is written for an agent running in Cursor on the
Mac. It emphasizes the two-machine reality (agent on Mac, Jetson over SSH — the
agent can't see the Jetson directly) and clarifies that the chunked-reboot loop
(~20 cycles) is fully SSH-drivable and autonomous — `sudo reboot` keeps power
applied and the device has no physical button — so once you confirm the run
should start, the agent can drive the whole loop unattended, only flagging you if
something genuinely goes wrong (e.g. a true power loss, which would need the
power cable physically unplugged/replugged).
