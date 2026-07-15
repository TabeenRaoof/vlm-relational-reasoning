# Pipeline Notes — required fixes and known gotchas

Two code changes to `analyze_pilot.py` are needed and are **pre-registered**
(they must be made before running the confirmatory analysis, and they must match
the frozen analysis plan). Plus a list of real gotchas found during the pilot.

---

## FIX 1 — Explicit model order (RRS / DT compute axis bug)

**Problem:** `analyze_pilot.py` infers the compute ordering of models from their
names via a `rank()`-style heuristic. The Mac model `qwen2.5vl:3b` matches none of
the expected patterns ("jetson", "7b", "gemini"), so it falls through to a default
(`return 99`) and is ranked as the LEAST-constrained / most-powerful model — which
is backwards (it's one of the MOST constrained). This scrambles the RRS (Relational
Retention Score) and DT (Degradation Trajectory) metrics, which depend on the axis
order. It also silently pulls the inverted-prompt and thinking-ablation arms onto
the main compute axis, where they don't belong.

**Fix:** add a `--model-order` CLI argument that takes an explicit,
comma-separated list from most-constrained to least-constrained, and use it
instead of the name heuristic. Example:
```
--model-order "jetson-qwen2.5vl:3b-q4_K_M,qwen2.5vl:3b,qwen2.5vl:7b,gemini-2.5-flash"
```
Only models on this list belong on the compute axis; control/ablation arms
(inverted, thinking) are excluded from RRS/DT and analyzed separately.

**Verify after fixing:** the printed "Inferred model order" must match the axis
you intend (most→least constrained), NOT put the 3B last.

---

## FIX 2 — Multiple-comparisons correction

**Problem:** the McNemar loop reports only raw p-values. For a peer-reviewed
claim this is insufficient.

**Fix:** after building the McNemar results table, apply correction and emit
extra columns. Use `statsmodels.stats.multitest.multipletests`. Emit BOTH:
1. Correction across the **confirmatory family only** (the 6 tests named in
   PREREGISTRATION.md §3) — this is the primary reported correction. Use the
   pre-registered method (recommended: Holm — `method="holm"`).
2. Correction across **all** pipeline tests (maximally conservative) — reported
   for transparency, so a strict reviewer sees you didn't hide it.

The confirmatory family must be identified by an explicit allowlist of
(model_a, model_b, category) tuples in the code — NOT inferred — so it exactly
matches the pre-registration. Hard-code that allowlist to match §3.

**Reference numbers from the pilot (n=300), to sanity-check your implementation:**
- Core test 3B-vs-7B projective: raw p=0.003.
  - Within a 9-test family: Holm/BH p_adj ≈ 0.027 (survives).
  - Across all ~45 tests: Bonferroni ≈ 0.135, BH ≈ 0.068 (does NOT survive).
- 3B-vs-7B containment: p≈0.79 (null, as intended).
At n=2000 these p-values will change; the point is only to confirm the correction
code behaves correctly on known inputs.

---

## Known gotchas (all real, all hit during the pilot)

1. **Filename/model_name collisions.** `load_all_results()` globs every `*.csv`
   directly in `results/` and dedups per (model_name, item_id). If two files share
   a `model_name` (e.g. a pilot run and a full run both labeled
   `jetson-qwen2.5vl:3b-q4_K_M`), they get silently merged with "last read wins."
   **Rule:** one authoritative file per model_name in `results/` at analysis time;
   move everything else to a subdirectory (the loader skips subdirs via the
   `csv_path.parent != results_dir` check).

2. **Jetson duplicate rows.** A chunked/crash-retried Jetson run produces a raw
   CSV with more rows than unique items (retried failures leave old rows behind).
   The dedup (keep-last per model_name+item_id) handles this correctly — do not
   "fix" the file by hand; let the loader dedup it.

3. **Gemini model version is not frozen.** "gemini-2.5-flash" can change under you.
   Record the exact version string the API returns (if exposed) at run time, and
   state in the paper that Gemini results are not guaranteed re-derivable after a
   provider-side model update. This is an accepted limitation — just disclose it.

4. **Jetson sustained-load OOM.** A single long continuous run gets OOM-killed.
   Always chunk (~100 items) with a full `sudo reboot` before each chunk. See
   docs/jetson_setup.md.

5. **`.env` must never be committed.** It holds the Gemini API key. `.gitignore`
   already lists it; verify with `git log --all --full-history -- .env` (must be
   empty) before making the repo public.
