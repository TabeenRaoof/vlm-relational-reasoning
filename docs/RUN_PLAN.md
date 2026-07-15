# RUN PLAN — n=2000 Confirmatory Study

Ordered, gated sequence for scaling the study from the n=300 pilot to the
pre-registered confirmatory n. **The gates are not optional** — they are what
make the pre-registration genuine (analysis plan frozen before data exists).

Notation: 🔴 GATE = do not proceed past this until its condition is met and the
result is recorded. Everything assumes you are in the repo root with the venv
active.

---

## PHASE 0 — Ceiling check and freeze (do this FIRST, before any data)

**Step 0.1 — Confirm VSR can support n=2000.**
```
python scripts/check_vsr_ceiling.py
```
Read the VERDICT block.
- If it says n=2000 is achievable → proceed, target stays 2000 (1000/category).
- If VSR caps lower → you have a real decision: pre-register the achievable n
  (and its power from the table), OR add a second dataset. **Do not proceed to
  freeze with a target the data can't support.** Bring this number to your
  advisor if it's below 2000.

🔴 **GATE 0.1:** the achievable n is known and the pre-registration target is set
to a number ≤ that ceiling.

**Step 0.2 — Finalize the pre-registration.**
- Fill every `<FILL>` in `preregistration/PREREGISTRATION.md`.
- Paste the `check_vsr_ceiling.py` VERDICT into §9.
- Delete the DRAFT banner.

**Step 0.3 — Commit the pre-registration BEFORE collecting data.**
```
git add preregistration/PREREGISTRATION.md scripts/check_vsr_ceiling.py
git commit -m "Freeze pre-registration for n=<N> confirmatory study (pre-data)"
git rev-parse HEAD    # copy this hash into PREREGISTRATION.md §9, then amend:
git add preregistration/PREREGISTRATION.md
git commit --amend --no-edit
```
(Optional, strongest: also register on OSF or post the frozen doc somewhere
timestamped, and put that URL in §9.)

🔴 **GATE 0.2 — THE CRITICAL ONE:** the pre-registration is committed to git
and you have NOT looked at any n=2000 results (there are none yet). From here on,
the analysis plan is frozen. Everything downstream just executes it.

---

## PHASE 1 — Build the frozen eval set

**Step 1.1 — Curate at the new n.**
```
python scripts/curate_eval_set.py --n-per-category <N/2> --seed 42 \
    --output data/eval_set_n<N>.csv
```
Seed 42 matches the pilot. Because IDs are content-derived and the rank-key
sampling nests, the pilot's items are a strict subset of this larger set.

**Step 1.2 — Verify nesting against the pilot (integrity check).**
```
python - <<'PY'
import pandas as pd
new = pd.read_csv("data/eval_set_n<N>.csv")
old = pd.read_csv("data/pilot_eval_set_n150.csv")   # the n=300 pilot set
old_ids, new_ids = set(old.item_id), set(new.item_id)
missing = old_ids - new_ids
print(f"new set: {len(new)} items; pilot: {len(old)} items")
print(f"pilot items NOT in new set: {len(missing)} (MUST be 0 for clean nesting)")
assert not missing, "NESTING BROKEN — investigate before proceeding"
print("OK: pilot nests inside the new set.")
PY
```
🔴 **GATE 1.2:** nesting confirmed (0 pilot items missing). If broken, stop —
the seed or taxonomy changed and results won't be comparable.

**Step 1.3 — Fetch any new images.** The larger set references COCO images not
yet cached. Fetch them (reuse the existing fetch mechanism — see how the pilot
image cache was populated; images come from the `image_url` column).

---

## PHASE 2 — Run every model over the frozen set

Each model runs the SAME `data/eval_set_n<N>.csv`. Outputs go to `results/`.
**Use a fresh, unique `--output` filename per model — never reuse a filename or
let two runs share a `model_name`** (this was a real bug source; see the
collision note in docs/PIPELINE_NOTES.md).

**Step 2.1 — Mac Qwen 3B (Q4):**
```
python scripts/run_ollama_eval.py --model qwen2.5vl:3b \
    --input data/eval_set_n<N>.csv --output results/mac_qwen25vl_3b_n<N>.csv
```

**Step 2.2 — Mac Qwen 7B:**
```
python scripts/run_ollama_eval.py --model qwen2.5vl:7b \
    --input data/eval_set_n<N>.csv --output results/mac_qwen25vl_7b_n<N>.csv
```

**Step 2.3 — Gemini 2.5 Flash (needs a NEW API key — the old one was deleted).**
Put the new key in `.env` (NEVER commit it; `.gitignore` already covers `.env`).
```
python scripts/run_gemini_eval.py --model gemini-2.5-flash \
    --input data/eval_set_n<N>.csv --output results/gemini_2_5_flash_n<N>.csv
```
User confirmed: run all three Gemini arms at n=2000 (base + thinking + inverted).
Repeat 2.3 for the thinking and inverted-prompt variants using whatever flags
run_gemini_eval.py exposes for those (see its --help). **Record the exact model
version string the API returns**, if available, for reproducibility (Gemini
"2.5-flash" is not a frozen artifact — note this as a limitation).

**Step 2.4 — Jetson Qwen 3B (Q4) — the slow one.**
Follow docs/jetson_setup.md exactly. Key points, learned the hard way:
- SSD must be mounted and `OLLAMA_MODELS` pointed at it (capacity).
- Run in ~100-item CHUNKS with a full `sudo reboot` before each chunk — a single
  continuous 2000-item run WILL get OOM-killed. At n=2000 that's ~20 chunks.
- Use `--limit` to advance chunk boundaries; resume skips completed items.
- Expect 4–33 hours total depending on the fast/slow state. This is the single
  most time-consuming part of the whole study. Budget for it.
- After the run, the raw file may have duplicate rows from retried failures —
  `analyze_pilot.py`'s dedup handles this (keep-last per model_name+item_id).

🔴 **GATE 2:** all model output CSVs exist in `results/`, each with a unique
filename and unique `model_name`. Sanity-check row counts and error rates before
analysis.

---

## PHASE 3 — Run the pre-registered analysis (execute §3/§6 of the pre-reg exactly)

**Step 3.1 — Clean results directory hygiene.** Ensure `results/` contains ONLY
the authoritative n=<N> files for this analysis. Move any pilot/intermediate CSVs
to a subdirectory (the loader skips subdirs), so no stale file with a colliding
`model_name` gets swept in.

**Step 3.2 — Run the analysis with an EXPLICIT model order** (do not rely on the
name-based rank heuristic — it mis-ranks the Mac 3B; pass the order explicitly):
```
python scripts/analyze_pilot.py \
    --results-dir results/ \
    --output-dir results/analysis_n<N>/ \
    --model-order "<explicit compute order, e.g. jetson-...q4,qwen2.5vl:3b,qwen2.5vl:7b,gemini-2.5-flash>"
```
(If `analyze_pilot.py` does not yet accept `--model-order` or emit
multiple-comparisons corrections, those two additions are pre-registered code
changes — see docs/PIPELINE_NOTES.md. Make them BEFORE running, and they must
match the frozen analysis plan.)

**Step 3.3 — The primary confirmatory test (RQ1 interaction).** This is the whole
reason for the scale-up. Run it exactly as pre-registered:
```
python scripts/interaction_test.py \
    --results-dir results/ \
    --model-a qwen2.5vl:3b --model-b qwen2.5vl:7b \
    --output results/analysis_n<N>/interaction.txt
```
(interaction_test.py is provided — it does the logistic regression + LR test.)

**Step 3.4 — Apply the pre-registered multiple-comparisons correction** across
the confirmatory family only (§3 of the pre-reg), and also report the
all-tests-corrected values for transparency.

🔴 **GATE 3:** results are in. Now — and only now — you learn whether the
interaction confirmed. Whatever it says, report it per §7 of the pre-registration.

---

## PHASE 4 — Write up

- Report the confirmatory result honestly (interaction significant or not).
- Include the "Deviations from Pre-Registration" subsection (even if empty).
- Frame pilot = exploratory, n=<N> = confirmatory replication+extension.
- Reproducibility package (README, requirements, LICENSE, this plan, the
  pre-reg, the frozen commit hash) goes public with the paper.
