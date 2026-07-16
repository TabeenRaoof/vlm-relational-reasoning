# Pre-Registration: Edge-Hardware Accuracy Equivalence for On-Device VLM Spatial Reasoning (H3, n=2000)

## 0. Relationship to Prior Work (read first — honesty about what we've already seen)

An earlier study (n=300) compared the same VLM (Qwen2.5-VL-3B, Q4_K_M) running on
edge hardware (NVIDIA Jetson Orin Nano 8GB) versus a desktop Mac baseline, at
byte-identical weights (SHA-256 verified) and the same inference engine (Ollama on
both). It found no significant accuracy difference (paired McNemar p=0.58; observed
difference +1.0pp, 95% CI [−1.4, +3.4]pp).

That result used a **difference test**, whose non-significance is weak evidence —
"we failed to find a difference" is not the same as "the two are equivalent." This
study re-tests H3 at n=2000 using a **formal equivalence test (TOST)**, which lets
us make the stronger, positive claim we actually care about: *edge deployment
changes accuracy by less than a pre-specified, practically-negligible margin.*

**Honesty note (optional-stopping):** we have already seen the n=300 result. This
is a confirmatory re-test at higher n with a pre-committed design and margin,
framed transparently. The n=300 result will be reported alongside. The design,
margin, and analysis below are frozen before the n=2000 Jetson data is collected.

## 1. Research Question and Hypotheses

**RQ (confirmatory).** Does running Qwen2.5-VL-3B (Q4_K_M) on the Jetson Orin Nano
edge device produce accuracy *equivalent* (within ±3 percentage points) to running
the identical model on the Mac desktop baseline, on the same items?

This is an **equivalence** framing, so the hypotheses are the TOST pair:
- **H0 (non-equivalence):** the true Jetson−Mac accuracy difference is ≥ +3pp OR
  ≤ −3pp (i.e., outside the equivalence margin).
- **H1 (equivalence):** the true Jetson−Mac accuracy difference lies strictly
  within ±3pp.

Equivalence is declared only if BOTH one-sided tests reject H0 at α=0.05.

## 2. The Equivalence Margin (the key pre-specified value)

**Margin = ±3 percentage points**, on the paired accuracy difference (Jetson − Mac).

Justification: a ≤3pp accuracy change is below the threshold at which edge
deployment decisions would change — it is well within run-to-run and
quantization-level noise for this class of model, and the n=300 point estimate
(~1pp) sits comfortably inside it. This margin is chosen a priori and is not
adjustable after seeing the n=2000 data.

## 3. Sample Size and Power

- **Target n:** 2,000 items (the existing frozen `data/eval_set_n2000.csv`, which
  nests the n=300 pilot). After expected ~1–2% unparseable/error exclusions,
  ~1,970 scoreable paired items.
- **Power (TOST, ±3pp margin, Monte Carlo):**
  - if true diff ≈ 1pp (the n=300 estimate): **~99% power** to declare equivalence.
  - if true diff = 0: ~100%.
  - if true diff ≈ 1.5pp: ~93%.
  - if true diff ≈ 2pp: ~68% (see §7 — a larger-than-expected difference may fail
    to declare equivalence even while remaining practically small).

## 4. Variables

- **Independent (manipulated):** hardware — Jetson Orin Nano (edge) vs. Mac desktop.
- **Dependent:** per-item correctness (binary; parsed answer == ground_truth_label).
- **Held constant (this is what makes it a clean hardware-only test):**
  - Model weights — byte-identical, verified by SHA-256 digest match between the
    Jetson and Mac Ollama blobs. **Mac side re-confirmed for this study:**
    `qwen2.5vl:3b` weight-blob digest =
    `sha256:e9758e589d443f653821b7be9bb9092c1bf7434522b70ec6e83591b1320fdb4d`
    (3,200,614,720 bytes; matches the pilot's recorded prefix). **Jetson side
    re-confirmed live for this study (2026-07-16),** via SSH, from the manifest at
    `/mnt/ssd/ollama-models/manifests/registry.ollama.ai/library/qwen2.5vl/3b-q4_K_M`
    on the Jetson (10.0.0.21): model-layer digest =
    `sha256:e9758e589d443f653821b7be9bb9092c1bf7434522b70ec6e83591b1320fdb4d`
    (3,200,614,720 bytes) — **matches the Mac digest exactly** (full hash and byte
    count). GATE 1.3 cleared: weights are byte-identical on both machines, which is
    what makes this a clean hardware-only comparison.
  - Quantization: Q4_K_M on both.
  - Inference engine: Ollama on both (same multimodal engine — NOT llama.cpp on
    either side, which would add an engine confound).
  - Prompt (system + user template), response parser, decoding settings
    (temperature=0.0, num_predict=10) — identical to the Mac run.
  - The item set: the exact same `data/eval_set_n2000.csv`, same items.
- **Confounds anticipated and controlled:**
  - *Engine/weight mismatch* → eliminated by the digest + same-engine controls above.
  - *Jetson sustained-load instability (OOM under long runs)* → mitigated by the
    chunked-run + reboot-per-chunk protocol (docs/jetson_setup.md); this affects
    run *completion*, not per-item correctness, and does not bias the comparison.
  - *Item-level failures (unparseable/OOM-killed items)* → handled as exclusions
    (§6), reported with counts; only items scoreable on BOTH Jetson and Mac enter
    the paired test.

## 5. Procedure (summary; full runnable version in RUN_PLAN_H3.md and docs/jetson_setup.md)

1. Freeze and commit this pre-registration. **(GATE)**
2. Confirm the Mac n=2000 3B result already exists
   (`results/mac_qwen25vl_3b_n2000.csv` — already collected).
3. Run Qwen2.5-VL-3B (Q4_K_M) over `data/eval_set_n2000.csv` on the Jetson, using
   the chunked-reboot protocol, producing `results/jetson_qwen25vl_3b_q4_n2000.csv`.
4. Dedup the Jetson output (keep-last per model_name+item_id) and run the
   pre-registered TOST analysis (§6).

## 6. Analysis Plan (exact, pre-committed)

- **Pairing:** restrict to items scoreable (parseable, non-errored) on BOTH the
  Jetson n=2000 run and the Mac n=2000 3B run. Report the excluded count and its
  per-category breakdown.
- **Primary test — TOST equivalence, ±3pp margin, α=0.05:** on the paired accuracy
  difference (Jetson − Mac). Declare equivalence iff both one-sided tests reject.
  Report the point estimate, its 90% CI (the CI corresponding to the TOST at
  α=0.05 — note it is the 90% CI, not 95%, that maps to two one-sided 0.05 tests),
  and the two one-sided p-values.
- **Secondary (context, not the primary claim):** the paired McNemar difference
  test (for continuity with the n=300 result) and the per-category (projective,
  containment) breakdown. Labeled secondary/descriptive.
- **Deduplication:** the Jetson raw file may contain duplicate rows from
  chunked/retried runs; dedup keep-last per (model_name, item_id) before analysis
  (as `analyze_pilot.py` already does).
- **Exclusions:** items that error or are unparseable on either side are dropped
  from the paired test; count + per-category rate reported. No other exclusions.
- **Deviations:** any departure from this plan reported in a "Deviations from
  Pre-Registration" subsection.

## 7. What Each Outcome Means (pre-committed interpretation)

- **Equivalence declared (both one-sided tests reject, diff within ±3pp):** we
  conclude accuracy transfers cleanly from Mac to Jetson within a ≤3pp margin — a
  strong, bounded, positive edge-deployment result.
- **Equivalence NOT declared because the difference is genuinely larger than ±3pp:**
  we report that edge deployment produces a measurable accuracy change exceeding
  our negligibility margin, and characterize its size and direction — a
  substantive (if less clean) finding.
- **Equivalence NOT declared due to insufficient precision (CI straddles the
  margin despite a small point estimate):** possible if the true difference is
  ~2pp (§3 shows ~68% power there). We report the point estimate and CI honestly
  and state that equivalence could not be formally established at this n, while
  noting the difference remains practically small. We do NOT collect more data to
  force a declaration (that would be optional stopping).

## 8. Ethics and Data Provenance

- Data: VSR (Apache-2.0; curated subset redistributed as IDs/captions/labels/
  category tags/COCO image_url references, not image bytes). No human subjects,
  no PII. Same determination as the n=2000 interaction study's pre-registration:
  secondary public benchmark data, no IRB review required (45 CFR 46 §102(e)).
- Compute: local Mac + Jetson only; no external API calls in this study (no Gemini).

## 9. Freeze Record (fill at commit time)

- Pre-registration finalized (UTC): `2026-07-16T07:43:28Z`
- Git commit hash of this file at freeze: `72d65c5` (short form; recorded per
  RUN_PLAN_H3.md Step 0.3's own prescribed sequence: commit → copy hash → amend
  to record it. Note the inherent self-reference limit this process accepts —
  writing a hash into the file and then amending necessarily changes the
  commit's actual hash again, so this value is one trivial metadata-only amend
  behind the true final hash. The authoritative record is `git log --oneline`
  on this branch: the freeze commit is titled "Freeze H3 n=2000 equivalence
  pre-registration (pre-data)" and is the sole commit directly after "Rename
  project to vlm-relational-reasoning.")
- Confirmed margin: ±3pp
- Confirmed target n: 2000 (existing frozen eval set)
- Mac baseline file used: results/mac_qwen25vl_3b_n2000.csv
- (Optional) public timestamp (OSF/arXiv): (none)
