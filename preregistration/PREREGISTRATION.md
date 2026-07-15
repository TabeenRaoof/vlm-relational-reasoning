# Pre-Registration: Compute-Sensitivity of Relational Reasoning in Vision-Language Models (Confirmatory Study)

> **STATUS: DRAFT — must be finalized and committed to git BEFORE running
> `curate_eval_set.py` at the new n, and BEFORE looking at any new results.**
> The whole point of this document is that it predates the confirmatory data.
> Fill every `<FILL>`, delete this banner, commit, record the commit hash in
> §9, and only then collect data.

---

## 0. Relationship to Prior Exploratory Work (read first — this is what keeps it honest)

An earlier study (n=300, "the pilot") was run and analyzed. It found:
- Projective-spatial accuracy dropped from the 7B to the 3B model (McNemar p=0.003,
  survives multiple-comparisons correction within the confirmatory family).
- Topological-containment accuracy did not detectably drop (p=0.79).
- BUT the formal category×model **interaction** test — the test of whether
  projective degrades *more than* containment — was **not significant**
  (logistic-regression interaction p=0.12; the study had only ~35% power to
  detect the observed effect).

**This document therefore treats the pilot as EXPLORATORY / hypothesis-generating,
and this new n=2000 study as CONFIRMATORY.** We are not pretending the pilot
did not happen. We are pre-registering the confirmatory test — with an a-priori
analysis plan and a sample size chosen by power analysis — before the new data
exists. Deviations from this plan, if any, will be reported as deviations.

This framing is the standard, defensible way to handle "we saw something
suggestive and now test it properly," and it must be stated plainly in the paper.

---

## 1. Research Questions and Hypotheses

**RQ1 (primary, confirmatory).** Does reduced model compute degrade projective-spatial
relational reasoning *more than* topological-containment reasoning in VLMs?

- **H1₀ (null):** The effect of model compute on accuracy does NOT differ between
  projective-spatial and topological-containment relations (no category×compute
  interaction).
- **H1₁ (alternative):** The compute effect is larger for projective-spatial than
  for topological-containment relations (a category×compute interaction, with
  projective more sensitive).

**RQ2 (confirmatory).** Does projective-spatial accuracy degrade from the higher-
compute to the lower-compute model?
- **H2₀:** No difference in projective accuracy between 3B and 7B.
- **H2₁:** Projective accuracy is lower for 3B than 7B.

**RQ3 (confirmatory, edge deployment / H3 in prior work).** Does running the same
model (identical weights, identical engine) on constrained edge hardware change
accuracy versus a desktop/cloud-simulated baseline?
- **H3₀:** No accuracy difference between Jetson and Mac at matched precision.
- **H3₁:** There is an accuracy difference.
- *(Note: prior work found H3₀ held at p=0.58; this is a confirmatory re-test at
  higher n. State direction of interest: we EXPECT to retain the null — a clean
  transfer result — but pre-register it as a two-sided test.)*

**RQ4 (secondary / exploratory — explicitly excluded from the confirmatory
family in §3).**
- The **inverted-prompt control** (`run_gemini_eval.py --invert-prompt`, Step 7
  bias diagnostic): does flipping the question polarity ("is the caption
  FALSE?", scored back into the same caption-truth space) change the yes/no
  response bias observed in the pilot (a NO-leaning bias / recall-specificity
  gap)? SECONDARY/EXPLORATORY.
- The **thinking-mode ablation** (`run_gemini_eval.py --thinking`): does
  leaving Gemini's reasoning ON (vs. the disabled-thinking default used for
  every confirmatory run) change accuracy? SECONDARY/EXPLORATORY.

No other pre-planned secondary analyses were found elsewhere in the repo for
this n=2000 cycle. Note: the pilot's Jetson Q4-vs-Q8 precision ablation (H2 in
`scripts/JETSON_DECISION_TREE_v2.md`) is **not** in scope for this scale-up —
`docs/RUN_PLAN.md` and `CLAUDE_CODE_INSTRUCTIONS.md` only specify a single
Q4_K_M Jetson arm at n=2000, so it isn't listed here as a planned RQ4 item.

---

## 2. Sample Size and Power Justification

- **Target total n:** 2000 (1000 projective-spatial + 1000 topological-containment),
  balanced ~50/50 True/False within each category. Confirmed achievable by
  `check_vsr_ceiling.py` (GATE 0.1, already cleared): pool sizes projective
  5753, containment 1554; max balanced 1508/category — comfortably above the
  1000/category needed. Full VERDICT block to be pasted into §9 at freeze.
- **Power analysis:** Monte Carlo simulation (2000 sims), assuming the pilot's
  observed interaction effect (log-odds ≈ 0.64) is the true effect, gives:
  n=1000→79%, n=1500→93%, **n=2000→98%** power to detect the interaction at α=0.05
  via the likelihood-ratio test. n=2000 chosen for ~98% power with margin to
  absorb multiple-comparisons correction.
- **If VSR ceiling < 2000:** pre-register the achievable n and its corresponding
  power from the table above; do not claim 2000 if the data cannot support it.

---

## 3. Confirmatory Test Family (THE key pre-specification)

The following, and ONLY the following, constitute the **confirmatory family**.
Multiple-comparisons correction (see §6) is applied within this family. Every
other comparison the pipeline produces is **exploratory** and will be labeled as
such in the paper.

| # | Test | Comparison | Method |
|---|------|-----------|--------|
| 1 | RQ1 interaction | correct ~ compute × category (3B vs 7B) | Logistic regression LR test on interaction term |
| 2 | RQ2 projective | 3B vs 7B, projective only | Paired McNemar |
| 3 | RQ2 containment (context) | 3B vs 7B, containment only | Paired McNemar |
| 4 | RQ3 edge | Jetson-3B-Q4 vs Mac-3B-Q4, all items | Paired McNemar |
| 5 | RQ3 edge, projective | Jetson vs Mac, projective | Paired McNemar |
| 6 | RQ3 edge, containment | Jetson vs Mac, containment | Paired McNemar |

The confirmatory family is closed at these 6 tests — no additional tests are
committed a priori.

**Everything NOT in this table is exploratory:** all Gemini pairwise comparisons,
the thinking-mode ablation, the inverted-prompt control, per-relation breakdowns,
RRS/DT/RCS retention metrics. These are reported without family-wise correction
and described as exploratory/descriptive.

---

## 4. Variables

- **Independent (manipulated):**
  - Model compute tier:
    - Qwen2.5-VL-3B — Ollama, Mac (desktop/cloud-simulated), Q4_K_M
    - Qwen2.5-VL-7B — Ollama, Mac (desktop/cloud-simulated), Q4_K_M
    - Gemini-2.5-Flash — Google Gen AI API (cloud)
    - Qwen2.5-VL-3B — Ollama, Jetson Orin Nano (edge), Q4_K_M
  - Relation category: projective_spatial vs topological_containment
  - Hardware (for RQ3): Mac (desktop, cloud-simulated) vs Jetson Orin Nano (edge)
- **Dependent:** per-item correctness (binary: model's parsed answer == ground_truth_label)
- **Controlled (held constant across all conditions):** the eval item set (same
  items via stable content-derived IDs), the system instruction, the user prompt
  template, the response parser, the quantization (Q4_K_M for the Mac-vs-Jetson
  comparison — weights verified byte-identical by SHA-256), decoding settings
  exactly as configured in the run scripts:
    - **Ollama, Mac and Jetson** (`run_ollama_eval.py`, `run_jetson_eval.py`):
      `temperature=0.0`, `num_predict=10` (Ollama's max-output-tokens
      equivalent) — the identical `options={...}` dict on both scripts by
      construction.
    - **Gemini** (`run_gemini_eval.py`), standard/confirmatory runs:
      `temperature=0.0`, `max_output_tokens=10`,
      `thinking_config=ThinkingConfig(thinking_budget=0)` (thinking explicitly
      disabled so it answers YES/NO directly, matching the single-word,
      no-thinking behavior of the Qwen tiers).
    - **Gemini, thinking-ablation arm** (exploratory, RQ4 only):
      `temperature=0.0`, `max_output_tokens=1024`, thinking left at its
      dynamic default (no `thinking_budget` override). The higher ceiling is
      required because thinking was empirically observed to consume ~980
      tokens before answering (per in-code note in `run_gemini_eval.py`);
      512 was observed to starve the answer.
    - **Gemini, inverted-prompt arm** (exploratory, RQ4 only): identical
      decoding settings to the standard run; only the prompt polarity changes
      (`SYSTEM_INSTRUCTION_INVERTED` / `USER_PROMPT_TEMPLATE_INVERTED`).
- **Confounding variables anticipated and controlled** (from lived pilot experience —
  state each and the mitigation):
  - *Prompt-framing artifact* — mitigated by the inverted-prompt control.
  - *Quantization mismatch* — mitigated by verifying identical weight digests
    across Mac and Jetson (SHA-256).
  - *Inference-engine mismatch* — mitigated by using Ollama on BOTH Mac and Jetson
    (same multimodal engine), NOT llama.cpp on one side.
  - *Sustained-load hardware instability on Jetson* — mitigated by the chunked-run
    + reboot protocol (see docs/jetson_setup.md); does not affect correctness,
    only completion.
  - `<FILL any others>`

---

## 5. Data Collection Procedure (summary — full runnable version in docs/RUN_PLAN.md)

1. Run `check_vsr_ceiling.py` → confirm achievable n. **(GATE)**
2. Finalize and **commit this pre-registration.** Record commit hash in §9. **(GATE)**
3. Run `curate_eval_set.py --n-per-category 1000 --seed 42` → the frozen eval set.
   Because IDs are content-derived and sampling nests, the pilot's items remain a
   strict subset — verify this (see RUN_PLAN.md).
4. Run each model over the frozen set (Mac 3B, Mac 7B, Gemini, Jetson 3B-Q4).
5. Run the pre-registered analysis (§3, §6) exactly as written.

---

## 6. Analysis Plan (exact, pre-committed)

- **Primary (RQ1):** logistic regression `correct ~ C(compute) * C(category)` on
  the 3B-vs-7B data; significance of the interaction via likelihood-ratio test
  (full vs additive model), α=0.05, two-sided.
- **McNemar tests (RQ2, RQ3):** paired, continuity-corrected, on items scoreable
  by both models in the pair.
- **Multiple-comparisons correction:** **Holm** (`method="holm"` in
  `statsmodels.stats.multitest.multipletests`) applied across the confirmatory
  family in §3.
  Report BOTH the family-wise-corrected p-values AND, for full transparency, the
  p-values under the maximally-conservative correction across all pipeline tests.
- **Exclusions:** items that error or are unparseable on a given model are dropped
  from that model's comparisons (report the count). No other exclusions.
- **Deviations:** any departure from this plan will be reported in a "Deviations
  from Pre-Registration" subsection, with rationale.

---

## 7. What Would Falsify / Change the Claim

- If the RQ1 interaction is **non-significant** even at n=2000 (98% power), we
  will NOT claim category-specific compute-sensitivity. We will report that
  projective degradation is real (RQ2) but that it does not differ significantly
  from containment — i.e., the pilot's suggestive pattern did not confirm.
- This is stated in advance so a null interaction is an honest, reportable
  outcome, not a failure to be worked around.

---

## 8. Ethics and Data Provenance

- Data: VSR (Visual Spatial Reasoning), `cambridgeltl/visual-spatial-reasoning`
  (GitHub) / `cambridgeltl/vsr_random` (Hugging Face; Liu, Emerson & Collier,
  TACL 2023). **License: Apache License 2.0** — verified directly against the
  repo's `LICENSE` file and the README's "License" section (this draft
  originally guessed CC-BY-4.0; that was wrong — corrected here). Apache-2.0
  §4 permits reproduction and distribution of the Work or Derivative Works
  (a curated subset qualifies), provided we: (a) include a copy of the
  License, (b) note that files were modified/subsetted, and (c) retain
  existing copyright/attribution notices. **Redistributing our curated
  n=2000 subset (item IDs, captions, ground-truth labels, relation/category
  tags, and COCO `image_url` links) is permitted** under these terms, as long
  as we ship the Apache-2.0 license text and attribute the original VSR
  authors alongside the release. We do not redistribute image bytes — only
  `image_url` references to `images.cocodataset.org` — so COCO/Flickr's own
  image licensing (a separate question we do not need to resolve) is not
  implicated. No human subjects, no PII collected.
- Compute: local hardware + Gemini API (paid). No sensitive data sent to the API
  beyond public VSR images/captions.
- `<FILL any IRB/approval note if your institution requires one — likely N/A>`

---

## 9. Freeze Record (fill at commit time)

- Pre-registration finalized (UTC): `<FILL>`
- Git commit hash of THIS file at freeze: `<FILL>`
- `check_vsr_ceiling.py` output (paste the VERDICT block): `<FILL>`
- Confirmed target n: `<FILL>`
- Curation command + seed used: `<FILL>`
- (Optional but recommended) public timestamp (OSF registration URL / arXiv): `<FILL>`
