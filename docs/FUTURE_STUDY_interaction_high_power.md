# Future Study Plan — High-Powered Test of the Compute × Category Interaction

**Status:** PLANNING ONLY. This is not a pre-registration and not a run plan. It
is the design brief for a separate, self-contained future study. Nothing here is
frozen; every open question below must be resolved *before* a pre-registration is
written, and the pre-registration must be committed *before* any new data for this
study is collected.

**This study is completely independent of the Jetson H3 (n=2000) work.** Different
question, different data, different design, different pre-registration. Do not
merge the two.

---

## 1. Why this study exists (the motivation, stated honestly)

An exploratory pilot (n=300) suggested that reducing model compute (7B → 3B) hurts
projective-spatial reasoning more than topological-containment reasoning — a
compute × category **interaction**. A first confirmatory study (n=2000, VSR,
pre-registered) tested this and found the interaction **non-significant**
(likelihood-ratio p = 0.10).

Crucially, that null is **not** evidence the interaction is absent. The pilot had
overestimated the effect (winner's curse): the true effect, estimated from the
n=2000 data, is ~0.27 log-odds — less than half the pilot's 0.64. At that real
effect size, the n=2000 study had only **~37% power**, so a non-significant result
is genuinely ambiguous ("no effect" vs. "real effect, underpowered" cannot be
distinguished).

This study is designed to resolve that ambiguity: reach **>90% power** for the
real effect size, so that whatever result comes back — significant interaction, or
a well-powered null — is *definitive* and publishable either way.

## 2. The honest methodological framing (READ THIS — it is what keeps the study valid)

We have already seen the n=2000 interaction result. Collecting more data now, if
done carelessly, is **optional stopping / sample-size ratcheting** — a classic way
to invalidate a p-value ("we collected until it was significant"). This study
avoids that trap by being explicit and pre-committed:

- It is framed as a **new, separate confirmatory study**, powered at a revised
  effect-size estimate (~0.27 log-odds) that came from the n=2000 data — not a
  quiet extension of the previous study.
- The n=2000 non-significant result **will be reported honestly** in any writeup;
  nothing is hidden.
- The stopping rule (whether fixed-n or group-sequential — see §5) is
  **pre-specified and frozen** before any new data is collected.
- Whatever the result, it is reported. No "collect until significant."

Done this way it is a principled powered replication, which is strong science. Any
deviation from this framing risks the study's credibility — do not shortcut it.

## 3. Sample size target

Power analysis (Monte Carlo, on the real n=2000 effect size of ~0.27 log-odds):

| Total n | Per category | Power (real effect) |
|---------|--------------|---------------------|
| 2,000   | 1,000        | ~37% (the study just run) |
| 3,016   | 1,508        | ~52% (VSR maximum) |
| 6,000   | 3,000        | ~80% |
| 8,000   | 4,000        | ~90% |
| 10,000  | 5,000        | ~97% |

Target for >90% power: **n ≈ 8,000 total (4,000 per category)**, possibly up to
10,000 for margin. The existing 2,000 VSR items nest inside this (content-derived
IDs), so they are reused, not re-collected — but see the ceiling problem in §4.

## 4. THE BINDING CONSTRAINT — a second dataset is mandatory, and it must have containment

**VSR alone cannot reach n=8000.** The `check_vsr_ceiling.py` run showed VSR's
**topological-containment** pool maxes at **1,508 balanced items** — the whole
dataset caps at ~3,016 total balanced. Projective is not the problem (VSR has
5,586 available); **containment is the bottleneck.**

Therefore this study **requires a second dataset**, and the second dataset must
supply a large number of **containment** items (roughly 2,500+ additional balanced
containment items to reach 4,000/category). A projective-only second dataset does
not help — it would just deepen the projective side while containment stays stuck
at 1,508.

### OPEN QUESTION 4a — which second dataset, and does it have containment?
Code comments in the current repo reference **What'sUp**. BUT it is unverified
whether What'sUp contains topological-containment relations at all. Many spatial-
reasoning datasets (What'sUp among the likely candidates) are heavily *projective*
(on/under/left/right/behind) and may have little or no containment ("in", "inside",
"contains", "enclosed by"). **If the chosen second dataset is projective-only, it
does not solve the binding constraint and the whole n=8000 target is infeasible
from it.**
→ **First action of this study: verify the containment coverage of candidate
  second datasets before committing.** Write a `check_<dataset>_ceiling.py` analog
  to the VSR one. Candidate datasets to evaluate for containment coverage:
  What'sUp, VG-Relation / ARO, GQA (has rich relations incl. containment), COCO
  spatial relations, Visual Genome directly. GQA and Visual Genome are the most
  likely to have real containment volume — worth checking first.

### OPEN QUESTION 4b — taxonomy mapping / confound risk
Whatever second dataset is chosen, its relation vocabulary must be mapped onto the
*exact same* projective / containment definitions used for VSR. If the second
dataset operationalizes a category even slightly differently, mixing it with VSR
introduces a dataset × category confound — you'd be testing "does compute affect
categories differently" while the categories themselves differ by source.
→ Requires an explicit, frozen **relation-to-category mapping table** in the
  pre-registration, plus a validation check that second-dataset items behave
  consistently with VSR items *within* each category (e.g., compare per-dataset
  accuracy within category at a fixed model — large gaps flag a mapping problem).

### OPEN QUESTION 4c — dataset licensing
VSR is Apache-2.0 (verified). The second dataset's license must be checked for
redistribution of a curated subset, same as was done for VSR. GQA, Visual Genome,
What'sUp each have their own terms — verify before building a public reproducible
release around them.

## 5. Design choice — fixed-n vs. group sequential (the early-stopping question)

The user asked whether the study can "stop as soon as an interaction is detected."
Yes — but ONLY via a properly pre-specified **group sequential design**, never
naive repeated peeking (which inflates false positives from 5% to 20-30%+ and is
p-hacking).

### OPEN QUESTION 5a — adopt a group sequential design?
A group sequential design allows pre-planned interim analyses with the option to
stop early for significance, while controlling the overall false-positive rate via
an **alpha-spending function**. Benefit: if the interaction is real and near the
estimated size, the study may stop well before the full n=8000 — saving substantial
data-collection cost. Cost: more complex to set up; each interim look slightly
raises the maximum n needed; boundaries must be computed in advance.
→ Decision needed: fixed-n (simpler, collect all 8000, analyze once) vs. group
  sequential (can stop early, more setup). Given the collection cost, group
  sequential is likely worth it.

### OPEN QUESTION 5b — alpha-spending function and number of looks
If group sequential: how many interim analyses (e.g., looks at n=4000, 6000, 8000?)
and which spending function?
- **O'Brien-Fleming** (recommended): spends very little alpha early, so you only
  stop early on *very* strong evidence; preserves near-full alpha for the final
  look. Conservative — best when you'd rather not stop on a marginal early result.
- **Pocock**: spends alpha evenly across looks; easier to stop early but each look
  uses a stricter constant threshold and the final look is less powerful.
→ Recommend O'Brien-Fleming. Boundaries computed via `rpact` (R) or a Python
  sequential-analysis package, and frozen in the pre-registration.

### OPEN QUESTION 5c — interleaved collection order (sequential-specific)
If using interim analyses AND two datasets, the order items are collected/scored
matters. If early looks are all-VSR and later looks introduce What'sUp/GQA, then
"interim look" is confounded with "dataset." → Pre-specify an **interleaved**
collection/scoring order so each interim analysis contains a representative mix of
both datasets. Must be defined in the pre-registration.

## 6. Models to run

At minimum the compute contrast that defines the interaction:
- Qwen2.5-VL-3B (Q4_K_M, Ollama)
- Qwen2.5-VL-7B (Ollama)

### OPEN QUESTION 6a — is Jetson in scope for this study?
The interaction (RQ1) is Mac-only by design (3B vs 7B; no Jetson-7B exists). So
Jetson is NOT needed for the interaction test itself. Only include Jetson if this
study also wants to re-examine H3 at the larger n — but H3 is being handled
separately (the n=2000 Jetson study). **Recommendation: exclude Jetson from this
study entirely** to avoid the multi-hour OOM-chunking cost for a hypothesis this
study isn't about.

### OPEN QUESTION 6b — is Gemini in scope?
Gemini was excluded from the pilot's confirmatory interaction family (different
architecture confounds scale with architecture). Same logic applies here — the
interaction test should stay 3B-vs-7B (same family, scale-only). Gemini could be a
separate descriptive/exploratory arm but should NOT enter the confirmatory
interaction test. → Recommend: exclude from the confirmatory test; optional as
exploratory only.

## 7. Analysis (carried forward from the current study's methodology)

- Primary: logistic regression `correct ~ C(compute) * C(category)`, interaction
  tested via likelihood-ratio test (full vs. additive). Same as `interaction_test.py`.
- If group sequential: apply the pre-specified alpha-spending boundaries at each
  interim look.
- Report the effect size and its CI, not just the p-value.
- Multiple comparisons: this study has essentially ONE confirmatory test (the
  interaction), so the multiple-comparisons burden is minimal — but any secondary
  per-category tests should be labeled exploratory.

## 8. Cost estimate (rough, for planning)

- Data collection: ~6,000 NEW items (8,000 − 2,000 existing) × 2 Mac models. Mac
  runs at ~3s/item → ~5 hours per model, ~10 hours total Mac compute. Free.
- Image fetching: potentially several thousand new images (COCO/VG/GQA source).
- Second-dataset curation + taxonomy mapping: real human/analysis time, not compute.
- NO Jetson (per §6a) → avoids the worst cost.
- NO Gemini in confirmatory test (per §6b) → avoids API cost, unless exploratory.
- Group-sequential early stopping (if adopted) may cut the data collection
  substantially if the effect is real.

## 9. Ordered next steps when this study is picked up

1. **Resolve OPEN QUESTION 4a first** — pick candidate second datasets, write a
   containment-coverage ceiling check for each, confirm one can supply ~2,500+
   containment items mapping cleanly to the VSR taxonomy. If none can, the study
   is infeasible as designed — rethink before proceeding.
2. Resolve 4b (taxonomy mapping table), 4c (license), 5a/5b/5c (design + spending +
   interleaving), 6a/6b (Jetson/Gemini scope).
3. Compute the sequential boundaries (if group sequential) and finalize the exact
   max n from a fresh power analysis on the combined-dataset effect estimate.
4. Write the pre-registration (reuse the structure of the current study's
   PREREGISTRATION.md; add the sequential design section and the taxonomy mapping
   table).
5. Freeze/commit the pre-registration BEFORE collecting any new data.
6. Collect (interleaved), run interim analyses per the frozen schedule, stop per
   the frozen rule, report whatever results.

## 10. Summary of all open questions to resolve before the pre-registration

- **4a** Which second dataset, and does it actually have enough containment items?
  (THE gating question — verify empirically, don't assume What'sUp works.)
- **4b** Exact relation-to-category mapping for the second dataset + a within-
  category consistency validation.
- **4c** Second dataset license terms for redistributing a curated subset.
- **5a** Fixed-n or group sequential?
- **5b** If sequential: number of interim looks + alpha-spending function
  (recommend O'Brien-Fleming).
- **5c** If sequential + two datasets: interleaved collection order.
- **6a** Confirm Jetson excluded (recommended).
- **6b** Confirm Gemini excluded from the confirmatory test (recommended).
- Final exact n after a fresh power analysis on the combined-dataset effect size.
