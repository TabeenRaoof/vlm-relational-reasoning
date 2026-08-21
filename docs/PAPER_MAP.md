# Paper → Repository Map

Every quantitative claim in *Compute Scale, Relation Type, and Edge Deployment in
Vision–Language Spatial Reasoning* (TMLR submission), mapped to the script that
produces it and the output file that holds it.

All values below were **recomputed from the committed per-item result files** and
checked against the submitted PDF. Where a value differs from the paper by more
than rounding, it is called out explicitly in the last section — nothing is
quietly adjusted.

Reproduce everything: `bash scripts/reproduce_paper.sh`

---

## Section 4 — Method

| Claim | Source | Value |
|---|---|---|
| VSR 10,972 items, 66 relations | `data/vsr_*.jsonl` | as stated |
| Projective pool 5,753 / max balanced 5,586 | `scripts/check_vsr_ceiling.py` | as stated |
| Containment pool 1,554 / max balanced 1,508 | `scripts/check_vsr_ceiling.py` | as stated |
| Max balanced two-category sample n≈3,016 | `scripts/check_vsr_ceiling.py` | as stated |
| Pilot nests inside the confirmatory set | `data/pilot_eval_set_n150.csv` ⊂ `data/eval_set_n2000.csv` | verified True |
| 1,337/1,337 images fetched, 0 failures | `results/image_fetch_failures_n2000.csv` | 0 rows |

## Section 5.1 — Study 1, exploratory pilot (Table 2)

Script: `scripts/analyze_pilot.py` → `results/analysis_full/accuracy_table.csv`

| Model | Overall | Projective | Containment |
|---|---|---|---|
| Qwen2.5-VL-3B | 76.7% | 70.7% | 82.7% |
| Qwen2.5-VL-7B | 82.3% | 82.0% | 82.7% |
| Gemini-2.5-Flash | 81.7% | 77.3% | 86.0% |

Pilot interaction χ²(1)=2.39, p=0.122, effect 0.637 log-odds, ~35% power.

## Section 5.2 — Study 2, confirmatory (Table 3)

Script: `scripts/interaction_test.py` → `results/analysis_n2000/interaction_primary.txt`
Script: `scripts/analyze_pilot.py` (confirmatory family) → `results/analysis_n2000/mcnemar_confirmatory.txt`

| Claim | Value | Status |
|---|---|---|
| Projective McNemar, b=56, c=109 | p = 5.161×10⁻⁵ | reproduced |
| Containment McNemar, b=58, c=66 | p = 0.530 | reproduced |
| Pre-registered interaction LRT | χ²(1)=2.6675, p=0.1024 | reproduced |
| Cell accuracies (Figure 1b) | 76.6 / 82.8 / 81.9 / 83.6 | reproduced |
| Holm across the 6-test family | only projective survives | reproduced |

## Section 5.2.1–5.2.2 — Corrected analyses (Table 4)

Script: `scripts/h1_corrected_analyses.py` → `results/analysis_paper/h1_corrected.txt`

| Sample | DiD (pp) | Logistic | GEE |
|---|---|---|---|
| Pooled (n=2,000) | +4.50 [1.18, 7.82] | p = 0.102 | p = 0.017 |
| New only (n=1,700) | +3.29 [−0.31, 6.90] | p = 0.272 | p = 0.112 |

- DiD p (pooled) = 0.0079; image-level cluster bootstrap over **1,674 distinct
  images**, 95% CI [+1.10, +7.88], p ≈ 0.010 (paper: [1.11, 7.84], p ≈ 0.009 —
  bootstrap RNG noise, seed fixed at 20260716).
- New-items-only McNemar: projective p = 0.0027, containment p = 0.50.
- New-items-only interaction effect ≈ 0.196 log-odds (paper: ≈0.20).

## Section 5.2 — Leave-one-relation-out robustness

Script: `scripts/robustness_checks.py` → `results/analysis_paper/robustness.txt`

Dropping each of the **21** projective relations in turn never raises p above
**2.414×10⁻⁴** (worst case: dropping `left of`). Paper: "never above 2.4×10⁻⁴".

## Section 5.3 — Study 3, edge equivalence (Table 5)

Script: `scripts/h3_equivalence_test.py` → `results/analysis_h3_n2000/equivalence.txt`

| Quantity | Value |
|---|---|
| Paired scoreable items | 1,996 (4 excluded, Jetson crashes) |
| Observed difference (Jetson − Mac) | +0.80 pp |
| 90% CI | [+0.08, +1.52] pp |
| TOST upper p at ±3 pp | 2.357×10⁻⁷ |
| Secondary McNemar | p = 0.0853 |

Margin and end-to-end sensitivity — `scripts/robustness_checks.py`:

| Margin | upper-tail p | Equivalence |
|---|---|---|
| ±3.0 pp | 2.357×10⁻⁷ | declared |
| ±2.0 pp | 0.00302 | declared |
| ±1.5 pp | 0.0548 | not declared |
| ±1.0 pp | 0.325 | not declared |

End-to-end (4 device failures scored **incorrect**, full n=2,000):
**+0.70 pp, p = 9.35×10⁻⁸** at ±3 pp. Paper: 9.4×10⁻⁸.

## Section 5.5 — Shrinkage and power (Table 6)

Script: `scripts/power_projection.py` → `results/analysis_paper/power_projection.txt`

Confirmatory estimate 0.2665 log-odds = **41.8%** of the pilot's 0.637.

| n | Power (true effect = 0.267 log-odds) |
|---|---|
| 2,000 | ~37% (this design) |
| 3,016 (VSR max) | ~52% |
| 6,000 | ~80% |
| 8,000 | ~90% |

Monte Carlo, cell probabilities fixed at observed values, seed 20260716.
Values move ±1 pp between seeds; the paper marks them approximate.

## Section 6 — Operational findings (Table 7)

Script: `scripts/latency_analysis.py` → `results/analysis_paper/latency.txt`

| Condition | Latency | Throughput |
|---|---|---|
| Mac M2 Max | 2.79 s | ~1,292/hr |
| Jetson, healthy | ~8 s (see note) | ~450/hr |
| Jetson, degraded | 57.95 s | ~62/hr (21×) |

- Regime transition detected at inference row **295**; **1,699 items (~85% of
  the run)** follow it — the paper's "~1,700 items, ~20 chunks".
- 4 of 2,000 items (0.2%) reproducibly crash the device; skip-list is in
  `scripts/run_jetson_eval.py::KNOWN_POISON_ITEM_IDS`.
- Per-chunk model-load cost (33 observations near 110 s) is reported separately
  and excluded from both regime means.

## Figures

Script: `scripts/make_paper_figures.py` → `results/figures/`

| Figure | File | Content |
|---|---|---|
| 1 | `figure1_accuracy_by_category.{pdf,png}` | pilot vs confirmatory, by category |
| 2 | `figure2_tost_equivalence.{pdf,png}` | TOST 90% CI vs ±3 pp margin |
| 3 | `figure3_shrinkage_and_power.{pdf,png}` | effect shrinkage; power curve |
| 4 | `figure4_jetson_latency.{pdf,png}` | latency trace; bimodal distribution |

Every plotted value is recomputed from the raw CSVs at render time, so a figure
cannot drift from the statistic it illustrates. The two-series orange/blue pair
was checked for colorblind separation rather than chosen by eye (ΔE 18.1 protan,
29.2 tritan, against a ΔE≥8 target).

---

## Known differences from the submitted PDF

Only one, and it is a rounding/characterization choice, not a disagreement:

1. **Table 7, Jetson healthy row.** The paper reports ~8 s / ~450 per hour,
   characterizing the healthy regime by the upper end of its ~1–8 s spread. The
   computed mean of that regime is **6.34 s** (~568/hr, 2.3× the Mac). Both are
   marked approximate in the paper, and the row is descriptive — no test or
   conclusion depends on it. The Mac (2.79 s) and degraded (57.95 s → 62/hr,
   21×) rows are exact computed means.

Everything else agrees to rounding, including all p-values, confidence
intervals, effect sizes, and power projections. The bootstrap CI and Monte
Carlo power values vary in the last digit with the RNG seed, which is fixed in
the scripts for reproducibility.

## Reproducibility gaps (paper Appendix C, unchanged)

- Exact Ollama/CUDA/OS package versions were not exhaustively logged.
- The relation taxonomy was single-coded; no inter-rater figure exists.
- Temperature-0 repeatability was assumed from configuration, not re-verified.
- No archival, DOI-bearing snapshot of this repository exists yet; one is
  recommended before external citation.
