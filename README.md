# Compute Scale, Relation Type, and Edge Deployment in Vision–Language Spatial Reasoning

Code, frozen evaluation sets, pre-registrations, per-item results, and analysis
for the paper of the same name.

The paper asks what kind of spatial reasoning survives two independent axes of
compute reduction — shrinking the model (7B → 3B) and moving inference from a
desktop to an 8 GB edge device — using Visual Spatial Reasoning (VSR) split into
**projective-spatial** (viewpoint-dependent: *left of*, *above*) and
**topological-containment** (viewpoint-invariant: *inside of*, *contains*)
relations.

**Start here:** [`docs/PAPER_MAP.md`](docs/PAPER_MAP.md) maps every quantitative
claim in the paper to the script that produces it and the file that holds it.

## Headline results

| | Result |
|---|---|
| **Projective vs containment (3B→7B)** | Projective degrades decisively (McNemar p ≈ 5.2×10⁻⁵); containment shows no detectable change (p = 0.53) |
| **The formal interaction (H1)** | Mixed. Pre-registered test non-significant (p = 0.102); corrected for item pairing p = 0.017; on data independent of the pilot p ≈ 0.11. A power problem, not a demonstrated null. |
| **Edge equivalence (H3)** | Equivalence declared within the pre-registered ±3 pp margin: +0.80 pp, 90% CI [0.08, 1.52], TOST p = 2.4×10⁻⁷ (in fact holds to ±2 pp) |
| **The real edge cost** | ~21× slower in the sustained regime, 8× context reduction to load at all, 0.2% of inputs hard-fail. Not accuracy. |
| **Quantization by scale (H2)** | Untestable as designed — Q8 will not fit in 8 GB. Reported as a finding. |

## Reproducing the paper

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

bash scripts/reproduce_paper.sh
```

This recomputes every statistic and figure in the paper from the committed
per-item result files (~2 minutes). It does **not** re-run inference — model
evaluation is the expensive, hardware-bound step, and its outputs are committed
under `results/`.

To re-run inference from scratch instead, fetch the images first
(`scripts/fetch_eval_images.py`, ~1,337 COCO images by URL) and then use the
eval harnesses (`run_ollama_eval.py`, `run_gemini_eval.py`,
`run_jetson_eval.py`). Read [`docs/RUN_PLAN.md`](docs/RUN_PLAN.md) and
[`docs/jetson_setup.md`](docs/jetson_setup.md) first — the Jetson run in
particular requires a chunked, reboot-between-chunks protocol.

## The three studies

Full detail in [`docs/STUDIES_INDEX.md`](docs/STUDIES_INDEX.md).

| # | Study | n | Pre-registered | Paper section |
|---|---|---|---|---|
| 1 | Exploratory pilot (hypothesis-generating) | 300 | No, by design | §5.1 |
| 2 | Confirmatory interaction test | 2,000 | Yes (frozen at `b176ed7`) | §5.2 |
| 3 | Edge equivalence, TOST | 2,000 | Yes (frozen at `792eb12`) | §5.3 |
| 4 | High-powered re-test | — | **Not started** — design brief only | §8 |

Both pre-registrations were frozen in version control and pushed **before any
confirmatory data existed**. Study 4 deliberately has no data: resolving H1
needs n ≈ 6,000–8,000, which exceeds VSR's containment ceiling (1,508 balanced
items), so it requires a second dataset *and* a fresh pre-registration. Adding
items under the existing pre-registration would be optional stopping.

## Directory layout

```
data/                        frozen evaluation sets + VSR splits (no images; see below)
  eval_set_n2000.csv           the confirmatory item set (Studies 2 and 3)
  pilot_eval_set_n150.csv      the n=300 pilot set — a strict subset of the above
  vsr_*.jsonl                  source VSR splits

preregistration/             frozen pre-registrations (do not edit; history is the point)

scripts/
  curate_eval_set.py           builds the frozen, content-hashed, nesting item sets
  check_vsr_ceiling.py         measures the containment ceiling that bounds the design
  fetch_eval_images.py         downloads COCO images by URL
  run_{ollama,gemini,jetson}_eval.py   the three eval harnesses
  analyze_pilot.py             Study 1 + the pre-registered confirmatory test family
  interaction_test.py          the pre-registered primary interaction test (RQ1)
  h3_equivalence_test.py       the pre-registered TOST (RQ3)
  h1_corrected_analyses.py     GEE, DiD, cluster bootstrap, pilot/new split (§5.2.1-2)
  robustness_checks.py         leave-one-relation-out, TOST margins, end-to-end (§5.2-3)
  power_projection.py          shrinkage + Monte Carlo power (§5.5)
  latency_analysis.py          regimes and throughput (§6)
  make_paper_figures.py        Figures 1-4
  reproduce_paper.sh           runs all of the above in order

results/
  *_n2000.csv                  per-item results, confirmatory runs
  pilot_n300/                  per-item results, pilot runs (isolated, see its README)
  analysis_full/               Study 1 output
  analysis_n2000/              Study 2 output
  analysis_h3_n2000/           Study 3 output
  analysis_paper/              corrected, robustness, power, latency
  figures/                     Figures 1-4 (pdf + png)

docs/                        PAPER_MAP, STUDIES_INDEX, run plans, pipeline notes,
                             Jetson setup, and the Study 4 design brief
```

## What is deliberately not in this repository

- **The evaluation images.** `data/eval_set_n2000.csv` carries item ids,
  captions, labels, and image URLs; the COCO images themselves are fetched at
  run time by `scripts/fetch_eval_images.py`. This follows VSR's license rather
  than redistributing the image set.
- **API keys.** `.env` is gitignored; `.env.example` documents the shape.
- **The virtual environment**, caches, and OS cruft.

## Known limitations

Stated in full in the paper (§7, Appendix C) and not softened here: the relation
taxonomy is single-coded with no inter-rater figure; the equivalence comparison
is device-as-configured rather than hardware-only (context length and KV-cache
precision differ, both forced by the 8 GB ceiling); scope is one benchmark, one
model family, one quantization level, one device; Gemini figures are pilot-only;
and exact Ollama/CUDA/OS package versions were not exhaustively logged.

## Citation

Tabeen Raoof. *Compute Scale, Relation Type, and Edge Deployment in Vision–Language Spatial Reasoning.* 2026. This repository is the accompanying artifact (code, data, and pre-registrations).
