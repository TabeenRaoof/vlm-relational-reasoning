# Studies Index

This repo contains **three completed studies** plus **one planned (not started)
follow-up**. This page is a map — pre-registration, run plan, raw data, and
analysis output for each — so you don't have to reconstruct study boundaries
from `results/` file names. For the full narrative and interpretation, see the
paper draft (held externally, not yet committed to this repo).

| # | Study | Status | n | Pre-registered? |
|---|---|---|---|---|
| 1 | Exploratory pilot | **Complete** | 300 | No (hypothesis-generating) |
| 2 | Confirmatory interaction study (RQ1/RQ2) | **Complete** | 2,000 | Yes |
| 3 | Edge equivalence study (RQ3 / "H3") | **Complete** | 2,000 | Yes |
| 4 | High-powered interaction re-test | **Not started** — planning brief only | TBD (target ~6,000–8,000) | Not yet — pre-registration doesn't exist |

---

## Study 1 — Exploratory pilot (n=300)

**Purpose.** Three-scale, hypothesis-generating comparison (Qwen2.5-VL-3B,
Qwen2.5-VL-7B, Gemini-2.5-Flash + inverted-prompt/thinking ablations) on 300
VSR items (150 projective, 150 containment). Not pre-registered — this is
what motivated Studies 2 and 3.

**Headline result.** 3B→7B costs 11.3 pp on projective, 0.0 pp on containment
(McNemar projective p=0.003, containment null). Interaction test itself:
LR χ²(1)=2.39, p=0.122, ~35% power — the non-significant interaction here is
why Study 2 exists.

**Key files:**
- Data: `data/pilot_eval_set_n150.csv` (300 items, nested inside the n=2000 set)
- Raw results (pilot-scale, n=300/n=100 for the inverted-prompt arm):
  `results/mac_qwen25vl_3b.csv`, `results/mac_qwen25vl_7b.csv`,
  `results/jetson_qwen25vl_3b_q4.csv`, `results/gemini_2_5_flash.csv`,
  `results/gemini_2_5_flash_thinking.csv`,
  `results/gemini_2_5_flash_inverted_prompt.csv`
- Isolated copy of the above (used so pilot-scale re-analysis doesn't collide
  with the n=2000 files below on `model_name` — see its README):
  `results/pilot_n300/`
- Analysis: `scripts/analyze_pilot.py` → `results/analysis_full/`
  (accuracy, McNemar, RRS/DT, RCS-proxy, signal detection)
- Superseded pilot-scale artifacts (an earlier n=100 Gemini run and the
  intermediate Jetson chunks) were removed when the repository was pruned for
  release. They are recoverable from git history at commit `e794ad8` if ever
  needed for provenance; nothing in the paper depends on them.

---

## Study 2 — Confirmatory interaction study, n=2000 (RQ1, RQ2)

**Purpose.** Pre-registered test of whether reduced compute (7B→3B) degrades
projective-spatial reasoning more than topological-containment reasoning
(RQ1: formal interaction test; RQ2: category-specific McNemar tests).

**Freeze.** `preregistration/PREREGISTRATION.md`, frozen at commit `b176ed7`
(before any n=2000 data existed). Ceiling-checked feasibility
(max balanced n ≈ 3,016) before freezing.

**Headline result.** RQ1 (interaction) is **not significant**: LR χ²(1)=2.67,
p=0.102, ~37% realized power (winner's-curse shrinkage from the pilot's 0.64
log-odds effect to 0.27). RQ2 (category-specific) is robust: projective
p≈5.16×10⁻⁵ (survives Holm), containment null (p=0.53).

**Key files:**
- Pre-registration: `preregistration/PREREGISTRATION.md`
- Run plan: `docs/RUN_PLAN.md`
- Eval set: `data/eval_set_n2000.csv` (nests the n=300 pilot set, verified)
- Raw results: `results/mac_qwen25vl_3b_n2000.csv`,
  `results/mac_qwen25vl_7b_n2000.csv`
- Image-fetch log: `results/image_fetch_failures_n2000.csv` (0/1,337 failures)
- Analysis: `scripts/interaction_test.py` →
  `results/analysis_n2000/interaction_primary.txt` (RQ1);
  standard McNemar suite + the full 6-test pre-registered confirmatory
  family (RQ1–RQ3, Holm-corrected) →
  `results/analysis_n2000/mcnemar_confirmatory.txt`

---

## Study 3 — Edge equivalence study, n=2000 ("H3", RQ3)

**Purpose.** Pre-registered **equivalence** (TOST, not difference) test of
whether accuracy transfers from an M2 Max desktop to a Jetson Orin Nano 8GB,
with byte-identical (SHA-256-verified) weights, identical engine (Ollama),
identical item set. Margin ±3 pp, α=0.05.

**Freeze.** `preregistration/PREREGISTRATION_H3_n2000.md`, frozen at commit
`792eb12` (2026-07-16T07:43:28Z, before any Jetson n=2000 data existed). Uses
the Study 2 Mac 3B baseline as its comparison arm (not re-run).

**Headline result.** **Equivalence declared.** Observed difference +0.80 pp
(Jetson favored), 90% CI [+0.08, +1.52] pp, upper one-sided TOST p=2.36×10⁻⁷
(n=1,996 scoreable of 2,000; 4 excluded — reproducible device-side crashes,
not data problems). Accuracy transfers essentially intact; the real
deployment cost is operational (~21× throughput penalty in the sustained
regime, 8× context-window reduction, 0.2% hard-failure rate on inputs the
desktop handled) — see the paper §7–8 for the full operational writeup.

**Key files:**
- Pre-registration: `preregistration/PREREGISTRATION_H3_n2000.md`
- Run plan: `docs/RUN_PLAN_H3.md`
- Jetson/SSH setup and gotchas: `docs/jetson_setup.md`
- Raw results: `results/jetson_qwen25vl_3b_q4_n2000.csv` (2,027 raw rows,
  2,000 unique items after dedup)
- Analysis: `scripts/h3_equivalence_test.py` →
  `results/analysis_h3_n2000/equivalence.txt`
- Known device-side poison items (skipped unconditionally, see
  `scripts/run_jetson_eval.py::KNOWN_POISON_ITEM_IDS`):
  `vsr_e76cbe6931fc`, `vsr_93309e3ab9fe`, `vsr_622d284e6fe7`,
  `vsr_1e406802b166`

---

## Study 4 — High-powered interaction re-test (planned, not started)

**Status: planning brief only.** No pre-registration exists yet, and per the
brief's own rules, none of this study's data may be collected before one is
written and committed. This is deliberately kept **separate** from Study 2 —
collecting more data now, having already seen the Study 2 null, would be
optional stopping / sample-size ratcheting and would invalidate any resulting
p-value.

**Purpose.** Study 2's interaction test was a genuine null at only ~37% power
(not "no effect," just underpowered). This study is designed to reach >90%
power for the *revised* effect estimate (~0.27 log-odds) via a second
containment-heavy dataset — VSR's containment pool (1,508 max balanced) is
the binding constraint and cannot supply the needed n on its own.

**Key file:** `docs/FUTURE_STUDY_interaction_high_power.md` — open design
questions (candidate second dataset, taxonomy-mapping-as-confound risk,
group-sequential stopping rule) that must be resolved before a
pre-registration can be written.

---

## Cross-cutting references

- `docs/PIPELINE_NOTES.md` — required `analyze_pilot.py` fixes
  (`--model-order`, multiple-comparisons correction) and real gotchas hit
  during data collection (filename/`model_name` collisions, Jetson
  duplicate rows, Gemini version drift, `.env` hygiene).
- `docs/PAPER_MAP.md` — every quantitative claim in the paper mapped to the
  script that produces it and the output file that holds it.
- `README.md` — top-level project overview and quickstart.
