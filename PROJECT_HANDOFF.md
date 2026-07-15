# CS587 Pilot — Comprehensive Handoff & Context Report

**Purpose.** A single self-contained document to restore full context in a new
conversation (Claude Code, Claude chat, or for you). It covers the plan, the
reasoning, every decision and pivot, all results, the exact state of the repo,
and how to resume. If you're starting fresh, **read §1–§3 first**, then jump to
whatever you need.

**Last updated:** 2026-07-10 · **Repo:** `~/cs587-pilot` (NOT a git repo) ·
**Deadline:** data-analysis report due Friday.

**Companion docs (all current):**
- `morning_summary.md` — the overnight autonomous run briefing (Steps 0–10).
- `notebooks/PILOT_RUN_LOG.md` — detailed methods/results log (⚠️ still reflects
  the OLD 100-item pilot numbers; superseded by the n=300 results here).
- `README.md` — the original 9-step pipeline plan and pilot-scope rationale.
- `scripts/JETSON_DECISION_TREE.md` — Jetson toolchain plan (NOT executed; see §9).

---

## 1. What this project is

A **pilot evaluation of vision-language models (VLMs) on visual spatial
reasoning**, measured across a compute spectrum:

```
Qwen2.5-VL-3B  ->  Qwen2.5-VL-7B (Mac/Ollama)  ->  Gemini 2.5 Flash (cloud)
```

Data source: **VSR (Visual Spatial Reasoning)**, `cambridgeltl/visual-spatial-reasoning`,
`splits/random`. Each item is an (image, caption, true/false label) triple.

Two relation categories (deliberately, not three):
- **projective_spatial** — viewpoint-dependent directional relations
  (`left of`, `above`, `in front of`, `facing`, …).
- **topological_containment** — viewpoint-independent enclosure/part-whole
  relations (`in`, `inside`, `contains`, `at the edge of`, …).

Research questions the pilot informs:
- **H1** — do relation types degrade non-uniformly as model scale drops?
- **H3** — cloud-vs-edge/hardware transfer (needs the Jetson tier — NOT yet done).

---

## 2. Current status at a glance (TL;DR)

**Done:** full pipeline through analysis on the corrected n=300 eval set, three
model tiers collected, plus two Gemini ablations (thinking, inverted-prompt).
**Not done:** Jetson tier (hardware-safety hold), the written report itself, a
matplotlib/env fix.

### Headline results (n=300 unless noted; scoreable accuracy)
| Model / condition | Projective | Containment | Overall | Notes |
|---|---|---|---|---|
| **qwen2.5vl:3b** | 70.7% | 82.7% | **76.7%** | user-run; see §8 |
| **qwen2.5vl:7b** (Mac) | 82.0% | 82.7% | **82.3%** | |
| **gemini-2.5-flash** (no-think) | 77.3% | 86.0% | **81.7%** | primary ceiling |
| gemini-2.5-flash (thinking) | 80.0% | 82.0% | 81.0% | ablation |
| gemini-2.5-flash (inverted, n=100) | 84.0% | 84.0% | 84.0% | bias probe |

### The three findings that matter
1. **Compute spectrum: 3B (76.7%) < 7B (82.3%) ≈ Gemini (81.7%).** There IS
   degradation from 7B→3B, but the frontier ceiling only *ties* the 7B — the task
   appears to **saturate ~82%** for these architectures above 7B.
2. **Degradation is non-uniform (supports H1):** projective drops hardest at 3B
   (70.7%) while containment holds (~82–86% across all tiers). RRS: projective
   0.780 < containment 0.835.
3. **The "NO-bias" is largely a PROMPT-WORDING ARTIFACT, not perception.**
   All standard conditions over-answer NO (high specificity ~90%, low TRUE-recall
   ~65–73%). Inverting the question ("is it FALSE?") collapses the 22pp
   recall/specificity gap to ~0 and significantly raises TRUE-recall (72→84%,
   p=0.031) with NO change in overall accuracy (p=1.0). Same bias appears in BOTH
   Qwen and Gemini → it's a property of the yes/no VSR framing, not one model.

**McNemar:** no pair of models differs significantly in *overall* accuracy at
n=300 (all p>0.1). The signal is in the per-category and per-tier breakdown, not
the aggregate.

**Total API cost to date: ~$0.83** (Gemini; billing enabled). Qwen tiers are
local (free).

---

## 3. Locked-in decisions (do NOT relitigate — set deliberately)

- **Two categories only** (projective_spatial, topological_containment).
  Physical support is out of scope for the pilot (would need PhysBench).
- **Yes/no prompting**, not open-ended (VSR is natively true/false).
- **gemini-2.5-flash** as the cloud ceiling.
- **Qwen2.5-VL family** at two scales: 3B and 7B — same architecture, clean
  scale isolation.
- **N=50/category (100)** pilot scale; **N=150/category (300)** full scale. The
  curation nests (n50 ⊂ n150) so results compose.
- **PhysBench excluded** — multiple-choice + ~50% video, incompatible with the
  yes/no scoring pipeline and with Qwen-on-Ollama (no video input).
- **Scripts were pre-written**; the rule was "don't rewrite unless actually
  broken, and ask/explain before changing." Every change below followed that.

### Jetson — hardware-safety STOP (critical)
The Jetson Orin Nano (8GB, firmware 36.4.7 / JetPack 6.2.1, already fixed) must
**NOT** be touched unattended — SSH, flash, or run — because an interrupted
firmware write can brick it and there's no safe unattended recovery. It was left
entirely alone. This is a safety hold, not a permissions issue.

---

## 4. Chronological narrative — plan, reasoning, decisions, pivots

### Session A — setup & first Gemini run (Steps 0–3 of README)
1. **Env (Step 0):** created `venv/` (Python 3.14.0), `pip install -r
   requirements.txt`. Risk flagged (3.14 very new) but cp314 wheels existed → no
   issue.
2. **Fetch VSR (Step 1):** downloaded train/dev/test (7,680 / 1,097 / 2,195 —
   verified against expected counts). Pooled 10,972 items, 66 relations.
3. **Curate (Step 2):** `curate_eval_set.py --n-per-category 50 --seed 42` →
   100 items, 25T/25F per category. Stopped at a checkpoint to show the user.
4. **API key handling:** created a **git-ignored `.env`** (+ `.env.example`,
   `.gitignore`). Decided against putting the key in README (leak risk). Billing
   was later enabled on Google project `edge_relational_reasoning`.
5. **Gemini run (Step 3) — four problems, four fixes** (all diagnosed live):
   - **429 rate limit:** free tier = 5 req/min. → user enabled billing (Tier 1).
   - **84% empty responses:** gemini-2.5-flash is a *thinking* model; it spent
     the 10-token output budget on hidden reasoning → empty `.text`. → set
     `thinking_config(thinking_budget=0)` for the primary run (also makes Gemini
     answer directly like the Qwen tiers — cleaner cross-model comparison).
   - **Indefinite hangs:** the SDK had no request timeout; one call stalled
     forever (confirmed via `lsof` — open HTTPS conn to Google). → 60s client
     timeout + 3-retry loop.
   - **Resume-skips-errors (design flaw, noted then):** resume skipped any
     written item_id *including errored rows*, silently shrinking n.
   - Result: no-thinking **81%** (pilot n=100), NO-bias observed.
6. **Thinking ablation:** added `--thinking`/`--label` flags; thinking gave 83%
   (pilot), McNemar p=0.79 → not significant. Validated the no-thinking choice.
7. Wrote `notebooks/PILOT_RUN_LOG.md`; merged `caption`+`image_url` into result
   CSVs for manual spot-checking.

### Session B — the item_id correctness bug (user-reported, critical)
The pre-written `curate_eval_set.py` had a **silent correctness bug**:
- `item_id` was **positional** (`vsr_0001` = enumeration index) → the same id
  meant a different VSR item on every run.
- `random.sample()` doesn't nest → n=50 was not a subset of n=150.
- Consequence: `analyze_pilot.py`'s McNemar join on `item_id` would silently pair
  different images across models and emit a bogus p-value.

**Fix (exactly as specified, verified empirically):**
- `item_id = "vsr_" + sha256(f"{image}||{caption}")[:12]` — content-derived,
  position-independent.
- Sampling now sorts each label-pool by a deterministic hash key
  `sha256(f"{seed}||{image}||{caption}")` and takes top-k → **nesting** (n50 ⊂
  n150) guaranteed. Deterministic `seed+1` interleave replaces the shuffle.
- 5-check verification passes (content nesting, id nesting, no collisions, hash
  ids, label balance). **Consequence:** the earlier positional-id Gemini CSVs
  were superseded and later re-run at full scale.

### Session C — overnight autonomous run (Steps 0–10, unattended)
Ran per an explicit 10-step overnight brief with a $8 cost ceiling and STOP
conditions (Jetson, cost, unfixable gate, destructive deletes). Summary:
- **Step 0** gate re-verified (nesting/stability PASS).
- **Step 1** resume-skips-errors fixed in `run_gemini_eval.py` AND
  `run_ollama_eval.py` (skip only rows with empty `error`).
- **Step 2** per-run **config sidecars** (`*.config.json`) added; `max_output_tokens`
  made explicit (10 no-think / **1024** think). **Deliberate deviation:** brief
  suggested ~500 for thinking; used 1024 because a probe showed thinking spends
  ~980 tokens — 512 would starve the answer and re-trigger the empty-response bug.
  Config parity confirmed: only `thinking`/`max_output_tokens` differ between
  conditions; temperature, system instruction, prompt text are byte-identical.
- **Step 3** signal-detection metrics: per-category AND pooled **d′ and criterion
  c** with log-linear correction and **Gourevitch-Galanter 95% CIs**, plus a
  **pooling-paradox warning** (fires when pooled d′ < every per-category d′).
- **Step 4** outlier-aware latency (median / raw mean / >10s-trimmed mean +
  outlier list). Also added a defensive dedup in `load_all_results`.
- **Step 5** re-curated n=150 (300 items), nests cleanly.
- **Step 6** full-scale Gemini: no-thinking **81.7%**, thinking **81.0%**
  (McNemar p=0.868). Old pilot CSVs **archived** (not deleted) to
  `results/archive_pilot_n100/`. 11 transient 503s were retried (Step-1 fix) and
  the append-mode dupes de-duplicated to a clean 300 rows.
- **Step 7** inverted-prompt bias probe → the artifact finding (see §2).
- **Step 8** Ollama installed **headlessly** via `brew install ollama` (CLI
  formula, no GUI), pulled `qwen2.5vl:7b` (6GB), ran 300 items → **82.3%**.
- **Step 9** full analysis → `results/analysis_full/`.
- **Step 10** wrote `morning_summary.md`.
- **Env anomaly:** `brew install ollama` upgraded `python@3.14` 3.14.0→3.14.6;
  the new `pyexpat` build breaks `matplotlib` import (libexpat symbol mismatch).
  Numeric analysis unaffected; generated an **SVG** figure substitute.
- `/fewer-permission-prompts` invoked mid-run → nothing eligible to add (all
  read-only commands already auto-allowed; the rest are interpreters/mutating).

### Session D — this session (report + 3B discovery)
- While preparing this report, discovered **`results/mac_qwen25vl_3b.csv`** —
  a 3B run (300 items, clean, correct hash ids) created 2026-07-08 20:43,
  **after** the overnight session, **by the user (not me)**. See §8.
- Recomputed: 3B = **76.7%**. Re-ran the analysis with the full 3-tier order
  (`qwen2.5vl:3b → qwen2.5vl:7b → gemini-2.5-flash`). This completes the compute
  spectrum and is the basis for the §2 findings.

---

## 5. Script changes ledger (what changed and why)

All four scripts were modified; every change was to fix a real defect or add a
requested capability, never a silent refactor.

**`scripts/curate_eval_set.py`** — content-derived `stable_item_id()` + hash-rank
`stratified_sample()` for nesting (Session B). Removed `random`, added `hashlib`.

**`scripts/run_gemini_eval.py`**
- `thinking_budget=0` default (fix empty responses).
- 60s client timeout + 3-retry loop (fix hangs).
- Resume skips only non-errored rows (Step 1).
- New flags: `--thinking`, `--label`, `--invert-prompt`, `--max-output-tokens`.
- `write_run_config()` → sidecar `*.config.json`.
- Inverted-prompt constants + scoring inversion (maps back to caption-truth space).

**`scripts/run_ollama_eval.py`** — resume skips only non-errored rows (Step 1).
Otherwise unchanged (uses the shared prompt/parser from the Gemini script).

**`scripts/analyze_pilot.py`**
- `signal_detection_table()` (d′/c, log-linear, G&G CIs) + `check_pooling_paradox()`.
- `latency_table()` + `latency_outliers()` (outlier-aware).
- Defensive dedup in `load_all_results()` (keep last per model+item_id).
- SDT/latency/pooling added to printed output and `summary.json`.

---

## 6. File & artifact map

```
data/
  vsr_{train,dev,test}.jsonl        # raw VSR (7680/1097/2195)
  pilot_eval_set_n50.csv            # 100 items (hash ids) — n50 ⊂ n150
  pilot_eval_set_n150.csv           # 300 items — full scale
  images/                           # cached COCO images (~300)
results/
  gemini_2_5_flash.csv              # no-thinking, n=300, clean (+.config.json)
  gemini_2_5_flash_thinking.csv     # thinking, n=300 (+.config.json)
  gemini_2_5_flash_inverted_prompt.csv  # inverted, n=100 (+.config.json)
  mac_qwen25vl_7b.csv               # Qwen-7B, n=300 (my run)
  mac_qwen25vl_3b.csv               # Qwen-3B, n=300 (USER's run — see §8)
  archive_pilot_n100/               # superseded old-id pilot CSVs (preserved)
  analysis_full/                    # accuracy_table, signal_detection, latency_*,
                                    # rrs, dt, rcs_proxy, mcnemar, summary.json,
                                    # accuracy_by_category.svg
notebooks/PILOT_RUN_LOG.md          # detailed log (OLD n=100 numbers)
morning_summary.md                  # overnight briefing (current)
PROJECT_HANDOFF.md                  # THIS FILE
scripts/                            # 4 modified scripts + JETSON_DECISION_TREE.md
.env / .env.example / .gitignore    # secret handling (key git-ignored)
```

Result-CSV columns: `item_id, category, relation, ground_truth_label,
model_name, raw_response, parsed_label, is_parseable, is_correct,
latency_seconds, error`. (Gemini CSVs also have `caption, image_url`.)

---

## 7. How to resume — environment & commands

```bash
cd ~/cs587-pilot
source venv/bin/activate           # Python 3.14.6 (upgraded mid-project)
set -a; source .env; set +a        # loads GEMINI_API_KEY (git-ignored)
```

- **API:** `GEMINI_API_KEY` lives only in `.env`. Billing is ON (project
  `edge_relational_reasoning`). Gemini calls cost ~cents; the pilot to date ~$0.83.
- **Ollama:** `ollama serve` may still be running (idle). Models pulled:
  `qwen2.5vl:7b` (and `qwen2.5vl:3b` if the user pulled it). Start server headless
  with `/opt/homebrew/opt/ollama/bin/ollama serve &`.
- **Re-run a Gemini condition:** `python scripts/run_gemini_eval.py --input
  data/pilot_eval_set_n150.csv --output results/<name>.csv [--thinking]
  [--invert-prompt]` (resumable; writes a config sidecar).
- **Re-run analysis:** `python scripts/analyze_pilot.py --results-dir results
  --output-dir results/analysis_full --model-order qwen2.5vl:3b qwen2.5vl:7b
  gemini-2.5-flash`.

### Environment gotchas
- **matplotlib is broken** (Python 3.14.6 `pyexpat`/libexpat symbol mismatch after
  brew upgraded python). Numeric analysis is fine; the PNG step is skipped. Use
  `results/analysis_full/accuracy_by_category.svg` (matplotlib-free) or fix the env
  (pin/downgrade `python@3.14`, or rebuild the venv) before regenerating the PNG.
- **Not a git repo** — no version control; be careful with overwrites (old data
  is archived, not deleted, by policy).

---

## 8. ⚠️ Open item: the `mac_qwen25vl_3b.csv` provenance

- It exists (300 rows, clean, correct hash `item_id`s, `model_name=qwen2.5vl:3b`),
  created 2026-07-08 20:43 — **after my overnight run, by you (or a run you
  started), not by me.**
- The filename says **"mac"** and it's a standard Ollama-schema CSV, so I've
  treated it as a **Mac/Ollama 3B run**, and included it in the analysis on that
  basis. **Please confirm:** was this run on the **Mac (Ollama qwen2.5vl:3b)** or
  on the **Jetson**? It changes what the 3-point spectrum *means* (pure
  scale-on-same-hardware vs. cloud-vs-edge). If it's a Jetson run, it should be
  relabeled (e.g. `jetson_qwen25vl_3b`) and it partially fulfills H3.
- Either way its numbers are in §2 and the regenerated `analysis_full/`.

---

## 9. What is NOT done / next steps (options, not decisions)

1. **Jetson tier (Qwen2.5-VL-3B on the Orin Nano).** Deliberately untouched
   (safety hold). If the 3B file in §8 is actually a *Mac* run, the true Jetson
   edge tier is still open and is the missing piece for **H3 (cloud-vs-edge)**.
   Do this only when you're physically present (per JETSON_DECISION_TREE.md).
2. **Write the report** (README Step 9). `analysis_full/` maps onto the rubric's
   Methods/Results/Interpretation. The §2 findings are the spine.
3. **Refresh `notebooks/PILOT_RUN_LOG.md`** from n=100 → n=300 (currently stale).
4. **Fix the matplotlib/python env** if you want the PNG figure.
5. **Decide how to treat the inverted-prompt finding:** report it as a
   methodological caveat (accuracy is criterion-sensitive), or adopt a balanced
   two-way prompt to neutralize the criterion artifact in the headline numbers.
6. **Optional:** scale to n=300/category for tighter CIs (curation nests, so
   existing runs remain valid subsets).

---

## 10. One-paragraph reset (paste this to bootstrap a new chat)

> CS587 VLM spatial-reasoning pilot (`~/cs587-pilot`, not a git repo). VSR
> yes/no items, two categories (projective_spatial, topological_containment),
> compute spectrum Qwen2.5-VL-3B → 7B (Mac/Ollama) → Gemini-2.5-flash. Eval set
> is content-hashed and nests (n50 ⊂ n150=300 items). Results (n=300): 3B 76.7%,
> 7B 82.3%, Gemini 81.7% — degradation 7B→3B (worst on projective, 70.7%), but
> 7B≈Gemini (task saturates ~82%). A strong "NO-bias" (high specificity, low
> TRUE-recall) is shown to be largely a prompt-wording artifact (inverting the
> question fixes recall, p=0.031, no overall change). Thinking mode doesn't help
> (p≈0.87). Four eval scripts were fixed/extended (content-hash ids, resume-error
> retry, timeout+retry, config sidecars, d′/c metrics). Jetson tier is a
> hardware-safety hold (untouched). matplotlib is broken (python 3.14.6/pyexpat);
> use the SVG. Full detail in PROJECT_HANDOFF.md; overnight log in
> morning_summary.md. Open question: is results/mac_qwen25vl_3b.csv a Mac or
> Jetson run? API cost so far ~$0.83.
