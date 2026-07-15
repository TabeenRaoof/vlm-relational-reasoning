# Morning Summary — Overnight Autonomous Run

**Session start:** 2026-07-08 (overnight, unattended)
**Repo:** ~/cs587-pilot · **Git:** not a git repo → commits skipped (per instructions)
**API key:** `GEMINI_API_KEY` present in `.env` (sourced for API steps) → Step 6 not blocked
**Cost ceiling:** $8 hard stop on Step 6. Running tally below.

This file is updated incrementally as steps complete, so it survives interruption.

---

## Running cost tally (FINAL)
| Item | Calls | Est. cost |
|---|---|---|
| No-thinking full (300) + 11 retries | 311 | ~$0.06 |
| Thinking full (300, ~980 out-tok/item) | 300 | ~$0.75 |
| Inverted-prompt (100) | 100 | ~$0.02 |
| Ollama Qwen-7B (300) | 300 | $0.00 (local) |
| **TOTAL API SPEND (est.)** | | **~$0.83** |

Well under the $8 ceiling; the backstop never came close to triggering.
(Estimate from item counts × observed token usage; Gemini list prices
~$0.30/M input, ~$2.50/M output. No billing API was queried.)

---

## Step 0 — GATE: item_id fix verification ✅ PASS
Re-ran curation at n50 and n150, seed 42, and ran the three-check gate:
- `nesting: PASS` (n50 content ⊂ n150 content)
- `ids nest: PASS` (n50 item_ids ⊂ n150 item_ids)
- `no collisions: PASS` (0 id→different-content mismatches)

n50 = 100 rows, n150 = 300 rows. IDs are content-derived hashes (e.g.
`vsr_4f766f327ec1`), not positional. Fix from the prior session is in place and
holding. Proceeding.

---

## Step 1 — resume-skips-errors fix ✅ DONE
`run_gemini_eval.py` and `run_ollama_eval.py`: resume now only skips rows that
completed **without** an error (`error` NaN/empty). Errored rows are retried on
resume instead of being silently treated as done (which had been shrinking the
effective n). Both scripts compile. Prints how many errored rows will be retried.

## Step 2 — per-run generation-config logging ✅ DONE
- Added `write_run_config()` → writes a sidecar `*.config.json` next to every
  results CSV (model, temperature, max_output_tokens, thinking flag/budget,
  prompt variant, full system instruction + user template, timestamp).
- Made `max_output_tokens` explicit and passed through: **10** for no-thinking,
  **1024** for thinking.
- **⚠️ Deliberate deviation, flagged:** the instructions suggested ~500 for
  thinking. I set **1024** instead. Empirical reason: an earlier thinking probe
  showed Gemini spends **~980 thinking tokens** before answering; a 512 ceiling
  would starve the final answer and reproduce the exact empty-response bug Step 2
  is meant to prevent. 1024 was already validated (0 unparseable across 100 items
  last night). Override with `--max-output-tokens` if you disagree.
- **Config-parity confirmation:** the only intended differences between the
  no-thinking and thinking conditions are (a) `thinking` on/off and (b)
  `max_output_tokens` (10 vs 1024) — and (b) is a *required consequence* of (a),
  not an independent confound. Temperature (0.0), system instruction, user
  prompt text, model, and eval set are **identical**. The sidecar JSONs make this
  auditable/diff-able after the runs.

## Step 3 — signal-detection metrics (d', c) ✅ DONE
Added to `analyze_pilot.py`:
- `signal_detection_table()` → per-(model, category) **and** pooled d' & c, with
  log-linear correction (Hautus) and **Gourevitch & Galanter 95% CIs**
  (se(c)=se(d')/2). Writes `signal_detection.csv`.
- `check_pooling_paradox()` → prints a first-class WARNING when a model's pooled
  d' is below every per-category d' (criterion divergence). **This fired on the
  existing thinking data** (pooled 1.928 < per-cat min 2.105) — the real
  phenomenon you flagged, now caught automatically.
- Reports BOTH aggregate and per-category, never aggregate alone.
- Signal def: signal=TRUE caption, "detect"=model says YES. c>0 ⇒ NO-bias.
- Defensive dedup added to `load_all_results` (keeps last row per
  model+item_id) so a resume-retry can't create duplicate-index McNemar joins.

## Step 4 — outlier-aware latency ✅ DONE
Added `latency_table()` (median, raw mean, >10s-trimmed mean, outlier count per
model×category) and `latency_outliers()` (lists every item >10s with its error).
Outliers are excluded only from the trimmed stat, never from the dataset.
Validated on existing data: e.g. containment mean_raw 4.03s vs median 1.28s vs
trimmed 1.54s — 2 timeout/retry items were distorting the mean. `latency_table.csv`
+ `latency_outliers.csv` written.

## Step 5 — re-curate full-scale set ✅ DONE
`--n-per-category 150 --seed 42` → `data/pilot_eval_set_n150.csv`, 300 items
(150/cat, 75T/75F each; projective 19 relations, containment 13). Nesting vs n50:
content PASS, id PASS, no collisions PASS.

## Step 6 — full-scale Gemini runs
Old 100-item pilot CSVs (+ their raw responses) were **archived**, not deleted,
to `results/archive_pilot_n100/` (STOP CONDITION 4). Then fresh 300-item runs.

### 6a. No-thinking (n=300) ✅ DONE — `results/gemini_2_5_flash.csv`
- Clean final: 300 rows, **0 errors, 0 unparseable** (after retry+dedup, below).
- **Overall 245/300 = 81.7%** (matches pilot's 81.0%).
- projective_spatial: 116/150 = **77.3%**
- topological_containment: 129/150 = **86.0%**
- TRUE-caption recall **70%**, FALSE-caption **93%** (NO-bias persists).
- Run note: first pass had **11 transient 503 "high demand" errors**; the Step 1
  resume-fix retried exactly those 11 (all succeeded), then I de-duplicated the
  append-mode CSV back to 300 clean rows (kept the successful retries; the 503
  rows carried no data). Sidecar `gemini_2_5_flash.config.json` written.
- ⚠️ **SURPRISE / look with fresh eyes:** the category ordering **flipped** vs the
  pilot. Pilot (n=100): projective 82% ≥ containment 80%. Full (n=300):
  containment 86% > projective 77%. The pilot's near-tie was noise; at 150/cat,
  containment is clearly easier for Gemini and projective clearly harder. This
  changes the H1 baseline narrative — worth eyeballing before you lean on it.

### 6b. Thinking (n=300) ✅ DONE — `results/gemini_2_5_flash_thinking.csv`
- Clean first pass: 300 rows, **0 errors, 0 unparseable** (the 1024 token ceiling
  held — no empty responses, validating the deviation from ~500).
- **Overall 243/300 = 81.0%** · projective 80.0% · containment 82.0%.
- TRUE recall 72%, FALSE 90% (NO-bias persists). Mean latency 3.48s.
- **Thinking vs no-thinking (paired, n=300): 81.0% vs 81.7%, McNemar p = 0.868 →
  NOT significant.** Confirms the pilot's finding (p=0.79) at 3× the sample:
  chain-of-thought does not move accuracy on these yes/no spatial items. Thinking
  fixed 17 items and broke 19 — pure reshuffling, no net gain.
- **Config parity CONFIRMED (Step 2.3):** diffing the two sidecar JSONs, the ONLY
  differences are `thinking` (False→True), `max_output_tokens` (10→1024), and
  `thinking_budget` (0→dynamic) — and the latter two are *required consequences*
  of enabling thinking, not independent confounds. `system_instruction`,
  `user_prompt_template`, and `temperature` are **byte-identical**. So the
  comparison isolates the thinking setting alone. ✅

### Cost tally (Step 6)
Estimated so far, well under the $8 ceiling:
| Run | Items | Est. cost |
|---|---|---|
| No-thinking (300) + 11 retries | 311 calls | ~$0.06 |
| Thinking (300) — in progress | 300 calls | ~$0.75 (proj., ~980 think-tok/item) |
| **Projected Step 6 total** | | **~$0.81** |

Nowhere near the ceiling; no risk of triggering the backstop.

## Step 7 — inverted-prompt bias check ✅ DONE
`results/gemini_2_5_flash_inverted_prompt.csv` (100 items, no-thinking, question
flipped to "is this caption FALSE?", scoring inverted back to caption-truth
space). Clean: 0 errors, 0 unparseable. Paired against the SAME 100 item_ids
from the full no-thinking run (n50 ⊂ n150, so identical items).

**Caption-truth-space metrics (same 100 items):**
| Condition | TRUE-recall | FALSE-specificity | gap | overall |
|---|---|---|---|---|
| Standard ("is it TRUE?") | 72% | 94% | **22 pp** | 83% |
| Inverted ("is it FALSE?") | 84% | 84% | **0 pp** | 84% |

**Paired significance (McNemar, n=50/cell):**
- TRUE items (recall): 72%→84%, inverted fixed 6 / broke 0, **p = 0.031 (sig.)**
- FALSE items (specificity): 94%→84%, broke 5 / fixed 0, **p = 0.062 (borderline)**
- ALL items: 83%→84%, **p = 1.000 (no net change)**

**VERDICT: the recall/specificity asymmetry is substantially a PROMPT-WORDING
(response-polarity) artifact, not a fixed perceptual bias.** Evidence: simply
rephrasing the question collapses the 22pp asymmetry to ~0 and **significantly
raises TRUE-caption recall (72→84%, p=0.031)** — a genuine perceptual inability
to confirm true relations could not be fixed by changing the question word.
BUT it is a trade, not a free win: specificity drops correspondingly (94→84%,
p=0.062) and **overall accuracy is unchanged (p=1.0)**. In signal-detection
terms: the model's discrimination (d') is stable; the phrasing moves the
*criterion*. So the "near-perfect specificity, poor recall" story from the pilot
is real under the standard prompt but is a property of the *question framing*,
not a stable claim about the model's spatial perception. Honest caveat: n=100
(50/cell) — the recall effect clears p<0.05, specificity is only borderline.

## Step 8 — Ollama / Mac Qwen2.5-VL-7B ✅ DONE (completed, not blocked)
- Ollama was **not installed**; installed the **CLI formula** headlessly via
  `brew install ollama` (v0.31.1) — no GUI cask, no interactive step needed.
  Started `ollama serve` in the background (logs/ollama_serve.log).
- Pulled `qwen2.5vl:7b` (6.0 GB) successfully. Smoke test: first call 9.8s
  (model load), warm 0.2s, coherent YES/NO.
- Full 300-item run → `results/mac_qwen25vl_7b.csv`: **300 rows, 0 errors,
  0 unparseable.** Mean latency 5.07s (median 3.35s) — local inference, no cost.
- **Results: OVERALL 247/300 = 82.3%** · projective 82.0% · containment 82.7%.
  TRUE recall 73%, FALSE 92%. Predicts 179 NO / 121 YES.
- ⚠️ **SURPRISE / headline for you:** the **7B open model (82.3%) essentially
  ties frontier Gemini (81.7% no-thinking / 81.0% thinking).** The cloud "ceiling"
  is barely above the mid-tier on this task — either the task saturates ~82% for
  these architectures, or Gemini isn't converting its scale advantage into
  spatial-reasoning accuracy here. This is the most important thing to eyeball.
- Both models show the **same NO-bias** (Qwen TRUE-recall 73% vs FALSE 92%;
  Gemini 70% vs 93%) — consistent with Step 7's finding that the asymmetry is
  largely a property of the yes/no question framing, and it reproduces across
  two independent model families.

## Step 9 — full analysis ✅ DONE
`results/analysis_full/` written: accuracy_table, signal_detection, latency_table,
latency_outliers, rrs, dt, rcs_proxy, mcnemar, summary.json, +
accuracy_by_category.**svg** (the PNG step failed — see Anomaly #6; SVG is a
matplotlib-free substitute). Included 4 conditions: `qwen2.5vl:7b`,
`gemini-2.5-flash`, `gemini-2.5-flash-thinking`, `gemini-2.5-flash-inverted`.
Compute-axis metrics (RRS/DT) used the 2 real tiers (Qwen-7B → Gemini); the
thinking/inverted variants are analysed in the tables but excluded from the
compute curve (they are config variants, not compute points).

**Headline analysis numbers (n=300 unless noted):**
| Condition | Proj | Cont | Overall | d' proj | d' cont | c (pooled) |
|---|---|---|---|---|---|---|
| qwen2.5vl:7b | 82.0% | 82.7% | 82.3% | 1.99 | 1.94 | +0.39 |
| gemini-2.5-flash | 77.3% | 86.0% | 81.7% | 1.64 | 2.38 | +0.48 |
| gemini-2.5-flash-thinking | 80.0% | 82.0% | 81.0% | 1.69 | 2.06 | +0.34 |
| gemini-2.5-flash-inverted (n=100) | 84.0% | 84.0% | 84.0% | 1.91 | 1.91 | **≈0.00** |

- **McNemar: NO pair differs significantly at the ALL level.** Gemini vs Qwen-7B
  p=0.885; Gemini vs thinking p=0.868; Gemini vs inverted p=1.0. On this task the
  three compute/config points are statistically indistinguishable in overall
  accuracy.
- **d'/c:** all standard conditions have c ≈ +0.35–0.48 (consistent NO-bias);
  the inverted prompt drives c to ≈0 (bias removed) — the SDT signature of the
  Step 7 finding. Discrimination d' is similar (~1.6–2.4) across all.
- **RRS:** projective 0.797, containment 0.843. Projective RRS is *lower* because
  accuracy actually DROPS from Qwen-7B (82%) to Gemini (77%) — an inverted
  "degradation" (more compute → slightly worse on projective). **DT** (70% floor):
  Qwen-7B already clears the floor in both categories.

## Aside — /fewer-permission-prompts (handled mid-run)
You invoked this. I scanned 17 recent transcripts. **No allowlist entries added:**
your read-only commands (git read-only, grep/ls/cat/find/lsof/sed/docker ps/
gh pr view) are already auto-allowed; your other frequent commands are
interpreters (python/java/source) and mutating ops (rm/npm run/pip/git push) —
both categories are unsafe/ineligible to allowlist. `.claude/settings.json` left
untouched (fabricating entries would be useless or dangerous). Ask me to add a
specific exact command if a particular prompt is bugging you.

---

# ============ MORNING BRIEFING (read this first) ============

## What ran successfully
| Step | Result |
|---|---|
| 0 Gate | item_id fix verified (nesting/stability PASS) |
| 1 Resume-errors fix | Applied to both eval scripts |
| 2 Config logging | Sidecar `*.config.json` per run; parity confirmed |
| 3 d'/c metrics | Per-category + pooled, G&G 95% CIs, pooling-paradox check |
| 4 Latency | Median / raw / trimmed + outlier list |
| 5 Re-curate n150 | 300 items, nests cleanly |
| 6 Gemini ×2 (n=300) | no-think 81.7%, think 81.0% (p=0.87, n.s.) |
| 7 Inverted prompt (n=100) | asymmetry is largely a PROMPT ARTIFACT |
| 8 Qwen-7B Mac (n=300) | 82.3% — ties Gemini; local, headless install |
| 9 Full analysis | `results/analysis_full/` complete |

## What did NOT run, and exactly why
- **Jetson tier (Qwen2.5-VL-3B): deliberately NOT attempted.** This was STOP
  CONDITION #1 (hardware safety — unattended firmware/boot risk). The board was
  not touched in any way. The compute spectrum is therefore **2 of 3 tiers**
  (Mac 7B + Gemini) this session; Jetson remains for when you're present.
- Nothing else was blocked. Ollama install did NOT require a GUI step (CLI
  formula), so Step 8 completed rather than falling back.

## Total estimated API cost: ~$0.83 (ceiling was $8)

## Bias-check verdict (Step 7)
The "near-perfect specificity, poor TRUE-recall" asymmetry is **substantially a
prompt-wording artifact, not a fixed perceptual bias.** Inverting the question
("is it FALSE?") collapsed the 22pp specificity–recall gap to ~0 and
significantly raised TRUE-recall (72→84%, p=0.031), with NO change in overall
accuracy (p=1.0) — i.e. rephrasing moves the *criterion*, not the *discrimination*.
The same NO-bias appears in Qwen-7B too, so it's a property of the yes/no VSR
framing shared across model families, not a quirk of one model.

## Wall-clock (approx, per major step)
- Steps 0–5 (code edits + curation): ~20 min
- Step 6 no-thinking (incl ~200 image downloads): ~20 min
- Step 6 thinking (cache hits): ~18 min
- Step 7 inverted: ~3 min
- Step 8 Ollama (brew install + 6GB pull + 300-item eval): ~28 min
- Step 9 analysis: ~1 min

## Anomalies / look with fresh eyes (not hidden)
1. **7B ≈ frontier.** Qwen-7B (82.3%) ties Gemini (81.7%); McNemar p=0.885. The
   cloud ceiling barely exceeds the mid-tier — the headline compute-spectrum
   story is weaker than the proposal assumed. Either the task saturates ~82% or
   Gemini isn't leveraging scale on spatial reasoning.
2. **Category order flipped vs the pilot.** Pilot: projective ≈ containment.
   Full n=300: Gemini does *worse* on projective (77%) than containment (86%),
   and Qwen is balanced (82/83). The pilot's near-tie was small-n noise.
3. **Inverted degradation on projective.** Going Qwen-7B→Gemini, projective
   accuracy DROPS (82→77%). "More compute" is not monotonically better here.
4. **One Qwen item took 521s** (8.7 min, `contains`, vsr_1ac3adde0f77) but
   completed with no error — a single stall, not repeated, so not a stop
   trigger. The median (3.5s) is unaffected; flagged in latency_outliers.csv.
5. **~14 Gemini no-thinking items hit ~63s** = the 60s-timeout+retry recovering
   from transient stalls. Expected behaviour of the Step-1/robustness fixes,
   just visible in the latency tail.
6. **⚠️ ENVIRONMENT: `brew install ollama` (Step 8) silently upgraded
   `python@3.14` 3.14.0 → 3.14.6.** The venv now runs 3.14.6, and its new
   `pyexpat` build references a libexpat symbol
   (`_XML_SetAllocTrackerActivationThreshold`) not present in the macOS system
   `/usr/lib/libexpat.1.dylib`. Net effect: **`import matplotlib` fails**, so
   `analyze_pilot.py` could not write the PNG (it caught the error and skipped).
   ALL numeric analysis is unaffected (pandas/scipy/numpy don't touch pyexpat)
   and complete. I generated a **matplotlib-free substitute**:
   `results/analysis_full/accuracy_by_category.svg`. Proper fixes for later
   (not done unattended — too risky to the working eval env): pin/downgrade
   `python@3.14`, or rebuild the venv once Homebrew ships a corrected build.
   Requirements.txt was "verified on 3.14.0"; you're now effectively on 3.14.6.

## What I'd do next — OPTIONS for you (not decisions)
- **Jetson tier (when you're present):** run Qwen2.5-VL-3B on the Orin Nano to
  complete the 3-point spectrum. This is the missing piece for the H3
  cloud-vs-edge story.
- **Re-examine the "7B ties frontier" result:** if real, it reframes the paper
  from "compute spectrum degradation" toward "task saturates below frontier —
  where's the ceiling?" Might warrant a harder item subset (the near-chance
  relations) to separate the tiers.
- **Decide how to treat the inverted-prompt finding:** either (a) report it as a
  methodological caveat (accuracy is criterion-sensitive), or (b) adopt a
  balanced two-way prompt (average of TRUE/FALSE phrasings) to neutralise the
  criterion artifact in the main results.
- **Scale n if you want tighter CIs:** current per-category CIs are ~±6pp at
  n=150; the curation nests cleanly so you can go to n=300/category without
  re-running what's done.
- **PILOT_RUN_LOG.md** still reflects the OLD 100-item pilot numbers — say the
  word and I'll refresh it to the n=300 results, or leave it as the pilot record.
