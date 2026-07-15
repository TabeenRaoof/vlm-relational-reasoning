# vlm-relational-reasoning — Detailed Run Log & Analysis Notebook

**Purpose.** A running, report-ready record of every step taken to produce the
pilot data, the decisions made, the problems encountered, and what the results
mean. Written so that sections map onto a data-analysis report:

| This document | Maps to report section |
|---|---|
| §1 Environment & reproducibility | Methods / Reproducibility appendix |
| §2 Data source & curation | Data |
| §3 Evaluation methodology | Analysis Methods |
| §4 Execution log & issues encountered | Methods + Limitations (and defensibility) |
| §5–6 Results (per model run) | Results |
| §7 Interpretation | Interpretation / Discussion |

Dates use ISO format. This log covers the **Gemini ceiling tier**. Mac
(Qwen2.5-VL-7B) and Jetson (Qwen2.5-VL-3B) tiers will be appended as they run.

---

## 1. Environment & reproducibility

- **Machine:** Apple Silicon Mac (darwin 25.2.0). Python **3.14.0** (Homebrew),
  isolated `venv/`.
- **Dependencies** (pinned in `requirements.txt`, installed 2026-07-07):
  pandas 3.0.3, requests 2.34.2, google-genai 2.10.0, ollama 0.6.2,
  matplotlib 3.11.0, scipy 1.18.0. All installed from prebuilt `cp314` wheels —
  no source builds despite the very new interpreter.
- **Determinism:** curation uses a fixed RNG seed (**42**); all model calls use
  `temperature=0.0`. The eval item set is therefore exactly reproducible.
- **Secrets:** `GEMINI_API_KEY` is kept only in a git-ignored `.env` (never in
  tracked files); `.gitignore` and `.env.example` document the pattern.

---

## 2. Data source & curation (README Steps 1–2)

### 2.1 Source
Visual Spatial Reasoning (VSR), `cambridgeltl/visual-spatial-reasoning`,
`splits/random`. Downloaded three JSONL splits and verified line counts against
the published dataset:

| Split | Items | Expected | OK |
|---|---:|---:|:--:|
| train | 7,680 | 7,680 | ✓ |
| dev | 1,097 | 1,097 | ✓ |
| test | 2,195 | 2,195 | ✓ |
| **Total pooled** | **10,972** | | |

Each VSR item is a `(image, caption, label)` triple where `label ∈ {0,1}` marks
whether the caption's spatial claim is TRUE for the image. Images are hosted on
COCO (`images.cocodataset.org`).

### 2.2 Taxonomy (deliberate 2-category scope)
`scripts/curate_eval_set.py` assigns **all 66 VSR relations** to exactly one of:
- **projective_spatial** — viewpoint-dependent directional relations
  (`left of`, `above`, `in front of`, `facing`, …). Flipping/moving the camera
  can change the answer.
- **topological_containment** — viewpoint-independent enclosure / part-whole
  relations (`in`, `inside`, `contains`, `at the edge of`, …).
- **excluded** — proximity (`near`, `next to`), contact/support (`on`,
  `touching`; deferred to PhysBench in the full paper), and low-count/ambiguous
  relations.

The script **fails loudly** if VSR ever introduces an uncategorized relation
(guards against silent dataset drift). On this run: "all 66 relations
accounted for."

### 2.3 Sampling
Command: `curate_eval_set.py --n-per-category 50 --seed 42` →
`data/pilot_eval_set_n50.csv`.

Stratified: 25 True + 25 False per category, spread across as many distinct
relations as possible, then shuffled so label is not positional.

| Category | Items | True | False | Distinct relations |
|---|---:|---:|---:|---:|
| projective_spatial | 50 | 25 | 25 | 17 |
| topological_containment | 50 | 25 | 25 | 14 |
| **Total** | **100** | 50 | 50 | |

Most-sampled relations: `in` (13), `at the right side of` (10), `contains` (7),
`in front of` (6), `under` (6), `at the edge of` (6), `inside` (5),
`beneath` (4), `facing` (4), `in the middle of` (4). `in` is the single heaviest
relation (13/50 of containment) — worth noting as a mild concentration but
within expectation given VSR's native relation frequencies.

---

## 3. Evaluation methodology (Gemini tier)

- **Model (API):** `gemini-2.5-flash` — the study's cloud "ceiling" reference.
- **Format:** yes/no. System instruction asks for a single word YES/NO; user
  prompt supplies the caption and asks whether it is TRUE for the image.
  Identical prompt across all model tiers (the whole point of the comparison).
- **Decoding:** `temperature=0.0` (deterministic).
- **Scoring:** responses parsed by `parse_response()` into 1 (YES/TRUE),
  0 (NO/FALSE), or `None` (unparseable). `is_correct = (parsed == ground_truth)`;
  unparseable items are scored as unparseable, **not** as random noise.
- **Robustness (added this run — see §4):** 60 s per-request timeout, 3 retries
  with backoff, resumable progressive CSV writes, per-item exception capture.
- **Two decoding conditions (ablation):**
  - **No-thinking (primary):** Gemini's internal reasoning disabled
    (`thinking_budget=0`), `max_output_tokens=10`. Model answers YES/NO directly
    — behaviourally matched to the open-weight Qwen tiers, which also answer in a
    single word without extended reasoning.
  - **Thinking (secondary/ablation):** reasoning left ON (dynamic),
    `max_output_tokens=1024`. Measures what Gemini's chain-of-thought buys on
    these items. Labeled `gemini-2.5-flash-thinking` in its own CSV.

---

## 4. Execution log & issues encountered (Gemini tier)

The pre-written eval script had never been run against the live Gemini API
(only the parser, curation, and analysis scripts were pre-tested). Four distinct
problems surfaced on first contact; each is documented here because they are
directly relevant to the report's Methods and Limitations, and because the fixes
are part of the reproducible methodology.

**Issue 1 — Free-tier rate limit (HTTP 429).**
First run errored on ~30/40 items with
`RESOURCE_EXHAUSTED … quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier,
limit: 5`. The free tier caps `gemini-2.5-flash` at **5 requests/minute**; the
script's default 0.5 s spacing (~120/min) far exceeds it.
*Resolution:* billing enabled on the Google project (`edge_relational_reasoning`,
Tier 1, ~1000 rpm). Verified with an 8-call burst that previously would have
429'd. Cost is negligible (see §5.4).

**Issue 2 — Empty responses (thinking consumed the token budget).**
After the rate limit was fixed, ~84% of responses came back **empty**
(unparseable). Root cause confirmed by inspecting `finish_reason` and token
usage: `gemini-2.5-flash` is a *thinking* model that spends output tokens on
hidden reasoning **before** the visible answer. With `max_output_tokens=10`, the
reasoning consumed the entire budget → `finish_reason = MAX_TOKENS` and empty
`response.text`.
*Resolution:* set `thinking_config.thinking_budget = 0` for the primary run.
This is also the more defensible cross-model choice — it makes Gemini answer
directly like the Qwen tiers, so the comparison isolates scale/architecture, not
decoding strategy. (The thinking condition is preserved separately as the §6
ablation.)

**Issue 3 — Intermittent indefinite hangs.**
The run repeatedly froze mid-stream (stuck at 41, then 46 items on successive
attempts). Diagnosis: `lsof` showed a single open HTTPS connection to a Google
IP that never returned, and the google-genai client was constructed with **no
request timeout**, so a stalled call blocked forever. Not item-specific (the
stuck items' images were already cached; it hung on the API call).
*Resolution:* client-level 60 s timeout (`HttpOptions(timeout=60_000)`) plus a
3-attempt retry loop with backoff around each call. A stalled call now aborts at
60 s and retries in place rather than producing a permanent error row (which the
resume logic would otherwise skip, silently dropping the item). After this fix
the run completed 100/100 with 0 errors; one item shows ~64 s latency — that is
the timeout+retry recovering from exactly this failure mode.

**Issue 4 — Resume-skips-errors interaction (design note).**
The script's resume logic skips any `item_id` already present in the output CSV,
**including rows written as errors**. This means a naive rerun does not retry
failed items. Handled operationally by discarding polluted partial CSVs before
each clean rerun, and structurally by Issue 3's in-place retry (so failures are
recovered before a row is ever written). Worth stating as a reproducibility
caveat.

**Net effect on data quality:** the final `results/gemini_2_5_flash.csv` has
**0 errors and 0 unparseable rows** across all 100 items — a fully clean ceiling
dataset. The `image_url` and `caption` columns were merged in afterward (from
the eval set, joined on `item_id`) to support manual spot-checking of individual
images against VSR's ground-truth labels.

---

## 5. Results — `gemini-2.5-flash`, no-thinking (primary)

**File:** `results/gemini_2_5_flash.csv` · 100/100 items · 0 errors · 0 unparseable.

### 5.1 Accuracy
| Category | Correct | Accuracy | Mean latency |
|---|---:|---:|---:|
| projective_spatial | 41/50 | **82.0%** | 1.36 s |
| topological_containment | 40/50 | **80.0%** | 4.03 s |
| **Overall** | **81/100** | **81.0%** | 2.70 s |

The two categories are nearly tied (82% vs 80%). At n=50/category this 2-point
gap is well within noise, so **at the ceiling tier there is no meaningful
projective-vs-containment difference** — a relevant baseline for testing
whether *smaller* models degrade non-uniformly across relation types (hypothesis
H1).

### 5.2 Answer asymmetry (bias analysis)
Ground truth is balanced 50 True / 50 False, but the model is not symmetric:

| | Correct | Accuracy |
|---|---:|---:|
| TRUE captions (should answer YES) | 35/50 | **70%** |
| FALSE captions (should answer NO) | 46/50 | **92%** |

Confusion matrix (rows = truth, cols = prediction):

|            | pred NO | pred YES |
|------------|--------:|---------:|
| **truth NO (false caption)**  | 46 | 4 |
| **truth YES (true caption)**  | 15 | 35 |

Predictions overall: **61 NO / 39 YES**. The model has a pronounced **negative
(rejection) bias** — it is excellent at rejecting incorrect spatial claims (92%)
but misses nearly a third of correct ones (30% false-negatives). This matters
for interpretation: a headline "81% accuracy" masks that most of the error is
concentrated in *failing to confirm true relations*, not in hallucinating false
ones.

Broken out by category, the asymmetry is stronger for containment:

| Category | TRUE-caption acc | FALSE-caption acc |
|---|---:|---:|
| projective_spatial | 19/25 (76%) | 22/25 (88%) |
| topological_containment | 16/25 (64%) | 24/25 (96%) |

So containment's overall 80% is built from near-perfect rejection (96%) but only
64% recall on true containment claims — the model is reluctant to affirm that
something is genuinely inside/contains/at-the-edge-of something else.

### 5.3 Latency
Mean 2.70 s/item; containment (4.03 s) slower than projective (1.36 s). Max
64.5 s on a single item (the Issue-3 timeout+retry recovery). Latency here is
dominated by API round-trips and is **not** a hardware-comparison signal — that
role belongs to the Mac and Jetson tiers.

### 5.4 Cost
Primary (no-thinking) run: ~1–3 cents total (100 small image calls, ~1-token
outputs). Trivial.

### 5.5 Manual verification hooks
Two disagreements flagged for eyeball checks against VSR ground truth (VSR labels
are known to contain some noise):
- `vsr_0010` — "The couch is under the truck." VSR=True, Gemini=No.
- `vsr_0015` — "The laptop is facing the person." VSR=False, Gemini=Yes.
Any confirmed VSR label errors should be logged here and reported as a measured
label-noise rate — a credibility point, not a weakness.

---

## 6. Results — `gemini-2.5-flash`, thinking (ablation)

**File:** `results/gemini_2_5_flash_thinking.csv` (label `gemini-2.5-flash-thinking`)
· 100/100 · 0 errors · 0 unparseable · mean latency 4.82 s/item
(~980 thinking tokens/item; full-run cost ≈ $0.25).

### 6.1 Accuracy (thinking ON)
| Category | Correct | Accuracy | Mean latency |
|---|---:|---:|---:|
| projective_spatial | 43/50 | **86.0%** | 4.91 s |
| topological_containment | 40/50 | **80.0%** | 4.73 s |
| **Overall** | **83/100** | **83.0%** | 4.82 s |

Answer asymmetry, thinking ON: TRUE-caption 38/50 (**76%**), FALSE-caption 45/50
(**90%**); predictions **57 NO / 43 YES**. The rejection bias persists but is
slightly softened relative to no-thinking (TRUE recall 76% vs 70%; predictions
57/43 vs 61/39) — reasoning makes the model marginally more willing to affirm
true relations.

### 6.2 Thinking vs no-thinking — the ablation result
| Condition | Overall | Projective | Containment | TRUE recall | Latency |
|---|---:|---:|---:|---:|---:|
| No-thinking (primary) | 81.0% | 82% | 80% | 70% | 2.70 s |
| Thinking | 83.0% | 86% | 80% | 76% | 4.82 s |
| **Δ** | **+2.0 pp** | +4 pp | 0 | +6 pp | +2.1 s |

Item-level paired agreement (n=100):

|            | thinking wrong | thinking right |
|------------|---------------:|---------------:|
| **no-thinking wrong** | 11 | 8 |
| **no-thinking right** | 6 | 75 |

- 75 items both got right, 11 both got wrong.
- Discordant pairs: thinking **fixed** 8, **broke** 6 → net +2 items.
- **McNemar exact test** on the 14 discordant pairs: **p = 0.791** →
  the +2 pp difference is **not statistically significant**.

**Conclusion (report-ready):** enabling Gemini's chain-of-thought yields a small,
statistically indistinguishable accuracy change (+2 pp, p = 0.79) at more than
1.7× the latency, and mostly reshuffles which items are right/wrong rather than
uniformly improving them. This **validates the primary no-thinking configuration**:
it costs no measurable accuracy while keeping the decoding strategy identical to
the open-weight Qwen tiers, so the cross-model comparison isolates scale and
architecture rather than reasoning strategy. The +6 pp on TRUE-caption recall is
the one directional signal worth mentioning (reasoning slightly mitigates the
under-confirmation bias), but it too falls inside pilot-scale noise.

---

## 7. Interpretation so far

- **Ceiling behaviour is strong but not saturated (81%).** There is clear
  headroom, which is desirable for a study about degradation: if the ceiling
  were ~99% the smaller tiers would have nowhere to fall from.
- **Relation type barely matters at the ceiling** (82% vs 80%). If the Qwen
  tiers show a *widening* projective-vs-containment gap as scale drops, that is
  positive evidence for H1 (non-uniform degradation by relation type).
- **The dominant error mode is under-confirmation of true relations**, strongest
  for topological containment (64% recall). This is a more interesting and
  specific finding than a single accuracy number and should anchor the
  interpretation section.
- **Reasoning is not the lever at the ceiling.** The thinking ablation shows
  chain-of-thought does not significantly move accuracy (+2 pp, p = 0.79) on
  these yes/no spatial items, which both (a) justifies the clean no-thinking
  comparison design and (b) suggests the errors are perceptual/grounding
  failures (the model mis-reads the spatial configuration in the image) rather
  than reasoning failures that more deliberation would fix. That framing will
  carry into the interpretation of the smaller Qwen tiers.

---

## 8. Artifacts produced (Gemini tier)

| Path | Contents |
|---|---|
| `data/pilot_eval_set_n50.csv` | 100 curated items (seed 42), + caption/image_url |
| `results/gemini_2_5_flash.csv` | No-thinking run; 100 rows; +caption/image_url |
| `results/gemini_2_5_flash_thinking.csv` | Thinking run; 100 rows; +caption/image_url |
| `logs/gemini_run_final.log`, `logs/gemini_thinking_run.log` | Run stdout |
| `notebooks/PILOT_RUN_LOG.md` | This document |

**Script changes made to `scripts/run_gemini_eval.py`** (all others untouched):
1. `thinking_config(thinking_budget=0)` default (Issue 2).
2. 60 s client timeout + 3-retry loop (Issue 3).
3. New `--thinking` and `--label` flags enabling the ablation without
   duplicating the script (one script, two configs, distinct CSV labels).

**Next tiers:** Mac Qwen2.5-VL-7B (Ollama, README Steps 4–5), then Jetson
Qwen2.5-VL-3B (README Steps 6–7). Both append to §5-style results blocks here.
