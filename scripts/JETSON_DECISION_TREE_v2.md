# Jetson Decision Tree — v2

**What changed from v1 (and why):**
1. **Path order flipped — Ollama first, llama.cpp second.** For `qwen2.5vl`, Mac ran
   Ollama's *own* multimodal engine (introduced May 2025), **not** llama.cpp's `mtmd`
   path. So Ollama-on-Jetson holds the inference engine constant with the Mac; llama.cpp
   would vary hardware *and* engine at once, muddying H3. v1's rationale for llama.cpp-first
   ("shares the quant family, minimizes confounds") was backwards for vision models.
2. **H2 rescoped** to a within-3B precision claim (see §H). This removes the need for a
   7B precision arm — the 3B Q4-vs-Q8 runs are sufficient for the rescoped claim.
3. **Both quant tags** (`q4_K_M` and `q8_0`) are now in scope on the Ollama path.
4. **Fallback reframed** (§5): a total Jetson failure does **not** drop you to two models —
   the submitted report already has three scales. Only H2/H3 defer.
5. Smoke-test command fixed, disk threshold raised, quant-digest pre-check added.

**Hard time budget:** 2 hours max for toolchain selection (sanity 10 + Ollama 30 +
llama.cpp 45 + MLC 30 = 115 worst-case). This cap is for *getting a toolchain working* —
the actual eval runs are a separate, uncapped block (§6).

**Hardware-risk note:** everything here is userspace (install + inference). No bootloader /
QSPI / firmware writes — no brick risk. Run attended anyway, per your own standing rule.

---

## Prep status (done Mac-side, before the on-device session)

- ✅ **Quant confirmed: Mac `qwen2.5vl:3b` = Q4_K_M** — verified two ways (Ollama config
  blob + the GGUF weight header itself, `general.file_type = 15 → Q4_K_M`). The live
  `ollama show` could not run (Mac port/TIME_WAIT exhaustion), so this was read from disk.
  Mac weight-blob digest: `e9758e589d44...`. So `mac_qwen25vl_3b.csv` is a valid
  matched-precision pair for Jetson `qwen2.5vl:3b-q4_K_M` — **pending the on-device digest
  check below.**
- ✅ **Nesting verified:** `data/pilot_eval_set_n50.csv` (100 rows) nests cleanly in
  `data/pilot_eval_set_n150.csv` (300 rows), 100/100, 0 missing. (Note: the "n50/n150"
  names do not equal literal row counts — confirm what the naming refers to and document it.)
- ✅ **`scripts/run_jetson_eval.py` written and reviewed.** Takes `--model` and input-CSV
  path as args (both quant tags run with one script); `--label` defaults to `jetson-<model>`
  so Jetson runs never collide with Mac runs in analysis; progressive flushed CSV write;
  per-item latency; per-item exception handling. Plus `--limit N` (first N input rows, pre-
  resume) for smoke tests and a per-item inference timeout (default 120s) that interrupts a
  genuine hang, not just an exception.
- ⚠️ **Mac Ollama currently unreachable** (TIME_WAIT port exhaustion — unrelated to Jetson).
  Doesn't block prep. **Reboot the Mac before driving the real runs over SSH**, or the same
  "can't assign requested address" error can hit mid-run.

---

## Phase 0 — precision match (DONE Mac-side; one on-device check remains)

**Done:** Mac `qwen2.5vl:3b` is confirmed Q4_K_M (see Prep status). No Mac rerun needed.

**Still to do, on the Jetson, after Path B pulls the model** — verify it's the same *weights*,
not just the same quant *scheme name*. "Q4_K_M" from a different conversion is not
guaranteed byte-identical:

```
ollama show qwen2.5vl:3b-q4_K_M    # compare its weight-blob digest to the Mac's e9758e589d44...
```

- Digests match → true matched-weights pair; H3 is "same weights, different hardware" (strong).
- Digests differ → still testable, but report H3 as "same quant scheme, different conversion
  + different hardware" (weaker but honest). Record whichever it is for the methodology.

---

## Phase 1 — Jetson sanity checks (10 min)

SSH in, confirm health:

```
ssh tabeen@<jetson-ip>
sudo nvbootctrl dump-slots-info    # expect 36.4.7
cat /etc/nv_tegra_release          # expect R36 (release), REVISION: 4.x
free -h                            # expect ~7-8G usable
df -h                              # need >10GB free on / (both quants ≈ 8GB + headroom)
```

`jtop` must report: Module **NVIDIA Jetson Orin Nano**, Memory **8 GB**, JetPack **6.x**.
If `jtop` is wrong or crashes, STOP and report back.

---

## Phase 2 — Path B→A→C (2-hour cap total)

### Path B: Ollama (TRY FIRST — same engine as Mac). Cap: 30 min.

```
curl -fsSL https://ollama.com/install.sh | sh   # native installer, version-adaptive for r36.x
ollama pull qwen2.5vl:3b-q4_K_M                  # 3.2 GB
ollama pull qwen2.5vl:3b-q8_0                    # 4.6 GB
```

**Smoke test — use the real code path, not a hand-typed command.** (v1's
`ollama run ... < image.jpg` stdin redirect is not how Ollama ingests images.) Run the actual
eval script with `--limit 1` against the pilot set:

```
python scripts/run_jetson_eval.py --model qwen2.5vl:3b-q4_K_M \
    --input data/pilot_eval_set_n50.csv --limit 1
```

- Success: loads without OOM, returns a parseable yes/no, **note the per-item seconds** —
  that number decides whether the full 600-inference run is reasonable.
- This is the first time the Ollama image-passing call runs for real; approval-on-review
  ≠ confirmed-working. This smoke test is the actual proof it works.
- If OOM: unlikely at 8GB for a 3B, but set `OLLAMA_KV_CACHE_TYPE=q8_0` / reduce context.
- If it works → proceed to Phase 4 (pilot).

Advantages locked in by staying on Ollama: projector is bundled in the manifest (no
mmproj version-match failure mode), and the engine matches the Mac exactly.

### Path A: llama.cpp + CUDA (only if Ollama fails). Cap: 45 min.

⚠️ **This changes the science.** llama.cpp uses a *different* multimodal engine than Mac
Ollama. If you land here, H3 now varies hardware AND engine — state that explicitly in the
paper; it is a real confound, not a footnote. Also: "both quants" means **two** GGUF
downloads here, not one.

```
sudo apt update && sudo apt install -y build-essential cmake git ccache libcurl4-openssl-dev
cd ~ && git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j 4          # 15-30 min; walk away, don't touch the terminal
ls build/bin/llama-mtmd-cli                         # or: ls build/bin/ | grep -i mtmd
```

Download BOTH quant GGUFs + the matching mmproj (note exact HF repo/version for the paper):

```
mkdir -p ~/models && cd ~/models
wget <HF>/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf
wget <HF>/Qwen2.5-VL-3B-Instruct-Q8_0.gguf
wget <HF>/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf
```

Smoke test one image with `llama-mtmd-cli` (see v1 §1E). OOM → `-c 2048` or Q4_0.
Garbled → mmproj/model version mismatch. `illegal instruction` → CPU-only build, rebuild
with `-DGGML_CUDA=ON`.

### Path C: MLC-LLM (only if A and B fail). Cap: 30 min.
Per https://llm.mlc.ai/docs/install/mlc_llm.html. More moving parts; last resort.

---

## Phase 3 — eval script (DONE — `scripts/run_jetson_eval.py`)

Already written and reviewed (see Prep status). Imports the shared prompt/parser; `--model`
+ input-CSV args; `--label` (`jetson-<model>` default); progressive flushed write; per-item
latency + exception handling; `--limit N` for smoke tests; per-item hang timeout (120s).

Still record verbatim for methodology when you run it: toolchain + exact version, model
digest/quant, any behavior-affecting flags, peak memory from `jtop`.

---

## Phase 4 — pilot before full (separate block, latency-gated)

Run the pilot set (`data/pilot_eval_set_n50.csv`, 100 rows) first, **both quant tags**. This
catches OOM / timeout / parser issues on the small set instead of partway through the full
run. The per-item latency here tells you how long the full run will actually take before you
commit to it.

---

## Phase 5 — full runs

Full set (`data/pilot_eval_set_n150.csv`, 300 rows), **both quant tags** = **600 Jetson
inferences**. Edge hardware is slow and the rate is unknown until Phase 4 — budget a
dedicated block, don't schedule it blind. 30+ s/item usually means SD-card I/O; move model
files to NVMe if available, else note the latency honestly.

**Before starting: reboot the Mac** if you're driving these runs over SSH — clears the
TIME_WAIT port exhaustion so the session doesn't die mid-run with "can't assign requested
address." The progressive-write design means a mid-run death only loses in-flight rows, but
avoid it.

---

## Phase 6 — analysis (tie each run to its hypothesis)

- **H2 (rescoped, within-3B):** 3B Q4 vs Q8 on Jetson, paired McNemar (same method as the
  report). The claim to test: precision reduction degrades the 3B, **concentrated in
  projective relations** — so check the category split, not just the aggregate. Do **not**
  phrase it as "disproportionately vs larger models" — that's out of scope now.
- **H3 (hardware transfer):** Mac-3B vs Jetson-3B at **matched precision** (from Phase 0).
  Same model, same quant, different hardware → is accuracy the same? Divergence = a real
  cloud-to-hardware gap worth reporting.

Extend `analyze_pilot.py` with the Jetson tier(s); reuse the existing McNemar + category
stratification code.

---

## §5. Fallback exit — corrected

If no toolchain works in 2 hours, stop. You do **NOT** fall back to a two-model study —
the submitted report already contains three scales (3B / 7B / Gemini). A Jetson failure
means only:

- **Deferred:** H2 (precision) and H3 (cloud-to-hardware transfer), and the "on real edge
  hardware" claim.
- **Fully preserved:** the entire three-scale H1 result (category-stratified degradation),
  methodology, McNemar framework, and the thinking-mode control — all already submitted.

State plainly: *"On-device (Jetson) evaluation deferred pending toolchain resolution; H2/H3
to be added to the full paper. Three-scale results (H1) are complete."* This is a bounded,
honest deferral of two hypotheses — not a collapse of the study.

---

## §H. Why H2 was rescoped (keep for the paper's framing)

H2 as originally written — "precision reduction affects smaller models *disproportionately*"
— is inherently comparative and needs the precision effect measured at **two** model sizes
(3B vs 7B). The current plan runs only the 3B at two precisions, which cannot establish
"disproportionate." Rather than add a 7B precision arm, H2 is rescoped to the claim the data
*can* support: **within the 3B, does lower precision (Q4 vs Q8) degrade accuracy, and is the
damage concentrated in projective-spatial relations** (mirroring the scale finding)? If the
full paper later wants the cross-size "disproportionate" claim, add 7B Q4-vs-Q8 on the Mac
(quantization loss is largely hardware-portable) — but that is explicitly out of scope for
this cycle.
