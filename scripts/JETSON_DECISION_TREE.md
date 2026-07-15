# Jetson Monday-Morning Decision Tree

**Purpose:** systematically determine which inference toolchain will actually
run Qwen2.5-VL-3B on your Orin Nano, without letting toolchain debugging eat
your Monday budget.

**Hard time budget:** 2 hours max for toolchain selection. If none of the paths
below works within that window, take the fallback exit (Section 5) and file
the report as pilot-methodology-with-Mac-and-Gemini-only. That is a defensible
outcome, not a failure.

## 0. Before you start — sanity checks (10 min)

SSH from your Mac into the Jetson:
```
ssh tabeen@<jetson-ip>
```

Confirm firmware and system health:
```
sudo nvbootctrl dump-slots-info    # should show 36.4.7
cat /etc/nv_tegra_release          # should show R36 (release), REVISION: 4.x
free -h                            # should show ~7-8G of usable memory
df -h                              # confirm you have >5GB free on / for model weights
```

Install `jtop` if not already installed (from last night's plan):
```
which pip3 || sudo apt install python3-pip -y
sudo -H pip3 install -U jetson-stats
sudo systemctl restart jtop
```

Log out and back in, then run `jtop` briefly. It must correctly report:
- Module: **NVIDIA Jetson Orin Nano** (not "Nano" without Orin)
- Memory: **8 GB**
- JetPack: **6.x**

If `jtop` shows wrong info or crashes, STOP and report back before continuing.

## 1. Path A: llama.cpp with CUDA (try first)

**Why first:** documented working on Orin Nano by external projects
(QuantFT-VL is one), supports GGUF format which Ollama also uses (so the
model file we download can be reused if we pivot), and produces the same
quantization family (GGUF k-quants) as the Mac's Ollama runtime, which
minimizes cross-runtime confounds in the paper.

**Time cap on this path: 45 minutes.**

### 1A. Install build tooling
```
sudo apt update
sudo apt install -y build-essential cmake git ccache libcurl4-openssl-dev
```

### 1B. Clone and build llama.cpp with CUDA support
```
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j 4
```

Build takes 15-30 minutes on Orin Nano. While it runs, do NOTHING that
touches the terminal — walk away.

### 1C. Verify build succeeded
```
ls build/bin/llama-mtmd-cli    # multimodal CLI; this is what we need
build/bin/llama-mtmd-cli --help | head -20
```

If `llama-mtmd-cli` doesn't exist, the multimodal build target may have been
renamed in a newer version. Check `ls build/bin/ | grep -i mtmd` and use
whatever multimodal binary exists.

### 1D. Pull a Qwen2.5-VL-3B GGUF from HuggingFace
```
mkdir -p ~/models
cd ~/models
# Verified path — this is a widely-used GGUF conversion
wget https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf
wget https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf
```

Download is roughly 2-3 GB total. If this exact HF path is 404, fall back to
searching the HF hub for "Qwen2.5-VL-3B-Instruct GGUF" and pick a maintained
conversion. Note the exact HF repo/version you use — this MUST go in the
paper's methodology section for reproducibility.

### 1E. Smoke test with a single image
```
cd ~/llama.cpp
build/bin/llama-mtmd-cli \
    -m ~/models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf \
    --mmproj ~/models/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf \
    --image /path/to/any/test.jpg \
    -p "What is in this image? Answer in one sentence." \
    -n 50
```

**Success criteria:**
- Model loads without OOM
- Produces coherent text output
- Completes in under 60 seconds

If yes → path A works, skip to Section 4.
If OOM → try Q4_0 quant or reduce context with `-c 2048`.
If garbled output → check the mmproj file matches the model version.
If takes >60 seconds → note the time and continue (may still be workable).

### 1F. If Path A fails or blows the 45-min cap
Move to Path B. Do NOT keep debugging past 45 min — write down what you
tried, what error you saw, and move on.

## 2. Path B: Ollama on Jetson (try second)

**Why second:** dramatically simpler install (one command), but Ollama on
Jetson uses a slightly different CUDA integration than on Mac, so we can't
be certain Qwen2.5-VL-3B will run cleanly there. Worth trying because if it
works, the eval script is a near-copy of the Mac one.

**Time cap on this path: 30 minutes.**

### 2A. Install
```
curl -fsSL https://ollama.com/install.sh | sh
```

### 2B. Pull the model
```
ollama pull qwen2.5vl:3b
```

Download is ~2 GB. If pull hangs or errors, path B fails.

### 2C. Smoke test
```
ollama run qwen2.5vl:3b "describe this image" < /path/to/any/test.jpg
```
(or use the Python client as in `run_ollama_eval.py`)

Same success criteria as Path A.

## 3. Path C: MLC-LLM (try only if A and B both fail)

**Time cap: 30 minutes.**

Follow: https://llm.mlc.ai/docs/install/mlc_llm.html for Jetson.

This path has more moving pieces (needs TVM-Unity, needs matching model conversion). Only worth attempting if A and B both fail, because setup can eat hours quickly.

## 4. When a path succeeds — write the eval script

Once you know which toolchain works, adapt `run_ollama_eval.py` as your
starting point. The core loop is unchanged; only the model invocation is
different. Save the working eval script as
`scripts/run_jetson_eval.py`.

Points to preserve exactly from the other eval scripts:
- Import `parse_response`, `SYSTEM_INSTRUCTION`, `USER_PROMPT_TEMPLATE`
  from `run_gemini_eval.py`. Do not rewrite these — the whole point of
  cross-model comparison is that the prompt is identical.
- Progressive CSV write (`fh.flush()` after every row).
- Latency measurement per item.
- Item-level exception handling so one bad item doesn't kill the run.

Record the following in the paper's methodology, verbatim:
- Toolchain and exact version (git commit hash for llama.cpp)
- Model file: exact HF repo, filename, quantization variant
- Any command-line flags that affected behavior
- Peak memory during a typical inference (from `jtop`)

## 5. Fallback exit — Jetson-less pilot (if all paths fail)

If after 2 hours you don't have a working toolchain, stop.

Rewrite the pilot report as: **two-model preliminary study (Mac Qwen2.5-VL-7B
and Gemini 2.5 Flash) with Jetson deployment scoped for follow-up.**

Real losses from this fallback:
- H3 (cloud-vs-hardware transfer) becomes untestable this week
- The paper's "on real edge hardware" claim gets deferred
- The report loses ~30% of its "innovativeness" story

Real preserved value:
- Methodology fully demonstrated (curation, scoring, metrics, McNemar)
- Two-point compute-spectrum evidence (7B open vs. frontier API)
- H1 (relation-type non-uniform degradation) still testable
- Full paper timeline is unaffected — Jetson slots into the next report cycle

Fallback is not the same as failure. State it plainly in the report:
*"Pilot data collection completed on 2 of 3 planned model tiers. Jetson tier
deferred to Week [n+1] pending toolchain resolution; findings from that tier
will be added to the full paper."*

## Common Failure Modes and What They Actually Mean

- **`CUDA out of memory` during llama.cpp inference** → try Q4_0 instead of
  Q4_K_M, or add `-c 2048` to reduce context window.
- **`illegal instruction (core dumped)`** → CUDA build didn't happen; the
  binary is CPU-only. Rebuild with `-DGGML_CUDA=ON` explicitly.
- **Model loads but produces garbled output** → mmproj file mismatch. The
  `.mmproj` must be the vision projector from the SAME model version as
  the main GGUF weights.
- **Inference is 30+ seconds per item** → SD card I/O is likely the
  bottleneck. Model files should ideally live on NVMe SSD if available.
  If stuck on SD, note the latency honestly in the paper.
