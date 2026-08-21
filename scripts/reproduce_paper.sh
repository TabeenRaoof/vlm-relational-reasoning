#!/usr/bin/env bash
#
# reproduce_paper.sh
#
# Regenerates every analysis output and figure reported in the paper, from the
# committed per-item result files. Run from the repository root:
#
#     bash scripts/reproduce_paper.sh
#
# Nothing here re-runs inference — the model evaluations are the expensive,
# hardware-bound part and their outputs are committed under results/. This
# script recomputes the STATISTICS from those outputs, which takes ~2 minutes.
#
# See docs/PAPER_MAP.md for which claim each output backs.

set -euo pipefail

PY="${PY:-./venv/bin/python}"
if [ ! -x "$PY" ]; then
    PY="python3"
fi
echo "Using interpreter: $PY"
echo

echo "=== [1/6] Study 1 — pilot analysis (n=300) ==="
$PY scripts/analyze_pilot.py \
    --results-dir results/pilot_n300 \
    --output-dir results/analysis_full \
    --model-order "jetson-qwen2.5vl:3b-q4_K_M,qwen2.5vl:3b,qwen2.5vl:7b,gemini-2.5-flash" \
    || echo "  (see docs/PIPELINE_NOTES.md if this step reports a model-order error)"
echo

echo "=== [2/6] Study 2 — pre-registered interaction test (n=2000) ==="
$PY scripts/interaction_test.py \
    --results-dir results \
    --model-a qwen2.5vl:3b \
    --model-b qwen2.5vl:7b \
    --output results/analysis_n2000/interaction_primary.txt
echo

echo "=== [3/6] Study 3 — pre-registered TOST equivalence (n=2000) ==="
$PY scripts/h3_equivalence_test.py \
    --jetson results/jetson_qwen25vl_3b_q4_n2000.csv \
    --mac    results/mac_qwen25vl_3b_n2000.csv \
    --margin 3.0 \
    --output results/analysis_h3_n2000/equivalence.txt
echo

echo "=== [4/6] Sections 5.2.1-5.2.2 — corrected H1 analyses (Table 4) ==="
$PY scripts/h1_corrected_analyses.py \
    --low  results/mac_qwen25vl_3b_n2000.csv \
    --high results/mac_qwen25vl_7b_n2000.csv \
    --eval-set  data/eval_set_n2000.csv \
    --pilot-set data/pilot_eval_set_n150.csv \
    --output results/analysis_paper/h1_corrected.txt
echo

echo "=== [5/6] Robustness, power, and latency (Sections 5.2, 5.3, 5.5, 6) ==="
$PY scripts/robustness_checks.py \
    --low  results/mac_qwen25vl_3b_n2000.csv \
    --high results/mac_qwen25vl_7b_n2000.csv \
    --jetson results/jetson_qwen25vl_3b_q4_n2000.csv \
    --eval-set data/eval_set_n2000.csv \
    --output results/analysis_paper/robustness.txt

$PY scripts/power_projection.py \
    --output results/analysis_paper/power_projection.txt

$PY scripts/latency_analysis.py \
    --jetson results/jetson_qwen25vl_3b_q4_n2000.csv \
    --mac    results/mac_qwen25vl_3b_n2000.csv \
    --output results/analysis_paper/latency.txt
echo

echo "=== [6/6] Figures 1-4 ==="
$PY scripts/make_paper_figures.py --outdir results/figures
echo

echo "Done. Outputs:"
echo "  results/analysis_full/        Study 1 (pilot)"
echo "  results/analysis_n2000/       Study 2 (confirmatory)"
echo "  results/analysis_h3_n2000/    Study 3 (equivalence)"
echo "  results/analysis_paper/       corrected + robustness + power + latency"
echo "  results/figures/              Figures 1-4"
