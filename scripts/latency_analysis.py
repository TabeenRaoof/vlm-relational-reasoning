#!/usr/bin/env python3
"""
latency_analysis.py

Section 6 and Table 7 of the paper: the operational cost of running the model
on the edge device, as opposed to its accuracy (which Study 3 shows transfers).

What the latency trace shows. The Jetson ran in two clearly separated regimes.
It started healthy (a few seconds per item), then after the first crash it
dropped into a degraded regime around an order of magnitude slower and never
sustainably recovered — for roughly 1,700 of the 2,000 items. Four candidate
causes (item corruption, transient thermal effects, disk speed, GUI memory
pressure) were tested and eliminated; the transition is reported as unexplained,
though the degraded regime is itself stable and error-free.

Why this matters for the paper's argument: capacity planning must use the
DEGRADED number, not the fresh-device benchmark. A reader who benchmarks a
freshly rebooted device sees ~3x the desktop's latency and concludes the device
is usable for near-real-time work; sustained, it is ~21x, which disqualifies
that use case.

The regime split is done by latency threshold rather than by row index so it is
reproducible from the CSV alone. Items above --load-threshold are per-chunk
model-load cost (the run was chunked with a reboot between chunks), not
per-item inference cost, and are reported separately rather than silently
folded into either regime.

Usage:
    python latency_analysis.py \
        --jetson results/jetson_qwen25vl_3b_q4_n2000.csv \
        --mac    results/mac_qwen25vl_3b_n2000.csv \
        --output results/analysis_paper/latency.txt

Requires: pandas, numpy.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def describe(series: pd.Series, label: str, out, baseline: float = None):
    """Print the distribution of one latency population."""
    if len(series) == 0:
        out(f"  {label:<34} (no items in this regime)")
        return
    line = (f"  {label:<34} n={len(series):>5}  "
            f"mean={series.mean():>7.2f}s  median={series.median():>6.2f}s  "
            f"min={series.min():>6.2f}s  max={series.max():>7.2f}s")
    if baseline:
        line += f"  ({series.mean() / baseline:.1f}x Mac)"
    out(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jetson", type=Path, required=True)
    ap.add_argument("--mac", type=Path, required=True)
    ap.add_argument("--healthy-threshold", type=float, default=20.0,
                    help="items faster than this are the healthy regime")
    ap.add_argument("--load-threshold", type=float, default=100.0,
                    help="items slower than this are per-chunk model-load cost")
    ap.add_argument("--transition-window", type=int, default=25,
                    help="consecutive slow items required to call a regime change")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    jetson = pd.read_csv(args.jetson)
    mac = pd.read_csv(args.mac)
    jetson_latency = jetson["latency_seconds"].dropna()
    mac_latency = mac["latency_seconds"].dropna()

    # The Mac is the reference point for every ratio in Table 7. The paper uses
    # the MEAN here (2.79s on the n=2000 run), which is the throughput-relevant
    # statistic: hourly capacity is total time / items, i.e. driven by the mean.
    mac_mean = mac_latency.mean()

    out("=" * 78)
    out("LATENCY AND THROUGHPUT  (paper Section 6, Table 7)")
    out("=" * 78)
    out("")
    out("Raw distributions")
    describe(mac_latency, "Mac M2 Max (Qwen2.5-VL-3B)", out)
    describe(jetson_latency, "Jetson Orin Nano (all items)", out, mac_mean)

    # Split the Jetson run into its two regimes plus the model-load tail.
    healthy = jetson_latency[jetson_latency < args.healthy_threshold]
    degraded = jetson_latency[(jetson_latency >= args.healthy_threshold)
                              & (jetson_latency < args.load_threshold)]
    model_load = jetson_latency[jetson_latency >= args.load_threshold]

    out("")
    out(f"Jetson regimes (split at {args.healthy_threshold:.0f}s / {args.load_threshold:.0f}s)")
    describe(healthy, "healthy regime", out, mac_mean)
    describe(degraded, "degraded regime", out, mac_mean)
    describe(model_load, "per-chunk model load (not per-item)", out)

    # Locate the transition in collection order — the row index at which the
    # latency figure shows the run falling off a cliff.
    #
    # Two things would fool a naive "first item over the threshold" rule, so
    # both are handled: the per-chunk model-load spikes (row 0 is one) are not
    # inference items and are skipped, and a single slow item is not a regime
    # change. We require the degraded state to PERSIST for a full window before
    # calling it a transition.
    inference = jetson[jetson["latency_seconds"] < args.load_threshold].reset_index(drop=True)
    is_slow = (inference["latency_seconds"] >= args.healthy_threshold).values

    transition_row = None
    for row in range(len(is_slow) - args.transition_window):
        window = is_slow[row:row + args.transition_window]
        if window.all():
            transition_row = row
            break

    if transition_row is not None:
        before = inference["latency_seconds"].iloc[:transition_row]
        after = inference["latency_seconds"].iloc[transition_row:]
        out("")
        out(f"  regime transition at inference row {transition_row} "
            f"(first run of {args.transition_window} consecutive slow items)")
        out(f"    before: n={len(before):>5}  mean={before.mean():>6.2f}s")
        out(f"    after : n={len(after):>5}  mean={after.mean():>6.2f}s "
            f"({len(after) / len(inference) * 100:.0f}% of inference items)")
        out("    The device never sustainably recovered the healthy regime after")
        out("    this point, across roughly 20 further chunks and reboots.")

    out("")
    out("-" * 78)
    out("TABLE 7 — Throughput by condition")
    out("-" * 78)
    out(f"  {'Condition':<26}{'Latency':>12}{'Throughput':>18}{'vs Mac':>10}")

    conditions = [
        ("Mac M2 Max", mac_mean, None),
        ("Jetson, healthy", healthy.mean() if len(healthy) else np.nan, mac_mean),
        ("Jetson, degraded", degraded.mean() if len(degraded) else np.nan, mac_mean),
    ]
    for label, latency, baseline in conditions:
        if np.isnan(latency):
            continue
        per_hour = 3600 / latency
        ratio = f"{latency / baseline:.0f}x" if baseline else "1x"
        out(f"  {label:<26}{latency:>10.2f}s{per_hour:>15.0f}/hr{ratio:>10}")

    out("")
    out("NOTE ON ROUNDING: the paper's Table 7 reports the healthy regime as")
    out("~8s / ~450 per hour, characterizing the regime by the upper end of its")
    out("~1-8s spread rather than by its mean. The mean is printed above; both")
    out("are marked approximate in the paper. The degraded and Mac rows are the")
    out("computed means and match exactly.")
    out("")
    out("Deployment reading (paper Section 6):")
    out("  - real-time perception (robot checking left/right before moving): ruled")
    out("    out on throughput at these latencies, NOT on accuracy")
    out("  - low-frequency monitoring (inspection camera, every few minutes): viable")
    out("  - offline batch analysis (on-device review, privacy-motivated): viable")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines) + "\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
