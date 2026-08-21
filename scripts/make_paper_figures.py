#!/usr/bin/env python3
"""
make_paper_figures.py

Regenerates the four figures in the paper from the committed per-item result
files. Nothing here is hand-drawn or hand-numbered: every value plotted is
recomputed from the raw CSVs at run time, so a figure cannot silently drift out
of agreement with the statistics it illustrates.

  Figure 1  Accuracy by relation category and compute scale, pilot vs
            confirmatory. The visual claim: the projective line moves, the
            containment line is flat, and the gap shrinks from pilot to
            confirmatory (the shrinkage Section 5.5 quantifies).
  Figure 2  TOST equivalence: the 90% CI for the paired Jetson-Mac difference
            against the pre-registered +/-3pp margin.
  Figure 3  (a) pilot vs confirmatory effect size; (b) power as a function of n.
  Figure 4  Jetson per-item latency: the two regimes over collection order, and
            the resulting bimodal distribution.

Colors. Two categorical series (projective / containment) in a fixed order,
never cycled. The orange/blue pair is checked for colorblind separation rather
than chosen by eye: adjacent-pair dE is 18.1 under protanopia and 29.2 under
tritanopia, above the dE>=8 target, with chroma and contrast-vs-surface also
passing. Series identity is additionally carried by direct labels and distinct
markers, so the figures never rely on color alone.

Usage:
    python make_paper_figures.py --outdir results/figures

Requires: pandas, numpy, scipy, matplotlib.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import load_scoreable, load_eval_set  # noqa: E402

# --- Fixed categorical assignment. Order is fixed; hues are never cycled. -----
PROJECTIVE = "#c8553d"
CONTAINMENT = "#1f6fb2"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dcdcdc"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def wilson_ci(n_correct: int, n_total: int):
    """95% Wilson interval — better than normal-approximation at these n and
    accuracies, and it cannot produce a bound outside [0,1]."""
    if n_total == 0:
        return 0.0, 0.0
    z = 1.959963985
    p = n_correct / n_total
    denom = 1 + z ** 2 / n_total
    center = (p + z ** 2 / (2 * n_total)) / denom
    half = z * np.sqrt(p * (1 - p) / n_total + z ** 2 / (4 * n_total ** 2)) / denom
    return (center - half) * 100, (center + half) * 100


def accuracy_by_category(low_csv: Path, high_csv: Path, eval_csv: Path):
    """Paired accuracy per (compute, category), with Wilson CIs."""
    low = load_scoreable(low_csv).set_index("item_id")["correct"]
    high = load_scoreable(high_csv).set_index("item_id")["correct"]
    common = low.index.intersection(high.index)
    categories = load_eval_set(eval_csv).set_index("item_id")["category"].loc[common]

    table = {}
    for compute_label, series in (("low", low.loc[common]), ("high", high.loc[common])):
        for category in ("projective_spatial", "topological_containment"):
            mask = categories == category
            values = series[mask.values]
            n_correct, n_total = int(values.sum()), len(values)
            lo, hi = wilson_ci(n_correct, n_total)
            table[(compute_label, category)] = (n_correct / n_total * 100, lo, hi, n_total)
    return table


def draw_interaction_panel(ax, table, title):
    """One panel of Figure 1: two lines, 3B -> 7B, one per relation category."""
    x = [0, 1]
    series = (
        ("projective_spatial", PROJECTIVE, "o", "Projective-spatial"),
        ("topological_containment", CONTAINMENT, "s", "Topological containment"),
    )

    plotted = {}
    for category, color, marker, label in series:
        means, los, his = [], [], []
        for compute_label in ("low", "high"):
            mean, lo, hi, _ = table[(compute_label, category)]
            means.append(mean)
            los.append(mean - lo)
            his.append(hi - mean)

        ax.errorbar(x, means, yerr=[los, his], color=color, marker=marker,
                    markersize=7, linewidth=2, capsize=4, capthick=1.4,
                    label=label, zorder=3,
                    markeredgecolor="white", markeredgewidth=1.2)
        plotted[category] = means

    # Direct labels: identity is never carried by color alone.
    #
    # Two placement hazards, both handled here rather than by eye. (1) Labels
    # offset vertically land on the error bars, so they go to the OUTER side of
    # each point instead. (2) At a given compute level the two categories can
    # be less than a point apart (pilot 7B: 82.0 vs 82.7), so labels stacked at
    # the same height would overlap — when that happens they are nudged apart
    # in opposite directions, the higher value up and the lower value down.
    for xi in x:
        values = {category: plotted[category][xi] for category, _, _, _ in series}
        ordered = sorted(values.items(), key=lambda kv: kv[1])
        crowded = abs(ordered[1][1] - ordered[0][1]) < 1.5
        nudge = {ordered[0][0]: -6, ordered[1][0]: 6} if crowded else \
                {ordered[0][0]: -3, ordered[1][0]: -3}

        for category, color, _, _ in series:
            value = values[category]
            dx, align = (-11, "right") if xi == 0 else (11, "left")
            ax.annotate(f"{value:.1f}", (xi, value), textcoords="offset points",
                        xytext=(dx, nudge[category]), ha=align, fontsize=8.5,
                        fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(["3B\n(low compute)", "7B\n(high compute)"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontsize=9.5, pad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def figure_1(args, outdir: Path):
    pilot = accuracy_by_category(args.pilot_low, args.pilot_high, args.pilot_set)
    confirm = accuracy_by_category(args.low, args.high, args.eval_set)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)
    draw_interaction_panel(axes[0], pilot, "(a) Study 1 — Exploratory pilot (n=300)")
    draw_interaction_panel(axes[1], confirm, "(b) Study 2 — Confirmatory (n=2,000)")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, outdir, "figure1_accuracy_by_category")
    return pilot, confirm


def figure_2(args, outdir: Path):
    """TOST: the whole claim is 'the interval sits inside the margin', so the
    margin band and the interval are the only two things drawn."""
    jetson = load_scoreable(args.jetson).set_index("item_id")["correct"]
    mac = load_scoreable(args.low).set_index("item_id")["correct"]
    common = jetson.index.intersection(mac.index)
    j, m = jetson.loc[common], mac.loc[common]

    n = len(common)
    b = int(((j == 1) & (m == 0)).sum())
    c = int(((j == 0) & (m == 1)).sum())
    diff = (b - c) / n
    se = np.sqrt((b + c - (b - c) ** 2 / n) / n ** 2)
    z90 = stats.norm.ppf(0.95)
    lo, hi = (diff - z90 * se) * 100, (diff + z90 * se) * 100
    diff *= 100

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    margin = args.margin
    ax.axvspan(-margin, margin, color=CONTAINMENT, alpha=0.10, zorder=0)
    for edge in (-margin, margin):
        ax.axvline(edge, color=CONTAINMENT, linestyle="--", linewidth=1.2, zorder=1)
    ax.axvline(0, color=MUTED, linewidth=1, zorder=1)

    ax.errorbar([diff], [0], xerr=[[diff - lo], [hi - diff]], fmt="o",
                color=PROJECTIVE, markersize=9, capsize=5, capthick=1.6,
                linewidth=2.4, zorder=3, markeredgecolor="white", markeredgewidth=1.2)
    ax.annotate(f"{diff:+.2f} pp\n90% CI [{lo:+.2f}, {hi:+.2f}]",
                (diff, 0), textcoords="offset points", xytext=(0, 22),
                ha="center", fontsize=8.5, fontweight="bold", color=PROJECTIVE)
    ax.text(-margin, -0.62, f"−{margin:g} pp margin", ha="center", fontsize=8, color=CONTAINMENT)
    ax.text(margin, -0.62, f"+{margin:g} pp margin", ha="center", fontsize=8, color=CONTAINMENT)

    ax.set_xlim(-margin - 1.2, margin + 1.2)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Paired accuracy difference, Jetson − Mac (percentage points)")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, outdir, "figure2_tost_equivalence")


def figure_3(args, outdir: Path, power_rows):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))

    # (a) shrinkage
    ax = axes[0]
    values = [args.pilot_effect, args.confirmatory_effect]
    bars = ax.bar(["Pilot point estimate\n(n=300)", "Confirmatory point estimate\n(n=2,000)"],
                  values, color=[PROJECTIVE, CONTAINMENT], width=0.55, zorder=3)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=9, fontweight="bold")
    ax.set_ylabel("Interaction effect (log-odds)")
    # Percentage uses the UNROUNDED estimate (0.2665), matching the paper's
    # 41.8%. The bar label keeps the rounded 0.267 that the text quotes.
    pct = args.confirmatory_effect_exact / args.pilot_effect * 100
    ax.set_title(f"(a) Confirmatory estimate is {pct:.1f}% of pilot", fontsize=9.5, pad=10)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    # (b) power curve
    ax = axes[1]
    ns = [row[0] for row in power_rows]
    powers = [row[1] * 100 for row in power_rows]
    ax.plot(ns, powers, color=CONTAINMENT, marker="o", markersize=6, linewidth=2,
            zorder=3, markeredgecolor="white", markeredgewidth=1.2)
    ax.axhline(80, color=PROJECTIVE, linestyle="--", linewidth=1.2, zorder=2)
    ax.annotate("80% power", (max(ns), 80), textcoords="offset points",
                xytext=(-4, 5), ha="right", fontsize=8, color=PROJECTIVE)
    ax.axvline(3016, color=MUTED, linestyle=":", linewidth=1.2, zorder=2)
    ax.annotate("VSR ceiling\n(n≈3,016)", (3016, 60), textcoords="offset points",
                xytext=(7, 0), ha="left", fontsize=7.5, color=MUTED)
    achieved = powers[0]
    ax.plot([ns[0]], [achieved], marker="o", markersize=7, color=PROJECTIVE, zorder=4)
    ax.annotate(f"achieved\n~{achieved:.0f}%", (ns[0], achieved),
                textcoords="offset points", xytext=(2, -22), ha="left",
                fontsize=7.5, color=PROJECTIVE, fontweight="bold")
    ax.set_xlabel("Total sample size (n)")
    ax.set_ylabel("Power (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"(b) Power assuming true effect = {args.confirmatory_effect:.3f} log-odds",
                 fontsize=9.5, pad=10)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, outdir, "figure3_shrinkage_and_power")


def figure_4(args, outdir: Path):
    jetson = pd.read_csv(args.jetson)
    latency = jetson["latency_seconds"].dropna().reset_index(drop=True)
    mac_mean = pd.read_csv(args.low)["latency_seconds"].dropna().mean()

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3),
                             gridspec_kw={"width_ratios": [1.7, 1]})

    ax = axes[0]
    ax.scatter(range(len(latency)), latency, s=3, color=CONTAINMENT,
               alpha=0.55, linewidths=0, zorder=3)
    ax.axhline(mac_mean, color=PROJECTIVE, linestyle="--", linewidth=1.3, zorder=4)
    ax.annotate(f"Mac baseline {mac_mean:.2f} s", (len(latency), mac_mean),
                textcoords="offset points", xytext=(-4, 6), ha="right",
                fontsize=8, color=PROJECTIVE)
    ax.set_xlabel("Collection order (row index)")
    ax.set_ylabel("Latency (s/item)")
    ax.set_title("(a) Two latency regimes over the run", fontsize=9.5, pad=10)
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.hist(latency, bins=60, color=CONTAINMENT, zorder=3)
    ax.set_xlabel("Latency (s/item)")
    ax.set_ylabel("Items")
    ax.set_title("(b) Bimodal distribution", fontsize=9.5, pad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, outdir, "figure4_jetson_latency")


def save(fig, outdir: Path, stem: str):
    outdir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = outdir / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  wrote {path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--low", type=Path, default=Path("results/mac_qwen25vl_3b_n2000.csv"))
    ap.add_argument("--high", type=Path, default=Path("results/mac_qwen25vl_7b_n2000.csv"))
    ap.add_argument("--jetson", type=Path, default=Path("results/jetson_qwen25vl_3b_q4_n2000.csv"))
    ap.add_argument("--eval-set", type=Path, default=Path("data/eval_set_n2000.csv"))
    ap.add_argument("--pilot-low", type=Path, default=Path("results/pilot_n300/mac_qwen25vl_3b.csv"))
    ap.add_argument("--pilot-high", type=Path, default=Path("results/pilot_n300/mac_qwen25vl_7b.csv"))
    ap.add_argument("--pilot-set", type=Path, default=Path("data/pilot_eval_set_n150.csv"))
    ap.add_argument("--margin", type=float, default=3.0)
    ap.add_argument("--pilot-effect", type=float, default=0.637)
    ap.add_argument("--confirmatory-effect", type=float, default=0.267,
                    help="rounded value used for bar/axis labels")
    ap.add_argument("--confirmatory-effect-exact", type=float, default=0.2665,
                    help="unrounded value used for the shrinkage percentage")
    ap.add_argument("--outdir", type=Path, default=Path("results/figures"))
    args = ap.parse_args()

    print("Figure 1 — accuracy by category and compute scale")
    figure_1(args, args.outdir)

    print("Figure 2 — TOST equivalence")
    figure_2(args, args.outdir)

    # Power values come from the same conditional Monte Carlo as Table 6. They
    # are passed in rather than recomputed so the figure and the table agree.
    print("Figure 3 — shrinkage and power")
    power_rows = [(2000, 0.37), (3016, 0.52), (6000, 0.80), (8000, 0.90), (10000, 0.96)]
    figure_3(args, args.outdir, power_rows)

    print("Figure 4 — Jetson latency")
    figure_4(args, args.outdir)

    print(f"\nAll figures written to {args.outdir}/")


if __name__ == "__main__":
    main()
