#!/usr/bin/env python3
"""
power_projection.py

Section 5.5 of the paper: effect-size shrinkage from pilot to confirmatory, and
the Monte Carlo power projection in Table 6.

The story this quantifies. The n=2000 study was sized off the pilot's
interaction estimate (0.637 log-odds), which implied ~98% power. The
confirmatory estimate came in at 0.267 log-odds — 41.8% of the pilot value — and
at THAT effect size the same design had only ~37% power. A single pilot point
estimate for an interaction carries large sampling uncertainty and is a fragile
basis for a power calculation. Note the pilot's own interaction test was not
even significant (p=0.122), which was the warning sign.

Method (matching Appendix B). Monte Carlo with the four cell probabilities
FIXED AT THEIR OBSERVED VALUES, so this is a conditional projection: "if the
confirmatory estimate is the truth, what power would n items buy?" It is not an
unconditional power analysis and should not be read as one.

The design simulated is the pre-registered, independence-assuming one, so the
projected n is what would be needed for the test as pre-registered.

Usage:
    python power_projection.py \
        --interaction-result results/analysis_n2000/interaction_primary.txt \
        --output results/analysis_paper/power_projection.txt

    (or pass the four cell accuracies explicitly with --cells)

Requires: numpy, scipy.
"""
import argparse
import re
from pathlib import Path

import numpy as np
from scipy import stats

# VSR's containment pool is the binding constraint on any single-dataset study:
# 1,508 max balanced containment items -> n ~ 3,016 total balanced.
VSR_CEILING = 3016


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def parse_cells(path: Path):
    """Pull the four observed cell accuracies out of interaction_primary.txt so
    this script cannot drift from the primary analysis it is conditioned on."""
    text = path.read_text()
    cells = {}
    # Lines look like:  'high    containment  0.836   1000'
    pattern = re.compile(
        r"^\s*(high|low)?\s*(containment|projective)\s+([01]\.\d+)\s+\d+\s*$",
        re.MULTILINE)
    current_compute = None
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        compute, category, accuracy = match.groups()
        if compute:
            current_compute = compute
        if current_compute is None:
            continue
        cells[(current_compute, category)] = float(accuracy)
    if len(cells) != 4:
        raise SystemExit(f"ERROR: expected 4 cell accuracies in {path}, got {cells}")
    return cells


def fit_deviance(design, y, weights):
    """IRLS logistic fit returning 2*loglik. Written out rather than calling
    statsmodels because this runs inside a Monte Carlo loop tens of thousands
    of times and the overhead of the full GLM machinery dominates."""
    beta = np.zeros(design.shape[1])
    for _ in range(50):
        eta = design @ beta
        prob = 1 / (1 + np.exp(-eta))
        weight = weights * prob * (1 - prob)
        gradient = design.T @ (weights * (y - prob))
        hessian = (design * weight[:, None]).T @ design
        step = np.linalg.solve(hessian + 1e-9 * np.eye(design.shape[1]), gradient)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    eta = design @ beta
    prob = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    return 2 * np.sum(weights * (y * np.log(prob) + (1 - y) * np.log(1 - prob)))


def simulate_power(cells, n_total: int, n_rep: int, alpha: float, rng) -> float:
    """Fraction of simulated studies whose interaction LRT rejects at alpha."""
    n_per_cell = n_total // 2   # balanced: half the items in each category
    rejections = 0

    for _ in range(n_rep):
        y, x_compute, x_category, weights = [], [], [], []
        for (compute, category), prob in cells.items():
            n_correct = rng.binomial(n_per_cell, prob)
            for label, count in ((1, n_correct), (0, n_per_cell - n_correct)):
                y.append(label)
                x_compute.append(1 if compute == "low" else 0)
                x_category.append(1 if category == "projective" else 0)
                weights.append(count)

        y = np.array(y, dtype=float)
        x_compute = np.array(x_compute, dtype=float)
        x_category = np.array(x_category, dtype=float)
        weights = np.array(weights, dtype=float)
        intercept = np.ones_like(x_compute)

        additive = np.column_stack([intercept, x_compute, x_category])
        full = np.column_stack([intercept, x_compute, x_category, x_compute * x_category])
        lr_stat = fit_deviance(full, y, weights) - fit_deviance(additive, y, weights)
        if stats.chi2.sf(lr_stat, 1) < alpha:
            rejections += 1

    return rejections / n_rep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interaction-result", type=Path,
                    default=Path("results/analysis_n2000/interaction_primary.txt"))
    ap.add_argument("--cells", type=float, nargs=4, default=None,
                    metavar=("LOW_PROJ", "LOW_CONT", "HIGH_PROJ", "HIGH_CONT"),
                    help="override observed cell accuracies")
    ap.add_argument("--pilot-effect", type=float, default=0.637,
                    help="pilot interaction estimate in log-odds")
    ap.add_argument("--sample-sizes", type=int, nargs="+",
                    default=[2000, VSR_CEILING, 6000, 8000, 10000])
    ap.add_argument("--n-rep", type=int, default=4000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260716)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if args.cells:
        cells = {("low", "projective"): args.cells[0],
                 ("low", "containment"): args.cells[1],
                 ("high", "projective"): args.cells[2],
                 ("high", "containment"): args.cells[3]}
    else:
        cells = parse_cells(args.interaction_result)

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    observed_effect = ((logit(cells[("high", "projective")]) - logit(cells[("low", "projective")]))
                       - (logit(cells[("high", "containment")]) - logit(cells[("low", "containment")])))

    out("=" * 70)
    out("EFFECT-SIZE SHRINKAGE AND POWER PROJECTION  (paper Section 5.5)")
    out("=" * 70)
    out("Observed cell accuracies (conditioned on, not estimated):")
    for (compute, category), prob in sorted(cells.items()):
        out(f"  {compute:>5} / {category:<12} {prob:.3f}")
    out("")
    out(f"pilot interaction estimate       : {args.pilot_effect:.3f} log-odds")
    out(f"confirmatory interaction estimate: {observed_effect:.3f} log-odds")
    out(f"shrinkage                        : {observed_effect / args.pilot_effect * 100:.1f}% of the pilot value")
    out("")
    out("A design powered off the pilot estimate (~98% power) in fact had the")
    out("power shown at n=2000 below. The pilot's own interaction test was not")
    out("significant (p=0.122) — that was the signal not to trust its magnitude.")
    out("")
    out("-" * 70)
    out("TABLE 6 — Monte Carlo power, conditional projection")
    out(f"(true effect assumed = {observed_effect:.3f} log-odds; "
        f"{args.n_rep} replicates; alpha={args.alpha})")
    out("-" * 70)
    out(f"  {'n':>8}   power")

    rng = np.random.default_rng(args.seed)
    for n_total in args.sample_sizes:
        power = simulate_power(cells, n_total, args.n_rep, args.alpha, rng)
        note = ""
        if n_total == 2000:
            note = "  <- this design"
        elif n_total == VSR_CEILING:
            note = "  <- VSR max balanced (containment pool is the ceiling)"
        elif power >= 0.80 and note == "":
            note = "  <- adequately powered"
        out(f"  {n_total:>8}   ~{power * 100:.0f}%{note}")

    out("")
    out("CONSEQUENCE: 80% power needs n ~ 6,000, which is more than twice VSR's")
    out("balanced ceiling. Resolving H1 therefore REQUIRES a second dataset")
    out("supplying containment volume, plus a fresh pre-registration — adding")
    out("items under the existing one would be optional stopping.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines) + "\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
