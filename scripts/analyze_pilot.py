#!/usr/bin/env python3
"""
analyze_pilot.py

Analysis for the CS587 pilot. Takes the per-model result CSVs produced by
run_gemini_eval.py, run_ollama_eval.py, and (Monday-morning) the Jetson
eval script, and computes:

  1. Descriptive statistics: accuracy per (model, category)
  2. Proposed metrics: RRS, DT, RCS
  3. Inferential statistics: McNemar's test on pairwise model comparisons
  4. Figures: bar charts by category, degradation-like plot

Design decisions worth stating:

  - "Unparseable" responses are reported honestly, not discarded silently.
    Accuracy is computed on scoreable items (is_parseable=True) with the
    unparseable count reported alongside. This is the difference between
    "78% accuracy" and "78% of the 87 scoreable items out of 100" — the
    latter is defensible.
  - McNemar's test uses only items where BOTH compared models produced
    parseable answers, so a model's refusal rate doesn't contaminate the
    other's accuracy comparison.
  - RCS (Relational Consistency Score) requires inverse-relation pairs to
    compute meaningfully. VSR does NOT have paired inverse items by design,
    so the pilot version of RCS is a proxy: consistency of answers on items
    that share the same relation. The full RCS from the proposal (paired
    L/R flips, etc.) requires the What'sUp dataset, which the pilot excluded
    for feasibility. This is disclosed in the report.

Usage:
    python analyze_pilot.py --results-dir results/ --output-dir results/analysis/
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# ============================================================================
# Loading & combining per-model results
# ============================================================================

def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load and concatenate every per-model CSV in results_dir.
    Skips files that don't look like model result CSVs."""
    frames = []
    for csv_path in sorted(results_dir.glob("*.csv")):
        # Skip anything in a subdirectory (e.g. results/analysis/*)
        if csv_path.parent != results_dir:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  Skipping {csv_path.name}: {e}")
            continue
        expected_cols = {"item_id", "model_name", "is_parseable", "is_correct"}
        if not expected_cols.issubset(df.columns):
            print(f"  Skipping {csv_path.name}: missing expected columns")
            continue
        df["_source_file"] = csv_path.name
        frames.append(df)
        print(f"  Loaded {csv_path.name}: {len(df)} rows, "
              f"model={df['model_name'].iloc[0]}")
    if not frames:
        raise SystemExit(f"No model result CSVs found in {results_dir}")
    combined = pd.concat(frames, ignore_index=True)

    # Defensive de-duplication. The resume-retry fix can, if a run was
    # interrupted and resumed, leave an old errored row AND a new successful row
    # for the same item (append-mode writer). Keep the LAST occurrence per
    # (model, item_id) so the successful retry wins and the McNemar join (which
    # set_index on item_id) never sees duplicate indices.
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["model_name", "item_id"], keep="last").reset_index(drop=True)
    if len(combined) < before:
        print(f"  De-duplicated {before - len(combined)} repeated "
              f"(model, item_id) rows (kept last / successful retry).")
    return combined


# ============================================================================
# Descriptive statistics
# ============================================================================

def accuracy_table(results: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per (model, category), with counts of unparseable + errors
    so the reader can judge how much of the eval set actually got scored."""
    rows = []
    for (model, category), sub in results.groupby(["model_name", "category"]):
        total = len(sub)
        scoreable = sub[sub["is_parseable"] == True]  # noqa: E712
        n_scoreable = len(scoreable)
        n_correct = scoreable["is_correct"].sum() if n_scoreable > 0 else 0
        acc = 100 * n_correct / n_scoreable if n_scoreable > 0 else float("nan")
        # 95% CI via normal approximation on proportion (Wald interval)
        # This is coarse; for a paper you'd want Wilson or Clopper-Pearson.
        if n_scoreable > 0:
            p = n_correct / n_scoreable
            se = (p * (1 - p) / n_scoreable) ** 0.5
            ci_lo = max(0, 100 * (p - 1.96 * se))
            ci_hi = min(100, 100 * (p + 1.96 * se))
        else:
            ci_lo = ci_hi = float("nan")
        rows.append({
            "model": model,
            "category": category,
            "n_total": total,
            "n_scoreable": n_scoreable,
            "n_unparseable": total - n_scoreable,
            "n_correct": int(n_correct),
            "accuracy_pct": round(acc, 1),
            "ci95_lo": round(ci_lo, 1),
            "ci95_hi": round(ci_hi, 1),
        })
    return pd.DataFrame(rows).sort_values(["category", "model"])


# ============================================================================
# Signal-detection theory: d' (sensitivity) and c (criterion/bias)
#
# This yes/no task is a detection problem. Define:
#   signal present = caption is genuinely TRUE (ground_truth_label == 1)
#   "detect"       = model answers YES (parsed_label == 1)
# Then:
#   Hit         = TRUE caption, model says YES
#   False alarm = FALSE caption, model says YES
#   HR = hits / (# TRUE items);  FA = false alarms / (# FALSE items)
#   d' = z(HR) - z(FA)                  (higher = better discrimination)
#   c  = -0.5 * (z(HR) + z(FA))         (>0 = biased toward NO, <0 = toward YES)
#
# Why this matters here: last night's Gemini run showed ~92% specificity but
# ~70% hit rate — a large positive c (NO-bias). Raw accuracy hides that; d'/c
# separate "can it tell the classes apart" (d') from "which way does it lean" (c).
#
# Corrections & CIs:
#   - Log-linear correction (Hautus 1995) so 0%/100% rates don't blow up to
#     infinity: HRc=(hits+0.5)/(N+1), FAc=(fa+0.5)/(M+1).
#   - Standard error of d' via Gourevitch & Galanter (1967):
#       se = sqrt( HRc(1-HRc)/(N*phi(zH)^2) + FAc(1-FAc)/(M*phi(zF)^2) )
#     where N=# signal trials (TRUE), M=# noise trials (FALSE), phi=normal pdf.
#     Since d'=zH-zF and c=-0.5(zH+zF) share the same two variance terms,
#     se(c) = se(d')/2.
# ============================================================================

def _sdt_stats(hits: int, n_signal: int, false_alarms: int, n_noise: int) -> dict:
    """Compute log-linear-corrected d', c and their G&G 95% CIs for one
    (hits, signal-trials, false-alarms, noise-trials) cell."""
    from scipy.stats import norm

    if n_signal == 0 or n_noise == 0:
        return {k: float("nan") for k in
                ("HRc", "FAc", "dprime", "c", "se_dprime",
                 "dprime_ci_lo", "dprime_ci_hi", "c_ci_lo", "c_ci_hi")}

    # Log-linear correction avoids z(0) = -inf and z(1) = +inf.
    HRc = (hits + 0.5) / (n_signal + 1)
    FAc = (false_alarms + 0.5) / (n_noise + 1)

    zH = norm.ppf(HRc)
    zF = norm.ppf(FAc)
    dprime = zH - zF
    c = -0.5 * (zH + zF)

    # Gourevitch & Galanter SE of d'. phi = standard normal pdf at the z-scores.
    var = (HRc * (1 - HRc) / (n_signal * norm.pdf(zH) ** 2)
           + FAc * (1 - FAc) / (n_noise * norm.pdf(zF) ** 2))
    se_dprime = var ** 0.5
    se_c = se_dprime / 2.0  # c and d' share the same variance components

    return {
        "HRc": round(HRc, 4),
        "FAc": round(FAc, 4),
        "dprime": round(dprime, 3),
        "c": round(c, 3),
        "se_dprime": round(se_dprime, 3),
        "dprime_ci_lo": round(dprime - 1.96 * se_dprime, 3),
        "dprime_ci_hi": round(dprime + 1.96 * se_dprime, 3),
        "c_ci_lo": round(c - 1.96 * se_c, 3),
        "c_ci_hi": round(c + 1.96 * se_c, 3),
    }


def _sdt_counts(sub: pd.DataFrame) -> tuple:
    """From a scoreable slice, return (hits, n_signal, false_alarms, n_noise)."""
    scoreable = sub[sub["is_parseable"] == True]  # noqa: E712
    signal = scoreable[scoreable["ground_truth_label"] == 1]
    noise = scoreable[scoreable["ground_truth_label"] == 0]
    hits = int((signal["parsed_label"] == 1).sum())
    false_alarms = int((noise["parsed_label"] == 1).sum())
    return hits, len(signal), false_alarms, len(noise)


def signal_detection_table(results: pd.DataFrame) -> pd.DataFrame:
    """Per-(model, category) AND per-(model, ALL-pooled) d'/c with 95% CIs.
    Reporting BOTH aggregate and per-category is required — see the pooling
    warning in check_pooling_paradox() for why the aggregate alone can mislead."""
    rows = []
    for model, msub in results.groupby("model_name"):
        # Per-category
        for category, csub in msub.groupby("category"):
            h, ns, fa, nn = _sdt_counts(csub)
            stats = _sdt_stats(h, ns, fa, nn)
            rows.append({"model": model, "category": category,
                         "hits": h, "n_signal": ns,
                         "false_alarms": fa, "n_noise": nn, **stats})
        # Pooled across categories ("ALL")
        h, ns, fa, nn = _sdt_counts(msub)
        stats = _sdt_stats(h, ns, fa, nn)
        rows.append({"model": model, "category": "ALL (pooled)",
                     "hits": h, "n_signal": ns,
                     "false_alarms": fa, "n_noise": nn, **stats})
    return pd.DataFrame(rows).sort_values(["model", "category"])


def check_pooling_paradox(sdt_df: pd.DataFrame) -> List[str]:
    """Warn when a model's POOLED d' is lower than EVERY per-category component
    d'. That inversion means response criteria diverge across categories, and
    pooling under different thresholds destroys signal — the pooled number is an
    artifact, not a summary. Returns the list of warning strings (also printed)."""
    warnings = []
    for model, msub in sdt_df.groupby("model"):
        per_cat = msub[msub["category"] != "ALL (pooled)"]["dprime"].dropna()
        pooled_row = msub[msub["category"] == "ALL (pooled)"]["dprime"].dropna()
        if len(per_cat) < 2 or len(pooled_row) == 0:
            continue
        pooled = pooled_row.iloc[0]
        if pooled < per_cat.min():
            warnings.append(
                f"  [POOLING PARADOX] {model}: pooled d'={pooled:.3f} is LOWER "
                f"than every per-category d' (min per-cat={per_cat.min():.3f}). "
                f"Criteria differ across categories; do NOT report the pooled d' "
                f"as a summary — use the per-category values.")
    return warnings


# ============================================================================
# Latency reporting (outlier-aware)
#
# A single stalled/retried call (e.g. a 60s API timeout+retry) badly distorts a
# mean. So we report the MEDIAN (robust), the raw mean, and an outlier-trimmed
# mean (excluding items > OUTLIER_S) side by side, and separately LIST the slow
# items. Outliers are only removed from the "trimmed" statistic — never dropped
# from the underlying dataset.
# ============================================================================

OUTLIER_S = 10.0  # latency_seconds above this = likely timeout/retry, not "slow"


def latency_table(results: pd.DataFrame) -> pd.DataFrame:
    """Per-(model, category) latency: median, raw mean, outlier-trimmed mean,
    and outlier count (> OUTLIER_S seconds)."""
    if "latency_seconds" not in results.columns:
        return pd.DataFrame()
    rows = []
    for (model, category), sub in results.groupby(["model_name", "category"]):
        lat = pd.to_numeric(sub["latency_seconds"], errors="coerce").dropna()
        if len(lat) == 0:
            continue
        trimmed = lat[lat <= OUTLIER_S]
        rows.append({
            "model": model,
            "category": category,
            "n": len(lat),
            "median_s": round(float(lat.median()), 2),
            "mean_raw_s": round(float(lat.mean()), 2),
            "mean_trimmed_s": (round(float(trimmed.mean()), 2)
                               if len(trimmed) > 0 else float("nan")),
            "n_outliers_gt10s": int((lat > OUTLIER_S).sum()),
        })
    return pd.DataFrame(rows).sort_values(["model", "category"])


def latency_outliers(results: pd.DataFrame) -> pd.DataFrame:
    """List individual items with latency > OUTLIER_S (likely timeout/retry),
    so they can be inspected rather than silently averaged in."""
    if "latency_seconds" not in results.columns:
        return pd.DataFrame()
    lat = pd.to_numeric(results["latency_seconds"], errors="coerce")
    out = results[lat > OUTLIER_S].copy()
    cols = [c for c in ["model_name", "item_id", "category", "relation",
                        "latency_seconds", "error"] if c in out.columns]
    return out[cols].sort_values("latency_seconds", ascending=False)


# ============================================================================
# Proposed metrics: RRS, DT, RCS
# ============================================================================

def compute_rrs(accuracy_df: pd.DataFrame,
                model_order_by_compute: List[str]) -> pd.DataFrame:
    """Relational Robustness Score.

    From the proposal: for each relation category, the normalized area under
    the accuracy-vs-compute-reduction curve. Higher = degrades more gracefully.

    Concretely for the pilot: we treat the ordered list of models
    [most-constrained -> least-constrained] as points along the compute axis
    at equally-spaced x positions, and compute the trapezoidal-rule area
    under the accuracy curve, normalized to [0, 1] by dividing by the maximum
    possible area (100% accuracy at every point).

    This is a *pilot-scale* implementation. In the full paper with more
    precision levels per model, the x-axis becomes measured compute (TOPS *
    bits), not just ordinal ranking.
    """
    if len(model_order_by_compute) < 2:
        raise ValueError("Need at least 2 models to compute RRS")

    rows = []
    for category in accuracy_df["category"].unique():
        sub = accuracy_df[accuracy_df["category"] == category].set_index("model")
        # Ordered accuracy values along the compute axis (constrained -> free)
        acc_values = []
        for m in model_order_by_compute:
            if m in sub.index:
                acc_values.append(sub.loc[m, "accuracy_pct"])
            else:
                acc_values.append(float("nan"))
        acc_values = np.array(acc_values)
        # Trapezoidal AUC using unit-spaced x positions; normalize to [0,1]
        if np.isnan(acc_values).any():
            rrs = float("nan")
        else:
            auc = np.trapezoid(acc_values, dx=1.0)
            max_auc = 100.0 * (len(acc_values) - 1)
            rrs = auc / max_auc if max_auc > 0 else float("nan")
        rows.append({
            "category": category,
            "model_order": " -> ".join(model_order_by_compute),
            "accuracy_series": [round(v, 1) if not np.isnan(v) else None
                                for v in acc_values],
            "RRS": round(rrs, 3) if not np.isnan(rrs) else None,
        })
    return pd.DataFrame(rows)


def compute_dt(accuracy_df: pd.DataFrame,
               model_order_by_compute: List[str],
               floor_pct: float = 70.0) -> pd.DataFrame:
    """Degradation Threshold.

    Simplest defensible version for the pilot: for each category, the
    index (position) of the highest-constrained model that still exceeds
    the usability floor. Reported as the model name for readability.
    """
    rows = []
    for category in accuracy_df["category"].unique():
        sub = accuracy_df[accuracy_df["category"] == category].set_index("model")
        # Walk from most-constrained to least-constrained; find first model
        # that passes the floor.
        threshold_model = None
        for m in model_order_by_compute:
            if m in sub.index and sub.loc[m, "accuracy_pct"] >= floor_pct:
                threshold_model = m
                break
        rows.append({
            "category": category,
            "floor_pct": floor_pct,
            "first_model_above_floor": threshold_model or "NONE",
        })
    return pd.DataFrame(rows)


def compute_rcs_by_relation(results: pd.DataFrame) -> pd.DataFrame:
    """Relational Consistency Score — pilot proxy version.

    Full RCS requires paired inverse relations (A left of B / B right of A);
    the pilot doesn't have those pairs, so this proxy measures within-relation
    accuracy variance: does the model do equally well on all instances of
    'inside', or is it flipping randomly?

    Reported per (model, relation): variance of correctness within the
    relation. Low variance = consistent; high variance = pattern-matching.

    This is honestly labeled as a proxy in the report, and the full RCS is
    scoped for the final paper.
    """
    rows = []
    grouped = results[results["is_parseable"] == True].groupby(  # noqa: E712
        ["model_name", "relation"]
    )
    for (model, relation), sub in grouped:
        if len(sub) < 3:
            continue  # too few items to meaningfully compute
        rows.append({
            "model": model,
            "relation": relation,
            "n": len(sub),
            "mean_correct": round(sub["is_correct"].mean(), 3),
            "std_correct": round(sub["is_correct"].std(), 3),
        })
    return pd.DataFrame(rows).sort_values(["model", "relation"])


# ============================================================================
# McNemar's test for pairwise model comparison on shared items
# ============================================================================

def mcnemar_pair(results: pd.DataFrame,
                 model_a: str, model_b: str,
                 category: Optional[str] = None) -> dict:
    """McNemar's test on paired predictions from two models on the same items.

    Only counts items where BOTH models produced parseable answers. This
    prevents one model's high refusal rate from contaminating comparison.
    """
    sub = results.copy()
    if category is not None:
        sub = sub[sub["category"] == category]
    sub = sub[sub["is_parseable"] == True]  # noqa: E712

    a = sub[sub["model_name"] == model_a].set_index("item_id")["is_correct"]
    b = sub[sub["model_name"] == model_b].set_index("item_id")["is_correct"]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return {"model_a": model_a, "model_b": model_b, "category": category or "ALL",
                "n_common": 0, "b": None, "c": None, "chi2": None, "p": None}
    a = a.loc[common].astype(int)
    b = b.loc[common].astype(int)

    # Contingency table cells:
    #   b = model_a correct AND model_b wrong
    #   c = model_a wrong AND model_b correct
    b_count = int(((a == 1) & (b == 0)).sum())
    c_count = int(((a == 0) & (b == 1)).sum())

    # McNemar's test with continuity correction (safe for small counts)
    if b_count + c_count == 0:
        chi2 = 0.0
        p = 1.0
    else:
        chi2 = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
        # Chi-square with 1 df; use scipy if available, else approximate.
        try:
            from scipy.stats import chi2 as chi2_dist
            p = 1 - chi2_dist.cdf(chi2, df=1)
        except ImportError:
            # Coarse approximation without scipy — good enough for pilot,
            # note this in the report.
            p = None

    return {
        "model_a": model_a,
        "model_b": model_b,
        "category": category or "ALL",
        "n_common": len(common),
        "a_correct_b_wrong": b_count,
        "a_wrong_b_correct": c_count,
        "chi2": round(chi2, 3),
        "p_value": round(p, 4) if p is not None else None,
    }


# ============================================================================
# Figures
# ============================================================================

def plot_accuracy_by_category(accuracy_df: pd.DataFrame, output_path: Path,
                              model_order: List[str]) -> None:
    """Grouped bar chart: accuracy per model per category, with 95% CIs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed; skipping figure. "
              "Run: pip install matplotlib")
        return

    categories = sorted(accuracy_df["category"].unique())
    n_models = len(model_order)
    n_cats = len(categories)
    bar_width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(n_cats)

    for i, model in enumerate(model_order):
        vals = []
        errs_lo = []
        errs_hi = []
        for cat in categories:
            row = accuracy_df[
                (accuracy_df["model"] == model) &
                (accuracy_df["category"] == cat)
            ]
            if len(row) == 0:
                vals.append(0)
                errs_lo.append(0)
                errs_hi.append(0)
            else:
                r = row.iloc[0]
                vals.append(r["accuracy_pct"])
                errs_lo.append(r["accuracy_pct"] - r["ci95_lo"])
                errs_hi.append(r["ci95_hi"] - r["accuracy_pct"])
        offset = (i - (n_models - 1) / 2) * bar_width
        ax.bar(x + offset, vals, bar_width, label=model,
               yerr=[errs_lo, errs_hi], capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy by model and relation category (95% CI)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Wrote figure: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/analysis"))
    parser.add_argument("--model-order", type=str, nargs="+", default=None,
                        help="Model names ordered most-constrained -> least. "
                             "Default: infer from data (imperfect).")
    parser.add_argument("--dt-floor", type=float, default=70.0,
                        help="Accuracy floor (%%) for Degradation Threshold")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model results...")
    results = load_all_results(args.results_dir)
    print(f"\nTotal rows: {len(results)}")
    print(f"Distinct models: {results['model_name'].unique().tolist()}")
    print(f"Distinct categories: {results['category'].unique().tolist()}\n")

    # Determine model order along compute axis
    if args.model_order:
        model_order = args.model_order
    else:
        # Best-guess ordering from names; explicit --model-order is safer.
        # For the pilot with jetson/qwen/gemini names:
        all_models = list(results["model_name"].unique())

        def rank(m: str) -> int:
            m_lo = m.lower()
            if "jetson" in m_lo or "orin" in m_lo:
                return 0  # most constrained
            if "qwen" in m_lo and "7b" in m_lo:
                return 1
            if "gemini" in m_lo:
                return 2  # least constrained
            return 99
        model_order = sorted(all_models, key=rank)
        print(f"Inferred model order (most -> least constrained): {model_order}")
        print("  If wrong, pass --model-order to override.\n")

    # 1. Accuracy table
    print("=" * 60)
    print("1. Accuracy by model and category")
    print("=" * 60)
    acc_df = accuracy_table(results)
    print(acc_df.to_string(index=False))
    acc_df.to_csv(args.output_dir / "accuracy_table.csv", index=False)

    # 1b. Signal-detection metrics: d' and c, per-category AND pooled
    print("\n" + "=" * 60)
    print("1b. Signal-detection: d' (sensitivity) and c (bias), 95% CI")
    print("=" * 60)
    sdt_df = signal_detection_table(results)
    print(sdt_df.to_string(index=False))
    sdt_df.to_csv(args.output_dir / "signal_detection.csv", index=False)
    print("\nInterpretation: d' higher = better class discrimination; "
          "c > 0 = biased toward answering NO, c < 0 = toward YES.")
    pooling_warnings = check_pooling_paradox(sdt_df)
    if pooling_warnings:
        print("\n!! POOLING WARNINGS (aggregate d' is misleading here) !!")
        for w in pooling_warnings:
            print(w)
    else:
        print("\nNo pooling paradox detected (pooled d' is not below all "
              "per-category d' for any model).")

    # 1c. Latency (outlier-aware): median, raw mean, trimmed mean, outliers
    print("\n" + "=" * 60)
    print("1c. Latency (median / raw mean / >10s-trimmed mean)")
    print("=" * 60)
    lat_df = latency_table(results)
    if len(lat_df) > 0:
        print(lat_df.to_string(index=False))
        lat_df.to_csv(args.output_dir / "latency_table.csv", index=False)
        outliers = latency_outliers(results)
        print(f"\nLatency outliers (> {OUTLIER_S:.0f}s), likely timeout/retry — "
              f"{len(outliers)} item(s):")
        if len(outliers) > 0:
            print(outliers.to_string(index=False))
            outliers.to_csv(args.output_dir / "latency_outliers.csv", index=False)
        else:
            print("  none")
    else:
        print("  No latency_seconds column in results; skipping.")

    # 2. Metrics
    print("\n" + "=" * 60)
    print("2. Proposed metrics")
    print("=" * 60)
    print("\nRelational Robustness Score (RRS):")
    rrs = compute_rrs(acc_df, model_order)
    print(rrs.to_string(index=False))
    rrs.to_csv(args.output_dir / "rrs.csv", index=False)

    print(f"\nDegradation Threshold (DT) at floor={args.dt_floor}%:")
    dt = compute_dt(acc_df, model_order, floor_pct=args.dt_floor)
    print(dt.to_string(index=False))
    dt.to_csv(args.output_dir / "dt.csv", index=False)

    print("\nRelational Consistency Score (RCS) proxy — per-relation "
          "correctness mean and std:")
    rcs = compute_rcs_by_relation(results)
    print(rcs.head(15).to_string(index=False))
    rcs.to_csv(args.output_dir / "rcs_proxy.csv", index=False)

    # 3. McNemar pairwise
    print("\n" + "=" * 60)
    print("3. McNemar's test — pairwise model comparisons")
    print("=" * 60)
    mcnemar_rows = []
    models = list(results["model_name"].unique())
    for i, ma in enumerate(models):
        for mb in models[i + 1:]:
            mcnemar_rows.append(mcnemar_pair(results, ma, mb))
            for cat in results["category"].unique():
                mcnemar_rows.append(mcnemar_pair(results, ma, mb, category=cat))
    mn = pd.DataFrame(mcnemar_rows)
    print(mn.to_string(index=False))
    mn.to_csv(args.output_dir / "mcnemar.csv", index=False)

    # 4. Figure
    print("\n" + "=" * 60)
    print("4. Figure")
    print("=" * 60)
    plot_accuracy_by_category(acc_df, args.output_dir / "accuracy_by_category.png",
                              model_order)

    # 5. Summary JSON for the report
    summary = {
        "n_items_total_per_model": int(len(results) / len(models)),
        "model_order_compute_axis": model_order,
        "accuracy_by_condition": acc_df.to_dict(orient="records"),
        "signal_detection": sdt_df.to_dict(orient="records"),
        "pooling_warnings": pooling_warnings,
        "latency": lat_df.to_dict(orient="records") if len(lat_df) else [],
        "dt_floor_pct": args.dt_floor,
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nAll analysis artifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
