#!/usr/bin/env python3
"""
interaction_test.py

The PRIMARY confirmatory test (RQ1 in the pre-registration): does model compute
affect projective-spatial accuracy DIFFERENTLY than topological-containment
accuracy? Tests the category × compute interaction via logistic regression, with
a likelihood-ratio test as the reported statistic.

This is a standalone, pre-registered script so the primary result is fully
reproducible and not an ad-hoc computation. It loads results the same way
analyze_pilot.py does (dedup per model_name+item_id, keep last).

Usage:
    python interaction_test.py --results-dir results/ \
        --model-a qwen2.5vl:3b --model-b qwen2.5vl:7b \
        --output results/analysis/interaction.txt

Requires: pandas, statsmodels, scipy.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def load_two_models(results_dir: Path, model_a: str, model_b: str) -> pd.DataFrame:
    """Load and dedup results for exactly two models. Mirrors analyze_pilot.py's
    dedup: keep last row per (model_name, item_id), parseable items only."""
    frames = []
    for csv_path in sorted(results_dir.glob("*.csv")):
        if csv_path.parent != results_dir:
            continue
        d = pd.read_csv(csv_path)
        if "model_name" not in d.columns:
            continue
        frames.append(d)
    if not frames:
        sys.exit(f"ERROR: no result CSVs found in {results_dir}")
    alld = pd.concat(frames, ignore_index=True)
    alld = alld[alld["model_name"].isin([model_a, model_b])].copy()
    if alld.empty:
        sys.exit(f"ERROR: no rows for {model_a!r} or {model_b!r}. "
                 f"Found: {sorted(alld['model_name'].unique())}")
    # dedup keep-last per (model_name, item_id)
    alld = alld.drop_duplicates(subset=["model_name", "item_id"], keep="last")
    # parseable only
    def truthy(v):
        return str(v).strip().lower() in ("true", "1", "1.0", "yes", "t")
    alld = alld[alld["is_parseable"].map(truthy)].copy()
    alld["correct"] = alld["is_correct"].map(truthy).astype(int)
    return alld


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--model-a", required=True, help="lower-compute model, e.g. qwen2.5vl:3b")
    ap.add_argument("--model-b", required=True, help="higher-compute model, e.g. qwen2.5vl:7b")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    try:
        import statsmodels.formula.api as smf
        from scipy.stats import chi2
    except ImportError:
        sys.exit("ERROR: pip install statsmodels scipy")

    d = load_two_models(args.results_dir, args.model_a, args.model_b)

    # Keep only items answered by BOTH models (paired design)
    common = set(d[d.model_name == args.model_a].item_id) & \
             set(d[d.model_name == args.model_b].item_id)
    d = d[d.item_id.isin(common)].copy()

    d["compute"] = d["model_name"].map({args.model_a: "low", args.model_b: "high"})
    d["cat"] = d["category"].map({
        "projective_spatial": "projective",
        "topological_containment": "containment",
    })
    d = d.dropna(subset=["cat"])

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 68)
    out("PRIMARY CONFIRMATORY TEST — category × compute interaction")
    out("=" * 68)
    out(f"model_a (low compute):  {args.model_a}")
    out(f"model_b (high compute): {args.model_b}")
    out(f"paired items (both models, categorized): {len(common)}")
    out("")
    out("Cell accuracies:")
    cell = d.groupby(["compute", "cat"])["correct"].agg(["mean", "count"])
    out(cell.round(4).to_string())
    out("")

    full = smf.logit("correct ~ C(compute) * C(cat)", data=d).fit(disp=0)
    reduced = smf.logit("correct ~ C(compute) + C(cat)", data=d).fit(disp=0)
    lr_stat = 2 * (full.llf - reduced.llf)
    lr_p = chi2.sf(lr_stat, df=1)

    out("Logistic regression: correct ~ C(compute) * C(cat)")
    inter = [t for t in full.params.index if ":" in t]
    for t in inter:
        out(f"  interaction term {t}:")
        out(f"    coef = {full.params[t]:.4f}   Wald p = {full.pvalues[t]:.4f}")
    out("")
    out("Likelihood-ratio test (full vs additive) — THE pre-registered statistic:")
    out(f"  LR chi2(1) = {lr_stat:.4f}")
    out(f"  p = {lr_p:.4f}")
    out("")
    if lr_p < 0.05:
        out("RESULT: interaction SIGNIFICANT at a=0.05.")
        out("  -> Compute affects the two categories differently (per pre-reg H1_1).")
    else:
        out("RESULT: interaction NOT significant at a=0.05.")
        out("  -> Per pre-reg H1_0 / §7: do NOT claim category-specific")
        out("     compute-sensitivity. Report projective degradation (RQ2) on its")
        out("     own; state the interaction did not confirm.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines) + "\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
