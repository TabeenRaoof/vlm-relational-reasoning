#!/usr/bin/env python3
"""
h3_equivalence_test.py

The primary confirmatory analysis for the H3 n=2000 study: a TOST (two one-sided
tests) equivalence test on the paired Jetson-vs-Mac accuracy difference, against a
pre-registered margin (default +/- 3pp). Declares equivalence iff both one-sided
tests reject at alpha.

Reproducible/pre-registered — not an ad-hoc computation.

Usage:
    python h3_equivalence_test.py \
        --jetson results/jetson_qwen25vl_3b_q4_n2000.csv \
        --mac    results/mac_qwen25vl_3b_n2000.csv \
        --margin 3.0 \
        --output results/analysis_h3_n2000/equivalence.txt

Requires: pandas, scipy.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_scoreable(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    # dedup keep-last per (model_name, item_id) — chunked Jetson runs leave dups
    if "model_name" in d.columns:
        d = d.drop_duplicates(subset=["model_name", "item_id"], keep="last")
    else:
        d = d.drop_duplicates(subset=["item_id"], keep="last")
    def truthy(v):
        return str(v).strip().lower() in ("true", "1", "1.0", "yes", "t")
    d = d[d["is_parseable"].map(truthy)].copy()
    d["correct"] = d["is_correct"].map(truthy).astype(int)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jetson", type=Path, required=True)
    ap.add_argument("--mac", type=Path, required=True)
    ap.add_argument("--margin", type=float, default=3.0, help="equivalence margin in percentage points")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    jet = load_scoreable(args.jetson)
    mac = load_scoreable(args.mac)

    # paired: items scoreable on BOTH
    j = jet.set_index("item_id")["correct"]
    m = mac.set_index("item_id")["correct"]
    common = j.index.intersection(m.index)
    j = j.loc[common]
    m = m.loc[common]
    cats = jet.set_index("item_id")["category"].loc[common]

    n = len(common)
    b = int(((j == 1) & (m == 0)).sum())   # jetson right, mac wrong
    c = int(((j == 0) & (m == 1)).sum())   # jetson wrong, mac right
    diff = (b - c) / n                       # jetson - mac, in proportion
    se = np.sqrt((b + c - (b - c) ** 2 / n) / n ** 2) if (b + c) > 0 else 1e-9
    margin = args.margin / 100.0

    # TOST: two one-sided tests
    z_lower = (diff - (-margin)) / se
    z_upper = (diff - margin) / se
    p_lower = 1 - stats.norm.cdf(z_lower)    # H0: diff <= -margin
    p_upper = stats.norm.cdf(z_upper)         # H0: diff >= +margin
    equivalence = (p_lower < args.alpha) and (p_upper < args.alpha)

    # 90% CI (the CI that corresponds to two one-sided alpha=0.05 tests)
    z90 = stats.norm.ppf(0.95)
    ci_lo, ci_hi = diff - z90 * se, diff + z90 * se

    # McNemar (secondary, continuity-corrected) for continuity with n=300
    if (b + c) > 0:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        mcnemar_p = stats.chi2.sf(chi2, 1)
    else:
        mcnemar_p = 1.0

    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out("=" * 70)
    out("H3 EQUIVALENCE TEST (TOST) — Jetson vs Mac, Qwen2.5-VL-3B Q4_K_M")
    out("=" * 70)
    out(f"paired scoreable items (both sides): {n}")
    out(f"discordant: b(jetson-right/mac-wrong)={b}, c(jetson-wrong/mac-right)={c}")
    out(f"equivalence margin: ±{args.margin:.1f}pp   alpha: {args.alpha}")
    out("")
    out(f"observed difference (Jetson − Mac): {diff*100:+.2f}pp")
    out(f"90% CI: [{ci_lo*100:+.2f}, {ci_hi*100:+.2f}]pp")
    out("")
    out("TOST two one-sided tests:")
    out(f"  H0: diff <= -{args.margin:.1f}pp  ->  p = {p_lower:.4g}")
    out(f"  H0: diff >= +{args.margin:.1f}pp  ->  p = {p_upper:.4g}")
    out("")
    if equivalence:
        out(f"RESULT: EQUIVALENCE DECLARED — the Jetson−Mac difference is")
        out(f"  statistically within ±{args.margin:.1f}pp. Accuracy transfers cleanly")
        out(f"  to the edge device within the pre-registered margin.")
    else:
        out(f"RESULT: equivalence NOT declared at ±{args.margin:.1f}pp.")
        out(f"  Per pre-reg §7: report the point estimate and 90% CI honestly.")
        out(f"  If the CI merely straddles the margin (small point estimate,")
        out(f"  insufficient precision), do NOT collect more data — that would be")
        out(f"  optional stopping. If the difference genuinely exceeds the margin,")
        out(f"  report it as a measurable edge-deployment accuracy change.")
    out("")
    out(f"[secondary] paired McNemar (continuity-corrected) p = {mcnemar_p:.4g}")
    out("")
    out("[secondary] per-category difference:")
    for cat in sorted(set(cats)):
        idx = cats[cats == cat].index
        jj, mm = j.loc[idx], m.loc[idx]
        d2 = (jj.mean() - mm.mean()) * 100
        out(f"  {cat:24s} n={len(idx):4d}  Jetson={jj.mean()*100:.1f}%  "
            f"Mac={mm.mean()*100:.1f}%  diff={d2:+.2f}pp")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines) + "\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
