#!/usr/bin/env python3
"""
h1_corrected_analyses.py

The post-hoc H1 analyses reported in Sections 5.2.1 and 5.2.2 of the paper, and
the four estimates in Table 4. These are NOT the pre-registered test — that one
lives in interaction_test.py and is reported first and unchanged. These are the
corrected/sensitivity analyses, and they exist because the pre-registered model
has two defects that were only visible after the data were in hand:

  DEFECT 1 (pairing). The pre-registered logistic regression treats all 4,000
  observations as independent. They are not: each item is answered by BOTH the
  3B and the 7B model, so the two observations share an image and an intrinsic
  difficulty. Ignoring that inflates the standard error of a within-item
  contrast. Fixes applied here: a GEE with item-clustered robust standard
  errors, and a per-item paired change score analyzed directly.

  DEFECT 2 (previously-seen data). The n=2000 confirmatory set NESTS the n=300
  pilot that generated H1. Pre-registering the analysis plan before collecting
  the remaining 1,700 items does not turn those 300 items into new evidence for
  the hypothesis they produced. So every analysis is run twice: pooled
  (n=2,000) and on the fully independent items only (n=1,700).

The honest summary, which the paper states and this script reproduces: the
pairing correction alone REVERSES the pre-registered conclusion (p=0.017), but
that reversal does not survive removing the pilot items (p=0.112). Direction is
consistent everywhere; significance is not.

Usage:
    python h1_corrected_analyses.py \
        --low  results/mac_qwen25vl_3b_n2000.csv \
        --high results/mac_qwen25vl_7b_n2000.csv \
        --eval-set data/eval_set_n2000.csv \
        --pilot-set data/pilot_eval_set_n150.csv \
        --output results/analysis_paper/h1_corrected.txt

Requires: pandas, numpy, scipy, statsmodels.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_common import paired_frame  # noqa: E402


def to_long(frame: pd.DataFrame) -> pd.DataFrame:
    """Reshape the per-item paired table back to one row per observation.

    The regression models want long format (item, compute, category, correct);
    the change-score analysis wants wide. We keep wide as the source of truth
    and expand here so the two can never drift apart.
    """
    parts = []
    for compute_level in ("low", "high"):
        part = pd.DataFrame({
            "item_id": frame.index,
            "correct": frame[compute_level].values,
            "compute": compute_level,
            "cat": frame["category"].values,
        })
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def logistic_lrt(long: pd.DataFrame):
    """The pre-registered model: independence-assuming logistic regression,
    likelihood-ratio test of the interaction term (full vs additive)."""
    full = smf.glm("correct ~ C(compute)*C(cat)", data=long,
                   family=sm.families.Binomial()).fit()
    additive = smf.glm("correct ~ C(compute)+C(cat)", data=long,
                       family=sm.families.Binomial()).fit()
    lr_stat = 2 * (full.llf - additive.llf)
    p = stats.chi2.sf(lr_stat, 1)
    term = [t for t in full.params.index if ":" in t][0]
    return lr_stat, p, float(full.params[term])


def gee_clustered(long: pd.DataFrame):
    """DEFECT 1 fix, model-based: the identical interaction, refit as a GEE with
    an exchangeable working correlation and robust standard errors clustered on
    item_id. Same point estimate, honest standard error."""
    model = smf.gee("correct ~ C(compute)*C(cat)", groups="item_id", data=long,
                    family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
    term = [t for t in model.params.index if ":" in t][0]
    return float(model.tvalues[term]), float(model.pvalues[term])


def did_change_score(frame: pd.DataFrame):
    """DEFECT 1 fix, estimand-matched: compare the per-item paired change score
    d = (7B correct - 3B correct) between categories.

    This is a difference-in-differences on the percentage-point scale — the
    scale H1 is actually worded in — rather than on the log-odds scale. Welch's
    t-test because the two category groups have different variances.
    """
    proj = frame[frame["category"] == "projective_spatial"]["d"]
    cont = frame[frame["category"] == "topological_containment"]["d"]
    did = (proj.mean() - cont.mean()) * 100
    se = np.sqrt(proj.var(ddof=1) / len(proj) + cont.var(ddof=1) / len(cont)) * 100
    p = stats.ttest_ind(proj, cont, equal_var=False).pvalue
    return did, did - 1.96 * se, did + 1.96 * se, float(p), len(proj), len(cont)


def image_cluster_bootstrap(frame: pd.DataFrame, n_boot: int, seed: int):
    """Non-parametric check on the DiD that assumes even less.

    Items are not independent at the IMAGE level either: 2,000 items come from
    only ~1,674 distinct COCO images, and two captions about the same image
    share whatever makes that image easy or hard. So we resample whole images
    with replacement (all items of a picked image travel together), which is
    the item-level analogue of the blockwise bootstrap used in ASR evaluation.
    """
    groups = {}
    for image_name, index in frame.groupby("image").groups.items():
        groups[image_name] = np.asarray(index)
    keys = np.array(list(groups.keys()), dtype=object)

    rng = np.random.default_rng(seed)
    stats_out = []
    for _ in range(n_boot):
        picked = rng.choice(keys, size=len(keys), replace=True)
        rows = np.concatenate([groups[k] for k in picked])
        sample = frame.loc[rows]
        proj = sample[sample["category"] == "projective_spatial"]["d"]
        cont = sample[sample["category"] == "topological_containment"]["d"]
        if len(proj) == 0 or len(cont) == 0:
            continue
        stats_out.append((proj.mean() - cont.mean()) * 100)

    draws = np.array(stats_out)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # Two-sided bootstrap p: how often does the resampled effect cross zero?
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return lo, hi, float(p), len(draws)


def mcnemar(frame: pd.DataFrame, category: str):
    """Paired McNemar within one category, continuity-corrected."""
    sub = frame[frame["category"] == category]
    b = int((sub["d"] == -1).sum())   # 3B right, 7B wrong
    c = int((sub["d"] == 1).sum())    # 3B wrong, 7B right
    if (b + c) == 0:
        return len(sub), b, c, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return len(sub), b, c, float(stats.chi2.sf(chi2, 1))


def analyze(frame: pd.DataFrame, label: str, n_boot: int, seed: int, out):
    """Run the whole battery on one sample (pooled or new-items-only)."""
    long = to_long(frame)
    lr_stat, p_lr, coef = logistic_lrt(long)
    z_gee, p_gee = gee_clustered(long)
    did, ci_lo, ci_hi, p_did, n_proj, n_cont = did_change_score(frame)

    out("")
    out("-" * 70)
    out(f"{label}   items={len(frame)}  distinct images={frame['image'].nunique()}")
    out("-" * 70)
    out(f"  projective items={n_proj}   containment items={n_cont}")
    out("")
    out("  [pre-registered model] logistic LR, observations treated independent")
    out(f"    interaction coef = {coef:+.4f} log-odds")
    out(f"    LR chi2(1) = {lr_stat:.4f}   p = {p_lr:.4f}")
    out("")
    out("  [pairing-corrected] GEE, item-clustered robust SE, same estimand")
    out(f"    z = {z_gee:.4f}   p = {p_gee:.4f}")
    out("")
    out("  [estimand-matched] per-item paired change score, difference-in-differences")
    out(f"    DiD = {did:+.2f} pp   95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]   p = {p_did:.4f}")

    lo_b, hi_b, p_b, n_ok = image_cluster_bootstrap(frame, n_boot, seed)
    out(f"    image-level cluster bootstrap (B={n_ok}, seed={seed})")
    out(f"      95% CI [{lo_b:+.2f}, {hi_b:+.2f}]   p ~ {p_b:.4f}")
    out("")
    out("  [per-category paired McNemar, 3B vs 7B]")
    for category in ("projective_spatial", "topological_containment"):
        n, b, c, p = mcnemar(frame, category)
        out(f"    {category:24s} n={n:4d}  b={b:3d}  c={c:3d}  p = {p:.4g}")

    return {"label": label, "n": len(frame), "did": did, "ci_lo": ci_lo,
            "ci_hi": ci_hi, "p_logistic": p_lr, "p_gee": p_gee}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--low", type=Path, required=True, help="lower-compute model CSV (3B)")
    ap.add_argument("--high", type=Path, required=True, help="higher-compute model CSV (7B)")
    ap.add_argument("--eval-set", type=Path, required=True)
    ap.add_argument("--pilot-set", type=Path, required=True,
                    help="pilot item set, used to identify previously-seen items")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260716,
                    help="fixed so the bootstrap CI is reproducible")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    frame = paired_frame(args.low, args.high, args.eval_set, args.pilot_set)

    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 70)
    out("H1 CORRECTED / SENSITIVITY ANALYSES  (paper Sections 5.2.1, 5.2.2)")
    out("=" * 70)
    out("The pre-registered result (interaction_test.py) is reported first and")
    out("is NOT superseded. What follows are corrections for (1) the paired item")
    out("structure and (2) the inclusion of the hypothesis-generating pilot.")

    pooled = analyze(frame, "POOLED — includes the 300 pilot items",
                     args.n_boot, args.seed, out)
    new_only = analyze(frame[frame["is_new"]].copy(),
                       "NEW ITEMS ONLY — independent of the pilot",
                       args.n_boot, args.seed, out)

    # Table 4 of the paper, assembled from the two runs above.
    out("")
    out("=" * 70)
    out("TABLE 4 — H1 interaction under four estimation choices")
    out("=" * 70)
    out(f"{'Sample':<26}{'DiD (pp)':<26}{'Logistic':<12}{'GEE':<12}")
    for row in (pooled, new_only):
        did_cell = f"{row['did']:+.2f} [{row['ci_lo']:+.2f}, {row['ci_hi']:+.2f}]"
        name = f"{'Pooled' if 'POOLED' in row['label'] else 'New only'} (n={row['n']})"
        out(f"{name:<26}{did_cell:<26}"
            f"p = {row['p_logistic']:.3f}  p = {row['p_gee']:.3f}")
    out("")
    out("Reading: all four agree in DIRECTION. The pairing correction flips the")
    out("pooled conclusion to significant; neither new-items-only estimate")
    out("crosses p<0.05. H1 is therefore reported as MIXED — a power problem at")
    out("the revised effect size, not a demonstrated null (see power_projection.py).")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines) + "\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
