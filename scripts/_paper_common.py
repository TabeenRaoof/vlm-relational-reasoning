#!/usr/bin/env python3
"""
_paper_common.py

Shared loading helpers for the post-hoc / sensitivity analyses reported in the
paper (Sections 5.2.1, 5.2.2, 5.3, 5.5, 6). These mirror the dedup and
scoreability rules already used by analyze_pilot.py, interaction_test.py, and
h3_equivalence_test.py, so every script in the repo agrees on which rows count.

The rules, stated once:
  1. Dedup keep-last per (model_name, item_id). Chunked/crash-retried Jetson
     runs leave stale rows behind; the last write is the good one.
  2. Keep only parseable rows. An unparseable response is not a wrong answer,
     it is a missing observation, and it is dropped rather than scored 0.
  3. Comparisons are paired: an item counts only if it is scoreable on BOTH
     sides of the comparison being made.
"""
from pathlib import Path

import pandas as pd


def truthy(v) -> bool:
    """The result CSVs were written by several scripts over several months, so
    booleans appear as True/true/1/1.0/yes. Normalize all of them."""
    return str(v).strip().lower() in ("true", "1", "1.0", "yes", "t")


def load_scoreable(path: Path) -> pd.DataFrame:
    """Load one result CSV, dedup, and keep only scoreable rows (rule 1 + 2)."""
    d = pd.read_csv(path)
    if "model_name" in d.columns:
        d = d.drop_duplicates(subset=["model_name", "item_id"], keep="last")
    else:
        d = d.drop_duplicates(subset=["item_id"], keep="last")
    d = d[d["is_parseable"].map(truthy)].copy()
    d["correct"] = d["is_correct"].map(truthy).astype(int)
    return d


def load_eval_set(path: Path) -> pd.DataFrame:
    """The frozen item set. Supplies category, relation, and — critically for
    the cluster bootstrap — image_filename, since many items share an image."""
    return pd.read_csv(path)


def pilot_item_ids(path: Path) -> set:
    """Item ids belonging to the n=300 exploratory pilot.

    These are the items that generated H1. They nest inside the n=2000
    confirmatory set by construction (content-hashed ids + deterministic rank
    key), which is exactly why Section 5.2.2 has to split them back out: they
    are not independent evidence for the hypothesis they produced.
    """
    return set(pd.read_csv(path)["item_id"])


def paired_frame(low_csv: Path, high_csv: Path, eval_csv: Path,
                 pilot_csv: Path = None) -> pd.DataFrame:
    """Build the per-item paired table used by every H1 analysis.

    One row per item, with the two models' correctness side by side, plus the
    per-item paired change score d = high - low in {-1, 0, +1}. That change
    score is the estimand H1 is actually phrased in ("scale reduction hurts
    projective MORE than containment" is a difference of differences on the
    percentage-point scale), which is why it is computed here once and reused.
    """
    low = load_scoreable(low_csv).set_index("item_id")["correct"]
    high = load_scoreable(high_csv).set_index("item_id")["correct"]

    # Rule 3: paired items only.
    common = low.index.intersection(high.index)
    low, high = low.loc[common], high.loc[common]

    ev = load_eval_set(eval_csv).set_index("item_id")
    frame = pd.DataFrame({
        "low": low,
        "high": high,
        "category": ev["category"].loc[common],
        "relation": ev["relation"].loc[common],
        "image": ev["image_filename"].loc[common],
    })
    frame["d"] = frame["high"] - frame["low"]

    # Mark which items the pilot had already seen, so Section 5.2.2 can split.
    if pilot_csv is not None:
        seen = pilot_item_ids(pilot_csv)
        frame["is_new"] = ~frame.index.isin(seen)
    return frame
