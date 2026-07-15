#!/usr/bin/env python3
"""
check_vsr_ceiling.py

GATE SCRIPT — run this BEFORE freezing the pre-registration or collecting any
data. It reports the maximum number of balanced items available per category
in the VSR source, so the pre-registered target n is set to a number that is
actually achievable rather than aspirational.

It reuses the EXACT taxonomy and label logic from curate_eval_set.py so the
ceiling it reports matches what curate_eval_set.py could actually produce.
It does not sample, write, or modify anything — read-only.

Usage:
    python check_vsr_ceiling.py
    python check_vsr_ceiling.py --data-dir ./data

Output: the per-category True/False pool sizes, and the max balanced
n-per-category and total n you can request without curate_eval_set.py
failing its "not enough items" check.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Import the taxonomy directly from the curation script so this can never
# drift out of sync with what curate_eval_set.py actually uses.
sys.path.insert(0, str(Path(__file__).parent))
try:
    from curate_eval_set import (
        PROJECTIVE, CONTAINMENT, EXCLUDED, CATEGORY_LABELS, load_all_splits,
        validate_taxonomy_coverage,
    )
except ImportError:
    sys.exit("ERROR: run this from the same scripts/ dir as curate_eval_set.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path,
                    default=Path(__file__).parent.parent / "data")
    args = ap.parse_args()

    items = load_all_splits(args.data_dir)
    print(f"Loaded {len(items)} VSR items across train+dev+test.\n")

    # Same guard curate uses — fail loud if VSR has relations we haven't placed.
    validate_taxonomy_coverage(items)
    print("Taxonomy validated: every VSR relation is categorized or excluded.\n")

    print("=" * 68)
    print("PER-CATEGORY POOL SIZES (this is the ceiling)")
    print("=" * 68)

    overall_max_balanced_per_cat = None
    for category, relset in CATEGORY_LABELS.items():
        pool = [r for r in items if r["relation"] in relset]
        true_n = sum(1 for r in pool if r["label"] == 1)
        false_n = sum(1 for r in pool if r["label"] == 0)
        # A balanced sample needs n//2 True and n-n//2 False, so the max
        # balanced n-per-category is 2 * min(true_n, false_n).
        max_balanced = 2 * min(true_n, false_n)
        overall_max_balanced_per_cat = (
            max_balanced if overall_max_balanced_per_cat is None
            else min(overall_max_balanced_per_cat, max_balanced)
        )
        print(f"\n  {category}:")
        print(f"    total pool: {len(pool)}")
        print(f"    True label:  {true_n}")
        print(f"    False label: {false_n}")
        print(f"    max BALANCED n for this category: {max_balanced} "
              f"(limited by the smaller label)")
        # distinct relations present, so we can see diversity
        rels = Counter(r["relation"] for r in pool)
        print(f"    distinct relations present: {len(rels)}")
        print(f"    relation breakdown: "
              f"{dict(rels.most_common())}")

    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    max_total = overall_max_balanced_per_cat * 2  # two categories
    print(f"Max balanced n-per-category (both categories): "
          f"{overall_max_balanced_per_cat}")
    print(f"==> Max achievable TOTAL n (balanced, 2 categories): {max_total}")
    print()
    target_total = 2000
    per_cat_needed = target_total // 2
    if overall_max_balanced_per_cat >= per_cat_needed:
        print(f"[OK] Target n={target_total} total "
              f"({per_cat_needed}/category) IS achievable from VSR alone.")
        print(f"     Headroom: {overall_max_balanced_per_cat - per_cat_needed} "
              f"extra items/category beyond target.")
    else:
        achievable_total = overall_max_balanced_per_cat * 2
        print(f"[!!] Target n={target_total} is NOT achievable from VSR alone.")
        print(f"     VSR caps at n={achievable_total} total "
              f"({overall_max_balanced_per_cat}/category).")
        print(f"     Options: (a) pre-register n={achievable_total} instead, or")
        print(f"              (b) add a second dataset (e.g. What'sUp) to reach "
              f"{target_total}.")
        print(f"     >>> Freeze the pre-registration at the number you can "
              f"actually hit. <<<")


if __name__ == "__main__":
    main()
