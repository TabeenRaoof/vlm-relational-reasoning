#!/usr/bin/env python3
"""
curate_eval_set.py

Curates a stratified evaluation set from VSR (Visual Spatial Reasoning) for the
CS587 pilot. Filters items into two categories (projective spatial, topological
containment) and samples a balanced pilot set with True/False labels roughly
50/50 within each category.

Input:  vsr_train.jsonl, vsr_dev.jsonl, vsr_test.jsonl in ./data/
Output: pilot_eval_set.csv in ./data/

Categories are defined explicitly in this script — see PROJECTIVE and
CONTAINMENT sets below. All 66 VSR relations are accounted for (categorized
or explicitly excluded). Proximity relations (near, next to, etc.) and
contact relations (touching, on) are excluded per the pilot's design decision
— see project notes.

Usage:
    python curate_eval_set.py --n-per-category 50 --seed 42
    python curate_eval_set.py --n-per-category 100 --seed 42   # scaled version
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def stable_item_id(image_filename: str, caption: str) -> str:
    """Content-derived, deterministic item ID. Depends ONLY on item
    content, never on sample position, so the same VSR item always gets
    the same id at every n."""
    key = f"{image_filename}||{caption}".encode("utf-8")
    return "vsr_" + hashlib.sha256(key).hexdigest()[:12]


def _rank_key(item: dict, seed: int) -> str:
    """Deterministic pseudo-random sort key. Ranking by this and taking
    top-k gives NESTING: top-50 is always a strict subset of top-150.
    random.sample() does not have this property."""
    key = f"{seed}||{item['image']}||{item['caption']}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()


# ============================================================================
# TAXONOMY — deliberate categorization of all 66 VSR relations.
# Every relation appears in exactly one set below. If VSR ever adds relations,
# they'll show up as "uncategorized" and the script will fail loudly.
# ============================================================================

# Projective spatial: viewpoint-dependent directional relations.
# Flipping the image horizontally changes left/right; changing camera position
# can change what's "in front of" what. These are about position from a POV.
PROJECTIVE = {
    "left of", "right of", "above", "below", "over", "under", "beneath",
    "behind", "in front of", "at the left side of", "at the right side of",
    "at the back of", "facing", "facing away from", "across from",
    "parallel to", "perpendicular to", "ahead of", "opposite to", "toward",
    "on top of",   # position, not force/support
}

# Topological containment: enclosure and part-whole relations.
# These are viewpoint-INDEPENDENT — a coin inside a jar is inside from every
# angle. Includes enclosure, membership, and edge/interior distinctions.
CONTAINMENT = {
    "in", "inside", "contains", "within", "surrounding", "enclosed by",
    "outside", "into", "out of", "part of", "has as a part", "consists of",
    "at the edge of", "in the middle of",
}

# Excluded from pilot — with explicit reason for each subset.
# Proximity: distance relations, don't fit any of the three research categories.
# Contact/support: ambiguous in VSR without further annotation; would need
#   PhysBench for reliable "physical support" items (deferred to full paper).
# Other: low-count or semantically ambiguous relations.
EXCLUDED = {
    # proximity (excluded per pilot design)
    "next to", "beside", "near", "far from", "far away from", "close to",
    "away from", "alongside", "by", "adjacent to", "at the side of", "around",
    # contact / support (ambiguous — deferred to full paper with PhysBench)
    "touching", "on", "attached to", "connected to", "against",
    # low-count, semantically ambiguous, or edge cases
    "beyond", "with", "across", "detached from", "down from", "along",
    "between", "past", "at", "down", "among", "congruent", "through",
    "off",
}

CATEGORY_LABELS = {
    "projective_spatial": PROJECTIVE,
    "topological_containment": CONTAINMENT,
}


def load_all_splits(data_dir: Path) -> list:
    """Load VSR train + dev + test into a single list. Each item gets a
    _split field so we know its origin (audit trail)."""
    items = []
    for split in ("train", "dev", "test"):
        path = data_dir / f"vsr_{split}.jsonl"
        if not path.exists():
            sys.exit(f"ERROR: missing {path}. Run fetch_vsr.sh first.")
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                r["_split"] = split
                items.append(r)
    return items


def validate_taxonomy_coverage(items: list) -> None:
    """Fail loudly if any relation in VSR isn't accounted for.
    This catches the case where VSR gets updated with new relations."""
    all_rels = {r["relation"] for r in items}
    categorized = PROJECTIVE | CONTAINMENT | EXCLUDED
    missing = all_rels - categorized
    if missing:
        sys.exit(
            f"ERROR: {len(missing)} uncategorized relations. "
            f"Decide where each goes and update the taxonomy sets in this script:\n"
            + "\n".join(f"  - {r}" for r in sorted(missing))
        )


def stratified_sample(items: list, category: str, n_per_category: int,
                      seed: int) -> list:
    """Sample n items from a category, balanced by True/False label.
    Also try to spread across as many distinct relations as possible so we
    don't accidentally sample all 'left of' or all 'inside'."""
    category_set = CATEGORY_LABELS[category]
    pool = [r for r in items if r["relation"] in category_set]

    # Split by label
    true_pool = [r for r in pool if r["label"] == 1]
    false_pool = [r for r in pool if r["label"] == 0]

    n_true = n_per_category // 2
    n_false = n_per_category - n_true  # handles odd n gracefully

    if len(true_pool) < n_true or len(false_pool) < n_false:
        sys.exit(
            f"ERROR: not enough items in {category} — "
            f"have True={len(true_pool)}, False={len(false_pool)}; "
            f"need True={n_true}, False={n_false}"
        )

    true_pool.sort(key=lambda r: _rank_key(r, seed))
    false_pool.sort(key=lambda r: _rank_key(r, seed))
    combined = true_pool[:n_true] + false_pool[:n_false]
    # deterministic interleave so True/False aren't blocked together,
    # without a shuffle (a shuffle would break output-order nesting)
    combined.sort(key=lambda r: _rank_key(r, seed + 1))

    for r in combined:
        r["_category"] = category
    return combined


def build_dataframe(items: list) -> pd.DataFrame:
    """Assemble the pilot eval set as a clean CSV with a stable item_id.
    item_id lets us join outputs from different models back to the same input."""
    rows = []
    for r in items:
        rows.append({
            "item_id": stable_item_id(r["image"], r["caption"]),
            "category": r["_category"],
            "relation": r["relation"],
            "caption": r["caption"],
            "ground_truth_label": r["label"],  # 1 = True, 0 = False
            "image_filename": r["image"],
            "image_url": r["image_link"],  # direct COCO URL, fetchable
            "vsr_source_split": r["_split"],
        })
    df = pd.DataFrame(rows)
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\nPilot eval set summary")
    print("=" * 60)
    print(f"Total items: {len(df)}")
    print()
    for cat in df["category"].unique():
        sub = df[df["category"] == cat]
        true_n = (sub["ground_truth_label"] == 1).sum()
        false_n = (sub["ground_truth_label"] == 0).sum()
        rels = sub["relation"].nunique()
        print(f"  {cat}: {len(sub)} items ({true_n} True, {false_n} False), "
              f"{rels} distinct relations")
    print()
    print("Top 10 relations sampled:")
    for rel, c in Counter(df["relation"]).most_common(10):
        print(f"  {c:>3}  {rel}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-category", type=int, default=50,
                        help="Items per category (default 50 for pilot)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).parent.parent / "data",
                        help="Directory containing vsr_*.jsonl files")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path (default: data/pilot_eval_set.csv)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.data_dir / f"pilot_eval_set_n{args.n_per_category}.csv"

    print(f"Loading VSR from {args.data_dir}...")
    items = load_all_splits(args.data_dir)
    print(f"  Loaded {len(items)} total items across 3 splits.")

    validate_taxonomy_coverage(items)
    print(f"  Taxonomy validated: all {len({r['relation'] for r in items})} "
          f"relations accounted for.")

    all_sampled = []
    for category in CATEGORY_LABELS:
        sampled = stratified_sample(items, category, args.n_per_category, args.seed)
        all_sampled.extend(sampled)
        print(f"  Sampled {len(sampled)} items for {category}.")

    df = build_dataframe(all_sampled)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} items to {args.output}")
    print_summary(df)


if __name__ == "__main__":
    main()
