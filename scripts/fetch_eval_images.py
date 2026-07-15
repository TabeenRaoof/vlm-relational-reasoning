#!/usr/bin/env python3
"""
fetch_eval_images.py

Populates data/images/ with every COCO image referenced by an eval-set CSV,
skipping images already cached. Images are fetched once per distinct
image_filename (many items share an image with different captions), then the
result is fanned back out to every item_id that depends on that image.

This directly feeds PREREGISTRATION.md §6's exclusion criterion: "Items whose
COCO image fails to download are excluded before model runs; count and
per-category failure rate are reported." Every failure is logged with its
item_id, category, image_filename, image_url, and the exact exception —
nothing is silently skipped.

Usage:
    python scripts/fetch_eval_images.py --input data/eval_set_n2000.csv \
        --cache-dir data/images \
        --failures-log results/image_fetch_failures_n2000.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import requests


def fetch_one(url: str, cache_path: Path, timeout: int, retries: int) -> tuple[bool, str]:
    """Attempt to fetch a single image, with retries on transient errors.
    Returns (success, error_message). error_message is "" on success."""
    last_err = ""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            if len(resp.content) == 0:
                last_err = "EmptyResponse: 0 bytes returned"
                continue
            cache_path.write_bytes(resp.content)
            return True, ""
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    return False, last_err


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True,
                     help="Eval-set CSV (must have item_id, category, "
                          "image_filename, image_url columns)")
    ap.add_argument("--cache-dir", type=Path,
                     default=Path(__file__).parent.parent / "data" / "images")
    ap.add_argument("--failures-log", type=Path, required=True,
                     help="Where to write the per-item failure CSV")
    ap.add_argument("--timeout", type=int, default=30,
                     help="Per-request timeout in seconds (default 30)")
    ap.add_argument("--retries", type=int, default=3,
                     help="Retry attempts per image before giving up (default 3)")
    args = ap.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.failures_log.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} items from {args.input}")

    # Fetch once per distinct image, not once per item — many items share an
    # image with different captions/relations.
    distinct_images = df[["image_filename", "image_url"]].drop_duplicates(
        subset="image_filename")
    already_cached = {
        row.image_filename for row in distinct_images.itertuples()
        if (args.cache_dir / row.image_filename).exists()
        and (args.cache_dir / row.image_filename).stat().st_size > 0
    }
    to_fetch = distinct_images[~distinct_images["image_filename"].isin(already_cached)]
    print(f"Distinct images referenced: {len(distinct_images)}")
    print(f"Already cached: {len(already_cached)}")
    print(f"New images to fetch: {len(to_fetch)}\n")

    failed_filenames = {}  # image_filename -> error message
    n_ok = 0
    n_fail = 0
    t_start = time.time()

    for i, row in enumerate(to_fetch.itertuples(), start=1):
        cache_path = args.cache_dir / row.image_filename
        ok, err = fetch_one(row.image_url, cache_path, args.timeout, args.retries)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            failed_filenames[row.image_filename] = err
        if i % 100 == 0 or i == len(to_fetch):
            elapsed = time.time() - t_start
            print(f"  [{i}/{len(to_fetch)}] ok={n_ok} failed={n_fail} "
                  f"elapsed={elapsed:.0f}s")

    print(f"\nFetch pass complete: {n_ok} succeeded, {n_fail} failed "
          f"(of {len(to_fetch)} new images attempted).")

    # Fan failures back out to every item_id that depends on a failed image,
    # and write the per-item failure log (item_id, category, and error).
    failure_rows = []
    for row in df.itertuples():
        if row.image_filename in failed_filenames:
            failure_rows.append({
                "item_id": row.item_id,
                "category": row.category,
                "image_filename": row.image_filename,
                "image_url": row.image_url,
                "error": failed_filenames[row.image_filename],
            })

    with open(args.failures_log, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["item_id", "category", "image_filename",
                             "image_url", "error"])
        writer.writeheader()
        writer.writerows(failure_rows)
    print(f"\nPer-item failure log written to: {args.failures_log} "
          f"({len(failure_rows)} affected items)")

    # Per-category breakdown (maps to PREREGISTRATION.md §6 exclusions).
    print("\n" + "=" * 68)
    print("PER-CATEGORY FAILURE BREAKDOWN (for §6 exclusion reporting)")
    print("=" * 68)
    total_by_cat = df["category"].value_counts()
    failed_item_ids = {r["item_id"] for r in failure_rows}
    for cat in sorted(total_by_cat.index):
        total = total_by_cat[cat]
        failed = sum(1 for r in failure_rows if r["category"] == cat)
        pct = 100 * failed / total if total else 0.0
        print(f"  {cat}: {failed}/{total} items excluded ({pct:.2f}%)")

    overall_total = len(df)
    overall_failed = len(failed_item_ids)
    overall_pct = 100 * overall_failed / overall_total if overall_total else 0.0
    print(f"\n  OVERALL: {overall_failed}/{overall_total} items excluded "
          f"({overall_pct:.2f}%)")


if __name__ == "__main__":
    main()
