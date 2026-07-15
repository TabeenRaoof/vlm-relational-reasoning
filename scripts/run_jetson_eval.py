#!/usr/bin/env python3
"""
run_jetson_eval.py

Runs the pilot eval set through Qwen2.5-VL-3B on the Jetson Orin Nano via Ollama
(Path B of JETSON_DECISION_TREE_v2.md — Ollama on-device holds the inference
engine constant with the Mac run, so H3 varies HARDWARE only).

This is a faithful adaptation of run_ollama_eval.py. The ONLY functional
additions are:
  1. --model is required and takes the explicit quant tag (qwen2.5vl:3b-q4_K_M or
     qwen2.5vl:3b-q8_0) — no default, because the whole point is running a
     specific precision.
  2. --label controls the model_name written to the CSV, defaulting to
     "jetson-<model>". This is REQUIRED for the analysis to separate:
       - Mac-3B-Q4  vs  Jetson-3B-Q4   (H3, hardware transfer)
       - Jetson-Q4  vs  Jetson-Q8      (H2, within-3B precision)
     Without a distinct label the Jetson rows would be indistinguishable from
     (or collide with) the Mac rows in analyze_pilot.py.
  3. --limit N restricts the run to the first N rows of --input, sliced BEFORE
     resume/skip logic runs. --limit 1 is a deterministic single-item smoke
     test: it always targets the same first row regardless of what --output
     already contains.
  4. --timeout SECONDS (default 120) bounds each inference call at the
     Ollama/httpx client level — a real transport-layer timeout, not a
     cooperative Python-side timer, so it interrupts a genuinely hung call
     (see the client construction in run_eval() for why this actually fires).
     A timed-out item is logged as a failed row via the existing per-item
     exception path, so a later resume retries it.

Everything else is IDENTICAL to run_ollama_eval.py by construction — the prompt,
parser, decoding options, CSV schema, progressive flush, per-item latency, and
per-item exception handling — because identical treatment across models is the
whole basis of the cross-model comparison.

Requirements on the Jetson (run attended, per the plan):
    1. Ollama installed on the Jetson and `ollama serve` running.
    2. Model pulled:  ollama pull qwen2.5vl:3b-q4_K_M   (and/or :3b-q8_0)
    3. Python deps in the Jetson venv:  pip install ollama pandas requests
    4. This repo (or at least scripts/ + data/) present on the Jetson so the
       shared prompt/parser import and the eval-set CSV resolve.

Record for the methodology section (NOT captured by this script — note manually):
    - Toolchain + exact version (`ollama --version`).
    - Model file: tag + confirmed quant + blob digest (`ollama show <tag>`).
    - Peak memory during a typical inference (from `jtop`).
    - Any behavior-affecting env vars (e.g. OLLAMA_KV_CACHE_TYPE).

Usage (run ON the Jetson, attended):
    # single-item smoke test (Phase 2):
    python run_jetson_eval.py --model qwen2.5vl:3b-q4_K_M \\
        --input data/pilot_eval_set_n50.csv \\
        --output results/jetson_smoketest.csv --limit 1

    # Phase 4 pilot — 100-item set, both quants, catches OOM/timeout early:
    python run_jetson_eval.py --model qwen2.5vl:3b-q4_K_M \\
        --input data/pilot_eval_set_n50.csv \\
        --output results/jetson_qwen25vl_3b_q4_n50.csv
    python run_jetson_eval.py --model qwen2.5vl:3b-q8_0 \\
        --input data/pilot_eval_set_n50.csv \\
        --output results/jetson_qwen25vl_3b_q8_n50.csv

    # Phase 5 full — 300-item set, both quants:
    python run_jetson_eval.py --model qwen2.5vl:3b-q4_K_M \\
        --input data/pilot_eval_set_n150.csv \\
        --output results/jetson_qwen25vl_3b_q4.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# Reuse the parsing logic and prompt from the Gemini script rather than
# duplicating them — cross-model scoring is IDENTICAL by construction. Do NOT
# rewrite these; importing them is the whole point of the comparison.
sys.path.insert(0, str(Path(__file__).parent))
from run_gemini_eval import parse_response, SYSTEM_INSTRUCTION, USER_PROMPT_TEMPLATE  # noqa: E402

import ollama  # noqa: E402


def fetch_image(url: str, cache_dir: Path, filename: str,
                timeout: int = 30) -> bytes:
    """Same fetch/cache pattern as the Mac Ollama script. On the Jetson the
    cache is likely cold, so images are pulled from their COCO URLs on first
    use (Jetson needs network); subsequent items reuse the local cache."""
    cache_path = cache_dir / filename
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return resp.content


def verify_ollama_running(model_tag: str) -> None:
    """Fail loudly and helpfully if Ollama isn't reachable or the model tag
    isn't pulled, instead of a cryptic failure mid-run."""
    try:
        models_response = ollama.list()
    except Exception as e:
        sys.exit(
            "ERROR: Cannot reach Ollama on the Jetson.\n"
            f"  Details: {type(e).__name__}: {e}\n"
            "  Is `ollama serve` running on this device?"
        )
    available = {m.model for m in models_response.models}
    if model_tag not in available and f"{model_tag}:latest" not in available:
        sys.exit(
            f"ERROR: Model '{model_tag}' not found in Ollama on this device.\n"
            f"  Available: {sorted(available)}\n"
            f"  Pull it first: ollama pull {model_tag}"
        )


def run_eval(
    eval_df: pd.DataFrame,
    output_path: Path,
    cache_dir: Path,
    model_tag: str,
    label: str,
    timeout_s: float,
) -> None:
    """Same overall shape as run_ollama_eval.py. model_tag is the tag sent to
    Ollama; label is what lands in the CSV's model_name column. timeout_s
    bounds each inference call via a dedicated ollama.Client (see below) so a
    hung call fails instead of blocking the run indefinitely."""
    verify_ollama_running(model_tag)

    # Dedicated client with an explicit timeout. The module-level ollama.chat()
    # convenience function (used by run_ollama_eval.py) is bound to a default
    # Client() built with timeout=None -- which ollama passes straight through
    # to httpx.Client(timeout=None), meaning NO timeout at all. A stuck read on
    # a wedged daemon or a hung generation would block forever. Passing a
    # numeric timeout here goes to the same httpx.Client(timeout=...) argument,
    # which httpx enforces at the socket/transport layer across connect, read,
    # write, and pool-checkout phases -- a genuine interrupt of a blocking
    # network read, not a cooperative check, so it fires even if the hang is
    # inside Ollama's inference loop and no response bytes ever arrive. Built
    # once here and reused for the whole run (not per-item).
    client = ollama.Client(timeout=timeout_s)

    # Resume support. Only skip items that completed WITHOUT an error — rows
    # written with a non-empty `error` were failures and must be retried on
    # resume, otherwise the effective n silently shrinks. (Matches the fixed
    # behavior in run_ollama_eval.py / run_gemini_eval.py.)
    already_done = set()
    if output_path.exists():
        prior = pd.read_csv(output_path)
        ok = prior[prior["error"].isna() | (prior["error"] == "")]
        already_done = set(ok["item_id"])
        n_err = len(prior) - len(ok)
        print(f"Resuming: found {len(already_done)} successfully-completed items."
              + (f" ({n_err} errored rows will be retried.)" if n_err else ""))

    # Output schema — IDENTICAL to the other eval scripts (do not change).
    fieldnames = [
        "item_id", "category", "relation", "ground_truth_label",
        "model_name", "raw_response", "parsed_label",
        "is_parseable", "is_correct", "latency_seconds", "error",
    ]
    is_new_file = not output_path.exists()
    fh = open(output_path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if is_new_file:
        writer.writeheader()

    n_processed = 0
    n_correct = 0
    n_unparseable = 0
    n_errors = 0
    total_latency = 0.0

    try:
        for _, row in eval_df.iterrows():
            if row["item_id"] in already_done:
                continue

            error = None
            raw_response = ""
            parsed_label = None
            latency = 0.0
            t0 = None   # set only once the timed inference call actually starts

            # Per-item exception handling: one bad item (bad image, transient
            # OOM, decode error, or a timeout from the client above) records an
            # error and moves on — it cannot kill the whole run. This matters
            # most on the Jetson, where a 300-item run is long and slow.
            try:
                image_bytes = fetch_image(
                    url=row["image_url"],
                    cache_dir=cache_dir,
                    filename=row["image_filename"],
                )
                user_prompt = USER_PROMPT_TEMPLATE.format(caption=row["caption"])

                t0 = time.time()
                # Same call and decoding options as the Mac Ollama run:
                # system instruction as a system-role message, image bytes in
                # `images`, temperature 0, num_predict 10. Held constant so the
                # only variable vs the Mac run is the hardware (and, across the
                # two Jetson runs, the quant). Uses the timeout-bound `client`
                # constructed above instead of the module-level ollama.chat().
                response = client.chat(
                    model=model_tag,
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTION},
                        {
                            "role": "user",
                            "content": user_prompt,
                            "images": [image_bytes],
                        },
                    ],
                    options={
                        "temperature": 0.0,
                        "num_predict": 10,   # ollama's max output tokens
                    },
                )
                latency = time.time() - t0
                total_latency += latency
                raw_response = response["message"]["content"].strip()
                parsed_label = parse_response(raw_response)

            except Exception as e:
                # If the timed call had already started, record the REAL
                # elapsed time (e.g. ~timeout_s on a timeout) instead of
                # leaving latency at its 0.0 default — so a hang shows up
                # honestly in latency_table.csv / latency_outliers.csv rather
                # than looking instant.
                if t0 is not None:
                    latency = time.time() - t0
                error = f"{type(e).__name__}: {e}"
                n_errors += 1

            is_parseable = parsed_label is not None
            if not is_parseable:
                n_unparseable += 1
                is_correct = None
            else:
                is_correct = int(parsed_label == row["ground_truth_label"])
                if is_correct:
                    n_correct += 1

            writer.writerow({
                "item_id": row["item_id"],
                "category": row["category"],
                "relation": row["relation"],
                "ground_truth_label": row["ground_truth_label"],
                "model_name": label,          # distinct hardware+quant label
                "raw_response": raw_response,
                "parsed_label": parsed_label if parsed_label is not None else "",
                "is_parseable": is_parseable,
                "is_correct": is_correct if is_correct is not None else "",
                "latency_seconds": round(latency, 3),
                "error": error or "",
            })
            fh.flush()   # crash-safe: every row hits disk immediately

            n_processed += 1
            if n_processed % 10 == 0 or n_processed == len(eval_df):
                scoreable = n_processed - n_unparseable - n_errors
                acc_str = f"{n_correct}/{scoreable}" if scoreable > 0 else "n/a"
                avg_lat = total_latency / max(n_processed - n_errors, 1)
                print(f"  [{n_processed}/{len(eval_df) - len(already_done)}] "
                      f"correct={acc_str} unparseable={n_unparseable} "
                      f"errors={n_errors} avg_latency={avg_lat:.1f}s")

    finally:
        fh.close()

    scoreable = n_processed - n_unparseable - n_errors
    print(f"\nRun complete: processed {n_processed} new items.")
    if scoreable > 0:
        print(f"Accuracy on scoreable items: {n_correct}/{scoreable} "
              f"({100 * n_correct / scoreable:.1f}%)")
    print(f"Unparseable: {n_unparseable}, Errors: {n_errors}")
    if n_processed > 0:
        print(f"Average latency: {total_latency / max(n_processed, 1):.2f}s per item")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to eval-set CSV (pilot_eval_set_n50.csv for "
                             "the Phase-4 pilot; pilot_eval_set_n150.csv for full)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(__file__).parent.parent / "data" / "images")
    parser.add_argument("--model", type=str, required=True,
                        help="Ollama model tag to run on the Jetson, e.g. "
                             "qwen2.5vl:3b-q4_K_M or qwen2.5vl:3b-q8_0")
    parser.add_argument("--label", type=str, default=None,
                        help="model_name written to the CSV (default: "
                             "'jetson-<model>'). Keep Jetson runs distinctly "
                             "labeled so the analysis can pair Mac-vs-Jetson (H3) "
                             "and Q4-vs-Q8 (H2).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N rows of --input, sliced "
                             "BEFORE resume/skip logic runs. --limit 1 gives a "
                             "deterministic single-item smoke test that always "
                             "hits the same first row regardless of what "
                             "--output already contains.")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Per-item inference timeout in seconds, enforced "
                             "at the Ollama/httpx client level (connect, read, "
                             "write, and pool-checkout phases). A hung call "
                             "raises an httpx timeout error after this many "
                             "seconds; the item is logged as a failed row and "
                             "the run continues. Default: 120.")
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    label = args.label if args.label is not None else f"jetson-{args.model}"

    eval_df = pd.read_csv(args.input)
    print(f"Loaded {len(eval_df)} items from {args.input}")
    if args.limit is not None:
        eval_df = eval_df.iloc[:args.limit].reset_index(drop=True)
        print(f"--limit {args.limit}: restricting to the first {len(eval_df)} "
              f"row(s), selected before resume/skip logic")
    print(f"Model tag: {args.model}  |  CSV label: {label}")
    print(f"Per-item timeout: {args.timeout}s")
    print(f"Output: {args.output}\n")

    run_eval(
        eval_df=eval_df,
        output_path=args.output,
        cache_dir=args.cache_dir,
        model_tag=args.model,
        label=label,
        timeout_s=args.timeout,
    )


if __name__ == "__main__":
    main()
