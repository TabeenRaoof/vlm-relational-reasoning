#!/usr/bin/env python3
"""
run_ollama_eval.py

Runs the pilot eval set through a local vision model via Ollama on your Mac.
By default, uses qwen2.5vl:7b as the mid-tier reference in the compute
spectrum (Jetson 3B < Mac 7B < Gemini frontier).

Requirements before running:
    1. Install Ollama on your Mac: https://ollama.com/download
    2. Pull the model:
          ollama pull qwen2.5vl:7b
    3. Ensure Ollama is running (open the Ollama.app or `ollama serve`).
    4. Install ollama-python in your venv:
          pip install ollama

Design notes (deliberate, and shared with run_gemini_eval.py):
    - Yes/No prompting format matching VSR's true/false structure.
    - Low temperature (0.0) for deterministic output; short output cap.
    - Same prompt and parser as Gemini eval so cross-model comparison is fair
      — the ONLY intentional difference across models is what runs the tokens.
    - Progressive write: crash-safe, one row per item as it completes.
    - Resume: skips items already in the output CSV.
    - Rate-limit is off by default since this is local, not an API.

Usage:
    python run_ollama_eval.py --input data/pilot_eval_set_n50.csv \\
                              --output results/mac_qwen25vl_7b.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# Reuse the parsing logic from the Gemini script rather than duplicating it —
# the whole point of parsing being a pure function is that it's reusable and
# testable. This ensures cross-model scoring is IDENTICAL by construction.
sys.path.insert(0, str(Path(__file__).parent))
from run_gemini_eval import parse_response, SYSTEM_INSTRUCTION, USER_PROMPT_TEMPLATE  # noqa: E402

import ollama  # noqa: E402


def fetch_image(url: str, cache_dir: Path, filename: str,
                timeout: int = 30) -> bytes:
    """Same fetch/cache pattern as the Gemini script."""
    cache_path = cache_dir / filename
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return resp.content


def verify_ollama_running(model_name: str) -> None:
    """Fail loudly and helpfully if Ollama isn't reachable or the model
    isn't pulled. This gives a clear Monday-morning error message
    instead of a cryptic connection failure mid-eval."""
    try:
        models_response = ollama.list()
    except Exception as e:
        sys.exit(
            "ERROR: Cannot reach Ollama.\n"
            f"  Details: {type(e).__name__}: {e}\n"
            "  Is the Ollama app running on your Mac?\n"
            "  Start it, or run `ollama serve` in a terminal."
        )
    # ollama.list() returns an object with .models; each has .model = 'name:tag'
    available = {m.model for m in models_response.models}
    if model_name not in available:
        # try with an implicit :latest if user typed bare name
        if f"{model_name}:latest" not in available:
            sys.exit(
                f"ERROR: Model '{model_name}' not found in Ollama.\n"
                f"  Available: {sorted(available)}\n"
                f"  Pull it first: ollama pull {model_name}"
            )


def run_eval(
    eval_df: pd.DataFrame,
    output_path: Path,
    cache_dir: Path,
    model_name: str,
) -> None:
    """Same overall shape as the Gemini eval — the *only* different code path
    is how we send the request. Everything else identical for comparability."""
    verify_ollama_running(model_name)

    # Resume support. Only skip items that completed WITHOUT an error — rows
    # written with a non-empty `error` were failures and must be retried on
    # resume, otherwise the effective n silently shrinks.
    already_done = set()
    if output_path.exists():
        prior = pd.read_csv(output_path)
        ok = prior[prior["error"].isna() | (prior["error"] == "")]
        already_done = set(ok["item_id"])
        n_err = len(prior) - len(ok)
        print(f"Resuming: found {len(already_done)} successfully-completed items."
              + (f" ({n_err} errored rows will be retried.)" if n_err else ""))

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

            try:
                image_bytes = fetch_image(
                    url=row["image_url"],
                    cache_dir=cache_dir,
                    filename=row["image_filename"],
                )
                user_prompt = USER_PROMPT_TEMPLATE.format(caption=row["caption"])

                t0 = time.time()
                # Ollama accepts image bytes directly in `images`.
                # System instruction goes via a separate system-role message.
                response = ollama.chat(
                    model=model_name,
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
                "model_name": model_name,
                "raw_response": raw_response,
                "parsed_label": parsed_label if parsed_label is not None else "",
                "is_parseable": is_parseable,
                "is_correct": is_correct if is_correct is not None else "",
                "latency_seconds": round(latency, 3),
                "error": error or "",
            })
            fh.flush()

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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(__file__).parent.parent / "data" / "images")
    parser.add_argument("--model", type=str, default="qwen2.5vl:7b",
                        help="Ollama model name (default: qwen2.5vl:7b)")
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    eval_df = pd.read_csv(args.input)
    print(f"Loaded {len(eval_df)} items from {args.input}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}\n")

    run_eval(
        eval_df=eval_df,
        output_path=args.output,
        cache_dir=args.cache_dir,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
