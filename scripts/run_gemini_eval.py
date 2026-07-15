#!/usr/bin/env python3
"""
run_gemini_eval.py

Runs the pilot eval set through Gemini (default: gemini-2.5-flash) as the
ceiling reference for the study. Each VSR item is a true/false judgment about
a spatial relation in an image.

Approach:
    1. Load pilot_eval_set CSV.
    2. For each item: fetch the image from its COCO URL, ask Gemini to judge
       whether the caption is TRUE or FALSE, parse the model's reply into a
       yes/no/unparseable label, write results to CSV.
    3. Progressive: writes each response as it comes in, so a mid-run failure
       doesn't lose finished work.
    4. Rate-limited politely (default 0.5s between calls) to stay under
       Gemini's free-tier RPM limits.

Environment:
    Requires GEMINI_API_KEY set. Get one at https://aistudio.google.com

Usage:
    export GEMINI_API_KEY="..."
    python run_gemini_eval.py --input data/pilot_eval_set_n50.csv \\
                              --output results/gemini_2_5_flash.csv
    # dry run (no API calls, just prints what it would do):
    python run_gemini_eval.py --input data/pilot_eval_set_n50.csv --dry-run
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# The new (2025+) Google Gen AI SDK. If you get an ImportError here, run:
#   pip install google-genai
# (Do NOT install the deprecated `google-generativeai` package.)
from google import genai
from google.genai import types


# ============================================================================
# Prompt design — DELIBERATE choices, explained.
#
# Format: yes/no true-false. Reasoning:
#   - VSR items are already true/false judgments about a caption's accuracy.
#   - Yes/No scoring is unambiguous; open-ended answers create scoring risk.
#   - Deterministic-ish output (low temperature, short max_output_tokens).
#
# We ask the model to answer with a single word to minimize verbose hedging
# ("Based on the image..."), which makes scoring cleaner.
# ============================================================================

SYSTEM_INSTRUCTION = (
    "You are evaluating whether a caption correctly describes a spatial "
    "relationship in an image. Answer with a single word: YES if the caption "
    "is TRUE for the image, NO if it is FALSE. Do not explain."
)

USER_PROMPT_TEMPLATE = (
    "Caption: {caption}\n"
    "Is this caption TRUE for the image? Answer with a single word: YES or NO."
)

# Inverted-phrasing variant (Step 7 bias diagnostic). The question is flipped to
# ask whether the caption is FALSE. Everything else (single-word yes/no, no
# explanation) is kept identical so the ONLY change is the polarity of the
# question. Scoring is inverted in the loop: a YES here means "the caption is
# false" -> predicted caption-truth = 0; a NO means "not false" -> predicted 1.
SYSTEM_INSTRUCTION_INVERTED = (
    "You are evaluating whether a caption correctly describes a spatial "
    "relationship in an image. Answer with a single word: YES if the caption "
    "is FALSE for the image, NO if it is TRUE. Do not explain."
)

USER_PROMPT_TEMPLATE_INVERTED = (
    "Caption: {caption}\n"
    "Is this caption FALSE for the image? Answer with a single word: YES or NO."
)


def write_run_config(output_path: Path, api_model: str, label: str,
                     thinking: bool, max_output_tokens: int,
                     invert_prompt: bool, input_path: Path) -> None:
    """Write the exact generation config to a sidecar JSON next to the results
    CSV (e.g. results/gemini_2_5_flash.config.json). Makes every run
    self-documenting and lets configs be diff-ed across runs to confirm the ONLY
    intentional difference between conditions is the one under study."""
    config = {
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_model": api_model,
        "csv_label": label,
        "input_eval_set": str(input_path),
        "temperature": 0.0,
        "max_output_tokens": max_output_tokens,
        "thinking": thinking,
        "thinking_budget": ("dynamic (default)" if thinking else 0),
        "prompt_variant": ("inverted" if invert_prompt else "standard"),
        "system_instruction": (
            SYSTEM_INSTRUCTION_INVERTED if invert_prompt else SYSTEM_INSTRUCTION),
        "user_prompt_template": (
            USER_PROMPT_TEMPLATE_INVERTED if invert_prompt
            else USER_PROMPT_TEMPLATE),
    }
    sidecar = output_path.with_suffix(".config.json")
    with open(sidecar, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote run config to: {sidecar}")


# ============================================================================
# Response parsing — case-insensitive, tolerant of leading punctuation and
# trailing commentary. If we can't confidently extract yes/no, return None
# and score it as "unparseable" downstream (not as random noise).
# ============================================================================

YES_PATTERNS = [
    r"^\s*yes\b",           # starts with "yes"
    r"^\s*true\b",          # starts with "true"
    r"^\s*\*+\s*yes\b",     # markdown bold
    r"^\s*answer\s*[:\-]?\s*yes\b",  # "Answer: yes"
]
NO_PATTERNS = [
    r"^\s*no\b",
    r"^\s*false\b",
    r"^\s*\*+\s*no\b",
    r"^\s*answer\s*[:\-]?\s*no\b",
]


def parse_response(text: str) -> Optional[int]:
    """Return 1 for YES/TRUE, 0 for NO/FALSE, or None if unparseable."""
    if not text:
        return None
    lo = text.lower()
    for pat in YES_PATTERNS:
        if re.search(pat, lo):
            return 1
    for pat in NO_PATTERNS:
        if re.search(pat, lo):
            return 0
    return None


# ============================================================================
# Image fetching — VSR image URLs point to COCO.
# We cache locally by filename so re-runs don't hammer the COCO server.
# ============================================================================

def fetch_image(url: str, cache_dir: Path, filename: str,
                timeout: int = 30) -> bytes:
    """Fetch an image, using a local cache. Returns raw bytes."""
    cache_path = cache_dir / filename
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    return resp.content


# ============================================================================
# Main eval loop
# ============================================================================

def run_eval(
    eval_df: pd.DataFrame,
    output_path: Path,
    cache_dir: Path,
    model_name: str,
    api_model: str,
    sleep_between_calls: float,
    dry_run: bool,
    thinking: bool,
    max_output_tokens: int,
    invert_prompt: bool,
) -> None:
    """Run the model on each eval item. Writes progressively to output CSV.

    model_name: the LABEL written to the CSV's model_name column (used by the
        analysis step to distinguish runs, e.g. 'gemini-2.5-flash-thinking').
    api_model: the actual model id sent to the API (e.g. 'gemini-2.5-flash').
        Kept separate from the label so the thinking/no-thinking ablation can
        share one API model while landing in distinct CSVs / columns.
    thinking: if True, leave Gemini's reasoning ON (see config note below).
    max_output_tokens: explicit output-token ceiling. Must be large enough in
        thinking mode to leave room for an answer after reasoning tokens.
    invert_prompt: if True, ask "is the caption FALSE?" and invert scoring
        (Step 7 bias diagnostic). parsed_label is stored in the SAME
        caption-truth space as the standard runs, so it stays comparable.
    """
    # Select the prompt polarity once (identical across all items in a run).
    system_instruction = (
        SYSTEM_INSTRUCTION_INVERTED if invert_prompt else SYSTEM_INSTRUCTION)
    user_prompt_template = (
        USER_PROMPT_TEMPLATE_INVERTED if invert_prompt else USER_PROMPT_TEMPLATE)
    if not dry_run:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            sys.exit(
                "ERROR: GEMINI_API_KEY environment variable not set.\n"
                "Get a key at https://aistudio.google.com and:\n"
                '  export GEMINI_API_KEY="your-key-here"'
            )
        # The client picks up GEMINI_API_KEY automatically from environment
        # when no api_key is passed, but we pass it explicitly for clarity.
        # A per-request timeout is REQUIRED: without it the SDK will block
        # forever if a call stalls (observed intermittently — one open HTTPS
        # connection to Google that never returns). HttpOptions.timeout is in
        # milliseconds; 60s is generous for a single yes/no image call.
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=60_000),
        )

    # Resume support: if output already exists, skip items already scored.
    # IMPORTANT: only skip items that completed WITHOUT an error. Rows written
    # with a non-empty `error` were failures; skipping them on resume would
    # silently shrink the effective n. Treat errored (and unwritten) items as
    # not-done so a rerun retries them.
    already_done = set()
    if output_path.exists():
        prior = pd.read_csv(output_path)
        ok = prior[prior["error"].isna() | (prior["error"] == "")]
        already_done = set(ok["item_id"])
        n_err = len(prior) - len(ok)
        print(f"Resuming: found {len(already_done)} successfully-completed items "
              f"in {output_path}. They will be skipped."
              + (f" ({n_err} errored rows will be retried.)" if n_err else ""))

    # Set up CSV writer in append mode. Write header only if the file is new.
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

                user_prompt = user_prompt_template.format(caption=row["caption"])

                if dry_run:
                    raw_response = "[DRY RUN]"
                    parsed_label = None
                else:
                    t0 = time.time()
                    # Two config modes, selected by the `thinking` flag:
                    #   thinking=False (default): gemini-2.5-flash is a THINKING
                    #     model that by default spends output tokens on hidden
                    #     reasoning before the answer. With only 10 output tokens,
                    #     thinking eats the whole budget -> finish_reason=MAX_TOKENS
                    #     and an EMPTY response.text (~84% empty in testing). We
                    #     DISABLE thinking so it answers YES/NO directly, matching
                    #     how the Qwen2.5-VL tiers answer (single word, no
                    #     thinking) and keeping the cross-model task identical.
                    #   thinking=True: leave thinking ON (dynamic default) and give
                    #     it room (max_output_tokens, >= ~500) for reasoning + the
                    #     final answer. Used for the thinking ablation.
                    # max_output_tokens is passed in explicitly so the exact value
                    # is logged to the sidecar config (see write_run_config).
                    if thinking:
                        gen_config = types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.0,
                            max_output_tokens=max_output_tokens,
                        )
                    else:
                        gen_config = types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.0,
                            max_output_tokens=max_output_tokens,
                            thinking_config=types.ThinkingConfig(
                                thinking_budget=0),
                        )
                    # Retry the call up to 3 times. Combined with the 60s client
                    # timeout above, this recovers from transient stalls/timeouts
                    # in-place instead of writing a permanent error row (which the
                    # resume logic would then skip, leaving the item unscored).
                    response = None
                    last_exc = None
                    for attempt in range(3):
                        try:
                            response = client.models.generate_content(
                                model=api_model,
                                contents=[
                                    types.Part.from_bytes(
                                        data=image_bytes,
                                        mime_type="image/jpeg",
                                    ),
                                    user_prompt,
                                ],
                                config=gen_config,
                            )
                            break  # success — stop retrying
                        except Exception as retry_exc:
                            # Transient stall/timeout/5xx — back off and retry.
                            last_exc = retry_exc
                            time.sleep(2 * (attempt + 1))
                    if response is None:
                        # All 3 attempts failed; re-raise so the outer handler
                        # records it as a per-item error.
                        raise last_exc
                    latency = time.time() - t0
                    raw_response = (response.text or "").strip()
                    parsed_label = parse_response(raw_response)
                    # Inverted-prompt scoring: the model answered "is the caption
                    # FALSE?". A YES (parse=1) means it asserts the caption is
                    # false -> caption-truth prediction 0; a NO (parse=0) -> 1.
                    # Map back into the SAME caption-truth space as the standard
                    # runs so is_correct and downstream joins stay comparable.
                    if invert_prompt and parsed_label is not None:
                        parsed_label = 1 - parsed_label

            except Exception as e:
                # Broad catch so ONE bad item doesn't abort the whole run.
                # Recorded per-item so we can inspect what failed.
                error = f"{type(e).__name__}: {e}"
                n_errors += 1

            is_parseable = parsed_label is not None
            if not is_parseable:
                n_unparseable += 1
                is_correct = None  # can't score
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
            fh.flush()  # write immediately so a crash doesn't lose data

            n_processed += 1
            if n_processed % 10 == 0 or n_processed == len(eval_df):
                scoreable = n_processed - n_unparseable - n_errors
                acc_str = f"{n_correct}/{scoreable}" if scoreable > 0 else "n/a"
                print(f"  [{n_processed}/{len(eval_df) - len(already_done)}] "
                      f"correct={acc_str} unparseable={n_unparseable} "
                      f"errors={n_errors}")

            if not dry_run and sleep_between_calls > 0:
                time.sleep(sleep_between_calls)

    finally:
        fh.close()

    scoreable = n_processed - n_unparseable - n_errors
    print(f"\nRun complete: processed {n_processed} new items.")
    if scoreable > 0:
        print(f"Accuracy on scoreable items: {n_correct}/{scoreable} "
              f"({100 * n_correct / scoreable:.1f}%)")
    print(f"Unparseable: {n_unparseable}, Errors: {n_errors}")
    print(f"Output written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to pilot_eval_set CSV")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path to output CSV (results/gemini_*.csv)")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(__file__).parent.parent / "data" / "images",
                        help="Directory to cache downloaded images")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                        help="Gemini model name")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Seconds to sleep between calls (rate-limit)")
    parser.add_argument("--thinking", action="store_true",
                        help="Leave Gemini's reasoning ON (default: disabled so "
                             "the model answers YES/NO directly). Use for the "
                             "thinking-vs-no-thinking ablation.")
    parser.add_argument("--label", type=str, default=None,
                        help="Value written to the CSV's model_name column "
                             "(defaults to --model, plus '-thinking'/'-inverted' "
                             "suffixes for those variants). Keep runs in distinct "
                             "CSVs with distinct labels so the analysis step "
                             "treats them as separate models.")
    parser.add_argument("--invert-prompt", action="store_true",
                        help="Ask 'is the caption FALSE?' instead of TRUE, and "
                             "invert scoring accordingly (Step 7 bias check). "
                             "parsed_label is stored in the same caption-truth "
                             "space, so results stay directly comparable.")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="Explicit output-token ceiling. Default: 10 for "
                             "no-thinking, 1024 for --thinking. (1024, not ~500: "
                             "a probe showed thinking spends ~980 tokens before "
                             "answering, so a lower ceiling starves the answer "
                             "and reproduces the empty-response bug.)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't call the API; just walk the eval set")
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Resolve the effective output-token ceiling. Thinking mode MUST have a high
    # enough ceiling to leave room for an answer after reasoning tokens are
    # spent; 10 (fine for direct yes/no) would starve it.
    if args.max_output_tokens is not None:
        max_out = args.max_output_tokens
    elif args.thinking:
        max_out = 1024  # empirically ~980 thinking tokens observed; 512 starves it
    else:
        max_out = 10

    # Resolve the CSV label. Auto-suffix so variant runs never silently collide
    # with the baseline in the analysis step.
    if args.label is not None:
        label = args.label
    else:
        label = args.model
        if args.thinking:
            label += "-thinking"
        if args.invert_prompt:
            label += "-inverted"

    eval_df = pd.read_csv(args.input)
    print(f"Loaded {len(eval_df)} items from {args.input}")
    print(f"API model: {args.model}  |  CSV label: {label}")
    print(f"thinking: {args.thinking}  |  invert_prompt: {args.invert_prompt}  "
          f"|  max_output_tokens: {max_out}")
    print(f"Output: {args.output}")
    if args.dry_run:
        print("DRY RUN — no API calls will be made.")
    print()

    # Write the exact generation config to a sidecar JSON next to the CSV, so
    # every run is self-documenting and configs are diff-able across runs.
    if not args.dry_run:
        write_run_config(
            output_path=args.output,
            api_model=args.model,
            label=label,
            thinking=args.thinking,
            max_output_tokens=max_out,
            invert_prompt=args.invert_prompt,
            input_path=args.input,
        )

    run_eval(
        eval_df=eval_df,
        output_path=args.output,
        cache_dir=args.cache_dir,
        model_name=label,
        api_model=args.model,
        sleep_between_calls=args.sleep,
        dry_run=args.dry_run,
        thinking=args.thinking,
        max_output_tokens=max_out,
        invert_prompt=args.invert_prompt,
    )


if __name__ == "__main__":
    main()
