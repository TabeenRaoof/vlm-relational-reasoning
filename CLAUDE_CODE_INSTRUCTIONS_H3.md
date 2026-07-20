# Instructions for Claude Code (in Cursor) — H3 n=2000 Equivalence Study

You are running in **Cursor on the user's Mac**, in the `vlm-relational-reasoning`
repo. This file is your brief. Read, in order: `docs/RUN_PLAN_H3.md`,
`preregistration/PREREGISTRATION_H3_n2000.md`, `docs/jetson_setup.md`.

## What this study is (short version)

A prior study found the same VLM (Qwen2.5-VL-3B, Q4_K_M) runs on a Jetson Orin Nano
edge device with accuracy indistinguishable from a Mac desktop (n=300, McNemar
p=0.58, diff ~1pp). This study re-tests at n=2000 with a formal **equivalence test
(TOST, ±3pp margin)** to make the stronger positive claim: edge deployment changes
accuracy by less than 3 percentage points.

## The two-machine reality — internalize this

**You run on the Mac. The Jetson is a separate physical device reachable ONLY over
SSH.** You cannot see the Jetson's files, run its commands directly, or observe its
state except by issuing SSH commands and reading their output. Consequences:
- Never assume a Jetson command worked — check its output every time.
- Reboots (`sudo reboot`) are fully autonomous over SSH: power stays applied,
  the connection drops, wait ~45-60s, reconnect, continue. **No physical
  intervention needed** — the device has no power button at all. You (the agent)
  can drive the entire ~20-chunk reboot loop yourself, end to end, without
  stopping for the user at each cycle.
  **The one real physical-intervention case:** if the device ever truly loses
  power or hangs in a way `sudo reboot` can't reach (rare), the only fix is
  unplugging/replugging the power cable — that DOES need the user. If a chunk's
  reboot doesn't come back after a few minutes and a retry or two, stop and tell
  the user rather than looping indefinitely.
- Shell prompt tells you the machine: `tabeen@tabeen-desktop` = Jetson,
  `tabeenraoof@...` = Mac.

## The single most important scientific rule

**Freeze (commit) the pre-registration BEFORE collecting any Jetson n=2000 data.**
The n=300 result is already known; the validity of this confirmatory re-test depends
on the design/margin being committed before the new data exists. Do not run the
Jetson evaluation until Phase 0's gate is cleared and the pre-reg is committed.

## Work in gated phases (docs/RUN_PLAN_H3.md) — stop at each 🔴 GATE

- **Phase 0:** finalize + commit the pre-registration. Get explicit user
  confirmation before committing. Do not fill scientific choices yourself (margin
  is already ±3pp; leave `<FILL>`s that need user/institutional input).
- **Phase 1:** prep the Jetson over SSH (reachability, copy eval set + images,
  re-verify the weight digest). The digest match (GATE 1.3) is what makes this a
  hardware-only test — confirm it.
- **Phase 2:** the chunked-reboot Jetson run — ~20 cycles, fully SSH-drivable and
  runs autonomously/unattended once started (no physical steps required). Confirm
  with the user before kicking off the whole multi-hour run, then drive the
  reboot+run loop yourself; report progress periodically (not every chunk) rather
  than pausing at each one. Watch each chunk's first latency checkpoint, and stop
  to tell the user only if something looks genuinely wrong (repeated failures, a
  reboot that doesn't come back after several minutes).
- **Phase 3:** pull the file, run `h3_equivalence_test.py` exactly as pre-registered.
  Report per PREREGISTRATION_H3_n2000.md §7 whatever the outcome.

## Guardrails

- **Do not re-run the Mac baseline** (`results/mac_qwen25vl_3b_n2000.csv`) — it's
  the frozen comparison baseline; re-running it would be changing the baseline
  after the fact.
- **Do not collect more data to force an equivalence declaration** — §7 of the
  pre-reg forbids this (optional stopping). If equivalence isn't declared, report
  honestly.
- **Never commit `.env`.** Verify `git log --all --full-history -- .env` is empty
  before any push. (This study makes no API calls, so no key is needed, but the
  repo hygiene rule stands.)
- `scripts/h3_equivalence_test.py` is provided and verified — use it, don't rewrite
  it unless it fails to run (then fix minimally and tell the user what changed).
- Preserve the user's data; you're adding files, not replacing.
- Show plans before executing multi-step or physical/SSH operations.

## Definition of done

- Pre-registration committed before Jetson n=2000 data (git log shows it).
- `results/jetson_qwen25vl_3b_q4_n2000.csv` complete (2000 unique items after
  dedup, minus reported exclusions).
- TOST run; result written to `results/analysis_h3_n2000/equivalence.txt`.
- Outcome reported per pre-reg §7, with the n=300 result alongside and a
  Deviations note.
