# RUN PLAN — H3 Edge-Equivalence Study (n=2000)

Ordered, gated plan for the Jetson H3 equivalence study. Designed for execution by
an agent in **Cursor / Claude Code running on the Mac**, driving the Jetson over
SSH. Read `docs/jetson_setup.md` and `preregistration/PREREGISTRATION_H3_n2000.md`
before starting.

**Two-machine reality (critical):** the agent runs on the Mac. The Jetson is a
separate device reached only via SSH; the agent cannot see its filesystem or state
except through SSH command output. Never assume a Jetson command succeeded without
checking its output. Prompt shell check: `tabeen@tabeen-desktop` = Jetson,
`tabeenraoof@...` = Mac.

🔴 GATE = do not proceed until the condition is met and recorded.

---

## PHASE 0 — Freeze the pre-registration (before any Jetson n=2000 data)

**0.1** Finalize `preregistration/PREREGISTRATION_H3_n2000.md` — fill every
`<FILL>`, confirm margin ±3pp and the digest-match note, remove the DRAFT banner.

**0.2** Confirm the Mac baseline already exists and is untouched:
`results/mac_qwen25vl_3b_n2000.csv` (collected in the prior n=2000 Mac run). Do NOT
re-run it — it's the frozen baseline for this comparison.

**0.3** Commit the pre-registration BEFORE collecting Jetson n=2000 data:
```
git add preregistration/PREREGISTRATION_H3_n2000.md scripts/h3_equivalence_test.py docs/jetson_setup.md docs/RUN_PLAN_H3.md
git commit -m "Freeze H3 n=2000 equivalence pre-registration (pre-data)"
git rev-parse HEAD   # record in §9, then amend per the one-amend-behind note
```
🔴 **GATE 0.3:** pre-registration committed; no Jetson n=2000 data collected yet.
(git log is the authoritative freeze record.)

---

## PHASE 1 — Prepare the Jetson

The Jetson has its own working copy at `~/cs587-pilot` (separate from the Mac repo).
See docs/jetson_setup.md §8.

**1.1** Verify SSH reachability and the Jetson's state (all over SSH):
```
ssh 10.0.0.21
# on the Jetson:
whoami                        # expect: tabeen
mount | grep ssd              # expect: /dev/sda1 on /mnt/ssd
ollama list                   # expect: qwen2.5vl:3b-q4_K_M present
```
If SSH fails with "Can't assign requested address", reboot the Mac (TIME_WAIT
exhaustion) and retry.

**1.2** Copy the frozen eval set + images to the Jetson (from the Mac):
```
scp "<mac-repo>/data/eval_set_n2000.csv" 10.0.0.21:~/cs587-pilot/data/
# images: ensure the Jetson can resolve every image referenced by the n2000 set.
# Either scp the data/images/ cache or run the fetch on the Jetson. A mid-run
# image miss wastes a reboot cycle, so verify coverage BEFORE the chunked run.
```

**1.3** (H3 integrity) Re-verify the Jetson Q4 weight-blob SHA-256 matches the
Mac's, from the manifest under `/mnt/ssd/ollama-models/manifests/.../qwen2.5vl/`.
Record the digest in the pre-reg §4. 🔴 **GATE 1.3:** digests match (this is what
makes H3 a hardware-only comparison).

---

## PHASE 2 — Run the Jetson (chunked-reboot protocol)

Follow docs/jetson_setup.md §5 exactly. **A single continuous 2000-item run WILL be
OOM-killed** — you MUST chunk with a reboot before each chunk.

**This entire loop is fully SSH-drivable and autonomous** — `sudo reboot` keeps
power applied and the device comes back on its own; no physical button exists or
is needed. An agent can run all ~20 chunks unattended once the user confirms the
run should start. The only scenario needing a person present is a genuine
power-loss/hang that a reboot can't recover from (rare) — if a chunk's reboot
doesn't come back after a few minutes and retries, stop and flag it rather than
looping indefinitely.

For each chunk (advance `--limit` by ~100: 100, 200, ... 2000):
```
# Jetson: reboot to reset memory state
sudo reboot
# Mac: wait ~45s, reconnect
ssh 10.0.0.21
# Jetson: run the next chunk
cd ~/cs587-pilot
python3 scripts/run_jetson_eval.py --model qwen2.5vl:3b-q4_K_M \
    --input data/eval_set_n2000.csv \
    --output results/jetson_qwen25vl_3b_q4_n2000.csv \
    --limit <chunk_boundary>
```
- Watch the first `[N/...]` checkpoint of each chunk. Healthy ~1-8s/item; slow
  ~55-65s/item (still correct, just slow); 0.0s latency + errors = load failure,
  reboot and retry.
- ~20 chunks total for n=2000. Resume skips completed items, so a re-run after a
  crash is safe and idempotent.

🔴 **GATE 2:** `results/jetson_qwen25vl_3b_q4_n2000.csv` covers all 2000 items
(unique item_ids after dedup = 2000, minus any that failed every attempt). Report
error/unparseable counts + per-category rate.

---

## PHASE 3 — Analysis (execute the pre-registered TOST exactly)

**3.1** Pull the Jetson file to the Mac:
```
scp 10.0.0.21:~/cs587-pilot/results/jetson_qwen25vl_3b_q4_n2000.csv "<mac-repo>/results/"
```

**3.2** Run the pre-registered equivalence test:
```
python scripts/h3_equivalence_test.py \
    --jetson results/jetson_qwen25vl_3b_q4_n2000.csv \
    --mac    results/mac_qwen25vl_3b_n2000.csv \
    --margin 3.0 \
    --output results/analysis_h3_n2000/equivalence.txt
```
(The script deduplicates the Jetson file keep-last per (model_name,item_id) and
pairs on items scoreable both sides — matching pre-reg §6.)

🔴 **GATE 3:** results in. Report per PREREGISTRATION_H3_n2000.md §7 — whichever of
the three outcomes occurred. Do NOT collect more data to force an equivalence
declaration (that's optional stopping; §7 forbids it).

---

## PHASE 4 — Record

- Commit the Jetson n=2000 result + analysis output.
- Report the outcome honestly (equivalence declared / not / insufficient precision).
- Include a "Deviations from Pre-Registration" note (even if empty).
- The n=300 result is reported alongside for continuity.
