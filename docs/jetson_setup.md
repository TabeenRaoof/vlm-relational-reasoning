# Jetson Setup & Run Protocol (consolidated)

Everything needed to run a VLM evaluation on the NVIDIA Jetson Orin Nano 8GB for
this project, in one place. Consolidated from hard-won trial-and-error across
prior sessions. **Read this fully before starting a Jetson run — several of these
points cost hours to discover.**

> **CRITICAL two-machine note for Cursor / Claude Code:** the agent runs on the
> **Mac**. The Jetson is a *separate physical device* reached only over SSH. The
> agent CANNOT see the Jetson's filesystem, run its commands directly, or observe
> its state except through SSH commands whose output is pasted back. Every Jetson
> command below must run in an SSH session. When in doubt about which machine a
> command is for, check the shell prompt: `tabeen@tabeen-desktop` = Jetson,
> `tabeenraoof@...` = Mac. Run `whoami` first in any pasted block if unsure.

---

## 1. Hardware / environment facts

- Device: NVIDIA Jetson Orin Nano 8GB dev kit, JetPack 6.2.1, 6-core ARM CPU +
  Ampere GPU sharing **7.4 GB usable unified memory**.
- Swap: **zram** (RAM-based compressed swap), NOT disk swap. (An earlier theory
  that swap lived on the SD card and caused slowness was WRONG — don't chase it.)
- Boot storage: 28 GB microSD (dev kit boots from microSD/NVMe; no onboard
  storage). ~21 GB consumed by base JetPack+GNOME+CUDA, leaving very little.
- Model storage: an external **WD Elements USB SSD** (~930 GB, NTFS, mounted at
  `/mnt/ssd`) holds the Ollama models. It also contains the user's personal files
  — **never reformat it.** Ollama points there via `OLLAMA_MODELS` (see §3).

## 2. Connecting from the Mac (SSH)

- `ssh 10.0.0.21` (user `tabeen`, host `tabeen-desktop`, Wi-Fi).
- **`BindAddress` in `~/.ssh/config` is NOT required.** A hardcoded local IP there
  breaks SSH (`bind: Can't assign requested address`) whenever the controlling
  machine's DHCP lease changes. Removing it was verified working, with no side
  effects. If TIME_WAIT exhaustion genuinely recurs, reboot the controlling
  machine rather than hardcoding a bind address.
- **Mac TIME_WAIT port exhaustion:** the Mac (Chrome/VSCode) can accumulate ~24k
  TIME_WAIT sockets causing `ssh: connect ... Can't assign requested address`.
  Fix: reboot the Mac before a session that drives many Jetson requests over SSH.
- **The Jetson's journal does not persist across reboots** (`journalctl
  --list-boots` shows a single boot), so runtime logs from past sessions are
  unrecoverable — capture anything needed (e.g. `dmesg`/`journalctl` evidence of
  an OOM-kill) at collection time, before the next reboot.
- After a Jetson `sudo reboot`, the SSH connection drops; wait ~30-60s and
  reconnect. This is a normal OS-level restart — power stays applied the whole
  time — and is **fully automatic, no physical intervention needed.** The chunked
  run protocol (§5) relies on this: all ~20 reboot cycles can be driven end-to-end
  over SSH by an agent, with no human needed to press anything.
  **Separately:** this device has no physical power button at all. If it ever
  truly loses power or hangs in a way `sudo reboot` can't reach (rare), the only
  recovery is physically unplugging and replugging the power cable — that DOES
  need a person present. A normal reboot does not.

## 3. Ollama configuration (already set — verify, don't recreate)

Systemd override at `/etc/systemd/system/ollama.service.d/override.conf`:
```
[Service]
Environment="OLLAMA_CONTEXT_LENGTH=4096"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_KEEP_ALIVE=30m"
Environment="OLLAMA_MODELS=/mnt/ssd/ollama-models"
```
- `OLLAMA_CONTEXT_LENGTH=4096` + `OLLAMA_KV_CACHE_TYPE=q8_0`: required or the model
  OOMs on load (default tries the model's 128k context).
- `OLLAMA_MODELS=/mnt/ssd/ollama-models`: keeps models off the full 28GB microSD.
- After any edit: `sudo systemctl daemon-reload && sudo systemctl restart ollama`.
- Verify SSD is mounted before running: `mount | grep ssd` (should show
  `/dev/sda1 on /mnt/ssd type fuseblk`). If missing, the model won't be found.

## 4. Model / weight verification (H3 requires this)

- Required model: `qwen2.5vl:3b-q4_K_M`. Confirm present: `ollama list`.
- **H3 depends on byte-identical weights** between Jetson and Mac. The weight-blob
  SHA-256 was verified matching in the pilot
  (`sha256:e9758e589d443f65...`). Before the n=2000 run, re-confirm the Jetson
  blob digest still matches the Mac's, from the manifest under
  `/mnt/ssd/ollama-models/manifests/registry.ollama.ai/library/qwen2.5vl/`.

## 5. THE RUN PROTOCOL — chunked with reboots (this is the important part)

**A single continuous long run WILL be OOM-killed by the kernel.** Confirmed in a
prior 300-item run (`dmesg`: `Out of memory: Killed process ... llama-server`,
~4.3GB resident). Short runs (~100 items) complete cleanly. Therefore:

**Run in ~100-item chunks, with a full `sudo reboot` immediately before each
chunk.** The reboot resets memory fragmentation/pressure; a `systemctl restart
ollama` is NOT sufficient (verified — only a full reboot restores the fast state).

For n=2000, that is ~20 chunks. The `run_jetson_eval.py` script's `--limit` +
resume logic makes this clean: `--limit` slices the first N rows; resume skips
already-successful items, so each chunk advances the boundary.

**Per-chunk loop (repeat, advancing --limit by ~100 each time):**
```
# 1. On the Jetson, reboot:
sudo reboot
# 2. From the Mac, wait ~45s, reconnect:
ssh 10.0.0.21
# 3. On the Jetson, run the next chunk (example: chunk covering up to item 200):
cd ~/cs587-pilot     # NOTE: Jetson repo path is still ~/cs587-pilot (rename it
                     # to match the Mac's vlm-relational-reasoning if desired, but
                     # it's a separate working copy — see §8)
python3 scripts/run_jetson_eval.py --model qwen2.5vl:3b-q4_K_M \
    --input data/eval_set_n2000.csv \
    --output results/jetson_qwen25vl_3b_q4_n2000.csv \
    --limit 200
```
Advance `--limit` (100, 200, 300, ... 2000) across chunks. The final chunk can omit
`--limit` to sweep any stragglers. **Watch the first `[N/...]` checkpoint of each
chunk:** ~1-8s/item = healthy; ~55-65s/item = memory-pressured (still produces
correct data, just slow — a reboot usually restores speed). 0.0s latency +
immediate errors = a load failure (OOM), stop and reboot.

**Realistic time:** at the healthy rate, 2000 items across chunks is ~1-2 hours of
compute plus reboot overhead (~20 reboots × ~1-2 min each). At the slow rate it
can be many hours. **The entire chunked-reboot loop is fully SSH-drivable — no
physical intervention needed** — so an agent can run it unattended once started;
budget generously for wall-clock time either way.

## 6. Data handling after the run

- The raw output will likely have **more rows than 2000** — chunked/retried runs
  leave duplicate rows for items that failed then succeeded. This is EXPECTED and
  correct; do not hand-edit the file.
- Dedup is handled downstream: `analyze_pilot.py` (and the H3 analysis) dedup
  keep-last per (model_name, item_id). 2000 unique items should remain, minus any
  that failed on every attempt (report those as exclusions per the pre-reg §6).
- Pull the file to the Mac when done:
  ```
  scp 10.0.0.21:~/cs587-pilot/results/jetson_qwen25vl_3b_q4_n2000.csv \
      "<mac-repo>/results/"
  ```

## 7. Known failure modes and fixes (quick reference)

| Symptom | Cause | Fix |
|---|---|---|
| `cudaMalloc failed` on load | memory fragmentation/pressure | full `sudo reboot`, then run immediately |
| slow ~60s/item | memory-pressured state | reboot; still-correct data if you let it run |
| 0.0s latency + all errors | model failed to load (OOM) | reboot; check `mount \| grep ssd` and `ollama list` |
| single long run dies mid-way | kernel OOM-kill of llama-server | use the chunked-reboot protocol (§5) |
| `no space left on device` on pull | 28GB microSD full | models live on /mnt/ssd; verify OLLAMA_MODELS; a failed pull leaves a reserved `-partial` blob that `ollama rm` can't clear at 0-free — `sudo rm -v` the exact blob path under the models `blobs/` dir |
| SSH `Can't assign requested address` | Mac TIME_WAIT exhaustion | reboot the Mac |
| Jetson unreachable ~45-60s after `sudo reboot` | still booting (normal) | wait and retry SSH — self-recovers, no action needed, fully autonomous |
| Jetson still unreachable after several minutes / genuinely hung | rare: true power loss or unrecoverable hang | device has no physical power button — the only fix is unplugging/replugging the power cable; needs a person present |

## 8. Repo path note

The Jetson has its OWN working copy of the repo at `~/cs587-pilot` (scp'd in an
earlier session, before the Mac repo was renamed to `vlm-relational-reasoning`).
It is a separate copy, not a git clone of the Mac repo. For the n=2000 run it needs
the frozen `data/eval_set_n2000.csv` and its images — copy them over before
starting:
```
# from the Mac:
scp "<mac-repo>/data/eval_set_n2000.csv" 10.0.0.21:~/cs587-pilot/data/
# images: the Jetson needs the images referenced by the n2000 set. Either scp the
# data/images/ cache, or run the fetch script on the Jetson. Confirm the run script
# can resolve every image before starting a long chunked run (a mid-run image miss
# wastes a reboot cycle).
```
Renaming the Jetson copy to `vlm-relational-reasoning` for consistency is optional
and cosmetic; if done, update the `cd` path in the §5 commands accordingly.
