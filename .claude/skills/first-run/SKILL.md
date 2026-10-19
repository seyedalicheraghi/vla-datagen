---
name: first-run
description: Bring the drone scene up on this PC for the first time (or after a big change) - environment check, unit tests, a quick probe render, log triage and fixes until it works, the full baseline probe, real NAIP references and a noise calibration for the realism loop.
argument-hint: "[--skip-calibration]"
disable-model-invocation: true
---
Bring the drone scene up on the owner's PC, fix what fails, and leave the realism loop ready. Arguments: $ARGUMENTS

Work from the main checkout (repo root). Keep the owner informed with one short line per step: what you are doing,
on which machine and in which folder. Renders go only through `scripts/drone_scene/realism/run_probe.sh` (start, then
wait in steps of up to 540 s); delegate them to the render-runner agent. Give long commands (`run_tests.sh`,
`run_probe.sh wait`) the Bash tool timeout 600000 ms.

1. **Machine.** If `echo $CLAUDE_CODE_REMOTE` prints `true`, stop: this is a cloud machine, and the owner runs code
   only on their own PC. Otherwise check the GPU (`nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version
   --format=csv`) and whether another Isaac Sim is running (`pgrep -af "isaac|kit"`); report both.
2. **Isaac Lab.** Find the folder whose `./isaaclab.sh -p` runs Isaac Sim's Python: this checkout if it has a working
   `./isaaclab.sh` (`./isaaclab.sh -p -c "import isaacsim, sys; print(sys.version)"`), otherwise ask the owner for the
   path. Write it to `.realism/config.env` as `ISAACLAB_DIR=<absolute path>` (create `.realism/`; it is git-ignored).
3. **Unit tests.** `scripts/drone_scene/realism/run_tests.sh`. `test_world` needs the map tools, which the first
   straight-route render installs: a "no map tools" failure is expected now and is re-checked after step 4. Fix other
   failures (the isaac-debugger agent can help) before rendering.
4. **Quick probe.** render-runner: `run_probe.sh start .realism/first-run -- --probe cruise_mid final --cameras front
   back nadir`, then wait. The first run also installs the map tools, downloads ~150 MB of map data and assets, and
   builds the map (several minutes, CPU). Then read `.realism/first-run/drone_scene.log` and report:
   - `HdrColor`: does the sensor simulation get it, or is there a "no HdrColor" warning?
   - the "mean scene radiance = ... x an 18 % grey card" line (0.03-30 is plausible);
   - the camera mount lines (% of view on the drone), the `[rtx_look] WARNING` lines, the `[probe]` timing lines;
   - any traceback.
   Look at the stills yourself (`.realism/first-run/probe/*.png`, `.realism/first-run/contact.jpg`).
   On failure, give the log to the isaac-debugger agent, let it fix and re-check, and repeat (at most 3 rounds; then
   stop and tell the owner exactly what fails, with the log lines).
5. **Tests again.** `scripts/drone_scene/realism/run_tests.sh` (now including `test_world`).
6. **Commit** the fixes you made, locally, with a clear message, on the branch the checkout is on (normally `main`).
   Do not push: ask the owner first.
7. **Baseline.** If the branch `realism` does not exist: `git branch realism HEAD`. Then
   `git worktree add .realism/worktrees/base realism` (if missing). If `realism` already existed and lacks the
   checkout's commits (`git merge-base --is-ancestor HEAD realism` fails), merge them:
   `git -C .realism/worktrees/base merge --no-edit $(git rev-parse HEAD)` (on a conflict: abort and tell the owner).
   Then render-runner: `run_probe.sh start .realism/baseline .realism/worktrees/base`, wait. Then `run_probe.sh naip .realism/baseline` for the real NAIP
   orthophotos of the nadir squares, and look at `.realism/baseline/metrics_vs_naip.json`.
8. **Noise calibration** (skip with --skip-calibration). Render the same code again into `.realism/calibration`, then
   `run_probe.sh compare .realism/baseline .realism/calibration 1`. Identical code, so every difference is noise: note
   the biggest metric differences. Run the realism-judge agent on the pictures in `.realism/calibration/ab/` (normal and
   `_swap` sets in separate calls, the "photo" question): with identical code the judges should mostly say tie or
   flip between orders. Write the noise levels and the judges' false-preference rate into
   `scripts/drone_scene/realism/LESSONS.md` under "Calibration" in `.realism/worktrees/base`, and commit it there
   (`git -C .realism/worktrees/base commit -am "realism: calibration"`).
9. **Report to the owner** in a few lines: what works, what was fixed (files), timings (map build, Isaac start, per
   probe), the paths to look at (`.realism/baseline/contact.jpg`, the stills), and ask whether they have real photos
   or footage (best: from their own LUCID camera) to put in `.realism/reference/footage/` for the critics.
