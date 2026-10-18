---
name: record
description: Record the whole straight flight (take-off, 2 km, landing) as videos from the front and back LUCID cameras and the chase camera, on this PC's GPU (queued), from the current code or the accepted realism branch.
argument-hint: "[realism] [extra quadcopter_pbr.py flags]"
disable-model-invocation: true
---
Record the full flight as videos. Arguments: $ARGUMENTS

1. If `echo $CLAUDE_CODE_REMOTE` prints `true`, stop: renders run only on the owner's PC.
2. Code: the current checkout, or with the argument `realism` the accepted state in `.realism/worktrees/base` (create
   it with `git worktree add .realism/worktrees/base realism` if missing). Other arguments are passed on as flags.
3. New folder `.realism/videos/<n>`. Delegate to the render-runner agent:
   `scripts/drone_scene/realism/run_probe.sh record .realism/videos/<n> [code dir] [-- flags]`, then wait (this takes
   long: about 4.5 minutes of flight at 25 fps for three cameras; keep calling wait).
4. Report the absolute paths of `drone_front.mp4`, `drone_back.mp4`, `drone_chase.mp4`, the camera_info JSON files,
   the render time, and any warnings from `drone_scene.log`.
