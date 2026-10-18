---
name: fleet-status
description: Say exactly what the drone-scene agents and renders are doing right now on this PC - running and queued renders, GPU use, the last realism round, branches and worktrees. Use when the owner asks "are you running?", "what is happening?", "what happened?" or "how is the scene?".
---
Answer the owner's question about the current state, plainly and exactly. Read-only: do not start or stop anything.

Gather (from the main checkout):
- Machine: `hostname`, and whether `CLAUDE_CODE_REMOTE` is `true` (a cloud machine: say so first).
- Renders: every `run.status` under `.realism/` (`find .realism -name run.status -newer .realism/config.env` or all of
  them), with its folder and the last line of its `drone_scene.log`; Isaac processes (`pgrep -af "quadcopter_pbr|kit"`).
- GPU: `nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv`.
- Loop: the last lines of `.realism/ledger.jsonl` (round, candidate, decision, why), the newest `.realism/rounds/*`.
- Code: current branch, `git log --oneline -3`, `git log --oneline main..realism` (accepted, not yet on main),
  `git worktree list`, and whether anything is not pushed (`git status -sb`).

Answer first in one or two sentences: what is running now (or that nothing is), on which machine, and what finished
last. Then only the details that matter, with absolute paths (contact sheet of the accepted state:
`.realism/baseline/contact.jpg`). If the owner asked how the scene looks, open that contact sheet and describe it.
