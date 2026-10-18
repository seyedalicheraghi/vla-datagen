---
name: realism
description: Run rounds of the agent-fleet loop that makes the drone scene look more like real drone footage - critics, planner, parallel builders in worktrees, queued GPU renders, blind A/B judges in both orders, merge of the winners onto the local branch realism.
argument-hint: "[rounds=2] [candidates=3] [focus words, e.g. light ground urban vegetation camera]"
disable-model-invocation: true
---
Run the realism loop for the owner. Arguments: $ARGUMENTS

Before starting:
- If `echo $CLAUDE_CODE_REMOTE` prints `true`, stop: the loop renders, and renders run only on the owner's PC.
- `/first-run` must have succeeded on this PC: `.realism/config.env` exists and `.realism/baseline/probe/probe.json`
  exists. If not, say so and offer to run /first-run.
- Parse the arguments: a number alone = rounds (default 2), `candidates=N` (default 3, at most 6), `lenses=N`
  (default 3), other words = focus (aspect keys light / ground / urban / vegetation / camera, or free text).
- Tell the owner in one line what will happen and roughly how long: each round renders every candidate one at a
  time (about 5-15 minutes each, more when the map is rebuilt) plus one combined render.

Each round:
1. Round id: `r` + three digits, one more than the highest folder in `.realism/rounds/` (first: r001).
2. Run the saved workflow **realism-round** with the Workflow tool:
   `{"name": "realism-round", "args": {"round": "<id>", "candidates": N, "lenses": N, "focus": "<text>"}}`.
3. When it finishes, report to the owner in at most 6 lines: what was tried, what won or lost and why (one phrase
   each), what was merged onto `realism`, and the path of the new contact sheet (`.realism/baseline/contact.jpg`).
4. Stop early when two rounds in a row merged nothing, or when the owner says stop.

If the Workflow tool is not available in this Claude Code, do the same steps by hand with the Agent tool, following
`.claude/workflows/realism-round.js` as the exact procedure (same prompts, same agents, same order): render-runner
for the baseline, realism-critic per aspect in parallel, realism-planner, then per candidate realism-implementer ->
candidate-reviewer -> render-runner -> realism-judge (each lens twice: normal pictures and `_swap` pictures, in
separate calls), then realism-integrator, render-runner, judges, and fleet-scribe. Score exactly as `scoreLens` in
that file: a verdict that flips with the picture order is a tie.

At the end: show `git log --oneline main..realism`, and ask the owner whether to merge `realism` into `main` and
push to `seyedalicheraghi/IsaacSim`. Never push without that yes.
