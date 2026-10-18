---
name: probe
description: Render probe stills of the drone scene from the current code on this PC's GPU (queued) and show where they are, compared with the accepted renders when there are some.
argument-hint: "[moments...] [cameras...]  e.g. cruise_mid final front nadir"
disable-model-invocation: true
---
Render probe stills of the current checkout and report them. Arguments: $ARGUMENTS

1. If `echo $CLAUDE_CODE_REMOTE` prints `true`, stop: renders run only on the owner's PC.
2. Split the arguments into moments (ground liftoff climb cruise_early cruise_mid cruise_late descent final, or a time
   in seconds) and cameras (front back chase nadir). None given: all moments, all four cameras.
3. Pick a new folder `.realism/probes/<n>` (n = 1 + the highest number there). Delegate to the render-runner agent:
   `scripts/drone_scene/realism/run_probe.sh start .realism/probes/<n> -- [--probe <moments>] [--cameras <cameras>]`,
   then wait until done or failed.
4. If it failed: show the important log lines and offer to hand it to the isaac-debugger agent.
5. If done: if `.realism/baseline/probe` exists, `scripts/drone_scene/realism/run_probe.sh compare .realism/baseline
   .realism/probes/<n> 0` and report the metric warnings. Look at the contact sheet yourself and describe in two or
   three sentences what you see.
6. Tell the owner the absolute paths: the contact sheet (`.realism/probes/<n>/contact.jpg`) and the stills folder, and
   the render time from `.realism/probes/<n>/run.status`.
