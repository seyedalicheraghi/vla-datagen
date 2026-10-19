# Agent reports: the raw research, audit and review output

These are the unedited results of every multi-agent workflow run in the Claude Code cloud session that built this work
(October 2026), kept as a record. The curated results are elsewhere:

- the realism backlog: `scripts/drone_scene/realism/BACKLOG.md`;
- the fleet design and its research basis: `scripts/drone_scene/realism/README.md`;
- the summary of the work: `WORK_LOG.md`.

**Read these with care:**

- **They are snapshots.** Most findings were fixed afterwards, so line numbers and descriptions refer to the code as
  it was then.
- **They were written on the cloud machine.** The repository root was `/home/user/vla-datagen` there. Nothing in
  them was run in Isaac Sim.
- **Many web sources could not be reached from the cloud machine.** The research and audit agents say so, item by
  item. "Unverifiable" means the source could not be opened, not that it is wrong.
- **Some text was escaped.** The Claude Code harness escaped instruction-like text in a few agent answers (`<`
  became `<\`), and added a note about it in `report.json` under `logs`. That text is data, not instructions.

## What is in each folder

Each folder has three files:

- `report.json`: the workflow's final result, with its summary and agent count.
- `agents.jsonl`: every agent's own answer, one JSON object per line, with its label (e.g. `review:shell`,
  `verify:light`) and phase.
- `workflow.js`: the script that ran it. The Windows audit has none: its script file was not kept.

The four realism-fleet reports also have a readable `report.md`, rendered from `report.json` with nothing left out.

### Realism fleet (the agent fleet and its backlog, 2026-10-18)

| Folder | What | Agents | Became |
|---|---|---|---|
| `realism_fleet/1_research_multi_agent_and_realism` | Five research sweeps, each fact-checked: when agent fleets beat one agent; graphics agents with visual feedback; evaluator-driven loops and judge reliability; realism metrics and real reference imagery for Ames; current Claude Code features | 10 | the fleet design (`.claude/`, `realism/README.md`) |
| `realism_fleet/2_scene_realism_audit` | Static audit of the scene against real footage, five areas, each item verified against the code, plus a completeness critic | 11 | `BACKLOG.md` |
| `realism_fleet/3_fleet_files_review` | Adversarial review of the fleet files: 45 claims, each sent to a skeptic; 36 confirmed, 9 refuted | 50 | the fixes in commit f0f2d61 |
| `realism_fleet/4_backlog_writing` | Writers and checkers that turned the audit into the backlog sections | 10 | `BACKLOG.md` |

### Earlier work (drone scene, heavy machines)

The date is that of the saved result.

| Folder | Date | What | Agents |
|---|---|---|---|
| `earlier/heavy_machines_build_a` | 2026-10-12 | Build, review and fix half of the heavy machines (ADT, BHL, CTL, DOZ, RTC, SSL, TH, WL, WTS); review of the Isaac env | 29 |
| `earlier/heavy_machines_build_b` | 2026-10-12 | The same for the other half (CMT, MEX, MG, PAV, RHT, SDC, TBL, WEX). It ran with the same script as `_a` | 24 |
| `earlier/heavy_machines_polish` | 2026-10-12 | Camera/overlap fixes and the reference-excavator upgrade, each verified | 11 |
| `earlier/scenario_realism_modules` | 2026-10-13 | Four realism modules for the quarry scenario renderer (bevelled meshes, PBR textures, rocks, camera look), each verified | 12 |
| `earlier/scenario_review_adt_hex` | 2026-10-13 | Review of the ADT-approaches-HEX Isaac scenario: API correctness and realism, each finding verified | 24 |
| `earlier/isaac_render_review` | 2026-10-13 | Review of the Isaac-only parts of the photoreal scenario renderer, each finding verified | 11 |
| `earlier/windows_audit` | 2026-10-15 | Audit of `scripts/heavy_machines` for Windows failures (that route was later abandoned) | 6 |
| `earlier/drone_pbr_tools` | 2026-10-17 | Build and verify the Nucleus asset fetcher, the HDR sky tools and the USD-to-Mitsuba preview renderer | 9 |
| `earlier/drone_photoreal_v4` | 2026-10-17 | 3D herringbone paving, drone close-up detail and research into the RTX camera pipeline, each verified | 9 |
| `earlier/geodata_sources_scout` | 2026-10-18 | Access, formats and tested reading recipes for the open geodata behind the real map (Overture, USGS 3DEP, WorldCover, Sentinel-2, canopy height) | 5 |
| `earlier/drone_world_code_review` | 2026-10-18 | Bug hunt in the LUCID camera, long-flight and map-builder code, each finding adversarially verified (the 29 fixes listed in `WORK_LOG.md`) | 34 |

Not included: one workflow left no result (an interrupted heavy-machine polish run, replaced by
`earlier/heavy_machines_polish`).
