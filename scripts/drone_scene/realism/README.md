# The realism fleet: Claude Code agents that make the drone scene look real

A team of Claude Code agents that improves the drone scene (`scripts/drone_scene/`) on your PC, with your GPU. Each
round runs these steps:

1. Render probe stills of the accepted state.
2. Critics, one per aspect, say what looks fake and why, in the code.
3. A planner picks a few independent fixes.
4. Builders implement them in parallel, each in its own git worktree.
5. Each fix is rendered, one GPU job at a time.
6. Blind judges compare it with the accepted state, every picture in both screen orders.
7. Only the winners are merged, on the local branch `realism`.

Everything is in the repo, so `git pull` brings it to your PC:

| Path | What |
|---|---|
| `.claude/agents/*.md` | the 10 agents (role, model, tools) |
| `.claude/workflows/realism-round.js` | one round of the loop, as a script |
| `.claude/skills/*/SKILL.md` | the commands `/first-run`, `/probe`, `/record`, `/realism`, `/fleet-status` |
| `.claude/output-styles/drone-owner.md` | how the main session talks to you (status first, exact paths, this PC only) |
| `.claude/settings.json`, `.claude/hooks/guard_bash.py` | the rules, enforced (see below) |
| `scripts/drone_scene/realism/` | the tools the agents use (this folder) |

## Using it

On the PC, in the repo folder (the one with `isaaclab.sh`):

```bash
git pull origin main
claude            # or open the folder in VS Code with the Claude Code extension
```

Accept the "trust this folder" question once; the project settings and hooks only apply after that. Restart Claude
Code after the pull, because output styles are read at start-up. Then type:

| Command | What it does |
|---|---|
| `/first-run` | Checks the PC (GPU, Isaac Lab folder) and runs the unit tests, then a quick probe render. Fixes failures with the debugger agent. Then renders the full baseline, fetches the real NAIP aerial photos and measures the render noise. **Do this first.** |
| `/probe [moments] [cameras]` | Probe stills of the current code, e.g. `/probe cruise_mid final front nadir`. |
| `/record [realism]` | The whole flight as videos (front, back, chase), from the current code or the accepted `realism` branch. |
| `/realism [rounds] [candidates=N] [focus]` | Rounds of the loop, e.g. `/realism 3` or `/realism 2 light`. Merges winners onto the local branch `realism`. At the end it asks before merging into `main` or pushing. |
| `/fleet-status` | Shows what is running now, on which machine, what finished, and where the pictures are. |

Or ask in your own words; the main session picks the right agent.

Where things go, all in `<repo>/.realism/` (git-ignored, on your PC only):

```
.realism/config.env              ISAACLAB_DIR=... (written by /first-run)
.realism/baseline/               the accepted state's probe: probe/<camera>_<moment>.png, contact.jpg, metrics
.realism/rounds/r001/r001-c1/    a candidate's probe, ab/ (blind pairs), metrics_vs_base.json
.realism/rounds/r001/merged/     the round's winners combined
.realism/reference/naip/         real NAIP orthophotos of the nadir squares (public domain)
.realism/reference/footage/      put real photos/footage here (best: from your own LUCID camera)
.realism/worktrees/              git worktrees: base (branch realism) and one per candidate (removed after a round)
.realism/ledger.jsonl            every candidate ever tried: change, votes, decision, why
```

The backlog (`BACKLOG.md`) and lessons (`LESSONS.md`) in this folder are kept up to date by the agents on the
`realism` branch.

## A round, step by step

```
render-runner     baseline probe of branch realism (reused if the code has not changed)
realism-critic x5 lighting | ground | buildings and roads | vegetation | camera and sensor     (parallel, read-only)
realism-planner   up to 3 candidates: one hypothesis each, disjoint files, written success criteria, stills to render
  per candidate, as a pipeline (candidate 1 can be rendering while 2 is still being built):
  realism-implementer   change + unit tests + commit, in .realism/worktrees/<id> on branch rc/<id>
  candidate-reviewer    real defects only (units, colour space, cache versions, Python 3.11); one fix round allowed
  render-runner         run_probe.sh start ... (queued: one render on the GPU at a time) + compare with baseline
  realism-judge x6      3 questions (photo-real? physically plausible? CG artifacts?), each in both screen orders
  score                 a verdict that flips with the order = tie; accept if most questions favour it, the total is
                        positive and no picture shows it broken
realism-integrator merge winners on rnext/<round>, clean cache-version bumps, tests
render-runner     full probe of the merged state; 4 judges compare it with the old baseline; adopted only if not worse
fleet-scribe      ledger, backlog, lessons; remove the round's worktrees
```

## The agents

| Agent | Model | Can | Why it exists |
|---|---|---|---|
| render-runner | haiku | run `run_probe.sh`, read | the only one that renders; renders are queued so the GPU runs one at a time |
| realism-critic | sonnet | read, web | looking at pictures in parallel is the part that gains most from many agents |
| realism-planner | opus | read, web | planning decides how many GPU-hours are wasted, so it gets the strongest model |
| realism-implementer | sonnet | edit, git, tests | one candidate per worktree, so parallel edits never collide |
| candidate-reviewer | sonnet | read, git, tests | cheap check before an expensive render |
| realism-judge | sonnet | read pictures only | blind: never told which side is the change |
| realism-integrator | sonnet | edit, git, tests | the only agent that merges |
| isaac-debugger | opus | edit, run probes | Isaac Sim failures need the strongest model and the installed sources |
| scene-researcher | sonnet | web | real numbers with sources (albedos, species, haze) |
| fleet-scribe | haiku | edit records | bookkeeping |

## Rules that are enforced, not just written down

- **No running on a cloud machine.** `guard_bash.py` blocks Isaac Sim, the drone scripts and their tests whenever
  `CLAUDE_CODE_REMOTE=true` (Claude Code sets this on cloud machines; on your PC it is not set).
- **One render at a time.** The hook refuses `./isaaclab.sh ... quadcopter_pbr.py` from any agent; renders must go
  through `run_probe.sh`, which takes a lock (`/tmp/drone_scene_gpu.lock`, also across separate sessions). It also
  survives the 10-minute command limit.
- **Ask before pushing.** `git push` is an "ask" rule, and the hook also catches `git -C dir push`. The hook refuses
  force pushes, mirror pushes and remote branch deletions outright, whatever the flag order. Accepted changes stay
  on the local branch `realism` until you say merge and push. The hook covers the Bash and Monitor tools.
- **No endless renders.** A render still running after `DRONE_PROBE_TIMEOUT` (2 h) or `DRONE_RECORD_TIMEOUT` (6 h)
  from when it got the GPU is killed, with everything it started, so the queue moves on. Set them in
  `.realism/config.env` if your renders are legitimately slower.
- **Cache safety.** Parallel candidates that change the map or textures give the version constants a suffix
  (`BUILD_VERSION = "6+r001-c2"`), so they never share a cache entry; the integrator turns them into clean bumps.
  Every generated texture carries `TEXTURE_VERSION` in its file name (the crop-row texture did not until now).

## Why it is built this way

These sources were checked against the original pages by a second agent (claims that could not be opened are
left out):

- **Orchestrator plus workers, with read-only exploration done in parallel.**
  - Anthropic's multi-agent research system: an Opus lead with Sonnet subagents beat a single Opus agent by 90.2 % on
    their research evaluation, using about 15 times the tokens of chat. Coding tasks have fewer parallel parts.
    ("How we built our multi-agent research system", Jun 2025.)
  - Anthropic recommends starting with orchestrator-subagent. A verifier needs concrete criteria, or it
    "will rubber-stamp the generator's output". ("Multi-agent coordination patterns", Apr 2026.)
  - Multi-agent setups typically cost 3-10 times the tokens. ("Building multi-agent systems: when and how", Jan 2026.)
  - Here, the critics and judges are parallel and read-only; only builders write, each in its own worktree.
- **A central manager, isolated worktrees, tests, and merges done by the manager.**
  - CAID (Geng and Neubig, arXiv 2603.21489, 2026) reports +26.7 points on PaperBench and +14.3 on Commit0 over a
    single agent with exactly this shape.
  - Two peer agents coordinating by messages succeeded about 50 % less often than one agent doing both
    (CooperBench, arXiv 2601.13295).
  - Agent pull requests conflict often (27.7 % of 142k in AgenticFlict, arXiv 2604.03551).
  - So the candidates touch disjoint files and never talk to each other.
- **Centralised beats independent agents.** Errors were amplified 17.2 times with independent agents against
  4.4 times with a central coordinator. Extra agents help little once a single agent already does well. (Kim et
  al., "Towards a science of scaling agent systems", arXiv 2512.08296.)
- **Cheap steps before the expensive one.**
  - AlphaEvolve and OpenEvolve run cascade evaluation: cheap checks first, the expensive evaluator last, a pass
    threshold per stage.
  - A Darwin Gödel Machine runs its full evaluation only when a candidate beats the archive.
  - Here, the order is review, then unit tests, then a render of only the stills that matter, then the judges.
- **Pairwise judging, both orders, ties allowed.**
  - With the default prompt, GPT-4 gave the same verdict after the two answers swapped places only 65 % of the time
    (Zheng et al., MT-Bench, arXiv 2306.05685).
  - Vision-language judges agree with humans on pair comparisons but not on 1-10 scores (MLLM-as-a-Judge, ICML 2024).
  - Vision-language judges "can rank but cannot score" (arXiv 2604.25235).
  - So: blind A/B pairs, each picture judged in both orders by separate calls, flips count as ties, no absolute
    scores.
- **Do not let the judge be the only measure.**
  - LOKI (ICLR 2025): multimodal models are clearly below humans at telling synthetic from real.
  - So every decision also carries image statistics (`metrics.py`) and, for the nadir view, a comparison with the
    real NAIP orthophoto of the same square. The judges never see either.
- **Measure noise first.** Infrastructure alone moved agent benchmark scores by 6 points (Anthropic, "Quantifying
  infrastructure noise", Feb 2026). `/first-run` therefore renders the same code twice and records how much the
  metrics and judges move.
- **Memory as small updates.**
  - Agentic Context Engineering (ACE, arXiv 2510.04618) evolves a playbook by small merged deltas: +10.6 % on
    AppWorld.
  - OpenEvolve feeds proposers the last attempts labelled improved or regressed.
  - Here, that is the ledger, `LESSONS.md` (short lessons, merged) and `BACKLOG.md`.
- **No agent teams.** Claude Code's agent teams are experimental, need an interactive session, and do not isolate
  teammates in worktrees. Subagents plus a workflow script fit a fixed loop around one GPU better (Claude Code
  docs, agent teams).

## What is not verified yet

- **Nothing here has run yet.** The fleet was written on a cloud machine without running anything. `/first-run` is
  the first real test of the probe mode, the tools and the agents on your PC.
- **Reading pictures.** That a subagent sees a PNG it reads follows from the Claude Code documentation, but no page
  says it in one sentence. `/first-run` step 8 shows it works, or not.
- **Judge reliability.** How well a vision model can judge the realism of these frames is open. The A/A calibration in
  `/first-run` measures how often judges prefer one of two identical renders. If that is high, raise the number of
  lenses or rely more on the metrics.
- **The probe's stills.** The drone holds still at each moment while exposure and anti-aliasing settle, so there is no
  motion blur in probes (there is none in the videos either, unless `--motion_blur`).
- **NAIP fetch.** `reference.py` follows the Planetary Computer recipe (STAC search, SAS token) that the research
  confirmed, but it has not been run.
