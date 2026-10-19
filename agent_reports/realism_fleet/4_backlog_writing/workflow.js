export const meta = {
  name: 'backlog-write',
  description: 'Write the realism BACKLOG.md sections (one per area) from the verified audit items, then check each section against the data and the code',
  phases: [
    { title: 'Write', detail: 'one writer per area', model: 'sonnet' },
    { title: 'Check', detail: 'one checker per section fixes omissions and errors in place', model: 'sonnet' },
  ],
}

const S = '/tmp/claude-0/-home-user-vla-datagen/39bfe37a-c345-5663-b8fb-e785cd7a1b70/scratchpad/backlog'
const RULES = `HARD RULES (from the owner): you are on a cloud machine. Do NOT run any code (no Python, no scripts, no tests). Do not install anything. Read files with Read/Grep/Glob; write ONLY the one output file named below (with Write/Edit). Do not edit anything in the repository.`

const FORMAT = `Context: the repo /home/user/vla-datagen is the owner's Isaac Lab fork. Its drone scene (scripts/drone_scene/) renders, in Isaac Sim 5.1 RTX, an LMF2 quadrotor with two simulated LUCID Triton cameras (front and back, 2048x1536, 4 mm lens, fx = 1159 px) flying 2 km at about 40 m over west Ames, Iowa in mid-July at 10:30 solar time, over a map built from open geodata by scripts/drone_scene/world/. A fleet of Claude Code agents on the owner's PC (see scripts/drone_scene/realism/README.md) improves realism round by round; its planner picks candidates from scripts/drone_scene/realism/BACKLOG.md, which you are helping to write. Nothing has been rendered yet.

The input is a JSON list of audit items. Each item has a "verdict" from a second agent who checked it against the code: verdict.keep (false = drop), verdict.code_claim_accurate, verdict.reference_supported, verdict.correction. THE CORRECTION OVERRIDES THE ITEM wherever they disagree (wrong line numbers, wrong derived numbers, wrong premises, better fixes).

Write markdown in exactly this form (no other headings):

## <AREA TITLE>

### <id>: <short title, plain words>
\`impact <1-5> · effort <S|M|L> · status: open · values: <sourced|partly sourced|unsourced>\`
- **Now:** what the code does, with repo-relative file:line and the quoted values (corrected).
- **Real world:** what real footage or measurements show, with numbers. Give the source URL when the item has one; otherwise write "(unsourced: have scene-researcher confirm before relying on it)".
- **Change:** the concrete change, with the verifier's corrections applied. If the first step is to measure, say so and give the measurement.
- **Check:** the probe stills that show it, by exact name <camera>_<moment> (cameras: front, back, chase, nadir; moments: ground, liftoff, climb, cruise_early, cruise_mid, cruise_late, descent, final) and an acceptance test: metrics.py keys (mean_luma, clip_hi, clip_lo, contrast, colorfulness, saturation, spectral_slope, noise_sigma, sharpness, repetition, haze_contrast, haze_blue, ground_lab), the NAIP comparison of nadir stills (colour_dE, ssim_coarse), and/or the blind judges.
- **Cost:** map rebuild needed (in a candidate: BUILD_VERSION / TEXTURE_VERSION / PREPARE_VERSION get the suffix "+<candidate id>"), render cost, dependencies on other ids, and whether the owner must decide something.

Map the audit's old probe-view names to stills: takeoff_front -> front_ground, front_liftoff (and back_*); climb_front -> front_climb; cruise_front -> front_cruise_mid (also front_cruise_early); cruise_back -> back_cruise_mid; cruise_nadir_chase -> nadir_cruise_mid, chase_cruise_mid; horizon_front -> back_cruise_mid (the back camera looks only about 8 deg down at cruise, so it shows the horizon) and front_cruise_early; landing_front -> front_descent, front_final.

Order inside the section: first items whose first step is a measurement or a check (gates), then the rest by impact (high first), then effort (small first). Keep every kept item; merge two items only if they are the same change (then list both ids in the heading). End the section with "### Dropped" and one line per item whose verdict.keep is false: id, title, and why (from the correction). Be exact and short: 5 bullet lines per item, each at most about 3 lines of text. Plain English, no hype.`

const AREAS = [
  { key: 'light', title: 'Lighting, sky, atmosphere and exposure' },
  { key: 'ground', title: 'Terrain, ground materials and land cover' },
  { key: 'urban', title: 'Buildings, roads and everyday objects' },
  { key: 'vegetation', title: 'Trees, grass and shrubs' },
  { key: 'camera', title: 'Camera, sensor, drone, renderer and probe tools' },
]

phase('Write')
const res = await pipeline(
  AREAS,
  a => agent(`${RULES}\n\n${FORMAT}\n\nYour area: ${a.title} (AREA TITLE = "${a.title}"). Input: ${S}/audit_${a.key}.json (read all of it). When a correction says a cited line is off, open the file and give the right line. Output: write the section to ${S}/section_${a.key}.md. Return a one-line summary: number of items written, merged, dropped.`,
    { label: `write:${a.key}`, phase: 'Write', model: 'sonnet' }),
  (summary, a) => summary === null ? null : agent(`${RULES}\n\n${FORMAT}\n\nCheck the section ${S}/section_${a.key}.md against its input ${S}/audit_${a.key}.json and the code. Fix problems directly in ${S}/section_${a.key}.md (that is the only file you may edit):
1. Every item with verdict.keep true appears (or is merged with a named id); every keep=false item is under Dropped.
2. Each verifier correction is applied; nothing the correction says is wrong survives (wrong numbers, wrong lines, wrong premises).
3. Spot-check at least 8 file:line references against the repo and fix wrong ones.
4. Still names use only the allowed cameras and moments; acceptance tests use only the listed metric keys.
5. The format is exactly as specified; values marked unsourced unless a URL is given.
Return a one-line summary of what you fixed.`,
    { label: `check:${a.key}`, phase: 'Check', model: 'sonnet' }).then(c => ({ area: a.key, writer: summary, checker: c })),
)
return res.filter(Boolean)
