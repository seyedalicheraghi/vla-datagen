export const meta = {
  name: 'realism-round',
  description: 'One round of the drone-scene realism loop on this PC: critique the accepted renders, build the best fixes in parallel worktrees, render each on the GPU (queued), judge blind A/B in both orders, merge the winners',
  whenToUse: 'Started by /realism (one call per round) after /first-run has passed on this PC. args: {"round": "r001", "candidates": 3, "lenses": 3, "focus": ""}',
  phases: [
    { title: 'Baseline', detail: 'the accepted state (branch realism) has a current probe render' },
    { title: 'Critique', detail: 'one critic per aspect looks at the accepted stills' },
    { title: 'Plan', detail: 'independent candidates with written success criteria' },
    { title: 'Build', detail: 'implement and review each candidate in its own worktree' },
    { title: 'Render', detail: 'GPU renders, queued one at a time by run_probe.sh' },
    { title: 'Judge', detail: 'blind A/B votes, every picture in both orders' },
    { title: 'Integrate', detail: 'merge the winners, render, check against the old baseline' },
    { title: 'Record', detail: 'ledger, backlog, lessons, clean up worktrees' },
  ],
}

// ---------------------------------------------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------------------------------------------
const MOMENTS = ['ground', 'liftoff', 'climb', 'cruise_early', 'cruise_mid', 'cruise_late', 'descent', 'final']
const CAMERAS = ['front', 'back', 'chase', 'nadir']
const RUN = 'scripts/drone_scene/realism/run_probe.sh'
const TESTS = 'scripts/drone_scene/realism/run_tests.sh'
const NO_RUN = 'Read files only: do not run anything except the commands named here, and do not edit anything.'
const LONG = '(give this command the Bash tool timeout 600000 ms; the default 2 minutes would cut it off)'

const ASPECTS = [
  { key: 'light', title: 'lighting, sky, atmosphere and exposure',
    files: 'scripts/drone_scene/rtx_look.py, hdr.py, prepare.py, assets.py (WORLD_SKY), scene.py (sun and sky), quadcopter_pbr.py (RenderCfg, scene_cfg), sensor_sim.py (auto exposure)',
    look: 'sun direction against shadows, shadow softness, sky colour and brightness near the horizon, aerial perspective (distant land hazier and bluer), blown or crushed regions, light in shadows, overall exposure' },
  { key: 'ground', title: 'terrain, ground materials and land cover',
    files: 'scripts/drone_scene/world/materials.py, textures.py, landcover.py, terrain.py, build.py (ground_materials, crop_fields, far_terrain, horizon_ring), pavers.py, scene.py',
    look: 'ground colours against the NAIP squares, texture tiling seen from altitude, crop rows, lawn variation, seams between terrain tiles, far terrain colour, the backyard at take-off' },
  { key: 'urban', title: 'buildings, roads and everyday objects',
    files: 'scripts/drone_scene/world/buildings.py, roads.py, textures.py, materials.py, data.py, build.py',
    look: 'roof and wall materials and colours, roof shapes and details, windows, driveways, road wear and markings, parking lots, missing everyday objects (cars, fences, poles, power lines, sheds)' },
  { key: 'vegetation', title: 'trees, grass and shrubs',
    files: 'scripts/drone_scene/world/vegetation.py, assets.py, build.py (trees, landing_grass), scene.py (backyard plants)',
    look: 'tree species and shapes, per-tree variation, density and clumping against the NAIP squares, canopy colour, grass from altitude, hedges, trees on the horizon' },
  { key: 'camera', title: 'camera, sensor, drone and renderer artifacts',
    files: 'scripts/drone_scene/lucid.py, sensor_sim.py, vehicles.py, flight.py, quadcopter_pbr.py, rtx_look.py',
    look: 'noise, sharpness and demosaicing against a real machine-vision camera, aliasing and shimmer, distortion and vignetting, drone parts in view, colour rendition, anything that marks the frames as computer graphics' },
]

const LENSES = [
  { key: 'photo', ask: 'Which side could be a real photograph from a machine-vision camera on a drone? Look at light, colour, texture detail, clutter and scale.' },
  { key: 'physics', ask: 'Which side is more physically plausible? Sun direction against shadows, exposure, how materials respond to light, haze and colour shift with distance, object sizes.' },
  { key: 'artifacts', ask: 'Which side has fewer computer-graphics artifacts? Repeating texture tiles, jagged or shimmering edges, seams, floating or buried objects, flat uniform colours, missing ground clutter, black or broken regions.' },
]

let input = args
if (typeof input === 'string') {
  try { input = JSON.parse(input) } catch (e) { input = { focus: input } }
}
input = input && typeof input === 'object' ? input : {}
const R = /^r\d{3}$/.test(String(input.round || '')) ? String(input.round) : 'r001'
const K = Math.max(1, Math.min(6, Number(input.candidates) || 3))
const NL = Math.max(1, Math.min(LENSES.length, Number(input.lenses) || 3))
const FOCUS = String(input.focus || '').trim()
const SEED = Number(R.slice(1)) || 1

// ---------------------------------------------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------------------------------------------
const str = { type: 'string' }
const strs = { type: 'array', items: str }
const BASE = { type: 'object', required: ['ok', 'fleet', 'commit', 'stills'], properties: {
  ok: { type: 'boolean' }, fleet: str, main_checkout: str, commit: str, rendered: { type: 'boolean' }, stills: strs,
  contact_sheet: str, seconds: { type: 'number' }, naip: { type: 'boolean' }, problems: str } }
const ISSUES = { type: 'object', required: ['issues'], properties: { issues: { type: 'array', items: { type: 'object',
  required: ['title', 'evidence', 'real_world', 'cause', 'fix', 'stills', 'confidence'], properties: {
    title: str, backlog_id: str, evidence: str, real_world: str, cause: str, fix: str, stills: strs,
    impact: { type: 'integer', minimum: 1, maximum: 5 }, confidence: { type: 'integer', minimum: 1, maximum: 3 } } } } } }
const PLAN = { type: 'object', required: ['candidates'], properties: {
  candidates: { type: 'array', items: { type: 'object',
    required: ['title', 'spec', 'files', 'success', 'probe_moments', 'probe_cameras'], properties: {
      title: str, backlog_id: str, spec: str, files: strs, success: str, probe_moments: strs, probe_cameras: strs,
      map_rebuild: { type: 'boolean' }, risk: str } } },
  skipped: { type: 'array', items: { type: 'object', properties: { title: str, reason: str } } } } }
const IMPL = { type: 'object', required: ['ok', 'commit', 'summary', 'tests_passed'], properties: {
  ok: { type: 'boolean' }, branch: str, worktree: str, commit: str, files_changed: strs, summary: str,
  tests_passed: { type: 'boolean' }, test_tail: str, notes: str } }
const REVIEW = { type: 'object', required: ['approve', 'blocking'], properties: {
  approve: { type: 'boolean' }, blocking: strs, nonblocking: strs } }
const SIDES = { type: 'object', required: ['1', '2'], properties: {
  1: { type: 'string', enum: ['A', 'B'] }, 2: { type: 'string', enum: ['A', 'B'] } } }
const RENDER = { type: 'object', required: ['ok'], properties: {
  ok: { type: 'boolean' }, seconds: { type: 'number' }, pictures: strs,
  key: { type: 'object', additionalProperties: SIDES }, warnings: { type: 'object', additionalProperties: strs },
  naip: { type: 'object', additionalProperties: { type: 'object' } }, failure_log: str } }
const VOTES = { type: 'object', required: ['votes'], properties: { votes: { type: 'array', items: { type: 'object',
  required: ['picture', 'choice', 'confidence', 'broken', 'reason'], properties: {
    picture: str, choice: { type: 'string', enum: ['1', '2', 'tie'] },
    confidence: { type: 'integer', minimum: 1, maximum: 3 }, broken: { type: 'string', enum: ['none', '1', '2', 'both'] },
    reason: str } } } } }
const INTEG = { type: 'object', required: ['ok', 'merged'], properties: {
  ok: { type: 'boolean' }, branch: str, worktree: str, commit: str, merged: strs, conflicted: strs,
  tests_passed: { type: 'boolean' }, notes: str } }
const DONE = { type: 'object', required: ['done'], properties: { done: { type: 'boolean' }, notes: str } }

// ---------------------------------------------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------------------------------------------
phase('Baseline')
const base = await agent(`Task: make sure the accepted state of the drone scene has a current probe render. Your working directory is the main checkout of the repo.
1. MAIN=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)"); FLEET=$MAIN/.realism; mkdir -p "$FLEET". Report both as absolute paths (fleet, main_checkout).
2. If the branch realism does not exist: git branch realism HEAD. If "$FLEET/worktrees/base" is not a worktree yet: git worktree add "$FLEET/worktrees/base" realism. If git -C "$FLEET/worktrees/base" status --porcelain shows changes: stop with ok=false and problems="uncommitted changes in $FLEET/worktrees/base".
3. commit = git -C "$FLEET/worktrees/base" rev-parse --short HEAD.
4. Reuse the accepted renders (rendered=false) when "$FLEET/baseline/probe/probe.json" exists and the code is unchanged since they were made: B=$(cat "$FLEET/baseline/code_commit") and git -C "$FLEET/worktrees/base" diff --quiet "$B" HEAD -- scripts/drone_scene ':(exclude)scripts/drone_scene/realism/*.md' exits 0.
5. Otherwise render them (rendered=true): ${RUN} start "$FLEET/baseline.new" "$FLEET/worktrees/base"; then run ${RUN} wait "$FLEET/baseline.new" 540 ${LONG} again and again, one call per command, until it exits 0 (done) or 1 (failed); exit 3 means still queued or running. On done: rm -rf "$FLEET/baseline" && mv "$FLEET/baseline.new" "$FLEET/baseline". On failed: ok=false and problems = the last 60 lines it printed.
6. If "$FLEET/reference/naip" has no nadir_*.png yet: ${RUN} naip "$FLEET/baseline" (if that fails, mention it in problems but keep ok=true).
Report ok, fleet, main_checkout, commit, rendered, stills (the PNG file names in $FLEET/baseline/probe), contact_sheet ($FLEET/baseline/contact.jpg), seconds (from $FLEET/baseline/run.status, 0 if reused), naip (true if $FLEET/reference/naip has nadir_*.png files), problems.`,
  { label: 'baseline', phase: 'Baseline', agentType: 'render-runner', schema: BASE })
if (!base || !base.ok) {
  return { round: R, status: 'baseline failed', problems: base ? base.problems : 'the baseline agent did not return' }
}
const F = base.fleet
const BW = `${F}/worktrees/base`
log(`baseline ${base.commit}: ${base.stills.length} stills${base.rendered ? `, rendered in ${Math.round(base.seconds || 0)} s` : ' (reused)'}`)

const CONTEXT = `Fleet folder: ${F}.
Accepted renders (branch realism, commit ${base.commit}): ${F}/baseline/probe/<camera>_<moment>.png. Cameras: front and back = the two LUCID cameras on the drone (20 deg down, the drone leans about 12 deg forward at cruise), chase = third person 10 m behind and 3.5 m above, nadir = straight down from 300 m above the take-off ground, north up, 60 deg (346 m square). Moments in flight order: ${MOMENTS.join(', ')}.
Contact sheet: ${base.contact_sheet}. Image statistics: ${F}/baseline/probe/metrics.json.${base.naip ? ` Real NAIP orthophotos of the nadir squares: ${F}/reference/naip/ (same file names as the nadir stills); comparison: ${F}/baseline/metrics_vs_naip.json.` : ''}
Real photos the owner added (may be empty): ${F}/reference/footage/.
Code of the accepted state: ${BW}. Backlog: ${BW}/scripts/drone_scene/realism/BACKLOG.md. Lessons: ${BW}/scripts/drone_scene/realism/LESSONS.md. Past attempts: ${F}/ledger.jsonl (one JSON object per line; may be missing).`

phase('Critique')
const wanted = ASPECTS.filter(a => FOCUS.toLowerCase().includes(a.key))
const aspects = wanted.length ? wanted : ASPECTS
const critiques = await parallel(aspects.map(a => () => agent(`You are the critic for ${a.title}.
${CONTEXT}
Look at the contact sheet first, then open at full size the stills that show your area (at most 10 pictures). Compare them with how real footage from a drone at 40 m over a west-Ames, Iowa suburb and farmland looks on a clear July morning, and with the NAIP squares and the owner's photos when there are any.
Relevant code: ${a.files}. Look at: ${a.look}.${FOCUS ? `\nThe owner asked to focus on: ${FOCUS}.` : ''}
Return up to 6 issues, most visible first. For each: evidence (still file names and where in the frame), what real footage shows instead, the cause in the code (open the code and name file and function), a concrete fix (values with a source where you can), the stills that would show the fix, impact 1-5, confidence 1-3, and the backlog id if the backlog already has it. Skip anything the ledger shows was rejected before, unless your fix is a different approach (then say how it differs).
${NO_RUN}`,
  { label: `critic:${a.key}`, phase: 'Critique', agentType: 'realism-critic', schema: ISSUES })))
const issues = critiques.filter(Boolean).flatMap((c, i) => c.issues.map(x => ({ aspect: aspects[i].key, ...x })))
log(`${issues.length} issues from ${critiques.filter(Boolean).length} critics`)
if (!issues.length) {
  return { round: R, status: 'no issues found', baseline_commit: base.commit }
}

phase('Plan')
const plan = await agent(`Plan round ${R} of the drone-scene realism loop: choose at most ${K} candidate changes to build and test in parallel.
${CONTEXT}
Issues from the critics (JSON): ${JSON.stringify(issues)}
Rules:
- Each candidate is ONE coherent change (one hypothesis) that rendering can confirm or refute. No two candidates may change the same file; list every file each one changes.
- Priority: if exposure, sun or sky are wrong, fix them first (they distort every other judgment); then large areas (ground, roofs, trees); then details.
- Prefer high impact x confidence for little effort. Use the backlog and lessons. Do not repeat a rejected attempt from the ledger unless the approach differs.
- spec: precise instructions for a programmer who has not seen the pictures: files, functions, the exact change, values with sources, and what not to touch.
- success: what should look different, in which stills, so a judge can tell.
- probe_moments and probe_cameras: only what is needed to see the change, plus at least two control stills elsewhere to catch regressions. Moments: ${MOMENTS.join(', ')}. Cameras: ${CAMERAS.join(', ')}. Fewer stills render faster.
- map_rebuild: true if the change alters the map built by scripts/drone_scene/world/ (that adds about 4 minutes to its render).
List the issues you did not choose under skipped with a short reason.
${NO_RUN}`,
  { label: 'plan', phase: 'Plan', agentType: 'realism-planner', schema: PLAN })
if (!plan || !plan.candidates.length) {
  return { round: R, status: 'no plan', baseline_commit: base.commit, issues }
}
const chosen = plan.candidates.slice(0, K)
if (plan.candidates.length > K) log(`planner proposed ${plan.candidates.length}; building the first ${K}`)

function idOf(i) { return `${R}-c${i + 1}` }
function listOk(xs, allowed, fallback) {
  const ok = (xs || []).filter(x => allowed.includes(x))
  return ok.length ? Array.from(new Set(ok)) : fallback
}

function implementPrompt(c, id, wt) {
  return `Implement ONE candidate change for the drone-scene realism loop, in its own git worktree.
Candidate ${id}: ${c.title}
Spec: ${c.spec}
Files planned: ${(c.files || []).join(', ')} (change others only if the spec cannot work without them, and say why).
What should look different once rendered (you cannot render): ${c.success}
Steps:
1. From the main checkout (your working directory): if ${wt} already exists from an earlier attempt at this same candidate, keep using it; otherwise git worktree add -b rc/${id} ${wt} realism
2. Change files under ${wt} only (absolute paths). Keep the surrounding style. Python 3.11: no f-string that reuses its own quote type inside. numpy-only modules stay numpy-only.
3. Shared caches: if the change alters the map (output of scripts/drone_scene/world/), set BUILD_VERSION in world/build.py to "<old value>+${id}"; if it alters generated textures, also set TEXTURE_VERSION in world/materials.py to "<old value>+${id}"; if it alters what prepare.py produces, set PREPARE_VERSION to "<old value>+${id}". Never just add 1: parallel candidates would collide in ~/.cache/drone_scene.
4. Add or update unit tests for logic you add. Run ${TESTS} ${wt} from the main checkout ${LONG}; every suite must pass.
5. Commit: git -C ${wt} add -A && git -C ${wt} commit -m "${id}: <short title>"
Do not run Isaac Sim or render (the render runner does), do not push, do not touch the main checkout or other worktrees.
Report ok, branch (rc/${id}), worktree, commit, files_changed, summary (what changed, with values and why), tests_passed, test_tail (the failing output if any), notes.`
}

function fixPrompt(c, id, wt, problem) {
  return `Candidate ${id} (${c.title}) in worktree ${wt}, branch rc/${id}, needs a fix.
${problem}
Spec it implements: ${c.spec}
Fix it under ${wt}, run ${TESTS} ${wt} from the main checkout ${LONG} (all suites must pass), and commit: git -C ${wt} add -A && git -C ${wt} commit -m "${id}: fix". Same rules: Python 3.11, cache-version suffixes "+${id}", no rendering, no push.
Report ok, branch, worktree, commit, files_changed, summary, tests_passed, test_tail, notes.`
}

function reviewPrompt(c, id, wt) {
  return `Review candidate ${id} before it is rendered (a render costs minutes on the only GPU).
Diff: git -C ${wt} diff realism...rc/${id}
Spec: ${c.spec}
Files planned: ${(c.files || []).join(', ')}
Check: (1) it does what the spec says and nothing unrelated; (2) real bugs: units, sRGB vs linear colour, axes, crashes, Python 3.11 syntax; (3) cache versions: BUILD_VERSION / TEXTURE_VERSION / PREPARE_VERSION end in "+${id}" when that output changes; (4) the tests pass (${TESTS} ${wt}, from the main checkout, if you need to check); (5) it will not make the render much slower (for example millions of new prims).
Set approve=false only for defects that would waste or invalidate the render; list each as "file:line - problem - fix". Put style remarks under nonblocking.
Do not edit anything.`
}

function renderPrompt(id, wt, out, extra, seed) {
  return `Render ${id} and compare it with the accepted renders.
1. ${RUN} start ${out} ${wt}${extra ? ` -- ${extra}` : ''}
   (if it answers "already queued or running", go on with step 2)
2. Run ${RUN} wait ${out} 540 ${LONG} again and again, one call per command, until it exits 0 (done) or 1 (failed). Exit 3 means it is still queued behind other renders or still running. Start nothing else on the GPU.
3. If it failed: ok=false, failure_log = the last 80 lines of ${out}/drone_scene.log (or of ${out}/run.log if that is missing), and stop.
4. If done: ${RUN} compare ${F}/baseline ${out} ${seed}
5. Read ${out}/ab/key.json and ${out}/metrics_vs_base.json.
Report: ok; seconds (from ${out}/run.status); pictures = the file names in ${out}/ab/ that end in .jpg, without .jpg; key = the "pairs" object of key.json exactly as it is; warnings = for each still with a non-empty "warnings" list under "B_minus_A" in metrics_vs_base.json, that list; naip = if metrics_vs_base.json has "pairs", for each still {"A_dE", "A_ssim", "B_dE", "B_ssim"} from pairs.A and pairs.B; failure_log.`
}

function judgePrompt(lens, out, pictures) {
  return `${lens.ask}
Each picture shows two versions of the same frame from the drone scene side by side, labelled 1 (left) and 2 (right). Open every picture with Read. For each picture: choice "1", "2", or "tie" when you cannot see a real difference on this question; confidence 1 (slight) to 3 (clear); broken = which side, if any, shows a rendering failure (black or missing regions, garbage pixels, broken geometry): "none", "1", "2" or "both"; and one sentence why. Judge each picture on its own merits. Do not open any other file (in particular no JSON) and do not try to find out which side is which version.
Pictures (give "picture" as the name without folder and .jpg):
${pictures.map(p => `${out}/ab/${p}.jpg`).join('\n')}`
}

// Votes of one lens in both orders -> net score for B (positive: the candidate is better).  A verdict that flips
// with the order counts as a tie.
function scoreLens(normal, swapped, key) {
  const byPic = v => Object.fromEntries((v ? v.votes : []).map(x => [x.picture.replace(/\.jpg$/, ''), x]))
  const vn = byPic(normal), vs = byPic(swapped)
  let net = 0, consistent = 0, flipped = 0, brokenB = false
  const reasons = []
  for (const pic of Object.keys(key)) {
    if (pic.endsWith('_swap')) continue
    const a = vn[pic], b = vs[`${pic}_swap`]
    for (const [v, k] of [[a, key[pic]], [b, key[`${pic}_swap`]]]) {
      if (!v || !k || v.broken === 'none') continue
      const sides = v.broken === 'both' ? ['1', '2'] : [v.broken]
      const hits = sides.map(s => k[s])
      if (hits.includes('B') && !hits.includes('A')) brokenB = true
    }
    if (!a || !b) continue
    const sa = a.choice === 'tie' ? null : key[pic][a.choice]
    const sb = b.choice === 'tie' ? null : key[`${pic}_swap`][b.choice]
    if (sa && sa === sb) {
      const w = Math.min(a.confidence, b.confidence)
      net += sa === 'B' ? w : -w
      consistent += 1
      reasons.push(`${pic}: ${sa === 'B' ? 'candidate' : 'baseline'} better - ${a.reason}`)
    } else if (sa || sb) {
      flipped += 1
    }
  }
  return { net, consistent, flipped, brokenB, reasons }
}

async function judgeAll(id, out, render, lenses, onlyFull) {
  const pics = (render.pictures || []).filter(p => render.key && render.key[p] && (!onlyFull || !p.includes('_detail')))
  const normal = pics.filter(p => !p.endsWith('_swap'))
  const swapped = pics.filter(p => p.endsWith('_swap'))
  const runs = await parallel(lenses.flatMap(l => [normal, swapped].map((set, j) => () =>
    agent(judgePrompt(l, out, set), { label: `judge:${id}:${l.key}${j ? ':swap' : ''}`, phase: 'Judge',
      agentType: 'realism-judge', schema: VOTES }))))
  const perLens = lenses.map((l, i) => ({ lens: l.key, ...scoreLens(runs[2 * i], runs[2 * i + 1], render.key),
    complete: Boolean(runs[2 * i] && runs[2 * i + 1]) }))
  const done = perLens.filter(p => p.complete)
  const total = done.reduce((s, p) => s + p.net, 0)
  const forB = done.filter(p => p.net > 0).length
  const brokenB = done.some(p => p.brokenB)
  return { total, lenses_for_candidate: forB, lenses_judged: done.length, broken_candidate: brokenB,
    flipped: done.reduce((s, p) => s + p.flipped, 0), per_lens: perLens }
}

async function build(c, i) {
  const id = idOf(i), wt = `${F}/worktrees/${id}`
  let impl = await agent(implementPrompt(c, id, wt), { label: `build:${id}`, phase: 'Build', agentType: 'realism-implementer', schema: IMPL })
  if (!impl || !impl.ok || !impl.tests_passed) {
    return { id, c, wt, done: true, decision: 'not built', why: impl ? (impl.test_tail || impl.notes || 'tests failed') : 'the implementer did not return' }
  }
  let review = await agent(reviewPrompt(c, id, wt), { label: `review:${id}`, phase: 'Build', agentType: 'candidate-reviewer', schema: REVIEW })
  if (review && !review.approve) {
    impl = await agent(fixPrompt(c, id, wt, `The reviewer found these defects:\n- ${review.blocking.join('\n- ')}`),
      { label: `fix:${id}`, phase: 'Build', agentType: 'realism-implementer', schema: IMPL })
    if (!impl || !impl.ok || !impl.tests_passed) return { id, c, wt, done: true, decision: 'review failed', why: review.blocking.join('; ') }
    review = await agent(reviewPrompt(c, id, wt), { label: `review2:${id}`, phase: 'Build', agentType: 'candidate-reviewer', schema: REVIEW })
    if (review && !review.approve) return { id, c, wt, done: true, decision: 'review failed', why: review.blocking.join('; ') }
  }
  return { id, c, wt, impl, review }
}

async function render(b) {
  if (b.done) return b
  const out = `${F}/rounds/${R}/${b.id}`
  const extra = `--probe ${listOk(b.c.probe_moments, MOMENTS, ['climb', 'cruise_mid', 'final']).join(' ')} --cameras ${listOk(b.c.probe_cameras, CAMERAS, ['front', 'back', 'nadir']).join(' ')}`
  let r = await agent(renderPrompt(b.id, b.wt, out, extra, SEED), { label: `render:${b.id}`, phase: 'Render', agentType: 'render-runner', schema: RENDER })
  if (r && !r.ok && r.failure_log) {
    const fix = await agent(fixPrompt(b.c, b.id, b.wt, `Its render failed. The end of the log:\n${r.failure_log}`),
      { label: `fix-render:${b.id}`, phase: 'Build', agentType: 'realism-implementer', schema: IMPL })
    if (fix && fix.ok && fix.tests_passed) {
      r = await agent(renderPrompt(b.id, b.wt, out, extra, SEED), { label: `rerender:${b.id}`, phase: 'Render', agentType: 'render-runner', schema: RENDER })
    }
  }
  if (!r || !r.ok || !r.key || !Object.keys(r.key).length) {
    return { ...b, out, done: true, decision: 'render failed', why: r ? String(r.failure_log || 'no pictures').slice(-2000) : 'the render runner did not return' }
  }
  return { ...b, out, render: r }
}

async function judge(r) {
  if (r.done) return r
  const s = await judgeAll(r.id, r.out, r.render, LENSES.slice(0, NL), false)
  const accepted = !s.broken_candidate && s.total > 0 && s.lenses_judged > 0 && 2 * s.lenses_for_candidate > s.lenses_judged
  log(`${r.id} ${accepted ? 'ACCEPTED' : 'rejected'}: score ${s.total}, ${s.lenses_for_candidate}/${s.lenses_judged} lenses for it, ${s.flipped} order flips`)
  return { ...r, score: s, decision: accepted ? 'accepted' : 'rejected' }
}

phase('Build')
const results = (await pipeline(chosen, (c, _, i) => build(c, i), render, judge)).filter(Boolean)

// ---------------------------------------------------------------------------------------------------------------
// Integrate: merge the winners on a new branch, render it fully, keep it only if it is not worse than the baseline
// ---------------------------------------------------------------------------------------------------------------
phase('Integrate')
const winners = results.filter(r => r.decision === 'accepted').sort((a, b) => b.score.total - a.score.total)
let integration = null, sanity = null, adopted = false
if (winners.length) {
  const nwt = `${F}/worktrees/${R}-next`
  integration = await agent(`Merge the winning candidates of round ${R} into a new branch, for a combined render.
From the main checkout (your working directory): git worktree add -b rnext/${R} ${nwt} realism  (if ${nwt} already exists from an earlier attempt of this round, remove it with git worktree remove --force ${nwt} and git branch -D rnext/${R} first).
Then in ${nwt}, merge in this order, one at a time: ${winners.map(w => `rc/${w.id}`).join(', ')}  (git -C ${nwt} merge --no-ff --no-edit rc/<id>).
- A conflict only in the cache-version lines (BUILD_VERSION, TEXTURE_VERSION, PREPARE_VERSION) is expected: resolve it as described below. Any other conflict: git -C ${nwt} merge --abort, skip that candidate and list it under conflicted.
- After merging: every cache version that carries a "+${R}-c<k>" suffix becomes a clean bump of the value on realism: "5" -> "6", "v2" -> "v3", "3" -> "4" (one bump per constant per round). Commit that as "${R}: cache versions".
- Run ${TESTS} ${nwt} from the main checkout ${LONG}. If it fails, find which merge broke it, revert that merge (git -C ${nwt} revert -m 1 --no-edit <merge commit>), list it under conflicted, and test again.
Do not render and do not push.
Report ok, branch (rnext/${R}), worktree, commit (short), merged (the candidate ids merged), conflicted, tests_passed, notes.`,
    { label: 'integrate', phase: 'Integrate', agentType: 'realism-integrator', schema: INTEG })
  if (integration && integration.ok && integration.merged.length && integration.tests_passed) {
    const mout = `${F}/rounds/${R}/merged`
    const mr = await agent(renderPrompt('the merged winners', nwt, mout, '', SEED + 500), { label: 'render:merged', phase: 'Integrate', agentType: 'render-runner', schema: RENDER })
    if (mr && mr.ok && mr.key && Object.keys(mr.key).length) {
      sanity = await judgeAll('merged', mout, mr, [LENSES[0], LENSES[2]], true)
      adopted = !sanity.broken_candidate && sanity.total >= 0
      log(`merged ${integration.merged.join(', ')}: sanity score ${sanity.total} -> ${adopted ? 'adopted' : 'NOT adopted'}`)
      if (adopted) {
        const fin = await agent(`Adopt the merged state of round ${R} as the new accepted state.
1. git -C ${BW} merge --ff-only rnext/${R}   (${BW} has branch realism checked out). If that is not a fast-forward, stop with done=false.
2. rm -rf ${F}/baseline.old; mv ${F}/baseline ${F}/baseline.old; cp -a ${mout} ${F}/baseline; then rm -rf ${F}/baseline.old
3. echo "$(git -C ${BW} rev-parse --short HEAD)" > ${F}/baseline/code_commit
Report done and notes. Do not push.`, { label: 'adopt', phase: 'Integrate', agentType: 'realism-integrator', schema: DONE })
        adopted = Boolean(fin && fin.done)
      }
    } else {
      log('the merged render failed; nothing adopted')
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Record
// ---------------------------------------------------------------------------------------------------------------
phase('Record')
const records = results.map(r => ({
  round: R, id: r.id, title: r.c.title, backlog_id: r.c.backlog_id || null, branch: `rc/${r.id}`,
  commit: r.impl ? r.impl.commit : null, files: r.impl ? r.impl.files_changed : r.c.files, decision: r.decision,
  merged: Boolean(adopted && integration && integration.merged.includes(r.id)),
  score: r.score ? { total: r.score.total, lenses_for: r.score.lenses_for_candidate, lenses: r.score.lenses_judged, flipped: r.score.flipped } : null,
  why: r.why || (r.score ? r.score.per_lens.flatMap(p => p.reasons).slice(0, 6).join(' | ') : ''),
  metric_warnings: r.render ? r.render.warnings || {} : {}, naip: r.render ? r.render.naip || {} : {},
  render_seconds: r.render ? r.render.seconds || null : null,
}))
const newIssues = issues.filter(x => !x.backlog_id).map(x => `[${x.aspect}] ${x.title}: ${x.fix}`)
await agent(`Record round ${R} of the drone-scene realism loop.
1. Append these objects to ${F}/ledger.jsonl, one JSON object per line, exactly as given: ${JSON.stringify(records)}
2. In ${BW} (branch realism) update scripts/drone_scene/realism/BACKLOG.md: accepted and merged items -> done (round ${R}); rejected or failed ones -> one line with the reason and the round; append new issues that are not in the backlog yet under "New from critics": ${JSON.stringify(newIssues)}
3. Update scripts/drone_scene/realism/LESSONS.md: at most 5 short, general lessons from this round (from the reasons and failures above), each tagged ${R}; merge with existing lessons rather than repeating them; keep the file under 150 lines.
4. git -C ${BW} add scripts/drone_scene/realism/BACKLOG.md scripts/drone_scene/realism/LESSONS.md && git -C ${BW} commit -m "realism ${R}: notes"
5. Remove this round's worktrees, keeping their branches: git worktree remove --force for each of ${results.map(r => `${F}/worktrees/${r.id}`).join(', ')}${winners.length ? `, ${F}/worktrees/${R}-next` : ''} (skip ones that do not exist).
Do not push. Report done and notes.`, { label: 'record', phase: 'Record', agentType: 'fleet-scribe', schema: DONE })

return {
  round: R,
  baseline_commit: base.commit,
  candidates: records.map(r => ({ id: r.id, title: r.title, decision: r.decision, merged: r.merged, score: r.score, why: r.why.slice(0, 400) })),
  merged: adopted ? integration.merged : [],
  merged_commit: adopted && integration ? integration.commit : null,
  sanity: sanity ? { total: sanity.total, flipped: sanity.flipped } : null,
  skipped: plan.skipped || [],
  new_baseline_contact_sheet: adopted ? `${F}/baseline/contact.jpg` : `${F}/baseline/contact.jpg (unchanged)`,
  ledger: `${F}/ledger.jsonl`,
}
