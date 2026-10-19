export const meta = {
  name: 'heavy-machines-polish-sonnet',
  description: 'Finish camera/overlap fixes and the reference-excavator upgrade on Sonnet, each independently verified',
  phases: [
    { title: 'Polish', detail: 'one agent per machine fixes its listed problems', model: 'sonnet' },
    { title: 'Verify', detail: 'independent reviewer tries to find regressions', model: 'sonnet' },
    { title: 'Refix', detail: 'only if the reviewer found real problems', model: 'sonnet' },
  ],
}

const MODEL = 'sonnet'
const PY = args.py
const PREV = args.render_root
const CONTEXT = `You are working in /home/user/vla-datagen on scripts/heavy_machines (pure-Python construction-machine specs -> URDFs for Isaac Sim, driven by a generic Isaac Lab env). Read scripts/heavy_machines/spec.py, checks.py (all rules, incl. the camera checks at the end: a camera inside a visual shape is an error, a view blocked within 0.3 m is a warning, and every tool must move >= 5 px in at least one 224 px camera), kinematics.py, and the machine file you are assigned. Python with matplotlib+torch: ${PY}. Run commands from /home/user/vla-datagen.
Camera conventions (machine_env.py): exactly three cameras named front/left/right, each looks along +x of (link frame ∘ xyz/rpy), pitch > 0 looks down, 10 mm lens on a 20.955 mm aperture (~92° horizontal FOV), 224x224. 'front' stays the operator's forward view (outside the opaque cab glass). 'left' and 'right' may be re-aimed (including rearward, like real rear-view / work cameras) so that EVERY tool is seen moving by at least one camera; keep them on the machine at realistic mounting points and keep the scene readable (not only sky or only ground).
Tools: ${PY} scripts/heavy_machines/check_machine.py <name> --render ${PREV}/polish2/<name> writes <name>_poses.png, <name>_sweeps.png and <name>_cameras.png (approximate camera images for init / all-tools-lower / all-tools-upper). LOOK at them with the Read tool.
Only edit scripts/heavy_machines/machines/<your machine>.py (check_machine.py also rewrites assets/heavy_machines/<name>/ — expected). Other agents edit other machines in parallel; never touch shared files. Finish with check_machine.py <name> printing OK, then run ${PY} scripts/heavy_machines/test/test_machines.py, ${PY} scripts/heavy_machines/test/test_env_logic.py and ${PY} scripts/heavy_machines/test/test_rosbag.py (no failures for your machine).`

const POLISH_SCHEMA = { type: 'object', properties: { status: { type: 'string', enum: ['ok', 'failed'] }, changes: { type: 'array', items: { type: 'string' } }, remaining_warnings: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' } }, required: ['status', 'changes', 'remaining_warnings', 'notes'] }
const VERIFY_SCHEMA = { type: 'object', properties: { verdict: { type: 'string', enum: ['pass', 'fail'] }, problems: { type: 'array', items: { type: 'object', properties: { severity: { type: 'string', enum: ['must_fix', 'should_fix', 'nit'] }, description: { type: 'string' }, evidence: { type: 'string' }, fix: { type: 'string' } }, required: ['severity', 'description', 'evidence', 'fix'] } } }, required: ['verdict', 'problems'] }

const results = await pipeline(
  args.jobs,
  (job) => job.mode === 'verify_only'
    ? Promise.resolve({ status: 'ok', changes: ['(already applied — see git diff)'], remaining_warnings: [], notes: job.note || '' })
    : agent(`${CONTEXT}\n\nYOUR MACHINE: ${job.name}\nTASK:\n${job.task}\n\nBefore editing, run check_machine.py ${job.name} --no-write and record the current warnings; afterwards there must be no new errors or warnings, and the listed problems must be gone (or explicitly justified if physically impossible). Keep everything else (dimensions, joint ranges, drive, tool signs) as it is unless the task says otherwise. Return status, the concrete changes, remaining warnings with justification, notes.`,
      { label: `polish:${job.name}`, phase: 'Polish', schema: POLISH_SCHEMA, model: MODEL }),
  (pol, job) => agent(`${CONTEXT}\n\nAdversarially verify the change made to scripts/heavy_machines/machines/${job.name}.py. Do NOT edit files.\nThe task was:\n${job.task}\nReport from the agent that made it: ${JSON.stringify(pol)}\nUse git diff -- scripts/heavy_machines/machines/${job.name}.py to see exactly what changed. Run check_machine.py ${job.name} --no-write --render ${PREV}/polish2_verify/${job.name} and LOOK at all three PNGs. Try hard to find: listed problems not actually solved; new overlaps/regressions; cameras that now see only sky/ground or are inside/behind geometry; unrealistic camera mounts; changed dimensions/ranges/signs that were not asked for; anything that makes the machine look less like the real one; leftovers from an interrupted edit (dead constants, stale comments/notes). Verify numbers with small FK scripts (heavy_machines.kinematics) where useful. Verdict pass only if the task is genuinely done without regressions.`,
    { label: `verify:${job.name}`, phase: 'Verify', schema: VERIFY_SCHEMA, model: MODEL }).then((v) => ({ pol, v })),
  async (r, job) => {
    if (!r || !r.v || r.v.verdict === 'pass') return { name: job.name, ...r, refix: null }
    const real = r.v.problems.filter((p) => p.severity !== 'nit')
    if (!real.length) return { name: job.name, ...r, refix: null }
    const refix = await agent(`${CONTEXT}\n\nYOUR MACHINE: ${job.name}. Original task:\n${job.task}\nA reviewer found these problems with the current state:\n${JSON.stringify(real, null, 1)}\nCheck each against the code and renders; fix the ones that hold up, explain the ones that do not. Same rules and finish criteria as before.`,
      { label: `refix:${job.name}`, phase: 'Refix', schema: POLISH_SCHEMA, model: MODEL })
    return { name: job.name, ...r, refix }
  },
)
return results
