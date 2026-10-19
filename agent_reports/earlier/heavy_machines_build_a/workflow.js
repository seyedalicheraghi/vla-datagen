export const meta = {
  name: 'heavy-machines-fleet-a',
  description: 'Build, adversarially review and fix half of the new heavy machines; review the Isaac env',
  phases: [
    { title: 'Build', detail: 'one agent per machine writes machines/<name>.py and iterates on renders + checks' },
    { title: 'Review', detail: 'independent skeptical review per machine (fidelity + kinematics)' },
    { title: 'Fix', detail: 'apply verified review findings, re-check' },
    { title: 'Env review', detail: 'Isaac Lab API + control-math review of machine_env.py' },
  ],
}

const PY = args.py
const PREV = args.render_root
const REPO = '/home/user/vla-datagen'

const BUILD_SCHEMA = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['ok', 'failed'] },
    summary: { type: 'string' },
    datasheet: { type: 'string' },
    remaining_warnings: { type: 'array', items: { type: 'string' } },
    core_issues: { type: 'array', items: { type: 'string' } },
  },
  required: ['status', 'summary', 'datasheet', 'remaining_warnings', 'core_issues'],
}
const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['good', 'needs_fixes'] },
    issues: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['must_fix', 'should_fix', 'nit'] },
          description: { type: 'string' },
          evidence: { type: 'string' },
          suggested_fix: { type: 'string' },
        },
        required: ['severity', 'description', 'evidence', 'suggested_fix'],
      },
    },
    core_issues: { type: 'array', items: { type: 'string' } },
  },
  required: ['verdict', 'issues', 'core_issues'],
}
const FIX_SCHEMA = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['ok', 'failed'] },
    applied: { type: 'array', items: { type: 'string' } },
    rejected: { type: 'array', items: { type: 'object', properties: { issue: { type: 'string' }, reason: { type: 'string' } }, required: ['issue', 'reason'] } },
    remaining_warnings: { type: 'array', items: { type: 'string' } },
  },
  required: ['status', 'applied', 'rejected', 'remaining_warnings'],
}
const ENV_SCHEMA = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['bug', 'risk', 'improvement'] },
          file: { type: 'string' },
          line: { type: 'integer' },
          description: { type: 'string' },
          evidence: { type: 'string' },
          fix: { type: 'string' },
        },
        required: ['severity', 'file', 'line', 'description', 'evidence', 'fix'],
      },
    },
  },
  required: ['findings'],
}

const CONTEXT = `You are working in the repo ${REPO} on the \`heavy_machines\` package (scripts/heavy_machines/), which turns pure-Python machine specs into URDFs for Isaac Sim / Isaac Lab and drives them with a generic Isaac Lab environment.
Read these files properly before doing anything:
- scripts/heavy_machines/spec.py — the format, frame/sign conventions (axis (0,-1,0) => +q raises a forward-pointing part), joint roles, helpers (box, cyl, sphere, beam, wheel_link, wheel_joint, track_shapes, sprocket_link).
- scripts/heavy_machines/kinematics.py — forward kinematics, root_height, ground contact.
- scripts/heavy_machines/checks.py — every validation rule (tree, limits, drive geometry, ground contact/clearance, envelope ±20%, mass ±10%, self-overlap, tool direction vocabulary, cameras).
- scripts/heavy_machines/machines/hydraulic_excavator.py — the reference machine: copy its style (module docstring listing the datasheet numbers used, constants at the top, a single build()).
- scripts/heavy_machines/machine_env.py — how roles are driven: tools (action channels, rate = joint.velocity at |cmd|=1, clamped to limits, continuous wraps), wheel / roll_left / roll_right (cosmetic spin), steer (per-wheel Ackermann from the virtual steer angle; steer joints revolute about z, lateral position taken from FK), articulation (root = FRONT frame, the articulation joint's child = rear frame; yaw-rate model needs drive.front_length = hinge->front axle and rear_length = hinge->rear axle/tandem centre), follower (q = k*q_src + c).
Python with matplotlib + torch: ${PY}. Run commands from ${REPO}.`

const BUILD_RULES = (m) => `
YOUR MACHINE: ${m.name} (abbrev ${m.abbrev}, category ${m.category}) — ${m.brief}
Drive: ${m.drive}
Tools: ${m.tools}

Create scripts/heavy_machines/machines/${m.name}.py defining build() -> MachineSpec with name="${m.name}", abbrev="${m.abbrev}", category="${m.category}".
Rules:
- Only create/edit that one file (check_machine.py also writes assets/heavy_machines/${m.name}/ — that is expected). Do NOT edit any other file: other agents are adding other machines in parallel. If a shared file (spec.py, checks.py, kinematics.py, machine_env.py, render.py, ...) has a bug or lacks something you need, work around it and report it in core_issues.
- Realism: model the named real machine class. Put the datasheet numbers you used in the module docstring (overall L/W/H in the init pose, operating mass, wheelbase/gauge, tyre/track sizes, reach/lift/dump numbers, speeds, joint ranges). Realistic proportions; masses split sensibly across links (sum within 10% of operating_mass); realistic joint ranges and rates (Joint.velocity = max joint speed in rad/s or m/s at full command); realistic drive.max_speed (m/s), yaw rate / max steer / articulation.
- Visual quality matters as much as passing checks: the silhouette must be instantly recognisable as that machine in a camera image. Cab with glass panes (visual-only), engine hood, counterweight/grille, tyres with hubs (wheel_link/wheel_joint) or tracks (track_shapes + sprocket_link), visible hydraulic cylinders where cheap, correct left/right placement of cab and booms as on the real machine. Use DEFAULT_MATERIALS; add a livery colour via materials={**DEFAULT_MATERIALS, ...} only if the real class is not yellow. Main masses collide; small details may be visual-only (collision=False).
- Init pose = travel/transport pose; every non-ground-contact part >= 2 cm above the ground; envelope = real overall L, W, H in that pose (checks allow ±20%; aim for ±10%). Wheels must touch the ground.
- turn = +1 must turn LEFT (the env handles signs if the drive fields are right).
- Each tool = one joint with role "tool"; snake_case label; positive/negative words from the auto-verified vocabulary where it fits (raise/lower/up/down/lift, extend/retract/out/in, left/right/ccw/cw); otherwise (curl/dump/tilt/open/close/...) verify the sign visually in the sweep render. Use role "follower" for joints that move in lock-step (second stabiliser leg, nested telescope sections, levelling links).
- Cameras: exactly 3 named front/left/right. front = operator view from inside/at the cab looking forward and down at the work tool; left/right = mounted on the machine looking outward-forward. Camera looks along +x of its frame; pitch > 0 looks down. If the cab slews, mount cameras on the slewing link.
- allowed_overlaps only for genuine design contact, each justified in a comment.

Work loop (repeat until done):
1. ${PY} scripts/heavy_machines/check_machine.py ${m.name} --render ${PREV}/${m.name}
   Fix every ERROR. Treat warnings seriously: fix them or justify them.
2. LOOK at ${PREV}/${m.name}/${m.name}_poses.png and ${m.name}_sweeps.png with the Read tool. Does it look like the real machine in side/top/front/3D? Do tools move the labelled way? Wheels/tracks on the ground? Nothing floating or detached? Iterate on geometry until it looks right.
3. Run ${PY} scripts/heavy_machines/test/test_machines.py and ${PY} scripts/heavy_machines/test/test_env_logic.py. There must be no failures for YOUR machine (subTest output names the machine). Failures in other machines may be other agents' in-progress files: ignore those.
4. Finish with check_machine.py ${m.name} printing OK (this writes the assets).
Return: status, a summary (key dims, mass, drive, tools with ranges), datasheet numbers/assumptions, remaining warnings with justification, core_issues.`

const reviewPrompt = (m, built) => `${CONTEXT}

Another agent just built machine ${m.name} (${m.abbrev}) — ${m.brief}
Drive brief: ${m.drive}
Tools brief: ${m.tools}
Builder's report: ${JSON.stringify(built)}

You are an adversarial reviewer. Your job is to find what is wrong, not to praise. Do NOT edit any file.
1. Read scripts/heavy_machines/machines/${m.name}.py closely.
2. Run ${PY} scripts/heavy_machines/check_machine.py ${m.name} --no-write --render ${PREV}/${m.name}_review and LOOK at both PNGs (Read tool).
3. Check with FK yourself where in doubt (e.g. a small ${PY} -c script using heavy_machines.kinematics: link_transforms, subtree_centroid, lowest_points, bounding_box) — evidence beats impressions.
Lenses:
A) Real-world fidelity vs the real machine class: overall dims, proportions, mass split, joint ranges & rates, travel speed, missing major DOFs/features, wrong placement (cab side, boom side, engine position), anything that makes the silhouette unrecognisable.
B) Kinematics/control: tool sign vs label (especially words checks.py cannot verify: curl/dump/tilt/open/close), joint origins at the real pivot points (bucket rotates about its pin, dump body hinges at its rear, boom foot location), follower formulas, drive fields consistent with geometry (wheelbase, fixed_axle_x, steer_axle, front/rear_length, track_width), steer limits vs inner-wheel angle, articulation range, cameras (front camera at the cab looking at the tool; mounted on the slewing link if the cab slews; not inside solid geometry that would block the view), init pose sensible, allowed_overlaps hiding real problems, collision shapes that would snag the ground/props, wheel_radius consistency.
C) Code: style matches hydraulic_excavator.py; datasheet numbers in the docstring are plausible for that class.
Return verdict and issues (must_fix = wrong behaviour or clearly unrealistic; should_fix = noticeable realism/usability problem; nit = minor). Include concrete evidence (numbers, render observations) and a concrete suggested fix for each. Also report core_issues in shared files if you notice any.`

const fixPrompt = (m, review) => `${CONTEXT}

Machine ${m.name} (${m.abbrev}) — ${m.brief}
A reviewer raised these issues on scripts/heavy_machines/machines/${m.name}.py:
${JSON.stringify(review.issues, null, 1)}

Evaluate each issue critically against the code, FK numbers and renders. Apply the ones that are correct (all must_fix and should_fix that hold up; nits if cheap). If a claim is wrong, do not apply it and explain why. Only edit scripts/heavy_machines/machines/${m.name}.py (other agents edit other machines in parallel).
After editing, repeat the loop: ${PY} scripts/heavy_machines/check_machine.py ${m.name} --render ${PREV}/${m.name}  (must end OK — this also rewrites assets/heavy_machines/${m.name}/), LOOK at the PNGs, run ${PY} scripts/heavy_machines/test/test_machines.py and ${PY} scripts/heavy_machines/test/test_env_logic.py (no failures for this machine).
Return status, applied fixes, rejected issues with reasons, remaining warnings.`

phase('Build')
const machineResults = pipeline(
  args.machines,
  (m) => agent(`${CONTEXT}\n${BUILD_RULES(m)}`, { label: `build:${m.abbrev}`, phase: 'Build', schema: BUILD_SCHEMA }),
  (built, m) => agent(reviewPrompt(m, built), { label: `review:${m.abbrev}`, phase: 'Review', schema: REVIEW_SCHEMA })
    .then((review) => ({ built, review })),
  async (r, m) => {
    if (!r || !r.review) return { machine: m.name, ...r, fix: null }
    const serious = r.review.issues.filter((i) => i.severity !== 'nit')
    if (r.built && r.built.status === 'ok' && serious.length === 0) return { machine: m.name, ...r, fix: null }
    const fix = await agent(fixPrompt(m, r.review), { label: `fix:${m.abbrev}`, phase: 'Fix', schema: FIX_SCHEMA })
    return { machine: m.name, ...r, fix }
  },
)

let envReview = null
if (args.include_env_review) {
  const ENV_FILES = 'scripts/heavy_machines/machine_env.py, scripts/heavy_machines/teleop_machine.py, scripts/heavy_machines/scripted_demo.py, scripts/heavy_machines/showcase.py (plus kinematics.py helpers they use)'
  envReview = parallel([
    () => agent(`${CONTEXT}

Review ${ENV_FILES} for Isaac Lab API and lifecycle correctness against the Isaac Lab source in THIS repo (source/isaaclab/isaaclab/...). Isaac Sim is NOT installed here, so verify by reading the source. Do not edit files.
For every Isaac Lab call and config field used, check the name, signature and semantics in this version, including: DirectRLEnv lifecycle (when _setup_scene runs relative to the subclass __init__ body; attributes set before super().__init__; _reset_idx contract and what super()._reset_idx does; DirectRLEnv.step order: _pre_physics_step, _apply_action x decimation, sim.step, scene.update, _get_dones/_get_rewards, reset of terminated envs, _get_observations; whether observation_space/action_space ints are fine), UrdfFileCfg/UrdfConverterCfg (are URDF joint names preserved; effect of merge_fixed_joints on joint/body names; joint_drive gains required fields; make_instanceable; fix_base), ArticulationCfg/ImplicitActuatorCfg fields (effort_limit_sim, velocity_limit_sim), Articulation writes (write_root_pose_to_sim, write_root_velocity_to_sim, write_joint_state_to_sim, set_joint_position_target with env_ids as tensor vs slice; shapes), Camera set_world_poses with convention="world" and env_ids before/after sensor init, scene.clone_environments with static props spawned under /World/envs/env_0 without rigid bodies, spawn_ground_plane/DomeLightCfg usage, shape cfg fields (ConeCfg/CylinderCfg axis, SphereCfg), carb keyboard enums in teleop, the standalone SimulationContext + Articulation usage in showcase.py (sim.reset before data access, write_data_to_sim, update), AppLauncher argument handling, and anything else that would crash or misbehave at runtime.
Compare with scripts/forklift/forklift_env.py and scripts/forklift/teleop_forklift.py, which are known to run on the user's machine (Isaac Sim 5.1, Isaac Lab 2.3).
Report concrete findings with file, line, evidence (quote the Isaac Lab source you relied on) and the exact fix.`, { label: 'env:isaac-api', phase: 'Env review', schema: ENV_SCHEMA }),
    () => agent(`${CONTEXT}

Review the control and math in scripts/heavy_machines/machine_env.py and the kinematics.py helpers it relies on. Do not edit repo files (you may copy them to ${PREV}/env_math_scratch to experiment).
Check with derivations and small numeric experiments (${PY} has torch): skid-steer model and left/right track speeds; Ackermann (virtual steer angle, per-wheel angles incl. rear-steer, yaw rate sign, root reference-point velocity v*e - x_ref*w*n with x_ref = fixed axle x); articulated model (yaw-rate formula and sign; the env integrates the ROOT (front frame) origin using x_ref = hinge_x + front_length — is that the right reference point and formula?); integration (yaw_mid), angle wrapping, steering/articulation rate limiting and return-to-centre, tool integration/clamping and reported qd, follower handling, cosmetic wheel spin sign for joints with axis +y (does +q roll the wheel forward?), camera pose composition base @ FK(link) @ mount and quaternion order (Isaac Lab uses (w, x, y, z); camera convention "world" = forward +X, up +Z), env_origins handling, reset, observation layout vs spec.state_labels, multi-env correctness (num_envs > 1), and anything else numerically wrong.
The existing offline tests are scripts/heavy_machines/test/test_env_logic.py (they stub Isaac Lab); run them and write extra experiments in your scratch dir.
Report concrete findings with file, line, evidence (numbers) and the exact fix.`, { label: 'env:control-math', phase: 'Env review', schema: ENV_SCHEMA }),
  ])
}

const [machinesDone, envDone] = await Promise.all([machineResults, envReview || Promise.resolve(null)])
return { machines: machinesDone, env_review: envDone }
