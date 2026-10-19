export const meta = {
  name: 'scenario-review',
  description: 'Sonnet review of the ADT-approaches-HEX Isaac scenario: API correctness + fidelity/realism, each finding skeptically verified',
  phases: [
    { title: 'Review', detail: 'Isaac/USD API review and fidelity/realism critique in parallel', model: 'sonnet' },
    { title: 'Verify', detail: 'one skeptic per finding tries to refute it', model: 'sonnet' },
  ],
}

const MODEL = 'sonnet'
const REPO = '/home/user/vla-datagen'
const PY = args.py
const PV = args.preview_dir

const FINDINGS = {
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
          title: { type: 'string' },
          description: { type: 'string' },
          evidence: { type: 'string' },
          fix: { type: 'string' },
        },
        required: ['severity', 'file', 'line', 'title', 'description', 'evidence', 'fix'],
      },
    },
  },
  required: ['findings'],
}
const VERDICT = {
  type: 'object',
  properties: { holds: { type: 'boolean' }, reason: { type: 'string' }, corrected_fix: { type: 'string' } },
  required: ['holds', 'reason', 'corrected_fix'],
}

const CONTEXT = `Repo: ${REPO}. New code to review (do NOT edit any repo file):
- scripts/heavy_machines/scenarios/terrain.py (numpy quarry terrain, meshes, baked textures, rocks)
- scripts/heavy_machines/scenarios/scenario.py (scenario dataclasses, camera transforms, motion profile)
- scripts/heavy_machines/scenarios/adt_approach_hex.py (the scenario: an articulated dump truck drives along a quarry loading lane and stops beside a parked excavator)
- scripts/heavy_machines/scenarios/render_scenario.py (Isaac Sim 5.1 / Isaac Lab 2.3 runner: builds the USD scene with pxr, spawns machines via heavy_machines.machine_env.articulation_cfg, drives them kinematically, records cameras to PNG + mp4)
- scripts/heavy_machines/scenarios/preview.py (offline matplotlib preview) and scripts/heavy_machines/test/test_scenarios.py
Isaac Sim is NOT installed here; the Isaac Lab source is in source/isaaclab/isaaclab/. Python with numpy/matplotlib/torch: ${PY}.`

const reviewers = [
  {
    key: 'isaac-api',
    prompt: `${CONTEXT}

Review render_scenario.py for anything that would crash or misbehave at runtime in Isaac Sim 5.1 / Isaac Lab 2.3. Verify against the Isaac Lab source in this repo (quote the lines you rely on): SimulationContext construction and attributes used (sim.stage, sim.device, step/render/reset semantics, RenderCfg fields incl. rendering_mode and antialiasing_mode values), AppLauncher args (rendering_mode default, device, headless), sim_utils.create_prim types, DomeLightCfg/DistantLightCfg fields and the light orientation convention (does a USD DistantLight shine along its local -Z?), Articulation usage without InteractiveScene for a fixed-base and a floating machine (cfg.init_state rot order w,x,y,z; joint_pos dict keys; write_* calls and shapes; write_data_to_sim; update), Camera usage (CameraCfg with update_period=0, OffsetCfg convention 'world', set_world_poses before/after sim.reset, data.output['rgb'] shape/dtype, update(dt)), pxr usage (UsdGeom.Mesh attributes, Vt.*.FromNumpy availability, normals interpolation, TexCoord2fArray primvar 'st', UsdPreviewSurface + UsdUVTexture + UsdPrimvarReader_float2 network and whether RTX renders it, absolute PNG paths as asset inputs, UsdGeom.Sphere with xform ops), material binding, the machine-material tweak (are URDF-imported materials editable / which shader type does the Isaac Sim 5.1 URDF importer author — check the importer config used by isaaclab.sim.converters.UrdfConverter), and the warmup/recording loop (does the camera image correspond to the state written before sim.step?). Compare with how scripts/forklift/forklift_env.py and scripts/heavy_machines/machine_env.py / showcase.py do the same things. Report concrete findings with file, line, evidence, and an exact fix. Do not report style nits.`,
  },
  {
    key: 'fidelity-realism',
    prompt: `${CONTEXT}

The scenario recreates an AI-generated clip whose prompt was:
"Realistic driving-simulator footage from a forward-facing camera mounted about 4 metres high on the front of a large autonomous articulated haul truck in an open-pit quarry at clear midday. The truck drives slowly and perfectly straight along a loading lane: it passes through a gap between two low earth berms, then a raised earth bench wall runs along its left side and the lane narrows. Ahead, slightly left, a large yellow tracked hydraulic excavator stands parked on the bench at the loading position, arm raised, growing larger as the truck approaches. The truck slows and stops smoothly at the loading spot beside the excavator, at a safe distance. Light sandy ground flowing toward the camera, pale sky. Steady camera, no shake, no people, no logos, no brand markings, no text, no watermarks. Clean simulator look, sharp detail." (1280x720, 30 fps, 10 s)

Render previews yourself: ${PY} scripts/heavy_machines/scenarios/preview.py adt_approach_hex --camera front --times 0 1 2 3 4 5 6 7 8 9 10 --out ${PV}/review_front.png (also --camera chase, and look at the _top.png layout). LOOK at the PNGs with the Read tool. Note the preview is flat-shaded painter's algorithm without a z-buffer (Isaac renders with RTX): judge layout, framing, timing and proportions, not shading quality. Also bake and look at crops of the terrain texture (terrain.bake_albedo at 24 px/m) for the lane, berms and bench face.
Critique (1) fidelity to every element of the clip description — ordering and timing of events (gate gap → bench appears on the left → lane narrows → excavator ahead slightly left, growing → smooth stop beside it at a safe distance), whether 'slightly left' and 'beside' hold, berm/bench heights and widths vs real quarry practice (safety berms ~half the wheel height of the largest vehicle, loading bench heights), camera height/pitch/FOV vs '4 m high forward camera', speed ('slowly'), whether the excavator's arm-raised pose reads clearly; (2) realism improvements that are achievable in this code (terrain/texture detail such as gravel, tyre-tread marks, wheel ruts, windrows at lane edges, bench face texture, rock scatter, distant pit walls, sun/sky settings, camera vibration is NOT wanted). For each finding give a concrete code/parameter change (file, line, exact new values). Prioritise by visual impact.`,
  },
]

phase('Review')
const reviewed = await pipeline(
  reviewers,
  (r) => agent(r.prompt, { label: `review:${r.key}`, phase: 'Review', schema: FINDINGS, model: MODEL }),
  (res, r) => parallel((res ? res.findings : []).map((f) => () =>
    agent(`${CONTEXT}

Skeptically verify this review finding about the scenario code. Default to holds=false unless you can confirm it from the code and (for Isaac APIs) the Isaac Lab source in source/isaaclab, or (for visual claims) by rendering ${PY} scripts/heavy_machines/scenarios/preview.py and looking at the PNG / computing numbers. Do NOT edit repo files.
Finding: ${JSON.stringify(f)}
Return holds, the reason with evidence, and a corrected_fix (the fix as it should actually be applied, or empty if it does not hold).`,
      { label: `verify:${r.key}:${f.line}`, phase: 'Verify', schema: VERDICT, model: MODEL, effort: 'medium' })
      .then((v) => ({ reviewer: r.key, ...f, verdict: v })))),
)
return reviewed.flat().filter(Boolean)
