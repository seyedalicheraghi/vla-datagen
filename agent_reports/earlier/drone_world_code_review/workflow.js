export const meta = {
  name: 'review-drone-world',
  description: 'Find bugs in the LUCID camera, long-flight and real-map builder code; adversarially verify each finding',
  phases: [{ title: 'Find', detail: 'one reviewer per area' }, { title: 'Verify', detail: 'one skeptic per finding' }],
}

const REPO = '/home/user/vla-datagen'
const PY = '/tmp/claude-0/-home-user-vla-datagen/39bfe37a-c345-5663-b8fb-e785cd7a1b70/scratchpad/venv/bin/python'
const WORLD = '/tmp/claude-0/-home-user-vla-datagen/39bfe37a-c345-5663-b8fb-e785cd7a1b70/scratchpad/worlds/world_5371363afc'

const CONTEXT = `Project: ${REPO} (branch claude/review-last-request-I6mXJ). New code makes an Isaac Lab 2.3 / Isaac Sim 5.1 (Python 3.11) drone demo:
- scripts/drone_scene/quadcopter_pbr.py: the Isaac script (run on the user's Ubuntu PC with an RTX GPU). Flags --route garden|straight, --vehicle lmf2|crazyflie, LUCID cameras front/back, chase camera, recording via ffmpeg pipe.
- scripts/drone_scene/flight.py: Flight (garden spline) and StraightFlight/SmoothMove (take-off, 2 km cruise with drag tilt, landing).
- scripts/drone_scene/lucid.py + sensor_sim.py: LUCID camera optics (OpenCV distortion, render overscan plan) and a torch sensor/ISP simulation fed by Replicator HdrColor.
- scripts/drone_scene/vehicles.py: LMF2 drone materials, camera mounts.
- scripts/drone_scene/world/: builds a large real-world USD map before Isaac starts: geo.py (local tmerc frame, Grid raster convention), data.py (adapters to sources/*), sources/ (overture, elevation, landcover (WorldCover + Sentinel-2), canopy recipes), landcover.py, terrain.py (LOD tiles, skirts, pads, UVs), buildings.py (walls + gable/hip/flat roofs), roads.py (ribbons), vegetation.py (trees from canopy height), site.py (choose backyard take-off and grass landing), materials.py, textures.py (procedural textures), usdio.py (USD meshes with GeomSubsets), build.py (orchestrator + manifest world.json), runner.py (installs deps with pip --target and runs the builder as a subprocess).
- Isaac Lab source for API checks is in ${REPO}/source/isaaclab (this repo is an Isaac Lab fork; the user runs Isaac Lab 2.3.2 + Isaac Sim 5.1).
- A real map was already built with this code: ${WORLD}/world.usdc and world.json (Ames, Iowa). Tests: python scripts/drone_scene/test/test_*.py.
Tools: Python with all deps (numpy, torch CPU, pxr/usd-core, rasterio, pyproj, shapely, scipy, Pillow, matplotlib): ${PY}. Isaac Sim is NOT available on this machine and nothing may be rendered here; do not run quadcopter_pbr.py. You may write small scratch scripts under /tmp/claude-0/review/ to check a claim (e.g. call a function with crafted inputs, inspect the built USD with pxr). Do NOT modify files under ${REPO}.`

const FINDINGS = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          file: { type: 'string', description: 'repo-relative path' },
          line: { type: 'integer' },
          severity: { type: 'string', enum: ['crash', 'wrong-output', 'quality', 'performance'] },
          scenario: { type: 'string', description: 'concrete inputs/state -> what goes wrong (be specific)' },
          evidence: { type: 'string', description: 'what you checked or ran that shows it' },
          fix: { type: 'string', description: 'suggested fix' },
        },
        required: ['title', 'file', 'line', 'severity', 'scenario', 'evidence', 'fix'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT = {
  type: 'object',
  properties: {
    real: { type: 'boolean', description: 'true only if the bug is real and would actually happen' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    reasoning: { type: 'string' },
    repro: { type: 'string', description: 'what you ran or read to decide' },
    better_fix: { type: 'string', description: 'a better fix if the suggested one is wrong; empty otherwise' },
  },
  required: ['real', 'confidence', 'reasoning', 'repro', 'better_fix'],
}

const AREAS = [
  { key: 'isaac', prompt: 'Area: Isaac integration. Files: scripts/drone_scene/quadcopter_pbr.py, scripts/drone_scene/world/runner.py, scripts/drone_scene/vehicles.py, scripts/heavy_machines/setup_linux.sh (drone path). Check against the Isaac Lab source in source/isaaclab: Camera/CameraCfg (offset convention opengl, render_product_paths, data_types, update), PinholeCameraCfg fields, Articulation/ArticulationCfg with a 0-joint USD (InitialStateCfg joint_pos={} etc.), permanent_wrench_composer, write_root_pose_to_sim, set_world_poses_from_view, SimulationContext API, sim.set_camera_view; Replicator HdrColor annotator usage; ordering (prepare/world build before AppLauncher; stage authoring before sim.reset); the straight-route code path (scene_cfg, make_flight, robot_z0 / ground clearance vs patio height, chase camera, record -1); VideoWriter (ffmpeg pipe: deadlocks, partial writes, closing, os._exit before close); runner (pip --target into Isaac python, uv fallback, PYTHONPATH order, subprocess env, failure messages); anything that would crash or misbehave on the user PC.' },
  { key: 'flight', prompt: 'Area: flight and camera geometry. Files: scripts/drone_scene/flight.py (SmoothMove, StraightFlight, _Trajectory camera_matrix/camera_local), how quadcopter_pbr.py builds StraightFlight (make_flight: start z, ground clearance, cruise height vs landing ground), vehicles.mount_cameras (pitch, positions), and world/build.py cruise_height. Check math: continuity of position/velocity/acceleration across phases, the drag-tilt attitude, heading convention (counter-clockwise from +x east) vs the map heading, that the drone lands exactly on the landing pad height (z of the flattened terrain + clearance) and takes off from the patio top, camera quaternion conventions vs Isaac Lab OffsetCfg convention="opengl".' },
  { key: 'geometry', prompt: 'Area: map geometry. Files: scripts/drone_scene/world/terrain.py, buildings.py, roads.py, usdio.py, textures.py, and their use in build.py (terrain_meshes, far_terrain, write_usd). Check: triangle winding and normals, skirts/cracks between LOD tiles of different cell size, the far terrain hole vs detail zone, UV conventions (faceVarying st count = 3 x faces), GeomSubset partition validity, roof geometry (gable/hip on rotated rectangles, overhang, cross-gables), walls vs ground (foundation sink), road ribbons at sharp turns / self-overlap / z-fighting with terrain and with each other, markings, PointInstancer authoring (orientations quath wxyz, scales, protoIndices, prototype scale op for grass), material binding paths, texture paths absolute and existing. Inspect the built map ' + WORLD + '/world.usdc with pxr to support claims (e.g. check normals, extents, subsets, a few buildings).' },
  { key: 'geodata', prompt: 'Area: geodata and map logic. Files: scripts/drone_scene/world/geo.py, data.py, landcover.py, site.py, vegetation.py, materials.py, build.py (find_site, ground_classes, ground_materials, cruise_height, landing_grass, manifest, cache key) and the adapters into world/sources/*.py. Check: frame/CRS consistency (frame z0 handling: heights relative to take-off ground everywhere, including far terrain and buildings via ground_any), Grid convention (x0, y1, res, north-up) vs rasterio transforms (from_origin, Affine), bbox_lonlat coverage, reprojection resampling, WorldCover code mapping, burn order of Overture land use, Sentinel-2 tint calibration math (reflectance vs texture albedo, clamps), tree detection/infill/species/scale (model heights, units 0.01), site selection correctness (backyard side, distances, grass check), the cache key including everything that changes output, and anything that would produce a wrong-looking map. Use the built map ' + WORLD + ' (world.json and the usdc) to check claims.' },
  { key: 'camera', prompt: 'Area: LUCID camera simulation. Files: scripts/drone_scene/lucid.py, scripts/drone_scene/sensor_sim.py, and their use in quadcopter_pbr.py (HdrReader, LucidSensor per camera, camera_info json). Check: OpenCV distortion and undistortion, render_plan overscan coverage and Isaac intrinsics consistency (fx = width*focal/aperture), grid_sample coordinate conventions (align_corners=False, pixel centres), lateral colour sign, vignetting, Malvar demosaic kernels and RGGB layout, noise model (Poisson in electrons, read noise, ADC, saturation), auto-exposure convergence and the reset on source change, white balance, gamma, dtype/device handling on CUDA (torch.Generator device, poisson on CUDA), and the HdrColor -> tensor conversion (float16, RGBA, shape).' },
]

const results = await pipeline(
  AREAS,
  a => agent(`${CONTEXT}\n\n${a.prompt}\n\nYou are a careful code reviewer. Find REAL bugs: crashes, wrong output, or clear quality/performance problems that matter for the user (photorealistic drone camera videos over a real 2 km map, rendered on their own GPU). Read the code closely; verify each claim (run small snippets where possible). Do not report style issues, missing docs or speculative concerns you could not support. Report at most 10 findings, most severe first.`,
    { label: `find:${a.key}`, phase: 'Find', schema: FINDINGS, model: 'sonnet' }),
  (res, a) => parallel((res?.findings || []).map(f => () =>
    agent(`${CONTEXT}\n\nA reviewer claims this bug in area "${a.key}":\nTitle: ${f.title}\nFile: ${f.file}:${f.line}\nSeverity: ${f.severity}\nScenario: ${f.scenario}\nEvidence: ${f.evidence}\nSuggested fix: ${f.fix}\n\nYour job is to try to REFUTE it. Read the code yourself and, where possible, reproduce it with a small script. Decide real=true only if the bug actually happens as described (or in a closely related way) for realistic inputs on the user's setup. Default to real=false if you cannot confirm it.`,
      { label: `verify:${a.key}:${(f.title || '').slice(0, 30)}`, phase: 'Verify', schema: VERDICT, model: 'sonnet' })
      .then(v => ({ area: a.key, ...f, verdict: v }))))
)
const all = results.flat().filter(Boolean)
log(`${all.length} findings, ${all.filter(f => f.verdict && f.verdict.real).length} confirmed`)
return all
