export const meta = {
  name: 'drone-photoreal-v4',
  description: 'UE5-style realism for the drone scene: 3D herringbone paving, drone close-up detail, RTX camera-pipeline research; each verified',
  phases: [
    { title: 'Build', detail: 'paving, drone detail, RTX research (Sonnet)', model: 'sonnet' },
    { title: 'Verify', detail: 'skeptical review of each piece', model: 'sonnet' },
    { title: 'Fix', detail: 'apply confirmed fixes', model: 'sonnet' },
  ],
}

const SP = '/tmp/claude-0/-home-user-vla-datagen/39bfe37a-c345-5663-b8fb-e785cd7a1b70/scratchpad'
const PY = `${SP}/venv/bin/python`
const COMMON = `
Project: /home/user/vla-datagen (Isaac Lab fork). Package scripts/drone_scene/ makes Isaac Lab's quadcopter demo
(a Crazyflie hovering 0.5 m above a backyard patio) photorealistic with PBR. Read these first:
- scripts/drone_scene/__init__.py
- scripts/drone_scene/scene.py (the layout, UsdPreviewSurface helper preview_surface(), _mesh(), box(),
  apply_drone_materials)
- scripts/drone_scene/assets.py
Isaac Sim is NOT installed here. The offline preview renders the same USD with Mitsuba 3:
scripts/drone_scene/mitsuba_preview.py (render(stage, camera_path, out_png, width, height, spp, exposure_ev),
Session, camera_from_matrix). scripts/drone_scene/preview.py shows how a full preview stage is built
(build_preview_stage, exposure_ev).

Environment:
- Python with numpy, Pillow, usd-core (pxr), mitsuba 3.9.1 and pyflakes: ${PY}. 4 CPU cores.
- Asset cache: ~/.cache/drone_scene, already holding the NVIDIA textures. Get paths with
  drone_scene.assets.material_textures(name, drone_scene.prepare.DEFAULT_CACHE); the manifest comes from
  drone_scene.prepare.prepare(cache, mirror=True).
- The Crazyflie is at ~/.cache/drone_scene/Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd.
- IMPORTANT: a 2-hour Mitsuba flight render is running in the background and uses all cores. Keep your own
  test renders SMALL: at most 480x270 and 16 spp, and few of them. Never kill python processes.
- Scratch files only under ${SP}/v4/<your piece>/.

Rules:
- Write ONLY the files named for your piece. Do NOT edit scene.py, preview.py, quadcopter_pbr.py,
  mitsuba_preview.py, render_flight.py or assets.py; the main agent integrates.
- No git add, commit or push.
- Style: like the existing package.
  - from __future__ import annotations, type hints, short docstrings.
  - Comments only where they explain why.
  - pxr imported inside functions, so modules import before Isaac Sim starts.
  - numpy/Pillow only at module level.
- Tests: unittest, run as ${PY} scripts/drone_scene/test/<file>.py. They must put scripts/ on sys.path.
- pyflakes must be clean.
- Your final message (it is the main agent's only input): public API with signatures, how to integrate it
  (exact calls), the test command and result, the paths of preview PNGs you rendered, and known
  limitations.
`

const PIECES = [
  {
    key: 'pavers',
    files: 'scripts/drone_scene/pavers.py and scripts/drone_scene/test/test_pavers.py',
    prompt: `Build scripts/drone_scene/pavers.py: replace the flat textured patio with REAL 3D herringbone clay
pavers, the way Unreal Engine 5 scenes get depth from real micro-geometry (Nanite) and scanned assets.

Current patio:
- scene.SceneCfg().patio = (x0, x1, y0, y1) = (-1.6, 2.0, -2.2, 2.0), top z = cfg.patio_top = 0.0;
- a concrete curb box surrounds it (scene.curb_boxes);
- material: NVIDIA Base "Brick_Pavers": albedo, ORM (R AO, G roughness, B metallic) and DirectX normal
  maps at 2048 px, one repeat = assets.MATERIALS["pavers"].tile_m = 1.6 m, axis-aligned herringbone, 200 x
  100 mm pavers;
- assets.MATERIALS["sand"] (Sand.mdl, tile 0.8 m) is available for the joints.

Requirements:
1. Brick atlas.
   - From the Brick_Pavers texture, find the individual paver rectangles (axis-aligned, 2:1 or 1:2, about
     256x128 px).
   - Use the ORM red channel (AO is dark in the joints) and/or albedo luminance to segment them: threshold,
     connected components, keep components whose bounding box has the right size and aspect, then shrink
     slightly to stay inside the brick.
   - Expect roughly 100+ usable bricks per texture.
   - Return them as UV rects in USD UV space. USD v=0 is the BOTTOM of the image; get this right and test it.
   - Cache the result as JSON next to the textures.
2. Layout.
   - A 90-degree herringbone of 200x100x60 mm pavers, axis-aligned, covering the patio rect with 3 mm
     joints.
   - Pavers crossing the edge are cut to the rect (clip the box, keep proper side faces).
3. Geometry.
   - Each paver is a box with 4 mm chamfered top edges: chamfer faces between top and sides.
   - Each gets a tiny random height offset (±1.0 mm) and tilt (±0.25 deg).
   - Top face UVs map into one randomly chosen atlas rect (rotated 90 deg when the brick's orientation
     differs from the crop's), so neighbouring bricks differ.
   - Chamfers and sides use the same crop's edge.
   - Merge everything into a few meshes, with vertex normals split at hard edges.
4. Joints. A flat sand surface (Sand material) at z = top - 8 mm under the whole patio, so joints show sand
   between bricks. Optionally darken joints near the curb with a tint.
5. Physics. Keep collision simple: the caller will keep a flat collider. Provide flag collision=False (no
   physics API on the visual bricks).
6. API.
   - author_patio(stage, root: str, rect, top_z: float, cache_dir: str, *, seed: int = 0) -> dict summary.
     It authors meshes under f"{root}/Ground/PatioBricks" and f"{root}/Ground/PatioSand", plus materials via
     scene.preview_surface with the same textures (flip_green for the DirectX normal maps, as
     scene.author_ground_materials does).
   - The brick-crop UV transform must work in both RTX and the Mitsuba preview: plain mesh UVs (st
     primvar), no UsdTransform2d.
7. Visual check.
   - Build a small test stage: the patio, a DistantLight sun at elevation 35 deg from azimuth 60 deg,
     intensity ~3800 (see manifest["sky"]), a DomeLight using manifest["sky"]["file_preview"] with its
     dome_intensity, and a camera at (0.5, 0.5, 1.0) looking at (0, 0, 0.5) (scene.look_at_matrix;
     focal 18.147 mm, aperture 20.955).
   - Render with mitsuba_preview.render at 480x270, 16 spp, exposure_ev = preview.exposure_ev(manifest),
     before and after: flat textured box vs your bricks.
   - Save PNGs in ${SP}/v4/pavers/ and look at them.
   - The bricks must read as real pavers: visible chamfer highlights, sand joints, brick-to-brick variation,
     no UV stretching, no gaps at the rect border.
8. Tests:
   - atlas rects inside [0,1] and of the right size/aspect;
   - herringbone coverage (sum of brick top areas ≈ rect area minus joints, ±3%);
   - no two bricks overlapping (sample points);
   - all bricks inside the rect;
   - chamfer normals between top and side normals;
   - UVs of each brick inside its crop;
   - USD stage validity (meshes, materials bound, st primvar);
   - the summary counts.`,
  },
  {
    key: 'drone',
    files: 'scripts/drone_scene/drone_look.py and scripts/drone_scene/test/test_drone_look.py',
    prompt: `Build scripts/drone_scene/drone_look.py: make the Crazyflie read as a real Crazyflie 2.1 in close-ups
(the hero camera is a 70 mm lens about 1 m away, the drone fills ~40% of the frame).

Current state: scene.apply_drone_materials(stage, drone_root) binds flat UsdPreviewSurface constants by mesh
name (DRONE_PARTS / DRONE_MATERIALS in scene.py). The asset (cf2x.usd, read it with pxr) has 10 meshes with
primvars:st:
- body_visual/{cf_body_001 (the PCB frame), motor_mount, Cylinder (motors), battery, battery_holder,
  pin_headers};
- m1..m4_prop/{ccw_prop|cw_prop}.

Requirements:
1. Inspect the UV layout of cf_body_001 and the props: render UV wireframes to PNG with Pillow.
2. PCB texture set for cf_body_001.
   - Generate procedurally with numpy/Pillow into cache_dir/drone/: albedo, roughness and normal PNGs at
     2048 px.
   - Content: matte-black solder mask with fine copper-trace relief (subtle normal and roughness variation
     along traces), white silkscreen outlines/labels, gold ENIG pads and vias, and a few small black IC
     packages with slightly different roughness.
   - Align it with the UV islands so features land on the frame, not across seams. If the UVs are unusable
     (overlapping or degenerate), use a planar top-down projection instead: compute new st from object-space
     x, y and author it as a new primvar (st_pcb), and use that varname.
3. Status LEDs. The Crazyflie 2.1 has 4 small LEDs near the motor arms: front two red/green, rear two blue,
   viewed from above. Add tiny emissive boxes (about 1.6x0.8x0.6 mm) on the PCB top near the arm roots, with
   UsdPreviewSurface emissiveColor. The emission must be bright enough to read as an LED in daylight but not
   light the scene.
4. Propeller motion blur. A hovering Crazyflie spins its props at ~250-350 rev/s, so in any photo they are a
   translucent blurred disc, not crisp blades.
   - Option prop_blur=True: make the blade meshes invisible and add under each m*_prop prim a thin disc mesh
     of the prop's radius (from the blade mesh extent), with a radial alpha/colour texture: faint dark blur
     where the blades sweep, slightly denser near the hub.
   - Use UsdPreviewSurface opacity (and opacityThreshold 0) for real translucency, which RTX supports with
     translucency enabled.
   - Check what mitsuba_preview does with non-cutout opacity, and make sure the preview still looks right.
5. Better plastics and metals for the other parts:
   - motor_mount: off-white ABS with a slight sheen;
   - motors: brushed steel with anisotropy approximated by roughness;
   - battery: silver foil with a printed label area if UVs allow;
   - pin_headers: black plastic + gold pins (one mesh, so pick gold-ish dark).
   Keep UsdPreviewSurface only, so RTX and the preview match.
6. API: apply_drone_look(stage, drone_root: str, cache_dir: str, *, prop_blur: bool = True,
   leds: bool = True) -> dict summary. Call it AFTER scene.apply_drone_materials; it rebinds the parts it
   improves.
7. Visual check.
   - Build a stage like preview.build_preview_stage (drone at (0,0,0.5), the real scene via
     prepare.prepare(..., mirror=True)), or a lighter one with just the drone, the sky dome and the sun from
     the manifest.
   - Render the hero camera (scene.SceneCfg().camera("hero"): 70 mm, f/8) at 480x270, 16 spp with
     mitsuba_preview.render, before and after. Save the PNGs in ${SP}/v4/drone/ and look at them.
   - Crop the drone region and upscale it to judge detail.
8. Tests:
   - textures written with the right sizes;
   - every mesh bound;
   - LEDs present and emissive;
   - prop discs present with radius ≈ blade extent and opacity texture;
   - blades invisible when prop_blur, and visible when not;
   - re-running is idempotent.`,
  },
  {
    key: 'rtx',
    files: 'scripts/drone_scene/rtx_look.py and scripts/drone_scene/test/test_rtx_look.py',
    prompt: `Research, then build scripts/drone_scene/rtx_look.py: the RTX renderer settings that give Isaac Sim 5.1
(Kit 107.x, RTX Real-Time 2.0 and the Path Tracer) a physically correct, cinematic camera pipeline.
The goal is the look of Unreal Engine 5 cinematics.

Research (use ToolSearch "select:WebSearch,WebFetch"; cite URLs; prefer docs.omniverse.nvidia.com Kit/RTX
renderer settings docs and docs.isaacsim.omniverse.nvidia.com):
1. Exact carb setting paths, types, defaults and meaning for each of these. Check whether each applies in
   Real-Time, Path Tracing or both.
   - the tonemapper: /rtx/post/tonemap/op values (e.g. ACES, filmic, ...), filmIso, cameraShutter, fNumber,
     cm2Factor, whitepoint, enableSrgbToGamma, maxWhiteLuminance;
   - auto exposure / histogram: /rtx/post/histogram/enabled and its parameters;
   - bloom / lens flares: /rtx/post/lensFlares/...;
   - chromatic aberration;
   - vignette;
   - film grain / tv noise;
   - depth of field enable;
   - motion blur;
   - /rtx/sceneDb/ambientLightIntensity: every Isaac Lab rendering preset in this repo sets it to 1.0 (see
     apps/rendering_modes/*.kit and apps/*.kit). Confirm what it does. We believe it is an artificial
     uniform fill light that flattens a physically lit scene, so 0 is correct when a DomeLight is present;
   - dome light sampling quality, /rtx/domeLight/upperLowerStrategy (the presets set 3 or 4: what do they
     mean?);
   - real-time GI / indirect diffuse quality;
   - reflections and translucency quality;
   - path-tracer settings: spp, totalSpp, maxBounces, maxSpecularAndTransmissionBounces, fireflyFilter /
     clamp, optixDenoiser;
   - DLSS/DLAA.
2. How UE5 gets photoreal imagery (Lumen GI, Nanite micro-geometry, virtual shadow maps, Megascans,
   physical camera exposure, ACES filmic tonemapping, post: bloom, vignette, chromatic aberration, grain).
   Map each to its Omniverse RTX equivalent or to what our scene does. Keep it short, with sources.

Then build rtx_look.py:
- SETTINGS: dict[str, dict[str, value]] with presets "realtime" (RTX Real-Time) and "pathtracing".
  Every value must come from your research and be commented with why.
  - Physical exposure tuned for our lighting: a DomeLight intensity of ~900 and a DistantLight sun of
    ~3800, in Isaac's units, where Isaac Lab's default exposure is tuned for a dome of 750 with the
    kloofendal sky (see assets.REFERENCE_INTENSITY and prepare.py's calibration notes).
  - Note: only set exposure-related keys if you are confident of their effect. Otherwise leave the
    defaults and say so.
- apply(set_setting, get_setting, preset: str, log) -> dict of applied keys.
  - Warn, without failing, for keys get_setting reports as None (unregistered in this Kit). The script uses
    sim.set_setting / sim.get_setting.
- A short module docstring summarising the UE5-to-RTX mapping, with the source URLs.

Tests: presets well formed (types), apply() calls the setter for each key and warns for unregistered ones,
and ambientLightIntensity is 0 in both presets.
If a web page is blocked, say so and use what you can verify (e.g. the repo's .kit files, Isaac Lab source
under source/isaaclab that sets RTX settings, or Kit settings names found in installed packages). Never
invent a setting path: list unverified ones separately in your final message.`,
  },
]

const BUILD = (p) => `${COMMON}\n\nYOUR PIECE: ${p.key}. Files you own: ${p.files}.\n\n${p.prompt}`
const VERIFY = (p, built) => `${COMMON}

You are a skeptical verifier of the piece "${p.key}" (files: ${p.files}). The builder reported:
---
${built}
---
1. Read the code and tests fully.
2. Run the tests and pyflakes.
3. Look at the builder's preview PNGs (and render at most 2 small ones yourself if needed: 480x270,
   ≤16 spp).
4. Try hard to find real problems:
   - spec violations;
   - wrong conventions (UV v direction, normal-map flip, units, transforms, RTX semantics);
   - visual artefacts;
   - integration traps (prim paths, rebinding order, idempotence);
   - for research: claims and setting paths not backed by a source.
Do NOT edit files. Report CONFIRMED problems with evidence, severity (blocker/major/minor) and exact fixes,
plus what you verified as correct.

Original spec:
${p.prompt}`
const FIX = (p, built, verdict) => `${COMMON}

YOUR PIECE: ${p.key}. Files you own: ${p.files}.
Builder report:
---
${built}
---
Verifier report:
---
${verdict}
---
Re-check each claim against the code first. Fix every confirmed blocker or major problem, and minor ones
when they are cheap. Re-run the tests and pyflakes, and re-render your check PNG once if visuals changed.
The final message replaces the builder's report as the main agent's input, so restate the full public
API, the integration calls, the test results, the PNG paths, what changed, what you skipped, and the
limitations.`

const results = await pipeline(
  PIECES,
  (p) => agent(BUILD(p), { label: `build:${p.key}`, phase: 'Build', model: 'sonnet' }),
  (built, p) => agent(VERIFY(p, built), { label: `verify:${p.key}`, phase: 'Verify', model: 'sonnet' })
    .then((verdict) => ({ built, verdict })),
  (r, p) => agent(FIX(p, r.built, r.verdict), { label: `fix:${p.key}`, phase: 'Fix', model: 'sonnet' })
    .then((fixed) => ({ piece: p.key, fixed })),
)
return results
