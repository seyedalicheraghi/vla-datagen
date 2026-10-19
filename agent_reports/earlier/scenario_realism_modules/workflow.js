export const meta = {
  name: 'scenario-realism-modules',
  description: 'Sonnet agents build 4 independent realism modules (bevelled machine meshes, PBR textures, rocks, camera look), each verified',
  phases: [
    { title: 'Build', detail: 'one agent per new module + its offline test', model: 'sonnet' },
    { title: 'Verify', detail: 'skeptic checks each module visually and numerically', model: 'sonnet' },
    { title: 'Fix', detail: 'only if the skeptic found real problems', model: 'sonnet' },
  ],
}

const MODEL = 'sonnet'
const PY = args.py
const OUT = args.out
const COMMON = `Repo /home/user/vla-datagen. Package scripts/heavy_machines (pure-Python construction-machine specs; read spec.py for Shape/Box/Cylinder/Sphere/Link/MachineSpec and kinematics.py). New scenario code lives in scripts/heavy_machines/scenarios/ — read terrain.py (QuarryTerrain, Mesh dataclass, grid_mesh, bake_albedo, value_noise/fbm), scenario.py, adt_approach_hex.py and render_scenario.py (the Isaac Sim runner that will consume your module; it authors USD UsdPreviewSurface materials with UsdUVTexture inputs: diffuseColor, roughness, metallic, normal, opacity, clearcoat; textures are PNG files referenced by absolute path, wrap 'repeat' for tileable sets).
GOAL OF THE WHOLE EFFORT: the rendered Isaac Sim video currently looks very synthetic; your module must make it look like real camera footage of a real quarry with real machines. Quality and realism matter more than code size.
Python with numpy, matplotlib, Pillow, torch: ${PY}. numpy only at runtime in your module (no scipy/opencv); Pillow allowed for saving.
Rules: create ONLY the files named in your task (other agents create other files in parallel; do not edit existing files). Write a unittest file scripts/heavy_machines/test/<your test>.py runnable as \`${PY} scripts/heavy_machines/test/<file>.py\` (sys.path insert like the other tests). Save demo images to ${OUT}/<module>/ and LOOK at them with the Read tool, iterating until they look photographic, not like CG. Return a summary of the API, design choices, test results and demo image paths.`

const SUMMARY = { type: 'object', properties: { status: { type: 'string', enum: ['ok', 'failed'] }, api: { type: 'string' }, summary: { type: 'string' }, demo_images: { type: 'array', items: { type: 'string' } }, test_result: { type: 'string' } }, required: ['status', 'api', 'summary', 'demo_images', 'test_result'] }
const VERDICT = { type: 'object', properties: { verdict: { type: 'string', enum: ['pass', 'fail'] }, problems: { type: 'array', items: { type: 'object', properties: { severity: { type: 'string', enum: ['must_fix', 'should_fix', 'nit'] }, description: { type: 'string' }, evidence: { type: 'string' }, fix: { type: 'string' } }, required: ['severity', 'description', 'evidence', 'fix'] } } }, required: ['verdict', 'problems'] }

const MODULES = [
  {
    key: 'visual_meshes',
    task: `Create scripts/heavy_machines/scenarios/visual_meshes.py + test/test_visual_meshes.py.
Render-quality meshes for machine shapes (the URDF/Isaac importer gives razor-sharp primitives with no UVs, which reads as CG).
API (exact):
  from .terrain import Mesh   # reuse; set rect=(0.0, 0.0, 0.0, 0.0)
  def shape_mesh(shape: Shape, bevel: float = 0.025, segments: int = 24) -> Mesh
      Triangle mesh of one Box/Cylinder/Sphere in its LINK frame (apply shape.xyz/shape.rpy via kinematics.Tf.from_xyz_rpy), outward unit vertex normals, UVs in METRES (1 UV unit = 1 m of surface) so tileable textures keep a constant physical scale.
      Box: rounded edges/corners with radius r = min(bevel, 0.3*min(size)) (fillets catch highlights — this is the main point), smooth normals on the fillets, flat on faces; per-face planar UVs.
      Cylinder (axis = local z): side with cylindrical UVs (u = arc length, v = height), caps with planar UVs, small fillet on both rims (r as above vs radius/length).
      Sphere: UV sphere.
      Triangle winding must be counter-clockwise seen from outside (right-handed, normal = cross(b-a, c-a) points outward).
  def machine_visual_meshes(spec: MachineSpec, bevel: float = 0.025) -> dict[str, list[tuple[str, Mesh]]]
      link name -> list of (material name, merged mesh) — all VISUAL shapes (shape.visual) of that link grouped by shape.material and merged into one mesh per material.
Tests: watertight-ish (each closed shape: Euler characteristic / edge use counts), outward normals (mean of dot(normal, vertex - centroid) > 0 for convex shapes), normals unit length, bounding box of a bevelled box equals its size within 1e-6, UV scale correct (distance between two vertices along a face edge ≈ UV distance), winding consistent with normals, works for every link of every registered machine (heavy_machines.registry) and stays under ~400k triangles per machine at defaults. Demo: render the hydraulic_excavator and articulated_dump_truck meshes with matplotlib (Lambert + a simple specular term) so the fillet highlights are visible, and look at it.`,
  },
  {
    key: 'pbr_textures',
    task: `Create scripts/heavy_machines/scenarios/pbr_textures.py + test/test_pbr_textures.py.
Procedural PBR texture sets (numpy) for machines and terrain.
API (exact):
  @dataclass class TextureSet: albedo: np.ndarray (H,W,3 uint8, sRGB); roughness: np.ndarray (H,W uint8, linear 0..255 -> 0..1); normal: np.ndarray (H,W,3 uint8, tangent space, OpenGL convention +Y up, flat = (128,128,255)); metallic: float; metres_per_tile: float; opacity: float = 1.0; clearcoat: float = 0.0
  def machine_textures(material_name: str, base_rgb: tuple[float,float,float], seed: int = 0, size: int = 1024) -> TextureSet
      TILEABLE (seamless when wrapped) — use periodic noise. Names come from spec.DEFAULT_MATERIALS keys: body (painted steel, yellow): realistic construction-machine paint with fine orange-peel normal, light dust film (more dust = lighter/desaturated, rougher), dirt speckles and splashes, small chips/scratches showing dark primer/steel, faint streaks (rain/oil); roughness ~0.35-0.55 varying with dust. accent/dark: black/dark-grey painted or plastic trim, dusty. steel: brushed/worn bare steel, metallic=1, roughness 0.3-0.5 with scratches. rubber: tyre/track rubber, near-black, roughness ~0.85, fine grain + dust in crevices. glass: dark tinted glass, low roughness 0.05, opacity 0.35-0.5, faint dust/water spots. hub/red/orange/white/rust: sensible variants. Unknown names: derive from base_rgb (painted). metres_per_tile ~1.0-2.0 so dust/wear features have real-world size.
  def terrain_detail_maps(terrain, rect, px_per_m: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]
      (normal RGB uint8 (H,W,3), roughness uint8 (H,W)) covering rect=(x0,x1,y0,y1) with EXACTLY the same pixel layout as terrain.bake_albedo(terrain, rect, px_per_m) (row 0 = y1, W = round((x1-x0)*px), H = round((y1-y0)*px)). Normal = fine micro-relief NOT represented in the mesh: gravel/pebble bumps (more off-lane), compacted lane with tyre-tread imprints inside the tyre tracks (terrain.tracks_y, terrain.extra_tracks, lane mask), small erosion rills on steep faces (terrain.surface_class slope), cracks on the dry compacted lane. Roughness: lane slightly smoother/compacted, loose sand rougher.
  def save_texture_set(ts: TextureSet, directory: str, prefix: str) -> dict[str, str]   # writes PNGs, returns {"albedo":..., "roughness":..., "normal":...} absolute paths
Tests: tileability (wrap-around seams: mean abs difference across the wrap edge comparable to interior neighbour differences), dtype/shape, normal maps decode to unit vectors with z > 0, flat regions -> (128,128,255), terrain maps match bake_albedo shape for a few rects/px values, determinism with seed. Demo: render sphere/plane swatches lit with Lambert+GGX-ish specular using albedo/normal/roughness for each machine material, and a terrain crop (lane + berm + bench face) combining bake_albedo with your normal map under a low sun; look at them and iterate until they look like photographs of real paint/rubber/ground.`,
  },
  {
    key: 'rocks',
    task: `Create scripts/heavy_machines/scenarios/rocks.py + test/test_rocks.py.
Natural rocks and gravel to replace the current perfect UsdGeom.Sphere rocks (terrain.scatter_rocks).
API (exact):
  from .terrain import Mesh, QuarryTerrain
  def rock_mesh(seed: int, detail: int = 3, angularity: float = 0.6) -> Mesh
      Irregular, believable quarry rock about 1 m across, centred at the origin, flattish bottom: start from a subdivided icosphere, apply anisotropic scaling + multi-octave displacement + a few planar 'fracture' cuts (angular blasted rock); vertex normals faceted-ish (angularity blends flat vs smooth); UVs (triplanar-style box projection in metres). Counter-clockwise outward winding.
  def rock_prototypes(n: int = 8, seed: int = 0) -> list[Mesh]
  @dataclass class Scatter: positions (N,3) float32, scales (N,3) float32, yaws (N,) float32, proto (N,) int32
  def scatter(terrain: QuarryTerrain, rect, seed: int = 0, keep_clear=None, boulders: int = 120, gravel_per_m2: float = 1.2) -> Scatter
      Two populations: boulders (0.25-1.2 m, biased to berm feet, bench toe, lane edges as windrow spill; none on the lane itself or in keep_clear rects) and gravel (0.03-0.15 m, everywhere off the compacted lane, sparse ON the lane (spillage, ~5% density), denser near lane edges). Sit them partly embedded: z = terrain height - 25..40% of their height. Deterministic for a seed.
Tests: rock meshes valid (Mesh.validate, closed, outward normals), bottoms roughly flat, scatter respects keep_clear and lane density rules, counts scale with area/density, embedding depth, determinism. Demo: matplotlib render of the 8 prototypes lit (Lambert), and a top-down + oblique render of a 20x15 m patch around the lane edge with the scatter, look at them and iterate until the rocks read as real blasted quarry rock, not blobs.`,
  },
  {
    key: 'camera_look',
    task: `Create scripts/heavy_machines/scenarios/camera_look.py + test/test_camera_look.py.
Post-processing that turns a clean render into believable camera footage (the renderer gives perfect, noiseless, hazeless CG).
API (exact):
  @dataclass class LookParams: (all with sensible defaults for a 4 m mast camera in a dusty open-pit quarry at hazy midday)
      haze_visibility_m: float = 900.0; haze_color: tuple = (0.82, 0.84, 0.86); haze_strength: float = 1.0
      exposure: float = 1.0; contrast: float = 1.0 (filmic S-curve strength); saturation: float = 0.9; white_balance: tuple = (1.02, 1.0, 0.97)
      highlight_rolloff: float = 0.8; bloom_strength: float = 0.06; bloom_radius_px: float = 18.0
      vignette: float = 0.22; chromatic_aberration_px: float = 0.7; distortion_k1: float = -0.025 (barrel)
      sharpen: float = 0.25; grain: float = 0.018 (luma-weighted sensor noise); dust_in_air: float = 0.0
  def apply(rgb: np.ndarray, depth_m: np.ndarray | None = None, params: LookParams = LookParams(), seed: int = 0) -> np.ndarray
      rgb (H,W,3) or (H,W,4) uint8 sRGB -> (H,W,3) uint8. depth_m (H,W) metres from the camera (inf/NaN/0 = sky, treat as far). Order: linearise -> aerial-perspective haze with depth (exp(-d/visibility)) -> exposure/white balance -> bloom on bright areas -> filmic tonemap/highlight rolloff + contrast + saturation -> lens distortion + chromatic aberration (radial, red/blue scaled differently) -> vignette -> sharpen -> grain (seeded, temporally varying via seed) -> sRGB uint8. Pure numpy; must process a 1280x720 frame in < 0.6 s here.
  def identity_params() -> LookParams   # every effect off; apply(...) must then return the input (±1 LSB)
Tests: identity passthrough, shapes/dtypes incl. RGBA input and None depth, haze increases with depth monotonically toward haze_color, grain changes with seed but mean brightness stable, vignette darkens corners not centre, distortion keeps the centre pixel fixed, speed bound. Demo: run it on the offline previews in ${OUT}/../scen/*.png frames is not ideal (they are flat-shaded); instead build a synthetic test scene: render a perspective view with matplotlib or numpy of a ground plane with checker/sand texture, a yellow box machine and sky gradient + a matching depth map, then show before/after side by side; look at it and tune defaults so the after image reads like a real camera frame (subtle: not Instagram-filtered).`,
  },
]

phase('Build')
const results = await pipeline(
  MODULES,
  (m) => agent(`${COMMON}\n\nYOUR TASK (${m.key}):\n${m.task}`, { label: `build:${m.key}`, phase: 'Build', schema: SUMMARY, model: MODEL }),
  (built, m) => agent(`${COMMON}\n\nA colleague built the module '${m.key}' for this task:\n${m.task}\nTheir report: ${JSON.stringify(built)}\n\nYou are the skeptical reviewer. Do NOT edit files. Read the code, run its test file, run your own extra numeric checks (edge cases, other machines/seeds/rects, API exactly as specified — the integrator will call it blindly), and LOOK at the demo images (re-generate them if needed). Judge hard: does the output actually look like real-world materials/rocks/camera footage, or still CG? Any API deviation, crash, seam, wrong normal convention, NaN, performance problem = must_fix. Return verdict and problems with evidence and concrete fixes.`,
    { label: `verify:${m.key}`, phase: 'Verify', schema: VERDICT, model: MODEL }).then((v) => ({ built, v })),
  async (r, m) => {
    const real = (r && r.v && r.v.problems || []).filter((p) => p.severity !== 'nit')
    if (!r || !r.v || (r.v.verdict === 'pass' && !real.length)) return { module: m.key, ...r, fix: null }
    const fix = await agent(`${COMMON}\n\nYOUR TASK (${m.key}) was:\n${m.task}\nA reviewer found these problems in the current files:\n${JSON.stringify(real, null, 1)}\nVerify each against the code/output; fix those that hold (you may edit only this module's two files), explain those that don't. Re-run the tests and re-check the demo images.`,
      { label: `fix:${m.key}`, phase: 'Fix', schema: SUMMARY, model: MODEL })
    return { module: m.key, ...r, fix }
  },
)
return results
