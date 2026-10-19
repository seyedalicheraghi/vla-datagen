export const meta = {
  name: 'drone-pbr-tools',
  description: 'Build, verify and fix three tools for the PBR drone scene: Nucleus asset fetcher, HDR sky tools, USD->Mitsuba preview renderer',
  phases: [
    { title: 'Build', detail: 'one Sonnet agent per tool, with tests', model: 'sonnet' },
    { title: 'Verify', detail: 'skeptical review + own tests per tool', model: 'sonnet' },
    { title: 'Fix', detail: 'apply confirmed fixes, rerun tests', model: 'sonnet' },
  ],
}

const SP = '/tmp/claude-0/-home-user-vla-datagen/39bfe37a-c345-5663-b8fb-e785cd7a1b70/scratchpad'
const PY = `${SP}/venv/bin/python`
const COMMON = `
Project: /home/user/vla-datagen, a fork of Isaac Lab. You are building one module of a new package
scripts/drone_scene/. The package gives Isaac Lab's quadcopter demo (scripts/demos/quadcopter.py: one
Crazyflie hovering at 0.5 m) photorealistic PBR scenery. A separate offline preview, using Mitsuba 3,
path-traces the same USD stage so looks can be iterated without Isaac Sim, which is NOT installed here.
The package's __init__.py already exists; read it.

Environment:
- Python with numpy, Pillow, usd-core (pxr), mitsuba 3.9.1 (variants llvm_ad_rgb and scalar_rgb) and
  pyflakes: ${PY}. 4 CPU cores. The system python3 has no packages; do not use it.
- Internet: NVIDIA's public asset server (Nucleus mirror on S3) is reachable at
  NUCLEUS_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1".
  Paths below it look like "Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd" and
  "NVIDIA/Materials/Base/Masonry/Brick_Pavers.mdl". Directory listing works with the S3 v2 API:
  GET https://omniverse-content-production.s3-us-west-2.amazonaws.com/?list-type=2&prefix=Assets/Isaac/5.1/<dir>/&delimiter=/
  which returns <Prefix>/<Key> XML.
- Already downloaded into ${SP}/drone_assets/:
  - cf2x/cf2x.usd: the Crazyflie, metres, Z-up, 10 meshes with primvars:st. It has 10 OmniPBR materials
    under /crazyflie/Looks, but none is bound to a mesh. Its sublayer is
    configuration/cf2x_robot_schema.usd, saved as cf2x/configuration_schema.usd.
  - veg/Boxwood.usd: a 24 MB shrub. Centimetres (metersPerUnit 0.01), Z-up, 2 meshes, 178k faces.
    Its materials are external MDL files ./materials/Boxwood_leaf_Mat.mdl and ../Trees/materials/bark3.mdl.
  - veg/Grass_Short_A.usd: a grass clump, 32k faces. Its MDL shader has USD inputs diffuse_texture,
    normalmap_texture and reflectionroughness_texture pointing to ./materials/textures/*.png.
  - mdl/Brick_Pavers.mdl and mdl/Grass_Cut.mdl: NVIDIA Base materials. They call ::OmniPBR::OmniPBR(...)
    with diffuse_texture: texture_2d("./X/X_BaseColor.png", ::tex::gamma_srgb),
    ORM_texture (R = AO, G = roughness, B = metallic) with enable_ORM_texture: true,
    normalmap_texture: texture_2d("./X/X_N.png", ...), texture_scale: float2(1.f), and so on.
- Put scratch files only under ${SP}/drone_tools_scratch/<your tool>/.

Rules:
- Write ONLY your module and its test file, both named below. Do not touch any other repo file. Do not
  run git commit, git add or git push.
- Code style:
  - Match scripts/heavy_machines/ (read e.g. scripts/heavy_machines/scenarios/bake.py and camera_look.py).
  - from __future__ import annotations, type hints, short docstrings.
  - Comments only where they explain why.
  - No print noise in library functions; accept a log callable instead.
- Tests use unittest and run as: ${PY} scripts/drone_scene/test/<test_file>.py.
  They must put scripts/ on sys.path (sys.path.insert(0, <repo>/scripts)) and import
  drone_scene.<module>.
  Network tests are allowed, but keep total download in tests under 60 MB and cache to a temp dir.
- The module must pass ${PY} -m pyflakes.
- Your final message: what you built (public API with signatures), how you tested it (exact commands and
  results), and known limitations. Be concise.
`

const TOOLS = [
  {
    key: 'nucleus',
    files: 'scripts/drone_scene/nucleus.py and scripts/drone_scene/test/test_nucleus.py',
    prompt: `Build scripts/drone_scene/nucleus.py: fetch and cache NVIDIA Nucleus assets over HTTPS with the
standard library only (urllib). It must also run on the user's Isaac Sim Python, which has no extra
packages except numpy and Pillow. Do not import pxr at module import time.

Public API:
- NUCLEUS_ROOT: str (the URL above).
- url(rel: str) -> str: rel is relative to NUCLEUS_ROOT. Percent-encode each path segment safely,
  because asset names may contain spaces.
- fetch(rel: str, cache_dir: str, *, timeout: float = 120, retries: int = 3, log=None) -> str:
  - Returns cache_dir/rel as a local path.
  - Skips the download if the file already exists and is non-empty.
  - Writes atomically: download to a .part file, then os.replace.
  - Retries with backoff and raises a clear RuntimeError naming the URL on final failure.
  - Streams large files in chunks.
- list_dir(rel_dir: str) -> tuple[list[str], list[str]]: (subdirs, files) from the S3 v2 listing API.
  Handle continuation tokens. Skip ".thumbs".
- mirror_usd(rel: str, cache_dir: str, *, log=None, max_files: int = 500) -> str:
  - Fetch a USD and, recursively, everything it needs, keeping the same relative layout under cache_dir,
    so that pxr opens the local copy with every dependency resolving locally.
  - Discover dependencies with pxr (import inside the function): sublayers, references, payloads and every
    asset-valued attribute such as textures. UsdUtils.ExtractExternalReferences or Sdf layer traversal
    both work; handle relative "./", "../" and absolute-root paths.
  - Also parse every downloaded .mdl file as text:
    - texture_2d("relative/path.png", ...) calls are textures to fetch;
    - relative module imports are other .mdl files to fetch, but only when they resolve to a file next to
      it. Skip core library imports like ::OmniPBR, ::base, ::tex, ::anno, ::state and ::math, which ship
      with Kit.
  - Skip missing optional dependencies with a log line instead of failing (e.g. a texture that 404s),
    but fail if the root USD itself cannot be fetched.
  - Return the local root path.
- local_or_url(rel: str, cache_dir: str | None) -> str: the local path if it is cached, otherwise
  url(rel). Isaac can load URLs directly.

Tests (network):
- url encoding;
- list_dir on NVIDIA/Materials/Base/Masonry/ includes Brick_Pavers.mdl;
- fetch of NVIDIA/Materials/Base/Masonry/Brick_Pavers.mdl is cached on the second call (time it or mock);
- mirror_usd of NVIDIA/Assets/Vegetation/Shrub/Grass_Short_A.usd (about 2.5 MB plus textures):
  - after mirroring, open with pxr and check that every asset path in the stage resolves to an existing
    local file, including the MDL file and the three textures;
- mirror_usd of Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd: its sublayer resolves;
- atomic write: a failing download leaves no partial file (mock urlopen to raise mid-stream).`,
  },
  {
    key: 'hdr',
    files: 'scripts/drone_scene/hdr.py and scripts/drone_scene/test/test_hdr.py',
    prompt: `Build scripts/drone_scene/hdr.py: HDR sky tools in pure numpy (no OpenCV, no imageio). It runs on the
user's Isaac Sim Python, which has numpy and Pillow only.

Purpose: Isaac renders the sky from an equirectangular (lat-long) HDR on a DomeLight. Real-time RTX
gives soft, noisy sun shadows from a dome, so we cut the sun out of the HDR and add a DistantLight with
the same direction and irradiance. We also calibrate the brightness against a reference that Isaac Lab
is known to expose well.

Public API:
- read_hdr(path) -> np.ndarray float32 (H, W, 3):
  - Radiance RGBE (.hdr). Support the "#?RADIANCE" and "#?RGBE" headers, FORMAT=32-bit_rle_rgbe and the
    "-Y H +X W" resolution string.
  - Support both new-style RLE scanlines and flat (uncompressed) scanlines. Apply EXPOSURE= if present.
- write_hdr(path, img): write a flat or RLE RGBE file that read_hdr, and standard readers, read back.
  Round-trip error must be within RGBE precision.
- Direction convention: define ONE convention and document it in the module docstring.
  - Image u in [0,1) left to right, v in [0,1] top (zenith) to bottom (nadir).
  - World is Z-up: the zenith is +Z.
  - Azimuth: u = 0.5 looks along +X, and azimuth increases counter-clockwise seen from above
    (+X towards +Y). Pick the sign so that this exact statement holds.
  - Provide direction_from_uv(u, v) -> unit vector (x,y,z) and uv_from_direction(d), vectorised and
    exact inverses, plus solid_angle_map(h, w): per-pixel solid angle (sum = 4π).
- find_sun(img, *, min_peak_ratio=50) -> SunInfo | None (a dataclass), with fields:
  - direction: unit vector pointing FROM the ground TO the sun, in the convention above;
  - elevation_deg and azimuth_deg;
  - irradiance: RGB irradiance at normal incidence = ∫ L dω over the sun region;
  - solid_angle (sr);
  - mask: bool (H, W) of the sun pixels.
  Detection:
  - Find the brightest region by luminance and grow the mask to the pixels well above the surrounding
    sky (e.g. > k × the local sky level), plus a small dilation for the halo.
  - Return None if there is no clear sun: the peak is < min_peak_ratio × the median sky luminance.
  - The direction is the luminance-weighted centroid in direction space (normalised).
- remove_sun(img, sun) -> img copy with the sun mask filled with plausible sky:
  - Smooth inpainting from the ring around the mask, e.g. iterative diffusion or a fit to the boundary,
    with no hard seam.
  - The result's peak luminance in that region must be within ~2× the surrounding sky.
- irradiance(img, normal=(0,0,1), *, mask=None) -> RGB ∫ L max(0, n·ω) dω. Use it for horizontal
  illuminance.
- luminance(rgb) using Rec.709 weights.
- calibrate(sky_img, sun: SunInfo | None, reference_img, reference_intensity=750.0) -> dict with
  dome_intensity and sun_intensity, computed so that:
  - (a) dome_intensity × (horizontal luminance irradiance of the sunless sky) + sun_intensity × cos(zenith)
    equals reference_intensity × the horizontal luminance irradiance of reference_img;
  - (b) the sun/sky ratio of the original HDR is preserved, i.e. sun_intensity = dome_intensity ×
    luminance(sun.irradiance).
  In other words, Isaac's DistantLight intensity is treated as irradiance (normalised), and the dome
  intensity as a radiance multiplier. Document this assumption.
- downsample(img, factor) by box filter; preview images use smaller skies.

Test data:
- Download NUCLEUS_ROOT + "/NVIDIA/Assets/Skies/Clear/noon_grass_4k.hdr" and
  "/Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr" (each ~20 MB) into a cache
  under ${SP}/drone_tools_scratch/hdr/ with urllib.
- If network tests are slow, downsample in memory after reading, but read the full files once to prove
  the RLE decoder.

Tests:
- read both files: shapes 2048x4096x3, finite, positive;
- write/read round trip on a crop;
- direction/uv inverse property on random samples; solid angles sum to 4π (1e-3 rel);
- find_sun on noon_grass gives a plausible sun:
  - elevation > 30 deg;
  - irradiance luminance > 5× the sky's horizontal luminance irradiance (sunny noon);
  - also print the values;
- synthetic test: a uniform sky of radiance 1 plus a disk of radiance 1e5 at a known direction;
  find_sun recovers the direction within 0.5 deg and the irradiance within 5%; remove_sun brings the
  region back to about 1 with no NaNs;
- irradiance of a uniform sky of radiance 1 with a horizontal normal ≈ π (1% rel);
- calibrate satisfies both equations (a) and (b) to 1e-6 rel.
Also write a tiny CLI (python hdr.py info <file>) printing the sun info and the calibration against the
reference.`,
  },
  {
    key: 'mitsuba',
    files: 'scripts/drone_scene/mitsuba_preview.py and scripts/drone_scene/test/test_mitsuba_preview.py',
    prompt: `Build scripts/drone_scene/mitsuba_preview.py: render a composed USD stage with Mitsuba 3
(mi.set_variant("llvm_ad_rgb"), with a fallback to "scalar_rgb"). The goal is an offline path-traced
preview that matches what Isaac Sim's RTX path tracer shows for the same stage, closely enough to judge
materials, lighting and composition.

Public API:
render(stage, camera: str, out_png: str, *, width=960, height=540, spp=64, max_depth=8, exposure_ev=0.0,
       tonemap="aces", save_exr: str | None = None, log=None) -> dict(stats)
- stage: a pxr Usd.Stage or a path. Assets are local files (references already mirrored locally).
- camera: path of a UsdGeom.Camera prim.

Requirements:
1. Geometry.
   - Every visible, renderable UsdGeom.Mesh. Include instance proxies (Usd.TraverseInstanceProxies) and
     prims under UsdGeom.PointInstancer (one instance per point, using
     ComputeInstanceTransformsAtTime), and honour visibility and purpose (render/default only).
   - World transforms from UsdGeom.XformCache. Referenced assets may be in centimetres. The referencing
     prim carries the scale, so world transforms are the truth.
   - Triangulate polygons by fan, keeping the face-to-triangle map for faceVarying and uniform
     primvars.
   - Normals come from the normals attribute or primvars:normals (vertex or faceVarying, indexed or not).
     When absent, compute smooth normals; subdivisionScheme "none" means flat.
   - UVs come from primvars:st, or any texCoord2f primvar ("st", "st0", "UVMap", "map1"), with
     vertex/faceVarying interpolation, indexed or not.
   - Build per-corner arrays where needed and pass numpy arrays to Mitsuba with
     mi.Mesh(...) + mesh.write_attribute / mi.traverse(mesh) parameters (vertex_positions, faces,
     vertex_normals, vertex_texcoords).
   - Handle GeomSubsets of familyName "materialBind": a different material per subset.
   - Also handle doubleSided, leftHanded orientation (flip winding), and extent-less meshes.
   - Reuse identical prototype meshes through Mitsuba shapegroup + instance when the same mesh appears
     many times (USD instancing or PointInstancer) to keep memory bounded.
2. Materials. Resolve with UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial() (all purposes, then
   preview), then the surface output's connected shader. Support:
   a. UsdPreviewSurface:
      - diffuseColor, roughness, metallic, opacity and opacityThreshold (cutout), normal, emissiveColor,
        and clearcoat (optional, approximate);
      - each input constant, or connected to UsdUVTexture (file, sourceColorSpace sRGB/raw/auto,
        scale/bias, wrapS/T, channel outputs r/g/b/a/rgb) fed by UsdPrimvarReader_float2 (varname), and
        optionally UsdTransform2d.
      Map to Mitsuba "principled" (base_color, roughness, metallic). Wrap it in "normalmap" when a normal
      texture is present and in "mask" for opacity cutouts. Use "twosided" as needed so leaves render
      from both sides. Textures become "bitmap" with raw=True for non-color data and the correct
      filter_type and wrap_mode.
   b. MDL shaders (info:implementationSource = sourceAsset, info:mdl:sourceAsset):
      - OmniPBR.mdl / OmniPBR_Opacity.mdl and any NVIDIA .mdl that calls ::OmniPBR::OmniPBR(...).
      - Read inputs from the USD shader prim first (diffuse_color_constant, diffuse_texture, diffuse_tint,
        albedo_brightness, reflection_roughness_constant, reflectionroughness_texture and its influence,
        metallic_constant, metallic_texture, enable_ORM_texture + ORM_texture (R AO, G roughness,
        B metallic), normalmap_texture, enable_opacity, opacity_constant, opacity_texture,
        opacity_threshold, opacity_mode, texture_scale, texture_translate, texture_rotate,
        emissive_color, enable_emission, emissive_intensity).
      - For any input not authored on the prim, parse the default from the .mdl source text: find the
        file relative to the shader's layer and the sub-identifier material, then regex
        name: value pairs, texture_2d("path", ::tex::gamma_srgb|gamma_linear) and floatN(...).
      - Unknown or custom MDL: fall back to whatever texture-like inputs are found (diffuse, albedo or
        base color names), else a neutral grey of roughness 0.6. Log each fallback once.
      - Note: for OmniPBR, texture_scale multiplies UVs (uv * texture_scale). Check the OmniPBR docs or
        source if you can find them; otherwise document the assumption.
   c. Unbound meshes: displayColor if authored, else grey 0.5, roughness 0.6.
   - Normal map convention:
     - UsdPreviewSurface normal = texture * 2 - 1 in tangent space, OpenGL-style (+Y up).
     - OmniPBR: NVIDIA's normal maps. Pick the convention empirically: mitsuba's normalmap plugin
       expects a specific encoding. Render a test where the normal map is a pure "bump to +U" tilt and
       verify that the shading direction flips as expected. Document what you found and add a
       flip_green option per material type.
3. Lights.
   - UsdLux DomeLight: inputs:texture:file latlong. Radiance = intensity × 2^exposure × color × texture.
     Without a texture it is a constant colour.
     - Orientation convention for a Z-up stage: the texture zenith is +Z, image centre u=0.5 looks along
       +X, azimuth increases counter-clockwise seen from above. This MUST match
       scripts/drone_scene/hdr.py, which another agent is writing in parallel with exactly this
       convention.
     - Honour the light prim's own transform on top of that.
     - Mitsuba "envmap" maps its own convention; work out the to_world transform empirically. Render an
       HDR with a single bright pixel at u=0.5 and v=0.25, check where it lights a test sphere, and add
       this as a unit test.
   - UsdLux DistantLight: emits along its local -Z axis. Irradiance at normal incidence =
     intensity × 2^exposure × color, normalised: the angle does not change the power. Map it to Mitsuba
     "directional" (irradiance). If angle > 0, optionally approximate soft shadows; sharp is fine.
   - SphereLight/RectLight/DiskLight: optional, approximate.
4. Camera.
   - UsdGeom.Camera: horizontalAperture, verticalAperture, focalLength, clippingRange, and optionally
     fStop + focusDistance as a thin lens when fStop > 0.
   - The USD camera looks down -Z with +Y up in camera space. Use the world transform. Mitsuba
     "perspective" with fov_axis "x" from horizontalAperture/focalLength. The aspect ratio comes from
     width/height (if it differs from the apertures, keep the horizontal FOV).
5. Output.
   - Linear HDR, then exposure_ev, then tonemap ("aces" fitted curve, or "none"), then sRGB encode, then
     an 8-bit PNG via Pillow. Optionally save EXR.
   - stats include triangle count, shape count, material count, render seconds and fallbacks used.
6. CLI: python mitsuba_preview.py <stage.usd> --camera /World/Cam --out x.png --width --height --spp.

Tests (no Isaac):
- a synthetic stage written with pxr in a temp dir:
  - a ground quad with a UsdPreviewSurface material and a generated checker texture file;
  - a sphere-like mesh with metallic 1;
  - a DomeLight with a constant colour;
  - a DistantLight;
  - a Camera.
  Render it at 160x90 with 16 spp. Assert the PNG exists, is not all black or all white, and shows the
  checker: the image has strong local contrast in the ground region.
- the camera convention: a red quad placed at +X in front of a camera at the origin looking at +X
  appears centred.
- the DistantLight direction: a light pointing down -Z lights a horizontal quad. Tilting it so it
  grazes reduces the brightness.
- the dome orientation unit test described above.
- ${SP}/drone_assets/cf2x/cf2x.usd referenced at 1:1 under a stage with a camera framing it: renders
  with non-trivial content.
- ${SP}/drone_assets/veg/Grass_Short_A.usd referenced with scale 0.01 under a metre stage: its textures
  load. Its MDL textures are next to it only if mirrored, so first mirror it into a temp dir by fetching
  the textures listed in the USD and the .mdl from NUCLEUS_ROOT + "/NVIDIA/Assets/Vegetation/Shrub/".
  The render shows green-ish pixels.
- Keep the total test time under 5 minutes on 4 cores.
- Save a few test renders as PNGs under ${SP}/drone_tools_scratch/mitsuba/ so the main agent can look
  at them.`,
  },
]

const BUILD = (t) => `${COMMON}

YOUR TOOL: ${t.key}. Files you own: ${t.files}.

${t.prompt}`

const VERIFY = (t, built) => `${COMMON}

You are a skeptical verifier for the tool "${t.key}" (files: ${t.files}). Another agent built it and
reported the following:
---
${built}
---
Your job:
1. Read the module and its tests completely.
2. Run the tests and pyflakes yourself.
3. Try hard to break it. Write extra throwaway checks in ${SP}/drone_tools_scratch/verify_${t.key}/.
   Check:
   - spec compliance;
   - correctness of conventions (directions, units, colour spaces, normal-map encoding, transforms);
   - edge cases;
   - robustness on the real assets;
   - performance;
   - anything that would make the downstream PBR drone scene look wrong.
   Compare against the original spec below. Do NOT edit the module or its tests: report only.

Original spec:
${t.prompt}

Report a list of CONFIRMED problems, each with:
- evidence (a command and its output);
- severity (blocker / major / minor);
- the exact fix you recommend.
Also list what you verified as correct. If nothing is wrong, say so plainly.`

const FIX = (t, built, verdict) => `${COMMON}

YOUR TOOL: ${t.key}. Files you own: ${t.files}.
The module was built. A skeptical verifier then reviewed it.

Builder report:
---
${built}
---
Verifier report:
---
${verdict}
---
Fix every CONFIRMED problem of blocker or major severity, and minor ones when they are cheap and safe.
Re-check each claim against the code first, and skip any claim that turns out to be wrong; say why.
Add or adjust tests for each fix. Run the full test file and pyflakes.

Final message:
- what you changed and why;
- what you skipped and why;
- final test results (command and summary);
- the final public API.`

const results = await pipeline(
  TOOLS,
  (t) => agent(BUILD(t), { label: `build:${t.key}`, phase: 'Build', model: 'sonnet' }),
  (built, t) => agent(VERIFY(t, built), { label: `verify:${t.key}`, phase: 'Verify', model: 'sonnet' })
    .then((verdict) => ({ built, verdict })),
  (r, t) => agent(FIX(t, r.built, r.verdict), { label: `fix:${t.key}`, phase: 'Fix', model: 'sonnet' })
    .then((fixed) => ({ tool: t.key, built: r.built, verdict: r.verdict, fixed })),
)
return results
