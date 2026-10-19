export const meta = {
  name: 'isaac-render-review',
  description: 'Sonnet review of the Isaac-only parts of the new photoreal scenario renderer, each finding verified',
  phases: [
    { title: 'Review', detail: 'Isaac Sim / RTX / USD runtime behaviour', model: 'sonnet' },
    { title: 'Verify', detail: 'skeptic per finding', model: 'sonnet' },
  ],
}
const MODEL = 'sonnet'
const FINDINGS = { type: 'object', properties: { findings: { type: 'array', items: { type: 'object', properties: { severity: { type: 'string', enum: ['bug', 'risk', 'improvement'] }, file: { type: 'string' }, line: { type: 'integer' }, title: { type: 'string' }, description: { type: 'string' }, evidence: { type: 'string' }, fix: { type: 'string' } }, required: ['severity', 'file', 'line', 'title', 'description', 'evidence', 'fix'] } } }, required: ['findings'] }
const VERDICT = { type: 'object', properties: { holds: { type: 'boolean' }, reason: { type: 'string' }, corrected_fix: { type: 'string' } }, required: ['holds', 'reason', 'corrected_fix'] }
const CTX = `Repo /home/user/vla-datagen. Do NOT edit repo files. Isaac Sim 5.1 / Isaac Lab 2.3 are NOT installed here; the Isaac Lab source is at source/isaaclab/isaaclab/ and apps/ (rendering presets). usd-core (pxr 26.08) + numpy are available via ${args.py}.
Code under review: scripts/heavy_machines/scenarios/render_scenario.py (Isaac runner) and scripts/heavy_machines/scenarios/usd_scene.py (USD authoring, already validated offline by scripts/heavy_machines/test/test_usd_scene.py with usd-core — do not re-litigate what that test proves).`
phase('Review')
const res = await agent(`${CTX}

Find what will break or look wrong ONLY at runtime in Isaac Sim's RTX renderer — things the offline test cannot catch. Check against the Isaac Lab source in the repo and the Omniverse/RTX behaviour you can establish with evidence:
1. RTX support for the UsdPreviewSurface network as authored (UsdUVTexture 'scale'/'bias' on a normal map, single-channel 'r' output to roughness, sourceColorSpace raw/sRGB, opacity < 1 with translucency, clearcoat, absolute PNG paths as asset inputs, 'clamp' vs 'repeat').
2. PointInstancer with prototypes authored as children of the instancer: does Omniverse/Hydra render the prototype meshes themselves at the origin in addition to the instances? (USD spec: prototypes beneath the instancer...). If so, the fix (e.g. 'invisible' on a Prototypes scope, or placing prototypes under an 'over'/inactive scope) — be exact.
3. Path tracing settings set via sim.set_setting: '/rtx/rendermode'='PathTracing', '/rtx/pathtracing/spp', '/rtx/pathtracing/totalSpp', '/rtx/pathtracing/optixDenoiser/enabled', '/rtx/pathtracing/maxBounces' — are these the right keys/values for Kit 107 / Isaac Sim 5.x, is SimulationContext.set_setting the right way, and does an Isaac Lab Camera (annotator) receive path-traced, accumulated frames when sim.step + extra sim.render() are called? Is 'accumulation reset' an issue when the scene changes every frame?
4. Camera 'distance_to_camera' data type: shape (N,H,W,1), value for sky pixels (inf?), and whether set_world_poses(convention='world') before sim.step yields the image for that pose in the same frame.
5. DomeLight with an HDR + DistantLight intensities 1000/3000 under the 'quality' preset + path tracing: exposure risk (blown out / too dark) and whether a camera exposure / tonemapper setting should be pinned (cite settings keys).
6. Memory/time: terrain textures (near strip ~48 px/m colour, ~120 px/m relief over 56x23 m; local 24/60 px/m over 130x65 m), 2-3M terrain triangles, ~5k gravel instances, ~135k-triangle machine twins — any hard limits (texture size caps, e.g. 16384), and the bake time on a workstation.
7. Anything else in render_scenario.py that would crash (imports such as isaacsim.core.simulation_manager after AppLauncher, check_file_path behaviour, os._exit flushing).
Report concrete findings with file, line, evidence (quote sources), and exact fixes.`, { label: 'review:isaac-render', phase: 'Review', schema: FINDINGS, model: MODEL })
phase('Verify')
const verified = await parallel((res ? res.findings : []).map((f) => () =>
  agent(`${CTX}

Skeptically verify this finding about the Isaac runtime behaviour of the scenario renderer. Default to holds=false unless the evidence (Isaac Lab source in the repo, USD semantics demonstrable with usd-core, or well-established documented Omniverse behaviour you can cite precisely) supports it. Finding: ${JSON.stringify(f)}
Return holds, reason with evidence, and the corrected exact fix (empty if it does not hold).`,
    { label: `verify:${f.line}`, phase: 'Verify', schema: VERDICT, model: MODEL, effort: 'medium' }).then((v) => ({ ...f, verdict: v }))))
return verified.filter(Boolean)
