export const meta = {
  name: 'realism-audit',
  description: 'Static (read-only) audit of the drone scene against real-world appearance; seeds the realism backlog for the local fleet',
  phases: [
    { title: 'Audit', detail: 'five dimension auditors read the code and look up real-world reference values', model: 'sonnet' },
    { title: 'Verify', detail: 'one skeptic per dimension checks file:line claims and reference values', model: 'sonnet' },
    { title: 'Critic', detail: 'completeness critic: what realism aspects were missed', model: 'sonnet' },
  ],
}

const RULES = `HARD RULES (from the owner): you are on a cloud machine. Do NOT run any code, tests, scripts, Python or Isaac Sim, and do not install anything; rendering is only allowed on the owner's PC. Only read files (Read/Grep/Glob) and use the web: load WebSearch and WebFetch with ToolSearch (query "select:WebSearch,WebFetch"). If the proxy blocks a site, do not get around it; use another source. Do NOT edit any file. Never invent a value: every real-world reference number needs a source URL, or say "unsourced".`

const CONTEXT = `Project: /home/user/vla-datagen (clone of the owner's private repo seyedalicheraghi/IsaacSim, an Isaac Lab fork). Read /home/user/vla-datagen/WORK_LOG.md and /home/user/vla-datagen/scripts/drone_scene/README.md first. The drone scene (scripts/drone_scene/) is rendered in NVIDIA Isaac Sim 5.1 RTX (real-time, optional path tracing): an LMF2 quadrotor with two simulated LUCID Triton 3.2 MP cameras (front and back, tilted 20 deg down) flies 2 km straight at about 40 m AGL and 10 m/s over west Ames, Iowa in mid-July at 10:30 solar time. The map is built by scripts/drone_scene/world/ from USGS 3DEP terrain, Overture buildings/roads/water/land use, ESA WorldCover, Sentinel-2 colours and the Meta canopy height map, with NVIDIA Omniverse PBR materials and tree/grass assets; far terrain out to 25 km. The take-off is a hand-built PBR backyard (scene.py, pavers.py). sensor_sim.py turns Replicator HdrColor into LUCID images (lens, auto exposure, noise, Bayer, demosaic). NOTHING has been run in Isaac Sim yet. Goal: make the two camera videos as close as possible to real drone footage of that place. Your audit seeds a backlog that a fleet of agents on the owner's GPU PC will test one item at a time by rendering probe frames, so each item must be concrete and testable.`

const ITEMS = {
  type: 'object',
  properties: {
    items: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'string', description: 'short kebab-case id, prefixed by the dimension key' },
      title: { type: 'string' },
      files: { type: 'array', items: { type: 'string', description: 'repo-relative path:line' } },
      current: { type: 'string', description: 'what the code does now, quoting the value' },
      real_world: { type: 'string', description: 'what real footage / measurements show, with numbers' },
      sources: { type: 'array', items: { type: 'string' } },
      proposed_change: { type: 'string', description: 'concrete code/material/setting change' },
      probe_view: { type: 'string', description: 'which probe view shows it best: takeoff_front, takeoff_back, climb_front, cruise_front, cruise_back, cruise_nadir_chase, horizon_front, landing_front, chase' },
      visual_impact: { type: 'integer', minimum: 1, maximum: 5 },
      effort: { type: 'string', enum: ['S', 'M', 'L'] },
      risk: { type: 'string', description: 'what could break; needs map rebuild (BUILD_VERSION bump)?' },
    }, required: ['id', 'title', 'files', 'current', 'real_world', 'proposed_change', 'probe_view', 'visual_impact', 'effort', 'risk'] } },
    notes: { type: 'string' },
  },
  required: ['items'],
}

const VERDICTS = {
  type: 'object',
  properties: {
    verdicts: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'string' },
      code_claim_accurate: { type: 'boolean', description: 'the file:line exists and the current behaviour is as described' },
      reference_supported: { type: 'string', enum: ['yes', 'partly', 'no', 'unsourced'] },
      keep: { type: 'boolean' },
      correction: { type: 'string' },
    }, required: ['id', 'code_claim_accurate', 'reference_supported', 'keep', 'correction'] } },
  },
  required: ['verdicts'],
}

const DIMS = [
  { key: 'light', prompt: `Dimension: LIGHTING, SKY, ATMOSPHERE, EXPOSURE. Read scripts/drone_scene/rtx_look.py, hdr.py, prepare.py, drone_look.py, assets.py (WORLD_SKY), scene.py light setup, quadcopter_pbr.py scene_cfg/sun setup, world/geo.py sun_position, sensor_sim.py exposure. Check: sun illuminance and sky-to-sun ratio for a clear July morning in Iowa, sun disc angular size and soft shadows, HDR dome rotation vs sun azimuth, sky colour near horizon, AERIAL PERSPECTIVE / haze over 1-25 km (RTX fog or atmosphere settings; real visibility on a clear Iowa summer day ~20-40 km with blue-grey haze), shadow quality settings, ambient occlusion, indirect light/GI in RTX real-time vs path tracing, auto-exposure behaviour when sky is in frame, and anything that would make images look CG.` },
  { key: 'ground', prompt: `Dimension: GROUND, TERRAIN, LAND COVER MATERIALS. Read scripts/drone_scene/world/materials.py, textures.py, landcover.py, terrain.py, build.py (ground_classes, crop_fields, ground_materials, far_terrain, far_canopy, horizon_ring, calibrate_tints), and the backyard pavers.py/scene.py ground. Check against measured broadband and visible albedo (lawn, mown turf, July corn and soybean canopies, bare soil, gravel, concrete, aged asphalt, water), roughness/specular, texture scale and visible tiling/repetition from 40 m and at grazing angles, colour calibration from Sentinel-2 (surface reflectance vs albedo vs sRGB albedo confusion), seams between LOD tiles, far-terrain colour realism, and the look of crop fields in mid-July Iowa (corn about 2 m tall with visible rows/canopy texture, soybeans about 0.5-1 m) from an oblique 40 m view.` },
  { key: 'urban', prompt: `Dimension: BUILDINGS, ROADS, URBAN DETAIL. Read scripts/drone_scene/world/buildings.py, roads.py, textures.py (facade, shingles), materials.py (BUILDING, ROAD, WALL_TILE), data.py (buildings, roads), build.py write_usd/layout. Check: roof colours and materials distribution for Iowa suburbs (asphalt shingles in greys/browns, metal roofs on farm/commercial), roof details (chimneys, vents, skylights, gutters, overhangs), wall materials (vinyl siding colours, brick), windows and doors, garages and driveways, sidewalks, curbs, road surface (aged asphalt colour, cracks, tar snakes, patches, lane markings condition), parking lots with cars, missing objects that real drone footage of a US suburb shows (parked cars, fences, power/utility poles and lines, street lights, mailboxes, play sets, sheds, pools, trampolines), and building-height realism.` },
  { key: 'vegetation', prompt: `Dimension: VEGETATION. Read scripts/drone_scene/world/vegetation.py, assets.py, build.py (trees, landing_grass, model heights), scene.py (backyard shrubs, maple, leaves, 3D grass). Check: species mix vs the real urban/riparian forest of Ames, Iowa (sources: Iowa DNR urban forest inventories, City of Ames street trees, USFS FIA), per-instance variation (rotation, scale, colour/hue jitter, leaf density), tree LOD and how trees look from 40 m and on the horizon, tree placement from the canopy height map (clumping, gaps), lawn grass appearance from altitude (mowing stripes, colour variation, dry patches in July), roadside/ditch vegetation, hedges and shrubs in yards, and leaf material PBR (translucency/subsurface, specular sheen) in RTX.` },
  { key: 'camera', prompt: `Dimension: CAMERA, SENSOR, DRONE AND RENDER SETTINGS, plus the TOOLING the local fleet needs. Read scripts/drone_scene/lucid.py, sensor_sim.py, vehicles.py, flight.py, quadcopter_pbr.py, rtx_look.py. Part A (realism): render resolution vs sensor, supersampling/anti-aliasing (DLSS/TAA ghosting at 10 m/s), motion blur at the real exposure, lens flare/ghosts when the sun is near frame, sensor noise and ISP realism vs real LUCID Triton footage (sharpening, colour matrix, compression of the final mp4), vibration/jitter of a real quadrotor (small high-frequency attitude noise), and drone parts visible in frame. Part B (tooling, very important): list exactly which command-line flags exist in quadcopter_pbr.py (with file:line) and what is MISSING for a fast, deterministic "probe" render that the fleet can run for every candidate change: e.g. rendering stills at chosen flight times (take-off, climb, mid-cruise, horizon, landing) without simulating the whole 4.5-minute flight, a fixed random seed, a time offset to start mid-route, writing named PNGs + per-frame stats JSON, a way to point the map cache at a separate directory (so parallel git worktrees with different world/ code do not share ~/.cache/drone_scene/worlds), and how long each step likely takes. Report Part B items with id prefix camera-tool-.` },
]

phase('Audit')
const results = await pipeline(
  DIMS,
  d => agent(`${RULES}\n\n${CONTEXT}\n\n${d.prompt}\n\nReturn 6-15 items, most visually important first. Quote current values exactly with repo-relative path:line. Prefix ids with "${d.key}-".`, {
    label: `audit:${d.key}`, phase: 'Audit', schema: ITEMS, model: 'sonnet',
  }),
  (r, d) => {
    if (!r) return null
    return agent(`${RULES}\n\nYou are a skeptical reviewer. For each backlog item below, (1) open the cited file:line and check the "current" description is accurate (default code_claim_accurate=false if the line does not show it), (2) check the real-world reference numbers against their sources or a quick search, (3) set keep=false for items that are wrong, already done in the code, or not testable by rendering. Give a correction where needed.\n\n${CONTEXT}\n\nITEMS:\n${JSON.stringify(r.items)}`, {
      label: `verify:${d.key}`, phase: 'Verify', schema: VERDICTS, model: 'sonnet',
    }).then(v => ({ key: d.key, items: r.items, notes: r.notes || '', verdicts: v ? v.verdicts : [] }))
  },
)

const done = results.filter(Boolean)
phase('Critic')
const titles = done.flatMap(d => d.items.map(i => `${i.id}: ${i.title}`)).join('\n')
const critic = await agent(`${RULES}\n\n${CONTEXT}\n\nA realism audit produced the backlog titles below. As a completeness critic, find what is MISSING that a viewer of real 40 m drone footage over an Iowa suburb and farmland in July would notice is wrong or absent in a CG render (think: scale cues, clutter, wear and dirt, colour variety, shadows, water, motion, the horizon, the drone's own parts, the take-off and landing moments, compression). Read the code to make sure each missing item is really missing. Return only new items, same format.\n\nEXISTING TITLES:\n${titles}`, {
  label: 'critic', phase: 'Critic', schema: ITEMS, model: 'sonnet',
})
return { dimensions: done, critic: critic ? critic.items : [] }
