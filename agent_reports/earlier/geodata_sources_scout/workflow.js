export const meta = {
  name: 'scout-world-data',
  description: 'Verify access, formats and tested reading recipes for the open geodata sources of a large real-world drone scene',
  phases: [{ title: 'Scout', detail: 'one agent per data source, tested recipes' }],
}

const SCRATCH = '/tmp/claude-0/-home-user-vla-datagen/39bfe37a-c345-5663-b8fb-e785cd7a1b70/scratchpad/world_scout'
const PY = '/tmp/claude-0/-home-user-vla-datagen/39bfe37a-c345-5663-b8fb-e785cd7a1b70/scratchpad/venv/bin/python'

const COMMON = `Context: we are building a large real-world 3D scene (several km across) for an Isaac Sim drone simulation from open geodata. A pre-processing step (plain Python, runs on the user's Linux PC before Isaac Sim starts) will download data for a bounding box and turn it into USD. Your job is ONLY to scout one data source and produce a tested, minimal Python recipe for reading it for a bounding box. Do not build the scene.

Rules:
- Work ONLY inside your own folder ${SCRATCH}/<your-source-name>/ (create it). Never modify anything under /home/user/vla-datagen.
- Python: ${PY} (has numpy, rasterio 1.4 with GDAL 3.10, pyproj, pyarrow 26, shapely 2, scipy, fsspec, aiohttp, requests, tifffile, Pillow, pxr/usd-core). You may pip install small extra packages into it if truly needed (say which).
- Network: outbound HTTPS goes through a proxy already configured in the environment (HTTPS_PROXY, CURL_CA_BUNDLE, REQUESTS_CA_BUNDLE, SSL_CERT_FILE are set). Reachable: *.s3.amazonaws.com and region S3 hosts (public buckets), pypi. NOT reachable from this machine (but reachable from the user's PC): openstreetmap.org, overpass, polyhaven, ambientcg, element84 STAC API, planetary computer, esri. Do NOT try to work around the proxy or the blocks (no alternative mirrors of blocked hosts, no disabling TLS). If something is blocked, say so and find an S3-hosted route instead.
- GDAL over HTTP: use '/vsicurl/https://...' URLs; set env GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR and CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif,.TIF,.tiff for speed. This was verified to work here for copernicus-dem-30m.
- Keep total downloads under 300 MB. Read windows / row groups, not whole files, wherever possible.
- Test area: a 4 km x 3 km box around lat 42.025, lon -93.690 (west edge of Ames, Iowa, USA: suburbs to farmland). bbox (lon/lat): west -93.715, south 42.012, east -93.665, north 42.038.
- Save your final recipe as ${SCRATCH}/<your-source-name>/recipe.py: importable functions with docstrings, plus a __main__ block that runs on the test area and prints a short summary. Run it and make sure it works.
- In your final structured answer be precise and factual: exact URLs/bucket paths and naming patterns, CRS, resolution, units, nodata, data types, field names, license, size of what you downloaded, time it took, and gotchas. If something did not work, say exactly what and why.`

const SOURCES = [
  {
    key: 'overture',
    prompt: `Data source: Overture Maps Foundation data on S3 (bucket overturemaps-us-west-2, region us-west-2, path release/<latest release>/theme=.../type=.../*.parquet; GeoParquet with a 'bbox' struct column and WKB 'geometry'). Needed types: buildings/building (footprints with height, num_floors, roof_shape, roof_height, class, facade/roof material/colour if present), transportation/segment (roads/paths/rail: class, subtype, road_surface, width if any, geometry linestrings), base/water, base/land_use, base/land_cover, base/infrastructure (if useful), and any tree/vegetation points if Overture has them.
Find the latest release by listing the bucket. Recipe: read only the features in a lon/lat bbox WITHOUT downloading whole files (they are hundreds of MB): e.g. pyarrow.parquet over an HTTP range-reading file object (fsspec HTTPFileSystem on https://overturemaps-us-west-2.s3.us-west-2.amazonaws.com/... honours the proxy) and prune row groups by the bbox column statistics; first find which files overlap the bbox (footer statistics; files are spatially sorted). Measure how many bytes it really fetches. Return per type: feature counts in the test bbox, the useful fields with example values (e.g. how many buildings have height / num_floors / roof_shape), and the recipe. Licence of each theme.`,
  },
  {
    key: 'elevation',
    prompt: `Data source: terrain elevation. Candidates (all on S3): (1) USGS 3DEP 1/3 arc-second seamless DEM tiles: s3 bucket prd-tnm, StagedProducts/Elevation/13/TIFF/current/n<lat>w<lon>/USGS_13_n<lat>w<lon>.tif (check exact naming by listing); (2) USGS 3DEP 1 m DEM: prd-tnm StagedProducts/Elevation/1m/Projects/... - find how to locate the tile(s) covering a point WITHOUT the TNM API if it is blocked (try tnmaccess.nationalmap.gov first and report whether it is reachable; otherwise use bucket listings / any index or footprint file in the bucket); (3) Copernicus DEM GLO-30 on copernicus-dem-30m (global fallback; verified readable); (4) AWS terrain tiles elevation-tiles-prod (terrarium PNG, global). For each: resolution, CRS, vertical datum and units, nodata, whether it is bare-earth (DTM) or surface (DSM, includes trees/buildings), coverage. Recipe: a function that returns a height grid (metres) on a regular local grid for a bbox at a requested spacing, preferring 3DEP 1 m, then 1/3 arc-second, then Copernicus, with reprojection handled (rasterio.warp / WarpedVRT to a local transverse-mercator or UTM CRS). Report for the test area: min/max/mean elevation, relief, timing, bytes.`,
  },
  {
    key: 'landcover_imagery',
    prompt: `Data sources: (1) ESA WorldCover 10 m land cover (bucket esa-worldcover, region eu-central-1, v200/2021/map/ESA_WorldCover_10m_2021_v200_<tile>_Map.tif, 3x3 degree COG tiles): class codes and meaning, recipe to read the class raster for a bbox; class histogram for the test area. (2) Sentinel-2 L2A cloud-optimised GeoTIFFs in bucket sentinel-cogs (region us-west-2), prefix sentinel-s2-l2a-cogs/<utm zone>/<latitude band>/<grid square>/<year>/<month>/<product id>/ with band files (B02/B03/B04 or TCI/visual) and a metadata json per product. The STAC API (earth-search.aws.element84.com) is BLOCKED here, so find scenes by S3 listing: determine the MGRS tile(s) for the test area, list products for summer months of a recent year, read each product's metadata (json) for cloud cover, pick the least cloudy leaf-on scene, and read a true-colour RGB window for the bbox (reprojected to a local metric CRS) with a sensible brightness mapping (e.g. reflectance -> sRGB). Report what files exist per product, cloud metadata fields, which scene you picked and why, and image statistics. Licences.`,
  },
  {
    key: 'canopy',
    prompt: `Data source: Meta / World Resources Institute global 1 m canopy height maps, on S3 bucket dataforgood-fb-data (find the exact prefix, e.g. forests/v1/alsgedi_global_v6_float/chm/<quadkey>.tif, and any tile index geojson). Recipe: (1) find the tile(s) covering a bbox (quadkey math or the index), (2) read a window of canopy height (metres) for the bbox, (3) detect individual trees: local maxima of the smoothed canopy height (e.g. scipy.ndimage maximum_filter with a height-dependent window), with min height ~3 m, returning tree positions (lon/lat and local metres), heights and an estimated crown diameter. Report: resolution, CRS, nodata, data type, value scaling, licence, timing/bytes, and for the test area: number of trees found, height distribution, and how it compares with ESA WorldCover tree class areas if you can (optional). Also note the imagery date / vintage of the canopy data.`,
  },
  {
    key: 'assets3d',
    prompt: `Data source: NVIDIA's public Omniverse asset mirror on S3: https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/ (folders NVIDIA/Assets/..., NVIDIA/Materials/..., Isaac/...). A helper exists: in /home/user/vla-datagen/scripts, 'from drone_scene import nucleus' gives nucleus.list_dir(rel_dir) -> (subdirs, files) and nucleus.url(rel) (READ ONLY; do not modify the repo; run python with cwd /home/user/vla-datagen/scripts). We need assets to dress a real-world suburban/rural landscape seen from a drone: houses or building parts (NVIDIA/Assets/ArchVis/Residential, Commercial, Industrial), fences, cars/vehicles, street furniture (lamps, signs, poles), playground/garden items, rocks, and every tree, shrub, grass and crop-like vegetation asset (NVIDIA/Assets/Vegetation/*), plus ground and building PBR materials (NVIDIA/Materials/Base/*: asphalt, gravel, dirt, grass variants, roof tiles/shingles, brick, siding, concrete, wood, water). Walk these folders (recursively, but with sensible limits) and produce a catalogue: for each useful asset its relative path, what it is, file size (HEAD request), and for a handful of representative ones (2-3 trees, 1 house if any, 1 car if any) open the USD with pxr (use nucleus.mirror_usd only if under ~50 MB total; otherwise read the root layer only) to report units (metersPerUnit), up axis, bounding box size in metres, and whether it is instanceable. For materials: which .mdl files exist per category and their texture sets (albedo/ORM/normal). Also check Isaac/Environments/Outdoor/Rivermark (what it is, size). Return the catalogue grouped by category.`,
  },
]

const RESULT = {
  type: 'object',
  properties: {
    source: { type: 'string' },
    works: { type: 'boolean' },
    recipe_path: { type: 'string' },
    summary: { type: 'string', description: 'what works and what was found, 5-15 lines' },
    details: { type: 'string', description: 'exact paths, naming, CRS, resolution, units, fields, gotchas' },
    license: { type: 'string' },
    test_output: { type: 'string', description: 'key numbers printed by the recipe on the test area' },
    problems: { type: 'string', description: 'what did not work and why; empty if none' },
  },
  required: ['source', 'works', 'recipe_path', 'summary', 'details', 'license', 'test_output', 'problems'],
}

phase('Scout')
const results = await parallel(SOURCES.map(s => () =>
  agent(`${COMMON}\n\nYour source name: ${s.key}\n\n${s.prompt}`, { label: `scout:${s.key}`, phase: 'Scout', schema: RESULT, model: 'sonnet' })))
return results.filter(Boolean)
