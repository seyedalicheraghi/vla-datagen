"""Build a large real-world scene for the drone (runs before Isaac Sim, see ``runner.py``).

    python -m drone_scene.world.build --out ~/.cache/drone_scene/worlds --cache ~/.cache/drone_scene \\
        [--region 42.025,-93.690] [--takeoff LAT,LON --landing LAT,LON] [--distance 2000] [--when 2025-07-15T15:30Z]

1. Site: a backyard to take off from and open grass ``--distance`` metres away to land on (``site.choose``),
   unless both are given.
2. Local frame centred on the take-off spot (x east, y north, z = height above the take-off ground).
3. Data for the detail zone (the route corridor) and the far zone (out to ``--far`` metres): bare-earth terrain
   (USGS 3DEP lidar / Copernicus), land cover (ESA WorldCover + Overture land use and water), buildings and roads
   (Overture: OpenStreetMap + Microsoft/Google footprints), individual trees (Meta/WRI 1 m canopy height),
   true colour (Sentinel-2, for the far terrain and to calibrate the ground textures).
4. Geometry: terrain tiles with distance-based detail (``terrain``), buildings with roofs (``buildings``), roads
   with markings (``roads``), instanced NVIDIA trees (``vegetation``), 3D grass on the landing meadow.
5. Output: ``<out>/<name>_<key>/world.usdc`` (default prim /Map) and ``world.json`` (frame, take-off, landing,
   cruise height, sun, attribution), reused when the same arguments are asked for again.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "drone_scene.world"

from .. import nucleus  # noqa: E402
from . import buildings as B  # noqa: E402
from . import data  # noqa: E402
from . import landcover as lc  # noqa: E402
from . import materials as M  # noqa: E402
from . import roads as R  # noqa: E402
from . import site as S  # noqa: E402
from . import terrain as T  # noqa: E402
from . import usdio  # noqa: E402
from . import vegetation as V  # noqa: E402
from .geo import Frame, Grid, compass_to_ccw_from_east, sun_position  # noqa: E402

BUILD_VERSION = "6"     # bump when the output changes for the same arguments (cache key)
DEFAULT_REGION = (42.0250, -93.6900)          # west Ames, Iowa: suburbs, parks and farmland
ATTRIBUTION = ("Map data: Overture Maps Foundation (OpenStreetMap contributors ODbL, Microsoft/Esri building "
               "footprints), ESA WorldCover 2021 (CC BY 4.0), Copernicus Sentinel-2 data, USGS 3DEP (public domain) / "
               "Copernicus DEM, Meta & World Resources Institute canopy height (CC BY 4.0). 3D models and materials: "
               "NVIDIA Omniverse.")


def _latlon(s: str) -> tuple[float, float]:
    a, b = (float(v) for v in s.split(","))
    return a, b


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--write-path", default=None, help="write the manifest path here when done")
    p.add_argument("--name", default="world")
    p.add_argument("--region", type=_latlon, default=DEFAULT_REGION, help="lat,lon to search for a route around")
    p.add_argument("--takeoff", type=_latlon, default=None, help="lat,lon of the take-off spot")
    p.add_argument("--landing", type=_latlon, default=None, help="lat,lon of the landing spot")
    p.add_argument("--distance", type=float, default=2000.0)
    p.add_argument("--when", default=None,
                   help="UTC date/time (e.g. 2025-07-15T15:30:00Z): sets the sun.  Default: 10:30 local solar time "
                        "on 15 July (north of the equator) or 15 January (south)")
    p.add_argument("--half-width", type=float, default=600.0, help="detail zone half width around the route (m)")
    p.add_argument("--margin", type=float, default=400.0, help="detail zone beyond take-off and landing (m)")
    p.add_argument("--far", type=float, default=6000.0, help="far terrain half size around the route centre (m)")
    p.add_argument("--horizon", type=float, default=25000.0, help="coarse terrain out to this half size (m)")
    p.add_argument("--cruise-agl", type=float, default=40.0, help="cruise height above the take-off ground (m)")
    p.add_argument("--clearance", type=float, default=15.0, help="minimum height above trees/buildings on the route")
    p.add_argument("--max-trees", type=int, default=60000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--force", action="store_true", help="rebuild even if cached")
    return p.parse_args(argv)


def _key(a) -> str:
    """What changes the geometry.  The sun (--when) and the cruise height only change world.json, which is
    refreshed on a cache hit (``finish_manifest``), so they are not in the key."""
    blob = repr((BUILD_VERSION, a.region if not a.takeoff else None, a.takeoff, a.landing, a.distance,
                 a.half_width, a.margin, a.far, a.horizon, a.max_trees, a.seed))
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def default_when(lat: float, lon: float) -> dt.datetime:
    """10:30 local solar time on 15 July (northern hemisphere) or 15 January (southern), in UTC."""
    day = dt.datetime(2025, 7, 15) if lat >= 0 else dt.datetime(2025, 1, 15)
    return day + dt.timedelta(hours=10.5 - lon / 15.0)


def finish_manifest(man: dict, a, log) -> dict:
    """The parts of world.json that depend on --when, --cruise-agl and --clearance (cheap to recompute)."""
    t = man["takeoff"]
    when = (dt.datetime.fromisoformat(a.when.replace("Z", "+00:00")) if a.when
            else default_when(t["lat"], t["lon"]))
    el, az = sun_position(t["lat"], t["lon"], when)
    if el < 10:
        log(f"[world] WARNING: the sun is only {el:.1f} deg above the horizon at {when.isoformat()} UTC (give --when)")
    man["sun"] = {"when_utc": when.isoformat(), "elevation_deg": el, "azimuth_compass_deg": az,
                  "azimuth_ccw_from_east_deg": compass_to_ccw_from_east(az)}
    obstacle = man["cruise"]["obstacle_top"]
    man["cruise_z"] = max(a.cruise_agl, obstacle + a.clearance)
    return man


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def find_site(a, cache, log):
    """(takeoff lat/lon, landing lat/lon, notes) from the arguments or the map."""
    if a.takeoff and a.landing:
        return a.takeoff, a.landing, {"chosen": "given"}
    if a.landing:
        raise SystemExit("--landing needs --takeoff too (or give neither and let the map choose)")
    lat0, lon0 = a.takeoff or a.region
    f = Frame(lat0, lon0)
    r = a.distance + 500.0 if not a.takeoff else a.distance + 300.0
    rect = (-r, r, -r, r)
    log(f"[world] looking for a route of {a.distance:.0f} m around {lat0:.5f},{lon0:.5f} ...")
    ov = data.overture(f, rect, ["building", "segment", "water", "land_use"], cache, log)
    fps = data.buildings(ov["building"])
    houses = [fp.polygon for fp in fps if (fp.cls or "house") in S.HOUSE_CLASSES and 70 < fp.polygon.area < 450]
    rd_all = [(r.line, R.road_spec(r)[0]) for r in data.roads(ov["segment"])]
    rd = [x for x in rd_all if x[1] >= 3.0]
    water = data.water_polygons(ov["water"])
    classes = ground_classes(f, rect, 10.0, cache, ov.get("land_use"), water, log,
                             streams=data.stream_lines(ov["water"]))
    canopy = data.canopy(f, rect, 2.0, cache, log)
    st = S.choose(houses, [fp.polygon for fp in fps], rd, water, classes, canopy, a.distance, log=log,
                  takeoff_roads=rd_all, fixed_takeoff=(0.0, 0.0) if a.takeoff else None)
    lon_t, lat_t = f.to_lonlat(*st.takeoff)
    lon_l, lat_l = f.to_lonlat(*st.landing)
    return (float(lat_t), float(lon_t)), (float(lat_l), float(lon_l)), dict(st.notes, score=st.score)


def ground_classes(frame, rect, res, cache, ov_land_use=None, water=None, log=print, seed=0, streams=(),
                   sentinel: Grid | None = None) -> Grid:
    """Our ground classes on the frame at ``res``: WorldCover upsampled, then Overture land use, water areas and
    creeks / ditches (water lines buffered to their width).  In an arid region (the satellite's mean colour is
    not green) unwatered 'lawn' and 'meadow' become soil: built-up land there is not a lawn."""
    wc = data.worldcover(frame, rect, cache, log)
    cls = lc.from_worldcover(wc)
    cls = lc.upsample_classes(cls, rect, res, seed=seed) if res < wc.res else cls.resample(rect, res, "nearest")
    if ov_land_use is not None:
        by = {}
        for g, k in data.land_use(ov_land_use):
            by.setdefault(k, []).append(g)
        # big generic areas first, specific ones on top
        for k in (lc.CROP, lc.MEADOW, lc.LAWN, lc.SOIL, lc.CONCRETE):
            lc.burn(cls, by.get(k, []), k)
    if water:
        lc.burn(cls, water, lc.WATER)
    if streams:
        lc.burn(cls, [g.buffer(w / 2) for g, w in streams], lc.WATER)
    if sentinel is not None:
        X, Y = cls.centres()
        rgb = np.stack([Grid(sentinel.data[..., c], sentinel.x0, sentinel.y1, sentinel.res).sample(X, Y)
                        for c in range(3)], -1)
        green = (rgb[..., 1] - rgb[..., 0]) / np.maximum(rgb[..., 1] + rgb[..., 0], 1e-6)
        grassy = np.isin(cls.data, [lc.LAWN, lc.MEADOW])
        if grassy.any() and float(np.median(green[grassy])) < 0.0:
            dry = grassy & (green < 0.02)
            cls.data[dry] = lc.SOIL
            log(f"[world] arid region: {100 * dry.mean():.0f}% of the ground is dry soil rather than grass")
    return cls


def _pads(takeoff_z: float, landing_xy, landing_z: float):
    return [T.Pad(0.0, 0.0, takeoff_z, flat_r=14.0, blend_r=30.0),
            T.Pad(landing_xy[0], landing_xy[1], landing_z, flat_r=10.0, blend_r=26.0)]


def crop_fields(classes: Grid, roads=()):
    """Field id grid: connected crop areas, cut along roads; and each field's row direction (rad), along its
    longest axis (rows run the long way, as they are planted)."""
    from scipy.ndimage import label
    crop = classes.data == lc.CROP
    if roads:
        cut = Grid(np.zeros(classes.shape, np.uint8), classes.x0, classes.y1, classes.res)
        lc.burn(cut, [g.buffer(max(w / 2, classes.res)) for g, w in roads], 1)
        crop &= cut.data == 0
    lab, n = label(crop)
    angles = {}
    if n:
        X, Y = classes.centres()
        idx = lab.ravel()
        cnt = np.bincount(idx, minlength=n + 1).astype(float)
        mx, my = (np.bincount(idx, w, n + 1) / np.maximum(cnt, 1) for w in (X.ravel(), Y.ravel()))
        cxx = np.bincount(idx, (X.ravel() - mx[idx]) ** 2, n + 1)
        cyy = np.bincount(idx, (Y.ravel() - my[idx]) ** 2, n + 1)
        cxy = np.bincount(idx, (X.ravel() - mx[idx]) * (Y.ravel() - my[idx]), n + 1)
        for f in range(1, n + 1):
            angles[f] = 0.5 * math.atan2(2 * cxy[f], cxx[f] - cyy[f])
    return Grid(lab.astype(np.int32), classes.x0, classes.y1, classes.res), angles


def ground_materials(classes: Grid, sentinel: Grid | None, specs: dict, faces_xy: np.ndarray, faces_cls: np.ndarray,
                     n_variants: int = 6, seed: int = 0, fields=None, log=print):
    """Material list + per-face material index + per-face UV params.

    Colour: each class's texture is tinted to the class's real reflectance, the median Sentinel-2 colour over
    faces deep inside that class (>= 90 % of the 22 m around them is the same class, so suburban 'lawn' is not
    calibrated against roofs and trees; for lawn, also only the greenest 30 % of those).  On top, every ~25 m patch (45 m for crops) gets the satellite's local
    brightness relative to the class (0.75-1.3x), quantised into ``n_variants`` materials.  The forest floor is
    not calibrated (the satellite sees the canopy, not the floor): it is the texture, darkened by the shade.
    UVs: crop rows run along each field's long axis, with one phase per field; other classes get a random
    rotation and phase per patch, so no texture repeat lines up across the ground."""
    from scipy.ndimage import uniform_filter
    rng = np.random.default_rng(seed)
    patch = T.patch_ids(faces_xy[:, 0], faces_xy[:, 1], 25.0, seed)
    crop_patch = T.patch_ids(faces_xy[:, 0], faces_xy[:, 1], 45.0, seed + 1)
    patch = np.where(faces_cls == lc.CROP, crop_patch, patch)
    s_rgb = None
    if sentinel is not None:
        s_rgb = np.column_stack([Grid(sentinel.data[..., c], sentinel.x0, sentinel.y1, sentinel.res).sample(
            faces_xy[:, 0], faces_xy[:, 1]) for c in range(3)])
    size = max(3, int(round(22.0 / classes.res)) | 1)
    mats, face_mat = [], np.zeros(len(faces_cls), np.int32)
    tile = np.ones(len(faces_cls))
    angle = np.zeros(len(faces_cls))
    clamped = {}
    for k in np.unique(faces_cls):
        sel = faces_cls == k
        spec = specs[int(k)]
        pid = patch[sel]
        upid, inv = np.unique(pid, return_inverse=True)
        base_tint = np.asarray(spec.tint, float)
        if k == lc.FOREST:
            base_tint = base_tint * 0.55
            lum_var = 1.0 + 0.06 * rng.standard_normal(len(upid))
        elif s_rgb is not None and spec.textures.get("albedo"):
            purity = Grid(uniform_filter((classes.data == k).astype(np.float32), size), classes.x0, classes.y1,
                          classes.res).sample(faces_xy[sel, 0], faces_xy[sel, 1])
            rgb = s_rgb[sel]
            pure = purity >= 0.9
            if k == lc.LAWN and pure.sum() > 50:
                # 'lawn' is also WorldCover's built-up land: even its pure 10 m pixels hold roofs and streets, so
                # take the greenest 30 % of them, the actual grass
                g = (rgb[:, 1] - rgb[:, 0]) / np.maximum(rgb[:, 1] + rgb[:, 0], 1e-6)
                pure &= g >= np.percentile(g[pure], 70)
            target = np.median(rgb[pure] if pure.sum() > 50 else rgb, axis=0)
            want = target / np.maximum(M.mean_albedo(spec.textures["albedo"]), 1e-4)
            base_tint = np.clip(want, 0.05, 8.0)
            clamped[spec.name] = bool((want != base_tint).any())
            lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
            sums = np.bincount(inv, lum, len(upid))
            counts = np.bincount(inv, minlength=len(upid))
            ref = float(target @ np.array([0.2126, 0.7152, 0.0722]))
            lum_var = np.clip((sums / np.maximum(counts, 1)) / max(ref, 1e-6), 0.75, 1.3)
        else:
            lum_var = 1.0 + 0.08 * rng.standard_normal(len(upid))
        # quantise the patch brightness into a few material variants
        nv = min(n_variants, len(upid))
        qs = np.quantile(lum_var, np.linspace(0, 1, nv + 1))
        lab = np.clip(np.searchsorted(qs[1:-1], lum_var, side="right"), 0, nv - 1)
        base = len(mats)
        for c in range(nv):
            level = float(np.median(lum_var[lab == c])) if (lab == c).any() else 1.0
            mats.append(spec.with_tint(f"{spec.name}_{c}", tuple(base_tint * level / np.asarray(spec.tint))))
        face_mat[sel] = base + lab[inv]
        tile[sel] = spec.tile_m
        angle[sel] = (pid * 2654435761 % 6283) / 1000.0
    offset = np.column_stack([(patch * 40503 % 1000) / 1000.0, (patch * 9973 % 1000) / 1000.0])
    if fields is not None and (faces_cls == lc.CROP).any():
        lab_grid, angles = fields
        crop = faces_cls == lc.CROP
        fid = lab_grid.sample(faces_xy[crop, 0], faces_xy[crop, 1], "nearest")
        angle[crop] = np.array([angles.get(int(f), 0.0) for f in fid])
        offset[crop] = np.column_stack([(fid * 40503 % 1000) / 1000.0, (fid * 9973 % 1000) / 1000.0])
    if any(clamped.values()):
        log(f"[world] note: colour calibration hit its limits for {', '.join(k for k, v in clamped.items() if v)}")
    return mats, face_mat, tile, angle, offset


def blended_height(height: Grid, fine: Grid | None, band: float = 5.0):
    """Heights from ``height``, with the finer lidar grid ``fine`` blended in where it exists (fading over
    ``band`` m inside its border)."""
    if fine is None:
        return height.sample
    fx0, fx1, fy0, fy1 = fine.rect

    def h(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        d = np.minimum.reduce([x - fx0, fx1 - x, y - fy0, fy1 - y]) - fine.res
        w = T.smoothstep(0.0, band, d)
        z = height.sample(x, y)
        if (w > 0).any():
            z = np.where(w > 0, z * (1 - w) + fine.sample(x, y) * w, z)
        return z
    return h


def terrain_meshes(height: Grid, fine: Grid | None, classes: Grid, sentinel, specs, rect, route, pads, log,
                   lod=T.Lod(), tile_size=250.0, seed=0, fields=None):
    """Detail-zone terrain: one merged MeshData, its material list, and the ``terrain.Surface`` of its triangles."""
    a, b = route
    hfun = blended_height(height, fine)
    meshes = []
    for tr in T.tiles(rect, tile_size):
        cell = lod.cell(T.rect_segment_distance(tr, a, b))
        meshes.append(T.tile_mesh(hfun, tr, cell, pads=pads))
    surface = T.Surface(meshes, rect, tile_size)
    allm = usdio.merge(meshes)
    cen = T.face_centroids(allm)
    # skirts take the class of the ground right above them
    fcls = classes.sample(cen[:, 0], cen[:, 1], "nearest").astype(np.int32)
    mats, fmat, tile, angle, offset = ground_materials(classes, sentinel, specs, cen[:, :2], fcls, seed=seed,
                                                       fields=fields, log=log)
    allm.material = fmat
    allm.uv = T.face_uvs(allm, tile, angle, offset)
    log(f"[world] terrain: {len(meshes)} tiles, {allm.n_faces / 1e6:.2f} M triangles, {len(mats)} ground materials")
    return allm, mats, surface


def _inside(r, outer):
    return r[0] >= outer[0] and r[1] <= outer[1] and r[2] >= outer[2] and r[3] <= outer[3]


def far_canopy(canopy: Grid | None, buildings, roads, hole, fade: float = 150.0) -> Grid | None:
    """The canopy raise of the far terrain: 0.85 x canopy height, cleared around buildings (+4 m) and roads
    (+3 m beyond their edges) so they are not buried under 'tree' terrain, and faded to 0 approaching the detail
    zone so the far mesh meets the detail terrain at ground level."""
    if canopy is None:
        return None
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    ny, nx = canopy.shape
    shapes = [(p.buffer(4.0), 1) for p in buildings] + [(line.buffer(w / 2 + 3.0), 1) for line, w in roads]
    mask = rasterize(shapes, out_shape=(ny, nx), transform=from_origin(canopy.x0, canopy.y1, canopy.res, canopy.res),
                     fill=0, all_touched=True, dtype="uint8").astype(bool) if shapes else np.zeros((ny, nx), bool)
    X, Y = canopy.centres()
    d = np.maximum.reduce([hole[0] - X, X - hole[1], hole[2] - Y, Y - hole[3]])      # > 0 outside the hole
    raise_ = np.where(mask, 0.0, np.nan_to_num(canopy.data) * 0.85) * T.smoothstep(0.0, fade, d)
    return Grid(raise_.astype(np.float32), canopy.x0, canopy.y1, canopy.res)


def far_terrain(height: Grid, raise_: Grid | None, sentinel: Grid | None, far_rect, hole, cache_dir, out_dir,
                cell=31.25, tile_size=1000.0):
    """Coarse terrain around the detail zone (with a hole for it), raised by the canopy so forests stand on the
    horizon, coloured by the satellite image.  31.25 m cells put vertex lines on every 250 m, so the hole's edges
    (the detail zone, 250 m aligned) fall exactly on them.  Returns (mesh, materials, Surface)."""
    meshes = []
    for tr in T.tiles(far_rect, tile_size):
        if _inside(tr, hole):
            continue
        z_off = raise_.sample if raise_ is not None else None
        m = T.tile_mesh(height, tr, cell, z_offset=z_off, skirt_m=4.0)
        cen = T.face_centroids(m)
        inside = (cen[:, 0] > hole[0]) & (cen[:, 0] < hole[1]) & (cen[:, 1] > hole[2]) & (cen[:, 1] < hole[3])
        m.faces = m.faces[~inside]
        meshes.append(m)
    surface = T.Surface(meshes, far_rect, tile_size)
    far = usdio.merge(meshes)
    if not far.n_faces:
        return None, None, surface
    x0, x1, y0, y1 = far_rect
    P = far.points[far.faces][..., :2]
    far.uv = np.stack([(P[..., 0] - x0) / (x1 - x0), (P[..., 1] - y0) / (y1 - y0)], -1)
    far.material = np.zeros(far.n_faces, np.int32)
    if sentinel is not None:
        from PIL import Image
        rgb = sentinel.data
        srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(np.clip(rgb, 1e-6, 1), 1 / 2.4) - 0.055)
        path = os.path.join(out_dir, "far_albedo.png")
        Image.fromarray((np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)
        # the image covers sentinel.rect; remap UVs to it
        sx0, sx1, sy0, sy1 = sentinel.rect
        far.uv = np.stack([(P[..., 0] - sx0) / (sx1 - sx0), (P[..., 1] - sy0) / (sy1 - sy0)], -1)
        spec = usdio.MaterialSpec("far_ground", {"albedo": path}, 1.0, roughness=0.9, flip_green=False)
    else:
        spec = usdio.MaterialSpec("far_ground", color=(0.07, 0.10, 0.05), roughness=0.9)
    return far, [spec], surface


def horizon_ring(height: Grid, far_rect, ring_rect, albedo_rgb, cell: float = 250.0):
    """Very coarse terrain from the far zone's edge out to ``ring_rect``, so the land reaches the horizon (from 40 m
    up the horizon is ~22 km away); one colour, the far zone's mean satellite albedo."""
    meshes = []
    # 1 km tiles on the far zone's 1 km grid: whole tiles are either inside the far zone (skipped) or outside it
    for tr in T.tiles(ring_rect, 1000.0):
        if _inside(tr, far_rect):
            continue
        meshes.append(T.tile_mesh(height, tr, cell, skirt_m=10.0))
    ring = usdio.merge(meshes)
    if not ring.n_faces:
        return None, None
    ring.material = np.zeros(ring.n_faces, np.int32)
    return ring, [usdio.MaterialSpec("horizon_ground", color=tuple(float(c) for c in albedo_rgb), roughness=0.95)]


def cruise_height(route, ground: Grid, canopy: Grid | None, footprints, a) -> tuple[float, dict]:
    """Absolute cruise z: cruise_agl above the take-off ground, raised to clear everything near the line."""
    import shapely
    (x0, y0), (x1, y1) = route
    t = np.linspace(0, 1, 401)
    xs, ys = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    top = ground.sample(xs, ys)
    if canopy is not None:
        for off in (-20, 0, 20):
            nx, ny = -(y1 - y0), (x1 - x0)
            n = math.hypot(nx, ny)
            top = np.maximum(top, ground.sample(xs + off * nx / n, ys + off * ny / n)
                             + np.nan_to_num(canopy.sample(xs + off * nx / n, ys + off * ny / n)))
    corridor = shapely.LineString(route).buffer(25)
    b_top = 0.0
    for fp in footprints:
        if fp.polygon.intersects(corridor):
            gz = ground.sample(np.array([fp.polygon.centroid.x]), np.array([fp.polygon.centroid.y]))[0]
            b_top = max(b_top, gz + (fp.height or 9.0))
    obstacle = float(max(top.max(), b_top))
    z = max(a.cruise_agl, obstacle + a.clearance)
    return z, {"obstacle_top": round(obstacle, 1), "ground_max": round(float(ground.sample(xs, ys).max()), 1),
               "ground_min": round(float(ground.sample(xs, ys).min()), 1)}


def landing_grass(landing_xy, ground, radius=22.0, seed=0):
    """3D meadow grass clumps around the landing spot, each standing on ``ground(x, y)``."""
    rng = np.random.default_rng(seed)
    from ..scene import poisson_disk
    lx, ly = landing_xy
    pts = poisson_disk((lx - radius, lx + radius, ly - radius, ly + radius), 0.55, rng,
                       keep=lambda x, y: math.hypot(x - lx, y - ly) < radius * (0.8 + 0.2 * rng.random()))
    n = len(pts)
    return {"protos": ["grass_a", "grass_b", "grass_c"], "index": rng.integers(0, 3, n),
            "positions": np.column_stack([pts, ground(pts[:, 0], pts[:, 1]) - 0.02]), "yaw": rng.uniform(0, 2 * math.pi, n),
            "scale": rng.uniform(0.85, 1.25, n)}


# ---------------------------------------------------------------------------
# USD
# ---------------------------------------------------------------------------

def write_usd(path, parts, log):
    from pxr import Gf, Usd, UsdGeom, Vt

    from .. import assets
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Map")
    stage.SetDefaultPrim(root.GetPrim())
    looks = "/Map/Looks"
    usdio.write_mesh(stage, "/Map/Terrain", parts["terrain"], parts["terrain_mats"], looks)
    if parts.get("far") is not None:
        usdio.write_mesh(stage, "/Map/FarTerrain", parts["far"], parts["far_mats"], looks)
    if parts.get("ring") is not None:
        usdio.write_mesh(stage, "/Map/HorizonTerrain", parts["ring"], parts["ring_mats"], looks)
    if parts["buildings"].n_faces:
        usdio.write_mesh(stage, "/Map/Buildings", parts["buildings"], parts["building_mats"], looks)
    if parts["roads"].n_faces:
        usdio.write_mesh(stage, "/Map/Roads", parts["roads"], parts["road_mats"], looks, double_sided=False)
    # trees: one PointInstancer per species, prototypes referenced from NVIDIA's server
    UsdGeom.Scope.Define(stage, "/Map/Trees")
    for key, inst in parts["trees"].items():
        sp = next(s for s in V.SPECIES if s.key == key)
        pi = UsdGeom.PointInstancer.Define(stage, f"/Map/Trees/{key}")
        proto = UsdGeom.Xform.Define(stage, f"/Map/Trees/{key}/Prototypes/tree")
        proto.GetPrim().GetReferences().AddReference(nucleus.url(sp.usd))
        pi.CreatePrototypesRel().SetTargets([proto.GetPath()])
        n = len(inst["positions"])
        pi.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(np.zeros(n, np.int32)))
        pi.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(inst["positions"].astype(np.float32)))
        pi.CreateOrientationsAttr(Vt.QuathArray([Gf.Quath(*map(float, q)) for q in inst["orientations"]]))
        pi.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(inst["scales"].astype(np.float32)))
    g = parts.get("grass")
    if g is not None and len(g["positions"]):
        pi = UsdGeom.PointInstancer.Define(stage, "/Map/LandingMeadow")
        rels = []
        for name in g["protos"]:
            x = UsdGeom.Xform.Define(stage, f"/Map/LandingMeadow/Prototypes/{name}")
            x.AddScaleOp().Set(Gf.Vec3f(*(3 * [assets.PROPS[name].units])))
            UsdGeom.Xform.Define(stage, f"/Map/LandingMeadow/Prototypes/{name}/asset").GetPrim().GetReferences() \
                .AddReference(nucleus.url(assets.PROPS[name].usd))
            rels.append(x.GetPath())
        pi.CreatePrototypesRel().SetTargets(rels)
        pi.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(g["index"].astype(np.int32)))
        pi.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(g["positions"].astype(np.float32)))
        pi.CreateOrientationsAttr(Vt.QuathArray([Gf.Quath(math.cos(a / 2), 0, 0, math.sin(a / 2)) for a in g["yaw"]]))
        pi.CreateScalesAttr(Vt.Vec3fArray([Gf.Vec3f(s, s, s * 0.75) for s in g["scale"]]))
    stage.GetRootLayer().Save()
    log(f"[world] wrote {path} ({os.path.getsize(path) / 1e6:.0f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(a, log=print) -> str:
    key = _key(a)
    out_dir = os.path.join(os.path.expanduser(a.out), f"{a.name}_{key}")
    man_path = os.path.join(out_dir, "world.json")
    if os.path.exists(man_path) and not a.force:
        with open(man_path, encoding="utf-8") as fh:
            man = finish_manifest(json.load(fh), a, log)
        with open(man_path, "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=1)
        log(f"[world] cached: {man_path}")
        return man_path
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.expanduser(a.cache)
    t0 = time.time()

    (lat_t, lon_t), (lat_l, lon_l), site_notes = find_site(a, cache, log)
    f0 = Frame(lat_t, lon_t)
    lx, ly = (float(v) for v in f0.to_local(lon_l, lat_l))
    dist = math.hypot(lx, ly)
    heading = math.degrees(math.atan2(ly, lx))
    route = ((0.0, 0.0), (lx, ly))
    hw, mg = a.half_width, a.margin
    ux, uy = lx / dist, ly / dist
    corners = np.array([[-mg * ux - hw * -uy, -mg * uy - hw * ux], [-mg * ux + hw * -uy, -mg * uy + hw * ux],
                        [lx + mg * ux - hw * -uy, ly + mg * uy - hw * ux], [lx + mg * ux + hw * -uy, ly + mg * uy + hw * ux]])
    detail = (float(np.floor(corners[:, 0].min() / 250) * 250), float(np.ceil(corners[:, 0].max() / 250) * 250),
              float(np.floor(corners[:, 1].min() / 250) * 250), float(np.ceil(corners[:, 1].max() / 250) * 250))
    cx, cy = lx / 2, ly / 2
    far = (float(np.floor((cx - a.far) / 1000) * 1000), float(np.ceil((cx + a.far) / 1000) * 1000),
           float(np.floor((cy - a.far) / 1000) * 1000), float(np.ceil((cy + a.far) / 1000) * 1000))
    log(f"[world] route {dist:.0f} m, heading {heading:.0f} deg; detail zone {detail[1] - detail[0]:.0f} x "
        f"{detail[3] - detail[2]:.0f} m, far terrain {far[1] - far[0]:.0f} x {far[3] - far[2]:.0f} m")

    # terrain heights (absolute), then the frame's z0 = take-off ground
    h_detail, h_info = data.heights(f0, detail, 2.0, cache, log)
    z0 = float(h_detail.sample(np.array([0.0]), np.array([0.0]))[0])
    frame = Frame(lat_t, lon_t, z0)
    h_detail.data -= z0
    fine_rect = (-60.0, 60.0, -60.0, 60.0)
    h_fine, _ = data.heights(frame, fine_rect, 1.0, cache, log)
    h_fine.data -= z0
    h_far, _ = data.heights(frame, far, 10.0, cache, log)
    h_far.data -= z0
    # the take-off pad is flattened to the hand-built backyard's lawn level (scene.SceneCfg.lawn_z), so its patio,
    # curb and mulch bed stand on it and the map's own lawn is the backyard's lawn
    from ..scene import SceneCfg
    z_take = SceneCfg().lawn_z
    z_land = float(h_detail.sample(np.array([lx]), np.array([ly]))[0])
    pads = _pads(z_take, (lx, ly), z_land)

    # map features
    ov = data.overture(frame, far, ["building", "segment", "water", "land_use", "land"], cache, log)
    water = data.water_polygons(ov["water"])
    sentinel = data.sentinel(frame, far, 10.0, cache, log)
    classes = ground_classes(frame, detail, 2.0, cache, ov["land_use"], water, log, seed=a.seed,
                             streams=data.stream_lines(ov["water"]), sentinel=sentinel)
    canopy_d = data.canopy(frame, detail, 1.0, cache, log)
    canopy_f = data.canopy(frame, far, 10.0, cache, log)
    if any(src in ("cop30", "terrarium") for src in h_info["sources"]):
        # a surface model (trees and roofs in it): take the canopy back off, or every wood stands on a mound
        log("[world] WARNING: no bare-earth lidar terrain here; using a surface model with the canopy height "
            "subtracted (expect ~1-3 m errors under woods and towns)")
        for g, c in ((h_detail, canopy_d), (h_fine, canopy_d), (h_far, canopy_f)):
            if c is not None:
                X, Y = g.centres()
                g.data -= (0.8 * c.sample(X, Y)).astype(g.data.dtype)

    # trees
    footprints = data.buildings(ov["building"])
    road_list = data.roads(ov["segment"])
    in_detail = __import__("shapely").box(detail[0], detail[2], detail[1], detail[3])
    blockers = [fp.polygon for fp in footprints if fp.polygon.intersects(in_detail)]
    paved = [(r.line, R.road_spec(r)[0]) for r in road_list if R.road_spec(r)[1] != "gravel" and r.line.intersects(in_detail)]
    clear = [(0.0, 0.0, 9.0), (lx, ly, 32.0)]
    trees = V.detect_trees(canopy_d) if canopy_d is not None else np.zeros((0, 4))
    trees = V.drop_trees(trees, blockers, paved, clear)
    n_detected = len(trees)
    if canopy_d is not None:
        extra = V.drop_trees(V.infill_trees(canopy_d, trees, seed=a.seed), blockers, paved, clear)
        trees = np.concatenate([trees, extra]) if len(extra) else trees
    if len(trees) > a.max_trees:
        trees = trees[np.argsort(-trees[:, 2])[:a.max_trees]]
    # geometry: terrain first, then everything stands on its exact triangles
    specs = M.ground_specs(cache, log=None)
    fields = crop_fields(classes, [(r.line, R.road_spec(r)[0]) for r in road_list if r.line.intersects(in_detail)])
    ter, ter_mats, ground = terrain_meshes(h_detail, h_fine, classes, sentinel, specs, detail, route, pads, log,
                                           seed=a.seed, fields=fields)
    far_buildings = [fp.polygon for fp in footprints if not fp.polygon.within(in_detail)]
    far_road_lines = [(r.line, R.road_spec(r)[0]) for r in road_list if not r.line.within(in_detail)]
    far_mesh, far_mats, far_surface = far_terrain(h_far, far_canopy(canopy_f, far_buildings, far_road_lines, detail),
                                                  sentinel, far, detail, cache, out_dir)
    far_surface.fallback = h_far.sample
    ground.fallback = far_surface

    def ground_any(x, y):
        return ground(x, y)

    mh = model_heights(cache, log)
    sp = V.assign_species(trees, seed=a.seed, model_heights=mh)
    tree_inst = V.tree_instances(trees, sp, mh, ground, seed=a.seed)
    per_species = ", ".join(f"{k} {len(v['positions'])}" for k, v in tree_inst.items())
    log(f"[world] trees: {len(trees)} ({n_detected} crowns detected, {len(trees) - n_detected} filled into closed "
        f"canopy; {per_species})")
    ring_rect = (float(np.floor((cx - a.horizon) / 1000) * 1000), float(np.ceil((cx + a.horizon) / 1000) * 1000),
                 float(np.floor((cy - a.horizon) / 1000) * 1000), float(np.ceil((cy + a.horizon) / 1000) * 1000))
    ring_mesh, ring_mats = None, None
    if a.horizon > a.far + 1000:
        try:
            h_ring, _ = data.heights(frame, ring_rect, 250.0, cache, log)
            h_ring.data -= z0
            mean_albedo = sentinel.data.reshape(-1, 3).mean(0) if sentinel is not None else (0.06, 0.08, 0.05)
            ring_mesh, ring_mats = horizon_ring(h_ring, far, ring_rect, mean_albedo)
        except Exception as exc:  # noqa: BLE001  (the horizon ring is cosmetic)
            log(f"[world] no horizon ring ({type(exc).__name__}: {exc})")
    bmesh, bstats = B.buildings_mesh(footprints, ground_any, M.WALL_TILE, (M.ROOF_TILE, M.ROOF_TILE))
    # all roads and paths in the detail zone, with markings; outside it only real roads (no paths), no markings
    road_tiles = {k: v[1] for k, v in M.ROAD.items()}
    near_roads = [r for r in road_list if r.line.intersects(in_detail)]
    far_only = [r for r in road_list if not r.line.intersects(in_detail) and R.road_spec(r)[3] >= 3]
    rm1, rs1 = R.roads_mesh(near_roads, ground_any, tiles=road_tiles)
    rm2, rs2 = R.roads_mesh(far_only, ground_any, tiles=road_tiles, markings=False)
    rmesh = usdio.merge([rm1, rm2])
    rstats = {"segments": rs1["segments"] + rs2["segments"], "length_km": rs1["length_km"] + rs2["length_km"]}
    log(f"[world] buildings: {bstats}; roads: {rstats['segments']} segments, {rstats['length_km']:.0f} km")
    grass = landing_grass((lx, ly), ground, seed=a.seed)
    _, cinfo = cruise_height(route, h_detail, canopy_d, footprints, a)

    write_usd(os.path.join(out_dir, "world.usdc"), {
        "terrain": ter, "terrain_mats": ter_mats, "far": far_mesh, "far_mats": far_mats,
        "ring": ring_mesh, "ring_mats": ring_mats,
        "buildings": bmesh, "building_mats": M.building_specs(cache), "roads": rmesh, "road_mats": M.road_specs(cache),
        "trees": tree_inst, "grass": grass}, log)

    man = {
        "version": BUILD_VERSION, "key": key, "dir": out_dir, "usd": os.path.join(out_dir, "world.usdc"),
        "origin": {"lat": lat_t, "lon": lon_t, "elevation_m": z0, "proj4": frame.proj4},
        "takeoff": {"x": 0.0, "y": 0.0, "z": z_take, "lat": lat_t, "lon": lon_t, "kind": "backyard"},
        "landing": {"x": lx, "y": ly, "z": z_land, "lat": lat_l, "lon": lon_l, "kind": "grass"},
        "distance_m": dist, "heading_deg": heading, "cruise": cinfo,
        "extent": {"detail": detail, "far": far}, "site": site_notes, "terrain_sources": h_info,
        "stats": {"buildings": bstats, "roads_km": round(rstats["length_km"], 1), "trees": int(len(trees)),
                  "terrain_triangles": int(ter.n_faces), "seconds": round(time.time() - t0)},
        "attribution": ATTRIBUTION,
    }
    man = finish_manifest(man, a, log)
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    log(f"[world] done in {time.time() - t0:.0f} s -> {man_path}")
    return man_path


def measure_model_height(path: str) -> float:
    """Height of a model in its own units: highest mesh point minus the lowest one above 0 (mesh points, not the
    authored extents, which NVIDIA's tree files get wrong in both directions)."""
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(path, Usd.Stage.LoadAll)
    up = 2 if UsdGeom.GetStageUpAxis(st) == "Z" else 1
    xc = UsdGeom.XformCache()
    lo, hi = np.inf, -np.inf
    for prim in Usd.PrimRange(st.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if pts is None or len(pts) == 0:
            continue
        m = np.array(xc.GetLocalToWorldTransform(prim))
        q = np.asarray(pts, float) @ m[:3, :3] + m[3, :3]
        lo, hi = min(lo, float(q[:, up].min())), max(hi, float(q[:, up].max()))
    if not np.isfinite(hi):
        raise ValueError("no mesh points")
    return hi - max(lo, 0.0)


def model_heights(cache, log) -> dict:
    """Height (m) of each tree model at scale 1, measured once from the USD (bounding box x units)."""
    path = os.path.join(cache, "world_data", "tree_heights.json")
    known = {}
    if os.path.exists(path):
        with open(path) as fh:
            known = json.load(fh)
    missing = [s for s in V.SPECIES if s.key not in known]
    fallback = {}
    if missing:
        for s in missing:
            if s.height_m:
                known[s.key] = s.height_m
                continue
            try:
                h = measure_model_height(nucleus.mirror_usd(s.usd, cache, log=None))
                if not 1.0 < h * s.units < 60.0:
                    raise ValueError(f"implausible height {h * s.units:.3g} m")
                known[s.key] = h * s.units
            except Exception as exc:  # noqa: BLE001
                log(f"[world] could not measure {s.key} ({exc}); assuming 15 m for this build only")
                fallback[s.key] = 15.0
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(known, fh, indent=1)                 # measured heights only; fallbacks are retried next time
    return dict(known, **fallback)


def main(argv=None):
    a = parse(argv)
    path = build(a, log=lambda m: print(m, flush=True))
    if a.write_path:
        with open(a.write_path, "w", encoding="utf-8") as fh:
            fh.write(path)


if __name__ == "__main__":
    main()
