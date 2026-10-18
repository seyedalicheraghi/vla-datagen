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

BUILD_VERSION = "3"     # bump when the output changes for the same arguments (cache key)
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
    p.add_argument("--when", default="2025-07-15T15:30:00Z", help="UTC date/time: sets the sun")
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
    blob = repr((BUILD_VERSION, a.region if not (a.takeoff and a.landing) else None, a.takeoff, a.landing, a.distance,
                 a.when, a.half_width, a.margin, a.far, a.horizon, a.cruise_agl, a.clearance, a.max_trees, a.seed))
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def find_site(a, cache, log):
    """(takeoff lat/lon, landing lat/lon, notes) from the arguments or the map."""
    if a.takeoff and a.landing:
        return a.takeoff, a.landing, {"chosen": "given"}
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
    classes = ground_classes(f, rect, 10.0, cache, ov.get("land_use"), water, log)
    canopy = data.canopy(f, rect, 2.0, cache, log)
    if a.takeoff:
        houses = [h for h in houses if h.distance(__import__("shapely").points(0, 0)) < 60] or houses
    st = S.choose(houses, [fp.polygon for fp in fps], rd, water, classes, canopy, a.distance, log=log,
                  takeoff_roads=rd_all)
    lon_t, lat_t = f.to_lonlat(*st.takeoff)
    lon_l, lat_l = f.to_lonlat(*st.landing)
    return (float(lat_t), float(lon_t)), (float(lat_l), float(lon_l)), dict(st.notes, score=st.score)


def ground_classes(frame, rect, res, cache, ov_land_use=None, water=None, log=print, seed=0) -> Grid:
    """Our ground classes on the frame at ``res``: WorldCover upsampled, then Overture land use and water."""
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
    return cls


def _pads(takeoff_z: float, landing_xy, landing_z: float):
    return [T.Pad(0.0, 0.0, takeoff_z, flat_r=14.0, blend_r=30.0),
            T.Pad(landing_xy[0], landing_xy[1], landing_z, flat_r=10.0, blend_r=26.0)]


def ground_materials(classes: Grid, sentinel: Grid | None, specs: dict, faces_xy: np.ndarray, faces_cls: np.ndarray,
                     n_variants: int = 6, seed: int = 0):
    """Material list + per-face material index + per-face UV params, with Sentinel-calibrated tint variants."""
    rng = np.random.default_rng(seed)
    patch = T.patch_ids(faces_xy[:, 0], faces_xy[:, 1], 25.0, seed)
    crop_patch = T.patch_ids(faces_xy[:, 0], faces_xy[:, 1], 45.0, seed + 1)
    patch = np.where(faces_cls == lc.CROP, crop_patch, patch)
    mats, face_mat = [], np.zeros(len(faces_cls), np.int32)
    tile = np.ones(len(faces_cls))
    angle = np.zeros(len(faces_cls))
    for k in np.unique(faces_cls):
        sel = faces_cls == k
        spec = specs[int(k)]
        pid = patch[sel]
        upid, inv = np.unique(pid, return_inverse=True)
        if sentinel is not None and spec.textures.get("albedo"):
            xy = faces_xy[sel]
            rgb = np.column_stack([Grid(sentinel.data[..., c], sentinel.x0, sentinel.y1, sentinel.res).sample(xy[:, 0], xy[:, 1])
                                   for c in range(3)])
            sums = np.zeros((len(upid), 3))
            np.add.at(sums, inv, rgb)
            counts = np.bincount(inv, minlength=len(upid))[:, None]
            patch_rgb = sums / np.maximum(counts, 1)
            tints = np.array([M.calibrate_tints(spec, c, strength=0.85) for c in patch_rgb])
        else:
            lum = 1.0 + 0.08 * rng.standard_normal(len(upid))
            tints = np.asarray(spec.tint)[None, :] * lum[:, None]
        # quantise the patch tints into a few material variants (k-means in log space)
        L = np.log(np.maximum(tints, 1e-3))
        nv = min(n_variants, len(upid))
        cent = L[np.linspace(0, len(L) - 1, nv).astype(int)]
        for _ in range(12):
            lab = np.argmin(((L[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
            cent = np.array([L[lab == c].mean(0) if (lab == c).any() else cent[c] for c in range(nv)])
        base = len(mats)
        for c in range(nv):
            mats.append(spec.with_tint(f"{spec.name}_{c}", tuple(np.exp(cent[c]) / np.asarray(spec.tint))))
        face_mat[sel] = base + lab[inv]
        tile[sel] = spec.tile_m
        if k == lc.CROP:
            ang = {int(p): float(rng.choice([0.0, math.pi / 2])) for p in upid}
            angle[sel] = np.array([ang[int(p)] for p in pid])
        else:
            angle[sel] = (pid * 2654435761 % 6283) / 1000.0
    offset = np.column_stack([(patch * 40503 % 1000) / 1000.0, (patch * 9973 % 1000) / 1000.0])
    return mats, face_mat, tile, angle, offset


def terrain_meshes(height: Grid, fine: Grid | None, classes: Grid, sentinel, specs, rect, route, pads, log,
                   lod=T.Lod(), tile_size=250.0, seed=0):
    """Detail-zone terrain tiles (MeshData list) and their material list."""
    a, b = route
    meshes = []
    for tr in T.tiles(rect, tile_size):
        cell = lod.cell(T.rect_segment_distance(tr, a, b))
        hsrc = fine if (fine is not None and cell <= fine.res * 1.01 and _inside(tr, fine.rect)) else height
        meshes.append(T.tile_mesh(hsrc, tr, cell, pads=pads))
    allm = usdio.merge(meshes)
    cen = T.face_centroids(allm)
    # skirts take the class of the ground right above them
    fcls = classes.sample(cen[:, 0], cen[:, 1], "nearest").astype(np.int32)
    mats, fmat, tile, angle, offset = ground_materials(classes, sentinel, specs, cen[:, :2], fcls, seed=seed)
    allm.material = fmat
    allm.uv = T.face_uvs(allm, tile, angle, offset)
    log(f"[world] terrain: {len(meshes)} tiles, {allm.n_faces / 1e6:.2f} M triangles, {len(mats)} ground materials")
    return allm, mats


def _inside(r, outer):
    return r[0] >= outer[0] and r[1] <= outer[1] and r[2] >= outer[2] and r[3] <= outer[3]


def far_terrain(height: Grid, canopy: Grid | None, sentinel: Grid | None, far_rect, hole, cache_dir, out_dir, cell=32.0,
                tile_size=1000.0):
    """Coarse terrain around the detail zone (with a hole for it), raised by the canopy so forests stand on the
    horizon, coloured by the satellite image."""
    meshes = []
    for tr in T.tiles(far_rect, tile_size):
        if _inside(tr, hole):
            continue
        z_off = (lambda x, y: np.nan_to_num(canopy.sample(x, y)) * 0.85) if canopy is not None else None
        m = T.tile_mesh(height, tr, cell, z_offset=z_off, skirt_m=4.0)
        cen = T.face_centroids(m)
        inside = (cen[:, 0] > hole[0]) & (cen[:, 0] < hole[1]) & (cen[:, 1] > hole[2]) & (cen[:, 1] < hole[3])
        m.faces = m.faces[~inside]
        meshes.append(m)
    far = usdio.merge(meshes)
    if not far.n_faces:
        return None, None
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
    return far, [spec]


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


def landing_grass(landing_xy, landing_z, radius=22.0, seed=0):
    """3D meadow grass clumps around the landing spot: (prototype index, x, y, z, yaw, scale)."""
    rng = np.random.default_rng(seed)
    from ..scene import poisson_disk
    lx, ly = landing_xy
    pts = poisson_disk((lx - radius, lx + radius, ly - radius, ly + radius), 0.55, rng,
                       keep=lambda x, y: math.hypot(x - lx, y - ly) < radius * (0.8 + 0.2 * rng.random()))
    n = len(pts)
    return {"protos": ["grass_a", "grass_b", "grass_c"], "index": rng.integers(0, 3, n),
            "positions": np.column_stack([pts, np.full(n, landing_z - 0.02)]), "yaw": rng.uniform(0, 2 * math.pi, n),
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

    def ground(x, y):
        z = h_detail.sample(x, y)
        return T.apply_pads(x, y, z, pads)

    # map features
    ov = data.overture(frame, far, ["building", "segment", "water", "land_use", "land"], cache, log)
    water = data.water_polygons(ov["water"])
    classes = ground_classes(frame, detail, 2.0, cache, ov["land_use"], water, log, seed=a.seed)
    canopy_d = data.canopy(frame, detail, 1.0, cache, log)
    canopy_f = data.canopy(frame, far, 10.0, cache, log)
    sentinel = data.sentinel(frame, far, 10.0, cache, log)

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
    mh = model_heights(cache, log)
    sp = V.assign_species(trees, seed=a.seed, model_heights=mh)
    tree_inst = V.tree_instances(trees, sp, mh, ground, seed=a.seed)
    per_species = ", ".join(f"{k} {len(v['positions'])}" for k, v in tree_inst.items())
    log(f"[world] trees: {len(trees)} ({n_detected} crowns detected, {len(trees) - n_detected} filled into closed "
        f"canopy; {per_species})")

    # geometry
    specs = M.ground_specs(cache, log=None)
    ter, ter_mats = terrain_meshes(h_detail, h_fine, classes, sentinel, specs, detail, route, pads, log, seed=a.seed)

    def ground_any(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        inside = (x >= detail[0]) & (x <= detail[1]) & (y >= detail[2]) & (y <= detail[3])
        return np.where(inside, ground(x, y), h_far.sample(x, y))

    far_mesh, far_mats = far_terrain(h_far, canopy_f, sentinel, far, detail, cache, out_dir)
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
    far_roads = [r for r in road_list if r.line.intersects(in_detail) or R.road_spec(r)[3] >= 3]
    rmesh, rstats = R.roads_mesh(far_roads, ground_any)
    log(f"[world] buildings: {bstats}; roads: {rstats['segments']} segments, {rstats['length_km']:.0f} km")
    grass = landing_grass((lx, ly), z_land, seed=a.seed)
    cz, cinfo = cruise_height(route, h_detail, canopy_d, footprints, a)

    write_usd(os.path.join(out_dir, "world.usdc"), {
        "terrain": ter, "terrain_mats": ter_mats, "far": far_mesh, "far_mats": far_mats,
        "ring": ring_mesh, "ring_mats": ring_mats,
        "buildings": bmesh, "building_mats": M.building_specs(cache), "roads": rmesh, "road_mats": M.road_specs(cache),
        "trees": tree_inst, "grass": grass}, log)

    when = dt.datetime.fromisoformat(a.when.replace("Z", "+00:00"))
    el, az = sun_position(lat_t, lon_t, when)
    man = {
        "version": BUILD_VERSION, "key": key, "dir": out_dir, "usd": os.path.join(out_dir, "world.usdc"),
        "origin": {"lat": lat_t, "lon": lon_t, "elevation_m": z0, "proj4": frame.proj4},
        "takeoff": {"x": 0.0, "y": 0.0, "z": z_take, "lat": lat_t, "lon": lon_t, "kind": "backyard"},
        "landing": {"x": lx, "y": ly, "z": z_land, "lat": lat_l, "lon": lon_l, "kind": "grass"},
        "distance_m": dist, "heading_deg": heading, "cruise_z": cz, "cruise": cinfo,
        "sun": {"when_utc": when.isoformat(), "elevation_deg": el, "azimuth_compass_deg": az,
                "azimuth_ccw_from_east_deg": compass_to_ccw_from_east(az)},
        "extent": {"detail": detail, "far": far}, "site": site_notes, "terrain_sources": h_info,
        "stats": {"buildings": bstats, "roads_km": round(rstats["length_km"], 1), "trees": int(len(trees)),
                  "terrain_triangles": int(ter.n_faces), "seconds": round(time.time() - t0)},
        "attribution": ATTRIBUTION,
    }
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
                log(f"[world] could not measure {s.key} ({exc}); assuming 15 m")
                known[s.key] = 15.0
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(known, fh, indent=1)
    return known


def main(argv=None):
    a = parse(argv)
    path = build(a, log=lambda m: print(m, flush=True))
    if a.write_path:
        with open(a.write_path, "w", encoding="utf-8") as fh:
            fh.write(path)


if __name__ == "__main__":
    main()
