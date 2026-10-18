"""Map data in the world's local frame: one function per kind of data, each wrapping a ``sources`` reader.

Every function takes the ``geo.Frame`` and a local rect (x0, x1, y0, y1) and returns ``geo.Grid``s or shapely
geometries in local metres, so the rest of the builder never deals with lon/lat or projections.  Downloads are
cached under ``<cache>/world_data``.
"""

from __future__ import annotations

import os

import numpy as np

from .geo import Frame, Grid


def _cache(cache_dir: str, name: str) -> str:
    d = os.path.join(cache_dir, "world_data", name)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Elevation
# ---------------------------------------------------------------------------

def heights(frame: Frame, rect, spacing: float, cache_dir: str, log=print) -> tuple[Grid, dict]:
    """Bare-earth heights (m above the DEM's datum) on the frame, covering rect, at ``spacing``."""
    from rasterio.crs import CRS

    from .sources import elevation
    bbox = frame.bbox_lonlat(rect, pad_m=3 * spacing)
    res = elevation.get_height_grid(bbox, spacing, crs=CRS.from_proj4(frame.proj4),
                                    cache_dir=_cache(cache_dir, "elevation"))
    t = res["transform"]
    g = Grid(res["height"].astype(np.float32), float(t.c), float(t.f), float(spacing))
    info = {"sources": res["source_names"], "seconds": res["seconds"], "mb": round(res["bytes"] / 1e6, 1)}
    log(f"[world] terrain {g.shape[1]}x{g.shape[0]} cells at {spacing:g} m from {', '.join(res['source_names'])} "
        f"({info['mb']} MB, {info['seconds']} s)")
    return g, info


# ---------------------------------------------------------------------------
# Overture: buildings, roads, water, land use, tree points
# ---------------------------------------------------------------------------

def _to_local(frame: Frame, geoms):
    import shapely
    fwd, _ = frame._transformers()

    def f(coords):
        x, y = fwd.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])
    return [shapely.transform(g, f) if g is not None else None for g in geoms]


def overture(frame: Frame, rect, kinds, cache_dir: str, log=print) -> dict:
    """{kind: (pyarrow table, local shapely geoms)} for Overture kinds ('building', 'segment', 'water', ...)."""
    from .sources import overture as ov
    bbox = frame.bbox_lonlat(rect)
    out = {}
    cnt = ov.ByteCounter()
    for kind in kinds:
        tbl = ov.read_bbox(kind, bbox, cache_dir=_cache(cache_dir, "overture"), counter=cnt)
        if tbl is None:                                 # nothing of this kind in the area (desert, open sea, ...)
            import pyarrow as pa
            tbl = pa.table({"geometry": pa.array([], pa.binary())})
            out[kind] = (tbl, [])
            log(f"[world] overture {kind}: none here")
            continue
        geoms = ov.to_shapely(tbl)
        if kind == "segment":
            geoms = ov.clip_to_bbox(geoms, bbox)
        out[kind] = (tbl, _to_local(frame, list(geoms)))
        log(f"[world] overture {kind}: {tbl.num_rows} features")
    log(f"[world] overture: {cnt.bytes / 1e6:.0f} MB in {cnt.requests} requests")
    return out


def _col(tbl, name):
    return tbl.column(name).to_pylist() if name in tbl.column_names else [None] * tbl.num_rows


def buildings(ov_buildings):
    """Overture building table + geoms -> list of buildings.Footprint."""
    from .buildings import Footprint
    tbl, geoms = ov_buildings
    out = []
    cols = {c: _col(tbl, c) for c in ("height", "num_floors", "roof_shape", "roof_height", "class", "subtype",
                                       "min_height", "is_underground")}
    for i, g in enumerate(geoms):
        if g is None or g.is_empty or cols["is_underground"][i]:
            continue
        polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
        for p in polys:
            if p.area < 6.0:
                continue
            out.append(Footprint(p, cols["height"][i], cols["num_floors"][i], cols["roof_shape"][i],
                                 cols["roof_height"][i], cols["class"][i] or cols["subtype"][i], cols["min_height"][i]))
    return out


def roads(ov_segments):
    """Overture segments -> list of roads.Road (roads only; rail and water lines dropped)."""
    from .roads import Road
    tbl, geoms = ov_segments
    sub, cls, subcls, surf = (_col(tbl, c) for c in ("subtype", "class", "subclass", "road_surface"))
    out = []
    for i, g in enumerate(geoms):
        if g is None or g.is_empty or (sub[i] or "road") != "road":
            continue
        c = subcls[i] if subcls[i] in ("driveway", "parking_aisle", "alley", "sidewalk") else (cls[i] or "unknown")
        s = None
        if surf[i]:
            vals = [r.get("value") for r in surf[i] if isinstance(r, dict) and r.get("value")]
            s = vals[0] if vals else None
        out.append(Road(g, c, s))
    return out


def water_polygons(ov_water):
    tbl, geoms = ov_water
    return [g for g in geoms if g is not None and g.geom_type in ("Polygon", "MultiPolygon")]


# typical widths (m) of mapped water lines by class / subtype
STREAM_WIDTH = {"river": 12.0, "canal": 6.0, "stream": 2.5, "drain": 1.5, "ditch": 1.5, "brook": 1.5, "tidal_channel": 6.0}


def stream_lines(ov_water):
    """[(line, width m)] for water mapped as lines (creeks, ditches, rivers too narrow for polygons)."""
    tbl, geoms = ov_water
    sub, cls = _col(tbl, "subtype"), _col(tbl, "class")
    out = []
    for i, g in enumerate(geoms):
        if g is not None and g.geom_type in ("LineString", "MultiLineString"):
            out.append((g, STREAM_WIDTH.get(cls[i] or "", STREAM_WIDTH.get(sub[i] or "", 2.0))))
    return out


def land_use(ov_land_use):
    """[(polygon, our ground class)] for the land-use polygons we understand."""
    from . import landcover as lc
    tbl, geoms = ov_land_use
    sub, cls = _col(tbl, "subtype"), _col(tbl, "class")
    out = []
    for i, g in enumerate(geoms):
        if g is None or g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        k = lc.overture_class(sub[i], cls[i], lc.OVERTURE_LAND_USE)
        if k is not None:
            out.append((g, k))
    return out


def tree_points(ov_land):
    """OpenStreetMap single trees (base/land subtype tree, Points) as (N, 2) local coordinates."""
    tbl, geoms = ov_land
    sub = _col(tbl, "subtype")
    pts = [(g.x, g.y) for i, g in enumerate(geoms) if g is not None and g.geom_type == "Point" and sub[i] == "tree"]
    return np.asarray(pts, float).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Rasters on the local frame: land cover, satellite colour, canopy height
# ---------------------------------------------------------------------------

def _grid_dict(frame: Frame, rect, res: float) -> dict:
    """The sources' grid description for a local rect (north-up, row 0 = north)."""
    from affine import Affine
    from rasterio.crs import CRS
    g = Grid.empty(rect, res)
    ny, nx = g.shape
    return dict(crs=CRS.from_proj4(frame.proj4), epsg=frame.proj4, transform=Affine(res, 0, g.x0, 0, -res, g.y1),
                width=nx, height=ny, res=res, bounds=(g.x0, g.y1 - ny * res, g.x0 + nx * res, g.y1))


def worldcover(frame: Frame, rect, cache_dir: str, log=print) -> Grid:
    """ESA WorldCover 2021 class codes at 10 m on the frame."""
    from .sources import landcover as L
    gd = _grid_dict(frame, rect, 10.0)
    arr = L.worldcover_on_grid(gd)
    g = Grid(arr, gd["transform"].c, gd["transform"].f, 10.0)
    vals, counts = np.unique(arr, return_counts=True)
    def name(v):
        e = L.WC_CLASSES.get(int(v), v)
        return e[0] if isinstance(e, tuple) else e
    top = ", ".join(f"{name(v)} {100 * c / arr.size:.0f}%" for v, c in
                    sorted(zip(vals, counts), key=lambda vc: -vc[1])[:5])
    log(f"[world] land cover (ESA WorldCover 10 m): {top}")
    return g


def sentinel(frame: Frame, rect, res: float, cache_dir: str, log=print, months=None) -> Grid | None:
    """Sentinel-2 L2A surface reflectance (linear RGB, ~albedo) of the clearest leaf-on scene, (ny, nx, 3).
    ``months`` default: June-August north of the equator, December-February south of it.  Cloud, cloud shadow,
    cirrus, snow and no-data pixels (the scene classification, SCL) are filled from their clear neighbours."""
    from .sources import landcover as L
    gd = _grid_dict(frame, rect, res)
    bbox = frame.bbox_lonlat(rect)
    months = months or ((6, 7, 8) if frame.lat0 >= 0 else (12, 1, 2))
    try:
        tiles = L.s2_tiles_for_bbox(bbox)
        cov = [t for t in tiles if t["covers"]]
        groups = [cov] if cov else [[t] for t in tiles]
        refl = np.zeros((gd["height"], gd["width"], 3), np.float32)
        valid = np.zeros((gd["height"], gd["width"]), bool)
        picked = []
        for grp in groups:
            best, _ = L.pick_s2_scene(bbox, gd, grp, months=months, verbose=False)
            if best is None:
                continue
            r = L.read_s2_rgb(best, gd, mode="srgb")
            try:
                from scipy.ndimage import binary_dilation
                bad = np.isin(L.scene_scl(best, gd), L.SCL_BAD)
                r["valid"] &= ~binary_dilation(bad, iterations=1)
            except Exception as exc:  # noqa: BLE001  (no SCL: keep the scene as is)
                log(f"[world] no cloud mask for {best.get('id', '?')} ({exc})")
            fill = ~valid & r["valid"]
            refl[fill] = r["refl"][fill]
            valid |= r["valid"]
            picked.append(best.get("id") or best.get("pid") or "?")
    except Exception as exc:  # noqa: BLE001  (no satellite colour: textures keep their own colour)
        log(f"[world] Sentinel-2 colour unavailable ({type(exc).__name__}: {exc}); using the textures' own colours")
        return None
    if not valid.any():
        log("[world] no clear Sentinel-2 scene found; using the textures' own colours")
        return None
    if not valid.all():                                 # fill holes (cloud masks, tile edges) from neighbours
        from scipy.ndimage import distance_transform_edt
        _, (ii, jj) = distance_transform_edt(~valid, return_indices=True)
        refl = refl[ii, jj]
    log(f"[world] satellite colour: Sentinel-2 {', '.join(map(str, picked))} ({100 * valid.mean():.0f}% clear)")
    return Grid(np.clip(refl, 0, 1), gd["transform"].c, gd["transform"].f, res)


def canopy(frame: Frame, rect, res: float, cache_dir: str, log=print) -> Grid | None:
    """Canopy height above ground (m) on the frame (Meta/WRI CHM v2, ~1 m); None if there is no coverage."""
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    from .sources import canopy as C
    bbox = frame.bbox_lonlat(rect, pad_m=30.0)
    try:
        chm = C.read_chm(bbox, version="v2")
        h = chm.height_m()
        if not np.isfinite(h).any():
            chm = C.read_chm(bbox, version="v1")
            h = chm.height_m()
    except Exception as exc:  # noqa: BLE001
        log(f"[world] canopy height unavailable ({type(exc).__name__}: {exc}); no trees from it")
        return None
    if not np.isfinite(h).any():
        log("[world] no canopy height data here; no trees from it")
        return None
    gd = _grid_dict(frame, rect, res)
    dst = np.full((gd["height"], gd["width"]), np.nan, np.float32)
    reproject(np.nan_to_num(h, nan=0.0).astype(np.float32), dst, src_transform=chm.transform, src_crs=CRS.from_epsg(3857),
              dst_transform=gd["transform"], dst_crs=gd["crs"],
              resampling=Resampling.average if res > 1.6 else Resampling.bilinear)
    g = Grid(np.nan_to_num(dst, nan=0.0), gd["transform"].c, gd["transform"].f, res)
    log(f"[world] canopy height ({chm.version}, {res:g} m): {100 * (g.data > 3).mean():.0f}% of the area under trees "
        f"taller than 3 m, tallest {g.data.max():.0f} m")
    return g
