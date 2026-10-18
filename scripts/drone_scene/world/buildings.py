"""Buildings from footprints: walls on the real outline, pitched roofs on houses, flat roofs on big buildings
(numpy + shapely).

Input: footprints in the local frame (shapely Polygons) with whatever the map knows: height, floors, roof shape,
building class.  Missing heights come from the class and footprint size (a US house is 1-2 storeys, eaves at
~3 m per storey).

Pitched roofs: the footprint, in its minimum rotated rectangle's frame, is covered with up to four rectangles
(greedy, on a 0.5 m raster); each rectangle gets its own gable or hip roof across its short side, with a 0.4 m
eave overhang.  Overlapping rectangles make the cross-gables of L- and T-shaped houses.  Flat roofs (large,
commercial, industrial or ``roof_shape == flat``) are the footprint triangulated at the wall top.

Output: one ``terrain.MeshData`` with face material ids ``WALL_IDS[k]`` / ``ROOF_IDS[k]`` (k = style variant),
chosen deterministically per building.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .terrain import MeshData

STOREY_M = 3.0
EAVE_OVERHANG = 0.4
FOUNDATION_SINK = 0.3                # walls start this far below the lowest ground point under the footprint

# material slots (the caller maps them to MaterialSpecs): wall styles then roof styles
WALL_STYLES = ["siding_white", "siding_beige", "siding_grey", "siding_blue", "brick_red", "metal_wall"]
ROOF_STYLES = ["shingle_dark", "shingle_brown", "shingle_grey", "metal_roof", "flat_roof"]
WALL_IDS = {k: i for i, k in enumerate(WALL_STYLES)}
ROOF_IDS = {k: len(WALL_STYLES) + i for i, k in enumerate(ROOF_STYLES)}
SLOT_NAMES = WALL_STYLES + ROOF_STYLES

FARM = {"farm", "farm_auxiliary", "barn", "silo", "shed", "agricultural", "greenhouse", "stable", "cowshed"}
COMMERCIAL = {"commercial", "retail", "industrial", "warehouse", "office", "school", "university", "college",
              "hospital", "civic", "public", "government", "church", "supermarket", "service", "kindergarten",
              "sports_hall", "parking", "transportation", "manufacture", "factory", "hangar", "storage_tank"}
SMALL = {"garage", "garages", "shed", "carport", "hut", "kiosk"}


@dataclass
class Footprint:
    polygon: object                  # shapely Polygon, local frame
    height: float | None = None      # total height (m) to the roof top
    floors: int | None = None
    roof_shape: str | None = None    # Overture/OSM: flat, gabled, hipped, ...
    roof_height: float | None = None
    cls: str | None = None           # building class / subtype
    min_height: float | None = None  # bottom of the building above ground (bridges, overhangs)


def _hash01(x: float, y: float, salt: int = 0) -> float:
    h = (int(x * 10) * 73856093) ^ (int(y * 10) * 19349663) ^ (salt * 83492791)
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    return (h & 0xFFFF) / 65535.0


def style(fp: Footprint, area: float) -> dict:
    """Eave height, ridge rise, roof kind and material slots for a footprint."""
    c = (fp.cls or "").lower()
    cx, cy = fp.polygon.centroid.x, fp.polygon.centroid.y
    r1, r2, r3 = _hash01(cx, cy, 1), _hash01(cx, cy, 2), _hash01(cx, cy, 3)
    farm, commercial = c in FARM, c in COMMERCIAL or area > 600
    small = c in SMALL or area < 45
    if fp.floors:
        eave = fp.floors * STOREY_M
    elif small:
        eave = 2.7
    elif commercial:
        eave = 6.0 + 4.0 * r1
    elif farm:
        eave = 4.5 + 2.0 * r1
    else:
        eave = STOREY_M * (1 if r1 < 0.45 else 2) + 0.3
    flat = commercial or (fp.roof_shape or "").lower() in ("flat", "skillion") or area > 900
    pitch_deg = 20.0 if farm else 22.0 + 12.0 * r2
    if fp.height:
        rise_guess = 0.0 if flat else min(3.5, 0.25 * math.sqrt(area))
        eave = max(2.2, fp.height - (fp.roof_height or rise_guess))
    hip = (fp.roof_shape or "").lower() in ("hipped", "hip", "pyramidal") or (fp.roof_shape is None and r3 < 0.35)
    if farm:
        wall, roof = "metal_wall", "metal_roof"
    elif commercial:
        wall = "brick_red" if r1 < 0.4 else "siding_grey"
        roof = "flat_roof"
    else:
        wall = ["siding_white", "siding_beige", "siding_grey", "siding_blue", "brick_red"][int(r2 * 4.999)]
        roof = ["shingle_dark", "shingle_brown", "shingle_grey"][int(r3 * 2.999)]
    if flat and roof != "flat_roof" and not farm:
        roof = "flat_roof"
    return {"eave": eave, "flat": flat, "hip": hip and not flat, "pitch": math.radians(pitch_deg),
            "wall": wall, "roof": roof, "roof_height": fp.roof_height}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

class _Builder:
    def __init__(self):
        self.pts, self.faces, self.uv, self.mat = [], [], [], []
        self.n = 0

    def tri(self, a, b, c, mat, uva, uvb, uvc):
        base = self.n
        self.pts += [a, b, c]
        self.n += 3
        self.faces.append((base, base + 1, base + 2))
        self.uv.append((uva, uvb, uvc))
        self.mat.append(mat)

    def quad(self, a, b, c, d, mat, uvs):
        """a-b-c-d counter-clockwise seen from the front."""
        self.tri(a, b, c, mat, uvs[0], uvs[1], uvs[2])
        self.tri(a, c, d, mat, uvs[0], uvs[2], uvs[3])

    def mesh(self) -> MeshData:
        if not self.faces:
            return MeshData(np.zeros((0, 3)), np.zeros((0, 3), np.int32))
        return MeshData(np.asarray(self.pts, float), np.asarray(self.faces, np.int32), None,
                        np.asarray(self.uv, float), np.asarray(self.mat, np.int32))


def _ring(poly):
    """Exterior ring, counter-clockwise, without the closing point."""
    from shapely.geometry.polygon import orient
    return np.asarray(orient(poly, 1.0).exterior.coords)[:-1, :2]


def obb_frame(poly):
    """(centre (2,), long axis unit (2,), short axis unit (2,), half_long, half_short) of the minimum rotated rect."""
    rect = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)[:4, :2]
    e1, e2 = rect[1] - rect[0], rect[2] - rect[1]
    if np.linalg.norm(e1) < np.linalg.norm(e2):
        e1, e2 = e2, e1
    u = e1 / max(np.linalg.norm(e1), 1e-9)
    v = np.array([-u[1], u[0]])
    c = rect.mean(axis=0)
    return c, u, v, np.linalg.norm(e1) / 2, np.linalg.norm(e2) / 2


def cover_rectangles(poly, cell: float = 0.5, max_rects: int = 6, min_area: float = 3.0, leave: float = 0.02):
    """Up to max_rects rectangles (in the OBB frame: (x0, x1, y0, y1)) covering the footprint greedily."""
    from shapely import contains_xy
    c, u, v, hl, hs = obb_frame(poly)
    xs = np.arange(-hl, hl, cell) + cell / 2
    ys = np.arange(-hs, hs, cell) + cell / 2
    X, Y = np.meshgrid(xs, ys)
    wx, wy = c[0] + X * u[0] + Y * v[0], c[1] + X * u[1] + Y * v[1]
    inside = contains_xy(poly, wx, wy)
    total = inside.sum()
    rects, left = [], inside.copy()
    while len(rects) < max_rects and left.sum() > leave * total:
        best, best_r = 0, None
        # largest all-inside rectangle covering the most uncovered cells (histogram method per row)
        h = np.zeros(len(xs), int)
        for j in range(len(ys)):
            h = np.where(inside[j], h + 1, 0)
            stack = []
            for i in range(len(xs) + 1):
                cur = h[i] if i < len(xs) else 0
                start = i
                while stack and stack[-1][1] >= cur:
                    s, hh = stack.pop()
                    if hh > 0:
                        j0 = j - hh + 1
                        gain = left[j0:j + 1, s:i].sum()
                        if gain > best:
                            best, best_r = gain, (s, i, j0, j + 1)
                    start = s
                stack.append((start, cur))
        if best_r is None or best * cell * cell < min_area:
            break
        i0, i1, j0, j1 = best_r
        left[j0:j1, i0:i1] = False
        rects.append((xs[i0] - cell / 2, xs[i1 - 1] + cell / 2, ys[j0] - cell / 2, ys[j1 - 1] + cell / 2))
    return (c, u, v), rects


def _roof_on_rect(b: _Builder, frame, rect, eave_z: float, pitch: float, hip: bool, roof_mat: int, wall_mat: int,
                  tile=(1.0, 1.0), wall_tile=(1.0, 1.0), floor_z: float = 0.0):
    """Gable or hip roof over an OBB-frame rectangle, ridge along its longer side, with overhang."""
    c, u, v = frame
    x0, x1, y0, y1 = rect
    o = EAVE_OVERHANG
    if (x1 - x0) < (y1 - y0):                      # ridge along the longer side: swap axes
        u, v = v, -u
        x0, x1, y0, y1 = y0, y1, -x1, -x0
    half_w = (y1 - y0) / 2
    ym = (y0 + y1) / 2
    rise = half_w * math.tan(pitch)
    ez = eave_z - o * math.tan(pitch)              # the overhang drops below the wall top
    rz = eave_z + rise

    def W(x, y, z):
        return (c[0] + x * u[0] + y * v[0], c[1] + x * u[1] + y * v[1], z)

    xa, xb = x0 - o, x1 + o
    ya, yb = y0 - o, y1 + o
    inset = half_w + o if hip else 0.0
    inset = min(inset, (xb - xa) / 2 - 0.01)
    ra, rb = xa + inset, xb - inset                # ridge ends
    slope = math.hypot(half_w + o, rise + o * math.tan(pitch))
    tu, tv = tile
    # two long slopes (UV: u along the ridge, v down the slope, in texture repeats)
    b.quad(W(xa, ya, ez), W(xb, ya, ez), W(rb, ym, rz), W(ra, ym, rz), roof_mat,
           [(xa / tu, 0), (xb / tu, 0), (rb / tu, slope / tv), (ra / tu, slope / tv)])
    b.quad(W(xb, yb, ez), W(xa, yb, ez), W(ra, ym, rz), W(rb, ym, rz), roof_mat,
           [(xb / tu, 0), (xa / tu, 0), (ra / tu, slope / tv), (rb / tu, slope / tv)])
    if hip:                                        # hip ends: triangular roof faces
        b.tri(W(xa, yb, ez), W(xa, ya, ez), W(ra, ym, rz), roof_mat, (yb / tu, 0), (ya / tu, 0), (ym / tu, slope / tv))
        b.tri(W(xb, ya, ez), W(xb, yb, ez), W(rb, ym, rz), roof_mat, (ya / tu, 0), (yb / tu, 0), (ym / tu, slope / tv))
    else:                                          # gable ends: wall triangles under the ridge, textured like the walls
        tw, th = wall_tile
        ve, vr = (eave_z - floor_z) / th, (rz - floor_z) / th
        b.tri(W(x0, y1, eave_z), W(x0, y0, eave_z), W(x0, ym, rz), wall_mat, (y1 / tw, ve), (y0 / tw, ve), (ym / tw, vr))
        b.tri(W(x1, y0, eave_z), W(x1, y1, eave_z), W(x1, ym, rz), wall_mat, (y0 / tw, ve), (y1 / tw, ve), (ym / tw, vr))
    return rz


def _flat_remainder(b: _Builder, poly, frame, rects, z: float, mat: int, tile) -> None:
    from shapely.geometry import Polygon
    c, u, v = frame
    covered = []
    for x0, x1, y0, y1 in rects:
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        covered.append(Polygon([(c[0] + x * u[0] + y * v[0], c[1] + x * u[1] + y * v[1]) for x, y in corners]))
    from shapely.ops import unary_union
    rest = poly.difference(unary_union(covered).buffer(0.05))
    parts = [rest] if rest.geom_type == "Polygon" else list(getattr(rest, "geoms", []))
    tu, tv = tile
    for part in parts:
        if part.geom_type != "Polygon" or part.area < 0.5:
            continue
        for t in _triangulate(part):
            a, bb, cc = t
            if (bb[0] - a[0]) * (cc[1] - a[1]) - (bb[1] - a[1]) * (cc[0] - a[0]) < 0:
                bb, cc = cc, bb
            b.tri((a[0], a[1], z), (bb[0], bb[1], z), (cc[0], cc[1], z), mat,
                  (a[0] / tu, a[1] / tv), (bb[0] / tu, bb[1] / tv), (cc[0] / tu, cc[1] / tv))


def _triangulate(poly):
    """Triangles (list of (3, 2) arrays) of a polygon."""
    import shapely
    try:
        tris = shapely.constrained_delaunay_triangles(poly)
        return [np.asarray(t.exterior.coords)[:3, :2] for t in tris.geoms]
    except Exception:  # noqa: BLE001  (older shapely: plain Delaunay, keep the triangles inside)
        from shapely.ops import triangulate
        return [np.asarray(t.exterior.coords)[:3, :2] for t in triangulate(poly) if poly.contains(t.centroid)]


def building_mesh(fp: Footprint, ground, wall_tile=(1.0, 1.0), roof_tile=(1.0, 1.0)) -> tuple[MeshData, dict]:
    """Walls + roof of one building.  ``ground(x, y)`` gives terrain heights.  Returns (mesh, info)."""
    poly = fp.polygon
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    poly = poly.buffer(0)
    area = poly.area
    st = style(fp, area)
    ring = _ring(poly)
    gz = ground(ring[:, 0], ring[:, 1])
    base = float(np.min(gz)) - FOUNDATION_SINK + (fp.min_height or 0.0)
    floor = float(np.max(gz)) + 0.3                # ground floor level, a step above the highest ground
    eave = float(np.max(gz)) + st["eave"]
    wall_mat, roof_mat = WALL_IDS[st["wall"]], ROOF_IDS[st["roof"]]
    b = _Builder()
    tw, th = wall_tile
    # walls along the outline (outward facing: the ring is counter-clockwise)
    s = 0.0
    for k in range(len(ring)):
        p, q = ring[k], ring[(k + 1) % len(ring)]
        L = float(np.hypot(*(q - p)))
        if L < 1e-3:
            continue
        # v = 0 at the ground floor's floor level, so the windows of the facade texture sit at sill height
        v0, v1 = (base - floor) / th, (eave - floor) / th
        b.quad((p[0], p[1], base), (q[0], q[1], base), (q[0], q[1], eave), (p[0], p[1], eave), wall_mat,
               [(s / tw, v0), ((s + L) / tw, v0), ((s + L) / tw, v1), (s / tw, v1)])
        s += L
    top = eave
    if st["flat"] or area < 12.0:
        tu, tv = roof_tile
        z = eave + 0.05
        for t in _triangulate(poly):
            a, bb, cc = t
            if (bb[0] - a[0]) * (cc[1] - a[1]) - (bb[1] - a[1]) * (cc[0] - a[0]) < 0:
                bb, cc = cc, bb
            b.tri((a[0], a[1], z), (bb[0], bb[1], z), (cc[0], cc[1], z), roof_mat,
                  (a[0] / tu, a[1] / tv), (bb[0] / tu, bb[1] / tv), (cc[0] / tu, cc[1] / tv))
    else:
        frame, rects = cover_rectangles(poly)
        if not rects:
            c, u, v, hl, hs = obb_frame(poly)
            frame, rects = (c, u, v), [(-hl, hl, -hs, hs)]
        for r in rects:
            top = max(top, _roof_on_rect(b, frame, r, eave, st["pitch"], st["hip"], roof_mat, wall_mat, roof_tile,
                                         wall_tile, floor))
        # whatever the rectangles leave uncovered (small wings, notches) gets a flat roof at the eaves
        _flat_remainder(b, poly, frame, rects, eave + 0.02, roof_mat, roof_tile)
    return b.mesh(), {"area": area, "eave": eave - float(np.max(gz)), "top": top - float(np.max(gz)),
                      "wall": st["wall"], "roof": st["roof"], "flat": st["flat"], "hip": st["hip"]}


def buildings_mesh(footprints, ground, wall_tile=(1.0, 1.0), roof_tile=(1.0, 1.0), log=None) -> tuple[MeshData, dict]:
    """All buildings in one mesh (material ids = SLOT_NAMES indices) and summary counts."""
    from .usdio import merge
    meshes, stats = [], {"buildings": 0, "flat": 0, "hip": 0, "gable": 0, "skipped": 0}
    for fp in footprints:
        try:
            m, info = building_mesh(fp, ground, wall_tile, roof_tile)
        except Exception as exc:  # noqa: BLE001  (bad geometry in the map: skip that one building)
            stats["skipped"] += 1
            if log:
                log(f"[buildings] skipped one footprint: {exc}")
            continue
        meshes.append(m)
        stats["buildings"] += 1
        stats["flat" if info["flat"] else ("hip" if info["hip"] else "gable")] += 1
    return merge(meshes), stats
