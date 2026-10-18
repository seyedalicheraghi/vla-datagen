"""Roads and paths as ribbons draped on the terrain, with lane markings on the bigger roads (numpy + shapely).

Each road segment (a polyline in the local frame) becomes a strip of its class width, resampled every
``step`` metres and lifted ``lift`` above the ground so it never fights with the terrain.  Junction overlaps are
left as they are (the strips are flat and coplanar there, separated by a tiny per-class height offset so the
bigger road wins).  UVs run along the road (u = distance / tile) and across it (v = 0..width / tile).

Widths are typical US values by OpenStreetMap / Overture road class when the map has no width.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .terrain import MeshData

# class -> (width m, surface, centre line marking, priority)
ROAD_CLASSES = {
    "motorway": (11.0, "asphalt", "yellow", 9), "trunk": (10.0, "asphalt", "yellow", 8),
    "primary": (9.0, "asphalt", "yellow", 7), "secondary": (8.0, "asphalt", "yellow", 6),
    "tertiary": (7.0, "asphalt", "yellow", 5), "residential": (7.5, "asphalt", None, 4),
    "unclassified": (6.0, "asphalt", None, 3), "living_street": (6.0, "asphalt", None, 3),
    "service": (4.0, "asphalt", None, 2), "driveway": (3.5, "concrete", None, 2), "parking_aisle": (6.0, "asphalt", None, 2),
    "alley": (4.0, "asphalt", None, 2), "track": (3.0, "gravel", None, 1), "unknown": (5.0, "asphalt", None, 1),
    "footway": (1.6, "concrete", None, 1), "sidewalk": (1.6, "concrete", None, 1), "cycleway": (2.2, "asphalt", None, 1),
    "path": (1.2, "gravel", None, 0), "pedestrian": (3.0, "concrete", None, 1), "steps": (1.5, "concrete", None, 0),
    "bridleway": (2.0, "gravel", None, 0),
}
SURFACES = ["asphalt", "concrete", "gravel", "marking_yellow", "marking_white"]
UNPAVED = {"unpaved", "gravel", "dirt", "ground", "compacted", "fine_gravel", "grass", "sand", "earth"}


@dataclass
class Road:
    line: object                     # shapely LineString, local frame
    cls: str = "unknown"
    surface: str | None = None       # map surface tag (paved / unpaved / gravel ...)
    width: float | None = None


def road_spec(r: Road) -> tuple[float, str, str | None, int]:
    w, surf, mark, prio = ROAD_CLASSES.get(r.cls or "unknown", ROAD_CLASSES["unknown"])
    if r.width and 1.0 < r.width < 40.0:
        w = r.width
    if r.surface and r.surface.lower() in UNPAVED:
        surf, mark = "gravel", None
    return w, surf, mark, prio


def _resample(coords: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return coords[:1]
    n = max(2, int(np.ceil(s[-1] / step)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(t, s, coords[:, 0]), np.interp(t, s, coords[:, 1])])


def stations(coords: np.ndarray, step: float):
    """Resampled centre line (n, 2), unit left normals (n, 2) and distance along it (n,)."""
    P = _resample(np.asarray(coords, float)[:, :2], step)
    if len(P) < 2:
        return P, np.zeros_like(P), np.zeros(len(P))
    d = np.gradient(P, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    nrm = np.column_stack([-d[:, 1], d[:, 0]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return P, nrm, s


def strip(P, nrm, s, offset: float, width: float, z_at, tile: float, mat: int, dashed: tuple | None = None,
          rows=None) -> MeshData:
    """A strip of ``width`` centred ``offset`` m left of the stations.  ``rows``: offsets across it (m, from
    ``offset - width/2`` to ``offset + width/2``; default just the two edges); every row is a line of vertices
    whose heights ``z_at(xy (n, 2), row_offset)`` gives.  ``dashed=(on, off)`` makes dashes."""
    n = len(P)
    if n < 2:
        return MeshData(np.zeros((0, 3)), np.zeros((0, 3), np.int32))
    rows = np.asarray(rows if rows is not None else (offset - width / 2, offset + width / 2), float)
    lines = [np.column_stack([P + nrm * o, z_at(P + nrm * o, o)]) for o in rows]
    pts = np.concatenate(lines)
    keep = np.ones(n - 1, bool)
    if dashed:
        on, off = dashed
        keep = ((s[:-1] + s[1:]) / 2) % (on + off) <= on
    k = np.nonzero(keep)[0]
    faces, uv = [], []
    for r in range(len(rows) - 1):
        a0, a1 = r * n + k, r * n + k + 1                      # row r (right side of this band)
        b0, b1 = (r + 1) * n + k, (r + 1) * n + k + 1          # row r + 1 (left side)
        faces += [np.column_stack([a0, a1, b1]), np.column_stack([a0, b1, b0])]
        u0, u1 = s[k] / tile, s[k + 1] / tile
        va, vb = np.full(len(k), (rows[r] - rows[0]) / tile), np.full(len(k), (rows[r + 1] - rows[0]) / tile)
        uv += [np.stack([np.column_stack([u0, va]), np.column_stack([u1, va]), np.column_stack([u1, vb])], 1),
               np.stack([np.column_stack([u0, va]), np.column_stack([u1, vb]), np.column_stack([u0, vb])], 1)]
    if not len(k):
        return MeshData(np.zeros((0, 3)), np.zeros((0, 3), np.int32))
    F = np.concatenate(faces).astype(np.int32)
    return MeshData(pts, F, None, np.concatenate(uv), np.full(len(F), mat, np.int32))


def cross_rows(width: float, extra=(), max_gap: float = 1.5) -> np.ndarray:
    """Offsets across a road (right edge -w/2 to left edge +w/2), at most ``max_gap`` apart, including ``extra``."""
    base = np.linspace(-width / 2, width / 2, max(2, int(np.ceil(width / max_gap)) + 1))
    r = np.unique(np.round(np.concatenate([base, [e for e in extra if -width / 2 < e < width / 2]]), 4))
    return r


def ribbon(coords: np.ndarray, width: float, ground, lift: float, step: float, tile: float, mat: int,
           offset: float = 0.0, dashed: tuple | None = None) -> MeshData:
    """A strip of ``width`` centred ``offset`` m left of the polyline, lifted ``lift`` above ``ground``."""
    P, nrm, s = stations(coords, step)
    return strip(P, nrm, s, offset, width, lambda xy, o: ground(xy[:, 0], xy[:, 1]) + lift, tile, mat, dashed)


DEFAULT_TILES = {"asphalt": 3.0, "concrete": 1.0, "gravel": 1.5}


def roads_mesh(roads, ground, step: float = 2.0, tiles: dict | None = None, markings: bool = True,
               log=None) -> tuple[MeshData, dict]:
    """All road ribbons + markings in one mesh (material ids = SURFACES indices) and summary counts.  ``tiles``:
    texture repeat (m) per surface (the materials' tile_m).  Markings lie on the road's own surface: their height
    is interpolated between the road edges, not taken from the ground under them."""
    tiles = dict(DEFAULT_TILES, **(tiles or {}))
    from .usdio import merge
    meshes, stats = [], {"segments": 0, "length_km": 0.0, "by_class": {}}
    for r in sorted(roads, key=lambda r: road_spec(r)[3]):
        w, surf, mark, prio = road_spec(r)
        geoms = r.line.geoms if r.line.geom_type.startswith("Multi") else [r.line]
        for g in geoms:
            coords = np.asarray(g.coords)
            if len(coords) < 2:
                continue
            lift = 0.04 + 0.004 * prio                 # bigger roads sit a hair higher at junctions
            P, nrm, s = stations(coords, step)
            if len(P) < 2:
                continue
            marks = []
            if markings and mark == "yellow":           # double yellow centre line, white edge lines
                marks = [(-0.12, "marking_yellow"), (0.12, "marking_yellow"),
                         (w / 2 - 0.35, "marking_white"), (-(w / 2 - 0.35), "marking_white")]
            # the road's cross-section follows the ground (rows <= 1.5 m apart) and has rows exactly under the
            # marking edges, so each marking lies 5 mm above the road surface it is painted on

            def surface(xy, o, lift=lift):
                return ground(xy[:, 0], xy[:, 1]) + lift
            rows = cross_rows(w, [o + d for o, _ in marks for d in (-0.05, 0.05)])
            meshes.append(strip(P, nrm, s, 0.0, w, surface, tiles.get(surf, 3.0), SURFACES.index(surf), rows=rows))
            for off, kind in marks:
                meshes.append(strip(P, nrm, s, off, 0.10, lambda xy, o: surface(xy, o) + 0.005, 1.0,
                                    SURFACES.index(kind)))
            stats["segments"] += 1
            stats["length_km"] += g.length / 1000
            stats["by_class"][r.cls] = stats["by_class"].get(r.cls, 0) + 1
    return merge(meshes), stats
