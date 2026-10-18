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


def ribbon(coords: np.ndarray, width: float, ground, lift: float, step: float, tile: float, mat: int,
           offset: float = 0.0, dashed: tuple | None = None) -> MeshData:
    """A strip of ``width`` centred ``offset`` m left of the polyline.  ``dashed=(on, off)`` makes dashes."""
    P = _resample(np.asarray(coords, float)[:, :2], step)
    if len(P) < 2:
        return MeshData(np.zeros((0, 3)), np.zeros((0, 3), np.int32))
    d = np.gradient(P, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    nrm = np.column_stack([-d[:, 1], d[:, 0]])
    C = P + nrm * offset
    L, R = C + nrm * width / 2, C - nrm * width / 2
    zl, zr = ground(L[:, 0], L[:, 1]) + lift, ground(R[:, 0], R[:, 1]) + lift
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
    pts = np.concatenate([np.column_stack([L, zl]), np.column_stack([R, zr])])
    n = len(P)
    faces, uv = [], []
    for k in range(n - 1):
        if dashed:
            on, off = dashed
            if ((s[k] + s[k + 1]) / 2) % (on + off) > on:
                continue
        l0, l1, r0, r1 = k, k + 1, n + k, n + k + 1
        u0, u1 = s[k] / tile, s[k + 1] / tile
        faces += [(r0, r1, l1), (r0, l1, l0)]
        uv += [((u0, 0), (u1, 0), (u1, width / tile)), ((u0, 0), (u1, width / tile), (u0, width / tile))]
    if not faces:
        return MeshData(np.zeros((0, 3)), np.zeros((0, 3), np.int32))
    return MeshData(pts, np.asarray(faces, np.int32), None, np.asarray(uv, float), np.full(len(faces), mat, np.int32))


def roads_mesh(roads, ground, step: float = 2.0, tile: float = 4.0, log=None) -> tuple[MeshData, dict]:
    """All road ribbons + markings in one mesh (material ids = SURFACES indices) and summary counts."""
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
            meshes.append(ribbon(coords, w, ground, lift, step, tile, SURFACES.index(surf)))
            if mark == "yellow":                       # double yellow centre line
                for off in (-0.12, 0.12):
                    meshes.append(ribbon(coords, 0.10, ground, lift + 0.005, step, 1.0,
                                         SURFACES.index("marking_yellow"), offset=off))
                for off in (w / 2 - 0.35, -(w / 2 - 0.35)):          # white edge lines
                    meshes.append(ribbon(coords, 0.10, ground, lift + 0.005, step, 1.0,
                                         SURFACES.index("marking_white"), offset=off))
            stats["segments"] += 1
            stats["length_km"] += g.length / 1000
            stats["by_class"][r.cls] = stats["by_class"].get(r.cls, 0) + 1
    return merge(meshes), stats
