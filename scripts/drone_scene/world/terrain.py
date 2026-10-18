"""Terrain meshes of a world: height field -> tiled triangle meshes with distance-based detail (numpy only).

The ground is cut into square tiles.  Each tile is a regular grid whose cell size depends on its distance from
the flight line (``Lod``): fine where the onboard cameras see the ground up close, coarse far away.  Neighbouring
tiles of different cell size do not share edge vertices, so every tile gets a *skirt*: a strip hanging
``skirt_m`` straight down from its border, which fills any crack between tiles (the trick terrain engines use).

Faces carry a material id (from a land-cover class grid and a per-patch variant id); every face corner gets
its own texture coordinate (USD faceVarying ``st``) in units of that material's texture repeat, optionally
rotated per patch (crop rows along each field) and offset per patch (so neighbouring patches do not repeat
the same texture phase).

Pads: the take-off and landing spots are flattened to an exact height within ``flat_r`` and blended back to
the real ground by ``blend_r`` (smoothstep), so the hand-built backyard and the landing meadow sit on level
ground.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geo import Grid


@dataclass(frozen=True)
class Pad:
    x: float
    y: float
    z: float                      # ground height on the pad (local frame)
    flat_r: float = 14.0
    blend_r: float = 30.0


@dataclass(frozen=True)
class Lod:
    """Cell size by distance from the flight line: [(max distance m, cell m), ...] ascending, then ``far``."""
    steps: tuple = ((60.0, 1.0), (200.0, 2.0), (450.0, 4.0), (900.0, 8.0))
    far: float = 16.0

    def cell(self, distance: float) -> float:
        for d, c in self.steps:
            if distance <= d:
                return c
        return self.far


@dataclass
class MeshData:
    """Triangles in the local frame.  ``uv`` is faceVarying: one (u, v) per face corner, (M, 3, 2)."""
    points: np.ndarray                         # (N, 3)
    faces: np.ndarray                          # (M, 3) int
    normals: np.ndarray | None = None          # (N, 3) per vertex
    uv: np.ndarray | None = None               # (M, 3, 2)
    material: np.ndarray | None = None         # (M,) int ids into the material list of the caller
    meta: dict = field(default_factory=dict)

    @property
    def n_faces(self) -> int:
        return len(self.faces)


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def apply_pads(x, y, z, pads) -> np.ndarray:
    """Heights with every pad flattened and blended in (pads may overlap; later pads win)."""
    z = np.asarray(z, float).copy()
    for p in pads:
        d = np.hypot(np.asarray(x, float) - p.x, np.asarray(y, float) - p.y)
        w = 1.0 - smoothstep(p.flat_r, p.blend_r, d)
        z = z * (1 - w) + p.z * w
    return z


def segment_distance(x, y, a, b) -> np.ndarray:
    """Distance from points to the segment a-b (2D)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((np.asarray(x) - a[0]) * ab[0] + (np.asarray(y) - a[1]) * ab[1]) / max(float(ab @ ab), 1e-12), 0, 1)
    return np.hypot(np.asarray(x) - (a[0] + t * ab[0]), np.asarray(y) - (a[1] + t * ab[1]))


def rect_segment_distance(rect, a, b) -> float:
    """Smallest distance between an axis-aligned rect and the segment a-b (0 if they touch)."""
    x0, x1, y0, y1 = rect
    xs, ys = np.linspace(x0, x1, 17), np.linspace(y0, y1, 17)
    X, Y = np.meshgrid(xs, ys)
    d = float(segment_distance(X.ravel(), Y.ravel(), a, b).min())
    # the segment may cross the rect between sample points
    for t in np.linspace(0, 1, 65):
        px, py = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        if x0 <= px <= x1 and y0 <= py <= y1:
            return 0.0
    return d


def grid_normals(Z: np.ndarray, cell: float) -> np.ndarray:
    """Vertex normals of a height grid (rows = -y, cols = +x), (ny, nx, 3)."""
    gy, gx = np.gradient(Z, cell)                  # d/drow (= -d/dy), d/dcol (= d/dx)
    n = np.stack([-gx, gy, np.ones_like(Z)], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def tile_mesh(height, rect, cell: float, pads=(), skirt_m: float = 2.0, z_offset=None) -> MeshData:
    """One tile: a (nx+1) x (ny+1) vertex grid over rect, heights from ``height(x, y)`` (vectorised callable or
    Grid), pads applied, plus a skirt along its border.  ``z_offset(x, y)``, if given, is added (e.g. canopy
    height on the far terrain).  Faces alternate their diagonal (no directional artefacts)."""
    x0, x1, y0, y1 = rect
    nx, ny = max(1, int(round((x1 - x0) / cell))), max(1, int(round((y1 - y0) / cell)))
    xs, ys = np.linspace(x0, x1, nx + 1), np.linspace(y1, y0, ny + 1)          # rows go north -> south
    X, Y = np.meshgrid(xs, ys)
    hf = height.sample if isinstance(height, Grid) else height
    Z = hf(X.ravel(), Y.ravel()).reshape(X.shape)
    if pads:
        Z = apply_pads(X, Y, Z, pads)
    if z_offset is not None:
        Z = Z + z_offset(X.ravel(), Y.ravel()).reshape(X.shape)
    N = grid_normals(Z, (x1 - x0) / nx).reshape(-1, 3)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    # two triangles per cell, counter-clockwise seen from above (+z normal)
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    v00 = (j * (nx + 1) + i).ravel()                 # north-west corner
    v01, v10, v11 = v00 + 1, v00 + (nx + 1), v00 + (nx + 1) + 1    # NE, SW, SE
    alt = ((i + j) % 2 == 0).ravel()
    t1 = np.where(alt[:, None], np.column_stack([v00, v10, v11]), np.column_stack([v00, v10, v01]))
    t2 = np.where(alt[:, None], np.column_stack([v00, v11, v01]), np.column_stack([v01, v10, v11]))
    faces = np.concatenate([t1, t2])
    n_ground = len(faces)
    if skirt_m > 0:
        border = np.concatenate([np.arange(nx + 1),                                      # north edge W->E
                                 (np.arange(1, ny + 1) * (nx + 1) + nx),                 # east edge N->S
                                 (ny * (nx + 1) + np.arange(nx - 1, -1, -1)),            # south edge E->W
                                 (np.arange(ny - 1, 0, -1) * (nx + 1))])                 # west edge S->N
        border = np.append(border, border[0])
        base = len(pts)
        low = pts[border].copy()
        low[:, 2] -= skirt_m
        pts = np.concatenate([pts, low])
        N = np.concatenate([N, N[border]])
        k = np.arange(len(border) - 1)
        a, b = border[k], border[k + 1]
        la, lb = base + k, base + k + 1
        # outward-facing quads (the border runs clockwise seen from above)
        faces = np.concatenate([faces, np.column_stack([a, lb, la]), np.column_stack([a, b, lb])])
    return MeshData(pts.astype(np.float64), faces.astype(np.int32), N.astype(np.float32),
                    meta={"cell": (x1 - x0) / nx, "n_ground_faces": n_ground})


def tiles(rect, size: float) -> list[tuple[float, float, float, float]]:
    x0, x1, y0, y1 = rect
    out = []
    for tx in np.arange(x0, x1 - 1e-9, size):
        for ty in np.arange(y0, y1 - 1e-9, size):
            out.append((float(tx), float(min(tx + size, x1)), float(ty), float(min(ty + size, y1))))
    return out


# ---------------------------------------------------------------------------
# Materials on faces
# ---------------------------------------------------------------------------

def face_centroids(mesh: MeshData) -> np.ndarray:
    return mesh.points[mesh.faces].mean(axis=1)


def patch_ids(x, y, size: float, seed: int = 0) -> np.ndarray:
    """Irregular patches ~``size`` across (jittered-grid Voronoi): id of the nearest jittered site."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    gi, gj = np.floor(x / size).astype(np.int64), np.floor(y / size).astype(np.int64)
    best = np.full(x.shape, np.inf)
    out = np.zeros(x.shape, np.int64)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ci, cj = gi + di, gj + dj
            h = _hash(ci, cj, seed)
            sx = (ci + 0.15 + 0.7 * ((h & 0xFFFF) / 65535.0)) * size
            sy = (cj + 0.15 + 0.7 * (((h >> 16) & 0xFFFF) / 65535.0)) * size
            d = (x - sx) ** 2 + (y - sy) ** 2
            better = d < best
            best = np.where(better, d, best)
            out = np.where(better, (ci * 73856093) ^ (cj * 19349663) ^ seed, out)
    return out


def _hash(i, j, seed):
    h = (np.asarray(i, np.int64) * 374761393 + np.asarray(j, np.int64) * 668265263 + seed * 2147483647) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return h ^ (h >> 16)


def face_uvs(mesh: MeshData, tile_m: np.ndarray, angle: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """faceVarying UVs (M, 3, 2): each face's corners in its material's texture repeats, rotated by its patch
    angle (rad) and shifted by its patch offset (repeats).  Skirt faces get the UVs of their top edge."""
    P = mesh.points[mesh.faces][..., :2]                          # (M, 3, 2)
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    u = (c * P[..., 0] + s * P[..., 1]) / tile_m[:, None] + offset[:, None, 0]
    v = (-s * P[..., 0] + c * P[..., 1]) / tile_m[:, None] + offset[:, None, 1]
    return np.stack([u, v], axis=-1)
