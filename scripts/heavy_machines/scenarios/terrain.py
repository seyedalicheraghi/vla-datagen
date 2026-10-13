"""Procedural open-pit quarry terrain (numpy only).

Coordinates: metres, +x along the haul lane (the direction the truck drives),
+y to the left, z up, z = 0 is the lane surface.

The terrain is a sum of smooth features:

* a gently undulating sandy pit floor and a flat, compacted haul lane with ruts;
* a "gate" of two low safety berms across the lane with a gap for the truck;
* a raised loading bench along the left side of the lane (steep earth face);
* a safety berm along the right side that converges, so the lane narrows;
* far away: the stepped benches of the pit walls and low hills beyond the rim.

``QuarryTerrain.height(x, y)`` is vectorised; ``grid_mesh`` turns a rectangle
of it into a triangle mesh with normals and UVs, and ``bake_albedo`` bakes a
colour texture (sand grain, compacted lane with tyre tracks, darker earth
faces, pebbles, slope darkening) that maps 1:1 onto that mesh.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def _hash2(ix, iy, seed):
    """Deterministic pseudo-random in [0, 1) for integer lattice points."""
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(np.float64) / float(0x1000000)


def value_noise(x, y, scale: float, seed: int = 0):
    """Smooth value noise in [-1, 1] with feature size ``scale`` metres."""
    fx, fy = np.asarray(x, dtype=np.float64) / scale, np.asarray(y, dtype=np.float64) / scale
    ix, iy = np.floor(fx), np.floor(fy)
    tx, ty = fx - ix, fy - iy
    tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
    ix, iy = ix.astype(np.int64), iy.astype(np.int64)
    v00, v10 = _hash2(ix, iy, seed), _hash2(ix + 1, iy, seed)
    v01, v11 = _hash2(ix, iy + 1, seed), _hash2(ix + 1, iy + 1, seed)
    v = (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty
    return 2.0 * v - 1.0


def fbm(x, y, scale: float, octaves: int = 4, seed: int = 0, gain: float = 0.5):
    out, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        out = out + amp * value_noise(x, y, scale / (2 ** o), seed + 101 * o)
        norm += amp
        amp *= gain
    return out / norm


# ---------------------------------------------------------------------------
# Terrain description
# ---------------------------------------------------------------------------

def berm_profile(t, t0: float = 0.18):
    """Dumped-berm cross-section: rounded crest for |t| < t0, straight flanks to the toe at t = 1."""
    t = np.clip(np.abs(t), 0.0, 1.0)
    a = 1.0 / (t0 * (1.0 - t0 / 2.0))
    m = a * t0
    return np.where(t < t0, 1.0 - 0.5 * a * t * t, m * (1.0 - t))


@dataclass
class Berm:
    """Straight earth berm from (x0, y0) to (x1, y1) with rounded ends."""
    x0: float
    y0: float
    x1: float
    y1: float
    height: float = 1.0
    base_width: float = 3.0

    def height_at(self, x, y):
        dx, dy = self.x1 - self.x0, self.y1 - self.y0
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        px, py = x - self.x0, y - self.y0
        along = px * ux + py * uy
        across = -px * uy + py * ux
        half = self.base_width / 2
        # distance outside the segment along its axis → rounded ends
        beyond = np.maximum(np.maximum(-along, along - length), 0.0)
        d = np.sqrt(across ** 2 + beyond ** 2)
        return self.height * berm_profile(d / half)


@dataclass
class QuarryTerrain:
    seed: int = 7
    # haul lane (z = 0, compacted) — left edge is the bench toe
    lane_left_y: float = 5.0
    # right safety berm along the lane: its centre line moves from y0 to y1 between x0 and x1
    right_berm: tuple[float, float, float, float] = (-9.5, -9.0, 4.0, -5.8)    # x0, y0, x1, y1
    right_berm_end_x: float = 60.0
    right_berm_height: float = 1.0           # safety berms ≈ wheel-centre height of the largest truck
    # gate of two berms across the lane
    gate_x: float = -11.0
    gate_gap: float = 7.5
    gate_len: float = 9.5
    gate_height: float = 1.0
    # loading bench on the left
    bench_start_x: float = -3.0
    bench_end_x: float = 75.0
    bench_height: float = 2.5
    bench_face_run: float = 2.2          # horizontal run of the steep face
    # pit
    pit_center: tuple[float, float] = (60.0, -40.0)
    pit_radii: tuple[float, float] = (300.0, 230.0)
    pit_bench_height: float = 12.0
    pit_bench_spacing: float = 34.0
    pit_benches: int = 5
    tracks_y: tuple[float, ...] = (1.16, -1.16)   # tyre-track centres in the lane (ADT gauge 2.32 m)
    flat_pads: list[tuple[float, float, float, float]] = field(default_factory=list)  # x0,x1,y0,y1: no top noise
    extra_tracks: list[float] = field(default_factory=lambda: [2.6, 0.2, -2.9])

    # --- feature pieces -------------------------------------------------------
    def right_berm_center(self, x):
        x0, y0, x1, y1 = self.right_berm
        return y0 + (y1 - y0) * smoothstep(x0, x1, x)

    def lane_mask(self, x, y):
        """1 on the compacted haul lane, 0 off it (soft edges)."""
        left = 1.0 - smoothstep(self.lane_left_y - 0.6, self.lane_left_y + 0.2, y)
        right_edge = self.right_berm_center(x) + 1.3
        right = smoothstep(right_edge - 0.2, right_edge + 0.8, y)
        start = smoothstep(self.gate_x - 30.0, self.gate_x - 20.0, x)   # lane fades in from far behind
        return left * right * np.maximum(start, 0.0)

    def bench_toe(self, x):
        """Toe line of the loading bench: a blast-cut face is never ruler-straight."""
        return (self.lane_left_y + 0.22 * value_noise(x, np.zeros_like(x), 7.0, self.seed + 31)
                + 0.10 * value_noise(x, np.zeros_like(x), 1.9, self.seed + 32) + 0.12)

    def bench(self, x, y):
        nose = smootherstep(self.bench_start_x - 3.5, self.bench_start_x, x)
        tail = 1.0 - smootherstep(self.bench_end_x, self.bench_end_x + 6.0, x)
        run = self.bench_face_run * (1.0 + 0.25 * value_noise(x, np.full_like(x, 3.0), 4.0, self.seed + 33))
        toe = self.bench_toe(x)
        face = smootherstep(toe, toe + run, y)
        # gullies and blocky texture on the face itself (zero at toe and crest)
        mid = 4.0 * face * (1.0 - face)
        gullies = 0.10 * fbm(x, y * 0.4, 0.9, 3, self.seed + 34) * mid
        return self.bench_height * face * nose * tail + gullies * nose * tail

    def berms(self, x, y):
        g = self.gate_x
        half_gap = self.gate_gap / 2
        pieces = [
            Berm(g, half_gap + 0.4, g, half_gap + self.gate_len, self.gate_height),
            Berm(g, -half_gap - 0.4, g, -half_gap - self.gate_len, self.gate_height),
        ]
        h = np.zeros_like(np.asarray(x, dtype=np.float64))
        for b in pieces:
            h = np.maximum(h, b.height_at(x, y))
        # right lane berm follows the converging centre line
        yc = self.right_berm_center(x)
        across = np.abs(y - yc)
        along_on = smoothstep(self.right_berm[0] - 2.0, self.right_berm[0], x) * \
            (1.0 - smoothstep(self.right_berm_end_x, self.right_berm_end_x + 4.0, x))
        rb = self.right_berm_height * berm_profile(across / 1.5) * along_on
        h = np.maximum(h, rb)
        # dozed berms are lumpy: height varies along them, clods on the surface
        lumpy = 1.0 + 0.10 * value_noise(x, y, 3.0, self.seed + 41) + 0.06 * fbm(x, y, 0.7, 2, self.seed + 42)
        return h * lumpy

    def windrows(self, x, y):
        """Ridges of loose graded material along both lane edges."""
        along = 0.65 + 0.35 * value_noise(x, np.full_like(x, 5.0), 2.5, self.seed + 51)
        left_c = self.bench_toe(x) - 0.55
        right_c = self.right_berm_center(x) + 1.75
        start = smoothstep(self.gate_x - 20.0, self.gate_x - 12.0, x)
        prof = lambda d: (1.0 - smoothstep(0.0, 0.8, np.abs(d))) ** 1.4
        w = 0.28 * (prof(y - left_c) * smoothstep(self.bench_start_x - 6.0, self.bench_start_x, x)
                    + prof(y - right_c) * smoothstep(self.right_berm[0], self.right_berm[0] + 3.0, x))
        clods = 1.0 + 0.35 * fbm(x, y, 0.35, 2, self.seed + 52)
        return w * along * clods * start

    def pit_walls(self, x, y):
        cx, cy = self.pit_center
        rx, ry = self.pit_radii
        r = np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)       # 1 at the pit edge
        dist = (r - 1.0) * min(rx, ry)                                 # ≈ metres beyond the edge
        h = np.zeros_like(dist)
        for k in range(self.pit_benches):
            start = k * self.pit_bench_spacing
            h = h + self.pit_bench_height * smootherstep(start, start + 9.0, dist)
        rim = self.pit_benches * self.pit_bench_spacing
        hills = 18.0 * smoothstep(rim, rim + 200.0, dist) * (0.6 + 0.4 * fbm(x, y, 260.0, 3, self.seed + 9))
        return h + hills

    def pad_mask(self, x, y):
        m = np.zeros_like(np.asarray(x, dtype=np.float64))
        for px0, px1, py0, py1 in self.flat_pads:
            pad = (smoothstep(px0 - 1.0, px0, x) * (1 - smoothstep(px1, px1 + 1.0, x))
                   * smoothstep(py0 - 1.0, py0, y) * (1 - smoothstep(py1, py1 + 1.0, y)))
            m = np.maximum(m, pad)
        return m

    # --- full height field ------------------------------------------------------
    def height(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        lane = self.lane_mask(x, y)
        floor = 0.10 * fbm(x, y, 18.0, 3, self.seed) + 0.03 * fbm(x, y, 2.5, 2, self.seed + 3)
        # compacted lane: tiny unevenness and a drainage crown (3 cm lower at the edges)
        right_edge = self.right_berm_center(x) + 1.3
        mid, half = 0.5 * (self.lane_left_y + right_edge), 0.5 * (self.lane_left_y - right_edge)
        crown = -0.03 * np.clip((y - mid) / half, -1.0, 1.0) ** 2
        floor = floor * (1.0 - lane) + (0.004 * fbm(x, y, 1.5, 2, self.seed + 5) + crown) * lane
        ruts = np.zeros_like(x)
        for ty in self.tracks_y:
            ruts = ruts + 0.035 * np.exp(-((y - ty) / 0.28) ** 2)
        floor = floor - ruts * lane
        bench = self.bench(x, y)
        bench_top_noise = 0.06 * fbm(x, y, 6.0, 2, self.seed + 11) * smoothstep(0.85, 1.0, bench / self.bench_height)
        pad = self.pad_mask(x, y)                 # level pads for parked machines: no undulation
        floor = floor * (1.0 - pad)
        bench_top_noise = bench_top_noise * (1.0 - pad)
        h = floor + bench + bench_top_noise + self.windrows(x, y) * (1.0 - pad)
        h = np.maximum(h, self.berms(x, y) + floor * 0.5)
        return h + self.pit_walls(x, y)

    def surface_class(self, x, y):
        """Per-point weights used for texturing: (lane, slope, bench_top)."""
        lane = self.lane_mask(x, y)
        h = self.height(x, y)
        e = 0.05
        gx = (self.height(x + e, y) - self.height(x - e, y)) / (2 * e)
        gy = (self.height(x, y + e) - self.height(x, y - e)) / (2 * e)
        slope = np.clip(np.sqrt(gx * gx + gy * gy) / 1.2, 0.0, 1.0)
        top = smoothstep(self.bench_height * 0.85, self.bench_height, h) * (1.0 - slope)
        return lane, slope, top

    # --- placement helpers --------------------------------------------------------
    def bench_top(self) -> float:
        return self.bench_height


# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------

@dataclass
class Mesh:
    points: np.ndarray      # (N, 3) float32
    triangles: np.ndarray   # (M, 3) int32
    normals: np.ndarray     # (N, 3) float32
    uvs: np.ndarray         # (N, 2) float32 — [0, 1] across the mesh rectangle
    rect: tuple[float, float, float, float]

    def validate(self):
        assert np.isfinite(self.points).all() and np.isfinite(self.normals).all()
        assert self.triangles.min() >= 0 and self.triangles.max() < len(self.points)
        n = np.linalg.norm(self.normals, axis=1)
        assert np.allclose(n, 1.0, atol=1e-3)


def grid_mesh(terrain: QuarryTerrain, rect, step: float, sink_inside=None, sink: float = 0.0) -> Mesh:
    """Regular grid over ``rect = (x0, x1, y0, y1)``; vertices inside ``sink_inside`` are lowered by ``sink``."""
    x0, x1, y0, y1 = rect
    nx = int(round((x1 - x0) / step)) + 1
    ny = int(round((y1 - y0) / step)) + 1
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")              # (ny, nx)
    Z = terrain.height(X, Y)
    # normals from central differences of the analytic field
    e = step * 0.5
    gx = (terrain.height(X + e, Y) - terrain.height(X - e, Y)) / (2 * e)
    gy = (terrain.height(X, Y + e) - terrain.height(X, Y - e)) / (2 * e)
    if sink_inside is not None:
        sx0, sx1, sy0, sy1 = sink_inside
        inside = (X > sx0) & (X < sx1) & (Y > sy0) & (Y < sy1)
        Z = np.where(inside, Z - sink, Z)
    N = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)
    uv = np.stack([(X - x0) / (x1 - x0), (Y - y0) / (y1 - y0)], axis=-1).reshape(-1, 2).astype(np.float32)
    idx = np.arange(nx * ny, dtype=np.int64).reshape(ny, nx)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]).astype(np.int32)
    return Mesh(pts, tris, N.reshape(-1, 3).astype(np.float32), uv, tuple(rect))


def graded_axis(lo: float, hi: float, fine: float = 0.25, growth: float = 1.2, cap: float = 3.0,
                limit: float = 900.0) -> np.ndarray:
    """Spacing ``fine`` on [lo, hi]; outside it the spacing grows by ``growth`` per cell up to ``cap``."""
    core = np.linspace(lo, hi, int(round((hi - lo) / fine)) + 1)

    def tail(p: float, sign: float) -> list[float]:
        out, step = [], fine
        while abs(p) < limit:
            step = min(step * growth, cap)
            p = float(np.clip(p + sign * step, -limit, limit))
            out.append(p)
        return out

    return np.array(tail(lo, -1.0)[::-1] + core.tolist() + tail(hi, +1.0))


def _tensor_mesh(terrain, xs, ys, uv_rect, cells) -> Mesh:
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    e = 0.5 * np.minimum(np.gradient(xs)[None, :], np.gradient(ys)[:, None])
    Z = terrain.height(X, Y)
    gx = (terrain.height(X + e, Y) - terrain.height(X - e, Y)) / (2 * e)
    gy = (terrain.height(X, Y + e) - terrain.height(X, Y - e)) / (2 * e)
    N = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    u0, u1, v0, v1 = uv_rect
    uv = np.stack([(X - u0) / (u1 - u0), (Y - v0) / (v1 - v0)], -1)
    ny, nx = X.shape
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = (idx[:-1, :-1][cells], idx[:-1, 1:][cells], idx[1:, 1:][cells], idx[1:, :-1][cells])
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    used = np.unique(tris)
    remap = np.full(nx * ny, -1, np.int64)
    remap[used] = np.arange(len(used))
    return Mesh(np.stack([X, Y, Z], -1).reshape(-1, 3)[used].astype(np.float32), remap[tris].astype(np.int32),
                N.reshape(-1, 3)[used].astype(np.float32), uv.reshape(-1, 2)[used].astype(np.float32),
                tuple(uv_rect))


def split_terrain_meshes(terrain, local_rect, far_rect, fine: float = 0.25, cap: float = 3.0,
                         inner_rect=None) -> tuple[Mesh, ...]:
    """One graded grid cut at ``local_rect`` (and optionally ``inner_rect`` inside it) into meshes that
    share their border vertices exactly — no overlap, z-fighting or cracks.

    Returns ``(local, far)`` or, with ``inner_rect``, ``(inner, local, far)``; each part's UVs map to its
    own rect, so each can carry its own texture resolution (e.g. a dense strip along the camera path)."""
    lx0, lx1, ly0, ly1 = local_rect
    xs = graded_axis(lx0, lx1, fine, cap=cap, limit=max(abs(far_rect[0]), abs(far_rect[1])))
    ys = graded_axis(ly0, ly1, fine, cap=cap, limit=max(abs(far_rect[2]), abs(far_rect[3])))
    cx, cy = 0.5 * (xs[:-1] + xs[1:]), 0.5 * (ys[:-1] + ys[1:])

    def cells_in(rect):
        x0, x1, y0, y1 = rect
        return ((cy[:, None] > y0) & (cy[:, None] < y1)) & ((cx[None, :] > x0) & (cx[None, :] < x1))

    local = cells_in(local_rect)
    if inner_rect is None:
        return (_tensor_mesh(terrain, xs, ys, local_rect, local), _tensor_mesh(terrain, xs, ys, far_rect, ~local))
    inner = cells_in(inner_rect) & local
    return (_tensor_mesh(terrain, xs, ys, inner_rect, inner),
            _tensor_mesh(terrain, xs, ys, local_rect, local & ~inner),
            _tensor_mesh(terrain, xs, ys, far_rect, ~local))


# ---------------------------------------------------------------------------
# Textures
# ---------------------------------------------------------------------------

SAND = np.array([0.80, 0.71, 0.55])          # light sandy pit floor
LANE = np.array([0.76, 0.71, 0.62])          # compacted, greyer haul lane
TRACK = np.array([0.62, 0.58, 0.52])         # tyre tracks
FACE = np.array([0.69, 0.55, 0.40])          # freshly cut earth faces (warmer, darker)
WALL = np.array([0.66, 0.58, 0.50])          # distant pit walls
PEBBLE_DARK = np.array([0.45, 0.42, 0.38])
PEBBLE_LIGHT = np.array([0.88, 0.84, 0.76])


def bake_albedo(terrain: QuarryTerrain, rect, px_per_m: float, seed: int = 0, far: bool = False) -> np.ndarray:
    """RGB uint8 texture covering ``rect`` (row 0 = y1, i.e. image 'up' = +y)."""
    x0, x1, y0, y1 = rect
    w = int(round((x1 - x0) * px_per_m))
    h = int(round((y1 - y0) * px_per_m))
    xs = x0 + (np.arange(w) + 0.5) / px_per_m
    ys = y1 - (np.arange(h) + 0.5) / px_per_m
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    rng = np.random.default_rng(seed)

    lane, slope, top = terrain.surface_class(X, Y)
    Z = terrain.height(X, Y)
    col = np.broadcast_to(SAND, X.shape + (3,)).copy()
    col = col * (1 - lane[..., None]) + LANE * lane[..., None]

    if not far:
        # tyre tracks on the lane: main pair + fainter older tracks, with tread marks
        tread = 0.5 + 0.5 * np.sin(X * 2 * math.pi / 0.32)
        for ty, strength in [(t, 0.55) for t in terrain.tracks_y] + [(t, 0.25) for t in terrain.extra_tracks]:
            wiggle = 0.15 * value_noise(X, np.full_like(X, ty * 10), 9.0, seed + int(ty * 10))
            band = np.exp(-((Y - ty - wiggle) / 0.30) ** 2) * lane * strength
            col = col * (1 - band[..., None]) + (TRACK * (0.92 + 0.08 * tread[..., None])) * band[..., None]

    # cut faces (bench, pit walls) vs dumped fill (berms, windrows)
    face_w = np.clip(slope * 1.3, 0, 1)
    berm_w = smoothstep(0.02, 0.12, terrain.berms(X, Y))
    cut_w = face_w * (1.0 - berm_w)
    strata = 0.05 * np.sin(Z * 2 * math.pi / (0.7 if not far else 3.0) + value_noise(X, Y, 4.0, seed + 2) * 2.0)
    streak = np.zeros_like(X) if far else 0.10 * fbm(X * 5.0, Y * 0.7, 1.0, 3, seed + 8)   # vertical scree streaks
    face_col = (FACE if not far else WALL) * (1.0 + strata[..., None] + streak[..., None])
    col = col * (1 - cut_w[..., None]) + face_col * cut_w[..., None]
    berm_col = np.array([0.74, 0.64, 0.50]) * (0.92 + 0.12 * (0.5 + 0.5 * fbm(X, Y, 0.5, 3, seed + 7)))[..., None]
    bw = (face_w * berm_w)[..., None]
    col = col * (1 - bw) + berm_col * bw
    col = col * (1.0 - 0.04 * top[..., None])

    # large-scale patches: darker/browner compacted or damp ground, paler dusty drifts
    if not far:
        patches = fbm(X, Y, 14.0, 3, seed + 31)
        damp = smoothstep(0.25, 0.6, patches)[..., None] * 0.12
        dusty = smoothstep(0.2, 0.55, -patches)[..., None] * 0.08
        col = col * (1.0 - damp) + col * np.array([0.82, 0.78, 0.74]) * damp
        col = col * (1.0 - dusty) + np.array([0.86, 0.80, 0.68]) * dusty
        wr = np.clip(terrain.windrows(X, Y) / 0.25, 0, 1)[..., None]
        col = col * (1.0 - 0.10 * wr) + np.array([0.70, 0.62, 0.50]) * 0.10 * wr

    # multi-scale brightness variation (patchiness, then grain)
    if far:
        var = 0.07 * fbm(X, Y, 60.0, 3, seed + 21) + 0.04 * fbm(X, Y, 8.0, 2, seed + 22)
    else:
        var = 0.06 * fbm(X, Y, 7.0, 3, seed + 21) + 0.035 * fbm(X, Y, 0.6, 2, seed + 22)
        var = var + 0.03 * rng.standard_normal(X.shape)          # sand grain
    col = col * (1.0 + var[..., None])
    if far:   # baked aerial perspective for the distant pit walls
        haze = 0.55 * smoothstep(150.0, 900.0, np.hypot(X, Y))[..., None]
        col = col * (1.0 - haze) + np.array([0.80, 0.82, 0.84]) * haze
    hue = 0.025 * fbm(X, Y, 12.0, 2, seed + 23)
    col[..., 0] *= 1.0 + hue
    col[..., 2] *= 1.0 - hue

    # slope-toe darkening (cheap cavity term from the height Laplacian)
    if not far:
        e = 0.25
        lap = (terrain.height(X + e, Y) + terrain.height(X - e, Y) + terrain.height(X, Y + e)
               + terrain.height(X, Y - e) - 4 * Z) / (e * e)
        col = col * (1.0 - np.clip(lap * 0.05, 0, 0.12))[..., None]

        # pebbles and small stones (denser off the lane and at slope toes)
        density = 0.0025 * (1.0 - 0.8 * lane) + 0.004 * face_w
        mask = rng.random(X.shape) < density
        shade = rng.random(X.shape)
        pebble = np.where(shade[..., None] < 0.65, PEBBLE_DARK, PEBBLE_LIGHT)
        grow = mask.copy()
        grow[1:, :] |= mask[:-1, :] & (rng.random(X.shape)[1:, :] < 0.6)
        grow[:, 1:] |= mask[:, :-1] & (rng.random(X.shape)[:, 1:] < 0.6)
        col = np.where(grow[..., None], col * 0.35 + pebble * 0.65, col)

    return (np.clip(col, 0, 1) * 255 + 0.5).astype(np.uint8)


def bake_rock_texture(size: int = 512, seed: int = 0) -> np.ndarray:
    """Tileable-ish grey-brown rock albedo for scattered stones."""
    s = np.arange(size) / size * 8.0
    X, Y = np.meshgrid(s, s, indexing="xy")
    n = 0.5 + 0.5 * fbm(X, Y, 1.5, 5, seed)
    base = np.array([0.55, 0.50, 0.45])
    col = base * (0.75 + 0.45 * n[..., None])
    return (np.clip(col, 0, 1) * 255 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Scattered rocks (visual only)
# ---------------------------------------------------------------------------

@dataclass
class Rock:
    x: float
    y: float
    z: float
    size: tuple[float, float, float]
    yaw: float
    shade: float


def scatter_rocks(terrain: QuarryTerrain, rect, count: int, seed: int = 3,
                  keep_clear: list[tuple[float, float, float, float]] | None = None) -> list[Rock]:
    """Rocks biased toward berm feet and the bench toe; none on the lane or in ``keep_clear`` rects."""
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = rect
    rocks: list[Rock] = []
    tries = 0
    while len(rocks) < count and tries < count * 60:
        tries += 1
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        lane, slope, _ = terrain.surface_class(np.array([x]), np.array([y]))
        if lane[0] > 0.05:
            continue
        if keep_clear and any(a <= x <= b and c <= y <= d for a, b, c, d in keep_clear):
            continue
        p = 0.12 + 0.88 * min(1.0, slope[0] * 2.0)
        if rng.random() > p:
            continue
        r = float(rng.choice([0.12, 0.2, 0.3, 0.45, 0.7], p=[0.35, 0.3, 0.2, 0.1, 0.05]))
        size = (r * rng.uniform(0.8, 1.3), r * rng.uniform(0.7, 1.1), r * rng.uniform(0.45, 0.8))
        z = float(terrain.height(np.array([x]), np.array([y]))[0]) + size[2] * 0.35
        rocks.append(Rock(x, y, z, size, float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(0.75, 1.15))))
    return rocks
