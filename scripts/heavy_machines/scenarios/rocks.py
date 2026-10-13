"""Blasted quarry rock and gravel (numpy only) to replace the perfect-sphere rocks.

Two things live here:

* :func:`rock_mesh` / :func:`rock_prototypes` - a handful of unit-sized (~1 m)
  triangle meshes of angular, blasted rock, meant to be authored once as
  ``UsdGeom.Mesh`` prototypes and instanced (``UsdGeom.PointInstancer``);
* :func:`scatter` - where, how big and how deep to put instances of them on a
  :class:`~heavy_machines.scenarios.terrain.QuarryTerrain`: boulders that pile up at
  the berm feet, the bench toe and along the lane edges, and gravel everywhere.

Instance transform (what :class:`Scatter` means)
-------------------------------------------------
A prototype vertex ``p`` ends up at ``positions + Rz(yaw) @ (scales * p)``: scale in
the rock's own axes first, then yaw about +z, then translate.  ``positions`` is the
centre of the rock's bounding box (prototypes are bbox-centred), so the rock bottom is
``0.5 * scale_z * height`` below ``positions.z``.  ``proto`` indexes the list
``sc.prototypes()`` = ``rock_prototypes(sc.n_protos, sc.proto_seed, sc.proto_detail)`` (defaults
``8, 0, 4``): **build the prototypes with the values stored in the Scatter**.  Boulders use the first
``proto_split(n)`` prototypes (fine meshes), gravel the remaining low-poly ones, so thousands of gravel
pieces stay cheap.  ``sc.shade`` (0.78-1.12) is a brightness variant per instance for picking a material.

``detail`` is purely a level of detail: a seed is the same rock at every detail (style, fracture planes,
noise, chips and scars are drawn without reference to it), bounding boxes agree within a few percent, so
sizes and embedding stay valid for any ``detail`` and an LOD swap does not pop.

Texture scale
-------------
A box-projected UV is in metres of the *unit* rock, but instances are scaled by 0.25-1.2 (boulders) and
0.03-0.15 (gravel); left alone, a 1 m rock texture would be squeezed 7-30x too small onto gravel.  The
prototypes' UVs are therefore pre-multiplied by :func:`proto_uv_scale` (the typical instance scale of the
group), so with a 1 m rock texture (``TEXTURE_METRES``) on ``repeat`` wrap the texel density is physical
on a typical instance of either group and within ~2-3x across the size range.  ``rock_mesh`` itself keeps
the plain unit-rock-metre UVs.

What a shared texture cannot do (dust, soil, contact)
-----------------------------------------------------
One UsdPreviewSurface texture on shared prototypes gives clean, uniformly coloured blocks that meet the
ground along a hard line.  Two helpers carry the missing information:

* :func:`rock_vertex_weights` / :func:`rock_vertex_tint` - per-vertex *dust* (up-facing) and *soil* (the
  foot of the stone).  Author the tint as ``primvars:displayColor`` and multiply it into the albedo in
  the shader (a primvar reader feeding a multiply; plain UsdPreviewSurface can only replace the texture
  with it, so without such a shader skip it and use the material variants picked by ``Scatter.shade``);
* :func:`contact_map` - a multiplicative map for the *terrain* texture: multiply it into
  ``bake_albedo`` of the same rect and every boulder gets a soft dark halo of contact shadow / packed
  soil.  This works with plain UsdPreviewSurface.

How a rock is made
------------------
A real blasted rock is not a blob: it is a block bounded by a few planar fracture
faces (bedding and joint planes, plus chips knocked off the corners), with sharp
edges, slightly curved/rough faces and a flat-ish side it lies on.  So:

1. the rock is a *convex polytope* ``{x : n_i . x <= d_i}`` made of three roughly
   orthogonal pairs of joint planes (random tilt/offset), a flat bottom, a flat /
   pitched / pyramidal top and a number of corner, edge and free-direction chips;
2. it is meshed by casting the vertices of a subdivided icosphere (stretched to the
   rock's aspect so sampling stays even) onto it - the planar fracture cuts are
   exact.  Wherever a mesh edge crosses a fracture edge, its nearer end point is
   snapped onto that edge, so every fracture edge is lined with a chain of vertices
   lying exactly on it (crisp, straight edges instead of a sawtooth) and the facets on
   both sides stay planar;
3. a rolling-ball fillet rounds the edges and corners with a radius set by
   ``angularity`` (weathered boulders are rounder than freshly blasted ones);
4. multi-octave 3-D noise bows the faces (conchoidal curvature) and roughens them: a ridged crease
   network at 3-6 cm and grain at 2-3 cm, each fracture face with its own roughness (some are cleaved
   smooth, others rough), bedded rock gets ledges; the edges are chipped by smooth noise that fades into
   both faces; a few shallow *scars* (local conchoidal chips) break the faces and 2-6 *notches* are
   knocked out of edges and corners;
5. every one of these moves goes through a guard that halves the move of any vertex
   whose triangle would flip, collapse or fold into a knife edge, so the mesh stays a
   clean closed manifold;
6. normals: each fracture face is a smoothing group (area-weighted vertex normals inside
   the face, so the tessellation never shows; creases stay sharp between faces),
   blended towards the face's plane normal by ``angularity``;
7. UVs: box projection per fracture face, in metres (1 UV unit = 1 m of the unit
   rock), so one tileable PBR rock texture keeps its physical scale (see *Texture scale*).

Winding is counter-clockwise seen from outside; ``cross(b - a, c - a)`` points out.
Vertices are split along creases and UV seams (welded where position, normal and UV
agree), so the mesh is closed only after welding vertices by position.

Placement in the clip
---------------------
Defaults are tuned for presence from the haul-road camera, not for statistical purity: a shallow size
power law (``size_alpha`` 1.6: median 0.45 m, ~40% over 0.5 m), boulders weighted to the lane-edge windrow
and berm toes and kept off the bench top, a minimum height/width of 0.5 so even slabs stand 10+ cm out
of the ground, and ``focus`` (a rect, e.g. the camera's view) to concentrate them where they are seen.
Gravel dominates the instance count; boulders are cheap, so ``boulders`` of 400-600 over the 130 x 65 m
local rect is a sensible setting.  Measured on the clip's terrain and front camera (eye x = -20.45, 92 deg),
counting blocks >= 0.45 m wide that stand >= 0.2 m above the ground in the first 40 m ahead: 160 boulders give
~9 (~12 with ``focus`` on the view zone; the module this replaces gave 0-3), 480 with ``focus`` give ~40.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .terrain import Mesh, QuarryTerrain, fbm, smoothstep

__all__ = ["rock_mesh", "rock_prototypes", "Scatter", "scatter", "STYLE_NAMES", "proto_extents", "proto_split",
           "proto_uv_scale", "rock_vertex_weights", "rock_vertex_tint", "contact_map", "TEXTURE_METRES"]


# ---------------------------------------------------------------------------
# 3-D value noise
# ---------------------------------------------------------------------------

def _hash3(ix, iy, iz, seed: int):
    h = (ix * 374761393 + iy * 668265263 + iz * 1274126177 + seed * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF).astype(np.float64) / float(0x1000000)


def _noise3(p, scale: float, seed: int):
    """Smooth value noise in [-1, 1] at points ``p`` (N, 3), feature size ``scale``."""
    q = np.asarray(p, dtype=np.float64) / scale
    i = np.floor(q)
    f = q - i
    f = f * f * (3.0 - 2.0 * f)
    i = i.astype(np.int64)
    out = 0.0
    for dx, dy, dz in itertools.product((0, 1), repeat=3):
        w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
             * (f[:, 2] if dz else 1 - f[:, 2]))
        out = out + w * _hash3(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz, seed)
    return 2.0 * out - 1.0


# ---------------------------------------------------------------------------
# Icosphere
# ---------------------------------------------------------------------------

_ICO_V = np.array([(-1, 1.6180339887, 0), (1, 1.6180339887, 0), (-1, -1.6180339887, 0), (1, -1.6180339887, 0),
                   (0, -1, 1.6180339887), (0, 1, 1.6180339887), (0, -1, -1.6180339887), (0, 1, -1.6180339887),
                   (1.6180339887, 0, -1), (1.6180339887, 0, 1), (-1.6180339887, 0, -1), (-1.6180339887, 0, 1)],
                  dtype=np.float64)
_ICO_T = np.array([(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
                   (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
                   (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)], dtype=np.int64)


@lru_cache(maxsize=None)
def _icosphere(level: int):
    """Unit icosphere, ``level`` midpoint subdivisions (20 * 4**level faces), CCW outward."""
    P = _ICO_V / np.linalg.norm(_ICO_V, axis=1, keepdims=True)
    T = _ICO_T.copy()
    for _ in range(level):
        n = len(P)
        e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
        uk, inv = np.unique(e[:, 0] * n + e[:, 1], return_inverse=True)
        mid = 0.5 * (P[uk // n] + P[uk % n])
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        nt = len(T)
        m01, m12, m20 = inv[:nt] + n, inv[nt:2 * nt] + n, inv[2 * nt:] + n
        P = np.concatenate([P, mid])
        T = np.concatenate([np.stack([T[:, 0], m01, m20], 1), np.stack([T[:, 1], m12, m01], 1),
                            np.stack([T[:, 2], m20, m12], 1), np.stack([m01, m12, m20], 1)])
    P.flags.writeable = False
    T.flags.writeable = False
    return P, T


# ---------------------------------------------------------------------------
# Convex polytope from half-spaces
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _combos(k: int) -> np.ndarray:
    return np.array(list(itertools.combinations(range(k), 3)), dtype=np.int64)


def _hull_vertices(N: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Vertices of ``{x : N x <= d}`` (assumed bounded), duplicates merged."""
    c = _combos(len(N))
    A = N[c]
    ok = np.abs(np.linalg.det(A)) > 1e-8
    A, c = A[ok], c[ok]
    x = np.linalg.solve(A, d[c][..., None])[..., 0]
    inside = (x @ N.T - d[None]).max(axis=1) <= 1e-7
    x = x[inside]
    if len(x) == 0:
        return x
    keep = []
    for i in range(len(x)):
        if all(np.linalg.norm(x[i] - x[j]) > 1e-4 for j in keep):
            keep.append(i)
    return x[keep]


def _plane_adjacency(N: np.ndarray, d: np.ndarray, V: np.ndarray) -> np.ndarray:
    """(K, K) bool: planes i, j meet in an edge of the polytope (share >= 2 vertices)."""
    on = np.abs(V @ N.T - d[None]) < 1e-6                       # (nV, K)
    share = on.T.astype(np.int64) @ on.astype(np.int64)
    adj = share >= 2
    np.fill_diagonal(adj, False)
    return adj


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _rot_xyz(ax, ay, az) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# ---------------------------------------------------------------------------
# Rock styles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Style:
    name: str
    a: tuple[float, float]          # half extent along x (the long axis), before normalisation
    b: tuple[float, float]          # half extent along y
    c: tuple[float, float]          # half height
    tilt: float                     # degrees of random tilt of the joint planes
    top: str                        # 'flat' | 'ridge' | 'pyramid' | 'dome'
    top_slope: tuple[float, float]  # degrees, slope of the pitched top faces
    chamfers: tuple[int, int]       # number of corner / edge chips
    depth: tuple[float, float]      # chips cut at this fraction of the corner distance (lower = deeper)
    angularity: float               # default rounding / shading
    wobble: float = 1.0             # amplitude of the face-bowing noise
    bedding: float = 0.0            # bedding ledge amplitude (m, before normalisation)
    scars: tuple[int, int] = (0, 2)
    weight: float = 1.0
    free: float = 0.15            # share of chips with a random direction


_STYLES = (
    #      name       a            b            c (half height)  tilt top        top slope   chips     depth        ang   wob  bed    scars   w    free
    _Style("blocky", (0.46, 0.58), (0.36, 0.48), (0.30, 0.40), 10, "flat", (8, 16), (7, 12), (0.66, 0.90), 0.86, 1.0, 0.0, (1, 3), 1.0, 0.30),
    _Style("chunk", (0.46, 0.54), (0.40, 0.50), (0.32, 0.44), 15, "pyramid", (16, 32), (9, 15), (0.64, 0.88), 0.82, 1.0, 0.0, (1, 3), 1.0, 0.55),
    _Style("slab", (0.52, 0.62), (0.38, 0.50), (0.10, 0.17), 6, "flat", (4, 9), (5, 9), (0.72, 0.92), 0.86, 0.8, 0.010, (1, 2), 1.0, 0.30),
    _Style("wedge", (0.50, 0.60), (0.34, 0.44), (0.20, 0.30), 11, "ridge", (14, 30), (7, 12), (0.66, 0.90), 0.84, 1.0, 0.0, (1, 3), 1.0, 0.35),
    _Style("rounded", (0.46, 0.54), (0.38, 0.48), (0.26, 0.36), 18, "dome", (14, 26), (12, 17), (0.80, 0.95), 0.46, 1.3, 0.0, (0, 2), 1.0, 0.45),
    _Style("long", (0.62, 0.74), (0.24, 0.34), (0.20, 0.30), 8, "flat", (6, 14), (6, 10), (0.70, 0.90), 0.86, 1.0, 0.007, (1, 2), 1.0, 0.30),
    _Style("shard", (0.46, 0.56), (0.26, 0.34), (0.32, 0.44), 12, "ridge", (18, 34), (7, 12), (0.64, 0.88), 0.88, 0.9, 0.0, (0, 2), 1.0, 0.45),
    _Style("tabular", (0.50, 0.60), (0.40, 0.50), (0.14, 0.22), 10, "flat", (4, 10), (9, 14), (0.70, 0.90), 0.84, 0.9, 0.008, (1, 2), 1.0, 0.40),
)
STYLE_NAMES = tuple(s.name for s in _STYLES)


def _uniform(rng, lohi):
    return float(rng.uniform(lohi[0], lohi[1]))


def _build_planes(rng: np.random.Generator, st: _Style, rough: float):
    """Half-space description (N, d) of the rock, origin roughly at its centre."""
    a, b, c = _uniform(rng, st.a), _uniform(rng, st.b), _uniform(rng, st.c)
    tilt = math.radians(st.tilt)
    N, D = [], []

    def add(n, d):
        N.append(_unit(n))
        D.append(float(d))

    # three roughly orthogonal joint sets; each face gets its own tilt so none are parallel
    for sign in (1.0, -1.0):
        R = _rot_xyz(*(rng.uniform(-tilt, tilt, 3) * 1.0))
        add(R @ np.array([sign, 0.0, 0.0]), a * rng.uniform(0.80, 1.20))
        R = _rot_xyz(*(rng.uniform(-tilt, tilt, 3) * 1.0))
        add(R @ np.array([0.0, sign, 0.0]), b * rng.uniform(0.80, 1.20))
    # flat-ish bottom (the face the rock lies on): at most ~5 degrees off horizontal
    phi, psi = math.radians(rng.uniform(0.5, 5.0)), rng.uniform(0, 2 * math.pi)
    add([math.sin(phi) * math.cos(psi), math.sin(phi) * math.sin(psi), -math.cos(phi)], c * rng.uniform(0.9, 1.1))
    # top
    hz = c * rng.uniform(0.95, 1.2)
    az0 = rng.uniform(0, 2 * math.pi)
    if st.top == "flat":
        th = math.radians(rng.uniform(0, 9)) * (1.0 if rng.random() < 0.8 else 2.0)
        al = rng.uniform(0, 2 * math.pi)
        add([math.sin(th) * math.cos(al), math.sin(th) * math.sin(al), math.cos(th)], hz)
    else:
        count = {"ridge": 2, "pyramid": int(rng.integers(3, 5)), "dome": int(rng.integers(5, 8))}[st.top]
        slope = lambda: math.radians(_uniform(rng, st.top_slope))  # noqa: E731
        apex = np.array([rng.uniform(-0.18, 0.18) * a, rng.uniform(-0.18, 0.18) * b, hz * 1.05])
        for k in range(count):
            al = az0 + (math.pi * k if st.top == "ridge" else 2 * math.pi * k / count) + rng.uniform(-0.45, 0.45)
            th = slope()
            n = np.array([math.sin(th) * math.cos(al), math.sin(th) * math.sin(al), math.cos(th)])
            add(n, float(n @ apex))
    Narr, Darr = np.array(N), np.array(D)
    # corner / edge chips cut against the current shape
    lo, hi = st.chamfers
    n_ch = int(rng.integers(lo, hi + 1))
    n_ch = max(0, int(round(n_ch * (0.6 + 0.8 * rough))))
    for _ in range(n_ch):
        V = _hull_vertices(Narr, Darr)
        if len(V) < 4:
            break
        ctr = V.mean(0)
        sx, sy, sz = (rng.choice([-1.0, 1.0]) for _ in range(3))
        kind = rng.random()
        if kind < st.free:     # free fracture: any direction (irregular polyhedron)
            n = rng.normal(size=3)
        elif kind < st.free + 0.5 * (1.0 - st.free):   # corner chip: plane through the three axis intercepts
            n = np.array([sx / a, sy / b, sz / c])
        else:                  # edge chip
            axis = int(rng.integers(0, 3))
            n = np.array([sx / a, sy / b, sz / c])
            n[axis] = 0.0
        n = _unit(n + rng.normal(scale=0.12, size=3) * np.linalg.norm(n))
        if n[2] < -0.55:       # keep the bottom face broad
            n[2] = -0.55
            n = _unit(n)
        h = float(((V - ctr) @ n).max())
        depth = _uniform(rng, st.depth)
        if n[2] < -0.25:
            depth = max(depth, 0.88)
        Narr = np.vstack([Narr, n])
        Darr = np.append(Darr, float(n @ ctr) + h * depth)
    # a few big fractures that take a real bite out of the block (a blasted rock is an irregular polyhedron,
    # not a brick with chamfers); mostly on the top half, the underside stays broad
    for _ in range(int(rng.integers(1, 4))):
        V = _hull_vertices(Narr, Darr)
        if len(V) < 4:
            break
        ctr = V.mean(0)
        sx, sy = (rng.choice([-1.0, 1.0]) for _ in range(2))
        sz = 1.0 if rng.random() < 0.8 else -1.0
        n = np.array([sx / a, sy / b, sz / c]) * np.array([rng.uniform(0.3, 1.2), rng.uniform(0.3, 1.2), rng.uniform(0.5, 1.5)])
        n = _unit(n + rng.normal(scale=0.25, size=3) * np.linalg.norm(n))
        if n[2] < -0.2:
            n[2] = -0.2
            n = _unit(n)
        h = float(((V - ctr) @ n).max())
        Narr = np.vstack([Narr, n])
        Darr = np.append(Darr, float(n @ ctr) + h * rng.uniform(0.52, 0.80))
    # recentre on the vertex centroid
    V = _hull_vertices(Narr, Darr)
    ctr = V.mean(0)
    Darr = Darr - Narr @ ctr
    return Narr, Darr


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------

def _round_edges(P: np.ndarray, Np: np.ndarray, Dp: np.ndarray, rho: float) -> np.ndarray:
    """Round every fracture edge and corner with radius ``rho``.

    A point of the polytope surface is replaced by the point at distance ``rho`` from the polytope
    shrunk by ``rho`` along the same direction (rolling-ball fillet).  Points on flat faces stay
    where they are; edge and corner points move inwards.  It is a per-vertex operation, so it is
    independent of the tessellation (no uneven smoothing along edge chains)."""
    if rho <= 1e-6:
        return P
    c = P.copy()
    d = Dp - rho
    for _ in range(24):
        viol = c @ Np.T - d[None]
        k = viol.argmax(axis=1)
        m = np.maximum(viol.max(axis=1), 0.0)
        if m.max() < 1e-9:
            break
        c -= m[:, None] * Np[k]
    v = P - c
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(n > 1e-9, c + rho * v / np.maximum(n, 1e-12), P)


def _vertex_faces(T: np.ndarray, nv: int) -> np.ndarray:
    """(nv, valence) padded (-1) table of the faces around each vertex."""
    vid = T.reshape(-1)
    fid = np.repeat(np.arange(len(T)), 3)
    order = np.argsort(vid, kind="stable")
    vs, fs = vid[order], fid[order]
    counts = np.bincount(vid, minlength=nv)
    starts = np.cumsum(counts) - counts
    rank = np.arange(len(vs)) - starts[vs]
    inc = -np.ones((nv, counts.max()), np.int64)
    inc[vs, rank] = fs
    return inc


def _face_normals(P: np.ndarray, T: np.ndarray):
    cr = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    fn = cr / np.maximum(2.0 * area, 1e-18)[:, None]
    return fn, area


def _edge_faces(T: np.ndarray):
    """Pairs of faces sharing each mesh edge (closed manifold): arrays (f1, f2, a, b)."""
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    f = np.tile(np.arange(len(T)), 3)
    nv = int(T.max()) + 1
    key = np.minimum(e[:, 0], e[:, 1]) * nv + np.maximum(e[:, 0], e[:, 1])
    o = np.argsort(key, kind="stable")
    return f[o][0::2], f[o][1::2], e[o][0::2, 0], e[o][0::2, 1]


def _dihedrals(fn: np.ndarray, ef) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", fn[ef[0]], fn[ef[1]]), -1.0, 1.0)))


def _guard(P0: np.ndarray, P1: np.ndarray, T: np.ndarray, ef, max_dihedral: float = 118.0, iters: int = 16) -> np.ndarray:
    """Move the mesh from ``P0`` to ``P1`` without creating flipped, collapsed or knife-edged triangles.

    Wherever the move would flip a triangle (its normal turns away from the original one), collapse it
    (area below 5%) or create an edge sharper than ``max_dihedral``, the offending vertices' moves are
    halved; repeated until the mesh is clean.  A rock edge never folds back onto itself."""
    P = P1.copy()
    n0, a0 = _face_normals(P0, T)
    d0 = _dihedrals(n0, ef)
    for _ in range(iters):
        n1, a1 = _face_normals(P, T)
        bad_t = (np.einsum("ij,ij->i", n0, n1) < 0.2) | (a1 < 0.05 * a0)
        d1 = _dihedrals(n1, ef)
        bad_e = (d1 > max_dihedral) & (d1 > d0 + 10.0)
        if not bad_t.any() and not bad_e.any():
            break
        v = np.unique(np.concatenate([T[bad_t].ravel(), T[ef[0][bad_e]].ravel(), T[ef[1][bad_e]].ravel()]))
        P[v] = P0[v] + 0.5 * (P[v] - P0[v])
    return P


def _cast(Np: np.ndarray, Dp: np.ndarray, detail: int):
    """Tessellate the polytope ``{x : Np x <= Dp}``: an icosphere stretched to the rock's aspect is cast
    onto it and snapped onto its edges.  Returns ``(P, T, act, group, edge_v, edge_in)``: vertices,
    triangles, the plane each vertex was cast onto, the plane (fracture face) each triangle lies on,
    and the vertices that sit on a fracture edge with their inward direction."""
    V = _hull_vertices(Np, Dp)
    half = np.maximum(np.abs(V).max(axis=0), 0.05)
    U, T = _icosphere(detail)
    dirs = _unit(U * half)
    nd = dirs @ Np.T
    with np.errstate(divide="ignore", invalid="ignore"):
        r_all = np.where(nd > 1e-9, Dp[None] / nd, np.inf)
    act = r_all.argmin(axis=1)                          # the plane each ray hits first
    P = dirs * r_all[np.arange(len(dirs)), act][:, None]

    # --- snap onto the fracture edges --------------------------------------------------------------
    # Every mesh edge whose end points were cast onto two *adjacent* planes crosses a fracture edge.
    # The end point nearer to that edge moves onto it (in its own plane), so each fracture edge
    # is lined with a connected chain of vertices lying exactly on it and the facets on either side
    # stay planar; no vertex moves by more than half a mesh edge.
    adj = _plane_adjacency(Np, Dp, V)
    me = np.unique(np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1), axis=0)
    ea, eb = me[:, 0], me[:, 1]
    pa, pb = act[ea], act[eb]
    cross = (pa != pb) & adj[pa, pb]
    ea, eb, pa, pb = ea[cross], eb[cross], pa[cross], pb[cross]
    cosab = np.einsum("ij,ij->i", Np[pa], Np[pb])
    sinab = np.sqrt(np.maximum(1.0 - cosab ** 2, 0.0))
    ok = sinab > 0.05
    ea, eb, pa, pb, sinab = ea[ok], eb[ok], pa[ok], pb[ok], sinab[ok]
    # in-plane distance of each end point to the edge line (towards the other plane)
    s_a = (Dp[pb] - np.einsum("ij,ij->i", Np[pb], P[ea])) / sinab
    s_b = (Dp[pa] - np.einsum("ij,ij->i", Np[pa], P[eb])) / sinab
    pick_a = s_a <= s_b
    sv = np.where(pick_a, ea, eb)                       # vertex to snap
    other = np.where(pick_a, pb, pa)                    # plane to snap it onto
    sdist = np.where(pick_a, s_a, s_b)
    order_s = np.argsort(-sdist, kind="stable")         # keep the smallest move if a vertex is picked twice
    best_other = -np.ones(len(P), np.int64)
    best_other[sv[order_s]] = other[order_s]
    snap = np.nonzero(best_other >= 0)[0]
    pj = best_other[snap]
    pi = act[snap]
    cosij = np.einsum("ij,ij->i", Np[pi], Np[pj])
    sinij = np.sqrt(np.maximum(1.0 - cosij ** 2, 1e-12))
    w_in = Np[pj] - cosij[:, None] * Np[pi]
    w_in /= np.maximum(np.linalg.norm(w_in, axis=1, keepdims=True), 1e-12)
    move = ((Dp[pj] - np.einsum("ij,ij->i", Np[pj], P[snap])) / sinij)[:, None] * w_in
    Ps = P[snap] + move
    good = (Ps @ Np.T - Dp[None]).max(axis=1) < 1e-6
    # Snapping must never flip or collapse a triangle (a vertex can overtake an unsnapped neighbour that
    # is nearer to the edge, or three vertices can land on one edge line): see _guard.
    P0 = P
    P1 = P0.copy()
    P1[snap[good]] = Ps[good]
    ef = _edge_faces(T)
    P = _guard(P0, P1, T, ef)
    snapped = np.zeros(len(P), bool)
    snapped[snap[good]] = True
    snapped &= np.linalg.norm(P - P1, axis=1) < 1e-9          # fully snapped (not reverted / halved)
    sel = snapped[snap] & good
    edge_v, e_i, e_j = snap[sel], pi[sel], pj[sel]
    edge_in = -_unit(Np[e_i] + Np[e_j])                                      # inward direction at fracture-edge vertices
    # fracture face of every triangle = the plane its vertices actually lie on (after snapping)
    res = np.abs(P @ Np.T - Dp[None])                           # (V, K) distance of each vertex to each plane
    group = res[T].mean(axis=1).argmin(axis=1)
    return P, T, act, group, edge_v, edge_in


def _nearest_edge(P: np.ndarray, pe: np.ndarray, chunk: int = 1024):
    """(distance, index) of the nearest point of ``pe`` for every row of ``P`` (chunked brute force)."""
    d_out = np.empty(len(P))
    i_out = np.empty(len(P), np.int64)
    for k in range(0, len(P), chunk):
        d2 = ((P[k:k + chunk, None, :] - pe[None, :, :]) ** 2).sum(axis=2)
        j = d2.argmin(axis=1)
        i_out[k:k + chunk] = j
        d_out[k:k + chunk] = np.sqrt(d2[np.arange(len(j)), j])
    return d_out, i_out


def _planar_cut(P: np.ndarray, ctr: np.ndarray, m: np.ndarray, rho: float, dep: float) -> np.ndarray:
    """Vertex positions after a shallow conchoidal chip: a planar cut with outward normal ``m`` through
    ``ctr - dep * m``, limited to radius ``rho`` and tapered so that there is no step at the rim."""
    dist = np.linalg.norm(P - ctr, axis=1)
    taper = 1.0 - smoothstep(0.55 * rho, rho, dist)
    sdist = (P - (ctr - dep * m)) @ m + (1.0 - taper) * dep      # distance beyond the (tilted-up at the rim) cut
    hit = (dist < rho) & (sdist > 0)
    P1 = P.copy()
    P1[hit] -= sdist[hit, None] * m
    return P1


def _build_rock(seed: int, detail: int, angularity: float, style: _Style | None = None):
    """Geometry of one rock: ``(P, T, group, Np, angularity, rng)`` - vertices, triangles, the fracture
    plane of each triangle, the plane normals, the clipped angularity and the generator (for the UV offset).

    ``detail`` only sets how finely the rock is tessellated: it takes no part in the random stream (style,
    planes, noise seeds, chip and scar positions are all drawn without reference to it), so one seed is the
    same rock at every detail level and a level-of-detail swap does not pop."""
    rng = np.random.default_rng([int(seed) & 0x7FFFFFFF, 0x0B10CC])
    angularity = float(np.clip(angularity, 0.0, 1.0))
    detail = int(np.clip(detail, 1, 5))
    if style is None:
        w = np.array([s.weight for s in _STYLES])
        style = _STYLES[int(rng.choice(len(_STYLES), p=w / w.sum()))]
    rough = 1.0 - angularity
    nseed = [int(v) for v in rng.integers(1, 1 << 20, 12)]

    Np, Dp = _build_planes(rng, style, rough)
    face_rough = rng.uniform(0.35, 1.5, len(Np))        # some fracture faces are cleaved smooth, others rough
    P, T, act, group, edge_v, edge_in = _cast(Np, Dp, detail)
    ef = _edge_faces(T)
    P = _guard(P, _round_edges(P, Np, Dp, 0.006 + 0.07 * rough ** 1.5), T, ef)
    fr = face_rough[act]

    # --- face bowing, roughness, ledges (displacement along the face normal) -------------------------
    wob = style.wobble * (0.6 + 1.2 * rough)             # a little more bowing on weathered stone
    amp = np.array([0.0125, 0.0120, 0.0045]) * wob
    disp = (amp[0] * _noise3(P, 0.62, nseed[0]) + amp[1] * _noise3(P, 0.23, nseed[1])
            + amp[2] * _noise3(P, 0.085, nseed[2]))
    # rough fracture surface: a network of crease lines (ridged noise) at 3-6 cm, plus grain at 2-3 cm,
    # strongest on the rough faces - the centimetre-scale relief a blasted face really has
    ridge = 1.0 - np.abs(_noise3(P, 0.062, nseed[6]))
    disp = disp + fr * (0.010 * (ridge - 0.62) + 0.0045 * _noise3(P, 0.028, nseed[7]))
    if style.bedding > 0:
        z = P[:, 2] / rng.uniform(0.085, 0.14) + 0.7 * _noise3(P, 0.4, nseed[3])
        saw = 2.0 * np.abs(z - np.floor(z) - 0.5)       # triangle wave in [0, 1]
        disp = disp + style.bedding * (saw - 0.5) * (1.0 - np.abs(Np[act][:, 2])) ** 2
    bottom = np.clip((-Np[act][:, 2] - 0.8) / 0.15, 0.0, 1.0)
    disp = disp * (1.0 - 0.8 * bottom)
    P = _guard(P, P + Np[act] * disp[:, None], T, ef)

    # --- chipped edges: blasted edges are never straight over more than a few centimetres ---------------
    # The chip depth follows smooth noise along the edge and fades over ~3 cm into the faces on both
    # sides, so the edge reads as spalled stone (soft lumps and steps), not as a regular saw-tooth.
    if angularity > 0.3 and len(edge_v):
        pe = P[edge_v]
        chip_e = (0.011 * np.clip(_noise3(pe, 0.16, nseed[4]), 0, 1) ** 1.3
                  + 0.026 * np.clip(_noise3(pe, 0.34, nseed[5]), 0, 1) ** 2.0)
        dist, j = _nearest_edge(P, pe)
        fall = np.exp(-(dist / 0.032) ** 2)
        P1 = P + edge_in[j] * (chip_e[j] * fall)[:, None]
        P = _guard(P, P1, T, ef)

    # --- scars: shallow conchoidal chips (local planar cuts of limited radius) -----------------------
    # Centres are picked by direction from the centre of the rock, never by vertex index, so they do not
    # depend on the tessellation.
    n_scar = int(rng.integers(style.scars[0], style.scars[1] + 1))
    for _ in range(n_scar if angularity > 0.35 else 0):
        dirs = _unit(P - P.mean(axis=0))
        k = int(np.argmax(dirs @ _unit(rng.normal(size=3))))
        m = _unit(Np[act[k]] + rng.normal(scale=0.35, size=3))
        rho, dep = rng.uniform(0.07, 0.15), rng.uniform(0.008, 0.020)
        P = _guard(P, _planar_cut(P, P[k], m, rho, dep), T, ef)

    # --- notches: stones are chipped where an edge or corner was knocked in handling ----------------------
    n_notch = int(rng.integers(2, 7))
    if angularity > 0.35 and len(edge_v):
        for _ in range(n_notch):
            pe = P[edge_v]
            dirs = _unit(pe - P.mean(axis=0))
            k = int(np.argmax(dirs @ _unit(rng.normal(size=3))))
            m = _unit(-edge_in[k] + rng.normal(scale=0.45, size=3))
            rho, dep = rng.uniform(0.035, 0.085), rng.uniform(0.010, 0.032) * (0.5 + angularity)
            P = _guard(P, _planar_cut(P, pe[k], m, rho, dep), T, ef)

    # --- normalise: ~1 m across, bbox centred at the origin ------------------------------------------
    lo, hi = P.min(axis=0), P.max(axis=0)
    size = rng.uniform(0.95, 1.05)
    P = (P - 0.5 * (lo + hi)) * (size / max(hi[0] - lo[0], hi[1] - lo[1]))
    P[:, 2] -= 0.5 * (P[:, 2].min() + P[:, 2].max())

    return P, T, group, Np, angularity, rng


def _shade_and_uv(P, T, group, Np, angularity, rng, uv_scale: float = 1.0):
    """Split-by-attribute vertex normals + box-projected UVs, welded where attributes agree.

    UVs are the box-projected coordinates in metres of the unit rock, times ``uv_scale``."""
    F = len(T)
    fn, area = _face_normals(P, T)
    inc = _vertex_faces(T, len(P))
    g = inc[T]                                            # (F, 3, M)
    valid = g >= 0
    gi = np.where(valid, g, 0)
    gn = fn[gi]                                           # (F, 3, M, 3)
    # smoothing groups are the fracture faces: a face shades smoothly (so the tessellation never
    # shows), a fracture edge stays sharp; the facet normal is then blended in by ``angularity``
    same = (group[gi] == group[:, None, None]) | (np.einsum("fj,fkmj->fkm", fn, gn) > math.cos(math.radians(10.0)))
    use = valid & same
    wgt = np.where(use, area[gi], 0.0)
    smooth = np.einsum("fkm,fkmj->fkj", wgt, gn)
    smooth /= np.maximum(np.linalg.norm(smooth, axis=2, keepdims=True), 1e-12)
    flat = 0.5 * angularity ** 1.5
    plane_n = Np[group]
    cn = (1.0 - flat) * smooth + flat * plane_n[:, None, :]
    cn /= np.maximum(np.linalg.norm(cn, axis=2, keepdims=True), 1e-12)

    # UV: one box-projection axis per fracture face
    gnorm = Np[group]
    ax = np.abs(gnorm).argmax(axis=1)
    sgn = np.sign(gnorm[np.arange(F), ax])
    sgn[sgn == 0] = 1.0
    Pf = P[T]                                             # (F, 3, 3)
    fi = np.arange(F)[:, None]
    ki = np.arange(3)[None, :]
    u = Pf[fi, ki, ((ax + 1) % 3)[:, None]] * sgn[:, None]
    v = Pf[fi, ki, ((ax + 2) % 3)[:, None]]
    off = rng.uniform(0.0, 16.0, 2)
    uv = np.stack([u + off[0], v + off[1]], axis=2)       # (F, 3, 2)

    # split corners, then weld corners that agree on vertex, normal and uv
    vid = T.reshape(-1)
    key = np.concatenate([vid[:, None], np.round(cn.reshape(-1, 3) * 2e3).astype(np.int64),
                          np.round(uv.reshape(-1, 2) * 1e4).astype(np.int64)], axis=1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    pts = P[vid][first]
    nrm = cn.reshape(-1, 3)[first]
    uvs = uv.reshape(-1, 2)[first] * uv_scale
    tris = inv.reshape(F, 3)
    return pts, tris, nrm, uvs


def _make_mesh(seed: int, detail: int, angularity: float, style: _Style | None, uv_scale: float = 1.0) -> Mesh:
    P, T, group, Np, ang, rng = _build_rock(seed, detail, angularity, style)
    pts, tris, nrm, uvs = _shade_and_uv(P, T, group, Np, ang, rng, uv_scale)
    mesh = Mesh(pts.astype(np.float32), tris.astype(np.int32), nrm.astype(np.float32), uvs.astype(np.float32),
                (float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 1].min()), float(pts[:, 1].max())))
    return mesh


def rock_mesh(seed: int, detail: int = 3, angularity: float = 0.6) -> Mesh:
    """One blasted rock, ~1 m across, bbox centred on the origin, flattish bottom, CCW outward.

    ``detail`` is the number of icosphere subdivisions (3 -> ~1300 triangles, 4 -> ~5000).
    ``angularity`` in [0, 1]: 1 = freshly blasted, sharp edges and faceted shading; 0 =
    weathered, rounded and smoothly shaded.  The rock shape family (blocky, slab, wedge,
    rounded, ...) is drawn from ``seed``.  UVs are in metres (box projection per face).
    """
    return _make_mesh(seed, detail, angularity, None)


# order in which the style families are dealt to boulder- and gravel-grade prototypes
_BOULDER_ORDER = (0, 1, 2, 3, 4, 5, 6, 7)       # blocky, chunk, slab, wedge, rounded, long, shard, tabular
_GRAVEL_ORDER = (6, 1, 7, 0, 5, 3, 2, 4)        # shards, chunks, flakes ... - what blasting and crushing produce

TEXTURE_METRES = 1.0                            # real-world size one tile of the rock texture is meant to cover
_TYPICAL_SCALE = (0.50, 0.06)                   # typical instance scale of boulder-grade / gravel-grade rocks


def _clip_detail(detail) -> int:
    return int(np.clip(detail, 1, 5))


def proto_split(n: int) -> int:
    """Number of boulder-grade prototypes among ``n``: they come first (indices ``[0, split)``), the
    remaining low-poly ones are gravel-grade.  ``scatter`` puts boulders on the first group and
    gravel on the second, so thousands of gravel instances stay cheap."""
    return n if n < 2 else max(1, int(round(n * 0.625)))


def proto_uv_scale(n: int = 8) -> np.ndarray:
    """(n,) factor ``rock_prototypes`` multiplies each prototype's UVs by.

    A prototype's box-projected UVs are in metres of the unit (~1 m) rock, but an instance is scaled
    down by 0.25-1.2 (boulders) or 0.03-0.15 (gravel), which would squeeze one texture tile into a few
    centimetres of gravel (sub-pixel, shimmering texture).  The UVs are therefore pre-scaled by the typical
    instance scale of the group, so that at a typical instance one UV unit covers
    ``TEXTURE_METRES`` of rock (1 m): the texel density is then physical, within ~2x across the
    size range of a group, and the same 1 m rock texture serves boulders and gravel."""
    n = int(n)
    nb = proto_split(n)
    return np.array([_TYPICAL_SCALE[0 if i < nb else 1] / TEXTURE_METRES for i in range(n)], dtype=np.float64)


@lru_cache(maxsize=8)
def _prototypes_cached(n: int, seed: int, detail: int) -> tuple[Mesh, ...]:
    nb = proto_split(n)
    uvs = proto_uv_scale(n)
    out = []
    for i in range(n):
        gravel = i >= nb
        k = i - nb if gravel else i
        st = _STYLES[(_GRAVEL_ORDER if gravel else _BOULDER_ORDER)[k % len(_STYLES)]]
        out.append(_make_mesh(seed * 7919 + i * 131 + 17 + (k // len(_STYLES)) * 1009,
                              max(1, detail - 2) if gravel else detail, st.angularity, st, float(uvs[i])))
    return tuple(out)


def _copy_mesh(m: Mesh) -> Mesh:
    return Mesh(m.points.copy(), m.triangles.copy(), m.normals.copy(), m.uvs.copy(), m.rect)


def rock_prototypes(n: int = 8, seed: int = 0, detail: int = 4) -> list[Mesh]:
    """``n`` distinct rocks covering the style families (blocky, chunk, slab, wedge, rounded,
    long, shard, tabular), deterministic for ``(n, seed, detail)``.

    The first ``proto_split(n)`` (5 of 8) are boulder-grade (``detail`` icosphere subdivisions,
    ~5000 triangles at the default 4); the rest are gravel-grade, two subdivisions coarser
    (~320 triangles) - instancing 10k pieces of gravel with the boulder meshes would be
    wasteful for pieces that cover a handful of pixels.  ``Scatter.proto`` follows that split.
    UVs are scaled by ``proto_uv_scale(n)`` (see there) so one 1 m rock texture has the right physical
    scale on every instance.  ``detail`` is only a level of detail: a given seed is the same rock at every
    level (bounding boxes agree within a few percent), so ``Scatter`` stays valid for any ``detail``, but
    pass the ``detail`` the Scatter was made with (``Scatter.prototypes()`` does)."""
    return [_copy_mesh(m) for m in _prototypes_cached(int(n), int(seed), _clip_detail(detail))]


def proto_extents(n: int = 8, seed: int = 0, detail: int = 4) -> np.ndarray:
    """(n, 3) bounding-box size (x, y, z) of each prototype, in metres of the unit rock."""
    return np.array([m.points.max(axis=0) - m.points.min(axis=0)
                     for m in _prototypes_cached(int(n), int(seed), _clip_detail(detail))], dtype=np.float64)


_FEET_K = 32


@lru_cache(maxsize=8)
def _feet(n: int, seed: int, detail: int) -> np.ndarray:
    """(n, K, 3) contact samples of every prototype: the outline of its underside (16 rim points), 12 points
    inside it, the lowest vertex and the centroid of the underside.  x, y are metres of the unit rock about the
    bbox centre; z is the height above the rock's lowest point.  These are the points that must not hang
    in the air over the ground when the rock is set down (the underside is a near-level plane, so its rim and
    a few interior points describe it).  The underside is found from the normals (facing down), not from
    the height, so a bevelled edge or a tilted underside is not mistaken for it."""
    out = []
    for m in _prototypes_cached(n, seed, detail):
        P = m.points.astype(np.float64)
        zmin = float(P[:, 2].min())
        h = float(P[:, 2].max()) - zmin
        sel = m.normals[:, 2] < -0.8
        if sel.sum() < 6:
            sel = P[:, 2] < zmin + 0.12 * h
        low = P[sel]
        ctr = low.mean(axis=0)
        d = low[:, :2] - ctr[:2]
        r = np.hypot(d[:, 0], d[:, 1])
        b = ((np.arctan2(d[:, 1], d[:, 0]) + math.pi) / (2 * math.pi) * 16).astype(int) % 16
        pts = [low[np.nonzero(b == k)[0][np.argmax(r[b == k])]] for k in range(16) if (b == k).any()]
        inner = low[r < 0.55 * r.max()]
        if len(inner):
            pts.extend(inner[np.linspace(0, len(inner) - 1, 12).astype(int)])
        pts.append(low[np.argmin(low[:, 2])])
        pts.append(ctr)
        arr = np.array(pts)
        arr = np.concatenate([arr, np.repeat(arr[-2:-1], _FEET_K - len(arr), axis=0)])[:_FEET_K]
        arr[:, 2] -= zmin
        out.append(arr)
    return np.array(out)


# ---------------------------------------------------------------------------
# Per-vertex dust / soil (what a single shared texture cannot do)
# ---------------------------------------------------------------------------

def rock_vertex_weights(mesh: Mesh) -> np.ndarray:
    """(N, 2) float32 per vertex of a rock: ``[:, 0]`` *dust* (0..1, settles on up-facing surfaces, patchy) and
    ``[:, 1]`` *base* (0..1, 1 on the underside fading out over the lower half: where soil is splashed and
    packed against the stone).  These are the two things that make a stone sit *in* the ground instead of
    on it; one shared texture cannot express them, per-vertex data can."""
    P = mesh.points.astype(np.float64)
    n = mesh.normals.astype(np.float64)
    up = np.clip(n[:, 2], 0.0, 1.0)
    patch = 0.5 + 0.5 * _noise3(P, 0.30, 7001)
    fine = 0.5 + 0.5 * _noise3(P, 0.07, 7002)
    dust = np.clip(up ** 1.5 * (0.35 + 0.8 * patch) * (0.7 + 0.3 * fine), 0.0, 1.0)
    z0, z1 = float(P[:, 2].min()), float(P[:, 2].max())
    hrel = (P[:, 2] - z0) / max(z1 - z0, 1e-6)
    base = (1.0 - smoothstep(0.10, 0.60, hrel)) * (0.7 + 0.3 * fine)
    return np.stack([dust, np.clip(base, 0.0, 1.0)], axis=1).astype(np.float32)


def rock_vertex_tint(mesh: Mesh, dust: float = 0.5, soil: float = 0.75, base_color=(0.58, 0.55, 0.50),
                     dust_color=(0.75, 0.68, 0.56), soil_color=(0.50, 0.43, 0.33)) -> np.ndarray:
    """(N, 3) float32 multiplicative per-vertex tint built from :func:`rock_vertex_weights`: pale warm dust
    on the tops, dark brown soil on the bottom, ~1 elsewhere (``base_color`` is the albedo the tint is
    relative to).  Author it as ``primvars:displayColor`` (vertex interpolation) and multiply it into the
    albedo in the shader (a ``UsdPrimvarReader_float3`` feeding a multiply - not available in plain
    UsdPreviewSurface, where per-vertex colour can only replace the texture)."""
    w = rock_vertex_weights(mesh).astype(np.float64)
    base = np.asarray(base_color, dtype=np.float64)
    wd = (np.clip(w[:, 0] * dust, 0, 1))[:, None]
    ws = (np.clip(w[:, 1] * soil, 0, 1))[:, None]
    c = base[None] * (1 - wd) + np.asarray(dust_color)[None] * wd
    c = c * (1 - ws) + np.asarray(soil_color)[None] * ws
    return (c / base[None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Scatter
# ---------------------------------------------------------------------------

@dataclass
class Scatter:
    """Instances of the rock prototypes (see the module docstring for the transform)."""
    positions: np.ndarray            # (N, 3) float32, bbox centre of each rock
    scales: np.ndarray               # (N, 3) float32, per-axis scale of the (~1 m) prototype
    yaws: np.ndarray                 # (N,) float32, radians about +z
    proto: np.ndarray                # (N,) int32, index into rock_prototypes(n_protos, proto_seed, proto_detail)
    n_boulders: int = 0              # the first ``n_boulders`` instances are boulders, the rest gravel
    n_protos: int = 8
    proto_seed: int = 0
    shade: np.ndarray | None = None   # (N,) float32 brightness variant in [0.78, 1.12] (material / colour pick)
    proto_detail: int = 4            # the ``detail`` the instances were sized and embedded for

    def __len__(self) -> int:
        return len(self.positions)

    def prototypes(self) -> list[Mesh]:
        """The prototype meshes ``proto`` indexes: ``rock_prototypes(n_protos, proto_seed, proto_detail)``."""
        return rock_prototypes(self.n_protos, self.proto_seed, self.proto_detail)

    def transform(self, i: int, points: np.ndarray) -> np.ndarray:
        """World-space points of instance ``i`` for prototype ``points`` (K, 3)."""
        c, s = math.cos(float(self.yaws[i])), math.sin(float(self.yaws[i]))
        q = points.astype(np.float64) * self.scales[i].astype(np.float64)
        return np.stack([c * q[:, 0] - s * q[:, 1], s * q[:, 0] + c * q[:, 1], q[:, 2]], 1) + self.positions[i]


class _Ground:
    """Height of the *rendered* terrain mesh: piecewise linear on the ``grid_mesh`` triangulation."""

    def __init__(self, terrain: QuarryTerrain, rect, step: float):
        x0, x1, y0, y1 = (float(v) for v in rect)
        step = max(float(step), math.sqrt((x1 - x0) * (y1 - y0) / 2.5e6), 1e-3)     # cap the grid at ~2.5 M nodes
        self.nx = max(2, int(round((x1 - x0) / step)) + 1)
        self.ny = max(2, int(round((y1 - y0) / step)) + 1)
        self.x0, self.y0 = x0, y0
        self.dx = (x1 - x0) / (self.nx - 1)
        self.dy = (y1 - y0) / (self.ny - 1)
        self.xs = np.linspace(x0, x1, self.nx)
        self.ys = np.linspace(y0, y1, self.ny)
        self.X, self.Y = np.meshgrid(self.xs, self.ys, indexing="xy")
        self.Z = terrain.height(self.X, self.Y)

    def height(self, x, y):
        fx = np.clip((np.asarray(x, dtype=np.float64) - self.x0) / self.dx, 0.0, self.nx - 1 - 1e-9)
        fy = np.clip((np.asarray(y, dtype=np.float64) - self.y0) / self.dy, 0.0, self.ny - 1 - 1e-9)
        i, j = fx.astype(np.int64), fy.astype(np.int64)
        tx, ty = fx - i, fy - j
        za, zb = self.Z[j, i], self.Z[j, i + 1]
        zc, zd = self.Z[j + 1, i + 1], self.Z[j + 1, i]
        return np.where(tx >= ty, za + tx * (zb - za) + ty * (zc - zb), za + tx * (zc - zd) + ty * (zd - za))


def _feet_ground(G: _Ground, feet: np.ndarray, proto, scales, yaw, x, y, chunk: int = 20000):
    """Where the rock's underside may sit on the ground, from its contact samples.

    Returns ``(low, relief)``: ``low`` is the highest height the rock's lowest point can have so that no contact
    sample hangs above the (rendered) ground - the underside is level, so this is the lowest of
    ``ground - height-of-sample-above-the-underside`` - and ``relief`` is how far the ground departs from that
    level underside (spread of the same quantity)."""
    n = len(x)
    low, relief = np.empty(n), np.empty(n)
    for k in range(0, n, chunk):
        sl = slice(k, k + chunk)
        q = feet[proto[sl]] * scales[sl][:, None, :]                          # (n, K, 3)
        c, s = np.cos(yaw[sl])[:, None], np.sin(yaw[sl])[:, None]
        wx = x[sl][:, None] + c * q[..., 0] - s * q[..., 1]
        wy = y[sl][:, None] + s * q[..., 0] + c * q[..., 1]
        allowed = G.height(wx, wy) - q[..., 2]
        low[sl] = allowed.min(axis=1)
        relief[sl] = allowed.max(axis=1) - low[sl]
    return low, relief


def _power_law(rng, n: int, lo, hi, alpha: float):
    """Samples of p(s) ~ s**-alpha on [lo, hi] (lo/hi may be arrays)."""
    u = rng.random(n)
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    if abs(alpha - 1.0) < 1e-6:                    # p ~ 1/s: log-uniform
        return lo * (hi / lo) ** u
    k = 1.0 - alpha
    return (lo ** k - u * (lo ** k - hi ** k)) ** (1.0 / k)


def _in_rects(x, y, rects, margin=0.0):
    m = np.zeros(np.shape(x), bool)
    mg = np.broadcast_to(np.asarray(margin, dtype=np.float64), np.shape(x))
    for a, b, c, d in rects:
        m |= (x > a - mg) & (x < b + mg) & (y > c - mg) & (y < d + mg)
    return m


def _node_fields(terrain: QuarryTerrain, G: _Ground):
    """Fields on the ground grid nodes: lane, slope, toe (concavity), edge proximity."""
    X, Y, Z = G.X, G.Y, G.Z
    lane = terrain.lane_mask(X, Y)
    k = np.array([1.0, 2.0, 1.0]) / 4.0
    pad = np.pad(Z, 1, mode="edge")
    Zs = sum(k[a] * k[b] * pad[a:a + Z.shape[0], b:b + Z.shape[1]] for a in range(3) for b in range(3))
    gy, gx = np.gradient(Zs, G.dy, G.dx)
    slope = np.clip(np.hypot(gx, gy) / 1.2, 0.0, 1.0)
    sx, sy = max(1, int(round(0.5 / G.dx))), max(1, int(round(0.5 / G.dy)))
    p2 = np.pad(Zs, ((sy, sy), (sx, sx)), mode="edge")
    h, w = Zs.shape
    lap = ((p2[sy:sy + h, 2 * sx:2 * sx + w] + p2[sy:sy + h, 0:w] - 2 * Zs) / (sx * G.dx) ** 2
           + (p2[2 * sy:2 * sy + h, sx:sx + w] + p2[0:h, sx:sx + w] - 2 * Zs) / (sy * G.dy) ** 2)
    toe = np.clip(lap / 0.9, 0.0, 1.0)
    left = Y - terrain.bench_toe(X)
    right = (terrain.right_berm_center(X) + 1.3) - Y
    out = np.maximum(left, right)                         # > 0 outside the lane, < 0 inside
    start = smoothstep(terrain.gate_x - 30.0, terrain.gate_x - 20.0, X)
    edge = np.exp(-(((out - 0.35) / 1.3) ** 2)) * start
    return lane, slope, toe, edge, out


def _cell_mean(a):
    return 0.25 * (a[:-1, :-1] + a[:-1, 1:] + a[1:, :-1] + a[1:, 1:])


def _blur(a: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Separable Gaussian blur (sigma ``sx`` / ``sy`` cells along x / y, edge-replicated)."""
    out = a
    for axis, sig in ((1, sx), (0, sy)):
        if sig < 0.3:
            continue
        r = int(math.ceil(3.0 * sig))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
        k /= k.sum()
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        pa = np.pad(out, pad, mode="edge")
        n = out.shape[axis]
        out = sum(k[i] * (pa[:, i:i + n] if axis == 1 else pa[i:i + n, :]) for i in range(2 * r + 1))
    return out


class _SpacingGrid:
    """Hash grid of placed boulders (x, y, size) for the minimum-spacing test."""

    def __init__(self, cell: float):
        self.cell = float(cell)
        self.cells: dict[tuple[int, int], list[tuple[float, float, float]]] = {}

    def add(self, x: float, y: float, s: float) -> None:
        self.cells.setdefault((int(x // self.cell), int(y // self.cell)), []).append((x, y, s))

    def ok(self, x: float, y: float, s: float, factor: float) -> bool:
        ci, cj = int(x // self.cell), int(y // self.cell)
        for i in (ci - 1, ci, ci + 1):
            for j in (cj - 1, cj, cj + 1):
                for bx, by, bs in self.cells.get((i, j), ()):
                    if math.hypot(bx - x, by - y) < factor * 0.5 * (bs + s):
                        return False
        return True


def _check_rect(rect) -> tuple[float, float, float, float]:
    try:
        x0, x1, y0, y1 = (float(v) for v in np.asarray(rect, dtype=np.float64).reshape(-1))
    except (TypeError, ValueError) as e:
        raise ValueError(f"rect must be (x0, x1, y0, y1), got {rect!r}") from e
    if not all(math.isfinite(v) for v in (x0, x1, y0, y1)) or not (x1 > x0 and y1 > y0):
        raise ValueError(f"rect must be finite with x1 > x0 and y1 > y0, got {rect!r}")
    return x0, x1, y0, y1


def _check_keep(keep_clear) -> list[tuple[float, float, float, float]]:
    if keep_clear is None:
        return []
    a = np.asarray(keep_clear, dtype=np.float64)
    if a.size == 0:
        return []
    if a.ndim == 1 and a.size == 4:
        a = a[None]
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError(f"keep_clear must be a sequence of (x0, x1, y0, y1) rects, got shape {a.shape}")
    return [tuple(float(v) for v in r) for r in a]


def scatter(terrain: QuarryTerrain, rect, seed: int = 0, keep_clear=None, boulders: int = 120,
            gravel_per_m2: float = 1.2, *, n_protos: int = 8, proto_seed: int = 0, detail: int = 4,
            grid_step: float = 0.25, size_alpha: float = 1.6, size_range=(0.25, 1.2),
            min_aspect: float = 0.5, max_aspect: float = 1.0, focus=None, focus_gain: float = 3.0) -> Scatter:
    """Boulders and gravel on ``terrain`` inside ``rect = (x0, x1, y0, y1)``.

    * boulders: ``size_range`` = 0.25-1.2 m across, power-law sizes ``p(s) ~ s**-size_alpha`` (1.6 gives a median
      of ~0.45 m with ~40% over 0.5 m and a few over 1 m; the 2.3 this replaces left the clip a handful of readable
      blocks), exactly ``boulders`` of them unless the rect is too small, some with smaller companions.  They
      pile up where blasted spoil comes to rest: the strip along the lane edges (the strongest weight; a
      windrow/rock-fall zone spread ~1 m about the edge and clumped in 2-3 m piles, not a bead-string), concave
      slope toes (berm feet, bench toe) and the shoulder just outside, thinning out quickly with distance
      from the haul road, so not on the bench top or far out.  ``focus`` (a rect or list of rects, e.g. the area
      the camera sees) multiplies the boulder weight there by ``focus_gain`` (3) and the gravel density by its
      square root.  Never on the lane (not overhanging it by more than ~20% of their size), never in
      ``keep_clear`` rects, never on a steep face, never where the ground departs from a level underside by
      more than 40% of the rock's height.  A boulder's height is held between ``min_aspect`` (0.5) and
      ``max_aspect`` (1.0) times its width, so a slab is a thick slab: the top of a typical boulder stands
      ~55% of its height above the ground (>= 20% for one lodged against a slope).
    * gravel (0.03-0.15 m across): ``gravel_per_m2`` pieces per m2 on open ground (so the
      total scales with area x density), ~3x denser along the lane edges, 5% of that on
      the compacted lane (small spillage only, <= 7 cm), denser at slope toes and sparser
      on steep faces, plus a debris halo around larger boulders.  Not in ``keep_clear`` and
      not under boulders.

    Sizes are the largest horizontal extent of the placed rock in metres.  Every rock sits partly embedded: the
    contact samples of its underside (rim and interior of the *actual* prototype's underside, along the
    rotated and scaled footprint) are all at or below the ground, the highest of them 25-40% of the rock's
    height below it, so no part of the underside hangs in the air; a rock on a slope is therefore buried on
    its uphill side and stands free on the downhill one.  The ground is the piecewise-linear surface of
    ``grid_mesh(terrain, rect, grid_step)`` - pass the rect and step the terrain mesh is built with (the
    scenario's ``local_rect``, 0.25 m) and the contact matches what is rendered.

    ``n_protos`` / ``proto_seed`` / ``detail`` select the prototypes: build them with
    ``rock_prototypes(sc.n_protos, sc.proto_seed, sc.proto_detail)`` (= ``sc.prototypes()``); a rock is
    the same at every ``detail`` (only the tessellation changes) so the choice is free, but the sizes and
    embedding are computed for the one given here.  Deterministic for a given argument set (any integer
    ``seed``); the first ``n_boulders`` instances are the boulders.  Raises ``ValueError`` for an empty or
    inverted rect, ``n_protos < 1`` or a bad ``size_range`` / ``min_aspect``.
    """
    seed31 = int(seed) & 0x7FFFFFFF
    rng = np.random.default_rng([seed31, 0x5CA77E2])
    x0, x1, y0, y1 = _check_rect(rect)
    keep = _check_keep(keep_clear)
    n_protos = int(n_protos)
    if n_protos < 1:
        raise ValueError("n_protos must be >= 1")
    proto_seed = int(proto_seed)
    detail = _clip_detail(detail)
    boulders = max(0, int(boulders))
    gravel_per_m2 = max(0.0, float(gravel_per_m2))
    s_lo, s_hi = (float(v) for v in size_range)
    if not (0.0 < s_lo < s_hi):
        raise ValueError(f"size_range must satisfy 0 < lo < hi, got {size_range!r}")
    size_alpha = float(size_alpha)
    min_aspect, max_aspect = float(min_aspect), float(max_aspect)
    if not (0.0 < min_aspect <= max_aspect):
        raise ValueError("need 0 < min_aspect <= max_aspect")
    relief_max = 0.40                                   # of a boulder's height (ground vs level underside)
    focus_r = _check_keep(focus)
    focus_gain = float(focus_gain)
    ext = proto_extents(n_protos, proto_seed, detail)
    feet = _feet(n_protos, proto_seed, detail)
    nbp = proto_split(n_protos)
    G = _Ground(terrain, (x0, x1, y0, y1), grid_step)
    lane_n, slope_n, toe_n, edge_n, out_n = _node_fields(terrain, G)
    cell_w, cell_h = G.dx, G.dy
    nxc = G.nx - 1
    xc = 0.5 * (G.X[:-1, :-1] + G.X[1:, 1:])
    yc = 0.5 * (G.Y[:-1, :-1] + G.Y[1:, 1:])
    clump = np.exp(1.0 * fbm(xc, yc, 9.0, 3, seed31 + 11))
    lane_c = _cell_mean(lane_n)
    free = ~_in_rects(xc, yc, keep, 0.0)
    boost = np.where(_in_rects(xc, yc, focus_r, 0.0), max(focus_gain, 0.0), 1.0)     # camera view zone

    def boulder_scales(sz, proto, ay, az):
        """Instance scales of boulders of horizontal size ``sz``; height / size is held in [min_aspect, max_aspect]."""
        e = ext[proto]
        f = sz / np.maximum(e[:, 0], e[:, 1] * ay)
        fz = np.maximum(f * e[:, 2], 1e-9)
        az_eff = np.clip(az, min_aspect * sz / fz, max_aspect * sz / fz)
        return np.stack([f, f * ay, f * az_eff], axis=1)

    def site_ok(cx, cy, sz, proto, ay, az, yaw):
        """Boulder rules for centres ``(cx, cy)``: inside the rect, footprint off the lane (overhang <= ~20% of
        the size), out of keep_clear, and the ground under the underside close to level."""
        ok = (cx > x0) & (cx < x1) & (cy > y0) & (cy < y1)
        for ddx, ddy in ((0.3, 0), (-0.3, 0), (0, 0.3), (0, -0.3), (0, 0)):
            ok &= terrain.lane_mask(cx + ddx * sz, cy + ddy * sz) < 0.05
        ok &= ~_in_rects(cx, cy, keep, 0.3 * sz)
        if ok.any():
            sc_ = boulder_scales(sz, proto, ay, az)
            _, relief = _feet_ground(G, feet, proto, sc_, yaw, cx, cy)
            ok &= relief < relief_max * sc_[:, 2] * ext[proto][:, 2]
        return ok

    # ---------------- boulders -------------------------------------------------------------------
    bld: list[tuple[float, float, float, int, float, float, float]] = []   # x, y, size, proto, ay, az, yaw
    if boulders > 0:
        slope_c = _cell_mean(slope_n)
        # loose blocks come to rest at slope toes and along the lane edge (the windrow), not stuck on steep faces
        out_c = _cell_mean(out_n)
        far = 0.008 + 0.05 * np.exp(-np.maximum(out_c, 0.0) / 5.0)       # spoil is scattered mostly near the haul road
        shoulder = np.exp(-(((out_c - 2.6) / 2.0) ** 2))
        high = smoothstep(1.2, 2.2, _cell_mean(G.Z))                      # bench top: seen only at grazing angles
        # Rock fall and spoil do not form a bead-string along the toe: spread the weight ~1 m about the edge
        # and toe lines, clump it in 2-3 m piles, and only then apply the hard rules
        spread = _blur(far + 2.6 * _cell_mean(toe_n) + 3.2 * _cell_mean(edge_n) * (out_c > -0.2) + 0.7 * shoulder,
                       0.9 / cell_w, 0.9 / cell_h)
        piles = np.exp(1.6 * fbm(xc, yc, 2.6, 2, seed31 + 53))
        wb = (spread * (0.45 + 0.55 * smoothstep(0.1, 1.4, out_c))            # not a bead-string right at the lane edge
              * (1.0 - 0.92 * smoothstep(0.30, 0.62, slope_c)) * (1.0 - 0.9 * high)) * clump * piles * boost
        wb = np.where((lane_c < 0.05) & free, wb, 0.0)
        wsum = wb.sum()
        if wsum > 0:
            p = (wb / wsum).ravel()
            grid = _SpacingGrid(max(1.5, 0.8 * s_hi))
            tries_per_round = max(400, boulders * 30)
            relax = 1.0
            for _ in range(5):
                if len(bld) >= boulders:
                    break
                T_ = tries_per_round
                cells = rng.choice(p.size, size=T_, p=p)
                cy_i, cx_i = np.divmod(cells, nxc)
                cx = G.x0 + (cx_i + rng.random(T_)) * cell_w
                cy = G.y0 + (cy_i + rng.random(T_)) * cell_h
                sz = _power_law(rng, T_, s_lo, s_hi, size_alpha)
                pr = rng.integers(0, nbp, T_)
                ay = rng.uniform(0.75, 1.15, T_)
                az = rng.uniform(1.00, 1.70, T_)
                yw = rng.uniform(-math.pi, math.pi, T_)
                ok = site_ok(cx, cy, sz, pr, ay, az, yw)
                sat_roll = rng.random((T_, 3))
                sat_ang = rng.uniform(0, 2 * math.pi, (T_, 3))
                sat_sz = rng.uniform(0.3, 0.55, (T_, 3))
                sat_gap = rng.uniform(0.78, 1.0, (T_, 3))
                sat_pr = rng.integers(0, nbp, (T_, 3))
                sat_ay = rng.uniform(0.75, 1.15, (T_, 3))
                sat_az = rng.uniform(1.00, 1.70, (T_, 3))
                sat_yw = rng.uniform(-math.pi, math.pi, (T_, 3))
                for k in np.nonzero(ok)[0]:
                    if len(bld) >= boulders:
                        break
                    if not grid.ok(cx[k], cy[k], sz[k], 0.8 * relax):
                        continue
                    grid.add(float(cx[k]), float(cy[k]), float(sz[k]))
                    bld.append((float(cx[k]), float(cy[k]), float(sz[k]), int(pr[k]), float(ay[k]), float(az[k]), float(yw[k])))
                    if sz[k] >= 0.55:
                        for m in range(3):
                            if len(bld) >= boulders or sat_roll[k, m] > 0.45:
                                continue
                            cs = max(s_lo, sz[k] * sat_sz[k, m])
                            dist = 0.5 * (sz[k] + cs) * sat_gap[k, m]
                            sx_ = cx[k] + dist * math.cos(sat_ang[k, m])
                            sy_ = cy[k] + dist * math.sin(sat_ang[k, m])
                            if not site_ok(np.array([sx_]), np.array([sy_]), np.array([cs]), sat_pr[k, m:m + 1],
                                           sat_ay[k, m:m + 1], sat_az[k, m:m + 1], sat_yw[k, m:m + 1])[0]:
                                continue
                            if grid.ok(sx_, sy_, cs, 0.7 * relax):
                                grid.add(float(sx_), float(sy_), float(cs))
                                bld.append((float(sx_), float(sy_), float(cs), int(sat_pr[k, m]), float(sat_ay[k, m]),
                                            float(sat_az[k, m]), float(sat_yw[k, m])))
                relax *= 0.6
    nb = len(bld)
    B = np.array(bld, dtype=np.float64).reshape(nb, 7)
    bx, by, bs = B[:, 0], B[:, 1], B[:, 2]

    # ---------------- gravel ---------------------------------------------------------------------
    gx_l, gy_l = [], []
    if gravel_per_m2 > 0:
        wg = ((1.0 - 0.95 * lane_c) * (1.0 + 2.0 * _cell_mean(edge_n))
              * (1.0 + 0.8 * _cell_mean(toe_n)) * (1.0 - 0.6 * smoothstep(0.35, 0.75, _cell_mean(slope_n)))
              * np.exp(0.35 * fbm(xc, yc, 4.0, 3, seed31 + 29)))
        wg = np.where(free, wg * np.sqrt(boost), 0.0)
        lam = gravel_per_m2 * wg * cell_w * cell_h
        cnt = rng.poisson(lam)
        ci = np.repeat(np.arange(cnt.size), cnt.ravel())
        cy_i, cx_i = np.divmod(ci, nxc)
        gx_l.append(G.x0 + (cx_i + rng.random(len(ci))) * cell_w)
        gy_l.append(G.y0 + (cy_i + rng.random(len(ci))) * cell_h)
        # debris halo around bigger boulders
        for x, y, s in zip(bx, by, bs):
            if s < 0.4:
                continue
            ring = math.pi * ((1.15 * s) ** 2 - (0.5 * s) ** 2)
            n_h = rng.poisson(gravel_per_m2 * 2.5 * ring)
            ang = rng.uniform(0, 2 * math.pi, n_h)
            rad = np.sqrt(rng.uniform((0.5 * s) ** 2, (1.15 * s) ** 2, n_h))
            gx_l.append(x + rad * np.cos(ang))
            gy_l.append(y + rad * np.sin(ang))
    gx = np.concatenate(gx_l) if gx_l else np.zeros(0)
    gy = np.concatenate(gy_l) if gy_l else np.zeros(0)
    keep_g = (gx > x0) & (gx < x1) & (gy > y0) & (gy < y1) & ~_in_rects(gx, gy, keep, 0.0)
    gx, gy = gx[keep_g], gy[keep_g]
    if len(gx) and len(bx):                                # not under boulders
        under = np.zeros(len(gx), bool)
        for k in range(0, len(gx), 20000):
            d = np.hypot(gx[k:k + 20000, None] - bx[None], gy[k:k + 20000, None] - by[None])
            under[k:k + 20000] = (d < 0.42 * bs[None]).any(axis=1)
        gx, gy = gx[~under], gy[~under]
    if len(gx):
        lane_g = terrain.lane_mask(gx, gy)
        gs = _power_law(rng, len(gx), 0.03, np.where(lane_g > 0.5, 0.07, 0.15), 2.0)
    else:
        gs = np.zeros(0)

    # ---------------- assemble ---------------------------------------------------------------------
    ng = len(gx)
    n = nb + ng
    X = np.concatenate([bx, gx]); Y = np.concatenate([by, gy])
    g_lo = nbp if nbp < n_protos else 0
    proto = np.concatenate([B[:, 3].astype(np.int64), rng.integers(g_lo, n_protos, ng)])
    ay = np.concatenate([B[:, 4], rng.uniform(0.7, 1.2, ng)])
    az = np.concatenate([B[:, 5], rng.uniform(0.55, 1.0, ng)])
    yaw = np.concatenate([B[:, 6], rng.uniform(-math.pi, math.pi, ng)])
    scales = np.empty((n, 3))
    if nb:
        scales[:nb] = boulder_scales(bs, proto[:nb], ay[:nb], az[:nb])
    if ng:
        e = ext[proto[nb:]]
        f = gs / np.maximum(e[:, 0], e[:, 1] * ay[nb:])
        scales[nb:] = np.stack([f, f * ay[nb:], f * az[nb:]], axis=1)
    height = scales[:, 2] * ext[proto][:, 2]
    embed = rng.uniform(0.25, 0.40, n)
    low, _ = _feet_ground(G, feet, proto, scales, yaw, X, Y)       # lowest the underside may sit on the ground
    Z = low - embed * height + 0.5 * height
    shade = rng.uniform(0.78, 1.12, n)
    return Scatter(np.stack([X, Y, Z], 1).astype(np.float32), scales.astype(np.float32), yaw.astype(np.float32),
                   proto.astype(np.int32), nb, n_protos, proto_seed, shade.astype(np.float32), detail)


# ---------------------------------------------------------------------------
# Contact shading for the terrain texture
# ---------------------------------------------------------------------------

def contact_map(sc: Scatter, rect, px_per_m: float = 8.0, min_size: float = 0.2, strength: float = 0.34) -> np.ndarray:
    """(H, W, 3) float32 multiplicative colour map over ``rect`` (row 0 = y1, like ``bake_albedo``) that darkens
    the ground around every boulder: the contact shadow / ambient occlusion and the packed soil at a stone's foot.

    A rock that touches the ground along a hard line is the classic CG tell, and a shared rock texture cannot
    soften it.  The *terrain* texture can: multiply this map into ``bake_albedo(...)`` of the same ``rect`` and
    every boulder gets a dark, soft halo (width ~0.1 m + 20% of the rock's size, warmer than neutral grey) that fades
    into the ground.  ``strength`` is the maximum darkening at the rock's edge."""
    x0, x1, y0, y1 = _check_rect(rect)
    W = max(1, int(round((x1 - x0) * px_per_m)))
    H = max(1, int(round((y1 - y0) * px_per_m)))
    out = np.ones((H, W), np.float32)
    if sc.n_boulders:
        ext = proto_extents(sc.n_protos, sc.proto_seed, sc.proto_detail)
        for i in range(sc.n_boulders):
            e = ext[int(sc.proto[i])]
            a, b = 0.5 * e[0] * float(sc.scales[i, 0]), 0.5 * e[1] * float(sc.scales[i, 1])
            if 2.0 * max(a, b) < min_size:
                continue
            cx, cy = float(sc.positions[i, 0]), float(sc.positions[i, 1])
            reach = max(a, b) + 0.35 + 0.5 * max(a, b)
            c0, c1 = int(math.floor((cx - reach - x0) * px_per_m)), int(math.ceil((cx + reach - x0) * px_per_m))
            r0, r1 = int(math.floor((y1 - (cy + reach)) * px_per_m)), int(math.ceil((y1 - (cy - reach)) * px_per_m))
            c0, c1, r0, r1 = max(c0, 0), min(c1, W), max(r0, 0), min(r1, H)
            if c0 >= c1 or r0 >= r1:
                continue
            xs = x0 + (np.arange(c0, c1) + 0.5) / px_per_m - cx
            ys = y1 - (np.arange(r0, r1) + 0.5) / px_per_m - cy
            dx, dy = np.meshgrid(xs, ys, indexing="xy")
            cyaw, syaw = math.cos(float(sc.yaws[i])), math.sin(float(sc.yaws[i]))
            u, v = cyaw * dx + syaw * dy, -syaw * dx + cyaw * dy
            r = np.hypot(u / a, v / b)
            d = np.maximum(r - 0.85, 0.0) * 0.5 * (a + b)             # ~metres beyond the rock's edge
            width = 0.07 + 0.20 * (a + b)
            f = 1.0 - strength * np.exp(-((d / width) ** 2))
            win = out[r0:r1, c0:c1]
            np.minimum(win, f.astype(np.float32), out=win)
    exponent = np.array([1.0, 1.06, 1.16], dtype=np.float32)          # deeper shadow = warmer, browner
    return (out[..., None] ** exponent).astype(np.float32)
