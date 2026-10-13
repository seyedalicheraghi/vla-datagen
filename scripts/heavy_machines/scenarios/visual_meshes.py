"""Render-quality meshes for the machine primitives (numpy only).

The URDF / Isaac importer turns every ``Box``, ``Cylinder`` and ``Sphere`` of a
:class:`~heavy_machines.spec.MachineSpec` into a razor-sharp primitive without
UVs.  Real machine panels are never razor sharp: every edge is a small fillet
that catches a highlight, and every surface carries texture coordinates so that
paint, steel and rubber textures can be applied at a constant physical scale.
This module builds those meshes in pure numpy so the Isaac Sim runner can author
them as ``UsdGeom.Mesh`` prims (instead of the importer's primitives) and bind
``UsdPreviewSurface`` materials to them.

Conventions
-----------
* Meshes live in the LINK frame (``shape.xyz`` / ``shape.rpy`` are applied).
* Triangle winding is counter-clockwise seen from outside, i.e.
  ``cross(b - a, c - a)`` points out of the solid.  Vertex normals are outward,
  unit length and smooth across fillets (flat on planar faces).
* UVs are in METRES of surface: one UV unit is one metre, so a tileable texture
  (``wrap = repeat``) keeps its physical size whatever the part size.  Each box
  face is mapped planar with *u x v = outward normal* (text is never mirrored,
  ``v`` points up on the sides); a cylinder side is unrolled (u = arc length,
  v = distance along the profile) and its caps are planar; spheres use
  u = equator arc length, v = meridian arc length.  Fillets are unfolded by arc
  length, so their texel density is correct too.  UV islands meet at hard seams
  (a face / fillet pair shares positions and normals but not UVs).

Construction
------------
A rounded box is the Minkowski sum of an inner box (half-extent ``h - r``) and a
sphere of radius ``r``.  Each of the six faces is a small grid whose samples are
points ``q`` of the sharp box surface, mapped by ``p = c + r * n`` with
``c = clamp(q, -(h - r), h - r)`` and ``n = normalize(q - c)``.  Along every axis
the grid has one flat interval between the fillet tangent lines and ``k``
angularly uniform intervals covering the half of each fillet that belongs to
that face, so neighbouring faces land on bit-identical positions along their
shared edge (the mesh is watertight once vertices are welded by position).

Machine-level improvements (:func:`machine_visual_meshes`)
----------------------------------------------------------
* **Tyres** (:func:`tire_shapes`): the rubber cylinder of a rolling link gets a
  real tyre cross-section instead of a rounded can: crowned tread, rounded
  shoulders and a sidewall that bulges out to the full width and dips towards
  the bead; the sidewall is mapped planar (lettering / dust never smears) and the
  tread is unrolled (u = arc length) so a tread-pattern texture keeps its scale.
  The rule is explicit (rubber material, child of a ``wheel`` / ``roll_*`` joint
  with ``wheel_radius > 0``, radius within 10 % of it, width <= 1.2 x radius), so
  a steel roller drum or a track idler is never mistaken for a tyre.
* **Panel fillets scale with the part** (``bevel_scale``): large painted panels
  (hoods, cab roofs, counterweights, dump bodies) get a radius proportional to
  their middle dimension, capped at ``max_bevel``; small parts keep ``bevel``.
* **Coplanar faces and coincident walls are separated**
  (:func:`resolve_coplanar_machine`, per link :func:`resolve_coplanar`): authored
  shapes often have faces in exactly the same plane (a trim plate flush with a
  panel, a rod cap flush with a barrel) or coaxial cylinders of equal radius (a
  sprocket inside the rounded end of its track).  A rasteriser or RTX renderer
  resolves such ties arbitrarily, so the surface z-fights and shimmers under
  camera motion, and because every part carries its own texture phase the
  flicker shows as a patchwork of paint / grain.  The smaller face (the
  later-listed one on a tie) is pushed 1 mm outward (a cylinder wall grows 1 mm
  in radius), which makes the result deterministic and invisible.  Chains are
  stacked by size, rings of identical segments need only two levels, and faces on
  different links are only separated when no joint motion can ever make them
  cross (every joint on the path is fixed, spins about the face normal / the
  cylinder axis, or slides inside the plane / along the axis).
"""

from __future__ import annotations

import dataclasses
import math
import zlib
from functools import lru_cache

import numpy as np

from ..kinematics import Tf, joint_frames, link_transforms
from ..spec import ROLLING_ROLES, Box, Cylinder, MachineSpec, Shape, Sphere
from .terrain import Mesh

__all__ = ["shape_mesh", "machine_visual_meshes", "merge_meshes", "fillet_radius",
           "tire_shapes", "resolve_coplanar", "resolve_coplanar_machine", "find_coplanar_pairs",
           "find_coaxial_pairs"]

_RECT = (0.0, 0.0, 0.0, 0.0)

# Longest chord-to-arc deviation tolerated on round silhouettes (metres).  Wheels
# and drums of a 100 t haul truck would look like polygons at the default 24
# segments, so big circles get more segments (see _circle_segments).
_SILHOUETTE_TOL = 0.003

# Tyre shoulders: a real tyre's tread edge is rounded with a radius of roughly a
# seventh of the section width, far more than the generic panel bevel.
TIRE_SHOULDER = 0.14
_TIRE_CROWN = 0.03            # radial drop of the tread from centre line to shoulder, x width
_TIRE_SIDEWALL_DIP = 0.05     # sidewall is this fraction of the half width narrower at the bead
_TIRE_BEAD = 0.48             # bead radius / tyre radius (the hub covers it)

# Large painted panels get fillets that scale with the part (see fillet_radius):
# radius = bevel_scale * middle dimension, within [bevel, max_bevel].
PANEL_BEVEL_SCALE = 0.035
MAX_PANEL_BEVEL = 0.10
PANEL_MATERIALS = ("body", "white", "red", "orange")

# Exactly coplanar faces closer than this (metres) are treated as z-fighting; the
# one that must win is pushed COPLANAR_OFFSET outwards.
COPLANAR_TOL = 5.0e-4
COPLANAR_OFFSET = 1.0e-3

# Deterministic per-part texture phase (metres): neighbouring panels of one
# material must not show the same grain / scratch pattern.
_UV_JITTER_PERIOD = 16.0


# ---------------------------------------------------------------------------
# Sizing helpers
# ---------------------------------------------------------------------------

def fillet_radius(bevel: float, *dims: float, bevel_scale: float = 0.0,
                  max_bevel: float = MAX_PANEL_BEVEL, scale_dim: float | None = None) -> float:
    """Fillet radius used for a part (0 if degenerate).

    The default rule is ``min(bevel, 0.3 * min(dims))``.  With ``bevel_scale > 0``
    the wanted radius grows with the part: ``max(bevel, min(max_bevel,
    bevel_scale * middle_dim))`` (``middle_dim`` = the median of ``dims`` unless
    ``scale_dim`` is given), still never more than ``0.3 * min(dims)`` so thin
    plates keep a thin fillet.
    """
    want = float(bevel)
    if bevel_scale > 0.0:
        sd = float(scale_dim) if scale_dim is not None else sorted(dims)[len(dims) // 2]
        want = max(want, min(float(max_bevel), float(bevel_scale) * sd))
    r = min(want, 0.3 * min(dims))
    return r if r > 1e-9 else 0.0


def _fillet_segments(r: float, segments: int) -> int:
    """Arc subdivisions of one 90 degree fillet."""
    base = max(2, int(segments) // 6)
    return int(min(12, max(base, min(10, math.ceil(r / 0.015)))))


def _circle_segments(radius: float, segments: int, shrink_small: bool = True) -> int:
    """Segments round a circle: ``segments`` for ordinary parts, fewer for tiny
    cylinders (bolts, pins), more for big ones so the polygonal silhouette stays
    under ~3 mm."""
    segments = max(int(segments), 4)
    if shrink_small and radius < 0.05:
        segments = max(8, segments // 2)
    need = math.ceil(math.pi * math.sqrt(radius / (2.0 * _SILHOUETTE_TOL)))
    n = min(max(segments, need), 4 * segments)
    return n + (-n) % 4            # multiple of 4: the axis-aligned extremes are exact vertices


class _Builder:
    """Accumulates indexed geometry from several patches."""

    def __init__(self) -> None:
        self.p: list[np.ndarray] = []
        self.n: list[np.ndarray] = []
        self.uv: list[np.ndarray] = []
        self.t: list[np.ndarray] = []
        self.count = 0

    def add(self, p, n, uv, tris) -> None:
        p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        self.p.append(p)
        self.n.append(np.asarray(n, dtype=np.float64).reshape(-1, 3))
        self.uv.append(np.asarray(uv, dtype=np.float64).reshape(-1, 2))
        self.t.append(np.asarray(tris, dtype=np.int64).reshape(-1, 3) + self.count)
        self.count += len(p)

    def arrays(self):
        return (np.concatenate(self.p), np.concatenate(self.n),
                np.concatenate(self.uv), np.concatenate(self.t))


def _grid_tris(nu: int, nv: int) -> np.ndarray:
    """Triangles of an (nu x nv) vertex grid indexed ``i * nv + j`` (u = i, v = j).

    With u x v pointing out of the surface the triangles are counter-clockwise
    seen from outside.
    """
    i, j = np.meshgrid(np.arange(nu - 1), np.arange(nv - 1), indexing="ij")
    a = (i * nv + j).ravel()
    b = ((i + 1) * nv + j).ravel()
    c = ((i + 1) * nv + j + 1).ravel()
    d = (i * nv + j + 1).ravel()
    return np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------

# (normal axis, normal sign, u axis, u sign, v axis, v sign), u x v = outward normal.
# v is "up" on the four side faces; text on a texture reads correctly from outside.
_BOX_FACES = (
    (2, +1, 0, +1, 1, +1),     # +z  u = +x  v = +y
    (2, -1, 0, +1, 1, -1),     # -z  u = +x  v = -y
    (0, +1, 1, +1, 2, +1),     # +x  u = +y  v = +z
    (0, -1, 1, -1, 2, +1),     # -x  u = -y  v = +z
    (1, +1, 0, -1, 2, +1),     # +y  u = -x  v = +z
    (1, -1, 0, +1, 2, +1),     # -y  u = +x  v = +z
)


def _axis_samples(h: float, r: float, k: int) -> np.ndarray:
    """Ascending sample positions along one face axis (symmetric about 0).

    ``k + 1`` samples on each side: the fillet tangent line ``h - r`` followed by
    points whose surface normals tilt by equal steps up to 45 degrees (the other
    45 degrees of the fillet belong to the neighbouring face), ending exactly at
    ``h`` (the sharp-box edge), so adjacent faces agree bit for bit.
    """
    if r <= 0.0:
        return np.array([-h, h])
    ang = (math.pi / 4.0) * np.arange(k + 1) / k
    pos = (h - r) + r * np.tan(ang)
    pos[0] = h - r
    pos[-1] = h
    return np.concatenate([-pos[::-1], pos])


@lru_cache(maxsize=2048)
def _box_local(size: tuple, r: float, segments: int):
    hv = np.array([0.5 * s for s in size], dtype=np.float64)
    k = max(1, (_fillet_segments(r, segments) + 1) // 2) if r > 0 else 0
    inner = hv - r
    out = _Builder()
    for ni, ns, ui, us, vi, vs in _BOX_FACES:
        tu = _axis_samples(hv[ui], r, k)
        tv = _axis_samples(hv[vi], r, k)
        nu, nv = len(tu), len(tv)
        q = np.empty((nu, nv, 3))
        q[..., ui] = (us * tu)[:, None]
        q[..., vi] = (vs * tv)[None, :]
        q[..., ni] = ns * hv[ni]
        if r > 0.0:
            c = np.clip(q, -inner, inner)
            d = q - c
            nrm = d / np.sqrt((d * d).sum(-1, keepdims=True))
            p = c + r * nrm
            # unfolded in-face coordinates (arc length along the fillet)
            fu = us * c[:, 0, ui] + r * np.arctan2(us * d[:, 0, ui], r)
            fv = vs * c[0, :, vi] + r * np.arctan2(vs * d[0, :, vi], r)
        else:
            nrm = np.zeros_like(q)
            nrm[..., ni] = ns
            p = q
            fu, fv = tu, tv
        uv = np.stack(np.broadcast_arrays((fu - fu[0])[:, None], (fv - fv[0])[None, :]), axis=-1)
        out.add(p, nrm, uv, _grid_tris(nu, nv))
    return out.arrays()


# ---------------------------------------------------------------------------
# Cylinder
# ---------------------------------------------------------------------------

def _cap_fans(out: _Builder, ca, sa, rc: float, z_lo: float, z_hi: float, R: float) -> None:
    """Planar triangle-fan caps of radius ``rc`` at ``z_lo`` (facing -z) and ``z_hi`` (+z)."""
    N = len(ca)
    ring = np.stack([rc * ca, rc * sa], -1)                       # (N, 2)
    idx = np.arange(N)
    nxt = (idx + 1) % N
    for sign, z in ((+1, z_hi), (-1, z_lo)):
        pts = np.concatenate([np.array([[0.0, 0.0, z]]),
                              np.column_stack([ring, np.full(N, z)])])
        nrm = np.zeros_like(pts)
        nrm[:, 2] = sign
        # seen from outside u x v = n:  top (x, y), bottom (x, -y)
        uv = np.column_stack([pts[:, 0] + R, sign * pts[:, 1] + R])
        if sign > 0:
            tris = np.stack([np.zeros(N, np.int64), idx + 1, nxt + 1], 1)
        else:
            tris = np.stack([np.zeros(N, np.int64), nxt + 1, idx + 1], 1)
        out.add(pts, nrm, uv, tris)


@lru_cache(maxsize=1024)
def _cylinder_local(radius: float, length: float, r: float, segments: int):
    R, L = float(radius), float(length)
    N = _circle_segments(R, segments)
    a = 2.0 * math.pi * np.arange(N) / N
    ca, sa = np.cos(a), np.sin(a)
    ca_r, sa_r = np.append(ca, ca[0]), np.append(sa, sa[0])      # seam vertices are bit-identical
    u_side = R * np.append(a, 2.0 * math.pi)                      # arc length around the axis
    out = _Builder()

    # ---- side wall including both rim fillets: revolve a profile -------------------
    if r > 0.0:
        nf = _fillet_segments(r, segments)
        phi = (math.pi / 2.0) * np.arange(nf + 1) / nf
        cp, sp = np.cos(phi), np.sin(phi)
        cp[-1], sp[-1] = 0.0, 1.0
        # bottom fillet: from the cap rim (phi = 90 deg) round to the wall (phi = 0)
        b_nr, b_nz = cp[::-1], -sp[::-1]
        b_rho = (R - r) + r * b_nr
        b_z = (-L / 2.0 + r) + r * b_nz
        b_z[0] = -L / 2.0
        b_s = r * (math.pi / 2.0 - phi[::-1])
        # top fillet: from the wall up to the cap rim
        t_nr, t_nz = cp, sp
        t_rho = (R - r) + r * t_nr
        t_z = (L / 2.0 - r) + r * t_nz
        t_z[-1] = L / 2.0
        t_s = r * (math.pi / 2.0) + (L - 2.0 * r) + r * phi
        rho = np.concatenate([b_rho, t_rho])
        zz = np.concatenate([b_z, t_z])
        nr = np.concatenate([b_nr, t_nr])
        nz = np.concatenate([b_nz, t_nz])
        s = np.concatenate([b_s, t_s])
    else:
        rho = np.array([R, R])
        zz = np.array([-L / 2.0, L / 2.0])
        nr = np.array([1.0, 1.0])
        nz = np.array([0.0, 0.0])
        s = np.array([0.0, L])
    rows = len(rho)
    # grid index = angle * rows + row  (u = angle, v = profile)
    px = (ca_r[:, None] * rho[None, :])
    py = (sa_r[:, None] * rho[None, :])
    pz = np.broadcast_to(zz[None, :], px.shape)
    nx = ca_r[:, None] * nr[None, :]
    ny = sa_r[:, None] * nr[None, :]
    nzz = np.broadcast_to(nz[None, :], px.shape)
    uu = np.broadcast_to(u_side[:, None], px.shape)
    vv = np.broadcast_to(s[None, :], px.shape)
    out.add(np.stack([px, py, pz], -1), np.stack([nx, ny, nzz], -1),
            np.stack([uu, vv], -1), _grid_tris(N + 1, rows))

    # ---- caps: planar fans ------------------------------------------------------------------
    _cap_fans(out, ca, sa, R - r, -L / 2.0, L / 2.0, R)
    return out.arrays()


# ---------------------------------------------------------------------------
# Tyre
# ---------------------------------------------------------------------------

def _tire_profile(R: float, L: float, shoulder: float, segments: int):
    """Cross-section of a tyre as a (rho, z) polyline from the lower bead over the
    tread to the upper bead, plus the row indices where the lower / upper
    sidewall hands over to the shoulder (the UV seams).

    From the crown centre line outwards: a parabolic crown, a quarter-ellipse
    shoulder (radius ``shoulder`` both ways), then a sidewall that is widest where
    it leaves the shoulder (exactly ``L / 2``) and dips towards the bead.  The
    mesh therefore spans exactly ``R`` x ``L``.
    """
    hw = 0.5 * L
    s = max(1e-4, min(float(shoulder), 0.3 * L, 0.2 * R))
    crown = min(_TIRE_CROWN * L, 0.25 * (hw - s))
    rb = _TIRE_BEAD * R
    dz = _TIRE_SIDEWALL_DIP * hw
    zt = hw - s                                    # tread edge (start of the shoulder arc)
    rho_e = R - crown
    nc = 4
    nsh = max(5, _fillet_segments(s, segments))
    nsw = 8
    # crown: z = 0 .. zt (exclusive)
    zc = zt * np.arange(nc) / nc
    rc = R - crown * (zc / zt) ** 2
    rc[0] = R
    # shoulder: edge -> sidewall, quarter ellipse
    phi = (math.pi / 2.0) * (1.0 - np.arange(nsh + 1) / nsh)
    rs = (rho_e - s) + s * np.sin(phi)
    zs = zt + s * np.cos(phi)
    rs[0], zs[0] = rho_e, zt
    rs[-1], zs[-1] = rho_e - s, hw
    # sidewall: outwards-to-bead (exclusive of the shoulder end), widest at the shoulder
    u = 1.0 - np.arange(1, nsw + 1) / nsw
    rw = rb + (rho_e - s - rb) * u
    zw = hw - dz * (1.0 - u) ** 2
    rw[-1], zw[-1] = rb, hw - dz
    rho_h = np.concatenate([rc, rs, rw])
    z_h = np.concatenate([zc, zs, zw])
    n_half = len(rho_h)
    rho = np.concatenate([rho_h[1:][::-1], rho_h])
    zz = np.concatenate([-z_h[1:][::-1], z_h])
    seam_hi = (n_half - 1) + nc + nsh
    seam_lo = (n_half - 1) - (nc + nsh)
    return rho, zz, seam_lo, seam_hi


@lru_cache(maxsize=256)
def _tire_local(radius: float, length: float, shoulder: float, segments: int):
    R, L = float(radius), float(length)
    N = _circle_segments(R, segments)
    a = 2.0 * math.pi * np.arange(N) / N
    ca, sa = np.cos(a), np.sin(a)
    ca_r, sa_r = np.append(ca, ca[0]), np.append(sa, sa[0])
    u_side = R * np.append(a, 2.0 * math.pi)
    rho, zz, seam_lo, seam_hi = _tire_profile(R, L, shoulder, segments)
    # outward normal of a profile traversed bottom -> top is (dz, -drho)
    d = np.diff(np.stack([rho, zz], 1), axis=0)
    seg = np.linalg.norm(d, axis=1)
    t = d / seg[:, None]
    tv = np.empty((len(rho), 2))
    tv[0], tv[-1] = t[0], t[-1]
    tv[1:-1] = t[:-1] + t[1:]
    tv /= np.linalg.norm(tv, axis=1, keepdims=True)
    nr, nz = tv[:, 1], -tv[:, 0]
    s = np.concatenate([[0.0], np.cumsum(seg)])
    out = _Builder()
    # three patches: lower sidewall (planar UV), shoulder-to-shoulder tread (unrolled UV), upper sidewall
    for lo, hi, mode in ((0, seam_lo, "lo"), (seam_lo, seam_hi, "tread"), (seam_hi, len(rho) - 1, "hi")):
        sl = slice(lo, hi + 1)
        rr, z_, nrr, nzz, ss = rho[sl], zz[sl], nr[sl], nz[sl], s[sl]
        px = ca_r[:, None] * rr[None, :]
        py = sa_r[:, None] * rr[None, :]
        pz = np.broadcast_to(z_[None, :], px.shape)
        nx = ca_r[:, None] * nrr[None, :]
        ny = sa_r[:, None] * nrr[None, :]
        nzm = np.broadcast_to(nzz[None, :], px.shape)
        if mode == "tread":
            uv = np.stack([np.broadcast_to(u_side[:, None], px.shape),
                           np.broadcast_to((ss - ss[0])[None, :], px.shape)], -1)
        else:
            sign = 1.0 if mode == "hi" else -1.0
            uv = np.stack([px + R, sign * py + R], -1)
        out.add(np.stack([px, py, pz], -1), np.stack([nx, ny, nzm], -1), uv,
                _grid_tris(N + 1, hi - lo + 1))
    _cap_fans(out, ca, sa, float(rho[0]), float(zz[0]), float(zz[-1]), R)
    return out.arrays()


# ---------------------------------------------------------------------------
# Sphere
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _sphere_local(radius: float, segments: int):
    R = float(radius)
    N = _circle_segments(R, segments, shrink_small=False)
    K = max(6, N // 2)                                            # latitude bands (even: equator ring exists)
    K += K & 1
    lon = 2.0 * math.pi * np.arange(N + 1) / N
    lon_c = np.append(np.cos(lon[:-1]), np.cos(lon[0]))
    lon_s = np.append(np.sin(lon[:-1]), np.sin(lon[0]))
    lat = -math.pi / 2.0 + math.pi * np.arange(1, K) / K          # interior rings only
    cl, sl = np.cos(lat), np.sin(lat)
    nx = lon_c[:, None] * cl[None, :]
    ny = lon_s[:, None] * cl[None, :]
    nz = np.broadcast_to(sl[None, :], nx.shape)
    ring_n = np.stack([nx, ny, nz], -1).reshape(-1, 3)            # index i * (K-1) + j
    ring_uv = np.stack([np.broadcast_to((R * lon)[:, None], nx.shape),
                        np.broadcast_to((R * (lat + math.pi / 2.0))[None, :], nx.shape)], -1).reshape(-1, 2)
    nr = K - 1
    # poles: one vertex per longitude cell so u is continuous (cell centre)
    mid = R * (lon[:-1] + math.pi / N)
    south_n = np.tile([0.0, 0.0, -1.0], (N, 1))
    north_n = np.tile([0.0, 0.0, 1.0], (N, 1))
    south_uv = np.column_stack([mid, np.zeros(N)])
    north_uv = np.column_stack([mid, np.full(N, R * math.pi)])
    nrm = np.concatenate([ring_n, south_n, north_n])
    uv = np.concatenate([ring_uv, south_uv, north_uv])
    s0 = len(ring_n)
    n0 = s0 + N
    tris = []
    ring_idx = np.arange((N + 1) * nr).reshape(N + 1, nr)
    # ring quads: u = longitude (i), v = latitude (j)
    i, j = np.meshgrid(np.arange(N), np.arange(nr - 1), indexing="ij")
    a, b = ring_idx[i, j].ravel(), ring_idx[i + 1, j].ravel()
    c, d = ring_idx[i + 1, j + 1].ravel(), ring_idx[i, j + 1].ravel()
    tris += [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]
    ii = np.arange(N)
    tris.append(np.stack([s0 + ii, ring_idx[ii + 1, 0], ring_idx[ii, 0]], 1))             # south fan
    tris.append(np.stack([ring_idx[ii, nr - 1], ring_idx[ii + 1, nr - 1], n0 + ii], 1))   # north fan
    return R * nrm, nrm, uv, np.concatenate(tris)


# ---------------------------------------------------------------------------
# Public API: single shapes
# ---------------------------------------------------------------------------

def _transform(p, n, rot, t):
    """Apply a rigid transform with explicit element-wise arithmetic, so equal
    input rows give bit-identical output rows (welded seams stay welded)."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    px = rot[0][0] * x + rot[0][1] * y + rot[0][2] * z + t[0]
    py = rot[1][0] * x + rot[1][1] * y + rot[1][2] * z + t[1]
    pz = rot[2][0] * x + rot[2][1] * y + rot[2][2] * z + t[2]
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    qx = rot[0][0] * nx + rot[0][1] * ny + rot[0][2] * nz
    qy = rot[1][0] * nx + rot[1][1] * ny + rot[1][2] * nz
    qz = rot[2][0] * nx + rot[2][1] * ny + rot[2][2] * nz
    return np.stack([px, py, pz], 1), np.stack([qx, qy, qz], 1)


def _shape_fillet(g, bevel: float, bevel_scale: float, max_bevel: float) -> float:
    if isinstance(g, Box):
        return fillet_radius(bevel, *g.size, bevel_scale=bevel_scale, max_bevel=max_bevel)
    return fillet_radius(bevel, g.radius, g.length, bevel_scale=bevel_scale,
                         max_bevel=max_bevel, scale_dim=2.0 * g.radius)


def _build(shape: Shape, bevel: float, segments: int, uv_offset=(0.0, 0.0),
           bevel_scale: float = 0.0, max_bevel: float = MAX_PANEL_BEVEL,
           tire_shoulder: float | None = None) -> Mesh:
    g = shape.geom
    bevel = max(float(bevel), 0.0)
    if isinstance(g, Box):
        if min(g.size) <= 0.0:
            raise ValueError(f"degenerate box {g.size}")
        r = _shape_fillet(g, bevel, bevel_scale, max_bevel)
        p, n, uv, tris = _box_local(tuple(float(s) for s in g.size), r, int(segments))
    elif isinstance(g, Cylinder):
        if g.radius <= 0.0 or g.length <= 0.0:
            raise ValueError(f"degenerate cylinder r={g.radius} l={g.length}")
        if tire_shoulder is not None and tire_shoulder > 0.0:
            p, n, uv, tris = _tire_local(float(g.radius), float(g.length),
                                         float(tire_shoulder) * float(g.length), int(segments))
        else:
            r = _shape_fillet(g, bevel, bevel_scale, max_bevel)
            p, n, uv, tris = _cylinder_local(float(g.radius), float(g.length), r, int(segments))
    elif isinstance(g, Sphere):
        if g.radius <= 0.0:
            raise ValueError(f"degenerate sphere r={g.radius}")
        p, n, uv, tris = _sphere_local(float(g.radius), int(segments))
    else:
        raise TypeError(f"unknown geometry {g!r}")
    tf = Tf.from_xyz_rpy(shape.xyz, shape.rpy)
    p, n = _transform(p, n, tf.R, tf.t)
    uv = uv + np.asarray(uv_offset, dtype=np.float64)
    return Mesh(p.astype(np.float32), tris.astype(np.int32), n.astype(np.float32),
                uv.astype(np.float32), _RECT)


def shape_mesh(shape: Shape, bevel: float = 0.025, segments: int = 24, *,
               bevel_scale: float = 0.0, max_bevel: float = MAX_PANEL_BEVEL,
               tire_shoulder: float | None = None) -> Mesh:
    """Triangle mesh of one Box / Cylinder / Sphere in its LINK frame.

    ``shape.xyz`` / ``shape.rpy`` are applied.  Normals are outward unit vectors,
    UVs are in metres, triangles are counter-clockwise seen from outside.

    * Box: edges and corners rounded with ``r = min(bevel, 0.3 * min(size))``
      (smooth normals on the fillets, flat on the faces); planar UVs per face.
    * Cylinder (axis = local z): cylindrical UVs on the wall, planar on the caps,
      both rims filleted with ``r = min(bevel, 0.3 * min(radius, length))``.
    * Sphere: UV sphere.

    ``segments`` is the number of segments round a circle for ordinary parts
    (and ``segments // 6`` arc steps per 90 degree fillet); very large circles
    automatically get more so silhouettes stay round.

    Optional keywords (the defaults reproduce the rules above exactly):

    * ``bevel_scale`` > 0 makes the radius grow with the part, ``max(bevel,
      min(max_bevel, bevel_scale * middle_dim))`` (see :func:`fillet_radius`).
    * ``tire_shoulder`` > 0 turns a Cylinder into a tyre cross-section (crowned
      tread, shoulders of ``tire_shoulder * length``, bulging sidewall); the mesh
      still spans exactly ``radius`` x ``length``.
    """
    return _build(shape, bevel, segments, bevel_scale=bevel_scale, max_bevel=max_bevel,
                  tire_shoulder=tire_shoulder)


def merge_meshes(meshes: list[Mesh]) -> Mesh:
    """Concatenate meshes (indices re-based, ``rect`` zeroed)."""
    if not meshes:
        return Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32),
                    np.zeros((0, 3), np.float32), np.zeros((0, 2), np.float32), _RECT)
    offs = np.cumsum([0] + [len(m.points) for m in meshes[:-1]])
    return Mesh(np.concatenate([m.points for m in meshes]).astype(np.float32),
                np.concatenate([m.triangles.astype(np.int64) + o for m, o in zip(meshes, offs)]).astype(np.int32),
                np.concatenate([m.normals for m in meshes]).astype(np.float32),
                np.concatenate([m.uvs for m in meshes]).astype(np.float32), _RECT)


# ---------------------------------------------------------------------------
# Tyre detection
# ---------------------------------------------------------------------------

def tire_shapes(spec: MachineSpec) -> set[tuple[str, int]]:
    """``(link name, shape index)`` of every visual tyre of the machine.

    A tyre is a ``rubber`` cylinder on the child link of a rolling joint (role in
    ``heavy_machines.spec.ROLLING_ROLES`` with ``wheel_radius > 0``) whose radius is within 10 %
    of the rolling radius and whose width is at most 1.2 x its radius.  Rollers,
    drums, hubs and idlers (other materials, or squat / wide proportions) never
    qualify.
    """
    rolling = {j.child: j.wheel_radius for j in spec.joints
               if j.role in ROLLING_ROLES and j.wheel_radius > 0.0}
    out: set[tuple[str, int]] = set()
    for lk in spec.links:
        wr = rolling.get(lk.name)
        if wr is None:
            continue
        for i, sh in enumerate(lk.shapes):
            g = sh.geom
            if (sh.visual and sh.material == "rubber" and isinstance(g, Cylinder)
                    and 0.9 * wr <= g.radius <= 1.1 * wr and g.length <= 1.2 * g.radius):
                out.add((lk.name, i))
    return out


# ---------------------------------------------------------------------------
# Coplanar face separation
# ---------------------------------------------------------------------------

_FACE_INSET = 5.0e-3          # faces are compared 5 mm inside their rim (rims are filleted)
_MIN_OVERLAP = 1.0e-4         # m^2: smaller overlaps are invisible


def _face_table(shapes: list[Shape], visible: list[int], frames: list | None = None):
    """Planar faces of the box faces / cylinder caps of the listed shapes (link frame).

    Returns arrays ``centre, normal, u, v, half (F, 2), disc (F,), owner (F,), key (F,)``
    where ``key`` identifies the face inside its shape (0..5 box faces, 6 / 7 caps).
    ``frames[i]`` (optional) is the transform from shape ``i``'s link frame to the common frame.
    """
    C, Nn, U, V, H, D, O, K = [], [], [], [], [], [], [], []
    for i in visible:
        sh = shapes[i]
        g = sh.geom
        tf = Tf.from_xyz_rpy(sh.xyz, sh.rpy)
        if frames is not None:
            tf = frames[i] @ tf
        R, t = np.array(tf.R, dtype=np.float64), np.array(tf.t, dtype=np.float64)
        if isinstance(g, Box):
            h = 0.5 * np.array(g.size, dtype=np.float64)
            for ax in range(3):
                for sg in (+1, -1):
                    n = np.zeros(3)
                    n[ax] = sg
                    u = np.zeros(3)
                    u[(ax + 1) % 3] = 1.0
                    v = np.zeros(3)
                    v[(ax + 2) % 3] = 1.0
                    C.append(R @ (n * h[ax]) + t)
                    Nn.append(R @ n)
                    U.append(R @ u)
                    V.append(R @ v)
                    H.append((h[(ax + 1) % 3], h[(ax + 2) % 3]))
                    D.append(False)
                    O.append(i)
                    K.append(ax * 2 + (0 if sg > 0 else 1))
        elif isinstance(g, Cylinder):
            for sg, key in ((+1, 6), (-1, 7)):
                n = np.array([0.0, 0.0, float(sg)])
                C.append(R @ (n * 0.5 * g.length) + t)
                Nn.append(R @ n)
                U.append(R @ np.array([1.0, 0.0, 0.0]))
                V.append(R @ np.array([0.0, 1.0, 0.0]))
                H.append((g.radius, g.radius))
                D.append(True)
                O.append(i)
                K.append(key)
    if not C:
        z3, z2 = np.zeros((0, 3)), np.zeros((0, 2))
        return z3, z3, z3, z3, z2, np.zeros(0, bool), np.zeros(0, int), np.zeros(0, int)
    return (np.array(C), np.array(Nn), np.array(U), np.array(V), np.array(H),
            np.array(D, bool), np.array(O, int), np.array(K, int))


def _effective_half(half: np.ndarray, inset: float) -> np.ndarray:
    return np.maximum(half - inset, 0.5 * half)


def _face_area(half_eff: np.ndarray, disc: bool) -> float:
    return math.pi * half_eff[0] ** 2 if disc else 4.0 * half_eff[0] * half_eff[1]


def _face_polygon(tab, row: int, inset: float) -> np.ndarray:
    """Corner points (n, 3) of a face shrunk by ``inset`` (a disc becomes a 32-gon)."""
    C, _, U, V, H, D, _, _ = tab
    he = _effective_half(H[row], inset)
    if D[row]:
        ang = 2.0 * math.pi * np.arange(32) / 32.0
        a, b = he[0] * np.cos(ang), he[0] * np.sin(ang)
    else:
        a = he[0] * np.array([1.0, -1.0, -1.0, 1.0])
        b = he[1] * np.array([1.0, 1.0, -1.0, -1.0])
    return C[row] + a[:, None] * U[row] + b[:, None] * V[row]


def _ccw(poly: np.ndarray) -> np.ndarray:
    x, y = poly[:, 0], poly[:, 1]
    return poly if float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) >= 0.0 else poly[::-1]


def _clip_convex(subject: list, clip: np.ndarray) -> list:
    """Sutherland-Hodgman: the part of ``subject`` inside the CCW convex polygon ``clip``."""
    out = subject
    n = len(clip)
    for i in range(n):
        if not out:
            break
        ax, ay = clip[i]
        bx, by = clip[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        inp, out = out, []
        sx, sy = inp[-1]
        s_in = ex * (sy - ay) - ey * (sx - ax) >= 0.0
        for px, py in inp:
            p_in = ex * (py - ay) - ey * (px - ax) >= 0.0
            if p_in != s_in:
                dx, dy = px - sx, py - sy
                den = ex * dy - ey * dx
                t = (ex * (ay - sy) - ey * (ax - sx)) / den if den != 0.0 else 0.0
                out.append((sx + t * dx, sy + t * dy))
            if p_in:
                out.append((px, py))
            sx, sy, s_in = px, py, p_in
    return out


def _overlap_area(tab, fi: int, fj: int, inset: float) -> float:
    """Exact area (m^2) where the two (inset) coplanar faces overlap."""
    C, _, U, V, _, _, _, _ = tab
    pa, pb = _face_polygon(tab, fi, inset), _face_polygon(tab, fj, inset)
    org, u, v = C[fi], U[fi], V[fi]                     # common 2D frame: the first face's
    to2d = lambda p: np.stack([(p - org) @ u, (p - org) @ v], 1)
    A, B = _ccw(to2d(pa)), _ccw(to2d(pb))
    poly = _clip_convex([tuple(q) for q in A], B)
    if len(poly) < 3:
        return 0.0
    x = np.array([q[0] for q in poly])
    y = np.array([q[1] for q in poly])
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _coplanar_candidates(tab, tol: float):
    """Row index pairs ``(i, j)`` of faces of different shapes (``owner_i < owner_j``) that
    share an outward normal and a plane (within ``tol``) and whose circumcircles touch."""
    C, Nn, _, _, H, _, O, _ = tab
    if len(C) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    plane = np.einsum("ij,ij->i", Nn, C)
    ok = ((Nn @ Nn.T > 1.0 - 1.0e-5) & (np.abs(plane[:, None] - Nn @ C.T) < tol)
          & (O[:, None] < O[None, :]))
    ii, jj = np.nonzero(ok)
    rad = np.hypot(H[:, 0], H[:, 1])
    near = np.linalg.norm(C[ii] - C[jj], axis=1) < rad[ii] + rad[jj]
    return ii[near], jj[near]


def _face_areas(tab) -> list[float]:
    return [_face_area(_effective_half(tab[4][r], _FACE_INSET), bool(tab[5][r]))
            for r in range(len(tab[0]))]


def _overlapping_pairs(tab, tol: float, allowed=None,
                       min_area: float = _MIN_OVERLAP) -> list[tuple[int, int, bool]]:
    """``(top, bottom, tie)`` face rows of every overlapping coplanar pair.

    The smaller face is the one that must stay visible (a trim plate on a panel).
    Faces of equal area (a ring of identical segments) have no natural winner:
    ``tie`` is True and ``top`` is just the later-listed shape.  ``allowed(owner_i,
    owner_j, normal)`` (optional) can veto a pair.
    """
    areas = _face_areas(tab)
    out = []
    for fi, fj in zip(*_coplanar_candidates(tab, tol)):             # owner(fi) < owner(fj)
        if allowed is not None and not allowed(int(tab[6][fi]), int(tab[6][fj]), tab[1][fi]):
            continue
        ai, aj = areas[fi], areas[fj]
        tie = abs(ai - aj) <= 1e-6 * (ai + aj)
        top, bot = (fi, fj) if ai < aj and not tie else (fj, fi)
        if _overlap_area(tab, fi, fj, _FACE_INSET) >= min_area:
            out.append((int(top), int(bot), bool(tie)))
    return out


def find_coplanar_pairs(shapes: list[Shape], tol: float = COPLANAR_TOL,
                        visible_only: bool = True,
                        min_area: float = 1.5 * _MIN_OVERLAP) -> list[tuple[int, int, int, int]]:
    """Overlapping, same-direction, coplanar faces between different shapes.

    Returns ``(top_shape, top_face_key, bottom_shape, bottom_face_key)`` tuples for
    box faces (keys 0..5) and cylinder caps (6, 7) that lie within ``tol`` metres of
    one plane with the same outward normal and overlap by at least ``min_area``
    (default 1.5 cm^2, measured 5 mm inside the rims; :func:`resolve_coplanar`
    separates everything from 1 cm^2 up, so a resolved set reports nothing).  The
    "top" face is the one :func:`resolve_coplanar` raises.
    """
    vis = [i for i, s in enumerate(shapes) if (s.visual or not visible_only)]
    tab = _face_table(shapes, vis)
    return [(int(tab[6][t]), int(tab[7][t]), int(tab[6][b]), int(tab[7][b]))
            for t, b, _ in _overlapping_pairs(tab, tol, None, min_area)]


def _assign_levels(areas: list[float], C: np.ndarray, Nn: np.ndarray,
                   pairs: list[tuple[int, int, bool]], offset: float, tol: float) -> np.ndarray:
    """Push (in whole ``offset`` steps) of every row so that overlapping coplanar
    (or coaxial) faces end up at least ``tol`` apart.

    Rows are processed from big to small.  A smaller face must stand above every
    bigger face it overlaps (strict relation), so nested details stack one step
    per level.  Faces of equal area only have to differ from their neighbours, in
    either direction, so a ring of identical overlapping segments needs two
    levels in total instead of a staircase.  ``C`` / ``Nn`` give each row a point
    and the direction along which it is pushed (the gap of a pair is measured
    along it).
    """
    bigger: dict[int, list[tuple[int, int]]] = {}
    tied: dict[int, list[tuple[int, int]]] = {}
    for top, bot, tie in pairs:
        gap = float(Nn[top] @ (C[top] - C[bot]))           # top above bottom by this much (|gap| < tol)
        if tie:
            k = max(1, int(math.ceil((tol - gap) / offset - 1e-9)), int(math.ceil((tol + gap) / offset - 1e-9)))
            tied.setdefault(top, []).append((bot, k))
            tied.setdefault(bot, []).append((top, k))
        else:
            bigger.setdefault(top, []).append((bot, max(1, int(math.ceil((tol - gap) / offset - 1e-9)))))
    level = np.zeros(len(C), dtype=np.int64)
    done: set[int] = set()
    involved = {r for top, bot, _ in pairs for r in (top, bot)}
    for r in sorted(involved, key=lambda r: (-areas[r], r)):
        lv = max([level[b] + need for b, need in bigger.get(r, []) if b in done], default=0)
        while any(t in done and abs(lv - level[t]) < k for t, k in tied.get(r, [])):
            lv += 1
        level[r] = lv
        done.add(r)
    return level


# ---------------------------------------------------------------------------
# Coaxial cylinder walls
# ---------------------------------------------------------------------------

def _wall_table(shapes: list[Shape], visible: list[int], frames: list | None = None):
    """Cylinder walls: ``axis point, axis direction, radius, length, owner`` arrays."""
    P, A, Rr, Ll, O = [], [], [], [], []
    for i in visible:
        sh = shapes[i]
        if not isinstance(sh.geom, Cylinder):
            continue
        tf = Tf.from_xyz_rpy(sh.xyz, sh.rpy)
        if frames is not None:
            tf = frames[i] @ tf
        R = np.array(tf.R, dtype=np.float64)
        P.append(np.array(tf.t, dtype=np.float64))
        A.append(R[:, 2])
        Rr.append(sh.geom.radius)
        Ll.append(sh.geom.length)
        O.append(i)
    if not P:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0, int)
    return np.array(P), np.array(A), np.array(Rr), np.array(Ll), np.array(O, int)


def _wall_pairs(wt, tol: float, allowed=None, min_area: float = _MIN_OVERLAP) -> list[tuple[int, int, bool]]:
    """``(top, bottom, tie)`` wall rows of coaxial cylinders of (nearly) equal radius whose
    walls overlap axially: their curved surfaces coincide and z-fight."""
    P, A, Rr, Ll, O = wt
    if len(P) == 0:
        return []
    d = P[None, :, :] - P[:, None, :]                        # (i, j): c_j - c_i
    along = np.einsum("ijk,ik->ij", d, A)
    perp = np.linalg.norm(d - along[..., None] * A[:, None, :], axis=2)
    parallel = np.abs(A @ A.T) > 1.0 - 1.0e-5
    overlap = np.minimum(0.5 * (Ll[:, None] + Ll[None, :]) - np.abs(along), np.minimum(Ll[:, None], Ll[None, :]))
    area = 2.0 * math.pi * Rr[:, None] * overlap
    ok = (parallel & (np.abs(Rr[:, None] - Rr[None, :]) < tol) & (perp < tol)
          & (area >= min_area) & (O[:, None] < O[None, :]))
    wall = 2.0 * math.pi * Rr * Ll
    out = []
    for fi, fj in zip(*np.nonzero(ok)):
        if allowed is not None and not allowed(int(O[fi]), int(O[fj]), A[fi]):
            continue
        ai, aj = wall[fi], wall[fj]
        tie = abs(ai - aj) <= 1e-6 * (ai + aj)
        top, bot = (fi, fj) if ai < aj and not tie else (fj, fi)
        out.append((int(top), int(bot), bool(tie)))
    return out


def find_coaxial_pairs(shapes: list[Shape], tol: float = COPLANAR_TOL, visible_only: bool = True,
                       min_area: float = 1.5 * _MIN_OVERLAP) -> list[tuple[int, int]]:
    """``(top_shape, bottom_shape)`` of coaxial cylinders of equal radius (within ``tol``)
    whose walls overlap by at least ``min_area`` m^2 (their curved surfaces z-fight)."""
    vis = [i for i, s in enumerate(shapes) if (s.visual or not visible_only)]
    wt = _wall_table(shapes, vis)
    return [(int(wt[4][t]), int(wt[4][b])) for t, b, _ in _wall_pairs(wt, tol, None, min_area)]


def _resolve_walls(shapes: list[Shape], offset: float, tol: float, frames: list | None,
                   allowed) -> tuple[list[Shape], bool]:
    vis = [i for i, s in enumerate(shapes) if s.visual]
    wt = _wall_table(shapes, vis, frames)
    pairs = _wall_pairs(wt, tol, allowed)
    if not pairs:
        return list(shapes), False
    P, A, Rr, Ll, O = wt
    C = np.stack([Rr, np.zeros(len(Rr)), np.zeros(len(Rr))], 1)           # gap = radius difference
    Nn = np.tile([1.0, 0.0, 0.0], (len(Rr), 1))
    levels = _assign_levels(list(2.0 * math.pi * Rr * Ll), C, Nn, pairs, offset, tol)
    out = list(shapes)
    for row in np.nonzero(levels)[0]:
        sh = shapes[int(O[row])]
        out[int(O[row])] = dataclasses.replace(
            sh, geom=Cylinder(float(sh.geom.radius + levels[row] * offset), float(sh.geom.length)))
    return out, True


def _resolve_once(shapes: list[Shape], offset: float, tol: float, frames: list | None = None,
                  allowed=None, allowed_axis=None) -> tuple[list[Shape], bool]:
    shapes, walls_changed = _resolve_walls(shapes, offset, tol, frames, allowed_axis if allowed_axis is not None else allowed)
    vis = [i for i, s in enumerate(shapes) if s.visual]
    tab = _face_table(shapes, vis, frames)
    O, K = tab[6], tab[7]
    pairs = _overlapping_pairs(tab, tol, allowed)
    if not pairs:
        return list(shapes), walls_changed
    steps = _assign_levels(_face_areas(tab), tab[0], tab[1], pairs, offset, tol)   # whole-offset pushes per face row
    out = list(shapes)
    by_shape: dict[int, list[int]] = {}
    for row in np.nonzero(steps)[0]:
        by_shape.setdefault(int(O[row]), []).append(int(row))
    for i, rows in by_shape.items():
        sh = shapes[i]
        tf = Tf.from_xyz_rpy(sh.xyz, sh.rpy)
        R = np.array(tf.R, dtype=np.float64)
        shift = np.zeros(3)                                     # in the shape frame
        if isinstance(sh.geom, Box):
            size = np.array(sh.geom.size, dtype=np.float64)
            for row in rows:
                ax, neg = divmod(int(K[row]), 2)
                grow = steps[row] * offset
                size[ax] += grow
                shift[ax] += (-0.5 if neg else 0.5) * grow
            geom = Box(tuple(float(x) for x in size))
        else:
            length = float(sh.geom.length)
            for row in rows:
                grow = steps[row] * offset
                length += grow
                shift[2] += (-0.5 if K[row] == 7 else 0.5) * grow
            geom = Cylinder(float(sh.geom.radius), float(length))
        xyz = tuple(float(c) for c in np.array(sh.xyz, dtype=np.float64) + R @ shift)
        out[i] = dataclasses.replace(sh, geom=geom, xyz=xyz)
    return out, True


def resolve_coplanar(shapes: list[Shape], offset: float = COPLANAR_OFFSET,
                     tol: float = COPLANAR_TOL) -> list[Shape]:
    """Copies of ``shapes`` in which no two visible faces z-fight.

    For every pair of overlapping coplanar same-direction faces (box faces and
    cylinder caps of different shapes) the smaller face (the later-listed shape
    on a tie) is pushed outwards by ``offset`` metres so it reliably covers the
    other: a box grows by ``offset`` along that axis and its centre shifts by
    ``offset / 2``; a cylinder gets ``offset`` more length.  Pushes accumulate in
    whole ``offset`` steps until the winner stands at least ``tol`` proud of every
    face it overlaps (so chains of coplanar plates end up stacked); the search is
    repeated because a pushed face can land on a face the author had offset by a
    millimetre.  Shapes that do not take part are returned unchanged (the same
    objects); invisible shapes are ignored.
    """
    out = list(shapes)
    for _ in range(6):
        out, changed = _resolve_once(out, offset, tol)
        if not changed:
            break
    return out


def resolve_coplanar_machine(spec: MachineSpec, offset: float = COPLANAR_OFFSET,
                             tol: float = COPLANAR_TOL) -> dict[str, list[Shape]]:
    """:func:`resolve_coplanar` for a whole machine: ``link name -> separated shapes``.

    Faces are compared in the root frame at the ``init`` pose, so a spinning
    sprocket whose disc is flush with its track, or a boom pin boss flush with
    the stick, are separated too.  A pair of faces on different links is only
    separated when no joint motion can ever make them cross: every joint on the
    path between the two links must be fixed, rotate about an axis parallel to
    the face normal, or slide inside the face plane.  Faces that merely happen to
    coincide in one pose are left alone.
    """
    tfs = link_transforms(spec)
    jfs = joint_frames(spec)
    flat: list[Shape] = []
    frames: list[Tf] = []
    link_of: list[str] = []
    for lk in spec.links:
        for sh in lk.shapes:
            flat.append(sh)
            frames.append(tfs[lk.name])
            link_of.append(lk.name)
    by_child = {j.child: j for j in spec.joints}
    axis_root = {}
    for j in spec.joints:
        if j.name in jfs:
            a = np.array(Tf(jfs[j.name].R).apply(tuple(j.axis)), dtype=np.float64)
            axis_root[j.name] = a / max(float(np.linalg.norm(a)), 1e-12)

    def chain(link: str) -> list:
        out = []
        while link in by_child:
            out.append(by_child[link])
            link = by_child[link].parent
        return out

    path_cache: dict[tuple[str, str], list] = {}

    def locked(i: int, j: int, vec, along: bool = False) -> bool:
        """Can no joint motion between the two shapes' links separate their faces?  ``vec`` is
        the face normal (``along`` False: planes must stay put) or a cylinder axis (``along``
        True: the coaxial walls must stay put)."""
        la, lb = link_of[i], link_of[j]
        if la == lb:
            return True
        if (la, lb) not in path_cache:
            ca, cb = chain(la), chain(lb)
            na, nb = {x.name for x in ca}, {x.name for x in cb}
            path_cache[(la, lb)] = [x for x in ca if x.name not in nb] + [x for x in cb if x.name not in na]
        for jt in path_cache[(la, lb)]:
            if jt.type == "fixed":
                continue
            d = abs(float(np.dot(vec, axis_root[jt.name])))
            if jt.type in ("revolute", "continuous"):
                if d < 0.9999:                              # rotation about the normal / the cylinder axis only
                    return False
            elif along:
                if d < 0.9999:                              # slide along the cylinder axis only
                    return False
            elif d > 1.0e-4:                                # slide inside the plane only
                return False
        return True

    out = list(flat)
    for _ in range(6):
        out, changed = _resolve_once(out, offset, tol, frames, locked,
                                     lambda i, j, a: locked(i, j, a, True))
        if not changed:
            break
    res: dict[str, list[Shape]] = {lk.name: [] for lk in spec.links}
    for sh, ln in zip(out, link_of):
        res[ln].append(sh)
    return res


# ---------------------------------------------------------------------------
# Public API: whole machines
# ---------------------------------------------------------------------------

def _uv_phase(key: str) -> tuple[float, float]:
    u = zlib.crc32((key + "/u").encode()) / 4294967296.0
    v = zlib.crc32((key + "/v").encode()) / 4294967296.0
    return (u * _UV_JITTER_PERIOD, v * _UV_JITTER_PERIOD)


def machine_visual_meshes(spec: MachineSpec, bevel: float = 0.025, segments: int = 24,
                          tire_shoulder: float = TIRE_SHOULDER,
                          uv_jitter: bool = True, *,
                          bevel_scale: float = PANEL_BEVEL_SCALE,
                          max_bevel: float = MAX_PANEL_BEVEL,
                          panel_materials: tuple[str, ...] = PANEL_MATERIALS,
                          resolve_coplanar_faces: bool = True) -> dict[str, list[tuple[str, Mesh]]]:
    """Visual meshes of every link: ``link name -> [(material name, merged mesh), ...]``.

    All ``shape.visual`` shapes of a link are grouped by ``shape.material`` (in
    order of first use) and merged into one mesh per material, in the link frame.
    Every link has an entry (an empty list if it has no visual shape).

    Extras on top of :func:`shape_mesh` (each can be switched off):

    * tyres (:func:`tire_shapes`) get a tyre cross-section with shoulders of
      ``tire_shoulder * width`` (``tire_shoulder = 0`` keeps the plain cylinder);
    * shapes whose material is in ``panel_materials`` get a fillet that scales with
      the part, ``max(bevel, min(max_bevel, bevel_scale * middle_dim))``
      (``bevel_scale = 0`` gives every part exactly ``bevel``);
    * ``resolve_coplanar_faces`` pushes one of every pair of exactly coplanar,
      overlapping faces 1 mm outward (see :func:`resolve_coplanar_machine`; this
      includes faces of different links that no joint motion can separate), so
      nothing z-fights; typically a face moves by a millimetre or two, a deeply
      nested plate stack by a few millimetres more;
    * ``uv_jitter`` shifts every part's UVs by a deterministic per-part offset
      (a metre-valued phase, harmless for ``repeat`` textures) so neighbouring
      panels of the same material do not repeat the same grain.
    """
    tires = tire_shapes(spec) if tire_shoulder and tire_shoulder > 0.0 else set()
    panel = set(panel_materials)
    resolved = resolve_coplanar_machine(spec) if resolve_coplanar_faces else {}
    out: dict[str, list[tuple[str, Mesh]]] = {}
    for lk in spec.links:
        shapes = resolved[lk.name] if resolve_coplanar_faces else list(lk.shapes)
        groups: dict[str, list[Mesh]] = {}
        for idx, sh in enumerate(shapes):
            if not sh.visual:
                continue
            is_tire = (lk.name, idx) in tires
            scale = bevel_scale if sh.material in panel else 0.0
            off = _uv_phase(f"{spec.name}/{lk.name}/{idx}") if uv_jitter else (0.0, 0.0)
            groups.setdefault(sh.material, []).append(
                _build(sh, bevel, segments, off, scale, max_bevel,
                       tire_shoulder if is_tire else None))
        out[lk.name] = [(mat, merge_meshes(ms)) for mat, ms in groups.items()]
    return out
