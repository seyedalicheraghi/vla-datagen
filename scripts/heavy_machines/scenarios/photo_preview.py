"""Offline 'photo' preview of a scenario camera (numpy only; no Isaac Sim).

Much closer to the final render than ``preview.py``: the terrain heightfield is ray-marched per
pixel and shaded with the baked albedo + normal + roughness maps, machines and rocks are rasterised
with a z-buffer using the render-twin meshes and their PBR texture sets, the sun casts shadows
(heightfield self-shadowing + a shadow map for machines and boulders), the sky is a physically
plausible gradient, and the camera look (haze, tone curve, lens, grain) is applied on top.

    python scripts/heavy_machines/scenarios/photo_preview.py adt_approach_hex --times 0 5 10 \
        --scale 0.5 --out /tmp/photo.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from heavy_machines import kinematics as K  # noqa: E402
from heavy_machines.scenarios import camera_look  # noqa: E402
from heavy_machines.scenarios.pbr_textures import (machine_textures, rock_textures,  # noqa: E402
                                                   terrain_detail_cavity, terrain_detail_maps)
from heavy_machines.scenarios.preview import load  # noqa: E402
from heavy_machines.scenarios.rocks import contact_map, scatter  # noqa: E402
from heavy_machines.scenarios.scenario import Scenario, sun_direction  # noqa: E402
from heavy_machines.scenarios.terrain import bake_albedo  # noqa: E402
from heavy_machines.scenarios.visual_meshes import machine_visual_meshes  # noqa: E402

SUN_E = 3.2          # relative sun irradiance
SKY_E = 0.75         # relative sky irradiance on an up-facing surface
SKY_ZENITH = np.array([0.38, 0.55, 0.80])
SKY_HORIZON = np.array([0.80, 0.84, 0.88])


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray, wrap: bool) -> np.ndarray:
    """Sample img (H,W[,C]) at continuous pixel coords u (x) and v (y)."""
    h, w = img.shape[:2]
    if wrap:
        u, v = np.mod(u, w), np.mod(v, h)
    else:
        u, v = np.clip(u, 0, w - 1.001), np.clip(v, 0, h - 1.001)
    x0, y0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    fx, fy = (u - x0).astype(np.float32), (v - y0).astype(np.float32)
    x1, y1 = (x0 + 1) % w if wrap else x0 + 1, (y0 + 1) % h if wrap else y0 + 1
    if img.ndim == 3:
        fx, fy = fx[:, None], fy[:, None]
    a = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    b = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return a * (1 - fy) + b * fy


# ---------------------------------------------------------------------------
# Scene assets (built once per scenario)
# ---------------------------------------------------------------------------

@dataclass
class Grid:
    rect: tuple
    step: float
    h: np.ndarray

    def sample(self, x, y):
        x0, _, y0, _ = self.rect
        return _bilinear(self.h, (x - x0) / self.step, (y - y0) / self.step, wrap=False)

    def inside(self, x, y):
        x0, x1, y0, y1 = self.rect
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


class Assets:
    def __init__(self, sc: Scenario, ppm: float = 16.0):
        T = sc.terrain
        self.sc = sc
        lx0, lx1, ly0, ly1 = sc.local_rect
        xs, ys = np.arange(lx0, lx1 + 1e-9, 0.1), np.arange(ly0, ly1 + 1e-9, 0.1)
        X, Y = np.meshgrid(xs, ys)
        self.local = Grid(sc.local_rect, 0.1, T.height(X, Y).astype(np.float32))
        frect = (-1200.0, 1200.0, -1200.0, 1200.0)
        xs, ys = np.arange(frect[0], frect[1] + 1e-9, 4.0), np.arange(frect[2], frect[3] + 1e-9, 4.0)
        X, Y = np.meshgrid(xs, ys)
        self.far = Grid(frect, 4.0, T.height(X, Y).astype(np.float32))
        self.ppm = ppm
        rk = scatter(T, sc.local_rect, seed=6, keep_clear=sc.keep_clear, gravel_per_m2=0.0, focus=sc.near_rect,
                     boulders=160)
        alb = bake_albedo(T, sc.local_rect, ppm, seed=1).astype(np.float32)
        alb *= (terrain_detail_cavity(T, sc.local_rect, ppm, seed=1).astype(np.float32) / 255.0)[..., None] ** 0.8
        alb *= contact_map(rk, sc.local_rect, ppm)
        self.albedo = srgb_to_linear(np.clip(alb, 0, 255) / 255.0)
        nrm, rough = terrain_detail_maps(T, sc.local_rect, ppm, seed=1)
        self.nmap = nrm.astype(np.float32) / 127.5 - 1.0
        self.rough = rough.astype(np.float32) / 255.0
        self.far_albedo = srgb_to_linear(bake_albedo(T, frect, 0.25, seed=2, far=True) / 255.0)
        self.frect = frect

        # triangles of machines (per time) are built on demand; static rocks once
        self.material_sets: dict[tuple[str, str], object] = {}
        self.link_meshes = {a.name: machine_visual_meshes(a.spec) for a in sc.actors}
        protos = rk.prototypes()
        self.rock_tex = srgb_to_linear(rock_textures(11, 256).albedo / 255.0)
        tris, uvs, nrms = [], [], []
        for p, s, yaw, k in zip(rk.positions, rk.scales, rk.yaws, rk.proto):
            m = protos[int(k)]
            c, sn = math.cos(float(yaw)), math.sin(float(yaw))
            R = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]], dtype=np.float32)
            pts = (m.points * s) @ R.T + p
            nr = (m.normals / np.maximum(s, 1e-6)) @ R.T
            nr /= np.linalg.norm(nr, axis=1, keepdims=True)
            tris.append(pts[m.triangles])
            nrms.append(nr[m.triangles])
            uvs.append(m.uvs[m.triangles])
        self.rocks = (np.concatenate(tris), np.concatenate(nrms), np.concatenate(uvs)) if tris else None

    def height(self, x, y):
        inside = self.local.inside(x, y)
        h = np.empty_like(x, dtype=np.float32)
        if inside.any():
            h[inside] = self.local.sample(x[inside], y[inside])
        if (~inside).any():
            h[~inside] = self.far.sample(x[~inside], y[~inside])
        return h

    def textures(self, spec, mname):
        key = (spec.name, mname)
        if key not in self.material_sets:
            rgb = spec.materials.get(mname, (0.6, 0.6, 0.6, 1.0))[:3]
            ts = machine_textures(mname, tuple(rgb), seed=17 + len(self.material_sets), size=256)
            self.material_sets[key] = (srgb_to_linear(ts.albedo / 255.0), ts.roughness / 255.0,
                                       ts.normal.astype(np.float32) / 127.5 - 1.0, ts)
        return self.material_sets[key]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _march_terrain(A: Assets, O, D, t_max=3000.0):
    """Per-ray first hit with the heightfield. Returns t (inf on miss).

    Steps scale with the ray's height above the terrain (a conservative fraction, since faces are
    steep) and with distance, so rays reach the pit walls hundreds of metres away in a few hundred
    steps; the crossing is then refined by bisection."""
    n = len(D)
    t = np.full(n, 0.3, dtype=np.float32)
    hit_t = np.full(n, np.inf, dtype=np.float32)
    active = np.ones(n, dtype=bool)
    prev_t = t.copy()
    for _ in range(900):
        idx = np.nonzero(active)[0]
        if len(idx) == 0:
            break
        tt = t[idx]
        P = O[idx] + D[idx] * tt[:, None]
        gap = P[:, 2] - A.height(P[:, 0], P[:, 1])
        below = gap <= 0
        if below.any():
            hi_, lo_ = tt[below], prev_t[idx][below]
            sub = idx[below]
            for _ in range(8):                                    # bisection refine
                mid = 0.5 * (hi_ + lo_)
                Pm = O[sub] + D[sub] * mid[:, None]
                b = Pm[:, 2] <= A.height(Pm[:, 0], Pm[:, 1])
                hi_ = np.where(b, mid, hi_)
                lo_ = np.where(b, lo_, mid)
            hit_t[sub] = hi_
            active[sub] = False
        go = idx[~below]
        prev_t[go] = t[go]
        step = np.maximum(np.maximum(0.04, 0.004 * t[go]), 0.3 * gap[~below])
        t[go] = t[go] + np.minimum(step, np.maximum(0.04, 0.03 * t[go]))
        active[go] &= t[go] < t_max
    return hit_t


def _terrain_shadow(A: Assets, P, L):
    """1 = lit, 0 = in the terrain's own shadow (march toward the sun)."""
    lit = np.ones(len(P), dtype=np.float32)
    t = np.full(len(P), 0.08, dtype=np.float32)
    for _ in range(70):
        Q = P + L * t[:, None]
        h = A.height(Q[:, 0], Q[:, 1])
        lit = np.minimum(lit, np.clip((Q[:, 2] - h) * 6.0 + 1.0, 0.0, 1.0))
        t = t + np.maximum(0.06, 0.08 * t)
    return lit


def _raster(tris, W, H, f, cam_R_T, C, zbuf, payload, attrs, mid):
    """Rasterise triangles (M,3,3) with per-vertex attrs list of (M,3,k); writes into zbuf/payload."""
    pc = (tris - C) @ cam_R_T.T                                     # camera frame (x fwd, y left, z up)
    front = (pc[:, :, 0] > 0.1).all(axis=1)
    pc = pc[front]
    attrs = [a[front] for a in attrs]
    if len(pc) == 0:
        return
    u = W / 2 - f * pc[:, :, 1] / pc[:, :, 0]
    v = H / 2 - f * pc[:, :, 2] / pc[:, :, 0]
    inv_z = 1.0 / pc[:, :, 0]
    umin, umax = np.floor(u.min(1)).astype(int), np.ceil(u.max(1)).astype(int)
    vmin, vmax = np.floor(v.min(1)).astype(int), np.ceil(v.max(1)).astype(int)
    vis = (umax >= 0) & (umin < W) & (vmax >= 0) & (vmin < H)
    for i in np.nonzero(vis)[0]:
        x0, x1 = max(umin[i], 0), min(umax[i], W - 1)
        y0, y1 = max(vmin[i], 0), min(vmax[i], H - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ua, ub, uc), (va, vb, vc) = u[i], v[i]
        den = (vb - vc) * (ua - uc) + (uc - ub) * (va - vc)
        if abs(den) < 1e-9:
            continue
        w0 = ((vb - vc) * (xs - uc) + (uc - ub) * (ys - vc)) / den
        w1 = ((vc - va) * (xs - uc) + (ua - uc) * (ys - vc)) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        iz = w0 * inv_z[i, 0] + w1 * inv_z[i, 1] + w2 * inv_z[i, 2]
        depth = 1.0 / iz
        py, px = np.nonzero(inside)
        py, px = py + y0, px + x0
        d = depth[inside]
        closer = d < zbuf[py, px]
        if not closer.any():
            continue
        py, px, d = py[closer], px[closer], d[closer]
        zbuf[py, px] = d
        # perspective-correct barycentrics
        b0 = (w0[inside][closer] * inv_z[i, 0]) * d
        b1 = (w1[inside][closer] * inv_z[i, 1]) * d
        b2 = (w2[inside][closer] * inv_z[i, 2]) * d
        payload["mid"][py, px] = mid[i] if np.ndim(mid) else mid
        for name, a in zip(payload["names"], attrs):
            payload[name][py, px] = b0[:, None] * a[i, 0] + b1[:, None] * a[i, 1] + b2[:, None] * a[i, 2]


def render(A: Assets, cam_name: str, t: float, scale: float = 0.5, look: bool = True, seed: int = 0) -> np.ndarray:
    sc = A.sc
    cam = next(c for c in sc.cameras if c.name == cam_name)
    W, H = int(cam.width * scale), int(cam.height * scale)
    f = W * cam.focal_length / cam.horizontal_aperture
    ctf = sc.camera_tf(cam, t)
    R = np.array(ctf.R, dtype=np.float32)
    C = np.array(ctf.t, dtype=np.float32)
    L = -np.array(sun_direction(sc.lighting), dtype=np.float32)          # toward the sun

    jj, ii = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dc = np.stack([np.ones_like(jj), -(jj - W / 2) / f, -(ii - H / 2) / f], axis=-1).reshape(-1, 3)
    D = (dc @ R.T).astype(np.float32)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    O = np.broadcast_to(C, D.shape).astype(np.float32)

    # --- terrain -----------------------------------------------------------------------
    t_hit = _march_terrain(A, O, D)
    # depth along the optical axis for z-buffering against rasterised meshes
    z_terrain = (t_hit * (D @ R[:, 0])).reshape(H, W)

    # --- machines + rocks (z-buffer) ----------------------------------------------------
    zbuf = np.where(np.isfinite(z_terrain), z_terrain, np.inf).astype(np.float32)
    payload = {"names": ["pos", "nrm", "uv"], "mid": np.full((H, W), -1, dtype=np.int32),
               "pos": np.zeros((H, W, 3), np.float32), "nrm": np.zeros((H, W, 3), np.float32),
               "uv": np.zeros((H, W, 2), np.float32)}
    mats = []
    for actor in sc.actors:
        base, q = actor.root_tf(t)
        tfs = K.link_transforms(actor.spec, q)
        for link, items in A.link_meshes[actor.name].items():
            T = base @ tfs[link]
            Rl, tl = np.array(T.R, np.float32), np.array(T.t, np.float32)
            for mname, mesh in items:
                pts = mesh.points @ Rl.T + tl
                nr = mesh.normals @ Rl.T
                mats.append((actor.spec, mname))
                _raster(pts[mesh.triangles], W, H, f, R.T, C, zbuf, payload,
                        [pts[mesh.triangles], nr[mesh.triangles], mesh.uvs[mesh.triangles]], len(mats) - 1)
    if A.rocks is not None:
        tri, nr, uv = A.rocks
        mats.append(("rock", "rock"))
        _raster(tri, W, H, f, R.T, C, zbuf, payload, [tri, nr, uv], len(mats) - 1)

    mid = payload["mid"].reshape(-1)
    on_mesh = mid >= 0
    on_terrain = (~on_mesh) & np.isfinite(t_hit)
    sky = (~on_mesh) & ~np.isfinite(t_hit)

    P = np.zeros_like(D)
    N = np.zeros_like(D)
    albedo = np.zeros_like(D)
    rough = np.full(len(D), 0.9, np.float32)
    metal = np.zeros(len(D), np.float32)

    # terrain surface attributes
    if on_terrain.any():
        Pt = O[on_terrain] + D[on_terrain] * t_hit[on_terrain, None]
        P[on_terrain] = Pt
        e = 0.08
        hx = (A.height(Pt[:, 0] + e, Pt[:, 1]) - A.height(Pt[:, 0] - e, Pt[:, 1])) / (2 * e)
        hy = (A.height(Pt[:, 0], Pt[:, 1] + e) - A.height(Pt[:, 0], Pt[:, 1] - e)) / (2 * e)
        Ng = np.stack([-hx, -hy, np.ones_like(hx)], 1)
        Ng /= np.linalg.norm(Ng, axis=1, keepdims=True)
        inl = A.local.inside(Pt[:, 0], Pt[:, 1])
        x0, _, _, y1 = A.sc.local_rect
        u, v = (Pt[:, 0] - x0) * A.ppm, (y1 - Pt[:, 1]) * A.ppm
        alb = np.empty((len(Pt), 3), np.float32)
        alb[inl] = _bilinear(A.albedo, u[inl], v[inl], wrap=False)
        fx0, _, _, fy1 = A.frect
        alb[~inl] = _bilinear(A.far_albedo, (Pt[~inl, 0] - fx0) * 0.25, (fy1 - Pt[~inl, 1]) * 0.25, wrap=False)
        albedo[on_terrain] = alb
        Nn = Ng.copy()
        if inl.any():                                            # tangent-space normal map (T=+x, B=+y)
            nm = _bilinear(A.nmap, u[inl], v[inl], wrap=False)
            Tg = np.stack([np.ones(inl.sum()), np.zeros(inl.sum()), hx[inl]], 1)
            Tg /= np.linalg.norm(Tg, axis=1, keepdims=True)
            Bg = np.cross(Ng[inl], Tg)
            Nn[inl] = nm[:, :1] * Tg + nm[:, 1:2] * Bg + nm[:, 2:3] * Ng[inl]
            Nn[inl] /= np.linalg.norm(Nn[inl], axis=1, keepdims=True)
            rr = rough[on_terrain]
            rr[inl] = _bilinear(A.rough, u[inl], v[inl], wrap=False)
            rough[on_terrain] = rr
        N[on_terrain] = Nn

    # mesh surface attributes
    if on_mesh.any():
        pos = payload["pos"].reshape(-1, 3)[on_mesh]
        nrm = payload["nrm"].reshape(-1, 3)[on_mesh]
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
        uvs = payload["uv"].reshape(-1, 2)[on_mesh]
        ids = mid[on_mesh]
        alb = np.zeros((len(ids), 3), np.float32)
        rg = np.full(len(ids), 0.6, np.float32)
        mt = np.zeros(len(ids), np.float32)
        nn = nrm.copy()
        for k in np.unique(ids):
            sel = ids == k
            spec, mname = mats[k]
            if spec == "rock":
                th, tw = A.rock_tex.shape[:2]
                alb[sel] = _bilinear(A.rock_tex, uvs[sel, 0] * tw, uvs[sel, 1] * th, wrap=True)
                rg[sel] = 0.88
                continue
            ta, tr, tn, ts = A.textures(spec, mname)
            th, tw = ta.shape[:2]
            uu, vv = uvs[sel, 0] / ts.metres_per_tile * tw, uvs[sel, 1] / ts.metres_per_tile * th
            alb[sel] = _bilinear(ta, uu, vv, wrap=True)
            rg[sel] = _bilinear(tr.astype(np.float32), uu, vv, wrap=True)
            mt[sel] = ts.metallic
            # approximate tangent frame from the normal (no UV derivatives here)
            nm = _bilinear(tn, uu, vv, wrap=True)
            up = np.where(np.abs(nrm[sel, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
            Tg = np.cross(up, nrm[sel])
            Tg /= np.maximum(np.linalg.norm(Tg, axis=1, keepdims=True), 1e-6)
            Bg = np.cross(nrm[sel], Tg)
            n2 = nm[:, :1] * Tg + nm[:, 1:2] * Bg + nm[:, 2:3] * nrm[sel]
            nn[sel] = n2 / np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-6)
        P[on_mesh], N[on_mesh], albedo[on_mesh] = pos, nn, alb
        rough[on_mesh], metal[on_mesh] = rg, mt

    # --- lighting -------------------------------------------------------------------------
    lit_mask = ~sky
    Pl, Nl = P[lit_mask], N[lit_mask]
    ndl = np.clip(Nl @ L, 0.0, 1.0)
    shadow = _terrain_shadow(A, Pl + Nl * 0.03, L)
    shadow *= _mesh_shadow(A, sc, t, Pl + Nl * 0.03, L)
    V = -D[lit_mask]
    Hh = V + L
    Hh /= np.linalg.norm(Hh, axis=1, keepdims=True)
    a = np.clip(rough[lit_mask], 0.05, 1.0) ** 2
    ndh = np.clip(np.sum(Nl * Hh, 1), 0, 1)
    ndv = np.clip(np.sum(Nl * V, 1), 1e-3, 1)
    dterm = a * a / (math.pi * (ndh * ndh * (a * a - 1) + 1) ** 2)
    f0 = np.where(metal[lit_mask, None] > 0.5, albedo[lit_mask], 0.04)
    fres = f0 + (1 - f0) * (1 - np.clip(np.sum(V * Hh, 1), 0, 1))[:, None] ** 5
    k = a / 2
    g = (ndl / (ndl * (1 - k) + k)) * (ndv / (ndv * (1 - k) + k))
    spec_term = fres * (dterm * g / np.maximum(4 * ndl * ndv, 1e-3))[:, None]
    diff = albedo[lit_mask] * (1 - metal[lit_mask, None]) / math.pi
    sun_rgb = np.array(sc.lighting.sun_color, np.float32) * SUN_E * math.pi
    sky_amb = SKY_E * (0.55 + 0.45 * np.clip(Nl[:, 2], -1, 1))[:, None] * np.array([0.92, 0.96, 1.05])
    bounce = 0.18 * np.clip(-Nl[:, 2] * 0.5 + 0.5, 0, 1)[:, None] * np.array([0.9, 0.8, 0.65])
    col = (diff + spec_term) * sun_rgb * (ndl * shadow)[:, None] + albedo[lit_mask] * (sky_amb + bounce) \
        + fres * 0.15 * SKY_E                                   # sky reflection on glossy parts

    img = np.zeros_like(D)
    img[lit_mask] = col
    # sky: gradient + sun glow
    up = np.clip(D[sky, 2], 0, 1)[:, None]
    sky_c = SKY_HORIZON * (1 - up ** 0.5) + SKY_ZENITH * up ** 0.5
    sky_c = sky_c + 0.35 * np.clip(D[sky] @ L, 0, 1)[:, None] ** 64
    img[sky] = srgb_to_linear(sky_c) * 2.6
    img = img.reshape(H, W, 3)
    srgb = (linear_to_srgb(img / 2.6) * 255 + 0.5).astype(np.uint8)       # exposure

    depth = np.full(H * W, np.inf, np.float32)
    depth[lit_mask] = np.linalg.norm(P[lit_mask] - C, axis=1)
    depth = depth.reshape(H, W)
    if look:
        row, tilt = camera_look.horizon_from_camera(H, f, pitch_down_deg=math.degrees(cam.rpy[1]))
        srgb = camera_look.apply(srgb, depth, camera_look.LookParams(horizon_row=row, horizon_tilt=tilt), seed=seed)
    return srgb


def _mesh_shadow(A: Assets, sc: Scenario, t: float, P: np.ndarray, L: np.ndarray, res: float = 0.06):
    """Orthographic shadow map of machines + boulders along the sun direction."""
    # light frame: z along L
    zl = L / np.linalg.norm(L)
    xl = np.cross([0.0, 0.0, 1.0], zl)
    xl = xl / np.linalg.norm(xl) if np.linalg.norm(xl) > 1e-6 else np.array([1.0, 0.0, 0.0])
    yl = np.cross(zl, xl)
    M = np.stack([xl, yl, zl]).astype(np.float32)               # world -> light
    tris = []
    for actor in sc.actors:
        base, q = actor.root_tf(t)
        tfs = K.link_transforms(actor.spec, q)
        for link, items in A.link_meshes[actor.name].items():
            T = base @ tfs[link]
            Rl, tl = np.array(T.R, np.float32), np.array(T.t, np.float32)
            for _, mesh in items:
                tris.append((mesh.points @ Rl.T + tl)[mesh.triangles])
    if A.rocks is not None:
        tris.append(A.rocks[0])
    if not tris:
        return np.ones(len(P), np.float32)
    tri = np.concatenate(tris) @ M.T                             # (M,3,3) light coords
    lo = tri[:, :, :2].reshape(-1, 2).min(0) - 1.0
    hi = tri[:, :, :2].reshape(-1, 2).max(0) + 1.0
    nx, ny = int((hi[0] - lo[0]) / res) + 1, int((hi[1] - lo[1]) / res) + 1
    if nx * ny > 6e6:
        res = math.sqrt((hi[0] - lo[0]) * (hi[1] - lo[1]) / 6e6)
        nx, ny = int((hi[0] - lo[0]) / res) + 1, int((hi[1] - lo[1]) / res) + 1
    top = np.full((ny, nx), -np.inf, np.float32)                 # highest occluder toward the sun
    gx, gy = (tri[:, :, 0] - lo[0]) / res, (tri[:, :, 1] - lo[1]) / res
    for i in range(len(tri)):
        x0, x1 = int(gx[i].min()), int(math.ceil(gx[i].max()))
        y0, y1 = int(gy[i].min()), int(math.ceil(gy[i].max()))
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        (ua, ub, uc), (va, vb, vc) = gx[i], gy[i]
        den = (vb - vc) * (ua - uc) + (uc - ub) * (va - vc)
        if abs(den) < 1e-12:
            continue
        w0 = ((vb - vc) * (xs - uc) + (uc - ub) * (ys - vc)) / den
        w1 = ((vc - va) * (xs - uc) + (ua - uc) * (ys - vc)) / den
        w2 = 1 - w0 - w1
        ins = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not ins.any():
            continue
        z = (w0 * tri[i, 0, 2] + w1 * tri[i, 1, 2] + w2 * tri[i, 2, 2])[ins]
        py, px = np.nonzero(ins)
        py, px = np.clip(py + y0, 0, ny - 1), np.clip(px + x0, 0, nx - 1)
        np.maximum.at(top, (py, px), z)
    pl = P @ M.T
    ix = np.clip(((pl[:, 0] - lo[0]) / res).astype(int), 0, nx - 1)
    iy = np.clip(((pl[:, 1] - lo[1]) / res).astype(int), 0, ny - 1)
    inside = (pl[:, 0] >= lo[0]) & (pl[:, 0] <= hi[0]) & (pl[:, 1] >= lo[1]) & (pl[:, 1] <= hi[1])
    occluded = inside & (top[iy, ix] > pl[:, 2] + 0.05)
    return np.where(occluded, 0.0, 1.0).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario")
    p.add_argument("--camera", default="front")
    p.add_argument("--times", type=float, nargs="*", default=[0.0, 5.0, 10.0])
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--no_look", action="store_true")
    p.add_argument("--out", required=True, help="PNG sheet; individual frames are written next to it")
    a = p.parse_args()
    from PIL import Image

    sc = load(a.scenario)
    A = Assets(sc)
    frames = []
    for i, t in enumerate(a.times):
        img = render(A, a.camera, t, a.scale, look=not a.no_look, seed=i)
        Image.fromarray(img).save(a.out.replace(".png", f"_{a.camera}_t{t:04.1f}.png"))
        frames.append(img)
    sheet = np.concatenate(frames, axis=0) if len(frames) > 1 else frames[0]
    Image.fromarray(sheet).save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
