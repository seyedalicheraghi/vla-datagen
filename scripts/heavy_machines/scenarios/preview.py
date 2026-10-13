"""Offline preview of a scenario camera (numpy + matplotlib; no Isaac Sim).

Flat-shaded painter's-algorithm render with the camera's real intrinsics,
terrain coloured from the baked textures and Lambert sun shading — good for
checking layout, framing and timing, not for final looks.

    python scripts/heavy_machines/scenarios/preview.py adt_approach_hex --camera front \
        --times 0 2 4 6 8 10 --out /tmp/preview.png
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from heavy_machines import kinematics as K  # noqa: E402
from heavy_machines.render import _faces_3d  # noqa: E402
from heavy_machines.scenarios.scenario import Scenario, sun_direction  # noqa: E402
from heavy_machines.scenarios.terrain import bake_albedo, grid_mesh  # noqa: E402

NEAR = 0.1


def load(name: str) -> Scenario:
    return importlib.import_module(f"heavy_machines.scenarios.{name}").build()


def _terrain_polys(sc: Scenario):
    """(N,3,3) triangles and (N,3) colours for a coarse local + far terrain."""
    out_tris, out_cols = [], []
    for rect, step, ppm, far in ((sc.local_rect, 1.0, 2.0, False), ((-600.0, 600.0, -600.0, 600.0), 15.0, 0.05, True)):
        mesh = grid_mesh(sc.terrain, rect, step,
                         sink_inside=None if not far else sc.local_rect, sink=0.4)
        tex = bake_albedo(sc.terrain, rect, ppm, far=far).astype(np.float64) / 255.0
        tris = mesh.points[mesh.triangles].astype(np.float64)              # (M,3,3)
        uv = mesh.uvs[mesh.triangles].mean(axis=1)
        if far:   # no z-buffer here: drop far triangles that touch the detailed area
            lx0, lx1, ly0, ly1 = sc.local_rect
            bx0, bx1 = tris[:, :, 0].min(1), tris[:, :, 0].max(1)
            by0, by1 = tris[:, :, 1].min(1), tris[:, :, 1].max(1)
            keep = (bx1 <= lx0) | (bx0 >= lx1) | (by1 <= ly0) | (by0 >= ly1)
            tris, uv = tris[keep], uv[keep]
        h, w = tex.shape[:2]
        px = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
        py = np.clip(((1.0 - uv[:, 1]) * h).astype(int), 0, h - 1)
        out_tris.append(tris)
        out_cols.append(tex[py, px])
    return np.concatenate(out_tris), np.concatenate(out_cols)


def _machine_polys(sc: Scenario, t: float):
    polys, cols = [], []
    for actor in sc.actors:
        base, q = actor.root_tf(t)
        tfs = {k: base @ v for k, v in K.link_transforms(actor.spec, q).items()}
        for verts, rgba in _faces_3d(actor.spec, tfs):
            polys.append(np.asarray(verts, dtype=np.float64))
            cols.append(np.asarray(rgba[:3], dtype=np.float64))
    return polys, cols


def _clip_near(poly_cam: np.ndarray) -> np.ndarray | None:
    """Sutherland–Hodgman clip of a camera-frame polygon against x >= NEAR."""
    out = []
    n = len(poly_cam)
    for i in range(n):
        a, b = poly_cam[i], poly_cam[(i + 1) % n]
        ina, inb = a[0] >= NEAR, b[0] >= NEAR
        if ina:
            out.append(a)
        if ina != inb:
            s = (NEAR - a[0]) / (b[0] - a[0])
            out.append(a + s * (b - a))
    return np.array(out) if len(out) >= 3 else None


def render_frame(sc: Scenario, cam_name: str, t: float, ax, terrain_cache=None):
    from matplotlib.collections import PolyCollection

    cam = next(c for c in sc.cameras if c.name == cam_name)
    W, H = cam.width, cam.height
    f = W * cam.focal_length / cam.horizontal_aperture
    ctf = sc.camera_tf(cam, t)
    Rt = np.array(K.transpose(ctf.R))
    C = np.array(ctf.t)
    sun = -np.array(sun_direction(sc.lighting))

    tris, tcols = terrain_cache if terrain_cache is not None else _terrain_polys(sc)
    mpolys, mcols = _machine_polys(sc, t)

    items = []

    def add(poly_w: np.ndarray, col: np.ndarray):
        pc = (poly_w - C) @ Rt.T                       # world -> camera frame (x fwd, y left, z up)
        if (pc[:, 0] < NEAR).all():
            return
        if (pc[:, 0] >= NEAR).all():
            clipped = pc
        else:
            clipped = _clip_near(pc)
            if clipped is None:
                return
        u = W / 2 - f * clipped[:, 1] / clipped[:, 0]
        v = H / 2 - f * clipped[:, 2] / clipped[:, 0]
        if u.max() < 0 or u.min() > W or v.max() < 0 or v.min() > H:
            return
        nrm = np.cross(poly_w[1] - poly_w[0], poly_w[2] - poly_w[0])
        nn = np.linalg.norm(nrm)
        lam = abs(float(nrm @ sun) / nn) if nn > 1e-9 else 1.0
        shade = 0.42 + 0.58 * lam
        items.append((float(clipped[:, 0].mean()), np.stack([u, v], 1), np.clip(col * shade, 0, 1)))

    # cull terrain triangles behind the camera quickly
    cen = tris.mean(axis=1)
    fwd = (cen - C) @ Rt.T
    keep = fwd[:, 0] > -5.0
    for tri, col in zip(tris[keep], tcols[keep]):
        add(tri, col)
    for poly, col in zip(mpolys, mcols):
        add(poly, col)

    items.sort(key=lambda it: -it[0])
    ax.set_facecolor((0.80, 0.85, 0.91))
    ax.add_collection(PolyCollection([it[1] for it in items], facecolors=[it[2] for it in items],
                                     edgecolors="none", antialiased=False))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def render_sheet(sc: Scenario, cam_name: str, times: list[float], path: str, cols: int = 3) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cache = _terrain_polys(sc)
    rows = math.ceil(len(times) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.25 * rows), squeeze=False)
    for i, t in enumerate(times):
        ax = axes[i // cols][i % cols]
        render_frame(sc, cam_name, t, ax, cache)
        x = sc.actor(sc.cameras[0].actor).state(t).x if sc.cameras[0].actor else 0.0
        ax.set_title(f"{cam_name}  t = {t:.1f} s  (truck x = {x:+.1f} m)", fontsize=8)
    for j in range(len(times), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"{sc.name}: offline preview ({cam_name} camera, flat shading, no textures detail)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=85)
    plt.close(fig)
    return path


def render_top(sc: Scenario, path: str) -> str:
    """Bird's-eye layout: height map, truck path at 1 s intervals, excavator, camera frustum."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0, x1, y0, y1 = sc.local_rect
    xs, ys = np.linspace(x0, x1, 360), np.linspace(y0, y1, 260)
    X, Y = np.meshgrid(xs, ys)
    Z = sc.terrain.height(X, Y)
    tex = bake_albedo(sc.terrain, sc.local_rect, 4.0)
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(tex, extent=(x0, x1, y0, y1), origin="upper")
    cs = ax.contour(X, Y, Z, levels=[0.3, 0.8, 1.2, 1.8, 2.4], colors="k", linewidths=0.4)
    ax.clabel(cs, fontsize=6)
    for actor in sc.actors:
        for t in np.arange(0.0, sc.duration + 1e-6, 1.0 if not actor.static else sc.duration):
            base, q = actor.root_tf(float(t))
            tfs = {k: base @ v for k, v in K.link_transforms(actor.spec, q).items()}
            for verts, rgba in _faces_3d(actor.spec, tfs):
                v = np.array(verts)
                ax.fill(v[:, 0], v[:, 1], color=rgba[:3], alpha=0.15 if not actor.static else 0.6, lw=0)
    for cam in sc.cameras:
        for t in (0.0, sc.duration):
            ctf = sc.camera_tf(cam, t)
            hf = math.atan(cam.horizontal_aperture / 2 / cam.focal_length)
            for s in (-1, 1):
                d = K.mat_vec(ctf.R, (math.cos(hf), s * math.sin(hf), 0.0))
                ax.plot([ctf.t[0], ctf.t[0] + 40 * d[0]], [ctf.t[1], ctf.t[1] + 40 * d[1]],
                        "--", lw=0.6, color="b" if cam.name == "front" else "m")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_title(f"{sc.name}: layout (truck every 1 s; dashed = camera FOV at start/end)")
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scenario")
    p.add_argument("--camera", default="front")
    p.add_argument("--times", type=float, nargs="*")
    p.add_argument("--out", required=True, help="output PNG (a _top.png layout is written next to it)")
    a = p.parse_args()
    sc = load(a.scenario)
    times = a.times or [round(sc.duration * i / 5, 2) for i in range(6)]
    print(render_sheet(sc, a.camera, times, a.out))
    print(render_top(sc, a.out.replace(".png", "_top.png")))


if __name__ == "__main__":
    main()
