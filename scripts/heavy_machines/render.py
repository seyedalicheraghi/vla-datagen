"""Preview renders of a MachineSpec (needs matplotlib; dev tool only).

``render_poses(spec, path)``   side / top / front / 3-D views for a few poses.
``render_tool_sweeps(spec, path)`` one side view per tool with its range overlaid.
"""

from __future__ import annotations

import math

from . import kinematics as K
from .spec import Box, Cylinder, MachineSpec, Sphere


def _hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _faces_3d(spec: MachineSpec, tfs):
    """Triangles/quads for a 3-D render: list of (verts, rgba)."""
    faces = []
    for lk in spec.links:
        for s in lk.shapes:
            if not s.visual:
                continue
            tf = K.shape_tf(tfs[lk.name], s)
            rgba = spec.materials.get(s.material, (0.5, 0.5, 0.5, 1.0))
            g = s.geom
            if isinstance(g, Box):
                hx, hy, hz = (v / 2 for v in g.size)
                c = [tf.apply((sx * hx, sy * hy, sz * hz))
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
                for idx in ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6),
                            (0, 2, 6, 4), (1, 3, 7, 5)):
                    faces.append(([c[i] for i in idx], rgba))
            elif isinstance(g, Cylinder):
                n = 14
                ring = [(g.radius * math.cos(2 * math.pi * i / n), g.radius * math.sin(2 * math.pi * i / n))
                        for i in range(n)]
                top = [tf.apply((x, y, g.length / 2)) for x, y in ring]
                bot = [tf.apply((x, y, -g.length / 2)) for x, y in ring]
                faces.append((top, rgba))
                faces.append((bot, rgba))
                for i in range(n):
                    k = (i + 1) % n
                    faces.append(([top[i], top[k], bot[k], bot[i]], rgba))
            elif isinstance(g, Sphere):
                pts = K.shape_points(tfs[lk.name], s, 12)
                faces.append((pts[:14], rgba))
    return faces


def _draw_ortho(ax, spec: MachineSpec, tfs, view: str, h: float, alpha: float = 1.0,
                edge: str = "k"):
    from matplotlib.patches import Polygon

    # (u, v, depth axis, sign of the direction TOWARD the viewer).
    #   side : seen from -y (x right, z up)     → nearest = most negative y
    #   top  : seen from +z (x right, y up)     → nearest = highest z
    #   front: seen from +x (y right, z up)     → nearest = largest x
    # Painter's algorithm: sort by distance toward the viewer, far first; ties
    # (e.g. a tyre and its hub) are broken by the face nearest the viewer.
    proj = {"side": (0, 2, 1, -1.0), "top": (0, 1, 2, 1.0), "front": (1, 2, 0, 1.0)}[view]
    u, v, w, sgn = proj
    items = []
    for lk in spec.links:
        for s in lk.shapes:
            if not s.visual:
                continue
            pts = K.shape_points(tfs[lk.name], s, 20)
            pts = [(p[0], p[1], p[2] + h) for p in pts]
            hull = _hull([(p[u], p[v]) for p in pts])
            toward = [sgn * p[w] for p in pts]
            depth = (round(sum(toward) / len(toward), 3), max(toward))
            rgba = spec.materials.get(s.material, (0.5, 0.5, 0.5, 1.0))
            items.append((depth, hull, rgba))
    for _, hull, rgba in sorted(items, key=lambda t: t[0]):
        if len(hull) >= 3:
            ax.add_patch(Polygon(hull, closed=True, facecolor=rgba[:3], edgecolor=edge,
                                 linewidth=0.4, alpha=alpha))


def _frame_axes(ax, title, view, lo, hi, h):
    names = {"side": ("x (m)", "z (m)"), "top": ("x (m)", "y (m)"), "front": ("y (m)", "z (m)")}[view]
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(names[0], fontsize=7)
    ax.set_ylabel(names[1], fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3)
    u, v = {"side": (0, 2), "top": (0, 1), "front": (1, 2)}[view]
    lo2 = (lo[0], lo[1], lo[2] + h)
    hi2 = (hi[0], hi[1], hi[2] + h)
    pad = 0.5
    ax.set_xlim(lo2[u] - pad, hi2[u] + pad)
    ax.set_ylim(min(lo2[v], 0.0) - pad, hi2[v] + pad)
    if view != "top":
        ax.axhline(0.0, color="saddlebrown", linewidth=1.2)


def default_poses(spec: MachineSpec) -> list[tuple[str, dict[str, float]]]:
    poses = [("init", {})]
    lo_q, hi_q = {}, {}
    for t in spec.tools:
        j = spec.joint(t.joint)
        if j.type == "continuous":
            lo_q[j.name], hi_q[j.name] = -math.pi / 2, math.pi / 2
        else:
            lo_q[j.name], hi_q[j.name] = j.lower, j.upper
    if spec.tools:
        poses.append(("all tools at lower limit", lo_q))
        poses.append(("all tools at upper limit", hi_q))
    d = spec.drive
    if d.type == "ackermann":
        steers = spec.joints_with_role("steer")
        jf = K.joint_frames(spec)
        delta = d.max_steer if d.steer_axle == "front" else -d.max_steer
        angles = K.ackermann_wheel_angles(delta, d.wheelbase, [jf[j.name].t[1] for j in steers],
                                          d.steer_axle)
        poses.append(("full left steer", {j.name: a for j, a in zip(steers, angles)}))
    elif d.type == "articulated":
        aj = spec.joint(d.articulation_joint)
        poses.append(("full left articulation", {aj.name: aj.lower}))
    return poses


def render_poses(spec: MachineSpec, path: str, poses=None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    poses = poses or default_poses(spec)
    h = K.root_height(spec)
    # Common limits over all poses so panels are comparable.
    los, his = [], []
    for _, q in poses:
        lo, hi = K.bounding_box(spec, q)
        los.append(lo)
        his.append(hi)
    lo = tuple(min(p[i] for p in los) for i in range(3))
    hi = tuple(max(p[i] for p in his) for i in range(3))

    fig = plt.figure(figsize=(18, 4.2 * len(poses)))
    for row, (title, q) in enumerate(poses):
        tfs = K.link_transforms(spec, q)
        for col, view in enumerate(("side", "top", "front")):
            ax = fig.add_subplot(len(poses), 4, row * 4 + col + 1)
            _draw_ortho(ax, spec, tfs, view, h)
            _frame_axes(ax, f"{title} — {view}", view, lo, hi, h)
        ax3 = fig.add_subplot(len(poses), 4, row * 4 + 4, projection="3d")
        faces = _faces_3d(spec, tfs)
        verts = [[(p[0], p[1], p[2] + h) for p in f] for f, _ in faces]
        coll = Poly3DCollection(verts, facecolors=[c[:3] for _, c in faces],
                                edgecolors=(0, 0, 0, 0.25), linewidths=0.2)
        ax3.add_collection3d(coll)
        span = max(hi[i] - lo[i] for i in range(3)) / 2 + 0.3
        cx, cy, cz = ((lo[i] + hi[i]) / 2 for i in range(3))
        ax3.set_xlim(cx - span, cx + span)
        ax3.set_ylim(cy - span, cy + span)
        ax3.set_zlim(0.0, 2 * span)
        ax3.view_init(elev=22, azim=-55)
        ax3.set_title(f"{title} — 3D", fontsize=8)
        ax3.tick_params(labelsize=5)
    d = K.bounding_box(spec)
    dims = tuple(round(d[1][i] - d[0][i], 2) for i in range(3))
    fig.suptitle(f"{spec.display_name} ({spec.abbrev}) — {spec.reference}\n"
                 f"model L×W×H {dims} m vs envelope {spec.envelope} m · "
                 f"mass {spec.total_mass:.0f} kg · root height {h:.2f} m · "
                 f"action {spec.action_labels}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=70)
    plt.close(fig)
    return path


def render_tool_sweeps(spec: MachineSpec, path: str) -> str | None:
    """One row per tool: side + top view with lower / init / upper overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not spec.tools:
        return None
    h = K.root_height(spec)
    fig = plt.figure(figsize=(12, 3.6 * len(spec.tools)))
    for row, t in enumerate(spec.tools):
        j = spec.joint(t.joint)
        if j.type == "continuous":
            qs = [(-math.pi / 2, 0.25, "r"), (j.init, 1.0, "k"), (math.pi / 2, 0.45, "b")]
        else:
            qs = [(j.lower, 0.25, "r"), (j.init, 1.0, "k"), (j.upper, 0.45, "b")]
        los, his = [], []
        for q, _, _ in qs:
            lo, hi = K.bounding_box(spec, {j.name: q})
            los.append(lo)
            his.append(hi)
        lo = tuple(min(p[i] for p in los) for i in range(3))
        hi = tuple(max(p[i] for p in his) for i in range(3))
        for col, view in enumerate(("side", "top")):
            ax = fig.add_subplot(len(spec.tools), 2, row * 2 + col + 1)
            for q, alpha, edge in qs:
                _draw_ortho(ax, spec, K.link_transforms(spec, {j.name: q}), view, h, alpha, edge)
            _frame_axes(ax, f"tool '{t.label}' ({j.name}): red=lower {qs[0][0]:.2f}, "
                            f"black=init {j.init:.2f}, blue=upper {qs[2][0]:.2f}  "
                            f"(+1 = {t.positive or '?'})", view, lo, hi, h)
    fig.suptitle(f"{spec.display_name} — tool sweeps", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=70)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Approximate camera images (pinhole, painter's algorithm, no lighting)
# ---------------------------------------------------------------------------

_CAM_PX = 224
_CAM_FOCAL = _CAM_PX * 10.0 / 20.955          # matches machine_env (10 mm lens, 20.955 mm aperture)


def _camera_faces(spec: MachineSpec, tfs, cam_tf, h: float):
    """Projected polygons (depth, pts2d, rgb) for one camera, ground tiles included."""
    Rt = K.transpose(cam_tf.R)

    def to_cam(p):
        return K.mat_vec(Rt, K.vsub(p, cam_tf.t))

    out = []

    def add(poly3, rgb):
        cps = [to_cam(p) for p in poly3]
        if any(c[0] < 0.12 for c in cps):            # crude near-plane rejection
            return
        pts = [(_CAM_PX / 2 - _CAM_FOCAL * c[1] / c[0], _CAM_PX / 2 - _CAM_FOCAL * c[2] / c[0]) for c in cps]
        if all(x < -200 or x > _CAM_PX + 200 for x, _ in pts) or all(y < -200 or y > _CAM_PX + 200 for _, y in pts):
            return
        out.append((sum(c[0] for c in cps) / len(cps), pts, rgb))

    # ground: tiles around the machine (root frame, z = -h)
    tile = 3.0
    for i in range(-12, 12):
        for k in range(-12, 12):
            x0, y0 = i * tile, k * tile
            shade = 0.40 if (i + k) % 2 else 0.36
            add([(x0, y0, -h), (x0 + tile, y0, -h), (x0 + tile, y0 + tile, -h), (x0, y0 + tile, -h)],
                (shade + 0.06, shade, shade - 0.08))
    for verts, rgba in _faces_3d(spec, tfs):
        add(verts, rgba[:3])
    return out


def render_camera_views(spec: MachineSpec, path: str, poses=None) -> str:
    """Rows = poses, columns = the front / left / right cameras (approximate)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    poses = poses or default_poses(spec)[:3]
    h = K.root_height(spec)
    cams = sorted(spec.cameras, key=lambda c: ("front", "left", "right").index(c.name))
    fig, axes = plt.subplots(len(poses), len(cams), figsize=(3.2 * len(cams), 3.3 * len(poses)),
                             squeeze=False)
    for row, (title, q) in enumerate(poses):
        tfs = K.link_transforms(spec, q)
        for col, cam in enumerate(cams):
            ax = axes[row][col]
            ax.set_facecolor((0.70, 0.80, 0.92))
            cam_tf = tfs[cam.link] @ K.Tf.from_xyz_rpy(cam.xyz, cam.rpy)
            for _, pts, rgb in sorted(_camera_faces(spec, tfs, cam_tf, h), key=lambda f: -f[0]):
                ax.add_patch(Polygon(pts, closed=True, facecolor=rgb, edgecolor=(0, 0, 0, 0.15),
                                     linewidth=0.2))
            ax.set_xlim(0, _CAM_PX)
            ax.set_ylim(_CAM_PX, 0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{cam.name} — {title}", fontsize=7)
    fig.suptitle(f"{spec.display_name}: approximate camera images (224 px, no lighting/occlusion culling)",
                 fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path
