"""Pure-Python rigid-body math and forward kinematics for MachineSpec.

No numpy: this module is imported by the Isaac environment (per-step camera
poses) and by the offline checks, so it must run anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .spec import Box, Cylinder, Joint, MachineSpec, Shape, Sphere, ROLLING_ROLES

Mat3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
Vec3 = tuple[float, float, float]

IDENTITY: Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 3x3 / vector helpers
# ---------------------------------------------------------------------------

def mat_mul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))  # type: ignore[return-value]


def mat_vec(a: Mat3, v: Vec3) -> Vec3:
    return (a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
            a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
            a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2])


def transpose(a: Mat3) -> Mat3:
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def vadd(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vscale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def vnorm(a: Vec3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def vunit(a: Vec3) -> Vec3:
    n = vnorm(a)
    if n < 1e-12:
        raise ValueError("zero-length vector")
    return (a[0] / n, a[1] / n, a[2] / n)


def rot_x(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))


def rot_y(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))


def rot_z(a: float) -> Mat3:
    c, s = math.cos(a), math.sin(a)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def rpy_to_mat(rpy: Vec3) -> Mat3:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) · Ry(pitch) · Rx(roll)."""
    return mat_mul(rot_z(rpy[2]), mat_mul(rot_y(rpy[1]), rot_x(rpy[0])))


def axis_angle_to_mat(axis: Vec3, angle: float) -> Mat3:
    x, y, z = vunit(axis)
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return ((t * x * x + c, t * x * y - s * z, t * x * z + s * y),
            (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
            (t * x * z - s * y, t * y * z + s * x, t * z * z + c))


def mat_to_quat_wxyz(m: Mat3) -> tuple[float, float, float, float]:
    """Rotation matrix → unit quaternion (w, x, y, z), w ≥ 0 (Isaac Lab order)."""
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w, x = 0.25 * s, (m[2][1] - m[1][2]) / s
        y, z = (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w, x = (m[2][1] - m[1][2]) / s, 0.25 * s
        y, z = (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w, x = (m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s
        y, z = 0.25 * s, (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w, x = (m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s
        y, z = (m[1][2] + m[2][1]) / s, 0.25 * s
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / n, x / n, y / n, z / n)


@dataclass(frozen=True)
class Tf:
    """Rigid transform: p_parent = R · p_child + t."""
    R: Mat3 = IDENTITY
    t: Vec3 = (0.0, 0.0, 0.0)

    def __matmul__(self, other: "Tf") -> "Tf":
        return Tf(mat_mul(self.R, other.R), vadd(mat_vec(self.R, other.t), self.t))

    def apply(self, p: Vec3) -> Vec3:
        return vadd(mat_vec(self.R, p), self.t)

    @staticmethod
    def from_xyz_rpy(xyz: Vec3, rpy: Vec3) -> "Tf":
        return Tf(rpy_to_mat(rpy), tuple(xyz))  # type: ignore[arg-type]

    @staticmethod
    def planar(x: float, y: float, z: float, yaw: float) -> "Tf":
        return Tf(rot_z(yaw), (x, y, z))


def joint_motion(jt: Joint, q: float) -> Tf:
    """Transform contributed by the joint variable (applied after the joint origin)."""
    if jt.type in ("revolute", "continuous"):
        return Tf(axis_angle_to_mat(jt.axis, q), (0.0, 0.0, 0.0))
    if jt.type == "prismatic":
        return Tf(IDENTITY, vscale(vunit(jt.axis), q))
    return Tf()


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

def init_q(spec: MachineSpec) -> dict[str, float]:
    return {j.name: j.init for j in spec.movable_joints}


def resolve_followers(spec: MachineSpec, q: dict[str, float]) -> dict[str, float]:
    """Return a copy of q with follower joints set from their sources."""
    out = dict(q)
    for jt in spec.joints:
        if jt.role == "follower" and jt.follows:
            src, k, c = jt.follows
            val = k * out.get(src, spec.joint(src).init) + c
            out[jt.name] = val if jt.type == "continuous" else min(max(val, jt.lower), jt.upper)
    return out


def link_transforms(spec: MachineSpec, q: dict[str, float] | None = None) -> dict[str, Tf]:
    """Pose of every link frame in the ROOT frame for joint positions ``q``.

    Missing entries in ``q`` use each joint's ``init``; followers are resolved.
    """
    qq = init_q(spec)
    if q:
        qq.update(q)
    qq = resolve_followers(spec, qq)
    children: dict[str, list[Joint]] = {}
    for jt in spec.joints:
        children.setdefault(jt.parent, []).append(jt)
    tfs: dict[str, Tf] = {spec.root: Tf()}
    stack = [spec.root]
    while stack:
        parent = stack.pop()
        for jt in children.get(parent, []):
            tf = tfs[parent] @ Tf.from_xyz_rpy(jt.xyz, jt.rpy) @ joint_motion(jt, qq.get(jt.name, 0.0))
            tfs[jt.child] = tf
            stack.append(jt.child)
    return tfs


def joint_frames(spec: MachineSpec, q: dict[str, float] | None = None) -> dict[str, Tf]:
    """Pose of each joint origin (before its motion) in the root frame."""
    tfs = link_transforms(spec, q)
    return {jt.name: tfs[jt.parent] @ Tf.from_xyz_rpy(jt.xyz, jt.rpy) for jt in spec.joints
            if jt.parent in tfs}


def subtree_links(spec: MachineSpec, link: str) -> list[str]:
    out, stack = [], [link]
    while stack:
        cur = stack.pop()
        out.append(cur)
        stack.extend(j.child for j in spec.joints if j.parent == cur)
    return out


# ---------------------------------------------------------------------------
# Shape sampling (bounding boxes, ground contact, rendering)
# ---------------------------------------------------------------------------

def shape_tf(link_tf: Tf, shape: Shape) -> Tf:
    return link_tf @ Tf.from_xyz_rpy(shape.xyz, shape.rpy)


def shape_local_points(shape: Shape, n: int = 16) -> list[Vec3]:
    """Surface sample points in the shape's own frame (exact hull for boxes)."""
    g = shape.geom
    if isinstance(g, Box):
        hx, hy, hz = (s / 2 for s in g.size)
        return [(sx * hx, sy * hy, sz * hz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    if isinstance(g, Cylinder):
        pts = []
        for i in range(n):
            a = 2 * math.pi * i / n
            for z in (-g.length / 2, g.length / 2):
                pts.append((g.radius * math.cos(a), g.radius * math.sin(a), z))
        return pts
    if isinstance(g, Sphere):
        pts = [(0.0, 0.0, g.radius), (0.0, 0.0, -g.radius)]
        for i in range(n):
            a = 2 * math.pi * i / n
            for el in (-math.pi / 4, 0.0, math.pi / 4):
                pts.append((g.radius * math.cos(el) * math.cos(a),
                            g.radius * math.cos(el) * math.sin(a), g.radius * math.sin(el)))
        return pts
    raise TypeError(f"unknown geometry {g!r}")


def shape_points(link_tf: Tf, shape: Shape, n: int = 16) -> list[Vec3]:
    tf = shape_tf(link_tf, shape)
    return [tf.apply(p) for p in shape_local_points(shape, n)]


def shape_min_z(link_tf: Tf, shape: Shape) -> float:
    """Exact lowest z of a shape (cylinders/spheres analytically)."""
    tf = shape_tf(link_tf, shape)
    g = shape.geom
    if isinstance(g, Sphere):
        return tf.t[2] - g.radius
    if isinstance(g, Cylinder):
        ax = (tf.R[0][2], tf.R[1][2], tf.R[2][2])      # cylinder axis in root frame
        radial = math.sqrt(max(0.0, 1.0 - ax[2] * ax[2]))
        return tf.t[2] - abs(ax[2]) * g.length / 2 - radial * g.radius
    return min(p[2] for p in shape_points(link_tf, shape))


def ground_contact_links(spec: MachineSpec) -> list[str]:
    links = [j.child for j in spec.joints if j.role in ROLLING_ROLES]
    links += list(spec.ground_contact)
    return sorted(set(links))


def lowest_points(spec: MachineSpec, q: dict[str, float] | None = None,
                  collision_only: bool = True) -> dict[str, float]:
    """Lowest z per link (root frame)."""
    tfs = link_transforms(spec, q)
    out: dict[str, float] = {}
    for lk in spec.links:
        zs = [shape_min_z(tfs[lk.name], s) for s in lk.shapes if s.collision or not collision_only]
        if zs:
            out[lk.name] = min(zs)
    return out


def root_height(spec: MachineSpec) -> float:
    """Height of the root origin above the ground in the init pose.

    Defined by the lowest collision point of the ground-contact links
    (wheels, track rollers, ``spec.ground_contact``).
    """
    lows = lowest_points(spec)
    contact = [lows[n] for n in ground_contact_links(spec) if n in lows]
    if not contact:
        contact = list(lows.values())
    return -min(contact)


def bounding_box(spec: MachineSpec, q: dict[str, float] | None = None,
                 visual: bool = True) -> tuple[Vec3, Vec3]:
    """Axis-aligned bounds (root frame) of all visual (or collision) shapes."""
    tfs = link_transforms(spec, q)
    xs, ys, zs = [], [], []
    for lk in spec.links:
        for s in lk.shapes:
            if (visual and s.visual) or (not visual and s.collision):
                for p in shape_points(tfs[lk.name], s, 24):
                    xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def subtree_centroid(spec: MachineSpec, link: str, q: dict[str, float] | None = None) -> Vec3:
    """Volume-weighted centre of all shapes in the subtree rooted at ``link``."""
    tfs = link_transforms(spec, q)
    acc, wsum = (0.0, 0.0, 0.0), 0.0
    for name in subtree_links(spec, link):
        for s in spec.link(name).shapes:
            w = shape_volume(s)
            acc = vadd(acc, vscale(shape_tf(tfs[name], s).t, w))
            wsum += w
    return vscale(acc, 1.0 / wsum) if wsum > 0 else tfs[link].t


def shape_volume(shape: Shape) -> float:
    g = shape.geom
    if isinstance(g, Box):
        return g.size[0] * g.size[1] * g.size[2]
    if isinstance(g, Cylinder):
        return math.pi * g.radius ** 2 * g.length
    if isinstance(g, Sphere):
        return 4.0 / 3.0 * math.pi * g.radius ** 3
    raise TypeError(g)


# ---------------------------------------------------------------------------
# Planar base kinematics (shared by the env and the tests)
# ---------------------------------------------------------------------------

def ackermann_wheel_angles(delta: float, wheelbase: float, lateral: list[float],
                           steer_axle: str = "front") -> list[float]:
    """Per-wheel steer angles for virtual centre-wheel angle ``delta``.

    ``lateral`` = y position of each steered wheel (root frame, + = left).
    Front steer: delta > 0 turns left.  Rear steer: delta < 0 turns left.
    """
    s = 1.0 if steer_axle == "front" else -1.0
    if abs(math.tan(delta)) < 1e-6:
        return [0.0 for _ in lateral]
    r_icr = s * wheelbase / math.tan(delta)       # lateral offset of the ICR, + = left
    out = []
    for y in lateral:
        out.append(s * math.atan(wheelbase / (r_icr - y)) if abs(r_icr - y) > 1e-6
                   else s * math.copysign(math.pi / 2, r_icr))
    return out


def articulated_yaw_rate(v: float, gamma: float, gamma_dot: float,
                         front_length: float, rear_length: float) -> float:
    """Yaw rate of the FRONT frame of an articulated vehicle.

    ``gamma`` = rear-frame angle relative to the front frame (CCW positive),
    ``v`` = speed of the front axle midpoint.  Derived from the no-slip
    constraints at both axles; gamma < 0 turns left when driving forward.
    """
    return -(v * math.sin(gamma) + rear_length * gamma_dot) / (
        front_length * math.cos(gamma) + rear_length)
