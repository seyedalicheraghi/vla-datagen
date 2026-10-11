"""Offline validation of a MachineSpec (pure Python).

``validate(spec)`` returns a :class:`Report`.  Errors mean the machine will
misbehave in Isaac Sim (or contradicts its own description); warnings are
worth a look but can be deliberate.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from . import kinematics as K
from .spec import (Box, CATEGORIES, Cylinder, MachineSpec, ROLLING_ROLES, Shape, Sphere,
                   valid_name)
from .urdf_builder import build_urdf, link_inertia

# Teleop key pairs assigned to tools in order (see teleop_machine.py).
TOOL_KEY_PAIRS = [("I", "K"), ("J", "L"), ("U", "O"), ("T", "G"),
                  ("Y", "H"), ("Z", "X"), ("C", "V"), ("N", "M")]

GROUND_TOL = 0.03          # m — wheels must touch, other parts must clear the ground
CLEARANCE = 0.02           # m — non-contact parts must be at least this high in the init pose
OVERLAP_TOL = 0.03         # m — allowed interpenetration between non-adjacent links
MASS_TOL = 0.10
ENVELOPE_ERR, ENVELOPE_WARN = 0.20, 0.10

# Direction vocabulary for Tool.positive / Tool.negative (auto-verified).
_RAISE, _LOWER = {"raise", "up", "lift"}, {"lower", "down"}
_OUT, _IN = {"extend", "out"}, {"retract", "in"}
_LEFT, _RIGHT = {"left", "ccw"}, {"right", "cw"}
_KNOWN = _RAISE | _LOWER | _OUT | _IN | _LEFT | _RIGHT


@dataclass
class Report:
    name: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        lines = [f"== {self.name}: {'OK' if self.ok else 'FAIL'} "
                 f"({len(self.errors)} errors, {len(self.warnings)} warnings)"]
        lines += [f"  ERROR   {e}" for e in self.errors]
        lines += [f"  warning {w}" for w in self.warnings]
        for k, v in self.info.items():
            lines.append(f"  info    {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Oriented-box overlap (separating axis theorem)
# ---------------------------------------------------------------------------

def _obb(link_tf: K.Tf, shape: Shape) -> tuple[K.Vec3, list[K.Vec3], K.Vec3]:
    tf = K.shape_tf(link_tf, shape)
    g = shape.geom
    if isinstance(g, Box):
        half = tuple(s / 2 for s in g.size)
    elif isinstance(g, Cylinder):
        half = (g.radius, g.radius, g.length / 2)
    elif isinstance(g, Sphere):
        half = (g.radius,) * 3
    else:
        raise TypeError(g)
    axes = [(tf.R[0][i], tf.R[1][i], tf.R[2][i]) for i in range(3)]
    return tf.t, axes, half  # type: ignore[return-value]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def obb_penetration(a, b) -> float:
    """Minimum overlap depth of two OBBs (≤ 0 means separated)."""
    ca, axa, ha = a
    cb, axb, hb = b
    d = K.vsub(cb, ca)
    cands = list(axa) + list(axb)
    for u in axa:
        for v in axb:
            c = _cross(u, v)
            n = K.vnorm(c)
            if n > 1e-6:
                cands.append(K.vscale(c, 1.0 / n))
    best = math.inf
    for L in cands:
        ra = sum(ha[i] * abs(_dot(axa[i], L)) for i in range(3))
        rb = sum(hb[i] * abs(_dot(axb[i], L)) for i in range(3))
        overlap = ra + rb - abs(_dot(d, L))
        if overlap <= 0:
            return overlap
        best = min(best, overlap)
    return best


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(spec: MachineSpec) -> Report:
    r = Report(spec.name)
    E, W = r.errors.append, r.warnings.append

    # --- identity -----------------------------------------------------------
    if not valid_name(spec.name):
        E(f"machine name {spec.name!r} must be snake_case")
    if spec.category not in CATEGORIES:
        E(f"category {spec.category!r} not in {CATEGORIES}")
    if not spec.abbrev or spec.abbrev.upper() != spec.abbrev:
        E(f"abbrev {spec.abbrev!r} must be non-empty upper case")
    for attr in ("display_name", "reference", "description"):
        if not getattr(spec, attr).strip():
            E(f"{attr} is empty")

    # --- names / tree -------------------------------------------------------
    link_names = [lk.name for lk in spec.links]
    joint_names = [j.name for j in spec.joints]
    for kind, names in (("link", link_names), ("joint", joint_names)):
        for n in names:
            if not valid_name(n):
                E(f"{kind} name {n!r} must be snake_case")
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            E(f"duplicate {kind} names: {sorted(dup)}")
    if set(link_names) & set(joint_names):
        W(f"names used for both a link and a joint: {sorted(set(link_names) & set(joint_names))}")
    if spec.root not in link_names:
        E(f"root link {spec.root!r} does not exist")
        return r
    parents: dict[str, list[str]] = {}
    for j in spec.joints:
        for end in (j.parent, j.child):
            if end not in link_names:
                E(f"joint {j.name}: link {end!r} does not exist")
        parents.setdefault(j.child, []).append(j.name)
    for n in link_names:
        if n == spec.root:
            if n in parents:
                E(f"root link {n} has a parent joint {parents[n]}")
        elif len(parents.get(n, [])) != 1:
            E(f"link {n} must have exactly one parent joint (has {parents.get(n, [])})")
    if r.errors:
        return r
    reachable = set(K.subtree_links(spec, spec.root))
    if reachable != set(link_names):
        E(f"links not reachable from root: {sorted(set(link_names) - reachable)}")
        return r

    # --- links ----------------------------------------------------------------
    for lk in spec.links:
        if lk.mass <= 0:
            E(f"link {lk.name}: mass must be > 0")
        if not lk.shapes:
            E(f"link {lk.name}: has no shapes")
        if lk.shapes and not any(s.visual for s in lk.shapes):
            W(f"link {lk.name}: no visual shapes (invisible)")
        for s in lk.shapes:
            g = s.geom
            dims = (g.size if isinstance(g, Box) else
                    (g.radius, g.length) if isinstance(g, Cylinder) else (g.radius,))
            if min(dims) <= 0:
                E(f"link {lk.name}: shape with non-positive dimension {dims}")
            if s.visual and s.material not in spec.materials:
                E(f"link {lk.name}: unknown material {s.material!r}")
        if lk.shapes and lk.mass > 0:
            _, _, I = link_inertia(lk)
            a, b, c = I[0][0], I[1][1], I[2][2]
            if min(a, b, c) <= 0 or a + b < c * 0.999 or a + c < b * 0.999 or b + c < a * 0.999:
                E(f"link {lk.name}: inertia violates triangle inequality ({a:.3g}, {b:.3g}, {c:.3g})")

    # --- joints ---------------------------------------------------------------
    for j in spec.joints:
        if (j.type == "fixed") != (j.role == "fixed"):
            E(f"joint {j.name}: type {j.type} incompatible with role {j.role}")
        if j.type != "fixed":
            if K.vnorm(j.axis) < 1e-6:
                E(f"joint {j.name}: zero axis")
            if j.velocity <= 0 or j.effort <= 0:
                E(f"joint {j.name}: velocity and effort must be > 0")
        if j.type in ("revolute", "prismatic"):
            if not j.lower < j.upper:
                E(f"joint {j.name}: lower {j.lower} must be < upper {j.upper}")
            if not j.lower - 1e-9 <= j.init <= j.upper + 1e-9:
                E(f"joint {j.name}: init {j.init} outside [{j.lower}, {j.upper}]")
        if j.role in ("steer", "articulation"):
            if j.type != "revolute" or abs(K.vunit(j.axis)[2]) < 0.99:
                E(f"joint {j.name}: {j.role} joints must be revolute about z")
            if abs(j.init) > 1e-9:
                E(f"joint {j.name}: {j.role} joints must start at 0")
        if j.role in ROLLING_ROLES:
            if j.type != "continuous":
                E(f"joint {j.name}: rolling joints must be continuous")
            if j.wheel_radius <= 0:
                E(f"joint {j.name}: wheel_radius must be > 0")
            cyls = [s.geom.radius for s in spec.link(j.child).shapes if isinstance(s.geom, Cylinder)]
            if cyls and abs(max(cyls) - j.wheel_radius) > 0.05 * j.wheel_radius:
                W(f"joint {j.name}: wheel_radius {j.wheel_radius} differs from the largest "
                  f"cylinder radius {max(cyls):.3f} on {j.child}")
        if j.role == "follower":
            if not j.follows:
                E(f"joint {j.name}: follower needs follows=(source, k, c)")
            elif j.follows[0] not in joint_names or spec.joint(j.follows[0]).type == "fixed":
                E(f"joint {j.name}: follows unknown/fixed joint {j.follows[0]!r}")
            elif spec.joint(j.follows[0]).role == "follower":
                E(f"joint {j.name}: followers may not chain")
        elif j.follows:
            E(f"joint {j.name}: follows= set but role is {j.role}")

    # --- tools / action layout ---------------------------------------------
    tool_joints = [t.joint for t in spec.tools]
    labels = [t.label for t in spec.tools]
    if len(set(tool_joints)) != len(tool_joints):
        E("a joint is used by more than one tool")
    if len(set(labels)) != len(labels) or {"drive", "turn"} & set(labels):
        E(f"tool labels must be unique and not 'drive'/'turn': {labels}")
    for t in spec.tools:
        if not valid_name(t.label):
            E(f"tool label {t.label!r} must be snake_case")
        if t.joint not in joint_names:
            E(f"tool {t.label}: joint {t.joint!r} does not exist")
        elif spec.joint(t.joint).role != "tool":
            E(f"tool {t.label}: joint {t.joint} has role {spec.joint(t.joint).role}, expected 'tool'")
    for j in spec.joints_with_role("tool"):
        if j.name not in tool_joints:
            E(f"joint {j.name} has role 'tool' but no Tool drives it")
    if len(spec.tools) > len(TOOL_KEY_PAIRS):
        E(f"{len(spec.tools)} tools but teleop has only {len(TOOL_KEY_PAIRS)} key pairs")

    # --- drive ----------------------------------------------------------------
    d = spec.drive
    if d.max_speed <= 0:
        E("drive.max_speed must be > 0")
    jf = K.joint_frames(spec)
    rolling = [j for j in spec.joints if j.role in ROLLING_ROLES]
    if d.type == "skid":
        if d.max_yaw_rate <= 0 or d.track_width <= 0:
            E("skid drive needs max_yaw_rate > 0 and track_width > 0")
        lefts = [j for j in rolling if j.role == "roll_left"]
        rights = [j for j in rolling if j.role == "roll_right"]
        if not lefts or not rights:
            E("skid drive needs at least one roll_left and one roll_right joint")
        else:
            yl = sum(jf[j.name].t[1] for j in lefts) / len(lefts)
            yr = sum(jf[j.name].t[1] for j in rights) / len(rights)
            if yl <= 0 or yr >= 0:
                E(f"roll_left joints must be at +y and roll_right at -y (got {yl:.2f}, {yr:.2f})")
            if abs((yl - yr) - d.track_width) > 0.15:
                W(f"track_width {d.track_width} vs rolling-element spacing {yl - yr:.2f}")
    elif d.type == "ackermann":
        steers = spec.joints_with_role("steer")
        if d.wheelbase <= 0 or d.max_steer <= 0 or d.steer_rate <= 0:
            E("ackermann drive needs wheelbase, max_steer, steer_rate > 0")
        if len(steers) < 2:
            E("ackermann drive needs at least two steer joints")
        if any(j.role in ("roll_left", "roll_right") for j in rolling):
            E("ackermann machines should use role 'wheel', not roll_left/roll_right")
        if steers and d.wheelbase > 0:
            sx = sum(jf[j.name].t[0] for j in steers) / len(steers)
            if abs(abs(sx - d.fixed_axle_x) - d.wheelbase) > 0.15:
                E(f"steer joints are {abs(sx - d.fixed_axle_x):.2f} m from fixed_axle_x "
                  f"but wheelbase is {d.wheelbase}")
            if (d.steer_axle == "front") != (sx > d.fixed_axle_x):
                E(f"steer_axle={d.steer_axle} but steer joints are on the other side of fixed_axle_x")
            angles = K.ackermann_wheel_angles(d.max_steer, d.wheelbase,
                                              [jf[j.name].t[1] for j in steers], d.steer_axle)
            angles += K.ackermann_wheel_angles(-d.max_steer, d.wheelbase,
                                               [jf[j.name].t[1] for j in steers], d.steer_axle)
            lim = min(min(abs(j.lower), abs(j.upper)) for j in steers)
            if max(abs(a) for a in angles) > lim + 1e-6:
                E(f"inner-wheel angle {max(abs(a) for a in angles):.3f} rad at max_steer exceeds "
                  f"steer joint limit {lim:.3f}")
            fixed = [j for j in rolling if abs(jf[j.name].t[0] - d.fixed_axle_x) < 1.0
                     and spec.joint(j.name).parent == spec.root]
            if not fixed:
                W("no root-mounted wheels within 1 m of fixed_axle_x")
    elif d.type == "articulated":
        if d.articulation_joint not in joint_names:
            E(f"articulation_joint {d.articulation_joint!r} does not exist")
        else:
            aj = spec.joint(d.articulation_joint)
            if aj.role != "articulation" or aj.parent != spec.root:
                E("articulation joint must have role 'articulation' and the root as parent")
            if d.front_length <= 0 or d.rear_length <= 0 or d.articulation_rate <= 0:
                E("articulated drive needs front_length, rear_length, articulation_rate > 0")
            hx = jf[aj.name].t[0]
            rear_links = set(K.subtree_links(spec, aj.child))
            front_w = [j for j in rolling if j.child not in rear_links]
            rear_w = [j for j in rolling if j.child in rear_links]
            if not front_w or not rear_w:
                E("articulated drive needs wheels on both the front and rear frames")
            else:
                fx = sum(jf[j.name].t[0] for j in front_w) / len(front_w) - hx
                rx = hx - sum(jf[j.name].t[0] for j in rear_w) / len(rear_w)
                if abs(fx - d.front_length) > 0.15:
                    E(f"front_length {d.front_length} but front wheels are {fx:.2f} m ahead of the hinge")
                if abs(rx - d.rear_length) > 0.15:
                    E(f"rear_length {d.rear_length} but rear wheels are {rx:.2f} m behind the hinge")
    else:
        E(f"unknown drive type {d.type!r}")
    if spec.joints_with_role("articulation") and d.type != "articulated":
        E("articulation joint present but drive type is not 'articulated'")
    if spec.joints_with_role("steer") and d.type != "ackermann":
        E("steer joints present but drive type is not 'ackermann'")

    # --- cameras --------------------------------------------------------------
    cam_names = sorted(c.name for c in spec.cameras)
    if cam_names != ["front", "left", "right"]:
        E(f"cameras must be exactly front/left/right, got {cam_names}")
    for c in spec.cameras:
        if c.link not in link_names:
            E(f"camera {c.name}: link {c.link!r} does not exist")

    # --- mass / size ------------------------------------------------------------
    total = spec.total_mass
    if abs(total - spec.operating_mass) > MASS_TOL * spec.operating_mass:
        E(f"link masses sum to {total:.0f} kg but operating_mass is {spec.operating_mass:.0f} kg")
    lo, hi = K.bounding_box(spec)
    dims = tuple(hi[i] - lo[i] for i in range(3))
    for axis, got, want in zip("LWH", dims, spec.envelope):
        rel = abs(got - want) / want
        if rel > ENVELOPE_ERR:
            E(f"model {axis} = {got:.2f} m vs envelope {want:.2f} m ({rel:.0%} off)")
        elif rel > ENVELOPE_WARN:
            W(f"model {axis} = {got:.2f} m vs envelope {want:.2f} m ({rel:.0%} off)")
    r.info["bbox_LWH_m"] = tuple(round(x, 2) for x in dims)
    r.info["mass_kg"] = round(total)

    # --- ground contact -----------------------------------------------------------
    contact = K.ground_contact_links(spec)
    if not contact:
        E("no ground-contact links (rolling joints or spec.ground_contact)")
    for n in spec.ground_contact:
        if n not in link_names:
            E(f"ground_contact link {n!r} does not exist")
    h = K.root_height(spec)
    r.info["root_height_m"] = round(h, 3)
    if h <= 0:
        E(f"root origin would be at or below the ground (root_height {h:.3f})")
    lows = K.lowest_points(spec)
    for n, z in lows.items():
        zg = z + h
        if n in contact:
            if n not in spec.ground_contact and zg > GROUND_TOL:
                E(f"rolling link {n} floats {zg:.3f} m above the ground in the init pose")
        elif zg < CLEARANCE:
            E(f"link {n} is {zg:.3f} m from the ground in the init pose "
              f"(non-contact parts need ≥ {CLEARANCE} m clearance)")

    # --- self-overlap in the init pose ---------------------------------------------
    adjacent = {frozenset((j.parent, j.child)) for j in spec.joints}
    allowed = {frozenset(p) for p in spec.allowed_overlaps}
    tfs = K.link_transforms(spec)
    cols = [(lk.name, s, _obb(tfs[lk.name], s)) for lk in spec.links for s in lk.shapes if s.collision]
    reported = set()
    for i in range(len(cols)):
        for k in range(i + 1, len(cols)):
            na, sa, oa = cols[i]
            nb, sb, ob = cols[k]
            pair = frozenset((na, nb))
            if na == nb or pair in adjacent or pair in allowed or pair in reported:
                continue
            pen = obb_penetration(oa, ob)
            if pen > OVERLAP_TOL:
                reported.add(pair)
                both_boxes = isinstance(sa.geom, Box) and isinstance(sb.geom, Box)
                msg = f"links {na} and {nb} interpenetrate by {pen:.2f} m in the init pose"
                (E if both_boxes else W)(msg + ("" if both_boxes else " (round shape bound; may be ok)"))

    # --- tool direction semantics ------------------------------------------------
    for t in spec.tools:
        if t.joint not in joint_names:
            continue
        j = spec.joint(t.joint)
        words = {t.positive.lower(), t.negative.lower()}
        if not (words & _KNOWN):
            W(f"tool {t.label}: directions {t.positive!r}/{t.negative!r} not auto-verifiable")
            continue
        # Compare the two ends of the range (robust for folded init poses);
        # continuous joints have no ends, so use ±0.1 rad around init.
        if j.type == "continuous":
            q0, q1 = j.init - 0.1, j.init + 0.1
        else:
            q0, q1 = j.lower, j.upper
        c0 = K.subtree_centroid(spec, j.child, {j.name: q0})
        c1 = K.subtree_centroid(spec, j.child, {j.name: q1})
        o = K.joint_frames(spec)[j.name].t
        delta = {
            "raise": c1[2] - c0[2],
            "out": K.vnorm(c1) - K.vnorm(c0),
            "left": _cross(K.vsub(c0, o), K.vsub(c1, o))[2],
        }
        for word, sign in ((t.positive.lower(), 1.0), (t.negative.lower(), -1.0)):
            for group, key, s in ((_RAISE, "raise", 1), (_LOWER, "raise", -1),
                                  (_OUT, "out", 1), (_IN, "out", -1),
                                  (_LEFT, "left", 1), (_RIGHT, "left", -1)):
                if word in group:
                    got = delta[key] * sign * s
                    if abs(delta[key]) < 1e-4:
                        W(f"tool {t.label}: '{word}' produces no measurable motion near init")
                    elif got < 0:
                        E(f"tool {t.label}: command {'+1' if sign > 0 else '-1'} is labelled "
                          f"'{word}' but moves the opposite way (flip the joint axis)")

    # --- URDF round trip -------------------------------------------------------------
    try:
        root = ET.fromstring(build_urdf(spec))
        if len(root.findall("link")) != len(spec.links) or len(root.findall("joint")) != len(spec.joints):
            E("URDF link/joint count mismatch")
    except ET.ParseError as exc:
        E(f"generated URDF does not parse: {exc}")

    r.info["action"] = spec.action_labels
    return r
