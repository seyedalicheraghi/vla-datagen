"""Single-drum vibratory soil compactor (SDC), 11-12 t class — modelled on a
CAT CS56B / Bomag BW 213 D-5 fitted with the optional front leveling blade.

Reference numbers (class datasheet values, rounded; CAT CS56B unless noted):
    operating mass with ROPS/FOPS cab ≈ 11.5 t (BW 213 D-5 ≈ 12.7 t), about
    half of it on the drum module; + optional leveling blade ≈ 0.5 t → 12.0 t ·
    smooth drum 2,134 mm wide × 1,534 mm diameter (25 mm shell) · overall
    length 5.83 m without the blade (BW 213 D-5: 5.94 m) · overall width
    2.30 m (BW 213: 2.25 m) · height to the top of the ROPS cab 3.09 m
    (BW 213: 2.99 m) · wheelbase 2.90 m (BW 213: 2.96 m) · ground clearance
    0.49 m · rear tyres 23.1-26 12PR (OD ≈ 1.57 m, section 0.59 m) on a ≈1.70 m
    gauge (width over the tyres ≈ 2.29 m) · centre articulation ±34°,
    oscillation ±15° (not modelled) · inside turning radius at the drum edge
    ≈ 3.7 m · hydrostatic travel 0-11.2 km/h (3.1 m/s) forward and reverse ·
    Cat C4.4 engine 117 kW · vibration 31.9 / 28 Hz, amplitude 2.1 / 1.0 mm
    (not modelled).
    Leveling blade option (assumed, typical for the class): 2.50 m wide,
    0.70 m high curved moldboard on two push arms pinned to the drum-frame
    side plates, one lift cylinder on each side plate; cutting edge 0.50 m
    above grade fully raised, 0.12 m cut depth, full stroke ≈ 2.5 s.

Model (offline FK): L × W × H = 6.31 × 2.68 × 3.12 m in the travel pose (blade
fully raised, beacon included; 2.53 m wide over the blade and push arms, 2.68 m
over the blade-camera posts); 11.97 t of which 6.1 t on the drum module; hinge
→ drum axis 1.40 m, hinge → rear axle 1.50 m (wheelbase 2.90 m); inside /
outside turning radius at the drum edges 3.70 / 5.83 m at full articulation;
0.52 m clearance under the hitch.  The envelope L / W (6.30 / 2.50 m) is the
datasheet 5.83 / 2.30 m plus the ASSUMED blade (+0.48 m length, 2.50 m width),
so the envelope check on L and W only confirms that assumption; H is the
datasheet cab height.

Joint ranges / rates:
    articulation   ±0.593 rad (34°), 0.30 rad/s (lock to lock in 4 s)
    leveling_blade -0.405 rad (cutting edge 0.12 m below grade) … 0 (fully
                   raised, edge 0.50 m above grade = travel pose); 0.16 rad/s
                   → full stroke in 2.5 s; edge at grade at -0.318.  The whole
                   blade rotates about the push-arm pins, so its pitch changes
                   with height; the moldboard is set 0.30 rad back on the arms
                   so its chord (edge → top) is vertical with the edge at grade
                   (it leans 17° back fully raised and 6° forward at full cut).
                   The bottom 0.2 m of the moldboard and end plates is visual
                   only: at full cut the visual edge is 0.12 m into the (rigid)
                   ground plane but nothing collidable is below grade.
    drum, rear tyres  cosmetic spin at v / r (role "wheel"; r = 0.767 / 0.785 m).

The drum frame is a box frame around the drum: side plates carrying the drum
bearings, a rear cross beam and a low front cross beam (top 1.10 m above the
ground, below the 1.53 m drum crown), and a light / camera post at each front
corner (driving light above the raised blade, blade camera on a short forward
arm).  The two blade lift cylinders are visual followers of the blade joint:
the barrels pivot on brackets outside the side plates and the rods slide out
of them to lugs on top of the push arms; both fits are least-squares linear in
the blade angle and the rod eyes stay within 1.5 cm of their lugs (see
``_cylinder_fit``).

Cameras (share of image pixels showing the blade, ray-cast with the env's
pinhole: raised / q -0.2 / edge at grade / full cut):
    front  operator eye height in the cab, looking over the drum crown at the
           ground ahead (driving view).  Only the top of the fully raised blade
           shows over the drum crown (0.8 %); once the edge is below ≈0.25 m
           (q < -0.15) the blade is hidden behind the drum, as it is for the
           real operator.
    left / right  blade cameras on the drum-frame corner posts, looking down
           and slightly inward at the back of the moldboard and the blade end
           on the ground: 15 / 10 / 8 / 6 %, independent of articulation (the
           blade and the cameras are both on the front frame).

Init pose = travel pose: frames straight, blade fully raised.

Root frame: FRONT drum frame, origin on the articulation hinge axis at drum-axle
height (z = 0 is 0.767 m above the ground); the rear tractor (cab, engine,
rear axle) is the child of the articulation joint.  Livery: Cat and Bomag
compactors are yellow, so DEFAULT_MATERIALS are used.
"""

from __future__ import annotations

import math

from ..kinematics import mat_mul, rot_y, rot_z
from ..spec import (Box, Camera, Drive, Joint, Link, MachineSpec, Shape, Tool, beam, box, cyl,
                    wheel_joint)

# --- drum (front frame) -------------------------------------------------------------
DRUM_R, DRUM_W = 0.767, 2.134         # 1,534 mm × 2,134 mm smooth drum
DRUM_X = 1.40                         # hinge → drum axis (drive.front_length)
Z0 = DRUM_R                           # ground is at z = -Z0 in the root frame


def agl(h: float) -> float:
    """Height above ground → root-frame z."""
    return h - Z0


PLATE_Y = 1.115                       # drum-frame side plates (0.05 thick, 1.09…1.14)
FRAME_TOP = 1.40                      # top of the side plates above ground (drum crown 1.53)

# --- rear tractor ---------------------------------------------------------------------
WHEEL_R, TYRE_W = 0.785, 0.59         # 23.1-26 tyres
GAUGE = 1.70
REAR_X = -1.50                        # hinge → rear axle (drive.rear_length)
ARTIC = 0.593                         # ±34° articulation
DECK = 1.67                           # operator deck / cab floor above ground
CAB_X0, CAB_X1, CAB_HW = -1.95, -0.35, 0.80
ROOF = 3.04                           # top of the cab roof above ground
HOOD_X0 = -3.38                       # rear face of the engine hood

# --- leveling blade ---------------------------------------------------------------------
PIVOT = (1.10, agl(1.12))             # push-arm pins on the side plates (root x, z)
ARM_Y = 1.185                         # push arms just outside the side plates
BLADE_W, BLADE_H = 2.50, 0.70
EDGE = (1.70, -0.62)                  # cutting edge in the blade frame (0.50 m AGL raised)
BLADE_TILT = 0.30                     # moldboard set back on the arms: chord ≈ vertical at grade
BLADE_LO, BLADE_HI = -0.405, 0.0      # edge 0.12 m below grade … 0.50 m above grade
CYL_BASE = (1.70, agl(1.30))          # lift-cylinder base pins on the side plates (outside)
CYL_EYE = (1.21, -0.18)               # lift-cylinder eyes on top of the push arms (blade frame)
CYL_Y = 1.20                          # cylinders ride on top of / just outside the push arms

# --- blade cameras on light/camera posts at the drum-frame front corners: lens (x, |y|,
# height AGL) on a short forward arm 0.16 m ahead of the post; pitch down, yaw inward
BLADE_CAM = (2.50, 1.28, 1.70)
BLADE_CAM_PITCH, BLADE_CAM_YAW = 0.85, 0.15


# -----------------------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------------------

def _rot(p: tuple[float, float], q: float) -> tuple[float, float]:
    """Rotate an (x, z) point by q about axis (0, -1, 0) (+q raises +x)."""
    c, s = math.cos(q), math.sin(q)
    return (c * p[0] - s * p[1], s * p[0] + c * p[1])


def _eye_root(q: float) -> tuple[float, float]:
    r = _rot(CYL_EYE, q)
    return (PIVOT[0] + r[0], PIVOT[1] + r[1])


def _mb(x: float, z: float) -> tuple[float, float]:
    """Moldboard frame (origin on the cutting edge, z up the blade) → blade frame (x, z)."""
    r = _rot((x, z), BLADE_TILT)
    return (EDGE[0] + r[0], EDGE[1] + r[1])


def _mb_box(sx: float, sy: float, sz: float, x: float, y: float, z: float, **kw) -> Shape:
    """Box centred at moldboard-frame (x, y, z), tilted with the moldboard."""
    bx, bz = _mb(x, z)
    return box(sx, sy, sz, at=(bx, y, bz), rpy=(0.0, -BLADE_TILT, 0.0), **kw)


def _blade_shapes() -> list[Shape]:
    """Curved moldboard with its back box, ribs and end plates, plus the push arms.

    Collision: the bottom 0.2 m of the moldboard and end plates is visual only, so
    nothing collidable reaches below grade at full cut (the env's ground plane is
    rigid; the visual cutting edge still goes 0.12 m into it).
    """
    hb, sides = BLADE_H, (1.0, -1.0)
    prof = [(0.0, 0.0), (-0.14, 0.20), (-0.17, 0.42), (-0.11, 0.61), (0.0, hb)]
    moldboard = [beam(_mb(*a), _mb(*b), BLADE_W, 0.05, collision=i > 0)
                 for i, (a, b) in enumerate(zip(prof, prof[1:]))]
    cutting_edge = [beam(_mb(0.01, -0.005), _mb(-0.07, 0.11), BLADE_W - 0.04, 0.035,
                         mat="steel", collision=False)]
    # back box and ribs (behind the concave face) kept low and shallow so that, with the
    # moldboard set back, they clear the front cross member and side plates when raised
    # and the drum at full cut
    back = [_mb_box(0.16, BLADE_W - 0.10, 0.30, -0.25, 0.0, 0.27)]
    ribs = [_mb_box(0.07, 0.05, 0.43, -0.19, y, 0.335, mat="dark", collision=False)
            for y in (-0.80, -0.27, 0.27, 0.80)]
    end_plates = [shp for s in sides for shp in (
        _mb_box(0.28, 0.04, hb, -0.11, s * (BLADE_W / 2 - 0.02), hb / 2, collision=False),
        _mb_box(0.28, 0.04, hb - 0.20, -0.11, s * (BLADE_W / 2 - 0.02), (hb + 0.20) / 2,
                visual=False))]
    arm_end = (EDGE[0] - 0.30, EDGE[1] + 0.28)                    # inside the back box
    arms = [beam((0.0, 0.0), arm_end, 0.07, 0.16, y=s * ARM_Y) for s in sides]
    arm_pins = [cyl(0.09, 0.11, at=(0.0, s * ARM_Y, 0.0), axis="y", mat="steel", collision=False)
                for s in sides]
    lugs = [box(0.14, 0.10, 0.16, at=(CYL_EYE[0], s * CYL_Y, CYL_EYE[1] - 0.04), mat="dark",
                collision=False) for s in sides]                  # lift-cylinder lugs on the arms
    return moldboard + cutting_edge + back + ribs + end_plates + arms + arm_pins + lugs


def _camera_post(s: float) -> list[Shape]:
    """Light / camera post at a drum-frame front corner (side s = +1 left, -1 right).

    Visual only.  The driving light (above the raised blade) and the post sit behind
    and below the lens, the camera housing behind it, so none of them shows up in
    the camera's own image.
    """
    x, y, h = BLADE_CAM[0], s * BLADE_CAM[1], BLADE_CAM[2]
    p, yaw = BLADE_CAM_PITCH, -s * BLADE_CAM_YAW
    fwd = (math.cos(yaw) * math.cos(p), math.sin(yaw) * math.cos(p), -math.sin(p))
    xp = x - 0.16                                                 # post axis, behind the lens
    z_foot, z_cam, z_lamp = agl(FRAME_TOP + 0.015), agl(h), agl(h - 0.24)
    return [
        box(0.12, abs(y) - PLATE_Y + 0.05, 0.03,
            at=(xp, s * (PLATE_Y + abs(y)) / 2, z_foot), mat="dark", collision=False),
        cyl(0.025, z_cam - z_foot, at=(xp, y, (z_cam + z_foot) / 2), mat="dark", collision=False),
        box(0.12, 0.04, 0.04, at=(xp + 0.05, y, z_cam - 0.03), mat="dark", collision=False),
        box(0.10, 0.09, 0.07, at=(x - 0.06 * fwd[0], y - 0.06 * fwd[1], z_cam - 0.06 * fwd[2]),
            rpy=(0.0, p, yaw), mat="accent", collision=False),               # camera housing
        box(0.08, 0.16, 0.11, at=(xp + 0.03, y - s * 0.06, z_lamp), mat="accent", collision=False),
        box(0.01, 0.13, 0.08, at=(xp + 0.075, y - s * 0.06, z_lamp), mat="white", collision=False),
    ]


def _fit1(f, lo: float, hi: float) -> tuple[float, float]:
    """Least-squares line k*q + c through f(q) on [lo, hi]."""
    qs = [lo + (hi - lo) * i / 40 for i in range(41)]
    ys = [f(q) for q in qs]
    mq, my = sum(qs) / len(qs), sum(ys) / len(ys)
    k = sum((q - mq) * (y - my) for q, y in zip(qs, ys)) / sum((q - mq) ** 2 for q in qs)
    return k, my - k * mq


def _cylinder_fit() -> dict:
    """Linear follower fits for the blade lift cylinders (barrel angle, rod stroke).

    Returns the fits plus the worst distance between the modelled rod eye and its
    lug on the push arm over the blade range (``err``).
    """
    def vec(q):
        e = _eye_root(q)
        return (e[0] - CYL_BASE[0], e[1] - CYL_BASE[1])

    ka, ca = _fit1(lambda q: math.atan2(vec(q)[1], vec(q)[0]), BLADE_LO, BLADE_HI)
    d0 = math.hypot(*vec(BLADE_HI))
    ks, cs = _fit1(lambda q: math.hypot(*vec(q)) - d0, BLADE_LO, BLADE_HI)
    dists = [math.hypot(*vec(BLADE_LO + (BLADE_HI - BLADE_LO) * i / 40)) for i in range(41)]
    err = 0.0
    for i in range(41):
        q = BLADE_LO + (BLADE_HI - BLADE_LO) * i / 40
        a, s = ka * q + ca, d0 + ks * q + cs
        eye = _eye_root(q)
        err = max(err, math.hypot(CYL_BASE[0] + s * math.cos(a) - eye[0],
                                  CYL_BASE[1] + s * math.sin(a) - eye[1]))
    return {"ka": ka, "ca": ca, "ks": ks, "cs": cs, "d0": d0,
            "d_min": min(dists), "d_max": max(dists), "err": err}


def _rpy_of(m) -> tuple[float, float, float]:
    """Rotation matrix → URDF roll-pitch-yaw (R = Rz·Ry·Rx)."""
    pitch = -math.asin(max(-1.0, min(1.0, m[2][0])))
    return (math.atan2(m[2][1], m[2][2]), pitch, math.atan2(m[1][0], m[0][0]))


def _rear_wheel(name: str, mass: float) -> Link:
    """23.1-26 traction tyre with chevron lugs, yellow rim and a spoke bar (spins about y)."""
    r, w, lug = WHEEL_R, TYRE_W, 0.045
    shapes = [
        cyl(r, w, axis="y", mat="rubber", visual=False),                 # rolling / collision
        cyl(r - lug, w, axis="y", mat="rubber", collision=False),        # carcass
        cyl(0.44, w * 1.02, axis="y", mat="hub", collision=False),       # rim
        cyl(0.18, w * 1.08, axis="y", mat="dark", collision=False),      # hub / final drive
        box(0.82, w * 1.10, 0.10, mat="accent", collision=False),        # spoke bar (shows spin)
    ]
    n = 18
    for i in range(n):
        for s in (1.0, -1.0):
            th = 2.0 * math.pi * (i + (0.5 if s < 0 else 0.0)) / n
            rpy = _rpy_of(mat_mul(rot_y(th), rot_z(s * 0.55)))
            rc = r - lug / 2
            shapes.append(Shape(Box((0.08, w * 0.50, lug)),
                                (rc * math.sin(th), s * w * 0.25, rc * math.cos(th)),
                                rpy, "rubber", False, True))
    return Link(name, mass, shapes)


def _cab(x0: float, x1: float, hw: float, z0: float, z1: float) -> list[Shape]:
    """ROPS/FOPS cab on the rear deck: glass all round, dark pillars, yellow roof."""
    cx, lx = (x0 + x1) / 2, x1 - x0
    zr = z1 - 0.15                     # underside of the roof
    zb = z0 + 0.30                     # top of the lower body panels
    sides = (1.0, -1.0)
    z_hood = agl(2.36)                 # rear window starts above the hood
    return [
        # collision volume (invisible) + lower body panels
        box(lx, 2 * hw, zr - z0, at=(cx, 0.0, (z0 + zr) / 2), visual=False),
        box(lx, 2 * hw, zb - z0, at=(cx, 0.0, (z0 + zb) / 2)),
        box(0.04, 2 * hw - 0.10, 0.10, at=(x1 + 0.005, 0.0, z0 + 0.12), mat="accent",
            collision=False),
        # roof (ROPS canopy) with overhang and a dark rim
        box(lx + 0.12, 2 * hw + 0.10, 0.15, at=(cx - 0.02, 0.0, z1 - 0.075)),
        box(lx + 0.14, 2 * hw + 0.12, 0.04, at=(cx - 0.02, 0.0, zr + 0.02), mat="accent",
            collision=False),
        # glass: windscreen, door / side windows, rear window above the hood
        box(0.03, 2 * hw - 0.12, zr - zb - 0.04, at=(x1 + 0.005, 0.0, (zb + zr) / 2),
            mat="glass", collision=False),
        *[box(lx - 0.12, 0.03, zr - zb - 0.04, at=(cx, s * (hw + 0.005), (zb + zr) / 2),
              mat="glass", collision=False) for s in sides],
        box(0.03, 2 * hw - 0.16, zr - z_hood - 0.04, at=(x0 - 0.005, 0.0, (z_hood + zr) / 2),
            mat="glass", collision=False),
        # pillars and door post (left side door)
        *[box(0.07, 0.07, zr - z0, at=(x, s * (hw - 0.025), (z0 + zr) / 2), mat="accent",
              collision=False) for x in (x0 + 0.035, x1 - 0.035) for s in sides],
        box(0.05, 0.035, zr - zb, at=(cx + 0.10, hw + 0.02, (zb + zr) / 2), mat="accent",
            collision=False),
        box(0.04, 0.04, 0.55, at=(cx + 0.02, hw + 0.06, zb + 0.35), mat="accent",
            collision=False),                                                     # door handle
        # mirrors on the front pillars, roof work lights, beacon
        *[box(0.03, 0.26, 0.03, at=(x1 + 0.02, s * (hw + 0.13), zr - 0.15), mat="accent",
              collision=False) for s in sides],
        *[box(0.04, 0.12, 0.22, at=(x1 + 0.04, s * (hw + 0.26), zr - 0.30), mat="accent",
              collision=False) for s in sides],
        *[box(0.08, 0.16, 0.08, at=(x1 + 0.06, s * (hw - 0.15), z1 - 0.08), mat="white",
              collision=False) for s in sides],
        *[box(0.08, 0.16, 0.08, at=(x0 - 0.06, s * (hw - 0.15), z1 - 0.08), mat="white",
              collision=False) for s in sides],
        cyl(0.06, 0.07, at=(x0 + 0.25, -0.45, z1 + 0.035), mat="orange", collision=False),
    ]


def _cyl_follower(name: str, parent: str, child: str, at: tuple[float, float], src: Joint,
                  k: float, c: float) -> Joint:
    """Barrel pivot at root (x, z) = ``at``: child +x elevation (about -y) = k*q_src + c,
    with the joint zero centred on the source's mid-range."""
    q_mid = (src.lower + src.upper) / 2
    base = k * q_mid + c
    c0 = -k * q_mid
    lo, hi = sorted((k * src.lower + c0, k * src.upper + c0))
    return Joint(name, "revolute", parent, child, xyz=(at[0], 0.0, at[1]),
                 rpy=(0.0, -base, 0.0), axis=(0, -1, 0), lower=lo - 0.02, upper=hi + 0.02,
                 init=k * src.init + c0, velocity=abs(k) * src.velocity + 0.1,
                 role="follower", follows=(src.name, k, c0))


# -----------------------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------------------

def build() -> MachineSpec:
    sides = (1.0, -1.0)
    y_w = GAUGE / 2

    # ------------------------------------------------------------- front (drum) frame
    xf0, xf1 = 0.45, 2.40                                         # side-plate extent
    inner_w = 2 * PLATE_Y - 0.05                                  # between the side plates
    # side plates around the drum ends (the drum crown stands 0.13 m above them),
    # bearing hangers down to the drum axle, drive / vibration covers, top trim, decal
    side_plates = [shp for s in sides for shp in (
        box(xf1 - xf0, 0.05, FRAME_TOP - 0.95, at=((xf0 + xf1) / 2, s * PLATE_Y,
            agl((FRAME_TOP + 0.95) / 2))),
        box(0.56, 0.05, 0.33, at=(DRUM_X, s * PLATE_Y, agl(0.785))),
        cyl(0.20, 0.06, at=(DRUM_X, s * PLATE_Y, 0.0), axis="y", mat="dark", collision=False),
        box(xf1 - xf0 + 0.02, 0.07, 0.05, at=((xf0 + xf1) / 2, s * PLATE_Y,
            agl(FRAME_TOP - 0.02)), mat="accent", collision=False),
        box(0.50, 0.01, 0.10, at=(0.80, s * (PLATE_Y + 0.03), agl(1.25)), mat="accent",
            collision=False),
    )]
    # front cross member low in front of the drum (top 1.10 m AGL, the drum crown stands
    # above the frame and it keeps the blade cameras' view clear), rear one behind it
    cross_members = [
        box(0.18, inner_w, 0.25, at=(2.28, 0.0, agl(0.975))),
        box(0.26, inner_w, FRAME_TOP - 1.05, at=(0.53, 0.0, agl((FRAME_TOP + 1.05) / 2))),
    ]
    # lift-cylinder base brackets on the outside of the side plates
    cyl_brackets = [box(0.20, 0.10, 0.16, at=(CYL_BASE[0], s * 1.19, CYL_BASE[1]), mat="dark")
                    for s in sides]
    # front and rear drum scrapers on holder bars between the side plates
    scrapers = [shp for sx in (1.0, -1.0) for shp in (
        box(0.025, DRUM_W - 0.06, 0.18, at=(DRUM_X + sx * (DRUM_R + 0.035), 0.0, 0.06),
            mat="dark", collision=False),
        box(0.06, inner_w, 0.08, at=(DRUM_X + sx * (DRUM_R + 0.075), 0.0, 0.15),
            mat="dark", collision=False),
    )]
    # neck down to the articulation hinge (0.52 m ground clearance)
    neck = [
        box(0.24, 0.84, 0.68, at=(0.39, 0.0, agl(0.86))),                    # x 0.27…0.51
        box(0.56, 0.62, 0.12, at=(-0.02, 0.0, agl(1.14)), mat="dark"),        # upper clevis
        box(0.56, 0.62, 0.12, at=(-0.02, 0.0, agl(0.58)), mat="dark"),        # lower clevis
        cyl(0.085, 0.16, at=(0.0, 0.0, agl(1.28)), mat="steel", collision=False),
        cyl(0.085, 0.16, at=(0.0, 0.0, agl(0.44)), mat="steel", collision=False),
    ]
    front = Link("front_frame", 2050.0, side_plates + cross_members + cyl_brackets + scrapers
                 + neck + _camera_post(1.0) + _camera_post(-1.0))

    # ------------------------------------------------------------------------- drum
    drum = Link("drum", 3400.0, [
        cyl(DRUM_R, DRUM_W, axis="y", mat="steel"),
        # end plates: dark rim ring, steel disc, black spoke bar (shows spin), yellow hub
        *[cyl(DRUM_R - 0.03, 0.012, at=(0.0, s * (DRUM_W / 2 + 0.006), 0.0), axis="y",
              mat="dark", collision=False) for s in sides],
        *[cyl(DRUM_R - 0.09, 0.012, at=(0.0, s * (DRUM_W / 2 + 0.012), 0.0), axis="y",
              mat="steel", collision=False) for s in sides],
        box(DRUM_R * 1.7, DRUM_W + 0.044, 0.10, mat="accent", collision=False),
        *[cyl(0.24, 0.02, at=(0.0, s * (DRUM_W / 2 + 0.02), 0.0), axis="y", mat="hub",
              collision=False) for s in sides],
        *[box(0.05, DRUM_W - 0.04, 0.012, at=((DRUM_R - 0.004) * math.sin(a), 0.0,
              (DRUM_R - 0.004) * math.cos(a)), rpy=(0.0, a, 0.0), mat="dark", collision=False)
          for a in (0.0, math.pi)],                               # weld seams (show spin)
    ])

    # ------------------------------------------------------------------ leveling blade
    blade = Link("blade", 520.0, _blade_shapes())

    # --------------------------------------------------------- blade lift cylinders
    blade_j = Joint("blade_joint", "revolute", "front_frame", "blade",
                    xyz=(PIVOT[0], 0.0, PIVOT[1]), axis=(0, -1, 0), lower=BLADE_LO,
                    upper=BLADE_HI, init=BLADE_HI, velocity=0.16, role="tool")
    cf = _cylinder_fit()
    if cf["err"] >= 0.02:
        raise ValueError(f"lift-cylinder follower misses its lug by {cf['err']:.3f} m")
    d0 = cf["d0"]
    lb = cf["d_min"] - 0.08                                       # barrel length
    lr = min(cf["d_min"] - 0.04, cf["d_max"] - lb + 0.12)         # rod length (stays inside)
    barrels = Link("blade_cylinders", 70.0, [shp for s in sides for shp in (
        cyl(0.055, lb, at=(lb / 2, s * CYL_Y, 0.0), axis="x", mat="dark", collision=False),
        cyl(0.065, 0.05, at=(lb - 0.025, s * CYL_Y, 0.0), axis="x", mat="steel", collision=False),
        cyl(0.055, 0.10, at=(0.0, s * CYL_Y, 0.0), axis="y", mat="dark", collision=False),
    )])
    rods = Link("blade_cylinder_rods", 30.0, [shp for s in sides for shp in (
        cyl(0.035, lr, at=(d0 - lr / 2, s * CYL_Y, 0.0), axis="x", mat="steel", collision=False),
        cyl(0.05, 0.10, at=(d0, s * CYL_Y, 0.0), axis="y", mat="steel", collision=False),
    )])
    s_lo, s_hi = sorted((cf["ks"] * BLADE_LO + cf["cs"], cf["ks"] * BLADE_HI + cf["cs"]))

    # ------------------------------------------------------------------ rear tractor
    rear = Link("rear_frame", 5000.0, [
        # hitch knuckle between the clevis plates + oscillation bearing
        cyl(0.24, 0.40, at=(0.0, 0.0, agl(0.86)), mat="dark"),
        box(0.62, 0.56, 0.40, at=(-0.31, 0.0, agl(0.86)), mat="dark"),
        cyl(0.30, 0.12, at=(-0.62, 0.0, agl(0.86)), axis="x", mat="dark"),
        # main frame between the tyres, rear axle
        box(2.72, 0.94, DECK - 0.07 - 0.52, at=(-1.98, 0.0, agl((DECK - 0.07 + 0.52) / 2))),
        cyl(0.14, GAUGE - TYRE_W - 0.02, at=(REAR_X, 0.0, agl(WHEEL_R)), axis="y", mat="dark"),
        # operator deck (also the fenders over the tyres)
        box(2.06, 2.26, 0.07, at=(-1.33, 0.0, agl(DECK - 0.035))),
        box(2.08, 2.28, 0.03, at=(-1.33, 0.0, agl(DECK - 0.085)), mat="accent", collision=False),
        # access ladders (both sides, ahead of the tyres) and grab rails
        *[box(0.04, 0.04, 1.30, at=(x, s * 1.02, agl(0.98)), mat="accent", collision=False)
          for x in (-0.60, -0.34) for s in sides],
        *[box(0.30, 0.18, 0.03, at=(-0.47, s * 1.02, agl(0.45 + 0.30 * i)), mat="accent",
              collision=False) for i in range(4) for s in sides],
        *[box(0.04, 0.04, 0.80, at=(-0.40, s * 1.10, agl(DECK + 0.40)), mat="accent",
              collision=False) for s in sides],
        # cab
        *_cab(CAB_X0, CAB_X1, CAB_HW, agl(DECK), agl(ROOF)),
        # engine hood behind the cab, sloping cap, side louvres
        box(CAB_X0 - HOOD_X0, 1.68, 2.22 - DECK, at=((CAB_X0 + HOOD_X0) / 2, 0.0,
            agl((2.22 + DECK) / 2))),
        beam((CAB_X0, agl(2.25)), (HOOD_X0, agl(2.16)), 1.62, 0.10),
        *[box(0.80, 0.02, 0.03, at=(-2.75, s * 0.845, agl(1.80 + 0.07 * i)), mat="dark",
              collision=False) for i in range(5) for s in sides],
        cyl(0.10, 0.22, at=(-2.25, 0.55, agl(2.36)), mat="accent", collision=False),  # pre-cleaner
        cyl(0.11, 0.05, at=(-2.25, 0.55, agl(2.49)), mat="dark", collision=False),
        cyl(0.06, 0.32, at=(-3.05, -0.55, agl(2.36)), mat="accent", collision=False),  # exhaust
        # rear body behind the tyres: radiator grille, tail lights, bumper
        box(1.04, 2.00, DECK - 0.56, at=(-2.88, 0.0, agl((DECK + 0.56) / 2))),
        box(0.03, 1.30, 0.45, at=(HOOD_X0 - 0.015, 0.0, agl(1.95)), mat="dark", collision=False),
        *[box(0.02, 1.26, 0.03, at=(HOOD_X0 - 0.035, 0.0, agl(1.78 + 0.07 * i)), mat="accent",
              collision=False) for i in range(6)],
        box(0.03, 1.50, 0.55, at=(-3.415, 0.0, agl(1.22)), mat="dark", collision=False),
        *[box(0.03, 0.16, 0.12, at=(-3.42, s * 0.85, agl(1.40)), mat="red", collision=False)
          for s in sides],
        *[box(0.03, 0.16, 0.08, at=(-3.42, s * 0.85, agl(1.26)), mat="orange", collision=False)
          for s in sides],
        box(0.10, 2.00, 0.24, at=(-3.43, 0.0, agl(0.70)), mat="dark"),
    ])

    # ------------------------------------------------------------------------ joints
    joints = [
        Joint("articulation_joint", "revolute", "front_frame", "rear_frame",
              xyz=(0.0, 0.0, 0.0), axis=(0, 0, 1), lower=-ARTIC, upper=ARTIC,
              velocity=0.30, role="articulation"),
        wheel_joint("drum_joint", "front_frame", "drum", (DRUM_X, 0.0, 0.0), DRUM_R),
        blade_j,
        _cyl_follower("blade_cylinder_joint", "front_frame", "blade_cylinders", CYL_BASE,
                      blade_j, cf["ka"], cf["ca"]),
        Joint("blade_cylinder_rod_joint", "prismatic", "blade_cylinders", "blade_cylinder_rods",
              axis=(1, 0, 0), lower=s_lo - 0.01, upper=s_hi + 0.01,
              init=cf["ks"] * BLADE_HI + cf["cs"], velocity=abs(cf["ks"]) * blade_j.velocity + 0.1,
              role="follower", follows=(blade_j.name, cf["ks"], cf["cs"])),
        *[wheel_joint(f"rear_{n}_wheel_joint", "rear_frame", f"rear_{n}_wheel",
                      (REAR_X, s * y_w, agl(WHEEL_R)), WHEEL_R)
          for n, s in (("left", 1.0), ("right", -1.0))],
    ]
    links = [front, drum, blade, barrels, rods, rear,
             _rear_wheel("rear_left_wheel", 450.0), _rear_wheel("rear_right_wheel", 450.0)]

    return MachineSpec(
        name="soil_compactor",
        abbrev="SDC",
        display_name="Soil Compactor",
        category="compaction",
        reference="11-12 t single-drum vibratory soil compactor (CAT CS56B / Bomag BW 213 class)",
        description="Articulated single smooth-drum vibratory roller: 2.13 m × 1.53 m drum in a "
                    "front drum frame with an optional front leveling blade, centre hinge "
                    "(±34°), rear tractor with ROPS cab, engine hood and two 23.1-26 tyres.",
        root="front_frame",
        links=links,
        joints=joints,
        drive=Drive(type="articulated", max_speed=3.1, articulation_joint="articulation_joint",
                    front_length=DRUM_X, rear_length=-REAR_X, articulation_rate=0.30),
        tools=[Tool("blade_joint", "leveling_blade", positive="raise", negative="lower")],
        cameras=[
            # Operator eye height, just ahead of the (opaque) windscreen, looking over the
            # drum frame and drum crown at the ground ahead (driving view).  The blade
            # sits below the sight line over the drum, as on the real machine.
            Camera("front", "rear_frame", (CAB_X1 + 0.06, 0.0, agl(2.84)), (0.0, 0.42, 0.0)),
            # Blade cameras on the drum-frame corner posts, looking down and slightly in
            # at the back of the moldboard and the blade end on the ground through the
            # whole stroke; the ground ahead of and beside the blade fills the rest.  On
            # the front frame, so articulation does not change the view.
            Camera("left", "front_frame", (BLADE_CAM[0], BLADE_CAM[1], agl(BLADE_CAM[2])),
                   (0.0, BLADE_CAM_PITCH, -BLADE_CAM_YAW)),
            Camera("right", "front_frame", (BLADE_CAM[0], -BLADE_CAM[1], agl(BLADE_CAM[2])),
                   (0.0, BLADE_CAM_PITCH, BLADE_CAM_YAW)),
        ],
        envelope=(6.30, 2.50, 3.09),
        operating_mass=12000.0,
        notes="Drum and rear tyres are cosmetic 'wheel' joints; vibration and the ±15° "
              "oscillation joint are not modelled. The blade lift cylinders are visual "
              "followers of blade_joint. Envelope L/W = datasheet 5.83 x 2.30 m plus the "
              "assumed blade (+0.48 m, 2.50 m wide), so the L/W check only confirms that "
              "assumption. left/right are blade cameras on the drum frame; front is the "
              "operator view, which loses the blade once it is lowered past about a third "
              "of its stroke.",
    )
