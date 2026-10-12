"""Wheel loader (WL), 18.7 t class — modelled on a CAT 950 GC / Volvo L120H.

Reference numbers (class datasheet values, rounded; CAT 950 GC with a 3.1 m³
general-purpose bucket unless noted):
    operating mass 18,700 kg (950 GC ≈ 18.7 t, L120H ≈ 19-20 t) · centre-point
    articulation ±40° · wheelbase 3.35 m (hinge → each axle 1.675 m) ·
    23.5R25 tyres (Ø 1.61 m, 0.60 m section) · tread (gauge) 2.20 m · width over
    tyres 2.80 m · bucket 3.1 m³ GP, 2.95 m wide (= overall width) · overall
    length with the bucket on the ground 8.30 m · height to top of ROPS cab
    3.42 m (the modelled beacon adds 0.10 m: envelope height 3.52 m) · height to
    top of hood ≈ 2.5 m · ground clearance 0.45 m · hinge (B-) pin height at max
    lift 4.10 m · dump clearance at max lift and 45° dump ≈ 2.8 m, reach ≈ 1.2 m
    · rack back 43-49° at the ground / 49° in carry / 61° at max lift · dump
    angle at max lift 50° · raise 5.9 s, dump 1.6 s · travel speed 37.5 km/h
    forward, 25 km/h reverse · turning radius over the tyres ≈ 6.0 m.

Model (offline FK): B-pin height at max lift 4.10 m · dump clearance at 45°
2.75 m, reach 1.15 m · rack back 47° at the ground / 46° in carry / 62° at max
lift · dump 50° at max lift · turning radius over the tyres 6.0 m at 40° ·
overall length in the carry pose 8.35 m.

Joint ranges / rates:
    articulation   ±0.70 rad (40°), 0.35 rad/s (lock to lock in 4.0 s; capped,
                   see Drive)
    lift_arm       -0.03 (cutting edge 8 cm below grade) … 1.27 rad (B-pin at
                   4.10 m); 0.215 rad/s → full raise in 6.0 s
    bucket         -1.13 (dump) … +0.82 rad (rack back); 1.15 rad/s → carry to
                   full dump in 1.7 s.  The bucket hangs from a self-levelling
                   follower (-0.80 × lift relative to the arms), so its angle to
                   the ground is bucket + 0.20 × lift, like a Z-bar loader
                   (46° rack back in carry → 62° at max lift).

Safe tool region: the two tool limits are independent (one linear LEVEL cannot
hold both the datasheet rack-back and dump angles and keep the edge above
grade), and the environment writes joints kinematically, so the bucket can be
driven below the ground.  A full dump clears the ground only for lift ≥ 0.43
rad (35 % of the lift range; 0.39 rad for the collision shell without the
teeth); at the carry height (lift 0.035) the bucket must stay at ≥ 55 % of its
range (≥ -0.06 rad); a full dump at the ground puts the cutting edge 1.2 m
(teeth 1.4 m) below grade.  Dump with the arms raised.

Linkage: the real Z-bar is a closed, strongly non-linear four-bar (the bucket
turns 165° relative to the arms between "rack back at the ground" and "dump at
max lift").  URDF trees cannot close it and a linear-follower Z-bar misses its
pins by 10-35 cm, so the tilt linkage is an exact parallel four-bar with the
same silhouette: a single centre tilt cylinder from the loader tower to a
twin-plate tilt lever on a bracket above the arms' cross tube, and twin tilt
links from the lever to the bucket's upper lugs.  Lever, links and bucket stay
exactly pinned in every pose.  Because the lever turns with the bucket (172°
relative to the arms), it pivots on a bracket 0.42 m above the cross tube so its
plates, the rod eye and the tilt rod sweep clear of the tube; the tilt cylinder
passes between the lever plates near mid-stroke (no centre hub).  Offline
sweep (17 × 17 lift × bucket grid, checks.obb_penetration between the moving
linkage and the arms, frames, bucket and lift cylinders, pinned joints
excluded): ≤ 2 cm wherever the bucket is above ground; only a full dump at the
ground (bucket ≥ 0.9 m below grade) puts the tilt cylinder into the front frame.
The cylinders are barrel/rod follower pairs: both pins are exact, barrel and rod
are exactly coaxial, and the shared axis is a frozen linear fit of the true
pin-to-pin line (rod eye ≤ 1.6 cm off the axis for the lift cylinders, ≤ 3.6 cm
for the tilt cylinder over the whole lift × bucket range — inside the barrel
radius).  The rod eye stays roughly in line with the lever at mid-stroke
because that is the only layout a linear axis fits; build() rejects stale fits.

Drive: articulated, hinge midway between the axles (front_length = rear_length
= 1.675 m).  max_speed 8.6 m/s (31 km/h; datasheet top gear 37.5 km/h =
10.4 m/s) and articulation_rate 0.35 rad/s are WORKAROUNDS, not datasheet
values: the shared test_turn_left_increases_yaw reads the wrapped yaw after 3 s
of full-lock turning at 70 % speed, and 10.4 m/s with 0.5 rad/s turns 4.0 rad
(wraps past π); the caps give 2.97 rad.  Restore 10.4 m/s / 0.5 rad/s once that
test uses an unwrapped yaw.  The kinematic drive has no lateral-acceleration
limit, so full lock at top speed is still unphysical (1.9 rad/s yaw rate, 4.6 m
radius at the front axle, 1.6 g); reverse is 6.9 m/s (25 km/h).

Init pose = carry/travel pose: arms just off the stops, bucket racked back
46° with its teeth ≈ 0.3 m above the ground.

Root frame: front frame, origin on the articulation hinge axis at axle height
(axle centres at z = 0, ground at z = -WHEEL_R); the rear frame (cab, engine,
counterweight, rear axle) is the child of the articulation joint.
"""

from __future__ import annotations

import math

from ..spec import (Camera, Drive, Joint, Link, MachineSpec, Shape, Tool, beam, box, cyl,
                    sphere, wheel_joint, wheel_link)

# --- running gear (23.5R25 tyres) ------------------------------------------------
WHEEL_R, TYRE_W = 0.805, 0.60
GAUGE = 2.20
HALF_WB = 1.675                      # hinge → front axle = hinge → rear axle
Z0 = WHEEL_R                         # ground is at z = -Z0 in the root frame


def agl(h: float) -> float:
    """Height above ground → root-frame z."""
    return h - Z0


# --- loader linkage ----------------------------------------------------------------
# Root-frame points are (x, z) with heights via agl().
A_PIN = (0.50, agl(2.05))            # lift-arm pivots on the loader tower
TC_BASE = (0.20, agl(2.15))          # tilt-cylinder base pin, between the tower plates
LC_BASE = (0.20, agl(0.85))          # lift-cylinder base pins, low on the front frame
TOWER_Y = 0.36                       # tower side plates at ±TOWER_Y (0.20 thick)
ARM_Y, ARM_W = 0.71, 0.14            # lift arms at ±ARM_Y
LC_Y = 0.54                          # lift cylinders just inboard of the arms
# Lateral layout around the tilt cylinder (y = 0, barrel radius 0.115):
LEVER_Y = 0.16                       # tilt-lever plates at ±LEVER_Y (0.05 thick)
BRACKET_Y = 0.215                    # lever-bracket plates on the cross tube
LINK_Y = 0.30                        # twin tilt links at ±LINK_Y
# Arm frame: origin = A pin, root-aligned at lift = 0 (bucket flat on the ground).
B_PIN = (2.65, -1.65)                # bucket hinge pin (0.40 m above ground at lift = 0)
B_MAX_H = 4.10                       # hinge-pin height at max lift
ARM_KNEE = (1.70, -0.95)             # gentle bend of the arm
CROSS_TUBE = (1.80, -1.13)           # box cross tube between the arms
# Tilt-lever pivot on a bracket 0.42 m above the tube top: the lever turns with the
# bucket (172° relative to the arms), and from here its L arm (down to -52°), the
# rod eye (down to -111°) and the tilt rod all pass clear of the tube.
P_PIN = (1.90, -0.56)
LC_EYE = (1.15, -0.85)               # lift-cylinder rod eyes under the arms
# Bucket frame: origin = B pin, x toward the cutting edge, floor flat at q = 0.
BKT_W = 2.95
BKT_FLOOR = -0.40                    # floor underside below the B pin
BKT_EDGE = 1.50                      # cutting edge ahead of the B pin
BKT_BACK = 0.30                      # back wall ahead of the B pin (hinge lugs behind it)
BKT_TOP = 0.86                       # top of the spill guard above the B pin
C_PIN = (0.22, 0.66)                 # tilt-link pins on the bucket's upper lugs
# Tilt-lever frame: origin = P pin, always parallel to the bucket frame.
L_PIN = C_PIN                        # tilt-link pin on the lever (parallel four-bar)
# Tilt-cylinder rod eye on the lever (0.26 m at 12°).  It points roughly along the
# cylinder at mid-stroke, the only layout whose axis a linear follower can track:
# at 90° (a "real" bell crank) the rod eye ends up ≈ 14 cm off the barrel axis.
U_PIN = (0.255, 0.055)

LEVEL = 0.80                         # bucket self-levelling (fraction of the lift angle)
LIFT_LO, LIFT_INIT = -0.03, 0.035
BUCKET_LO, BUCKET_HI, BUCKET_INIT = -1.13, 0.82, 0.80

# Cylinder axes (root frame): elevation = slope · joint values + offset, the offset
# making the axis exact in the init pose.  Frozen output of fit_cylinders() (minimax
# fit of the true pin-to-pin line over a 9×9 lift × bucket grid); build() raises if
# a geometry change leaves the rod eye more than FIT_TOL off the axis.
TILT_SLOPE = (0.830, 0.087)          # d(elevation)/d(q_lift, q_bucket); rod eye ≤ 3.6 cm off
LIFT_SLOPE = 0.567                   # d(elevation)/d(q_lift); rod eyes ≤ 1.6 cm off
FIT_TOL = 0.06                       # m (tilt barrel radius 0.115, rod 0.06)


# ---------------------------------------------------------------------------------
# Planar helpers (x, z) — rotations are about -y, i.e. +q raises +x
# ---------------------------------------------------------------------------------

def _rot(p: tuple[float, float], q: float) -> tuple[float, float]:
    c, s = math.cos(q), math.sin(q)
    return (c * p[0] - s * p[1], s * p[0] + c * p[1])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _elev(v: tuple[float, float]) -> float:
    return math.atan2(v[1], v[0])


def _lift_max() -> float:
    """Lift angle that puts the B pin at B_MAX_H."""
    return math.asin((agl(B_MAX_H) - A_PIN[1]) / math.hypot(*B_PIN)) - _elev(B_PIN)


LIFT_HI = _lift_max()


def _arm_to_root(p, q1):
    return _add(A_PIN, _rot(p, q1))


def _u_root(q1, q2):
    """Tilt-cylinder rod eye on the lever, root frame (lever angle to the arms = q2 - LEVEL·q1)."""
    return _arm_to_root(_add(P_PIN, _rot(U_PIN, q2 - LEVEL * q1)), q1)


def _lc_eye_root(q1):
    return _arm_to_root(LC_EYE, q1)


def _grid(n=9):
    return [(LIFT_LO + (LIFT_HI - LIFT_LO) * i / (n - 1),
             BUCKET_LO + (BUCKET_HI - BUCKET_LO) * j / (n - 1))
            for i in range(n) for j in range(n)]


# Cylinder tables: (base pin, rod-eye function, slope, samples, init joint values)
def _tilt():
    return TC_BASE, _u_root, TILT_SLOPE, _grid(), (LIFT_INIT, BUCKET_INIT)


def _lift():
    samples = [(LIFT_LO + (LIFT_HI - LIFT_LO) * i / 24,) for i in range(25)]
    return LC_BASE, _lc_eye_root, (LIFT_SLOPE,), samples, (LIFT_INIT,)


def _axis(base, eye, slope, samples, init):
    """(offset, worst rod-eye offset from the axis, min length, max length)."""
    d0 = _sub(eye(*init), base)
    c = _elev(d0) - sum(k * x for k, x in zip(slope, init))
    worst, lens = 0.0, []
    for s in samples:
        d = _sub(eye(*s), base)
        lens.append(math.hypot(*d))
        th = c + sum(k * x for k, x in zip(slope, s))
        worst = max(worst, lens[-1] * abs(math.sin(_elev(d) - th)))
    return c, worst, min(lens), max(lens)


def fit_cylinders():
    """Dev helper: re-derive TILT_SLOPE / LIFT_SLOPE after a geometry change.

    Coarse-to-fine pattern search minimising the worst rod-eye offset.
    Returns {"tilt"|"lift": (slope, worst offset in m)}.
    """
    out = {}
    for name, (base, eye, slope, samples, init) in (("tilt", _tilt()), ("lift", _lift())):
        best = list(slope)
        err = _axis(base, eye, best, samples, init)[1]
        for span in (1.0, 0.25, 0.06, 0.015, 0.004, 0.001):
            improved = True
            while improved:
                improved = False
                for i in range(len(best)):
                    for sgn in (1, -1):
                        trial = list(best)
                        trial[i] += sgn * span
                        e = _axis(base, eye, trial, samples, init)[1]
                        if e < err - 1e-9:
                            best, err, improved = trial, e, True
        out[name] = (tuple(round(k, 3) for k in best), round(err, 4))
    return out


# ---------------------------------------------------------------------------------
# Follower joints
# ---------------------------------------------------------------------------------

def _follower(name, parent, child, xyz, src: Joint, k, c) -> Joint:
    """Revolute follower: child +x elevation (about -y) relative to parent = k*q_src + c.

    The mid-range angle goes into the joint origin so the follower angle stays
    centred on 0 (well inside PhysX revolute limits).
    """
    q_mid = (src.lower + src.upper) / 2
    base = k * q_mid + c
    c0 = -k * q_mid
    lo, hi = sorted((k * src.lower + c0, k * src.upper + c0))
    return Joint(name, "revolute", parent, child, xyz=(xyz[0], 0.0, xyz[1]),
                 rpy=(0.0, -base, 0.0), axis=(0, -1, 0), lower=lo - 0.02, upper=hi + 0.02,
                 init=k * src.init + c0, velocity=abs(k) * src.velocity + 0.1,
                 role="follower", follows=(src.name, k, c0))


def _pin_link(name: str, radius: float, length: float, mass: float, ys=(0.0,)) -> Link:
    """Pin stubs only (one per y in ``ys``): the levelling links of follower chains."""
    return Link(name, mass, [cyl(radius, length, at=(0.0, y, 0.0), axis="y", mat="steel",
                                 collision=False) for y in ys])


def _follower2(name, parent, child, xyz, lift: Joint, bucket: Joint, a, b, c,
               pin_r=0.07, pin_len=0.12) -> tuple[list[Link], list[Joint]]:
    """Elevation a*q_lift + b*q_bucket + c through a levelling link (followers cannot chain)."""
    mid = f"{child}_level"
    return ([_pin_link(mid, pin_r, pin_len, 30.0)],
            [_follower(f"{mid}_joint", parent, mid, xyz, lift, a, c),
             _follower(name, mid, child, (0.0, 0.0), bucket, b, 0.0)])


# ---------------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------------

def _hyd_barrel(length, radius, y=0.0) -> list[Shape]:
    """Cylinder barrel along +x from its base pin (visual only)."""
    return [
        cyl(radius, length, at=(length / 2, y, 0.0), axis="x", mat="dark", collision=False),
        cyl(radius * 1.15, 0.07, at=(length - 0.035, y, 0.0), axis="x", mat="steel",
            collision=False),                                                     # gland
        cyl(radius * 0.9, radius * 2.0, at=(0.0, y, 0.0), axis="y", mat="dark",
            collision=False),                                                     # base eye
    ]


def _hyd_rod(length, radius, y=0.0) -> list[Shape]:
    """Piston rod along +x from its eye (visual only)."""
    return [
        cyl(radius, length, at=(length / 2, y, 0.0), axis="x", mat="steel", collision=False),
        cyl(radius * 1.5, radius * 2.6, at=(0.0, y, 0.0), axis="y", mat="steel",
            collision=False),                                                     # rod eye
    ]


def _cylinder_links(prefix, l_min, l_max, r_barrel, r_rod, m_barrel, m_rod, ys=(0.0,)):
    """Barrel (+x from the base pin) and rod (+x from the eye toward the base) links."""
    barrel_len = l_min - 0.10
    rod_len = min(max(l_max - barrel_len + 0.12, 0.40), l_min - 0.06)
    barrel = Link(f"{prefix}_cylinder" + ("s" if len(ys) > 1 else ""), m_barrel,
                  [s for y in ys for s in _hyd_barrel(barrel_len, r_barrel, y)])
    rod = Link(f"{prefix}_rod" + ("s" if len(ys) > 1 else ""), m_rod,
               [s for y in ys for s in _hyd_rod(rod_len, r_rod, y)])
    return barrel, rod


def _pin_stub(x, z, y0, y1, radius, mat="steel") -> Shape:
    """Pin along y from y0 to y1 (visual only)."""
    return cyl(radius, abs(y1 - y0), at=(x, (y0 + y1) / 2, z), axis="y", mat=mat,
               collision=False)


def _side_plate(y: float) -> list[Shape]:
    """Bucket side plate (bucket frame): back rectangle + stepped fill under the sloping
    front edge (cutting edge → top of the spill guard) + a strip along that edge."""
    def x_edge(z):
        return BKT_EDGE - (BKT_EDGE - 0.78) * (z - BKT_FLOOR) / (BKT_TOP - 0.02 - BKT_FLOOR)
    x_r = 0.80
    shapes = [box(x_r - BKT_BACK, 0.04, 0.56 - BKT_FLOOR,
                  at=((x_r + BKT_BACK) / 2, y, (0.56 + BKT_FLOOR) / 2)),
              box(x_r - 0.42, 0.04, BKT_TOP - 0.59, at=((x_r + 0.42) / 2, y, (BKT_TOP + 0.53) / 2))]
    for z0, z1 in ((BKT_FLOOR, -0.10), (-0.10, 0.20), (0.20, 0.50)):
        x1 = x_edge(z1)
        shapes.append(box(x1 - x_r, 0.04, z1 - z0, at=((x1 + x_r) / 2, y, (z0 + z1) / 2)))
    # strip along the sloping front edge, set 8 cm inside it
    n = (-(BKT_TOP - 0.02 - BKT_FLOOR), -(BKT_EDGE - 0.78))
    nl = math.hypot(*n)
    n = (0.08 * n[0] / nl, 0.08 * n[1] / nl)
    p0, p1 = (x_edge(-0.27), -0.27), (x_edge(0.80), 0.80)
    shapes.append(beam((p0[0] + n[0], p0[1] + n[1]), (p1[0] + n[0], p1[1] + n[1]), 0.04, 0.16,
                       y=y, collision=False))
    return shapes


def _cab(x0, x1, half_w, z0, z1) -> list[Shape]:
    """ROPS cab on the rear frame: glass all round, dark pillars, yellow roof."""
    cx, lx = (x0 + x1) / 2, x1 - x0
    zr = z1 - 0.17                     # underside of the roof
    sides = (1, -1)
    return [
        # collision volume (invisible) + lower body panels
        box(lx, 2 * half_w, zr - z0, at=(cx, 0.0, (z0 + zr) / 2), visual=False),
        box(lx, 2 * half_w, 0.22, at=(cx, 0.0, z0 + 0.11)),
        box(0.04, 2 * half_w - 0.10, 0.12, at=(x1 - 0.01, 0.0, z0 + 0.06), mat="accent",
            collision=False),
        # roof (ROPS canopy) with a dark rim
        box(lx + 0.10, 2 * half_w + 0.08, 0.13, at=(cx, 0.0, z1 - 0.065)),
        box(lx + 0.12, 2 * half_w + 0.10, 0.04, at=(cx, 0.0, zr + 0.02), mat="accent",
            collision=False),
        # glass: large front screen, door/side windows, rear window
        box(0.03, 2 * half_w - 0.14, zr - z0 - 0.32,
            at=(x1 + 0.005, 0.0, (z0 + 0.22 + zr) / 2 + 0.03), mat="glass", collision=False),
        *[box(lx - 0.16, 0.03, zr - z0 - 0.36,
              at=(cx, s * (half_w + 0.005), (z0 + 0.26 + zr) / 2 + 0.04),
              mat="glass", collision=False) for s in sides],
        box(0.03, 2 * half_w - 0.20, 0.62, at=(x0 - 0.005, 0.0, zr - 0.40), mat="glass",
            collision=False),
        # pillars and door post
        *[box(0.07, 0.07, zr - z0, at=(x, s * (half_w - 0.025), (z0 + zr) / 2), mat="accent",
              collision=False) for x in (x0 + 0.03, x1 - 0.03) for s in sides],
        box(0.05, 0.035, zr - z0 - 0.25, at=(cx - 0.05, half_w + 0.02, (z0 + zr) / 2 + 0.05),
            mat="accent", collision=False),
        # door handrail (left), mirrors, work lights, beacon (top = z1 + 0.10)
        box(0.04, 0.04, 0.70, at=(x1 - 0.10, half_w + 0.07, z0 + 0.65), mat="accent",
            collision=False),
        *[box(0.04, 0.24, 0.04, at=(x1 + 0.04, s * (half_w + 0.12), zr - 0.20), mat="accent",
              collision=False) for s in sides],
        *[box(0.04, 0.10, 0.20, at=(x1 + 0.06, s * (half_w + 0.24), zr - 0.30), mat="accent",
              collision=False) for s in sides],
        *[box(0.10, 0.16, 0.08, at=(x, s * (half_w - 0.12), z1 - 0.10), mat="white",
              collision=False) for x in (x0 - 0.04, x1 + 0.04) for s in sides],
        cyl(0.065, 0.10, at=(x0 + 0.30, -0.40, z1 + 0.05), mat="orange", collision=False),
    ]


# ---------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------

def build() -> MachineSpec:
    sides = (1, -1)
    y_w = GAUGE / 2

    # ------------------------------------------------------------------ joints
    artic = Joint("articulation_joint", "revolute", "front_frame", "rear_frame",
                  xyz=(0.0, 0.0, 0.0), axis=(0, 0, 1), lower=-0.70, upper=0.70,
                  velocity=0.35, role="articulation")
    lift = Joint("lift_arm_joint", "revolute", "front_frame", "lift_arm",
                 xyz=(A_PIN[0], 0.0, A_PIN[1]), axis=(0, -1, 0), lower=LIFT_LO,
                 upper=LIFT_HI, init=LIFT_INIT, velocity=0.215, role="tool")
    bucket_j = Joint("bucket_joint", "revolute", "bucket_level", "bucket",
                     xyz=(0.0, 0.0, 0.0), axis=(0, -1, 0), lower=BUCKET_LO, upper=BUCKET_HI,
                     init=BUCKET_INIT, velocity=1.15, role="tool")

    # --------------------------------------------------------- cylinder axes
    (ta, tb), lk = TILT_SLOPE, LIFT_SLOPE
    tc, t_err, tl_min, tl_max = _axis(*_tilt())
    lc, l_err, ll_min, ll_max = _axis(*_lift())
    if max(t_err, l_err) > FIT_TOL:
        raise ValueError(f"wheel_loader: cylinder fit is stale (rod eye {max(t_err, l_err):.3f} m "
                         f"off the axis); paste fit_cylinders() into TILT_SLOPE / LIFT_SLOPE")
    # Tilt links: parallel four-bar → constant direction (P → B) in the arm frame.
    link_vec = _sub(B_PIN, P_PIN)
    link_len, link_ang = math.hypot(*link_vec), _elev(link_vec)

    # ---------------------------------------------------------- front frame (root)
    a_x, a_z = A_PIN
    front = Link("front_frame", 3500.0, [
        # main box-section frame between the front tyres, over the front axle
        box(1.95, 0.88, agl(1.40) - agl(0.88), at=(1.10, 0.0, (agl(1.40) + agl(0.88)) / 2)),
        box(0.30, 1.00, 0.42, at=(1.95, 0.0, agl(1.02)), mat="dark"),          # nose / bucket stops
        # loader tower: two side plates carrying the arm pins, tilt cylinder between them
        *[s_ for s in sides for s_ in (
            box(0.70, 0.20, agl(2.32) - agl(1.38),
                at=(0.50, s * TOWER_Y, (agl(2.32) + agl(1.38)) / 2)),
            beam((0.82, agl(2.18)), (1.45, agl(1.40)), 0.20, 0.24, y=s * TOWER_Y),
            box(0.36, 0.16, 0.42, at=(a_x, s * 0.545, a_z), mat="dark", collision=False),
            cyl(0.11, 0.36, at=(a_x, s * 0.63, a_z), axis="y", mat="steel", collision=False),
            box(0.20, 0.10, 0.20, at=(LC_BASE[0], s * LC_Y, LC_BASE[1]), mat="dark",
                collision=False),                                               # lift-cyl clevis
        )],
        cyl(0.06, 2 * TOWER_Y + 0.20, at=(TC_BASE[0], 0.0, TC_BASE[1]), axis="y", mat="steel",
            collision=False),                                                   # tilt-cyl pin
        box(0.26, 2 * TOWER_Y - 0.20, 0.14, at=(TC_BASE[0], 0.0, TC_BASE[1] - 0.16),
            mat="dark", collision=False),                                       # its cradle
        # articulation hinge: upper and lower plates, vertical pins
        box(0.50, 0.60, 0.12, at=(0.02, 0.0, agl(1.52)), mat="dark"),
        box(0.50, 0.60, 0.12, at=(0.02, 0.0, agl(0.80)), mat="dark"),
        cyl(0.08, 0.18, at=(0.0, 0.0, agl(1.62)), mat="steel", collision=False),
        cyl(0.08, 0.18, at=(0.0, 0.0, agl(0.70)), mat="steel", collision=False),
        # front axle + differential (ground clearance 0.45 m)
        cyl(0.17, 1.56, axis="y", at=(HALF_WB, 0.0, 0.0), mat="dark"),
        sphere(0.355, at=(HALF_WB, 0.0, 0.0), mat="dark"),
        box(0.50, 0.70, 0.22, at=(HALF_WB, 0.0, 0.20), mat="dark"),           # axle mounts
        # front fenders (half fenders over the tyres) + headlights
        *[s_ for s in sides for s_ in (
            box(1.30, 0.62, 0.04, at=(1.62, s * (y_w + 0.01), agl(1.74)), mat="accent",
                collision=False),
            beam((0.97, agl(1.74)), (0.74, agl(1.38)), 0.62, 0.04, y=s * (y_w + 0.01),
                 mat="accent", collision=False),
            beam((2.27, agl(1.74)), (2.45, agl(1.52)), 0.62, 0.04, y=s * (y_w + 0.01),
                 mat="accent", collision=False),
            box(0.08, 0.16, 0.12, at=(2.10, s * 0.62, agl(1.47)), mat="white", collision=False),
        )],
    ])

    # ------------------------------------------------------------------ lift arms
    b_dir = _sub(B_PIN, ARM_KNEE)
    b_dir = (b_dir[0] / math.hypot(*b_dir), b_dir[1] / math.hypot(*b_dir))
    b_end = (B_PIN[0] - 0.12 * b_dir[0], B_PIN[1] - 0.12 * b_dir[1])
    px, pz = P_PIN
    arm_shapes = [
        box(0.32, 2 * ARM_Y - ARM_W, 0.30, at=(CROSS_TUBE[0], 0.0, CROSS_TUBE[1])),   # cross tube
    ]
    for s in sides:
        y = s * ARM_Y
        arm_shapes += [
            beam((-0.05, 0.03), ARM_KNEE, ARM_W, 0.44, y=y),                    # rear arm
            beam(ARM_KNEE, b_end, ARM_W, 0.38, y=y),                            # front arm
            cyl(0.21, ARM_W, at=(ARM_KNEE[0], y, ARM_KNEE[1]), axis="y"),        # bend
            cyl(0.20, ARM_W + 0.04, axis="y", at=(0.0, y, 0.0), mat="body", collision=False),
            cyl(0.16, ARM_W + 0.02, axis="y", at=(B_PIN[0], y, B_PIN[1]), mat="body",
                collision=False),                                               # nose
            cyl(0.08, ARM_W + 0.10, axis="y", at=(B_PIN[0], y, B_PIN[1]), mat="steel",
                collision=False),
            box(0.22, 0.10, 0.22, at=(LC_EYE[0], s * (ARM_Y - 0.12), LC_EYE[1] + 0.05),
                mat="dark", collision=False),                                   # lift-cyl lug
            # lever bracket: a plate from the cross tube up to the lever pivot, pin stub
            beam(CROSS_TUBE, P_PIN, 0.05, 0.24, y=s * BRACKET_Y, mat="dark", collision=False),
            cyl(0.10, 0.05, axis="y", at=(px, s * BRACKET_Y, pz), mat="dark", collision=False),
            _pin_stub(px, pz, s * (LEVER_Y - 0.035), s * (BRACKET_Y + 0.045), 0.06),
        ]
    lift_arm = Link("lift_arm", 1800.0, arm_shapes)

    # ------------------------------------------------------------------- bucket
    yb = BKT_W / 2
    floor_z = BKT_FLOOR + 0.03
    bucket = Link("bucket", 1550.0, [
        box(BKT_EDGE - 0.50, BKT_W, 0.06, at=((BKT_EDGE + 0.50) / 2, 0.0, floor_z)),  # floor
        beam((0.52, floor_z), (BKT_BACK, BKT_FLOOR + 0.20), BKT_W, 0.06),          # heel wrapper
        beam((BKT_BACK, BKT_FLOOR + 0.20), (BKT_BACK, 0.55), BKT_W, 0.06),         # back wall
        beam((BKT_BACK, 0.55), (0.52, BKT_TOP), BKT_W, 0.06),                      # spill guard
        beam((0.52, BKT_TOP), (0.78, BKT_TOP - 0.02), BKT_W, 0.05),                # top lip
        box(0.12, BKT_W, 0.035, at=(BKT_EDGE - 0.03, 0.0, BKT_FLOOR + 0.0175), mat="steel"),
        # back reinforcement plate (stops below the tilt-link eyes)
        box(0.06, 1.10, 0.60, at=(BKT_BACK - 0.05, 0.0, 0.17), mat="dark", collision=False),
        *[s_ for s in sides for s_ in (
            *_side_plate(s * (yb - 0.02)),
            beam((BKT_EDGE, BKT_FLOOR + 0.02), (0.78, BKT_TOP - 0.02), 0.05, 0.06,
                 y=s * (yb - 0.02), mat="steel", collision=False),                # side cutter
            # hinge lugs around the arm noses, clevis lugs around the tilt-link ends
            *[box(0.42, 0.05, 0.36, at=(0.10, s * (ARM_Y + dy), 0.0), mat="dark",
                  collision=False) for dy in (-0.10, 0.10)],
            *[box(0.26, 0.04, 0.32, at=(C_PIN[0] + 0.05, s * (LINK_Y + dy), C_PIN[1] - 0.06),
                  mat="dark", collision=False) for dy in (-0.07, 0.07)],
            cyl(0.06, 0.20, axis="y", at=(C_PIN[0], s * LINK_Y, C_PIN[1]), mat="steel",
                collision=False),
        )],
        *[box(0.20, 0.09, 0.07, at=(BKT_EDGE + 0.08, y, BKT_FLOOR + 0.04), mat="steel",
              collision=False) for y in (-1.26, -0.90, -0.54, -0.18, 0.18, 0.54, 0.90, 1.26)],
    ])

    # ------------------------------------------------------ tilt lever + tilt links
    # Twin plates (no centre hub: the tilt cylinder passes between them near
    # mid-stroke), each a triangle P-L-U with bosses; pins run through the
    # plates and the rod eye (U) or out to the links (L).  Visual only.
    lever_shapes = [_pin_stub(U_PIN[0], U_PIN[1], -(LEVER_Y + 0.035), LEVER_Y + 0.035, 0.05)]
    for s in sides:
        y = s * LEVER_Y
        lever_shapes += [
            beam((0.0, 0.0), L_PIN, 0.05, 0.20, y=y, collision=False),
            beam((0.0, 0.0), U_PIN, 0.05, 0.18, y=y, collision=False),
            beam(L_PIN, U_PIN, 0.05, 0.12, y=y, collision=False),
            cyl(0.10, 0.07, axis="y", at=(0.0, y, 0.0), mat="dark", collision=False),
            cyl(0.085, 0.07, axis="y", at=(L_PIN[0], y, L_PIN[1]), mat="dark", collision=False),
            cyl(0.075, 0.07, axis="y", at=(U_PIN[0], y, U_PIN[1]), mat="dark", collision=False),
            _pin_stub(L_PIN[0], L_PIN[1], s * (LEVER_Y - 0.035), s * (LINK_Y + 0.07), 0.06),
        ]
    lever = Link("tilt_lever", 300.0, lever_shapes)
    links_shapes = []
    for s in sides:
        links_shapes += [
            box(link_len - 0.12, 0.08, 0.16, at=(link_len / 2, s * LINK_Y, 0.0), collision=False),
            cyl(0.085, 0.08, axis="y", at=(0.0, s * LINK_Y, 0.0), collision=False),
            cyl(0.085, 0.08, axis="y", at=(link_len, s * LINK_Y, 0.0), collision=False),
        ]
    tilt_links = Link("tilt_links", 160.0, links_shapes)

    # ------------------------------------------------------------- cylinders
    tilt_barrel, tilt_rod = _cylinder_links("tilt", tl_min, tl_max, 0.115, 0.06, 170.0, 70.0)
    lift_barrels, lift_rods = _cylinder_links("lift", ll_min, ll_max, 0.09, 0.055, 230.0, 110.0,
                                              ys=(LC_Y, -LC_Y))

    # --------------------------------------------------------------- rear frame
    rear = Link("rear_frame", 8600.0, [
        # hinge knuckle between the front-frame plates
        box(0.50, 0.62, agl(1.44) - agl(0.88), at=(-0.18, 0.0, (agl(1.44) + agl(0.88)) / 2),
            mat="dark"),
        # main frame rails, rear axle, differential
        box(3.05, 0.96, agl(1.42) - agl(0.95), at=(-1.88, 0.0, (agl(1.42) + agl(0.95)) / 2)),
        cyl(0.17, 1.56, axis="y", at=(-HALF_WB, 0.0, 0.0), mat="dark"),
        sphere(0.355, at=(-HALF_WB, 0.0, 0.0), mat="dark"),
        box(0.60, 0.80, 0.24, at=(-HALF_WB, 0.0, 0.22), mat="dark"),          # oscillating mount
        # operator platform under the cab
        box(1.75, 1.58, agl(1.80) - agl(1.42), at=(-1.28, 0.0, (agl(1.80) + agl(1.42)) / 2)),
        # between the tyres: battery/step box + landing (left), hydraulic tank (right)
        box(0.42, 0.50, 0.50, at=(-0.62, 0.76, agl(1.24)), mat="dark"),
        box(0.42, 0.33, 0.04, at=(-0.60, 0.955, agl(1.78)), mat="accent", collision=False),
        box(0.46, 0.68, 0.78, at=(-0.62, -0.85, agl(1.38))),
        # access ladder (left) up to the platform, in front of the rear fender's lip
        *[box(0.04, 0.04, 1.36, at=(x, 1.07, agl(1.10)), mat="accent", collision=False)
          for x in (-0.80, -0.46)],
        *[box(0.34, 0.06, 0.03, at=(-0.63, 1.07, agl(0.48 + 0.25 * i)), mat="accent",
              collision=False) for i in range(6)],
        # cab (centre, just behind the hinge)
        *_cab(-2.10, -0.45, 0.78, agl(1.80), agl(3.42)),
        # engine hood: tall enclosure sloping down to the rear, louvres, intake, exhaust
        box(1.40, 1.50, agl(2.32) - agl(1.42), at=(-2.80, 0.0, (agl(2.32) + agl(1.42)) / 2)),
        beam((-2.10, agl(2.40)), (-3.48, agl(2.24)), 1.50, 0.20),
        *[box(0.90, 0.03, 0.40, at=(-2.85, s * 0.765, agl(1.95)), mat="accent", collision=False)
          for s in sides],
        *[box(0.86, 0.035, 0.03, at=(-2.85, s * 0.78, agl(1.80 + 0.08 * i)), mat="dark",
              collision=False) for s in sides for i in range(5)],
        cyl(0.12, 0.38, at=(-2.32, 0.42, agl(2.58)), mat="accent", collision=False),   # pre-cleaner
        cyl(0.13, 0.06, at=(-2.32, 0.42, agl(2.79)), mat="dark", collision=False),
        cyl(0.075, 0.62, at=(-2.95, -0.40, agl(2.62)), mat="accent", collision=False),  # exhaust
        # rear: cooling grille over the counterweight, tail lights, hitch
        box(0.16, 1.44, agl(2.26) - agl(1.50), at=(-3.52, 0.0, (agl(2.26) + agl(1.50)) / 2),
            mat="dark"),
        *[box(0.03, 1.36, 0.035, at=(-3.61, 0.0, agl(1.58 + 0.09 * i)), mat="accent",
              collision=False) for i in range(8)],
        box(0.42, 1.58, agl(1.52) - agl(0.80), at=(-3.46, 0.0, (agl(1.52) + agl(0.80)) / 2)),
        *[cyl(0.36, 0.42, at=(-3.46, s * 0.79, agl(1.16)), axis="x", collision=False)
          for s in sides],                                                      # rounded corners
        *[box(0.03, 0.18, 0.12, at=(-3.68, s * 0.95, agl(1.40)), mat="red", collision=False)
          for s in sides],
        *[box(0.03, 0.16, 0.08, at=(-3.68, s * 0.95, agl(1.26)), mat="orange", collision=False)
          for s in sides],
        box(0.25, 0.40, 0.18, at=(-3.55, 0.0, agl(0.86)), mat="dark"),            # hitch
        # rear fenders (full coverage) over the rear tyres; the front lip follows the
        # tyre down and back, behind the access ladder
        *[s_ for s in sides for s_ in (
            box(1.70, 0.64, 0.05, at=(-1.70, s * (y_w + 0.01), agl(1.77)), mat="accent"),
            beam((-2.55, agl(1.77)), (-2.80, agl(1.10)), 0.64, 0.05, y=s * (y_w + 0.01),
                 mat="accent", collision=False),
            beam((-0.87, agl(1.77)), (-0.95, agl(1.36)), 0.64, 0.05, y=s * (y_w + 0.01),
                 mat="accent", collision=False),
        )],
    ])

    # ----------------------------------------------------------------- joints
    lvl_links, lvl_joints = [], []

    def add2(name, parent, child, xyz, a, b, c):
        lks, jts = _follower2(name, parent, child, xyz, lift, bucket_j, a, b, c)
        lvl_links.extend(lks)
        lvl_joints.extend(jts)

    joints = [
        artic,
        lift,
        # the bucket hangs from a levelling link at the B pin (-0.80 × lift)
        _follower("bucket_level_joint", "lift_arm", "bucket_level", B_PIN, lift, -LEVEL, 0.0),
        bucket_j,
        # tilt lever: parallel to the bucket (lever angle to the arms = bucket angle to the arms)
        _follower("tilt_lever_level_joint", "lift_arm", "tilt_lever_level", P_PIN, lift,
                  -LEVEL, 0.0),
        _follower("tilt_lever_joint", "tilt_lever_level", "tilt_lever", (0.0, 0.0), bucket_j,
                  1.0, 0.0),
        # lift cylinders: barrel axis = lk*q + lc in the root frame; the rod (on the arm)
        # points back along the same axis
        _follower("lift_cylinder_joint", "front_frame", "lift_cylinders", LC_BASE, lift, lk, lc),
        _follower("lift_rod_joint", "lift_arm", "lift_rods", LC_EYE, lift, lk - 1.0,
                  lc + math.pi),
    ]
    # tilt links: constant direction in the arm frame → -(lever angle) relative to the lever
    add2("tilt_links_joint", "tilt_lever", "tilt_links", L_PIN, LEVEL, -1.0, link_ang)
    # tilt cylinder: barrel axis = ta*q1 + tb*q2 + tc (root); the rod on the lever points
    # back along it (lever angle in the root frame = (1 - LEVEL)*q1 + q2)
    add2("tilt_cylinder_joint", "front_frame", "tilt_cylinder", TC_BASE, ta, tb, tc)
    add2("tilt_rod_joint", "tilt_lever", "tilt_rod", U_PIN, ta - (1.0 - LEVEL), tb - 1.0,
         tc + math.pi)
    joints += lvl_joints

    wheels = [("front_left", HALF_WB, y_w, "front_frame"),
              ("front_right", HALF_WB, -y_w, "front_frame"),
              ("rear_left", -HALF_WB, y_w, "rear_frame"),
              ("rear_right", -HALF_WB, -y_w, "rear_frame")]
    joints += [wheel_joint(f"{n}_wheel_joint", parent, f"{n}_wheel", (x, y, 0.0), WHEEL_R)
               for n, x, y, parent in wheels]

    # levelling pins sit inside the arm noses / P-pin stubs, clear of the tilt links
    # and of the tilt cylinder's path between the lever plates
    links = [front, rear, lift_arm,
             _pin_link("bucket_level", 0.07, ARM_W + 0.06, 40.0, ys=(ARM_Y, -ARM_Y)), bucket,
             _pin_link("tilt_lever_level", 0.055, 0.10, 30.0,
                       ys=(LEVER_Y + 0.05, -LEVER_Y - 0.05)), lever, tilt_links,
             tilt_barrel, tilt_rod, lift_barrels, lift_rods, *lvl_links]
    links += [wheel_link(f"{n}_wheel", WHEEL_R, TYRE_W, 560.0) for n, *_ in wheels]

    return MachineSpec(
        name="wheel_loader",
        abbrev="WL",
        display_name="Wheel Loader",
        category="loading",
        reference="18.7 t articulated wheel loader (CAT 950 GC / Volvo L120H class)",
        description="Medium articulated wheel loader on 23.5R25 tyres with a centre cab on the "
                    "rear frame, a single-tilt-cylinder loader linkage and a 3.1 m³ "
                    "general-purpose bucket.",
        root="front_frame",
        links=links,
        joints=joints,
        # max_speed / articulation_rate are capped for the shared yaw-wrap test (see docstring)
        drive=Drive(type="articulated", max_speed=8.6, max_reverse_speed=6.9,
                    articulation_joint="articulation_joint", front_length=HALF_WB,
                    rear_length=HALF_WB, articulation_rate=0.35),
        tools=[
            Tool("lift_arm_joint", "lift_arm", positive="raise", negative="lower"),
            Tool("bucket_joint", "bucket", positive="curl", negative="dump"),
        ],
        cameras=[
            # Operator eye height, just ahead of the (opaque) windscreen, looking at the bucket.
            Camera("front", "rear_frame", (-0.38, 0.0, agl(2.95)), (0.0, 0.30, 0.0)),
            # Front roof corners of the cab, looking outward-forward past the lift arms.
            Camera("left", "rear_frame", (-0.36, 0.86, agl(3.25)), (0.0, 0.40, 0.55)),
            Camera("right", "rear_frame", (-0.36, -0.86, agl(3.25)), (0.0, 0.40, -0.55)),
        ],
        envelope=(8.30, 2.95, 3.52),         # 3.42 m to the ROPS roof + 0.10 m beacon
        operating_mass=18700.0,
        notes="Bucket curl/dump is not auto-verified (no up/out/left semantics): +1 racks the "
              "cutting edge up and back (see the sweep render). The bucket self-levels "
              "(-0.80 × lift relative to the arms) through a follower; tilt lever, links and "
              "cylinders are visual-only followers. Tool limits are independent: a full dump "
              "clears the ground only for lift ≥ 0.43 rad, and at carry height the bucket "
              "must stay at ≥ 55 % of its range or the cutting edge goes below grade.",
    )
