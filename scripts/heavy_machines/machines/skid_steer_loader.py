"""Skid steer loader (SSL), 3.5 t class — modelled on a Bobcat S650 / CAT 262D3.

Reference numbers (class datasheet values, rounded):
    operating mass 3,600 kg (S650 ≈ 3,770 kg, 262D3 ≈ 3,500 kg) · rated
    operating capacity ≈ 1,100 kg · 4 × 12-16.5 tyres (Ø 0.84 m, 0.31 m section,
    16.5 in rim) · wheelbase 1.16 m · tread (gauge) 1.45 m · width over tyres
    1.76 m · 72 in bucket: width 1.83 m, ≈ 0.40 m³ heaped · length with bucket
    on the ground 3.45 m · length without attachment ≈ 2.8 m · height to top of
    cab 2.06 m · ground clearance 0.20 m · arm pivot (rear towers) ≈ 1.54 m ·
    hinge-pin height at max lift 3.13 m · dump height ≈ 2.45 m · dump reach
    ≈ 0.6 m · dump angle at max height 42° · rollback 30° at the ground, 97° at
    full height · raise ≈ 3.3 s, dump ≈ 2 s · travel speed 11.9 km/h (3.3 m/s,
    same in reverse) · counter-rotation (spin turn) 1.0–1.2 rad/s; the model
    uses 1.0 rad/s (360° in 6.3 s).

Model numbers: length 3.48 m with the bucket flat on the ground (3.55 m in the
carry pose) · cutting edge 2.49 m high and 0.61 m ahead of the front tyres at
full height and full dump.

Joint ranges:
    lift_arm  0 (arms on the stops, bucket flat on the ground) … 1.16 rad
              (hinge pin at 3.13 m); 0.35 rad/s → full raise in 3.3 s.
    bucket    −1.88 (dump) … +0.55 rad (rollback) relative to the arms, i.e.
              +31.5° rollback at the ground, −41.5° dump / +98° rollback at
              full height; 0.9 rad/s.

Model simplifications:
    * The lift arms swing about a single rear-tower pin (radial lift path); the
      S650 / 262D3 vertical-lift linkage is a closed loop that URDF cannot
      express.  Pin heights, reach and bucket angles at both ends of the lift
      range match the datasheet.
    * Quick coupler (Bob-Tach style): the coupler plate sits 0.15–0.23 m ahead
      of the arm pins and the bucket back wall in front of it, so the bucket
      clears the front legs of the arms over the whole bucket range (tightest
      at full rollback, where the wall top passes just in front of the legs).
    * Lift and tilt cylinders are visual-only follower links.
      Lift: barrel pinned at its base, rod at its eye; their common direction
      is a linear fit of the true two-pin geometry (coaxial within 0.7 cm).
      Tilt: the barrel is pinned on the arm and the rod telescopes out of it
      (prismatic follower), so the cylinder is always straight and extends
      monotonically, 1.21 m (full curl) → 1.52 m (full dump), with its axis
      0.09–0.11 m from the coupler pin.  A real two-pin cylinder swinging the
      coupler through 139° cannot be matched by follower motions that are
      linear in q.  So the rod eye stays within 3 cm of a bracket line on the
      back of the coupler and slides ≈ 0.11 m along it; the bracket covers
      that slide (see ``_tilt_fit``).

Init pose = carry/travel pose: arms just off the stops, bucket rolled back with
its heel ≈ 0.26 m above the ground.

Dumping at low lift drives the bucket into the ground.  The full bucket range
clears the ground only from lift ≈ 0.25 rad (22 % of the lift range).  At the
init lift, bucket angles below −0.26 rad (≈ 67 % of the range, counted from
full dump as tasks.py does) put it as much as 0.49 m below ground.  A real loader cannot do this either: its cutting edge
would dig in and lift the front axle.  The kinematic environment does not
prevent it, so raise the arms before dumping.

Side cameras sit on brackets on top of the rear towers, behind and just
outboard of the arm pivots.  The arms never enter the left/right images.

Root frame: midway between the axles at axle height (root height = tyre radius).
"""

from __future__ import annotations

import math

from .. import kinematics as K
from ..checks import OVERLAP_TOL, obb_penetration
from ..spec import (Box, Camera, Drive, Joint, Link, MachineSpec, Tool, beam, box, cyl,
                    wheel_joint, wheel_link)

# --- running gear (12-16.5 tyres) -------------------------------------------
WHEEL_R, WHEEL_W = 0.42, 0.31
WHEELBASE, GAUGE = 1.16, 1.45
Z0 = WHEEL_R                       # ground is at z = -Z0 in the root frame

# --- body (x, z in the root frame; "- Z0" converts heights above ground) ------
BELLY_X = (-1.12, 0.86)            # main frame / chain cases between the tyres
BELLY_Y = 0.54
BELLY_Z = (0.20 - Z0, 0.67 - Z0)   # 0.20 m ground clearance
REAR_X = -1.42                     # rear bumper / counterweight face
HOOD_TOP = 1.20 - Z0               # engine cover
CAB_X = (-0.50, 0.72)
CAB_Y = 0.48
CAB_TOP = 2.06 - Z0                # top of ROPS = overall height
TOWER_X = (-1.32, -1.02)
TOWER_Y, TOWER_W = 0.58, 0.16      # rear uprights carrying the arm pins
SIDE_CAM = (-1.26, 0.88, 1.82 - Z0)  # on tower-top brackets, behind and outboard of the arm pins

# --- lift arms (root frame at lift q = 0, i.e. bucket flat on the ground) ---
ARM_Y, ARM_W = 0.73, 0.14
PIVOT = (-1.08, 1.54 - Z0)         # rear tower pin
KNEE = (0.92, 1.40 - Z0)           # arm turns down in front of the cab
HINGE = (1.26, 0.17 - Z0)          # coupler pin; front leg leans back 15.5 deg
HINGE_MAX_Z = 3.13 - Z0            # hinge-pin height at full lift
LEG_D = 0.18                       # front-leg depth (side view)
CYL_Y = 0.58                       # lift + tilt cylinders run just inboard of the arms
LIFT_CYL_BASE = (-1.06, 0.82 - Z0)  # on the tower
LIFT_CYL_EYE = (-0.28, 1.28 - Z0)   # lug under the arm
TILT_CYL_BASE = (0.72, 1.40 - Z0)   # lug on the inner face of the upper arm

# --- bucket (bucket frame: origin at the coupler pin, +x toward the cutting edge)
# The coupler plate sits in front of the pin and the bucket in front of it (as
# on a Bob-Tach), so the back wall clears the front legs at full rollback.
BKT_W = 1.83
BKT_BACK = 0.23                    # rear face of the back wall ahead of the pin
BKT_FLOOR = -0.17                  # floor below the pin
BKT_TOP = 0.45                     # top of the back wall above the pin
BKT_EDGE = 0.74                    # cutting edge (centre) ahead of the pin
TILT_EYE_X = 0.09                  # tilt-rod eye bracket on the back of the coupler
TILT_EYE_Z0 = 0.20                 # start value for the eye-height fit (see _tilt_fit)
TILT_LEVER_MIN = 0.09              # min. tilt-cylinder moment arm about the coupler pin

LIFT_INIT, BUCKET_INIT = 0.06, 0.40
BUCKET_LO, BUCKET_HI = -1.88, 0.55


def _rot(p: tuple[float, float], q: float) -> tuple[float, float]:
    """Rotate an (x, z) point by q about axis (0, -1, 0) (+q raises +x)."""
    c, s = math.cos(q), math.sin(q)
    return (c * p[0] - s * p[1], s * p[0] + c * p[1])


def _fit_cylinder(base, centre, eye, q_lo, q_hi, q_init, tol=0.02, n=40):
    """Linear fit of the direction of a two-pin hydraulic cylinder.

    ``base`` is fixed in the parent frame; the rod eye is ``centre + rot(eye, q)``.
    Returns (k, c, len_min, len_max, max_misalignment_m) with the direction
    angle ≈ k*q + c: minimax over [q_lo, q_hi], at most ``tol`` off at q_init.
    """
    def polar(q):
        r = _rot(eye, q)
        dx, dz = centre[0] + r[0] - base[0], centre[1] + r[1] - base[1]
        return math.atan2(dz, dx), math.hypot(dx, dz)

    qs = [q_lo + (q_hi - q_lo) * i / (n - 1) for i in range(n)]
    pts = [polar(q) for q in qs]
    th0, l0 = polar(q_init)
    best = None
    for i in range(601):
        k = -1.5 + 3.0 * i / 600
        res = [th - k * q for q, (th, _) in zip(qs, pts)]
        c0 = th0 - k * q_init
        c = min(max((max(res) + min(res)) / 2, c0 - tol / l0), c0 + tol / l0)
        err = max(abs(r - c) * ln for r, (_, ln) in zip(res, pts))
        if best is None or err < best[0]:
            best = (err, k, c)
    err, k, c = best
    return k, c, min(p[1] for p in pts), max(p[1] for p in pts), err


def _linear_fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return k, my - k * mx


def _tilt_fit(base, hinge, n=41) -> dict:
    """Linear follower fits (vs. bucket angle) for the tilt cylinders, arm frame.

    The barrel pivots on the arm (angle ``k*q + c``) and the rod slides out of
    it coaxially (stroke ``ks*q + cs`` from the base pin), so the cylinder is
    always straight, pinned exactly at its base and extends monotonically
    (extend = dump).  A real two-pin cylinder swinging the coupler through
    139 deg cannot be matched by link angles that are linear in q, so the rod
    eye wanders on the coupler: the fit keeps it within a few cm of the
    bracket line x = TILT_EYE_X (bucket frame) and lets it slide along it,
    while the cylinder axis stays at least TILT_LEVER_MIN from the coupler
    pin (no dead centre).  The slide range is returned in ``eye_z`` (bucket
    frame), the axis-to-pin distance in ``lever``.
    """
    qs = [BUCKET_LO + (BUCKET_HI - BUCKET_LO) * i / (n - 1) for i in range(n)]
    eye = (TILT_EYE_X, TILT_EYE_Z0)
    ang, dist = [], []
    for q in qs:
        r = _rot(eye, q)
        dx, dz = hinge[0] + r[0] - base[0], hinge[1] + r[1] - base[1]
        ang.append(math.atan2(dz, dx))
        dist.append(math.hypot(dx, dz))

    def eye_miss(p):
        k, c, ks, cs = p
        xs, zs = [], []
        for q in qs:
            a, s = k * q + c, ks * q + cs
            b = _rot((base[0] + s * math.cos(a) - hinge[0], base[1] + s * math.sin(a) - hinge[1]), -q)
            xs.append(b[0] - TILT_EYE_X)
            zs.append(b[1])
        return xs, zs

    hb = (hinge[0] - base[0], hinge[1] - base[1])

    def lever(p):     # distance of the visible cylinder axis from the hinge pin, per q
        return [hb[0] * math.sin(p[0] * q + p[1]) - hb[1] * math.cos(p[0] * q + p[1]) for q in qs]

    def cost(p):
        xs, zs = eye_miss(p)
        weak = max(0.0, TILT_LEVER_MIN - min(lever(p)))   # keep real leverage on the coupler
        return max(4.0 * max(abs(x) for x in xs), max(zs) - min(zs)) + 10.0 * weak

    p = [*_linear_fit(qs, ang), *_linear_fit(qs, dist)]
    steps = [0.04, 0.04, 0.08, 0.08]
    best = cost(p)
    for _ in range(12):                                         # coordinate descent, halving steps
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for sgn in (1.0, -1.0):
                    trial = list(p)
                    trial[i] += sgn * steps[i]
                    c_t = cost(trial)
                    if c_t < best - 1e-9:
                        p, best, improved = trial, c_t, True
        steps = [s / 2 for s in steps]
    xs, zs = eye_miss(p)
    k, c, ks, cs = p
    arms = lever(p)
    return {"k": k, "c": c, "ks": ks, "cs": cs, "eye_x": (min(xs), max(xs)),
            "eye_z": (min(zs), max(zs)), "lever": (min(arms), max(arms)),
            "stroke": (ks * BUCKET_HI + cs, ks * BUCKET_LO + cs)}


def _assert_bucket_clears_arms(spec: MachineSpec, n: int = 49) -> None:
    """Lift arm and bucket are parent/child, so checks.validate never compares
    them; sweep the bucket range here and refuse geometry where the collision
    boxes of the arm legs cut into the bucket."""
    def obb(link_tf, shape):
        tf = K.shape_tf(link_tf, shape)
        return tf.t, [(tf.R[0][i], tf.R[1][i], tf.R[2][i]) for i in range(3)], \
            tuple(v / 2 for v in shape.geom.size)

    arm = [s for s in spec.link("lift_arm").shapes if s.collision and isinstance(s.geom, Box)]
    bkt = [s for s in spec.link("bucket").shapes if s.collision and isinstance(s.geom, Box)]
    for i in range(n):
        q = BUCKET_LO + (BUCKET_HI - BUCKET_LO) * i / (n - 1)
        tfs = K.link_transforms(spec, {"bucket_joint": q})
        worst = max(obb_penetration(obb(tfs["lift_arm"], a), obb(tfs["bucket"], b))
                    for a in arm for b in bkt)
        assert worst <= OVERLAP_TOL, f"bucket cuts {worst:.3f} m into the lift arms at q={q:.2f}"


def _cylinder_pair(prefix, l_min, l_max, r_barrel, r_rod, m_barrel, m_rod):
    """Barrel link (+x from the base pin) and rod link (+x from the eye toward the base)."""
    barrel_len = l_min - 0.08
    rod_len = min(max(l_max - barrel_len + 0.06, 0.30), l_min - 0.06)
    barrel, rod = [], []
    for y in (CYL_Y, -CYL_Y):
        barrel += [
            cyl(r_barrel, barrel_len, at=(barrel_len / 2, y, 0.0), axis="x", mat="dark",
                collision=False),
            cyl(r_barrel * 1.2, 0.09, at=(0.0, y, 0.0), axis="y", mat="dark", collision=False),
            cyl(r_barrel * 1.12, 0.05, at=(barrel_len - 0.025, y, 0.0), axis="x", mat="steel",
                collision=False),                                              # gland
        ]
        rod += [
            cyl(r_rod, rod_len, at=(rod_len / 2, y, 0.0), axis="x", mat="steel", collision=False),
            cyl(r_rod * 1.7, 0.07, at=(0.0, y, 0.0), axis="y", mat="steel", collision=False),
        ]
    return (Link(f"{prefix}_cylinders", m_barrel, barrel), Link(f"{prefix}_rods", m_rod, rod))


def build() -> MachineSpec:
    x_ax, y_w = WHEELBASE / 2, GAUGE / 2

    # ------------------------------------------------------------------ chassis
    bx0, bx1 = BELLY_X
    tower_top = PIVOT[1] + 0.10
    cab_bot, cab_body_top = BELLY_Z[1], CAB_TOP - 0.08
    cab_len, cab_cx = CAB_X[1] - CAB_X[0], (CAB_X[0] + CAB_X[1]) / 2
    hood_x0 = REAR_X + 0.02
    sides = (1, -1)
    chassis = Link("chassis", 2440.0, [
        # main frame + chain cases between the tyres
        box(bx1 - bx0, 2 * BELLY_Y, BELLY_Z[1] - BELLY_Z[0],
            at=((bx0 + bx1) / 2, 0.0, (BELLY_Z[0] + BELLY_Z[1]) / 2)),
        box(0.12, 2 * BELLY_Y - 0.04, 0.04, at=(bx1 - 0.04, 0.0, BELLY_Z[1] + 0.02),
            mat="accent", collision=False),                                   # entry step
        # rear counterweight / bumper, full width behind the rear tyres
        box(bx0 - REAR_X, 1.32, 0.40, at=((REAR_X + bx0) / 2, 0.0, BELLY_Z[1] - 0.20),
            mat="dark"),
        # engine compartment, rear door with grille, intake screen, exhaust
        box(CAB_X[0] - hood_x0, 2 * (TOWER_Y - TOWER_W / 2), HOOD_TOP - BELLY_Z[1],
            at=((hood_x0 + CAB_X[0]) / 2, 0.0, (HOOD_TOP + BELLY_Z[1]) / 2)),
        box(0.03, 0.88, 0.44, at=(REAR_X + 0.005, 0.0, BELLY_Z[1] + 0.27),
            collision=False),
        *[box(0.02, 0.70, 0.035, at=(REAR_X - 0.012, 0.0, BELLY_Z[1] + 0.14 + 0.07 * i),
              mat="accent", collision=False) for i in range(5)],
        box(0.50, 0.40, 0.03, at=(-0.95, 0.0, HOOD_TOP + 0.015), mat="dark", collision=False),
        cyl(0.045, 0.30, at=(-1.20, -0.30, HOOD_TOP + 0.15), mat="accent", collision=False),
        # ROPS cab: body (collides) + roof, posts, glass (visual only)
        box(cab_len, 2 * CAB_Y, cab_body_top - cab_bot,
            at=(cab_cx, 0.0, (cab_bot + cab_body_top) / 2)),
        box(cab_len + 0.08, 2 * CAB_Y + 0.06, 0.08, at=(cab_cx + 0.02, 0.0, CAB_TOP - 0.04),
            mat="accent"),                                                    # roof
        box(0.02, 2 * CAB_Y - 0.12, 1.04, at=(CAB_X[1] + 0.01, 0.0, cab_bot + 0.13 + 0.52),
            mat="glass", collision=False),                                    # front door
        box(0.025, 2 * CAB_Y - 0.08, 0.05, at=(CAB_X[1] + 0.013, 0.0, cab_bot + 0.50),
            mat="accent", collision=False),                                   # door rail
        box(0.02, 2 * CAB_Y - 0.30, 0.36, at=(CAB_X[0] - 0.01, 0.0, HOOD_TOP + 0.26),
            mat="glass", collision=False),                                    # rear window
        *[box(cab_len - 0.20, 0.02, 0.74,
              at=(cab_cx + 0.02, s * (CAB_Y + 0.01), cab_body_top - 0.44),
              mat="glass", collision=False) for s in sides],                  # side windows
        *[box(0.07, 0.07, cab_body_top - cab_bot,
              at=(CAB_X[1] - 0.02, s * (CAB_Y - 0.02), (cab_bot + cab_body_top) / 2),
              mat="accent", collision=False) for s in sides],                 # front posts
        *[box(0.04, 0.03, 0.50, at=(CAB_X[1] + 0.05, s * (CAB_Y - 0.10), cab_bot + 0.60),
              mat="accent", collision=False) for s in sides],                 # grab handles
        *[box(0.10, 0.12, 0.07, at=(CAB_X[1] + 0.03, s * 0.36, CAB_TOP - 0.12),
              mat="white", collision=False) for s in sides],                 # work lights
        # rear towers carrying the arm pins, tail lights, lift-cylinder clevises
        *[box(TOWER_X[1] - TOWER_X[0], TOWER_W, tower_top - BELLY_Z[1],
              at=((TOWER_X[0] + TOWER_X[1]) / 2, s * TOWER_Y, (tower_top + BELLY_Z[1]) / 2))
          for s in sides],
        *[cyl(0.12, TOWER_W + 0.02, at=(PIVOT[0], s * TOWER_Y, PIVOT[1]), axis="y",
              collision=False) for s in sides],
        *[box(0.02, 0.10, 0.14, at=(TOWER_X[0] - 0.01, s * TOWER_Y, PIVOT[1] - 0.25),
              mat="red", collision=False) for s in sides],
        *[box(0.12, 0.12, 0.14, at=(LIFT_CYL_BASE[0], s * CYL_Y, LIFT_CYL_BASE[1]),
              mat="dark", collision=False) for s in sides],
        # side-camera brackets: post on the tower top, outrigger behind the arm pin, housing
        *[sh for s in sides for sh in (
            box(0.05, 0.05, SIDE_CAM[2] - tower_top, at=(SIDE_CAM[0] - 0.01, s * TOWER_Y,
                                                         (SIDE_CAM[2] + tower_top) / 2),
                mat="accent", collision=False),
            box(0.05, SIDE_CAM[1] - TOWER_Y, 0.04,
                at=(SIDE_CAM[0] - 0.01, s * (SIDE_CAM[1] + TOWER_Y) / 2, SIDE_CAM[2] - 0.02),
                mat="accent", collision=False),
            box(0.08, 0.07, 0.07, at=(SIDE_CAM[0] - 0.04, s * SIDE_CAM[1], SIDE_CAM[2]),
                mat="accent", collision=False))],
    ])

    # ------------------------------------------------------------- lift arms
    # Arm frame = rear pivot pin, aligned with the root frame at q = 0.
    def rel(p):
        return (p[0] - PIVOT[0], p[1] - PIVOT[1])

    knee, hinge = rel(KNEE), rel(HINGE)
    lift_eye, tilt_base = rel(LIFT_CYL_EYE), rel(TILT_CYL_BASE)
    arm_shapes = [cyl(0.07, 2 * ARM_Y + ARM_W, at=(0.36, 0.0, -0.05), axis="y")]  # cross tube
    for s in sides:
        y = s * ARM_Y
        arm_shapes += [
            beam((0.0, 0.0), knee, ARM_W, 0.20, y=y),                          # upper arm
            beam(knee, hinge, ARM_W, LEG_D, y=y),                              # front leg
            cyl(0.13, ARM_W, at=(knee[0], y, knee[1]), axis="y"),              # nose
            beam((knee[0] - 0.50, knee[1] - 0.03), (knee[0] + 0.05, knee[1] - 0.48),
                 ARM_W * 0.85, 0.26, y=y, collision=False),                    # web at the bend
            cyl(0.11, ARM_W + 0.04, at=(0.0, y, 0.0), axis="y", mat="steel",
                collision=False),                                              # pivot boss
            cyl(0.085, ARM_W + 0.04, at=(hinge[0], y, hinge[1]), axis="y", mat="steel",
                collision=False),                                              # hinge boss
            box(0.14, 0.12, 0.16, at=(lift_eye[0], s * 0.62, lift_eye[1] + 0.04),
                collision=False),                                              # lift-cyl lug
            box(0.16, 0.12, 0.18, at=(tilt_base[0], s * 0.62, tilt_base[1]),
                collision=False),                                              # tilt-cyl lug
        ]
    lift_arm = Link("lift_arm", 430.0, arm_shapes)

    # ----------------------------------------------------------------- bucket
    tilt = _tilt_fit(tilt_base, hinge)
    ez0, ez1 = tilt["eye_z"]
    yb, xb, xf = BKT_W / 2, BKT_BACK, BKT_BACK + 0.02      # xf: floor / side plates start
    bucket = Link("bucket", 260.0, [
        # quick coupler: plate against the bucket back, pivot tube on the arm pins,
        # bottom flange, and tilt-rod brackets covering the rod-eye slide (_tilt_fit)
        box(0.08, 1.30, BKT_TOP + 0.12, at=(xb - 0.04, 0.0, (BKT_TOP - 0.12) / 2), mat="dark"),
        cyl(0.07, 1.28, axis="y", mat="dark", collision=False),
        box(xb - 0.08, 1.28, 0.08, at=((xb - 0.08) / 2, 0.0, -0.08), mat="dark", collision=False),
        *[box(xb - TILT_EYE_X - 0.02, 0.10, ez1 - ez0 + 0.14,
              at=((xb + TILT_EYE_X) / 2 - 0.07, s * CYL_Y, (ez0 + ez1) / 2), mat="dark",
              collision=False) for s in sides],
        # bucket shell
        box(0.05, BKT_W, BKT_TOP - BKT_FLOOR, at=(xb + 0.025, 0.0, (BKT_TOP + BKT_FLOOR) / 2)),
        box(BKT_EDGE - xf, BKT_W, 0.05, at=((BKT_EDGE + xf) / 2, 0.0, BKT_FLOOR + 0.025)),
        box(0.24, BKT_W, 0.04, at=(xb + 0.16, 0.0, BKT_TOP - 0.02)),                # top lip
        box(0.08, BKT_W, 0.025, at=(BKT_EDGE, 0.0, BKT_FLOOR + 0.0125), mat="steel"),
        *[box(BKT_EDGE - xf, 0.03, 0.20, at=((BKT_EDGE + xf) / 2, s * (yb - 0.015),
                                            BKT_FLOOR + 0.10)) for s in sides],
        *[beam((xb + 0.06, BKT_TOP - 0.10), (BKT_EDGE - 0.02, BKT_FLOOR + 0.12), 0.03, 0.16,
               y=s * (yb - 0.015)) for s in sides],                              # side plates
    ])

    # ------------------------------------------------------------ joint ranges
    r_hinge = math.hypot(*hinge)
    lift_max = math.asin((HINGE_MAX_Z - PIVOT[1]) / r_hinge) + math.atan2(-hinge[1], hinge[0])
    # Lift cylinders: base on the tower, eye on the arm (rotates about PIVOT).
    lk, lc, ll0, ll1, _ = _fit_cylinder(LIFT_CYL_BASE, PIVOT, lift_eye, 0.0, lift_max, LIFT_INIT)
    lift_cyl, lift_rod = _cylinder_pair("lift", ll0, ll1, 0.06, 0.035, 60.0, 30.0)
    # Tilt cylinders: barrel pinned on the arm, rod telescoping out of it (_tilt_fit).
    s0, s1 = tilt["stroke"]                       # pin-to-eye at full curl / full dump
    t_barrel, t_rod = s0 - 0.10, s0 - 0.07
    assert t_barrel - (s1 - t_rod) > 0.15, "tilt rod leaves the barrel"
    tilt_cyl = Link("tilt_cylinders", 40.0, [sh for y in (CYL_Y, -CYL_Y) for sh in (
        cyl(0.055, t_barrel, at=(t_barrel / 2, y, 0.0), axis="x", mat="dark", collision=False),
        cyl(0.066, 0.09, at=(0.0, y, 0.0), axis="y", mat="dark", collision=False),
        cyl(0.062, 0.05, at=(t_barrel - 0.025, y, 0.0), axis="x", mat="steel",
            collision=False))])                                               # + gland
    tilt_rod = Link("tilt_rods", 20.0, [sh for y in (CYL_Y, -CYL_Y) for sh in (
        cyl(0.032, t_rod, at=(-t_rod / 2, y, 0.0), axis="x", mat="steel", collision=False),
        cyl(0.054, 0.07, at=(0.0, y, 0.0), axis="y", mat="steel", collision=False))])

    def follower(name, parent, child, xyz, src, k, c, src_range, src_init):
        """Link +x direction = k*q_src + c (angle above parent +x, about -y).

        The mid-range direction goes into the joint origin so the follower
        angle itself stays centred on 0 (well inside PhysX revolute limits).
        """
        q_mid = (src_range[0] + src_range[1]) / 2
        c0 = -k * q_mid
        lo, hi = sorted((k * src_range[0] + c0, k * src_range[1] + c0))
        return Joint(name, "revolute", parent, child, xyz=(xyz[0], 0.0, xyz[1]),
                     rpy=(0.0, -(k * q_mid + c), 0.0), axis=(0, -1, 0),
                     lower=lo - 0.02, upper=hi + 0.02, init=k * src_init + c0,
                     velocity=2.0, role="follower", follows=(src, k, c0))

    lift_rng, bkt_rng = (0.0, lift_max), (BUCKET_LO, BUCKET_HI)
    q_mid = (BUCKET_LO + BUCKET_HI) / 2
    ks = tilt["ks"]                               # < 0: curling retracts, dumping extends
    s_lo, s_hi = sorted((ks * (BUCKET_LO - q_mid), ks * (BUCKET_HI - q_mid)))
    joints = [
        Joint("lift_arm_joint", "revolute", "chassis", "lift_arm",
              xyz=(PIVOT[0], 0.0, PIVOT[1]), axis=(0, -1, 0), lower=0.0, upper=lift_max,
              init=LIFT_INIT, velocity=0.35, role="tool"),
        Joint("bucket_joint", "revolute", "lift_arm", "bucket",
              xyz=(hinge[0], 0.0, hinge[1]), axis=(0, -1, 0), lower=BUCKET_LO, upper=BUCKET_HI,
              init=BUCKET_INIT, velocity=0.9, role="tool"),
        # Lift: barrel direction = k*q + c in the parent frame; the rod hangs from its
        # eye pointing back at the base, i.e. direction + pi - q in the arm's frame.
        follower("lift_cylinder_joint", "chassis", "lift_cylinders", LIFT_CYL_BASE,
                 "lift_arm_joint", lk, lc, lift_rng, LIFT_INIT),
        follower("lift_rod_joint", "lift_arm", "lift_rods", lift_eye,
                 "lift_arm_joint", lk - 1.0, lc + math.pi, lift_rng, LIFT_INIT),
        follower("tilt_cylinder_joint", "lift_arm", "tilt_cylinders", tilt_base,
                 "bucket_joint", tilt["k"], tilt["c"], bkt_rng, BUCKET_INIT),
        # Rod frame origin = rod eye, at ks*q + cs from the base pin along the barrel.
        Joint("tilt_rod_joint", "prismatic", "tilt_cylinders", "tilt_rods",
              xyz=(ks * q_mid + tilt["cs"], 0.0, 0.0), axis=(1, 0, 0),
              lower=s_lo - 0.02, upper=s_hi + 0.02, init=ks * (BUCKET_INIT - q_mid),
              velocity=2.0, role="follower", follows=("bucket_joint", ks, -ks * q_mid)),
    ]

    wheels = [("front_left", +x_ax, +y_w, "roll_left"), ("rear_left", -x_ax, +y_w, "roll_left"),
              ("front_right", +x_ax, -y_w, "roll_right"), ("rear_right", -x_ax, -y_w, "roll_right")]
    joints += [wheel_joint(f"{n}_wheel_joint", "chassis", f"{n}_wheel", (x, y, 0.0), WHEEL_R,
                           role=role) for n, x, y, role in wheels]
    links = [chassis, lift_arm, bucket, lift_cyl, lift_rod, tilt_cyl, tilt_rod]
    links += [wheel_link(f"{n}_wheel", WHEEL_R, WHEEL_W, 80.0) for n, *_ in wheels]

    spec = MachineSpec(
        name="skid_steer_loader",
        abbrev="SSL",
        display_name="Skid Steer Loader",
        category="loading",
        reference="3.5 t wheeled skid steer loader (Bobcat S650 / CAT 262D3 class)",
        description="Four-wheel skid-steered compact loader with a centre ROPS cab, "
                    "rear-pivot lift arms and a 1.83 m bucket on a quick coupler.",
        root="chassis",
        links=links,
        joints=joints,
        drive=Drive(type="skid", max_speed=3.3, max_yaw_rate=1.0, track_width=GAUGE),
        tools=[
            Tool("lift_arm_joint", "lift_arm", positive="raise", negative="lower"),
            Tool("bucket_joint", "bucket", positive="curl", negative="dump"),
        ],
        cameras=[
            # Operator eye height, just ahead of the (opaque) front door glass.
            Camera("front", "chassis", (CAB_X[1] + 0.05, 0.0, 1.59 - Z0), (0.0, 0.50, 0.0)),
            # Rear tower tops, outboard of and behind the arm pivots, looking
            # forward-outward: the arms swing in a plane just inboard of the lens,
            # so they stay out of the side images over the whole lift range.
            Camera("left", "chassis", SIDE_CAM, (0.0, 0.40, 0.90)),
            Camera("right", "chassis", (SIDE_CAM[0], -SIDE_CAM[1], SIDE_CAM[2]), (0.0, 0.40, -0.90)),
        ],
        envelope=(3.45, 1.83, 2.06),
        operating_mass=3600.0,
        notes="Bucket curl/dump is not auto-verified (no up/out/left semantics); +1 rolls the "
              "cutting edge up and back (see the sweep render). Cylinders are visual-only "
              "followers; the tilt rods telescope coaxially out of their barrels and their "
              "eyes slide ~0.11 m along brackets on the coupler (linear-follower limit). "
              "Below lift ~0.25 rad, bucket targets under ~67% of the bucket range (q < -0.26 "
              "rad at the init lift) put the bucket below the ground (up to 0.49 m), so "
              "raise the arms before dumping; the kinematic env and tasks.py do not "
              "prevent it.",
    )
    _assert_bucket_clears_arms(spec)
    return spec
