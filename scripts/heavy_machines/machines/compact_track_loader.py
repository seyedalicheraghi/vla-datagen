"""Compact track loader (CTL), 4 t class — modelled on a CAT 259D3 / Bobcat T66.

Reference numbers (typical datasheet values for the class):
    operating mass ≈ 4.1 t (CAT 259D3 4 076 kg, Bobcat T66 ≈ 4 050 kg) ·
    rated operating capacity (35 % of tipping load) ≈ 0.9–1.1 t (259D3 915 kg,
    T66 ≈ 1 110 kg) ·
    length with bucket on the ground 3.56 m (2.87 m without bucket) ·
    height to top of cab 2.11 m · width over the 320 mm rubber tracks 1.68 m
    (track gauge 1.36 m) · 72 in (1.83 m) general-purpose bucket ·
    track 2.05 m overall / ≈ 1.5 m on the ground, ground clearance ≈ 0.23 m ·
    bucket pin height 0.22 m in carry, ≈ 3.10 m at max lift ·
    dump clearance at max lift ≈ 2.4 m, reach at max lift and dump ≈ 0.75 m ·
    rollback at ground ≈ 31°, dump angle at max lift ≈ 42° ·
    raise time ≈ 3.4 s, full bucket stroke ≈ 2.3 s ·
    travel 11.5 km/h (3.2 m/s, two-speed high range), counter-rotation ≈ 1 rad/s.

Model joints:
    lift_arm_joint  -0.020 … 1.06 rad, 0.32 rad/s.  At the lower limit (arms on
                    their stops) the bucket pin is 0.17 m up and a level bucket
                    rests its floor on the ground; in carry (q = 0) the pin is at
                    0.22 m, fully raised at 3.12 m.  A single pivot at the top of
                    the rear towers stands in for the vertical-lift linkage (a
                    closed loop URDF cannot express), so the pin path is radial:
                    it starts and ends at about the same x (1.27 m / 1.24 m ahead
                    of the track centre) but bulges 0.40 m forward at mid height,
                    where a true vertical-lift path would not.
    bucket_joint    -1.80 … +0.55 rad relative to the arms (135° tilt range),
                    1.0 rad/s; +1 = curl (roll back), -1 = dump.  Gives 31°
                    rollback at the ground, 42° dump at max lift (cutting edge
                    2.41 m high, 0.74 m ahead of the tracks) and ≈ 92° rollback
                    fully raised (the arms rotate 61°), as on the real machine.
    Quick coupler (Bob-Tach style): the coupler plate sits between the arms,
    0.16–0.24 m ahead of the arm pins, and the full-width bucket shell is in
    front of it.  The back wall therefore clears the front legs of the arms over
    the whole tilt range and comes to rest against them at full rollback.
    checks.validate never compares parent/child links, so build() re-checks
    this (``_assert_bucket_clears_arms``).
    The lift cylinders are visual-only followers: the barrel pivots on its
    frame pin and the rod on its arm pin, both linear in the lift angle (the
    geometry is chosen so that the error stays below 0.1°).  The tilt
    cylinders are left out: over a 135° tilt range a two-pin cylinder is far
    from linear in the bucket angle.  The best barrel + telescoping-rod
    follower fit leaves the rod eye ±4 cm off the coupler and sliding 0.15 m
    along it, and a barrel without a rod would look broken.

Layout as on the real machine: rubber tracks with the drive sprocket at the
rear, black ROPS cab in the middle, lift arms outboard of the cab on both
sides pivoting on the rear towers, engine under the rear hood behind the cab,
rear door with grille and tail lights.

Init pose = carry/travel pose: arms down, bucket rolled back 20° with its
lowest point ≈ 0.14 m above the ground.  Model envelope in that pose
3.63 × 1.83 × 2.11 m (3.59 m long with the bucket flat on the ground); max
height with the bucket raised ≈ 3.95 m.

Dumping at low lift drives the bucket into the ground.  The full bucket range
clears the ground only from lift ≈ 0.26 rad.  At the init lift, bucket angles
below ≈ -0.06 rad put the cutting edge below z = 0 (down to -0.64 m).  A real
loader cannot do this either, and the kinematic environment does not prevent
it, so raise the arms before dumping.

Camera coverage: the front camera sits just outside the (solid) cab at the door
glass.  No camera sees the bucket once the pin is above ≈ 2.6 m (lift ≥ 0.86
rad with the bucket in carry tilt, ≥ 0.96 rad when dumped).  The dump-at-height
phase of truck loading therefore has no visual feedback; the real operator looks
up through the roof window there.  Keep pin heights ≤ 2.3 m when the bucket has
to stay in view.

Root frame: centre of the undercarriage at track mid-height (0.28 m above the
ground).  Geometry below is written in ground coordinates (x, z above ground).
"""

from __future__ import annotations

import math

from .. import kinematics as K
from ..checks import OVERLAP_TOL, obb_penetration
from ..spec import (Box, Camera, Drive, Joint, Link, MachineSpec, Tool, beam, box, cyl,
                    sprocket_link)

# Undercarriage
TRACK_L, TRACK_H, SHOE_W, GAUGE = 2.05, 0.56, 0.32, 1.36
Z0 = TRACK_H / 2                      # root origin height above the ground
SPROCKET_R = 0.25
# Cab (ROPS) extents
CAB_X, CAB_Y, CAB_Z = (-0.50, 0.70), 0.48, (0.88, 2.03)
ROOF_TOP = 2.11
# Lift arms (ground coordinates in the carry pose)
PIVOT = (-1.20, 1.62)                 # arm pivot on top of the rear towers
KNEE = (0.92, 1.42)
PIN = (1.30, 0.22)                    # bucket / coupler pin
ARM_Y, ARM_W = 0.76, 0.10
LEG_D = 0.21                          # depth of the front leg (knee → pin) in side view
# Lift cylinders (ground coordinates in the carry pose)
LIFT_BASE = (-1.00, 0.85)             # pin on the frame
LIFT_EYE = (-0.40, 1.48)              # pin under the arm
LIFT_Y = 0.655
# Bucket (bucket frame: origin on the coupler pin, x toward the cutting edge, floor level
# at q = 0).  The coupler plate sits between the arms just ahead of the pin and the full-width
# bucket shell in front of it, so the back wall clears the front legs over the whole range.
BKT_W = 1.83
BKT_BACK = 0.24                       # rear face of the back wall ahead of the pin
BKT_TOP = 0.54                        # top of the back wall / spill guard above the pin
BKT_FLOOR = -0.17                     # underside of the floor below the pin
BKT_LIP = 0.18                        # depth of the spill guard (top lip)
BKT_EDGE = 0.86                       # front of the cutting edge ahead of the pin
COUPLER_Y = 0.62                      # half width of the coupler plate (inside the arms)
# Joint ranges
LIFT_LO, LIFT_HI, LIFT_INIT = -0.020, 1.06, 0.0
BUCKET_LO, BUCKET_HI, BUCKET_INIT = -1.80, 0.55, 0.35


def _bx(x0, x1, y0, y1, z0, z1, mat="body", collision=True, origin=(0.0, Z0)):
    """Box from extents; x/z relative to ``origin`` (x, z) — ground coords by default."""
    ox, oz = origin
    return box(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0),
               at=((x0 + x1) / 2 - ox, (y0 + y1) / 2, (z0 + z1) / 2 - oz),
               mat=mat, collision=collision)


def _rot(p, q):
    """Rotate (x, z) by q about the origin with the axis (0, -1, 0) convention (+q raises +x)."""
    c, s = math.cos(q), math.sin(q)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _cylinder_fit(base, pivot, eye0, lo, hi, n=61):
    """Least-squares line (k, c) for a cylinder's elevation angle vs. a joint angle q.

    ``base`` is fixed in the parent frame; ``eye0`` (eye at q = 0) rotates about
    ``pivot`` by q (axis (0, -1, 0)).  Returns (k, c, max_err, (len_min, len_max)).
    """
    qs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    ang, ln = [], []
    for q in qs:
        r = _rot((eye0[0] - pivot[0], eye0[1] - pivot[1]), q)
        dx, dz = pivot[0] + r[0] - base[0], pivot[1] + r[1] - base[1]
        ang.append(math.atan2(dz, dx))
        ln.append(math.hypot(dx, dz))
    mq, ma = sum(qs) / n, sum(ang) / n
    k = sum((q - mq) * (a - ma) for q, a in zip(qs, ang)) / sum((q - mq) ** 2 for q in qs)
    c = ma - k * mq
    err = max(abs(k * q + c - a) for q, a in zip(qs, ang))
    return k, c, err, (min(ln), max(ln))


def _assert_bucket_clears_arms(spec: MachineSpec, n: int = 25) -> None:
    """lift_arm and bucket are parent/child, so checks.validate never compares them:
    sweep the bucket range and refuse geometry where the bucket cuts into the arms."""
    def obb(link_tf, shape):
        tf = K.shape_tf(link_tf, shape)
        return tf.t, [(tf.R[0][i], tf.R[1][i], tf.R[2][i]) for i in range(3)], \
            tuple(v / 2 for v in shape.geom.size)

    arm = [s for s in spec.link("lift_arm").shapes if s.collision and isinstance(s.geom, Box)]
    bkt = [s for s in spec.link("bucket").shapes if s.collision and isinstance(s.geom, Box)]
    for i in range(n):
        q = BUCKET_LO + (BUCKET_HI - BUCKET_LO) * i / (n - 1)
        tfs = K.link_transforms(spec, {"bucket_joint": q})
        arm_obbs = [obb(tfs["lift_arm"], s) for s in arm]
        worst = max(obb_penetration(a, obb(tfs["bucket"], b)) for b in bkt for a in arm_obbs)
        if worst > OVERLAP_TOL:
            raise ValueError(f"bucket cuts {worst:.3f} m into the lift arms at bucket q = {q:.2f}")


def _track(y: float) -> list:
    """Rubber track loop + track frame, bogie rollers and front idler, centred at (0, y, 0)."""
    r = TRACK_H / 2
    straight = TRACK_L - TRACK_H
    belt = 0.07
    return [
        box(straight, SHOE_W, belt, at=(0.0, y, -r + belt / 2), mat="rubber"),        # bottom run
        box(straight, SHOE_W, belt, at=(0.0, y, r - belt / 2), mat="rubber"),         # top run
        cyl(r, SHOE_W, at=(straight / 2, y, 0.0), axis="y", mat="rubber"),           # front bend
        cyl(r, SHOE_W, at=(-straight / 2, y, 0.0), axis="y", mat="rubber"),          # rear bend
        box(straight - 0.10, SHOE_W * 0.7, 0.26, at=(0.0, y, 0.02), mat="dark"),      # track frame
        cyl(0.22, SHOE_W + 0.02, at=(straight / 2, y, 0.0), axis="y", mat="steel",
            collision=False),                                                        # front idler
        cyl(0.10, SHOE_W + 0.03, at=(straight / 2, y, 0.0), axis="y", mat="dark",
            collision=False),                                                        # idler hub
        *[cyl(0.095, SHOE_W * 0.9, at=(x, y, -r + belt + 0.095), axis="y", mat="steel",
              collision=False) for x in (-0.40, -0.135, 0.135, 0.40)],               # bogie rollers
    ]


def build() -> MachineSpec:
    y_t = GAUGE / 2
    cx0, cx1 = CAB_X
    cz0, cz1 = CAB_Z

    # ------------------------------------------------------------------ chassis
    chassis_shapes = [
        *_track(+y_t),
        *_track(-y_t),
        _bx(-1.28, 0.95, -0.50, 0.50, 0.24, 0.62),                           # main frame between tracks
        _bx(-1.32, 0.80, -0.62, 0.62, 0.60, 0.88),                           # upper frame / deck
        _bx(0.80, 0.98, -0.40, 0.40, 0.34, 0.80, mat="dark"),                # front nose / step
        _bx(-1.40, -0.50, -0.58, 0.58, 0.88, 1.22),                          # engine hood
        _bx(-1.46, -1.30, -0.62, 0.62, 0.30, 0.50, mat="dark"),              # rear bumper / counterweight
        _bx(-1.43, -1.40, -0.47, 0.47, 0.52, 1.16, mat="accent", collision=False),   # rear door grille
        *[_bx(-1.445, -1.43, -0.42, 0.42, z, z + 0.035, mat="dark", collision=False)
          for z in (0.60, 0.70, 0.80, 0.90, 1.00)],                         # grille slats
        cyl(0.05, 0.30, at=(-1.15, -0.30, 1.37 - Z0), mat="accent", collision=False),  # exhaust
        # ROPS cab: black frame, glass all round, roof with work lights
        _bx(cx0, cx1, -CAB_Y, CAB_Y, cz0, cz1, mat="accent"),
        _bx(cx0 - 0.03, cx1 + 0.04, -CAB_Y - 0.02, CAB_Y + 0.02, cz1, ROOF_TOP, mat="accent"),
        _bx(cx1, cx1 + 0.02, -0.40, 0.40, 0.98, 1.96, mat="glass", collision=False),     # front door
        beam((cx1 + 0.025, 1.04 - Z0), (cx1 + 0.025, 1.50 - Z0), 0.03, 0.015, y=0.10,
             mat="dark", collision=False),                                  # wiper
        _bx(cx0 - 0.015, cx0, -0.36, 0.36, 1.32, 1.95, mat="glass", collision=False),   # rear window
    ]
    for sy in (1.0, -1.0):
        chassis_shapes += [
            _bx(-0.40, 0.62, sy * CAB_Y, sy * (CAB_Y + 0.015), 1.12, 1.95, mat="glass",
                collision=False),                                            # side window
            _bx(cx1 + 0.02, cx1 + 0.05, sy * 0.30, sy * 0.42, 1.97, 2.02, mat="white",
                collision=False),                                            # work light
            # rear tower carrying the lift-arm pivot, with brace and tail light
            _bx(-1.40, -0.95, sy * 0.47, sy * 0.60, 0.88, 1.72),
            # brace ends low on the tower so the arm cross tube clears it at the lift stop
            beam((-1.02, 1.52 - Z0), (-0.50, 0.98 - Z0), 0.13, 0.16, y=sy * 0.535),
            _bx(-1.42, -1.40, sy * 0.49, sy * 0.58, 1.30, 1.52, mat="red", collision=False),
            cyl(0.11, 0.16, at=(PIVOT[0], sy * 0.58, PIVOT[1] - Z0), axis="y", mat="steel",
                collision=False),                                            # pivot boss
            _bx(LIFT_BASE[0] - 0.08, LIFT_BASE[0] + 0.08, sy * 0.60, sy * (LIFT_Y + 0.07),
                LIFT_BASE[1] - 0.10, LIFT_BASE[1] + 0.05, mat="dark", collision=False),  # cyl lug
        ]
    chassis = Link("chassis", 3030.0, chassis_shapes)

    # ------------------------------------------------------------------ lift arms
    # Arm frame = ground coordinates relative to the pivot (carry pose, q = 0).
    def a(p):
        return (p[0] - PIVOT[0], p[1] - PIVOT[1])

    knee, pin = a(KNEE), a(PIN)
    arm_shapes = []
    for sy in (1.0, -1.0):
        y = sy * ARM_Y
        arm_shapes += [
            beam(a((-1.30, 1.64)), a((-0.30, 1.56)), ARM_W, 0.30, y=y),      # rear section
            beam(a((-0.30, 1.56)), knee, ARM_W, 0.24, y=y),                  # along the cab
            beam(knee, pin, ARM_W, LEG_D, y=y),                              # front leg
            cyl(0.13, ARM_W, at=(knee[0], y, knee[1]), axis="y"),            # knee
            cyl(0.10, ARM_W + 0.04, at=(pin[0], y, pin[1]), axis="y", mat="steel",
                collision=False),                                            # coupler pin boss
            cyl(0.12, 0.22, at=(0.0, sy * 0.70, 0.0), axis="y", mat="steel", collision=False),
            _bx(LIFT_EYE[0] - 0.08, LIFT_EYE[0] + 0.08, sy * (LIFT_Y - 0.04), sy * (ARM_Y - 0.05),
                LIFT_EYE[1] - 0.05, LIFT_EYE[1] + 0.10, origin=PIVOT, collision=False),  # cyl lug
        ]
    cross = a((-0.82, 1.64))
    arm_shapes.append(cyl(0.075, 2 * ARM_Y, at=(cross[0], 0.0, cross[1]), axis="y"))  # cross tube
    lift_arm = Link("lift_arm", 520.0, arm_shapes)

    # ------------------------------------------------------------------ bucket
    # Bucket frame: origin on the coupler pin; at q = 0 the floor is level.
    o = (0.0, 0.0)
    bw, xb, xl = BKT_W / 2, BKT_BACK, BKT_BACK + BKT_LIP
    xc = xb - 0.08                                                     # coupler plate rear face
    # Side-plate top edge: from the front of the spill guard down to the cutting edge.
    e0, e1 = (xl, BKT_TOP - 0.02), (BKT_EDGE - 0.03, BKT_FLOOR + 0.04)
    slope = (e1[1] - e0[1]) / (e1[0] - e0[0])
    edge_d = 0.06                                                      # edge-rim depth
    # Stepped boxes fill the side plate under the edge.  Each step top meets the edge line at
    # its mid-point; steps are narrow enough that their corners stay inside the rim, which
    # reaches edge_d/2 (measured square to the edge) either side of the line, so there are
    # no gaps and nothing pokes above the rim.
    step_max = edge_d * math.hypot(1.0, slope) / abs(slope)
    n_steps = math.ceil((e1[0] - e0[0]) / (0.85 * step_max))
    step_w = (e1[0] - e0[0]) / n_steps
    bucket_shapes = [
        # quick coupler (between the arms): plate, pivot tube on the arm pins, bottom flange
        _bx(xc, xb, -COUPLER_Y, COUPLER_Y, -0.12, BKT_TOP + 0.06, mat="dark", origin=o),
        cyl(0.07, 2 * COUPLER_Y, axis="y", mat="dark", collision=False),
        _bx(0.0, xc, -COUPLER_Y, COUPLER_Y, -0.12, -0.05, mat="dark", origin=o, collision=False),
        # bucket shell
        _bx(xb, xb + 0.04, -bw, bw, BKT_FLOOR + 0.02, BKT_TOP, origin=o),               # back wall
        _bx(xb, BKT_EDGE - 0.06, -bw, bw, BKT_FLOOR, BKT_FLOOR + 0.04, origin=o),       # floor
        _bx(BKT_EDGE - 0.10, BKT_EDGE, -bw, bw, BKT_FLOOR, BKT_FLOOR + 0.025, mat="steel",
            origin=o),                                                                  # cutting edge
        _bx(xb, xl, -bw, bw, BKT_TOP - 0.04, BKT_TOP, origin=o),                        # spill guard
    ]
    for sy in (1.0, -1.0):
        y0, y1 = sy * (bw - 0.03), sy * bw
        bucket_shapes += [                                                 # solid side plates
            _bx(xb, xl, y0, y1, BKT_FLOOR, BKT_TOP, origin=o),
            *[_bx(x0, x0 + step_w, y0, y1, BKT_FLOOR,
                  e0[1] + slope * (x0 + step_w / 2 - e0[0]), origin=o)
              for x0 in (e0[0] + i * step_w for i in range(n_steps))],
            beam(e0, e1, 0.03, edge_d, y=sy * (bw - 0.015)),               # edge rim
            _bx(-0.07, xc, sy * 0.60, sy * 0.70, -0.10, 0.14, mat="dark", origin=o,
                collision=False),                                          # coupler ear to the pin
        ]
    bucket = Link("bucket", 290.0, bucket_shapes)

    # ------------------------------------------------------------------ lift cylinders
    k, c, err, (lmin, lmax) = _cylinder_fit(LIFT_BASE, PIVOT, LIFT_EYE, LIFT_LO, LIFT_HI)
    if err > 0.01:   # barrel and rod would visibly misalign
        raise ValueError(f"lift cylinder angle is not linear enough in the lift angle ({err:.3f} rad)")
    barrel_len = lmin - 0.06                      # barrel never reaches the rod eye
    rod_len = lmax + 0.08 - barrel_len            # rod still 8 cm inside at full stroke
    barrels = Link("lift_cylinders", 70.0, [
        s for sy in (1.0, -1.0) for s in (
            cyl(0.06, barrel_len + 0.05, at=((barrel_len - 0.05) / 2, sy * LIFT_Y, 0.0), axis="x",
                mat="dark", collision=False),
            cyl(0.075, 0.05, at=(barrel_len - 0.025, sy * LIFT_Y, 0.0), axis="x", mat="dark",
                collision=False),
        )])
    rods = Link("lift_rods", 40.0, [
        s for sy in (1.0, -1.0) for s in (
            cyl(0.036, rod_len, at=(rod_len / 2, sy * LIFT_Y, 0.0), axis="x", mat="steel",
                collision=False),
            cyl(0.055, 0.10, at=(0.0, sy * LIFT_Y, 0.0), axis="y", mat="steel", collision=False),
        )])
    # The rod (on the arm) points back at the base: angle = barrel angle + pi - q.
    k_rod, c_rod = k - 1.0, _wrap(c + math.pi)

    def follower(name, parent, child, xyz, kk, cc):
        vals = (kk * LIFT_LO + cc, kk * LIFT_HI + cc)
        return Joint(name, "revolute", parent, child, xyz=xyz, axis=(0, -1, 0),
                     lower=min(vals) - 0.05, upper=max(vals) + 0.05, init=kk * LIFT_INIT + cc,
                     velocity=1.0, role="follower", follows=("lift_arm_joint", kk, cc))

    sprocket_x = -(TRACK_L - TRACK_H) / 2
    links = [chassis, lift_arm, bucket, barrels, rods,
             sprocket_link("left_sprocket", SPROCKET_R, SHOE_W + 0.03, 45.0),
             sprocket_link("right_sprocket", SPROCKET_R, SHOE_W + 0.03, 45.0)]

    joints = [
        Joint("lift_arm_joint", "revolute", "chassis", "lift_arm",
              xyz=(PIVOT[0], 0.0, PIVOT[1] - Z0), axis=(0, -1, 0),
              lower=LIFT_LO, upper=LIFT_HI, init=LIFT_INIT, velocity=0.32, role="tool"),
        Joint("bucket_joint", "revolute", "lift_arm", "bucket",
              xyz=(pin[0], 0.0, pin[1]), axis=(0, -1, 0),
              lower=BUCKET_LO, upper=BUCKET_HI, init=BUCKET_INIT, velocity=1.0, role="tool"),
        follower("lift_cylinder_joint", "chassis", "lift_cylinders",
                 (LIFT_BASE[0], 0.0, LIFT_BASE[1] - Z0), k, c),
        follower("lift_rod_joint", "lift_arm", "lift_rods",
                 (a(LIFT_EYE)[0], 0.0, a(LIFT_EYE)[1]), k_rod, c_rod),
        Joint("left_sprocket_joint", "continuous", "chassis", "left_sprocket",
              xyz=(sprocket_x, y_t, 0.0), axis=(0, 1, 0), velocity=20.0, role="roll_left",
              wheel_radius=SPROCKET_R),
        Joint("right_sprocket_joint", "continuous", "chassis", "right_sprocket",
              xyz=(sprocket_x, -y_t, 0.0), axis=(0, 1, 0), velocity=20.0, role="roll_right",
              wheel_radius=SPROCKET_R),
    ]

    cam_z = 1.85 - Z0
    spec = MachineSpec(
        name="compact_track_loader",
        abbrev="CTL",
        display_name="Compact Track Loader",
        category="loading",
        reference="4 t compact track loader (CAT 259D3 / Bobcat T66 class)",
        description="Rubber-tracked skid-steer loader with lift arms either side of the cab "
                    "(single-pivot approximation of vertical-lift arms, radial path) and a "
                    "1.83 m general-purpose bucket on a quick coupler.",
        root="chassis",
        links=links,
        joints=joints,
        drive=Drive(type="skid", max_speed=3.2, max_yaw_rate=1.0, track_width=GAUGE),
        tools=[
            Tool("lift_arm_joint", "lift_arm", positive="raise", negative="lower"),
            Tool("bucket_joint", "bucket", positive="curl", negative="dump"),
        ],
        cameras=[
            # operator view at the front door glass, looking down at the bucket
            Camera("front", "chassis", (cx1 + 0.05, 0.0, cam_z), (0.0, 0.42, 0.0)),
            # on the front corners of the roof, looking out over the lift arms
            Camera("left", "chassis", (0.62, CAB_Y + 0.04, ROOF_TOP + 0.04 - Z0), (0.0, 0.50, 0.75)),
            Camera("right", "chassis", (0.62, -CAB_Y - 0.04, ROOF_TOP + 0.04 - Z0), (0.0, 0.50, -0.75)),
        ],
        envelope=(3.56, 1.83, 2.11),
        operating_mass=4080.0,
        ground_contact=["chassis"],
        notes="Bucket curl/dump is not auto-verified (no up/out/left semantics); +1 rolls the "
              "cutting edge up (see sweep render). Lift cylinders are visual-only followers; "
              "tilt cylinders are omitted. No camera sees the bucket once the pin is above "
              "~2.6 m (lift >= 0.86 rad in carry tilt, >= 0.96 rad dumped), so the "
              "dump-at-height phase of truck loading has no visual feedback; keep pin heights "
              "<= 2.3 m when the bucket must stay in view. Below lift ~0.26 rad, dumping "
              "(bucket < ~-0.06 rad at the init lift) puts the bucket below the ground; the "
              "kinematic env does not prevent it, so raise the arms before dumping.",
    )
    _assert_bucket_clears_arms(spec)
    return spec
