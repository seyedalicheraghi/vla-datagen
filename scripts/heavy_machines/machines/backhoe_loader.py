"""Backhoe loader (BHL), 8 t class — modelled on a JCB 3CX / CAT 420.

Datasheet numbers (JCB 3CX Sitemaster, standard dipper, rounded):
    operating mass 8,070 kg (class 7.5-8.8 t; CAT 420F2 from 7.7 t) ·
    travel (roading) length 5.62 m · overall width 2.36 m (loader shovel) ·
    travel height 3.61 m (top of the folded backhoe; CAT 420 3.58 m) · height
    to top of cab 2.87 m · wheelbase 2.17 m (CAT 420 2.2 m) · ground clearance
    0.37 m · rear tyres 16.9-28 (Ø 1.44 m, 0.43 m section) · front tyres
    12.5/80-18 (Ø 0.97 m, 0.32 m) · travel speed 40 km/h (4F/4R shuttle) ·
    backhoe dig depth 4.24 m · reach from the slew centre 5.37 m · load height
    3.84 m · slew arc 180° · 1.0 m³ GP loader shovel, 2.35 m wide.
Assumed / typical for the class:
    track (gauge) 1.86 m rear / 1.84 m front (width over rear tyres 2.29 m) ·
    loader hinge-pin height at full lift 3.40 m · shovel rollback 43° / dump
    54°, self-levelling "parallel lift" (the shovel keeps its angle to the
    ground while the arms lift) · loader raise 4.7 s, carry-to-dump 1.4 s ·
    backhoe boom 2.35 m pin to pin, dipper 2.05 m, 0.6 m (24") bucket, 0.80 m
    pin to tip · boom 151°, dipper 138°, bucket 203° rotation · boom raise
    2.6 s, dipper 2.4 s, bucket 2.4 s, slew 172° in 3 s · vertical (slide)
    stabiliser legs, 0.45 m stroke from the travel position to the pads on the
    ground · front-wheel steer, inner wheel 44° at the 0.6 rad virtual lock.

Model check (offline FK):
    travel L × W × H 6.05 × 2.36 × 3.64 m (L +8 %: the shovel is carried
    rolled back clear of the nose with its lowest point 0.33 m above the
    ground; real 3CX shovels tuck closer) · dig depth 4.14 m · reach at ground
    level from the slew centre 5.23 m · max bucket-pin height 4.82 m · loader
    hinge pin 3.40 m, dump clearance at 45° 2.58 m, digs 0.08 m below grade ·
    turning radius over the outer front tyre 4.8 m at full lock · mass 8,070 kg
    (chassis/cab/engine/axles 4.47 t, wheels + knuckles 0.95 t, loader 1.19 t,
    backhoe 1.28 t, stabiliser legs 0.19 t).

Joint ranges / rates at full command:
    loader_arm      -0.04 … +1.38 rad (hinge pin 3.40 m), 0.30 rad/s
    loader_bucket   -0.95 (dump) … +0.75 rad (rollback), 1.15 rad/s; the angle
                    of the shovel floor to the ground (self-levelling follower)
    backhoe_swing   ±1.5 rad about the king post, 1.0 rad/s; +1 = "left" = CCW
                    seen from above (to the left of an operator facing the hoe)
    backhoe_boom    -1.15 … +1.48 rad (pin line to horizontal), 1.0 rad/s
    backhoe_stick   -1.45 (in) … +0.95 rad (out), 1.0 rad/s
    backhoe_bucket  -1.10 (dump) … +2.45 rad (curl), 1.5 rad/s
    stabilizers     0 … 0.45 m down (pads at grade), 0.25 m/s; right leg is a
                    follower of the left one

Drive: Ackermann, front-wheel steer, rear (fixed) axle at x = 0.  max_speed
11.1 m/s (40 km/h, same in reverse on the 4F/4R shuttle), max_steer 0.6 rad.
Virtual steer rate 0.16 rad/s (lock to lock ≈ 7.5 s, a road-speed steering
rate): the kinematic drive has no lateral-acceleration limit and on a 2.17 m
wheelbase a site-speed rate (0.3 rad/s) spins the machine > 180° in 3 s at
70 % of top speed.

Hydraulics (visual only): lift, boom and dipper rams are barrel/rod follower
pairs whose axis is a linear fit of the true pin-to-pin line (≤ 1 cm off in
the roading pose; worst 5 / 7 / 12 cm at the range ends).  The shovel rams are
fixed on the arms (the parallel-lift tipping linkage is not modelled) and the
bucket ram / dogbone links are fixed on the dipper.

Init pose = roading pose: loader shovel rolled back just above the ground,
backhoe boom raised, dipper folded in and hanging behind the boom, bucket just
curled off the vertical, slew centred, stabilisers up.

Root frame: chassis, origin at the rear-axle centre (rear axle at z = 0,
ground at z = -REAR_R), +x towards the loader end.
"""

from __future__ import annotations

import math

from ..spec import (Camera, Drive, Joint, Link, MachineSpec, Shape, Tool, beam, box, cyl,
                    sphere, wheel_joint, wheel_link)

# --- running gear --------------------------------------------------------------------
REAR_R, REAR_W, REAR_Y = 0.72, 0.43, 0.93        # 16.9-28, gauge 1.86 m
FRONT_R, FRONT_W, FRONT_Y = 0.48, 0.32, 0.92     # 12.5/80-18, gauge 1.84 m
WB = 2.17                                         # wheelbase (rear axle at x = 0)


def agl(h: float) -> float:
    """Height above ground -> root-frame z."""
    return h - REAR_R


# --- cab / body ----------------------------------------------------------------------
CAB_X0, CAB_X1 = -0.55, 1.05                      # cab rear / front wall
CAB_HW = 0.83                                     # half width of the glazed upper cab
CAB_TOP = 2.87                                    # top of the cab roof (AGL)
NOSE_X = 2.70                                     # front of the grille / counterweight

# --- loader (root frame (x, z) via agl(); arm frame root-aligned at q = 0) ---------------
A_PIN = (1.00, agl(1.70))                         # loader-arm pivots on the cab-front towers
ARM_Y, ARM_W = 0.64, 0.14
HOOD_W = 0.82                                     # bonnet width
B_PIN = (1.93, -1.42)                             # shovel hinge pin (arm frame), 0.28 m AGL at q = 0
ARM_KNEE = (1.20, -0.66)                          # gentle bend of the arm above the front tyre
B_MAX_H = 3.40                                    # hinge-pin height at full lift (AGL)
LIFT_LO, LIFT_INIT = -0.04, 0.11                  # roading: hinge pin 0.50 m AGL
BKT_LO, BKT_HI, BKT_INIT = -0.95, 0.75, 0.70
LEVEL = 1.0                                       # parallel lift: shovel keeps its ground angle
LC_BASE = (2.57, agl(0.87))                       # lift-ram base pins at the nose (root frame)
LC_EYE = (0.30, -0.70)                            # lift-ram eyes on drop lugs under the arms
LC_Y = 0.49                                       # lift rams run along the bonnet sides
TILT_Y = 0.78                                     # shovel (tilt) rams outboard of the arms
BKT_W = 2.34
# shovel profile (bucket frame: origin = hinge pin, +x toward the cutting edge, floor flat)
SH_FLOOR, SH_EDGE, SH_BACK, SH_TOP, SH_LIP = -0.28, 0.88, 0.06, 0.50, 0.40
TILT_BASE = (0.75, -0.36)                         # shovel-ram base lug (arm frame)
TILT_LEVER = (1.83, -1.14)                        # tipping-lever pin above the hinge (arm frame)

# --- backhoe ---------------------------------------------------------------------------
SWING_X, SWING_H = -1.20, 0.85                    # king-post axis, swing frame origin height
FOOT = (0.24, 0.05)                               # boom foot pin (swing frame, +x = rearward)
BOOM_L, BOOM_KNEE = 2.35, (0.95, 0.33)
STICK_L, STICK_TAIL, STICK_PITCH = 2.05, 0.34, 1.55
HOE_W, HOE_R = 0.60, 0.80
BOOM_LO, BOOM_HI, BOOM_INIT = -1.15, 1.48, 1.40
STICK_LO, STICK_HI, STICK_INIT = -1.45, 0.95, -1.45
HOE_LO, HOE_HI, HOE_INIT = -1.10, 2.45, 0.20
BC_BASE = (0.05, -0.20)                           # boom-ram base inside the casting (swing frame)
BC_EYE = (0.90, -0.05)                            # boom-ram eye under the boom (boom frame)
DC_BASE = (0.95, 0.55)                            # dipper-ram base on top of the boom (boom frame)
DC_EYE = (-0.30, 0.10)                            # dipper-ram eye on the dipper tail (stick frame)
RAM_W_INIT = 10.0                                 # ram fits: weight of the roading pose

# --- stabilisers -------------------------------------------------------------------------
STAB_X, STAB_Y = -0.92, 0.98
STAB_TOP = 0.70                                   # casing bottom (AGL) = leg joint origin
PAD_H = 0.45                                      # pad underside in the travel position (AGL)

# --- masses (kg) ---------------------------------------------------------------------------
M = dict(chassis=4470.0, rear_wheel=310.0, front_wheel=110.0, knuckle=55.0, arm=620.0,
         shovel=440.0, swing=300.0, boom=430.0, stick=250.0, hoe=140.0, leg=95.0)


# ---------------------------------------------------------------------------------
# Planar helpers: (x, z) pairs, rotation about -y (positive raises +x)
# ---------------------------------------------------------------------------------

def _rot(p, q):
    c, s = math.cos(q), math.sin(q)
    return (c * p[0] - s * p[1], s * p[0] + c * p[1])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _lerp(a, b, t):
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def _elev(v):
    return math.atan2(v[1], v[0])


def _unwrap(ys):
    out = list(ys)
    for i in range(1, len(out)):
        while out[i] - out[i - 1] > math.pi:
            out[i] -= 2 * math.pi
        while out[i] - out[i - 1] < -math.pi:
            out[i] += 2 * math.pi
    return out


def _minimax_fit(qs, ys, ws):
    """Chebyshev line k*q + c minimising max w*|y - (k*q + c)| -> (k, c)."""
    ys = _unwrap(ys)

    def err(k):
        rs = [y - k * q for q, y in zip(qs, ys)]
        c = (max(rs) + min(rs)) / 2
        return max(w * abs(r - c) for r, w in zip(rs, ws)), c
    lo, hi = -4.0, 4.0
    for _ in range(80):                       # ternary search on a convex max(|.|)
        k1, k2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        lo, hi = (k1, hi) if err(k1)[0] > err(k2)[0] else (lo, k2)
    k = (lo + hi) / 2
    return k, err(k)[1]


def cylinder_fit(base, origin, a0, eye, lo, hi, init=None, w_init=0.0, n=61):
    """Hydraulic ram between ``base`` (parent frame) and ``eye`` (child frame).

    The child frame sits at ``origin`` in the parent with its +x at elevation
    a0 + q (joint axis -y).  The barrel axis elevation in the parent frame is
    modelled as k*q + c, minimising the worst lateral pin miss over [lo, hi]
    (errors near ``init`` weighted (1 + w_init) times, so the roading pose is
    nearly exact); the rod follows the same line back from the eye.  Returns
    the barrel (k, c) in the parent frame, the rod (k, c) in the child frame,
    the worst lateral pin miss (m) and the min / max pin distance.
    """
    def geo(q):
        e = _add(origin, _rot(eye, a0 + q))
        return _elev(_sub(e, base)), math.hypot(*_sub(e, base))
    qs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    if init is not None:
        qs = sorted(qs + [init])
    th, lens = zip(*[geo(q) for q in qs])
    ws = [ln * (1.0 + (w_init * math.exp(-((q - init) / 0.08) ** 2) if init is not None else 0.0))
          for q, ln in zip(qs, lens)]
    kb, cb = _minimax_fit(qs, th, ws)
    th = _unwrap(th)
    miss = max(ln * abs(math.sin(t - (kb * q + cb))) for q, t, ln in zip(qs, th, lens))
    return (kb, cb), (kb - 1.0, cb + math.pi - a0), miss, min(lens), max(lens)


def _follower(name, parent, child, xz, src: Joint, k, c, y=0.0) -> Joint:
    """Revolute follower: child +x elevation (about -y) relative to parent = k*q_src + c.

    The mid-range angle goes into the joint origin so the follower angle stays
    centred on 0.
    """
    q_mid = (src.lower + src.upper) / 2
    base = k * q_mid + c
    c0 = -k * q_mid
    lo, hi = sorted((k * src.lower + c0, k * src.upper + c0))
    return Joint(name, "revolute", parent, child, xyz=(xz[0], y, xz[1]),
                 rpy=(0.0, -base, 0.0), axis=(0, -1, 0), lower=lo - 0.02, upper=hi + 0.02,
                 init=k * src.init + c0, velocity=abs(k) * src.velocity + 0.1,
                 role="follower", follows=(src.name, k, c0))


# ---------------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------------

def _rod(p0, p1, radius, y=0.0, mat="steel") -> Shape:
    """Visual-only cylinder from p0 to p1 in the link's x-z plane."""
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    return cyl(radius, math.hypot(dx, dz), at=((p0[0] + p1[0]) / 2, y, (p0[1] + p1[1]) / 2),
               rpy=(0.0, math.atan2(dx, dz), 0.0), mat=mat, collision=False)


def _ram_links(prefix, l_min, l_max, r_barrel, r_rod, m_barrel, m_rod, ys=(0.0,),
               barrel_mat="body"):
    """Barrel link (+x from the base pin) and rod link (+x from the eye toward the base)."""
    barrel_len = l_min - 0.10
    rod_len = min(max(l_max - barrel_len + 0.12, 0.35), l_min - 0.05)
    bs, rs = [], []
    for y in ys:
        bs += [cyl(r_barrel, barrel_len, at=(barrel_len / 2, y, 0.0), axis="x", mat=barrel_mat,
                   collision=False),
               cyl(r_barrel * 1.18, 0.07, at=(barrel_len - 0.035, y, 0.0), axis="x", mat="dark",
                   collision=False),
               cyl(r_barrel * 0.95, r_barrel * 2.2, at=(0.0, y, 0.0), axis="y", mat="dark",
                   collision=False)]
        rs += [cyl(r_rod, rod_len, at=(rod_len / 2, y, 0.0), axis="x", mat="steel",
                   collision=False),
               cyl(r_rod * 1.5, r_rod * 2.8, at=(0.0, y, 0.0), axis="y", mat="steel",
                   collision=False)]
    plural = "s" if len(ys) > 1 else ""
    return (Link(f"{prefix}_cylinder{plural}", m_barrel, bs),
            Link(f"{prefix}_rod{plural}", m_rod, rs))


def _cab() -> list[Shape]:
    """Centre cab: glazed all round, dark pillars, yellow roof (chassis frame)."""
    x0, x1, hw = CAB_X0, CAB_X1, CAB_HW
    cx, lx = (x0 + x1) / 2, x1 - x0
    z0, zr, z1 = agl(1.50), agl(CAB_TOP - 0.15), agl(CAB_TOP)
    sides = (1, -1)
    return [
        # lower body between the rear tyres (floor, consoles) and glazed upper cab volume
        box(lx, 1.36, z0 - agl(1.00), at=(cx, 0.0, (z0 + agl(1.00)) / 2)),
        box(lx, 2 * hw, zr - z0, at=(cx, 0.0, (z0 + zr) / 2), visual=False),
        box(lx, 2 * hw, 0.12, at=(cx, 0.0, z0 + 0.06), mat="accent", collision=False),
        # roof with a dark rim
        box(lx + 0.10, 2 * hw + 0.08, 0.15, at=(cx, 0.0, z1 - 0.075)),
        box(lx + 0.12, 2 * hw + 0.10, 0.04, at=(cx, 0.0, zr + 0.02), mat="accent",
            collision=False),
        # glass: windscreen, doors (both sides), rear window
        box(0.03, 2 * hw - 0.16, zr - z0 - 0.20, at=(x1 + 0.005, 0.0, (z0 + 0.14 + zr) / 2),
            mat="glass", collision=False),
        *[box(lx - 0.18, 0.03, zr - z0 - 0.20, at=(cx, s * (hw + 0.005), (z0 + 0.14 + zr) / 2),
              mat="glass", collision=False) for s in sides],
        box(0.03, 2 * hw - 0.16, zr - z0 - 0.30, at=(x0 - 0.005, 0.0, (z0 + 0.20 + zr) / 2),
            mat="glass", collision=False),
        # corner pillars and door posts
        *[box(0.08, 0.08, zr - z0, at=(x, s * (hw - 0.03), (z0 + zr) / 2), mat="accent",
              collision=False) for x in (x0 + 0.04, x1 - 0.04) for s in sides],
        *[box(0.05, 0.04, zr - z0 - 0.10, at=(cx + 0.05, s * (hw + 0.02), (z0 + zr) / 2),
              mat="accent", collision=False) for s in sides],
        # mirrors, roof work lights (front and rear), beacon
        *[box(0.04, 0.22, 0.04, at=(x1 + 0.04, s * (hw + 0.10), zr - 0.25), mat="accent",
              collision=False) for s in sides],
        *[box(0.04, 0.09, 0.20, at=(x1 + 0.06, s * (hw + 0.21), zr - 0.35), mat="accent",
              collision=False) for s in sides],
        *[box(0.10, 0.16, 0.08, at=(x, s * (hw - 0.15), z1 - 0.10), mat="white",
              collision=False) for x in (x0 - 0.04, x1 + 0.04) for s in sides],
        cyl(0.065, 0.10, at=(x0 + 0.35, 0.45, z1 + 0.05), mat="orange", collision=False),
    ]


def _loader_bucket() -> list[Shape]:
    """GP shovel (bucket frame: origin = hinge pin, +x toward the cutting edge, floor flat)."""
    fl, edge, back, top, lip = SH_FLOOR, SH_EDGE, SH_BACK, SH_TOP, SH_LIP
    yb = BKT_W / 2

    def x_front(z):
        return edge - (edge - lip) * (z - fl) / (top - fl)

    heel = (back + 0.12, fl + 0.025)
    shapes = [
        box(edge - heel[0], BKT_W, 0.05, at=((edge + heel[0]) / 2, 0.0, fl + 0.025)),  # floor
        beam(heel, (back, fl + 0.15), BKT_W, 0.05),                                   # heel
        beam((back, fl + 0.15), (back, top - 0.12), BKT_W, 0.05),                     # back wall
        beam((back, top - 0.12), (back + 0.20, top), BKT_W, 0.05),                    # spill guard
        beam((back + 0.20, top), (lip, top - 0.01), BKT_W, 0.05),                     # top lip
        box(0.10, BKT_W, 0.03, at=(edge - 0.03, 0.0, fl + 0.015), mat="steel"),       # cutting edge
        box(0.05, 1.00, 0.50, at=(back - 0.04, 0.0, 0.10), mat="dark", collision=False),  # back rib
    ]
    for s in (1, -1):
        y = s * (yb - 0.02)
        shapes.append(box(lip - back, 0.04, top - fl, at=((lip + back) / 2, y, (top + fl) / 2)))
        steps = [fl + (top - 0.04 - fl) * i / 5 for i in range(5)]
        for z0, z1 in zip(steps, steps[1:] + [top - 0.04]):
            x1 = x_front(z1)
            if x1 > lip + 0.01:
                shapes.append(box(x1 - lip, 0.04, z1 - z0,
                                  at=((x1 + lip) / 2, y, (z0 + z1) / 2)))
        shapes.append(beam((edge, fl + 0.02), (lip, top - 0.02), 0.05, 0.06, y=y, mat="steel",
                           collision=False))                                          # side cutter
        # hinge lugs (behind the back wall) around the arm noses
        shapes += [box(0.20, 0.05, 0.30, at=(back - 0.06, s * (ARM_Y + dy), 0.02), mat="dark",
                       collision=False) for dy in (-0.09, 0.09)]
    return shapes


def _hoe_bucket() -> list[Shape]:
    """24" digging bucket (frame: origin = pin, +x toward the teeth at q = 0 along the
    dipper, mouth on -z (machine side when digging), curved back on +z)."""
    w = HOE_W
    return [
        beam((-0.02, 0.04), (0.19, 0.28), w, 0.05, mat="dark"),
        beam((0.19, 0.28), (0.52, 0.31), w, 0.05, mat="dark"),
        beam((0.52, 0.31), (0.74, 0.14), w, 0.05, mat="dark"),
        beam((0.74, 0.14), (HOE_R, -0.05), w, 0.05, mat="dark"),
        *[s_ for y in (-w / 2, w / 2) for s_ in (
            beam((0.02, 0.09), (0.77, 0.03), 0.03, 0.16, y=y, mat="dark"),
            beam((0.10, 0.20), (0.66, 0.18), 0.03, 0.10, y=y, mat="dark"),
            beam((0.22, 0.26), (0.54, 0.26), 0.03, 0.06, y=y, mat="dark"))],
        box(0.18, 0.26, 0.16, at=(0.03, 0.0, 0.07), mat="dark"),                    # pin bosses
        box(0.12, 0.22, 0.12, at=(0.12, 0.0, 0.26), mat="dark", collision=False),   # link lugs
        *[box(0.10, 0.06, 0.05, at=(HOE_R + 0.03, y, -0.08), mat="steel", collision=False)
          for y in (-0.22, -0.075, 0.075, 0.22)],                                   # teeth
    ]


# ---------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------

def _lift_hi() -> float:
    """Lift angle that puts the shovel hinge pin at B_MAX_H."""
    r = math.hypot(*B_PIN)
    return math.asin((agl(B_MAX_H) - A_PIN[1]) / r) - _elev(B_PIN)


def build() -> MachineSpec:
    sides = (1, -1)

    # ------------------------------------------------------------------ tool joints
    lift = Joint("loader_arm_joint", "revolute", "chassis", "loader_arm",
                 xyz=(A_PIN[0], 0.0, A_PIN[1]), axis=(0, -1, 0), lower=LIFT_LO,
                 upper=round(_lift_hi(), 3), init=LIFT_INIT, velocity=0.30, role="tool")
    shovel = Joint("loader_bucket_joint", "revolute", "shovel_level", "loader_bucket",
                   axis=(0, -1, 0), lower=BKT_LO, upper=BKT_HI, init=BKT_INIT, velocity=1.15,
                   role="tool")
    swing = Joint("swing_joint", "revolute", "chassis", "swing_frame",
                  xyz=(SWING_X, 0.0, agl(SWING_H)), rpy=(0.0, 0.0, math.pi), axis=(0, 0, 1),
                  lower=-1.5, upper=1.5, init=0.0, velocity=1.0, role="tool")
    boom_j = Joint("boom_joint", "revolute", "swing_frame", "boom",
                   xyz=(FOOT[0], 0.0, FOOT[1]), axis=(0, -1, 0), lower=BOOM_LO, upper=BOOM_HI,
                   init=BOOM_INIT, velocity=1.0, role="tool")
    stick_j = Joint("stick_joint", "revolute", "boom", "stick",
                    xyz=(BOOM_L, 0.0, 0.0), rpy=(0.0, STICK_PITCH, 0.0), axis=(0, -1, 0),
                    lower=STICK_LO, upper=STICK_HI, init=STICK_INIT, velocity=1.0, role="tool")
    hoe_j = Joint("backhoe_bucket_joint", "revolute", "stick", "backhoe_bucket",
                  xyz=(STICK_L, 0.0, 0.0), axis=(0, 1, 0), lower=HOE_LO, upper=HOE_HI,
                  init=HOE_INIT, velocity=1.5, role="tool")
    stab = Joint("left_stabilizer_joint", "prismatic", "chassis", "left_stabilizer",
                 xyz=(STAB_X, STAB_Y, agl(STAB_TOP)), axis=(0, 0, -1), lower=0.0,
                 upper=PAD_H, init=0.0, velocity=0.25, role="tool")
    stab_r = Joint("right_stabilizer_joint", "prismatic", "chassis", "right_stabilizer",
                   xyz=(STAB_X, -STAB_Y, agl(STAB_TOP)), axis=(0, 0, -1), lower=0.0,
                   upper=PAD_H, init=0.0, velocity=0.25, role="follower",
                   follows=(stab.name, 1.0, 0.0))

    # ---------------------------------------------------------------- ram fits
    lift_fit = cylinder_fit(LC_BASE, A_PIN, 0.0, LC_EYE, lift.lower, lift.upper,
                            lift.init, RAM_W_INIT)
    boom_fit = cylinder_fit(BC_BASE, FOOT, 0.0, BC_EYE, boom_j.lower,
                            boom_j.upper, boom_j.init, RAM_W_INIT)
    dip_fit = cylinder_fit(DC_BASE, (BOOM_L, 0.0), -STICK_PITCH, DC_EYE,
                           stick_j.lower, stick_j.upper, stick_j.init, RAM_W_INIT)

    # ----------------------------------------------------------------- chassis
    a_x, a_z = A_PIN
    hood_x0, hood_x1 = CAB_X1, NOSE_X - 0.03
    chassis = Link("chassis", M["chassis"], [
        # main frame rails from the king-post mount to the front counterweight
        box(NOSE_X - 0.05 + 1.00, 0.78, 0.40, at=((NOSE_X - 0.05 - 1.00) / 2, 0.0, agl(0.82))),
        # rear axle + differential, front (steer) axle + differential (0.37 m clearance)
        cyl(0.13, 1.42, axis="y", mat="dark"),
        sphere(0.22, mat="dark"),
        cyl(0.10, 1.40, at=(WB, 0.0, agl(FRONT_R)), axis="y", mat="dark"),
        sphere(0.15, at=(WB, 0.0, agl(0.52)), mat="dark"),
        box(0.40, 0.50, 0.14, at=(WB, 0.0, agl(0.66)), mat="dark"),
        # cab
        *_cab(),
        # rear mudguards over the big tyres (attached to the cab sides)
        *[s_ for s in sides for s_ in (
            box(1.10, 0.50, 0.04, at=(0.0, s * REAR_Y, agl(1.50)), mat="accent", collision=False),
            beam((0.55, agl(1.50)), (0.80, agl(1.08)), 0.50, 0.04, y=s * REAR_Y, mat="accent",
                 collision=False),
            beam((-0.55, agl(1.50)), (-0.80, agl(1.08)), 0.50, 0.04, y=s * REAR_Y, mat="accent",
                 collision=False),
        )],
        # side pods between the axles: toolbox / steps (left), fuel + hydraulic tank (right)
        box(0.80, 0.42, 0.34, at=(1.20, 0.62, agl(0.67)), mat="dark"),
        box(0.80, 0.42, 0.38, at=(1.20, -0.62, agl(0.69))),
        *[box(0.40, 0.10, 0.03, at=(1.05, 0.88, agl(h)), mat="steel", collision=False)
          for h in (0.55, 0.84)],
        # bonnet: engine box, sloping top, grille, lights, louvres, exhaust, pre-cleaner
        box(hood_x1 - hood_x0, HOOD_W, 0.40, at=((hood_x0 + hood_x1) / 2, 0.0, agl(1.18))),
        beam((hood_x0, agl(1.60)), (hood_x1, agl(1.33)), HOOD_W, 0.26),
        box(0.03, HOOD_W - 0.12, 0.34, at=(hood_x1 + 0.015, 0.0, agl(1.20)), mat="dark",
            collision=False),
        *[box(0.035, HOOD_W - 0.16, 0.03, at=(hood_x1 + 0.03, 0.0, agl(1.08 + 0.06 * i)), mat="accent",
              collision=False) for i in range(5)],
        *[box(0.04, 0.14, 0.10, at=(hood_x1, s * (HOOD_W / 2 - 0.10), agl(1.40)), mat="white",
              collision=False)
          for s in sides],
        *[box(0.70, 0.03, 0.20, at=(1.95, s * (HOOD_W / 2 + 0.005), agl(1.25)), mat="accent",
              collision=False)
          for s in sides],
        cyl(0.06, 0.60, at=(1.25, -0.30, agl(1.88)), mat="accent", collision=False),
        cyl(0.09, 0.30, at=(1.25, 0.30, agl(1.78)), mat="accent", collision=False),
        # front counterweight / bumper and lift-ram base brackets
        box(0.22, 1.02, 0.36, at=(NOSE_X - 0.11, 0.0, agl(0.80))),
        *[box(0.22, LC_Y + 0.08 - 0.36, 0.22, at=(LC_BASE[0], s * (LC_Y + 0.08 + 0.36) / 2,
                                                  LC_BASE[1] - 0.04),
              mat="dark", collision=False) for s in sides],
        cyl(0.05, 2 * LC_Y + 0.20, at=(LC_BASE[0], 0.0, LC_BASE[1]), axis="y", mat="steel",
            collision=False),
        # loader towers at the cab front corners carrying the arm pivots
        *[box(0.40, 0.08, 0.84, at=(a_x, s * 0.49, agl(1.42))) for s in sides],
        cyl(0.06, 2 * ARM_Y + 0.20, at=(a_x, 0.0, a_z), axis="y", mat="steel", collision=False),
        # rear frame: stabiliser cross beam, king-post brackets, stabiliser casings
        box(0.20, 2.10, 0.36, at=(STAB_X, 0.0, agl(1.18))),
        # king-post brackets: top and bottom plates reaching back to the slew axis
        box(STAB_X - SWING_X - 0.12, 0.50, 0.10,
            at=((STAB_X + SWING_X + 0.12) / 2, 0.0, agl(SWING_H + 0.60))),
        box(STAB_X - SWING_X, 0.50, 0.10, at=((STAB_X + SWING_X) / 2, 0.0, agl(SWING_H - 0.42)),
            mat="dark"),
        box(0.10, 0.50, 0.30, at=(STAB_X - 0.05, 0.0, agl(SWING_H - 0.27)), mat="dark"),
        *[box(0.22, 0.22, 0.80, at=(STAB_X, s * STAB_Y, agl(STAB_TOP + 0.40))) for s in sides],
        *[cyl(0.05, 0.30, at=(STAB_X, s * STAB_Y, agl(STAB_TOP + 0.95)), mat="steel",
              collision=False) for s in sides],                                    # leg rams
        *[box(0.03, 0.14, 0.10, at=(STAB_X - 0.125, s * STAB_Y, agl(1.30)), mat="red",
              collision=False) for s in sides],                                    # tail lights
    ])

    # ------------------------------------------------------------- loader arm
    # The colliding arm section stops 0.50 m short of the hinge pin; the nose that
    # sits between the shovel's hinge lugs is visual only.
    b_end = _lerp(B_PIN, ARM_KNEE, 0.50 / math.hypot(*_sub(ARM_KNEE, B_PIN)))
    arm_shapes = []
    for s in sides:
        y = s * ARM_Y
        arm_shapes += [
            beam((-0.10, 0.04), ARM_KNEE, ARM_W, 0.30, y=y),
            beam(ARM_KNEE, b_end, ARM_W, 0.26, y=y),
            beam(b_end, B_PIN, ARM_W, 0.20, y=y, collision=False),
            cyl(0.15, ARM_W, at=(ARM_KNEE[0], y, ARM_KNEE[1]), axis="y"),
            cyl(0.16, ARM_W + 0.04, axis="y", at=(0.0, y, 0.0), collision=False),
            cyl(0.12, ARM_W + 0.02, axis="y", at=(B_PIN[0], y, B_PIN[1]), collision=False),
            cyl(0.06, ARM_W + 0.10, axis="y", at=(B_PIN[0], y, B_PIN[1]), mat="steel",
                collision=False),
            box(0.22, ARM_Y - LC_Y + 0.10, 0.48,
                at=(LC_EYE[0], s * (ARM_Y + LC_Y) / 2, LC_EYE[1] + 0.20), mat="dark",
                collision=False),                                                    # lift-ram drop lug
        ]
        # shovel (tilt) ram outboard of each arm: from a lug behind the knee to the
        # tipping-lever plate on the arm nose (the levelling linkage is not modelled)
        t0, t1 = TILT_BASE, TILT_LEVER
        tm = _lerp(t0, t1, 0.62)
        ty = s * TILT_Y
        arm_shapes += [
            box(0.22, TILT_Y - ARM_Y + 0.08, 0.16, at=(t0[0], s * (ARM_Y + TILT_Y) / 2, t0[1] - 0.06),
                mat="dark", collision=False),
            _rod(t0, tm, 0.07, y=ty, mat="dark"),
            cyl(0.08, 0.06, at=(tm[0], ty, tm[1]), rpy=(0.0, math.atan2(t1[0] - t0[0], t1[1] - t0[1]), 0.0),
                mat="steel", collision=False),
            _rod(tm, t1, 0.04, y=ty),
            cyl(0.06, TILT_Y - ARM_Y + 0.10, at=(t1[0], s * (ARM_Y + TILT_Y) / 2, t1[1]), axis="y",
                mat="steel", collision=False),
            beam(t1, B_PIN, 0.05, 0.20, y=s * (TILT_Y - 0.06), mat="dark", collision=False),
        ]
    loader_arm = Link("loader_arm", M["arm"], arm_shapes)

    # ---------------------------------------------------------- swing frame + backhoe
    swing_frame = Link("swing_frame", M["swing"], [
        box(0.32, 0.44, 0.84, at=(0.10, 0.0, 0.07)),                                # casting
        cyl(0.10, 0.08, at=(0.0, 0.0, 0.53), mat="steel", collision=False),          # king pins
        cyl(0.10, 0.08, at=(0.0, 0.0, -0.39), mat="steel", collision=False),
        box(0.24, 0.40, 0.26, at=(FOOT[0], 0.0, FOOT[1]), mat="dark"),               # foot clevis
        cyl(0.07, 0.48, at=(FOOT[0], 0.0, FOOT[1]), axis="y", mat="steel", collision=False),
        box(0.20, 0.30, 0.16, at=(BC_BASE[0] + 0.06, 0.0, BC_BASE[1]), mat="dark",
            collision=False),
        # slew rams either side of the casting (visual)
        *[_rod((-0.05, -0.05), (0.30, -0.05), 0.06, y=y, mat="dark") for y in (-0.30, 0.30)],
    ])

    boom = Link("boom", M["boom"], [
        beam((0.0, 0.0), BOOM_KNEE, 0.30, 0.36),
        beam(BOOM_KNEE, (BOOM_L, 0.0), 0.26, 0.30),
        cyl(0.17, 0.30, at=(BOOM_KNEE[0], 0.0, BOOM_KNEE[1]), axis="y"),
        cyl(0.14, 0.36, axis="y", mat="steel", collision=False),
        cyl(0.11, 0.34, at=(BOOM_L, 0.0, 0.0), axis="y", mat="steel", collision=False),
        box(0.30, 0.20, 0.30, at=(DC_BASE[0], 0.0, DC_BASE[1] - 0.12), mat="dark",
            collision=False),                                                       # dipper-ram bracket
        box(0.24, 0.24, 0.14, at=(BC_EYE[0], 0.0, BC_EYE[1] + 0.07), mat="dark",
            collision=False),                                                       # boom-ram lug
    ])

    stick = Link("stick", M["stick"], [
        beam((0.0, 0.0), (STICK_L, 0.0), 0.22, 0.28),
        beam((-STICK_TAIL, 0.0), (0.0, 0.0), 0.20, 0.24),
        cyl(0.12, 0.32, axis="y", mat="steel", collision=False),
        cyl(0.08, 0.28, at=(-STICK_TAIL, 0.0, 0.0), axis="y", mat="steel", collision=False),
        cyl(0.09, 0.30, at=(STICK_L, 0.0, 0.0), axis="y", mat="steel", collision=False),
        # bucket ram on the back of the dipper + dogbone links (visual)
        _rod((0.15, 0.20), (1.15, 0.19), 0.07, mat="body"),
        _rod((1.10, 0.19), (1.62, 0.18), 0.04),
        *[_rod((1.62, 0.18), (1.86, 0.07), 0.03, y=y, mat="dark") for y in (-0.10, 0.10)],
        box(0.20, 0.16, 0.12, at=(0.15, 0.0, 0.14), mat="dark", collision=False),
    ])

    def leg(name):
        """Sliding inner leg + foot pad (frame origin = casing bottom, pad below it)."""
        pad_z = PAD_H - STAB_TOP                                    # pad underside
        return Link(name, M["leg"], [
            box(0.16, 0.16, 0.86, at=(0.0, 0.0, pad_z + 0.07 + 0.43)),               # inner leg
            box(0.40, 0.32, 0.07, at=(0.0, 0.0, pad_z + 0.035), mat="dark"),          # foot pad
            box(0.10, 0.30, 0.10, at=(0.0, 0.0, pad_z + 0.12), mat="dark", collision=False),
        ])

    # ------------------------------------------------------------- rams
    (lkb, lcb), (lkr, lcr), _, l_min, l_max = lift_fit
    lift_cyl, lift_rod = _ram_links("lift", l_min, l_max, 0.075, 0.045, 60.0, 35.0,
                                    ys=(LC_Y, -LC_Y), barrel_mat="dark")
    (bkb, bcb), (bkr, bcr), _, b_min, b_max = boom_fit
    boom_cyl, boom_rod = _ram_links("boom", b_min, b_max, 0.085, 0.05, 55.0, 30.0)
    (dkb, dcb), (dkr, dcr), _, d_min, d_max = dip_fit
    dip_cyl, dip_rod = _ram_links("dipper", d_min, d_max, 0.075, 0.045, 45.0, 25.0)

    # ------------------------------------------------------------- wheels
    wheel_links, wheel_joints, knuckles, steer_joints = [], [], [], []
    for side, s in (("left", 1.0), ("right", -1.0)):
        kn = f"front_{side}_knuckle"
        knuckles.append(Link(kn, M["knuckle"], [
            cyl(0.11, 0.14, at=(0.0, -s * 0.22, 0.0), axis="y", mat="dark"),
            cyl(0.07, 0.30, at=(0.0, -s * 0.26, 0.0), mat="dark", collision=False),
        ]))
        steer_joints.append(Joint(f"front_{side}_steer_joint", "revolute", "chassis", kn,
                                  xyz=(WB, s * FRONT_Y, agl(FRONT_R)), axis=(0, 0, 1),
                                  lower=-0.80, upper=0.80, velocity=1.0, role="steer"))
        fw, rw = f"front_{side}_wheel", f"rear_{side}_wheel"
        wheel_links += [wheel_link(fw, FRONT_R, FRONT_W, M["front_wheel"]),
                        wheel_link(rw, REAR_R, REAR_W, M["rear_wheel"])]
        wheel_joints += [wheel_joint(f"{fw}_joint", kn, fw, (0.0, 0.0, 0.0), FRONT_R),
                         wheel_joint(f"{rw}_joint", "chassis", rw, (0.0, s * REAR_Y, 0.0), REAR_R)]

    links = [chassis, loader_arm,
             Link("shovel_level", 30.0, [cyl(0.06, 2 * ARM_Y + 0.24, axis="y", mat="steel",
                                             collision=False)]),
             Link("loader_bucket", M["shovel"], _loader_bucket()), lift_cyl, lift_rod,
             swing_frame, boom, stick, Link("backhoe_bucket", M["hoe"], _hoe_bucket()),
             boom_cyl, boom_rod, dip_cyl, dip_rod,
             leg("left_stabilizer"), leg("right_stabilizer"),
             *knuckles, *wheel_links]

    joints = [
        lift,
        _follower("shovel_level_joint", "loader_arm", "shovel_level", B_PIN, lift, -LEVEL, 0.0),
        shovel,
        _follower("lift_cylinder_joint", "chassis", "lift_cylinders", LC_BASE, lift, lkb, lcb),
        _follower("lift_rod_joint", "loader_arm", "lift_rods", LC_EYE, lift, lkr, lcr),
        swing, boom_j, stick_j, hoe_j,
        _follower("boom_cylinder_joint", "swing_frame", "boom_cylinder", BC_BASE, boom_j, bkb, bcb),
        _follower("boom_rod_joint", "boom", "boom_rod", BC_EYE, boom_j, bkr, bcr),
        _follower("dipper_cylinder_joint", "boom", "dipper_cylinder", DC_BASE, stick_j, dkb, dcb),
        _follower("dipper_rod_joint", "stick", "dipper_rod", DC_EYE, stick_j, dkr, dcr),
        stab, stab_r,
        *steer_joints, *wheel_joints,
    ]

    return MachineSpec(
        name="backhoe_loader",
        abbrev="BHL",
        display_name="Backhoe Loader",
        category="excavation",
        reference="8 t backhoe loader (JCB 3CX / CAT 420 class)",
        description=("Rigid-frame backhoe loader on big 16.9-28 rear and small 12.5/80-18 "
                     "front tyres with front-wheel Ackermann steering, a centre cab, a "
                     "self-levelling front loader with a 1.0 m³ shovel, a centre-mount rear "
                     "backhoe (180° slew, boom / dipper / 24\" bucket) and two vertical "
                     "rear stabiliser legs."),
        root="chassis",
        links=links,
        joints=joints,
        drive=Drive(type="ackermann", max_speed=11.1, wheelbase=WB, max_steer=0.60,
                    steer_rate=0.16, fixed_axle_x=0.0, steer_axle="front"),
        tools=[
            Tool("loader_arm_joint", "loader_arm", positive="raise", negative="lower"),
            Tool("loader_bucket_joint", "loader_bucket", positive="curl", negative="dump"),
            Tool("swing_joint", "backhoe_swing", positive="left", negative="right"),
            Tool("boom_joint", "backhoe_boom", positive="raise", negative="lower"),
            Tool("stick_joint", "backhoe_stick", positive="out", negative="in"),
            Tool("backhoe_bucket_joint", "backhoe_bucket", positive="curl", negative="dump"),
            Tool("left_stabilizer_joint", "stabilizers", positive="lower", negative="raise"),
        ],
        cameras=[
            # operator eye height, just ahead of the (opaque) windscreen, over the bonnet
            Camera("front", "chassis", (CAB_X1 + 0.07, 0.0, agl(2.32)), (0.0, 0.30, 0.0)),
            # outside the front cab corners under the roof rim, looking outward-forward
            Camera("left", "chassis", (CAB_X1 + 0.06, CAB_HW + 0.10, agl(2.64)), (0.0, 0.40, 0.55)),
            Camera("right", "chassis", (CAB_X1 + 0.06, -CAB_HW - 0.10, agl(2.64)), (0.0, 0.40, -0.55)),
        ],
        envelope=(5.62, 2.36, 3.61),
        operating_mass=8070.0,
        notes=("loader_bucket / backhoe_bucket curl-dump are not auto-verified: +1 on "
               "loader_bucket racks the shovel's cutting edge up and back (rollback), +1 on "
               "backhoe_bucket swings the teeth toward the machine (see the sweep render). "
               "backhoe_swing +1 = CCW seen from above. The shovel self-levels through a "
               "follower (-1.0 x lift relative to the arms); ram barrels/rods are visual-only "
               "followers; the right stabiliser follows the left. Stabiliser pads stop at "
               "grade (the kinematic base cannot be jacked up)."),
    )
