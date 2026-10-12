"""Telehandler (TH), 17 m / 4 t class — modelled on a JCB 540-170 Loadall (CAT TH417D class).

Datasheet numbers used (JCB 540-170 static-dimension / performance sheet, rounded):
    max lift height 16.7 m · max forward reach 12.5 m · max payload 4.0 t ·
    operating mass 12,060 kg (class 11.2-12.3 t) · overall length 6.36 m to the
    carriage face, 5.08 m to the front tyres (7.56 m with 1.2 m pallet forks) ·
    overall width 2.44 m over tyres · overall height 2.69 m (cab roof) ·
    wheelbase 2.75 m · stabiliser width deployed 3.55 m · 4-speed powershift.
    (The brief named the CAT TH514D, which is CAT's 5 t / 14 m machine; the
    4 t / 17 m CAT equivalent of the 540-170 is the TH417D.)
Assumed / typical for the class:
    ≈ 30 km/h top speed · 500/70 R24 tyres (Ø 1.31 m, 0.50 m section) → track 1.94 m · ground
    clearance ≈ 0.40 m · JCB side-engine layout: cab on the left, boom pivot at
    the rear (1.75 m above the ground, 1.30 m behind the rear axle) with the
    boom running down the right-hand side of the cab, engine bay on the right
    under the boom · box-frame side rails ending behind the front tyres and a
    narrow (0.50 m) front nose carrying the steer axle and the folded
    stabilisers ahead of the tyres, so the steered tyres clear everything at
    full lock · four-section boom (base + 3 telescopic sections, rope/chain
    synchronised so each strokes a third of the total), 5.57 m retracted /
    16.77 m extended pivot-to-carriage-pin (11.2 m total extension), carriage
    pin in a rounded drop nose 0.30 m below the boom axis · single lift ram
    between a chassis lug behind the rear axle and the boom belly (2.49 →
    3.47 m pin to pin) · automatic carriage compensation (the master/slave
    levelling rams keep the fork angle constant while luffing) · quick-hitch
    carriage with two side plates straddling the boom nose · folded front
    stabilisers (not driven) · front-wheel Ackermann steer (the real machine
    also has 4WS / crab modes).
Model check (offline FK): fork underside 0.56 m above the ground in the travel
    pose (boom fully lowered and retracted) and 0.03 m at full extension with
    the boom fully lowered, so no luff / telescope combination puts the
    carriage below grade; fork underside 16.70 m above the ground at boom
    69.2° / full extension (forks level); 12.50 m reach from the front tyres
    to the fork heel at boom 0° / full extension; carriage exactly level at
    every luff angle with fork_tilt = 0 · lift ram ≈ 0.6 m minimum lever arm
    about the foot pin over the whole luff range (barrel / rod are linear-fit
    followers, ≤ 8 cm lateral pin miss) · inner / outer front wheel 46.1° /
    30.9° at max_steer 0.65 rad (2WS turning radius ≈ 5.6 m over the outer
    tyre); no tyre / mudguard / chassis overlap anywhere in the lock range ·
    no carriage / boom overlap over the luff × telescope × tilt range ·
    travel-pose envelope 7.63 × 2.49 × 2.69 m.

Joint ranges and rates at full command:
    boom_luff       -0.048 (forks reach grade at full extension) … +1.208 rad
                    (69.2°), 0.14 rad/s (≈ 9 s boom up, the class figure is 9-10 s)
    boom_telescope  0 … 11.20 m TOTAL extension (the fly section; the two
                    intermediate sections follow at 1/3 and 2/3), 0.9 m/s
                    (≈ 12.4 s extend; real ≈ 13-16 s)
    fork_tilt       -0.95 (tilt forward 54°) … +0.40 rad (tilt back 23°) relative to
                    level (compensated), 0.45 rad/s; +1 = tilt back (fork tips up).
                    Relative to the boom that is -124° … +26° (≈ 150° of carriage
                    rotation, class figure ≈ 140-150°).

Drive: Ackermann, front-wheel steer, rear (fixed) axle at x = 0, max_speed
8.5 m/s (30.6 km/h) both ways (powershift shuttle), max_steer 0.65 rad,
steer rate 0.38 rad/s (1.7 s centre to full lock, ≈ 3.4 s lock to lock).
The environment limits the lock at speed (drive.max_lat_accel).

Init pose = travel pose: boom fully lowered and retracted, forks level-ish
(tilted back 3°) at a 0.56 m carry height.  (Lower would let a fully lowered
boom bury the carriage up to 1.3 m when extended; the env cannot limit the
luff by the extension.)

Root frame: chassis, origin at the rear-axle centre (axle at z = 0, ground at
z = -WHEEL_R), +x forward (towards the forks), +y left (cab side).
"""

from __future__ import annotations

import math

from ..spec import (Camera, Drive, Joint, Link, MachineSpec, Shape, Tool, beam, box, cyl,
                    sphere, wheel_joint, wheel_link)

# --- running gear ------------------------------------------------------------------------
WHEEL_R, TYRE_W = 0.655, 0.50          # 500/70 R24
WB = 2.75                              # wheelbase, rear axle at x = 0
Y_W = 0.97                             # wheel centre → track 1.94 m, 2.44 m over tyres
REAR_X = -1.675                        # rear face of the counterweight (5.08 m to front tyres)
MAX_STEER = 0.65                       # virtual centre-wheel lock → 46.1° inner / 30.9° outer
# Front chassis kept out of the steered tyres' sweep (each tyre swings 0.70 m about its
# steer axis in plan view, reaching |y| = 0.33 behind the axle and x = 3.41 ahead of it).
FRAME_X0, FRAME_X1 = -1.10, 2.05       # side rails end behind the front tyres
NOSE_HW, NOSE_X1 = 0.25, 3.70          # narrow front nose (|y| ≤ 0.25) carries axle + stabilisers
AXLE_HL = 0.42                         # front axle housing half length (kingpins in the rims)
STAB_X = 3.56                          # folded stabilisers, ahead of the front tyres' sweep


def agl(h: float) -> float:
    """Height above the ground → root-frame z."""
    return h - WHEEL_R


# --- boom (boom frame: origin = foot pin, +x along the boom axis) -------------------------
BOOM_Y = -0.12                         # boom centre plane (right of the cab)
PIVOT = (-1.30, agl(1.75))             # foot pin (x, z) in the root frame
BASE_SEC, MID1_SEC, MID2_SEC, FLY_SEC = (0.50, 0.60), (0.42, 0.50), (0.34, 0.41), (0.27, 0.33)
WALL = 0.035
BASE_X0, BASE_X1 = -0.35, 4.50         # base section extent (tail behind the foot pin)
MID1_X0, MID1_LEN = -0.10, 4.80        # section origins (retracted) in the parent section
MID2_X0, MID2_LEN = 0.10, 4.90
FLY_X0, FLY_LEN = 0.10, 4.95
SEC_X0 = MID1_X0 + MID2_X0 + FLY_X0    # retracted fly origin ahead of the foot pin
DROP = 0.30                            # carriage pin below the boom axis (drop nose)
NOSE_R = 0.20                          # rounded boom-head nose around the carriage pin
HEEL_X = 0.44                          # fork-shank front face ahead of the carriage pin
FORK_DROP = 0.62                       # fork underside below the carriage pin (level)
FORK_LEN = 1.20
FORK_GRADE = 0.03                      # level-fork underside above the ground at FULL
                                       # extension with the boom fully lowered
CARRIAGE_FACE_X = REAR_X + 6.36        # overall length to the carriage face
FRONT_TYRE_X = WB + WHEEL_R
LIFT_HEIGHT, REACH = 16.7, 12.5

# Lift ram: base lug on the chassis behind the rear axle, eye under the boom belly.
RAM_BASE = (-0.90, agl(1.00))
RAM_EYE = (2.90, -0.40)                # boom frame

# --- body --------------------------------------------------------------------------------
CAB_Y0, CAB_Y1 = 0.17, 1.13            # cab inner (boom side) / outer wall
CAB_X0, CAB_X1 = 0.62, 2.12            # glazed upper cab
CAB_LX0, CAB_LX1 = 0.74, 2.02          # lower cab between the tyres
CAB_FLOOR, CAB_BELT, CAB_ROOF = 0.82, 1.40, 2.58
ENG_Y0, ENG_Y1 = -1.15, -0.45          # engine bay (right side)
RCAM = (CAB_LX1 - 0.04, ENG_Y0 - 0.10, agl(1.95))   # right camera, outboard of the mudguard

# --- masses (kg) ---------------------------------------------------------------------------
M = dict(chassis=7180.0, wheel=280.0, knuckle=60.0, base=1300.0, mid1=700.0, mid2=520.0,
         fly=430.0, level=20.0, carriage=420.0, barrel=160.0, rod=90.0)


# -----------------------------------------------------------------------------------------
# Planar helpers: (x, z) pairs, rotation about -y (positive raises +x)
# -----------------------------------------------------------------------------------------

def _rot(p, q):
    c, s = math.cos(q), math.sin(q)
    return (c * p[0] - s * p[1], s * p[0] + c * p[1])


def _elev(v):
    return math.atan2(v[1], v[0])


def _luff_for_pin_height(length: float, dz: float) -> float:
    """Boom angle that puts the pin (length, -DROP) at height dz above the foot pin."""
    return math.asin(dz / math.hypot(length, DROP)) - math.atan2(-DROP, length)


def _extended_len() -> float:
    """Foot pin → carriage pin at full extension, from the datasheet reach (boom at 0°,
    fork heel 12.5 m ahead of the front tyres)."""
    return REACH + FRONT_TYRE_X - PIVOT[0] - HEEL_X


def _luff_lo() -> float:
    """Lowest boom angle: level forks FORK_GRADE above the ground at FULL extension, so
    no telescope position can push the carriage into the ground."""
    return _luff_for_pin_height(_extended_len(), agl(FORK_GRADE + FORK_DROP) - PIVOT[1])


def _retracted_len() -> float:
    """Foot pin → carriage pin, retracted, so the travel pose (boom fully lowered) matches
    the datasheet length to the carriage face."""
    q = _luff_lo()
    return (CARRIAGE_FACE_X - HEEL_X - PIVOT[0] - math.sin(q) * DROP) / math.cos(q)


def _pin_x_fly() -> float:
    """Carriage-pin x in the fly frame."""
    return _retracted_len() - SEC_X0


def _stroke() -> float:
    """Per-section stroke (three equal telescopic sections)."""
    return (_extended_len() - _retracted_len()) / 3.0


def _ram_fit(lo: float, hi: float, n: int = 61):
    """Lift ram barrel elevation (root frame) as a Chebyshev line k*q + c over [lo, hi].

    Returns (k, c), the worst lateral pin miss (m) and the min / max pin distance.
    """
    def geo(q):
        e = _rot(RAM_EYE, q)
        e = (PIVOT[0] + e[0], PIVOT[1] + e[1])
        v = (e[0] - RAM_BASE[0], e[1] - RAM_BASE[1])
        return _elev(v), math.hypot(*v)
    qs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    th, lens = zip(*[geo(q) for q in qs])

    def err(k):
        rs = [t - k * q for q, t in zip(qs, th)]
        c = (max(rs) + min(rs)) / 2
        return max(ln * abs(r - c) for r, ln in zip(rs, lens)), c
    a, b = -3.0, 3.0
    for _ in range(80):
        k1, k2 = a + (b - a) / 3, b - (b - a) / 3
        a, b = (k1, b) if err(k1)[0] > err(k2)[0] else (a, k2)
    k = (a + b) / 2
    miss, c = err(k)
    return (k, c), miss, min(lens), max(lens)


def _follower(name, parent, child, xz, src: Joint, k, c, y=0.0) -> Joint:
    """Revolute follower whose child +x elevation (about -y) in the parent is k*q_src + c."""
    q_mid = (src.lower + src.upper) / 2
    base = k * q_mid + c
    c0 = -k * q_mid
    lo, hi = sorted((k * src.lower + c0, k * src.upper + c0))
    return Joint(name, "revolute", parent, child, xyz=(xz[0], y, xz[1]),
                 rpy=(0.0, -base, 0.0), axis=(0, -1, 0), lower=lo - 0.02, upper=hi + 0.02,
                 init=k * src.init + c0, velocity=abs(k) * src.velocity + 0.05,
                 role="follower", follows=(src.name, k, c0))


# -----------------------------------------------------------------------------------------
# Shapes
# -----------------------------------------------------------------------------------------

def _rod(p0, p1, radius, y=0.0, mat="steel") -> Shape:
    """Visual-only cylinder from p0 to p1 in the link's x-z plane."""
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    return cyl(radius, math.hypot(dx, dz), at=((p0[0] + p1[0]) / 2, y, (p0[1] + p1[1]) / 2),
               rpy=(0.0, math.atan2(dx, dz), 0.0), mat=mat, collision=False)


def _tube(x0: float, x1: float, sec: tuple[float, float], mat: str = "body") -> list[Shape]:
    """Hollow rectangular boom section from x0 to x1 (four colliding plates)."""
    w, h = sec
    lx, cx = x1 - x0, (x0 + x1) / 2
    return [
        box(lx, w, WALL, at=(cx, 0.0, h / 2 - WALL / 2), mat=mat),
        box(lx, w, WALL, at=(cx, 0.0, -h / 2 + WALL / 2), mat=mat),
        box(lx, WALL, h - 2 * WALL, at=(cx, w / 2 - WALL / 2, 0.0), mat=mat),
        box(lx, WALL, h - 2 * WALL, at=(cx, -w / 2 + WALL / 2, 0.0), mat=mat),
    ]


def _collar(x: float, sec: tuple[float, float], mat: str = "dark") -> list[Shape]:
    """Visual wear-pad collar around the mouth of a boom section."""
    w, h = sec
    t = 0.025
    return [
        box(0.10, w + 2 * t, t, at=(x, 0.0, h / 2 + t / 2), mat=mat, collision=False),
        box(0.10, w + 2 * t, t, at=(x, 0.0, -h / 2 - t / 2), mat=mat, collision=False),
        box(0.10, t, h, at=(x, w / 2 + t / 2, 0.0), mat=mat, collision=False),
        box(0.10, t, h, at=(x, -w / 2 - t / 2, 0.0), mat=mat, collision=False),
    ]


def _cab() -> list[Shape]:
    """Left-hand cab: yellow lower body between the tyres, glazed upper cab, roof."""
    x0, x1, y0, y1 = CAB_X0, CAB_X1, CAB_Y0, CAB_Y1
    cx, lx, cy, ly = (x0 + x1) / 2, x1 - x0, (y0 + y1) / 2, y1 - y0
    zf, zb, zr = agl(CAB_FLOOR), agl(CAB_BELT), agl(CAB_ROOF)
    zm = (zb + zr) / 2
    corners = [(x, y) for x in (x0 + 0.04, x1 - 0.04) for y in (y0 + 0.04, y1 - 0.04)]
    return [
        # lower body (floor, consoles) between the rear and front tyres
        box(CAB_LX1 - CAB_LX0, ly, zb - zf, at=((CAB_LX0 + CAB_LX1) / 2, cy, (zf + zb) / 2)),
        # glazed upper volume: collision only, drawn as pillars + glass
        box(lx, ly, zr - zb, at=(cx, cy, zm), visual=False),
        box(lx, ly, 0.06, at=(cx, cy, zb + 0.03), mat="accent", collision=False),
        *[box(0.08, 0.08, zr - zb, at=(x, y, zm), mat="accent", collision=False)
          for x, y in corners],
        box(0.06, 0.05, zr - zb, at=(x0 + 0.62, y1 - 0.02, zm), mat="accent", collision=False),
        # roof (yellow) with a dark rim, roof window, work lights, beacon
        box(lx + 0.10, ly + 0.06, 0.08, at=(cx, cy, zr + 0.04)),
        box(lx + 0.12, ly + 0.08, 0.03, at=(cx, cy, zr + 0.015), mat="accent", collision=False),
        box(0.80, ly - 0.30, 0.01, at=(cx + 0.15, cy, zr + 0.085), mat="glass", collision=False),
        *[box(0.06, 0.16, 0.08, at=(x1 + 0.05, y, zr - 0.02), mat="white", collision=False)
          for y in (y0 + 0.14, y1 - 0.14)],
        box(0.06, 0.12, 0.06, at=(x0 - 0.04, y1 - 0.14, zr - 0.02), mat="white", collision=False),
        cyl(0.055, 0.03, at=(x0 + 0.25, y1 - 0.20, zr + 0.095), mat="orange", collision=False),
        # glass: windscreen, door (left), side window facing the boom, rear window
        box(0.025, ly - 0.14, zr - zb - 0.10, at=(x1 + 0.005, cy, zm), mat="glass",
            collision=False),
        box(0.025, ly - 0.10, 0.36, at=(CAB_LX1 + 0.005, cy, zb - 0.22), mat="glass",
            collision=False),                                                  # lower front pane
        box(lx - 0.16, 0.025, zr - zb - 0.08, at=(cx, y1 + 0.005, zm), mat="glass",
            collision=False),
        box(0.75, 0.025, 0.42, at=(CAB_LX0 + 0.62, y1 + 0.005, zb - 0.25), mat="glass",
            collision=False),                                                  # door lower glass
        box(lx - 0.16, 0.025, zr - zb - 0.12, at=(cx, y0 - 0.005, zm), mat="glass",
            collision=False),
        box(0.025, ly - 0.14, zr - zb - 0.30, at=(x0 - 0.005, cy, zm + 0.08), mat="glass",
            collision=False),
        # door handle, mirror arm + mirror on the front pillar, entry steps + grab handle
        box(0.10, 0.03, 0.03, at=(CAB_LX0 + 0.20, y1 + 0.02, zb + 0.05), mat="steel",
            collision=False),
        box(0.04, 0.06, 0.04, at=(x1 - 0.02, y1 + 0.03, zr - 0.30), mat="accent", collision=False),
        box(0.05, 0.04, 0.22, at=(x1 + 0.02, y1 + 0.06, zr - 0.40), mat="accent", collision=False),
        *[box(0.40, 0.20, 0.03, at=(CAB_LX0 + 0.62, y1 - 0.05, agl(h)), mat="steel",
              collision=False) for h in (0.48, 0.72)],
        *[box(0.03, 0.20, 0.40, at=(CAB_LX0 + 0.62 + dx, y1 - 0.05, agl(0.65)), mat="accent",
              collision=False) for dx in (-0.215, 0.215)],                    # step hangers
        box(0.03, 0.04, 0.70, at=(CAB_LX0 + 0.30, y1 + 0.03, agl(1.30)), mat="steel",
            collision=False),
    ]


def _engine_bay() -> list[Shape]:
    """Right-hand side engine bay between the wheels, cooling pack behind the right
    rear wheel, hydraulic tank behind the left rear wheel."""
    y0, y1 = ENG_Y0, ENG_Y1
    cy, ly = (y0 + y1) / 2, y1 - y0
    ex0, ex1 = CAB_LX0, CAB_LX1
    return [
        box(ex1 - ex0, ly, 0.86, at=((ex0 + ex1) / 2, cy, agl(0.93))),             # engine cover
        beam((ex0 + 0.05, agl(1.36)), (ex1 - 0.05, agl(1.36)), ly - 0.10, 0.12, y=cy),  # top
        box(0.90, 0.02, 0.48, at=((ex0 + ex1) / 2, y0 - 0.01, agl(1.02)), mat="accent",
            collision=False),                                                      # side grille
        *[box(0.86, 0.025, 0.025, at=((ex0 + ex1) / 2, y0 - 0.015, agl(0.84 + 0.08 * i)),
              mat="dark", collision=False) for i in range(6)],
        box(0.30, 0.02, 0.18, at=(ex1 - 0.25, y0 - 0.01, agl(0.66)), mat="dark",
            collision=False),                                                      # access panel
        cyl(0.05, 0.42, at=(ex1 - 0.20, y0 + 0.30, agl(1.62)), mat="accent", collision=False),
        cyl(0.08, 0.10, at=(ex0 + 0.25, y0 + 0.25, agl(1.47)), mat="dark", collision=False),
        # cooling pack (rear right) with its grille, hydraulic / fuel tank (rear left)
        box(0.36, ly, 0.86, at=(-0.90, cy, agl(0.93))),
        box(0.30, 0.02, 0.60, at=(-0.90, y0 - 0.01, agl(0.96)), mat="accent", collision=False),
        box(0.36, 0.80, 0.70, at=(-0.90, 0.72, agl(0.92)), mat="dark"),
        cyl(0.06, 0.05, at=(-0.90, 0.95, agl(1.29)), mat="accent", collision=False),
    ]


def _counterweight_and_pivot() -> list[Shape]:
    px, pz = PIVOT
    sides = (1, -1)
    return [
        # cast counterweight across the rear, rounded rear corners
        box(0.50, 2.20, 0.85, at=(-1.35, 0.0, agl(0.875))),
        box(0.08, 1.80, 0.85, at=(REAR_X + 0.04, 0.0, agl(0.875))),
        *[cyl(0.20, 0.85, at=(REAR_X + 0.20, s * 0.90, agl(0.875))) for s in sides],
        box(0.14, 1.70, 0.16, at=(REAR_X + 0.05, 0.0, agl(0.53)), mat="accent"),  # bumper
        cyl(0.05, 0.10, at=(REAR_X + 0.01, 0.0, agl(0.70)), axis="x", mat="steel",
            collision=False),                                                     # tow pin
        *[box(0.02, 0.18, 0.10, at=(REAR_X - 0.005, s * 0.80, agl(1.12)), mat="red",
              collision=False) for s in sides],
        *[box(0.02, 0.10, 0.08, at=(REAR_X - 0.005, s * 0.62, agl(1.12)), mat="orange",
              collision=False) for s in sides],
        box(0.02, 0.50, 0.12, at=(REAR_X - 0.005, -0.45, agl(0.82)), mat="white",
            collision=False),                                                     # number plate
        # boom pivot tower: two side plates straddling the boom on the counterweight
        *[beam((px - 0.32, agl(1.30)), (px + 0.45, agl(1.30)), 0.06, 0.02, y=BOOM_Y + s * 0.30,
               mat="body") for s in sides],
        *[box(0.62, 0.06, pz - agl(1.30) + 0.22, at=(px + 0.05, BOOM_Y + s * 0.30,
                                                      (agl(1.30) + pz + 0.22) / 2))
          for s in sides],
        cyl(0.11, 0.70, at=(px, BOOM_Y, pz), axis="y", mat="steel", collision=False),
    ]


def _chassis_frame() -> list[Shape]:
    x0, x1 = FRAME_X0, FRAME_X1
    cx, lx = (x0 + x1) / 2, x1 - x0
    nx0 = x1 - 0.10
    sides = (1, -1)
    return [
        # box frame: two side rails with the boom (and lift ram) nesting between them; they
        # end behind the front tyres, which sweep to |y| = 0.33 at full lock
        *[box(lx, 0.08, 0.50, at=(cx, BOOM_Y + s * 0.33, agl(0.67)), mat="dark") for s in sides],
        box(lx, 0.58, 0.08, at=(cx, BOOM_Y, agl(0.46)), mat="dark"),
        # narrow front nose (symmetric, under the lowered boom) carrying the front axle and
        # the stabiliser beam
        box(NOSE_X1 - nx0, 2 * NOSE_HW, 0.44, at=((nx0 + NOSE_X1) / 2, 0.0, agl(0.70)), mat="dark"),
        # axles (0.40 m ground clearance under the differentials); the steer-axle housing
        # stops short of the tyres' sweep, the knuckle hubs bridge the gap
        cyl(0.15, 1.44, axis="y", mat="dark"),
        sphere(0.24, at=(0.0, 0.0, -0.01), mat="dark"),
        *[cyl(0.20, 0.14, at=(0.0, s * (Y_W - 0.32), 0.0), axis="y", mat="dark") for s in sides],
        cyl(0.15, 2 * AXLE_HL, at=(WB, 0.0, 0.0), axis="y", mat="dark"),
        sphere(0.24, at=(WB, 0.0, -0.01), mat="dark"),
        # lift ram base clevis
        box(0.26, 0.30, 0.24, at=(RAM_BASE[0], BOOM_Y, RAM_BASE[1] - 0.06), mat="dark",
            collision=False),
        # folded front stabilisers (beam, legs and pads up for travel), ahead of the tyres
        box(0.24, 1.20, 0.30, at=(STAB_X, 0.0, agl(0.64)), mat="dark"),
        *[box(0.16, 0.14, 0.72, at=(STAB_X, s * 0.55, agl(0.83)), mat="body") for s in sides],
        *[box(0.32, 0.26, 0.05, at=(STAB_X, s * 0.55, agl(0.455)), mat="dark") for s in sides],
        *[cyl(0.045, 0.50, at=(STAB_X - 0.08, s * 0.42, agl(1.00)), mat="steel",
              collision=False) for s in sides],
    ]


def _knuckle(name: str, s: float) -> Link:
    """Front steering knuckle: wheel hub + steering mudguard with a road light."""
    r = WHEEL_R
    return Link(name, M["knuckle"], [
        # hub reaching in toward the axle housing (≤ 0.50 m inboard, so it never sweeps into
        # the housing end 0.55 m inboard when steering)
        cyl(0.17, 0.25, at=(0.0, -s * 0.375, 0.0), axis="y", mat="dark"),
        cyl(0.07, 0.36, at=(0.0, -s * 0.33, 0.0), mat="dark", collision=False),      # king pin
        box(0.70, 0.50, 0.04, at=(0.0, 0.0, r + 0.09), mat="accent", collision=False),
        beam((0.35, r + 0.09), (0.62, r - 0.18), 0.50, 0.04, mat="accent", collision=False),
        beam((-0.35, r + 0.09), (-0.55, r - 0.12), 0.50, 0.04, mat="accent", collision=False),
        box(0.06, 0.14, 0.10, at=(0.40, s * 0.12, r + 0.16), mat="white", collision=False),
        box(0.04, 0.03, 0.30, at=(0.0, -s * 0.30, r - 0.05), mat="dark", collision=False),
    ])


def _carriage() -> list[Shape]:
    """Fork carriage (frame: origin = carriage pin, +x toward the fork tips when level)."""
    cx = HEEL_X - 0.09                  # carriage frame plates just behind the fork shanks
    fz = -FORK_DROP                     # fork underside
    shapes = [
        box(0.12, 1.28, 0.14, at=(cx, 0.0, 0.45), mat="dark"),                   # top bar
        box(0.12, 1.28, 0.14, at=(cx, 0.0, fz + 0.13), mat="dark"),               # bottom bar
        *[box(0.12, 0.10, 0.80, at=(cx, s * 0.59, -0.01), mat="dark") for s in (1, -1)],
        # hitch side plates straddling the boom nose (outboard of its ±0.20 m), so the
        # carriage can rotate about the pin without cutting into the boom head
        *[box(0.10, 0.05, 0.80, at=(cx - 0.01, s * 0.24, -0.01), mat="dark") for s in (1, -1)],
        # pivot lugs around the boom-head pin + tilt-ram lug (visual)
        *[box(0.34, 0.05, 0.24, at=(0.15, s * 0.24, 0.0), mat="dark", collision=False)
          for s in (1, -1)],
        cyl(0.06, 0.58, axis="y", mat="steel", collision=False),
        box(0.20, 0.20, 0.10, at=(0.20, 0.0, 0.50), mat="dark", collision=False),
    ]
    for y in (0.38, -0.38):                                                   # pallet forks
        shapes += [
            box(0.06, 0.12, 1.04, at=(HEEL_X - 0.03, y, fz + 0.52), mat="accent"),   # shank
            box(FORK_LEN, 0.12, 0.06, at=(HEEL_X + FORK_LEN / 2, y, fz + 0.03), mat="accent"),
            box(0.10, 0.12, 0.10, at=(HEEL_X + 0.02, y, fz + 0.07), mat="accent",
                collision=False),                                                # heel
            box(0.08, 0.14, 0.10, at=(HEEL_X - 0.05, y, 0.40), mat="accent",
                collision=False),                                                # top hook
        ]
    return shapes


# -----------------------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------------------

def build() -> MachineSpec:
    sides = (1, -1)
    pin_x = _pin_x_fly()
    ext = round(3 * _stroke(), 3)                  # total extension (three sections)
    l_ext = _retracted_len() + ext
    luff_lo = round(_luff_lo(), 4)
    luff_hi = round(_luff_for_pin_height(l_ext, LIFT_HEIGHT + FORK_DROP - (PIVOT[1] + WHEEL_R)), 4)

    # ------------------------------------------------------------------ tool joints
    luff = Joint("boom_luff_joint", "revolute", "chassis", "boom_base",
                 xyz=(PIVOT[0], BOOM_Y, PIVOT[1]), axis=(0, -1, 0), lower=luff_lo, upper=luff_hi,
                 init=luff_lo, velocity=0.14, role="tool")
    # The telescope tool drives the fly section directly (total extension); the two
    # intermediate sections are siblings on the base section that follow it at 1/3 and
    # 2/3 (rope/chain synchronised), so no follower chains.
    tele = Joint("boom_telescope_joint", "prismatic", "boom_base", "boom_fly",
                 xyz=(SEC_X0, 0.0, 0.0), axis=(1, 0, 0), lower=0.0, upper=ext, init=0.0,
                 velocity=0.90, role="tool")
    # Fork tilt relative to level: 54° forward … 23° back, i.e. ≈ 150° of carriage
    # rotation relative to the boom over the luff range (class figure ≈ 140-150°).
    tilt = Joint("fork_tilt_joint", "revolute", "carriage_level", "carriage",
                 axis=(0, -1, 0), lower=-0.95, upper=0.40, init=0.05, velocity=0.45, role="tool")

    # ------------------------------------------------------------------ lift ram fit
    (kb, cb), _, l_min, l_max = _ram_fit(luff.lower, luff.upper)
    barrel_len = l_min - 0.12
    rod_len = l_max - barrel_len + 0.15

    # ------------------------------------------------------------------ chassis
    chassis = Link("chassis", M["chassis"], [
        *_chassis_frame(), *_counterweight_and_pivot(), *_cab(), *_engine_bay(),
        # right-side camera post on the engine cover, with an arm carrying the camera
        # outboard of the front-right mudguard
        box(0.05, 0.05, 0.50, at=(RCAM[0], ENG_Y0 + 0.06, agl(1.67)), mat="accent",
            collision=False),
        box(0.05, ENG_Y0 + 0.085 - RCAM[1], 0.04,
            at=(RCAM[0], (ENG_Y0 + 0.085 + RCAM[1]) / 2, RCAM[2] - 0.04), mat="accent",
            collision=False),
    ])

    # ------------------------------------------------------------------ boom
    bw, bh = BASE_SEC
    base = Link("boom_base", M["base"], [
        *_tube(BASE_X0 + 0.23, BASE_X1, BASE_SEC),
        box(0.23, bw, bh, at=(BASE_X0 + 0.115, 0.0, 0.0)),                         # foot casting
        cyl(0.20, bw, axis="y", collision=False),                                    # foot boss
        cyl(0.12, bw + 0.08, axis="y", mat="steel", collision=False),
        *_collar(BASE_X1 - 0.05, BASE_SEC),
        # ram-eye lug under the belly, decal stripes, hose guide on the left side
        box(0.36, 0.26, 0.18, at=(RAM_EYE[0], 0.0, -bh / 2 - 0.07), mat="dark", collision=False),
        cyl(0.07, 0.34, at=(RAM_EYE[0], 0.0, RAM_EYE[1]), axis="y", mat="steel",
            collision=False),
        *[box(3.60, 0.01, 0.07, at=(2.40, s * (bw / 2 + 0.005), -0.14), mat="accent",
              collision=False) for s in sides],
        box(3.80, 0.08, 0.10, at=(2.20, bw / 2 + 0.04, 0.16), mat="dark", collision=False),
    ])
    mid1 = Link("boom_mid_1", M["mid1"], [*_tube(0.0, MID1_LEN, MID1_SEC),
                                          *_collar(MID1_LEN - 0.05, MID1_SEC)])
    mid2 = Link("boom_mid_2", M["mid2"], [*_tube(0.0, MID2_LEN, MID2_SEC),
                                          *_collar(MID2_LEN - 0.05, MID2_SEC)])
    fw, fh = FLY_SEC
    head_x0, head_x1 = FLY_LEN - 0.05, pin_x - 0.05
    head_z0, head_z1 = -fh / 2 - 0.005, 0.16
    fly = Link("boom_fly", M["fly"], [
        box(FLY_LEN, fw, fh, at=(FLY_LEN / 2, 0.0, 0.0)),
        # boom head: block ending 5 cm behind the carriage pin, short drop web and a
        # rounded nose around the pin (nothing below the fly section behind the pin, so
        # the carriage frame can swing under the head when tilted forward at high luff)
        box(head_x1 - head_x0, 0.40, head_z1 - head_z0,
            at=((head_x0 + head_x1) / 2, 0.0, (head_z0 + head_z1) / 2), mat="dark"),
        box(0.15, 0.40, head_z0 + DROP + 0.02, at=(pin_x - 0.125, 0.0, (head_z0 - DROP + 0.02) / 2),
            mat="dark"),
        cyl(NOSE_R, 0.40, at=(pin_x, 0.0, -DROP), axis="y", mat="dark"),
        # compensation (slave) ram on top of the head, hydraulic quick couplers
        _rod((FLY_LEN - 1.30, fh / 2 + 0.07), (FLY_LEN - 0.30, fh / 2 + 0.07), 0.065, mat="body"),
        _rod((FLY_LEN - 0.40, fh / 2 + 0.07), (head_x1 - 0.05, 0.18), 0.04),
        box(0.12, 0.30, 0.08, at=(head_x0 + 0.10, 0.0, fh / 2 + 0.04), mat="accent",
            collision=False),
    ])
    level = Link("carriage_level", M["level"], [
        cyl(0.065, 0.50, axis="y", mat="steel", collision=False),
    ])
    carriage = Link("carriage", M["carriage"], _carriage())

    barrel = Link("lift_cylinder", M["barrel"], [
        cyl(0.14, barrel_len, at=(barrel_len / 2, 0.0, 0.0), axis="x", mat="dark", collision=False),
        cyl(0.16, 0.08, at=(barrel_len - 0.04, 0.0, 0.0), axis="x", mat="accent", collision=False),
        cyl(0.10, 0.30, axis="y", mat="steel", collision=False),
    ])
    rod = Link("lift_rod", M["rod"], [
        cyl(0.075, rod_len, at=(rod_len / 2, 0.0, 0.0), axis="x", mat="steel", collision=False),
        cyl(0.11, 0.24, axis="y", mat="steel", collision=False),
    ])

    # ------------------------------------------------------------------ wheels
    wheel_links, wheel_joints, knuckles, steer_joints = [], [], [], []
    for side, s in (("left", 1.0), ("right", -1.0)):
        kn = f"front_{side}_knuckle"
        knuckles.append(_knuckle(kn, s))
        steer_joints.append(Joint(f"front_{side}_steer_joint", "revolute", "chassis", kn,
                                  xyz=(WB, s * Y_W, 0.0), axis=(0, 0, 1),
                                  lower=-0.85, upper=0.85, velocity=1.0, role="steer"))
        fwn, rwn = f"front_{side}_wheel", f"rear_{side}_wheel"
        wheel_links += [wheel_link(fwn, WHEEL_R, TYRE_W, M["wheel"]),
                        wheel_link(rwn, WHEEL_R, TYRE_W, M["wheel"])]
        wheel_joints += [wheel_joint(f"{fwn}_joint", kn, fwn, (0.0, 0.0, 0.0), WHEEL_R),
                         wheel_joint(f"{rwn}_joint", "chassis", rwn, (0.0, s * Y_W, 0.0), WHEEL_R)]

    links = [chassis, base, mid1, mid2, fly, level, carriage, barrel, rod,
             *knuckles, *wheel_links]

    joints = [
        luff,
        tele,
        # intermediate sections synchronised to the fly: each section strokes ext / 3
        *[Joint(f"boom_mid_{i}_joint", "prismatic", "boom_base", f"boom_mid_{i}",
                xyz=(x0, 0.0, 0.0), axis=(1, 0, 0), lower=0.0, upper=ext * i / 3, init=0.0,
                velocity=tele.velocity * i / 3, role="follower", follows=(tele.name, i / 3, 0.0))
          for i, x0 in ((1, MID1_X0), (2, MID1_X0 + MID2_X0))],
        # carriage compensation: the levelling frame cancels the boom angle
        Joint("carriage_level_joint", "revolute", "boom_fly", "carriage_level",
              xyz=(pin_x, 0.0, -DROP), axis=(0, -1, 0), lower=-luff.upper - 0.05,
              upper=-luff.lower + 0.05, init=-luff.init, velocity=luff.velocity,
              role="follower", follows=(luff.name, -1.0, 0.0)),
        tilt,
        _follower("lift_cylinder_joint", "chassis", "lift_cylinder", RAM_BASE, luff, kb, cb,
                  y=BOOM_Y),
        _follower("lift_rod_joint", "boom_base", "lift_rod", RAM_EYE, luff, kb - 1.0,
                  cb + math.pi),
        *steer_joints,
        *wheel_joints,
    ]

    cab_y = (CAB_Y0 + CAB_Y1) / 2
    return MachineSpec(
        name="telehandler",
        abbrev="TH",
        display_name="Telehandler",
        category="lifting",
        reference="17 m / 4 t telescopic handler (JCB 540-170 Loadall / CAT TH417D class)",
        description=("Rigid-frame 4x4 telescopic handler with a left-hand cab, side engine, "
                     "rear-pivot four-section telescopic boom (16.7 m lift height, 12.5 m "
                     "reach), compensated tilting fork carriage with pallet forks, folded "
                     "front stabilisers and front-wheel Ackermann steering."),
        root="chassis",
        links=links,
        joints=joints,
        drive=Drive(type="ackermann", max_speed=8.5, wheelbase=WB, max_steer=MAX_STEER,
                    steer_rate=0.38, fixed_axle_x=0.0, steer_axle="front"),
        tools=[
            Tool("boom_luff_joint", "boom_luff", positive="raise", negative="lower"),
            Tool("boom_telescope_joint", "boom_telescope", positive="extend", negative="retract"),
            Tool("fork_tilt_joint", "fork_tilt", positive="tilt_back", negative="tilt_forward"),
        ],
        cameras=[
            # operator eye point just ahead of the windscreen, looking down the boom at the forks
            # (level: keeps the forks in view from ground work up to mid-height lifts)
            Camera("front", "chassis", (CAB_X1 + 0.07, cab_y - 0.03, agl(2.05)),
                   (0.0, 0.0, -0.15)),
            # cab front-left roof corner (mirror position) looking out to the left
            Camera("left", "chassis", (CAB_X1 + 0.04, CAB_Y1 - 0.03, agl(2.50)),
                   (0.0, 0.35, 0.65)),
            # arm on the front-right corner of the engine cover looking out to the right
            Camera("right", "chassis", RCAM, (0.0, 0.32, -0.75)),
        ],
        envelope=(7.56, 2.44, 2.69),
        operating_mass=12060.0,
        notes=("Fork tilt is relative to level (automatic carriage compensation follows the "
               "boom with k = -1), so 0 = forks level whatever the boom angle; tilt_back / "
               "tilt_forward are verified visually (not in the auto-verified vocabulary). The "
               "telescope tool position is the TOTAL boom extension (fly section); the two "
               "intermediate sections follow at 1/3 and 2/3. Limitation: with the boom fully "
               "lowered (travel pose) tilting the forks forward past ≈ 23° (fork_tilt below "
               "≈ 40%) pushes the fork tips below grade (0.5 m at full forward tilt); the "
               "kinematic env does not stop it, so goal sampling should avoid such tilt goals "
               "at luff 0%. Stabilisers are modelled folded and are not operable."),
    )
