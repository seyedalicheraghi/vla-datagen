"""Telescopic boom lift (TBL), 60 ft class — modelled on a JLG 600SJ / Genie S-60 (J).

Datasheet numbers used (JLG 600S / 600SJ, Genie S-60 XC / S-60 J; rounded):
    platform height 60 ft 5 in = 18.4 m (working height ≈ 20.4 m) ·
    horizontal outreach 50 ft 5 in = 15.4 m · platform capacity 227-300 kg ·
    stowed length 31 ft 0 in = 9.45 m (600SJ, jib lowered) · stowed height
    8 ft 5 in = 2.57 m · width 8 ft 2 in = 2.49 m (2.44 m / 8 ft platform) ·
    wheelbase 8 ft 1 in = 2.46 m · ground clearance 0.30 m · tailswing 4 ft =
    1.22 m (Genie S-60 X sheet; some listings give 1.41 m for the XC's larger
    counterweight) · 41/18LL-22.5 foam-filled tyres (OD 1.04 m, section
    0.46 m) · operating mass 10.4 t (S-60 XC 4WD 22 930 lb in dealer listings,
    Genie's own sheet gives 9.96 t; S-60 X 4WD 9.4 t; 600S 9.7 t, 600SJ
    10.8 t) · 4WD, 2WS: outside turning radius 19 ft 7 in = 5.97 m · drive
    3.8 mph = 1.7 m/s (stowed) · 45 % gradeability · 360° continuous turntable
    · 6 ft = 1.83 m jib with ≈ 135° vertical articulation · boom 0..+75° ·
    platform rotation 160° (Genie S-60 X / XC; JLG 600S / 600SJ 180°) ·
    normal driving position = boom over the non-steer (drive) axle (JLG's
    drive-orientation system and Genie's drive-enable trip when the boom is
    swung past those wheels), i.e. the steered axle is under the counterweight.
Assumed / typical for the class:
    boom = base + mid + fly (the hydraulic telescope cylinder drives the mid
    section; the fly is chain-synchronised to it, so it is a follower with
    k = 1). A single 2-stage telescope cannot reach 18 m from a 6.2 m stowed
    boom, which is why the 60 ft class uses three sections · base section
    0.56 × 0.62 m, mid 0.46 × 0.52 m, fly 0.36 × 0.42 m · boom pivot 2.25 m
    above ground, 1.45 m behind the slew axis, base-section tail 0.10 m behind
    the pivot (so it clears the counterweight top while the boom rises) ·
    pivot-to-jib-pin length 6.17 m retracted, 15.34 m extended (2 × 4.585 m
    stroke) · jib pivot 0.45 m ahead of the fly tip pin on the levelling
    bracket (clear of the fly head at every boom/jib angle) · rotator axis
    0.32 m ahead of the jib-tip pin · counterweight 2.9 t, underside 1.07 m
    above the ground (clears the 1.04 m tyre tops at every slew angle) ·
    platform 2.44 × 0.91 m, rails 1.10 m · steer lock 0.70 rad.
Model check (offline FK): platform floor height 18.42 m at boom 75° / full
    extension / jib +70°, floor exactly level in every pose · no link pair
    interpenetrates over slew × steer and boom × telescope × jib × platform
    rotation sweeps · unrestricted horizontal outreach 17.7 m to the platform
    front (the datasheet gives 15.4 m; the real machine's reach envelope near
    horizontal is extension-limited, which independent joint limits cannot
    express) · stowed 9.93 × 2.53 × 2.59 m (platform floor 0.22 m above the
    ground) · outside turning radius 5.97 m at max_steer 0.53 rad (inner
    wheel 37.7°).
Stability (no runtime effect: the env disables gravity and moves the base
    kinematically): the unrestricted reach envelope is statically UNSTABLE.
    Boom horizontal, fully extended, jib level, slewed 90°: whole-machine CoM
    0.96 m off the centreline unloaded, 1.31 m with a 230 kg basket load,
    against a tyre centreline at 1.015 m (outer tyre edge 1.245 m).  Real
    machines stay stable through the extension-limited envelope and an
    upper-structure mass that sits further rearward than this model's.

Joint ranges and rates at full command: slew continuous at 0.10 rad/s
(≈ 63 s per turn, the class figure); boom 0..1.31 rad at 0.11 rad/s, telescope
(mid section, fly follows) 0..4.585 m at 0.36 m/s, jib -1.13..+1.22 rad at
0.20 rad/s, platform rotation ±1.40 rad (±80°) at 0.25 rad/s.  The boom,
telescope and jib rates are ≈ 3x the ANSI-limited real function speeds (boom up
≈ 45 s, telescope out ≈ 40 s, jib ≈ 25 s) because the offline tests require
every tool to cover its full range within 15 s (and a task goal within 40 s).

Platform levelling (physically: master/slave levelling cylinders + a
parallelogram jib).  The jib is mounted on a levelling bracket at the fly tip
(two clevis plates outside the fly head) that follows the boom angle with
k = -1 (the slave cylinder keeps the bracket level whatever the boom angle),
so the jib joint angle IS the jib elevation above horizontal.  The jib is a
parallelogram, so the rotator housing at its tip follows the jib joint with
k = -1.  Both followers follow a tool joint (no follower chains) and the
platform floor stays exactly level.  The rotator (a tool) yaws the basket
about its vertical axis; the front camera is on the basket, so it turns with
it like the operator.  The lift cylinder barrel is a cosmetic follower of the
boom (linear fit of the true pin-to-pin angle).

Drive: +x (forward) is the direction the operator faces from the platform
console in the normal driving position, i.e. platform first over the
non-steered axle.  The steered axle is therefore the trailing one
(steer_axle = "rear", like a forklift); the env still maps turn = +1 to a left
turn.

Init pose = stowed travel pose: boom over the non-steer axle (slew 0), boom
down on its rest, telescope fully retracted, jib fully lowered so the platform
hangs just above the ground for entry, platform rotation centred.

Root frame: chassis centre at wheel-axle height (axle centres at z = 0),
+x towards the platform and the non-steered axle.
"""

from __future__ import annotations

import math

from ..spec import (DEFAULT_MATERIALS, Camera, Drive, Joint, Link, MachineSpec, Shape, Tool,
                    beam, box, cyl, wheel_joint, wheel_link)

JLG_ORANGE = (0.93, 0.43, 0.07, 1.0)

# --- Chassis ------------------------------------------------------------------------
WHEEL_R, TYRE_W = 0.52, 0.46          # 41/18LL-22.5 foam-filled
WHEELBASE = 2.46
WIDTH = 2.49
Y_W = (WIDTH - TYRE_W) / 2            # wheel centre (1.015 m) → gauge 2.03 m
X_FIXED, X_STEER = WHEELBASE / 2, -WHEELBASE / 2   # non-steer axle under the boom, steer axle under the counterweight
Z_GND = -WHEEL_R                      # ground in the root frame
SLEW_AGL = 1.12                       # top of the slew bearing above the ground

# --- Turntable (frame z = 0 at the top of the slew bearing) -----------------------------
PIVOT = (-1.45, 2.25 - SLEW_AGL)      # boom foot pin (x, z), 2.25 m above the ground
TAIL_R = 2.465                        # counterweight radius → tailswing 1.22 m
LIFT_BASE = (-1.05, 0.40)             # lift-cylinder base pin (turntable frame)
LIFT_EYE = (3.00, -0.42)              # lift-cylinder rod eye (boom-base frame)

# --- Boom (frame origin = foot pin, +x along the boom) ------------------------------------
BASE_SEC = (0.56, 0.62)               # outer width (y) × depth (z)
MID_SEC = (0.46, 0.52)
FLY_SEC = (0.36, 0.42)
WALL = 0.04
BASE_X0, BASE_X1 = -0.10, 5.57        # base section extent (short tail: clears the counterweight)
SEC_LEN = 5.64                        # mid and fly section lengths
MID_X0 = 0.13                         # mid rear end in the base frame (retracted)
FLY_X0 = 0.20                         # fly rear end in the mid frame (retracted)
TIP_X = SEC_LEN + 0.20                # jib-bracket pin in the fly frame
STROKE = 4.585                        # per section → 9.17 m total extension

# --- Jib and platform -------------------------------------------------------------------
JIB_PIVOT = (0.45, 0.0)               # in the levelling-bracket frame (clears the fly head at boom 75°)
JIB_LEN = 1.83
ROT_X = 0.32                          # platform-rotator axis ahead of the jib-tip pin (clear of
                                      # the jib-arm ends, which sweep r ≈ 0.17 m round the pin)
ROT_LIM = math.radians(80.0)          # ±80° → 160° platform rotation (Genie S-60 X/XC)
PLAT_D, PLAT_W = 0.91, 2.44           # depth (x) × width (y)
PLAT_X0 = ROT_X + 0.20                # platform rear edge ahead of the jib-tip pin (rotation 0)
FLOOR_Z = -0.37                       # floor top below the jib-tip pin
RAIL_H = 1.10


# -----------------------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------------------

def rod(p0: tuple[float, float], p1: tuple[float, float], radius: float, y: float = 0.0,
        mat: str = "steel") -> Shape:
    """Visual-only cylinder from p0 to p1 in the link's x-z plane."""
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    return cyl(radius, math.hypot(dx, dz), at=((p0[0] + p1[0]) / 2, y, (p0[1] + p1[1]) / 2),
               rpy=(0.0, math.atan2(dx, dz), 0.0), mat=mat, collision=False)


def tube(x0: float, x1: float, w: float, h: float, mat: str = "body") -> list[Shape]:
    """Hollow rectangular boom section from x0 to x1 (four plates, all colliding)."""
    lx, cx = x1 - x0, (x0 + x1) / 2
    return [
        box(lx, w, WALL, at=(cx, 0.0, h / 2 - WALL / 2), mat=mat),
        box(lx, w, WALL, at=(cx, 0.0, -h / 2 + WALL / 2), mat=mat),
        box(lx, WALL, h - 2 * WALL, at=(cx, w / 2 - WALL / 2, 0.0), mat=mat),
        box(lx, WALL, h - 2 * WALL, at=(cx, -w / 2 + WALL / 2, 0.0), mat=mat),
    ]


def collar(x: float, w: float, h: float, mat: str = "dark") -> list[Shape]:
    """Visual wear-pad collar around the mouth of a boom section."""
    t = 0.03
    return [
        box(0.10, w + 2 * t, t, at=(x, 0.0, h / 2 + t / 2), mat=mat, collision=False),
        box(0.10, w + 2 * t, t, at=(x, 0.0, -h / 2 - t / 2), mat=mat, collision=False),
        box(0.10, t, h, at=(x, w / 2 + t / 2, 0.0), mat=mat, collision=False),
        box(0.10, t, h, at=(x, -w / 2 - t / 2, 0.0), mat=mat, collision=False),
    ]


def _raise(p: tuple[float, float], q: float) -> tuple[float, float]:
    """Rotate an x-z point by a joint with axis (0, -1, 0) at angle q."""
    c, s = math.cos(q), math.sin(q)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


def _elev(v: tuple[float, float]) -> float:
    return math.atan2(v[1], v[0])


def _lift_cylinder_fit(lo: float, hi: float) -> tuple[float, float, float, float]:
    """Linear fit of the lift-cylinder barrel elevation vs boom angle.

    Returns (k, c, rod angle fixed in the boom frame, retracted length).
    """
    qs = [lo + (hi - lo) * i / 100 for i in range(101)]
    eyes = []
    for q in qs:
        e = _raise(LIFT_EYE, q)
        eyes.append((PIVOT[0] + e[0], PIVOT[1] + e[1]))
    ys = [_elev((e[0] - LIFT_BASE[0], e[1] - LIFT_BASE[1])) for e in eyes]
    n = len(qs)
    mq, my = sum(qs) / n, sum(ys) / n
    k = sum((q - mq) * (y - my) for q, y in zip(qs, ys)) / sum((q - mq) ** 2 for q in qs)
    err = [y - k * q for q, y in zip(qs, ys)]
    c = (max(err) + min(err)) / 2
    rod_ang = [_elev((LIFT_BASE[0] - e[0], LIFT_BASE[1] - e[1])) - q for e, q in zip(eyes, qs)]
    rod_mid = (max(rod_ang) + min(rod_ang)) / 2
    lmin = min(math.hypot(e[0] - LIFT_BASE[0], e[1] - LIFT_BASE[1]) for e in eyes)
    return k, c, rod_mid, lmin


# -----------------------------------------------------------------------------------------
# Links
# -----------------------------------------------------------------------------------------

def _chassis() -> Link:
    agl = lambda h: h + Z_GND          # noqa: E731 — height above ground → root z
    top = agl(SLEW_AGL - 0.10)          # top of the chassis deck
    shapes = [
        # box frame: wide centre section between the wheels, narrower noses at the axles
        box(1.36, 1.40, 0.70, at=(0.0, 0.0, agl(0.30 + 0.35))),
        box(3.30, 0.90, 0.58, at=(0.0, 0.0, agl(0.36 + 0.29))),
        box(1.36, 2.20, 0.40, at=(0.0, 0.0, agl(0.55 + 0.20))),       # side tanks / boxes
        box(1.90, 1.40, 0.06, at=(0.0, 0.0, top - 0.03)),               # bearing deck
        cyl(0.72, 0.10, at=(0.0, 0.0, top + 0.05), mat="dark"),         # slew bearing
        # axle housings (the steer axle carries the steering knuckles)
        cyl(0.14, 1.29, at=(X_FIXED, 0.0, 0.0), axis="y", mat="dark"),
        cyl(0.14, 1.29, at=(X_STEER, 0.0, 0.0), axis="y", mat="dark"),
        # non-steered wheel-motor hubs
        cyl(0.17, 0.14, at=(X_FIXED, Y_W - 0.30, 0.0), axis="y", mat="dark"),
        cyl(0.17, 0.14, at=(X_FIXED, -(Y_W - 0.30), 0.0), axis="y", mat="dark"),
        # visual details: tie-down / lift lugs, steering cylinder, side-box lids
        *[box(0.10, 0.20, 0.16, at=(sx * 1.70, sy * 0.30, agl(0.62)), mat="accent",
              collision=False) for sx in (-1, 1) for sy in (-1, 1)],
        cyl(0.05, 0.90, at=(X_STEER + 0.22, 0.0, 0.10), axis="y", mat="steel", collision=False),
        *[box(1.10, 0.02, 0.26, at=(0.0, sy * 1.105, agl(0.78)), mat="accent", collision=False)
          for sy in (-1, 1)],
        *[box(0.20, 0.02, 0.08, at=(sx * 0.40, sy * 1.11, agl(0.70)), mat="steel",
              collision=False) for sx in (-1, 1) for sy in (-1, 1)],
    ]
    return Link("chassis", 2150.0, shapes)


def _turntable() -> list[Shape]:
    px, pz = PIVOT
    tower_x0, tower_x1 = px - 0.30, px + 0.32
    shapes = [
        box(2.65, 1.10, 0.25, at=(-0.225, 0.0, 0.125)),                    # base weldment
        # engine tray (left) and hydraulic / fuel tray (right) with their covers
        box(2.30, 0.68, 0.80, at=(-0.30, 0.79, 0.40)),
        box(2.30, 0.68, 0.80, at=(-0.30, -0.79, 0.40)),
        cyl(0.40, 0.80, at=(0.85, 0.85, 0.40)),                             # rounded front corners
        cyl(0.40, 0.80, at=(0.85, -0.85, 0.40)),
        # boom pivot tower (two side plates) and foot-pin bosses
        box(0.62, 0.07, 1.30, at=((tower_x0 + tower_x1) / 2, 0.355, 0.65), mat="body"),
        box(0.62, 0.07, 1.30, at=((tower_x0 + tower_x1) / 2, -0.355, 0.65), mat="body"),
        cyl(0.16, 0.86, at=(px, 0.0, pz), axis="y", mat="steel", collision=False),
        # lift-cylinder base clevis and boom rest (two posts straddling the cylinder)
        box(0.26, 0.30, 0.20, at=(LIFT_BASE[0], 0.0, LIFT_BASE[1] - 0.08), mat="dark"),
        box(0.10, 0.06, 0.55, at=(0.95, 0.22, 0.25 + 0.275), mat="dark"),
        box(0.10, 0.06, 0.55, at=(0.95, -0.22, 0.25 + 0.275), mat="dark"),
        box(0.14, 0.50, 0.04, at=(0.95, 0.0, 0.81), mat="rubber", collision=False),
        # covers: grilles, louvres, ground-control box, exhaust, beacon, fuel cap, steps
        box(1.30, 0.02, 0.45, at=(-0.45, 1.135, 0.45), mat="accent", collision=False),
        *[box(1.20, 0.025, 0.03, at=(-0.45, 1.14, 0.30 + 0.07 * i), mat="dark", collision=False)
          for i in range(5)],
        box(0.45, 0.10, 0.50, at=(0.30, -1.17, 0.45), mat="dark", collision=False),
        box(0.30, 0.03, 0.10, at=(0.30, -1.22, 0.62), mat="red", collision=False),
        box(0.70, 0.02, 0.35, at=(-0.80, -1.135, 0.40), mat="accent", collision=False),
        cyl(0.06, 0.40, at=(-0.95, 0.95, 0.98), mat="accent", collision=False),
        cyl(0.07, 0.12, at=(0.55, 0.95, 0.86), mat="orange", collision=False),
        cyl(0.06, 0.04, at=(0.20, -0.70, 0.82), mat="dark", collision=False),
        box(0.30, 0.02, 0.04, at=(0.55, 1.14, 0.12), mat="steel", collision=False),
        box(0.02, 1.30, 0.20, at=(1.25, 0.0, 0.50), mat="accent", collision=False),
    ]
    return shapes


def _counterweight() -> Link:
    """Cast counterweight hung on the turntable tail (turntable frame).

    Its underside is 1.07 m above the ground, i.e. above the 1.04 m tyre tops,
    so it clears the wheels at every slew angle.
    """
    r0 = TAIL_R
    return Link("counterweight", 2900.0, [
        box(0.70, 2.00, 0.85, at=(-1.85, 0.0, 0.375), mat="dark"),          # main block
        box(0.30, 1.40, 0.85, at=(-(r0 - 0.15), 0.0, 0.375), mat="dark"),   # rear face
        cyl(0.30, 0.85, at=(-(r0 - 0.30), 0.70, 0.375), mat="dark"),        # rounded corners
        cyl(0.30, 0.85, at=(-(r0 - 0.30), -0.70, 0.375), mat="dark"),
        # tail lights and stripes (visual)
        box(0.02, 0.18, 0.08, at=(-r0 - 0.005, 0.50, 0.55), mat="red", collision=False),
        box(0.02, 0.18, 0.08, at=(-r0 - 0.005, -0.50, 0.55), mat="red", collision=False),
        box(0.02, 1.10, 0.06, at=(-r0 - 0.005, 0.0, 0.72), mat="body", collision=False),
    ])


def _rotator() -> Link:
    """Platform rotator on the jib tip; levelled frame at the jib-tip pin, does not yaw."""
    bx0, bx1 = -0.12, ROT_X
    return Link("rotator", 70.0, [
        cyl(0.13, 0.62, at=(ROT_X, 0.0, -0.08), mat="dark"),                          # rotary actuator
        # clevis plates from the jib-tip pins to the actuator housing (outside the jib arms)
        *[box(bx1 - bx0, 0.03, 0.32, at=((bx0 + bx1) / 2, sy * 0.11, 0.0), mat="dark",
              collision=False) for sy in (-1.0, 1.0)],
        cyl(0.15, 0.06, at=(ROT_X, 0.0, 0.26), mat="dark", collision=False),         # motor cap
    ])


def _platform() -> Link:
    """Platform basket; frame on the rotator axis at jib-tip-pin height, +x away from the jib."""
    x0, x1 = PLAT_X0 - ROT_X, PLAT_X0 - ROT_X + PLAT_D
    cx, hw = (x0 + x1) / 2, PLAT_W / 2
    zf = FLOOR_Z
    zt, zm = zf + RAIL_H, zf + RAIL_H / 2
    rail = 0.05
    s: list[Shape] = [
        # mounting weldment from the rotator output to the support frame under the floor
        # (kept within r ≈ 0.18 m of the axis behind it, so it clears the jib when rotated)
        box(0.50, 0.30, 0.20, at=(0.15, 0.0, zf - 0.02), mat="dark"),
        cyl(0.15, 0.04, at=(0.0, 0.0, zf + 0.10), mat="steel", collision=False),    # rotator flange
        box(PLAT_D, PLAT_W, 0.08, at=(cx, 0.0, zf - 0.04), mat="dark"),      # floor
        box(PLAT_D * 0.6, 0.12, 0.06, at=(cx, 0.0, zf - 0.11), mat="dark"),  # floor beam
        # toe boards
        box(PLAT_D, 0.02, 0.15, at=(cx, hw - 0.01, zf + 0.075), collision=False),
        box(PLAT_D, 0.02, 0.15, at=(cx, -hw + 0.01, zf + 0.075), collision=False),
        box(0.02, PLAT_W, 0.15, at=(x1 - 0.01, 0.0, zf + 0.075), collision=False),
        box(0.02, PLAT_W, 0.15, at=(x0 + 0.01, 0.0, zf + 0.075), collision=False),
    ]
    # guard rails: top + mid rails all round (top rails collide), posts
    for z, col in ((zt, True), (zm, False)):
        s += [
            box(PLAT_D, rail, rail, at=(cx, hw - rail / 2, z), mat="body", collision=col),
            box(PLAT_D, rail, rail, at=(cx, -hw + rail / 2, z), mat="body", collision=col),
            box(rail, PLAT_W, rail, at=(x1 - rail / 2, 0.0, z), mat="body", collision=col),
            box(rail, PLAT_W, rail, at=(x0 + rail / 2, 0.0, z), mat="body", collision=col),
        ]
    for px in (x0 + rail / 2, x1 - rail / 2):
        for py in (-hw + rail / 2, -0.40, 0.40, hw - rail / 2):
            s.append(box(rail, rail, RAIL_H, at=(px, py, zf + RAIL_H / 2), mat="body",
                         collision=False))
    # control console on the front rail, foot switch, entry gate, work light
    s += [
        box(0.22, 0.62, 0.34, at=(x1 - 0.14, 0.15, zf + 0.98), mat="dark"),
        box(0.18, 0.56, 0.02, at=(x1 - 0.18, 0.15, zf + 1.16), mat="white", collision=False),
        cyl(0.025, 0.10, at=(x1 - 0.20, 0.30, zf + 1.20), mat="red", collision=False),
        box(0.30, 0.22, 0.06, at=(x1 - 0.35, 0.15, zf + 0.03), mat="accent", collision=False),
        box(0.40, 0.02, 0.04, at=(x0 + 0.30, hw - 0.04, zf + 0.85), mat="accent", collision=False),
        box(0.10, 0.18, 0.10, at=(x1 + 0.03, -0.70, zt - 0.10), mat="white", collision=False),
    ]
    return Link("platform", 190.0, s)


# -----------------------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------------------

def build() -> MachineSpec:
    boom_lo, boom_hi = 0.0, math.radians(75.0)
    k_lift, c_lift, rod_ang, lift_len = _lift_cylinder_fit(boom_lo, boom_hi)

    # ------------------------------------------------------------------- links
    chassis = _chassis()
    turntable = Link("turntable", 1700.0, _turntable())
    counterweight = _counterweight()

    bw, bh = BASE_SEC
    rod_len = lift_len - 0.35
    rod_end = (LIFT_EYE[0] + rod_len * math.cos(rod_ang), LIFT_EYE[1] + rod_len * math.sin(rod_ang))
    boom_base = Link("boom_base", 1000.0, [
        *tube(BASE_X0, BASE_X1, bw, bh),
        box(0.20, bw - 2 * WALL, bh - 2 * WALL, at=(BASE_X0 + 0.10, 0.0, 0.0)),    # tail block
        cyl(0.14, bw + 0.04, axis="y", mat="steel", collision=False),             # foot-pin boss
        *collar(BASE_X1 - 0.05, bw, bh),
        # lift-cylinder lug + rod (barrel is a follower link on the turntable)
        box(0.34, 0.26, 0.16, at=(LIFT_EYE[0], 0.0, -bh / 2 - 0.05), mat="dark", collision=False),
        rod(LIFT_EYE, rod_end, 0.075),
        # external hose / cable track on the right side and decal stripes
        box(4.40, 0.10, 0.16, at=(2.60, -bw / 2 - 0.05, 0.12), mat="accent", collision=False),
        box(4.80, 0.01, 0.06, at=(2.90, bw / 2 + 0.005, -0.18), mat="accent", collision=False),
        box(4.80, 0.01, 0.06, at=(2.90, -bw / 2 - 0.005, -0.18), mat="accent", collision=False),
        # levelling master cylinder on the left side, near the foot pin
        rod((0.25, -0.12), (1.35, -0.08), 0.06, y=bw / 2 + 0.07, mat="dark"),
    ])

    mw, mh = MID_SEC
    boom_mid = Link("boom_mid", 560.0, [
        *tube(0.0, SEC_LEN, mw, mh),
        *collar(SEC_LEN - 0.05, mw, mh),
        box(0.12, mw - 2 * WALL, 0.03, at=(0.10, 0.0, mh / 2 - WALL - 0.015), mat="dark",
            collision=False),                                                     # wear pads
    ])

    fw, fh = FLY_SEC
    head_x0 = SEC_LEN
    boom_fly = Link("boom_fly", 400.0, [
        box(SEC_LEN, fw, fh, at=(SEC_LEN / 2, 0.0, 0.0)),
        box(0.26, fw + 0.06, fh + 0.10, at=(head_x0 + 0.13, 0.0, -0.02)),         # fly head
        cyl(0.11, fw + 0.18, at=(TIP_X, 0.0, 0.0), axis="y", mat="steel", collision=False),
        # slave levelling cylinder under the head, hose loop, stripes
        rod((head_x0 - 0.90, -fh / 2 - 0.08), (head_x0 + 0.05, -fh / 2 - 0.08), 0.055, mat="dark"),
        box(0.60, 0.08, 0.10, at=(head_x0 - 0.30, -fw / 2 - 0.04, 0.10), mat="accent",
            collision=False),
        box(3.00, 0.01, 0.05, at=(head_x0 - 1.80, fw / 2 + 0.005, -0.12), mat="accent",
            collision=False),
    ])

    # levelling bracket: two clevis side plates outside the fly head (±0.21 m), from the
    # tip pin to past the jib pins, so the head can rotate between them
    br_x0, br_x1 = -0.05, JIB_PIVOT[0] + 0.07
    jib_base = Link("jib_base", 70.0, [
        *[box(br_x1 - br_x0, 0.03, 0.56, at=((br_x0 + br_x1) / 2, sy * 0.245, -0.06), mat="dark")
          for sy in (-1.0, 1.0)],
        cyl(0.07, 0.52, at=(JIB_PIVOT[0], 0.0, JIB_PIVOT[1] + 0.12), axis="y", mat="steel",
            collision=False),
        cyl(0.07, 0.52, at=(JIB_PIVOT[0], 0.0, JIB_PIVOT[1] - 0.12), axis="y", mat="steel",
            collision=False),
    ])

    jib = Link("jib", 120.0, [
        beam((0.0, 0.12), (JIB_LEN, 0.12), 0.16, 0.10),                              # upper arm
        beam((0.0, -0.12), (JIB_LEN, -0.12), 0.16, 0.10),                            # lower arm
        rod((0.25, -0.02), (1.25, 0.02), 0.05, mat="dark"),                           # jib cylinder
        cyl(0.06, 0.30, at=(JIB_LEN, 0.0, 0.12), axis="y", mat="steel", collision=False),
        cyl(0.06, 0.30, at=(JIB_LEN, 0.0, -0.12), axis="y", mat="steel", collision=False),
    ])

    lift_barrel = Link("lift_cylinder", 120.0, [
        cyl(0.115, lift_len - 0.30, at=((lift_len - 0.30) / 2, 0.0, 0.0), axis="x", mat="dark",
            collision=False),
        cyl(0.13, 0.10, at=(lift_len - 0.35, 0.0, 0.0), axis="x", mat="steel", collision=False),
        cyl(0.09, 0.28, axis="y", mat="steel", collision=False),                      # base pin
    ])

    rotator = _rotator()
    platform = _platform()

    # ------------------------------------------------------------------ wheels
    wheel_links, wheel_joints, knuckles, steer_joints = [], [], [], []
    for side, sign in (("left", 1.0), ("right", -1.0)):
        kn = f"rear_{side}_knuckle"
        knuckles.append(Link(kn, 50.0, [
            cyl(0.17, 0.14, at=(0.0, -sign * 0.30, 0.0), axis="y", mat="dark"),        # wheel motor
            cyl(0.07, 0.30, at=(0.0, -sign * 0.40, 0.0), axis="z", mat="dark", collision=False),
        ]))
        steer_joints.append(Joint(f"rear_{side}_steer_joint", "revolute", "chassis", kn,
                                  xyz=(X_STEER, sign * Y_W, 0.0), axis=(0, 0, 1),
                                  lower=-0.70, upper=0.70, velocity=0.6, role="steer"))
        fw_name, rw_name = f"front_{side}_wheel", f"rear_{side}_wheel"
        wheel_links += [wheel_link(fw_name, WHEEL_R, TYRE_W, 260.0, hub="body"),
                        wheel_link(rw_name, WHEEL_R, TYRE_W, 260.0, hub="body")]
        wheel_joints += [
            wheel_joint(f"{fw_name}_joint", "chassis", fw_name, (X_FIXED, sign * Y_W, 0.0), WHEEL_R),
            wheel_joint(f"{rw_name}_joint", kn, rw_name, (0.0, 0.0, 0.0), WHEEL_R),
        ]

    links = [chassis, turntable, counterweight, boom_base, boom_mid, boom_fly, jib_base, jib,
             rotator, platform, lift_barrel, *knuckles, *wheel_links]

    # ------------------------------------------------------------------ joints
    boom_j = Joint("boom_joint", "revolute", "turntable", "boom_base",
                   xyz=(PIVOT[0], 0.0, PIVOT[1]), axis=(0, -1, 0),
                   lower=boom_lo, upper=boom_hi, init=0.0, velocity=0.11, role="tool")
    tele_j = Joint("telescope_joint", "prismatic", "boom_base", "boom_mid",
                   xyz=(MID_X0, 0.0, 0.0), axis=(1, 0, 0),
                   lower=0.0, upper=STROKE, init=0.0, velocity=0.36, role="tool")
    jib_lo, jib_hi = math.radians(-65.0), math.radians(70.0)
    jib_j = Joint("jib_joint", "revolute", "jib_base", "jib",
                  xyz=(JIB_PIVOT[0], 0.0, JIB_PIVOT[1]), axis=(0, -1, 0),
                  lower=jib_lo, upper=jib_hi, init=jib_lo, velocity=0.20, role="tool")
    lift_ends = (k_lift * boom_lo + c_lift, k_lift * boom_hi + c_lift)

    joints = [
        Joint("slew_joint", "continuous", "chassis", "turntable",
              xyz=(0.0, 0.0, SLEW_AGL + Z_GND), axis=(0, 0, 1), velocity=0.10, role="tool"),
        Joint("counterweight_mount", "fixed", "turntable", "counterweight"),
        boom_j,
        tele_j,
        # fly section: chain-synchronised with the mid section (same stroke)
        Joint("fly_joint", "prismatic", "boom_mid", "boom_fly",
              xyz=(FLY_X0, 0.0, 0.0), axis=(1, 0, 0), lower=0.0, upper=STROKE, init=0.0,
              velocity=tele_j.velocity, role="follower", follows=(tele_j.name, 1.0, 0.0)),
        # levelling bracket: slave cylinder cancels the boom angle
        Joint("jib_level_joint", "revolute", "boom_fly", "jib_base",
              xyz=(TIP_X, 0.0, 0.0), axis=(0, -1, 0), lower=-boom_hi - 0.05, upper=-boom_lo + 0.05,
              init=-boom_j.init, velocity=boom_j.velocity, role="follower",
              follows=(boom_j.name, -1.0, 0.0)),
        jib_j,
        # parallelogram jib keeps the platform rotator level
        Joint("platform_level_joint", "revolute", "jib", "rotator",
              xyz=(JIB_LEN, 0.0, 0.0), axis=(0, -1, 0), lower=-jib_hi - 0.05, upper=-jib_lo + 0.05,
              init=-jib_j.init, velocity=jib_j.velocity, role="follower",
              follows=(jib_j.name, -1.0, 0.0)),
        # platform rotator: yaws the basket about the vertical rotator axis (+ = left / CCW)
        Joint("platform_rotate_joint", "revolute", "rotator", "platform",
              xyz=(ROT_X, 0.0, 0.0), axis=(0, 0, 1), lower=-ROT_LIM, upper=ROT_LIM, init=0.0,
              velocity=0.25, role="tool"),
        Joint("lift_cylinder_joint", "revolute", "turntable", "lift_cylinder",
              xyz=(LIFT_BASE[0], 0.0, LIFT_BASE[1]), axis=(0, -1, 0),
              lower=min(lift_ends) - 0.1, upper=max(lift_ends) + 0.1,
              init=k_lift * boom_j.init + c_lift, velocity=abs(k_lift) * boom_j.velocity + 0.05,
              role="follower", follows=(boom_j.name, k_lift, c_lift)),
        *steer_joints,
        *wheel_joints,
    ]

    floor_cam_z = FLOOR_Z + 1.65
    return MachineSpec(
        name="telescopic_boom_lift",
        abbrev="TBL",
        display_name="Telescopic Boom Lift",
        category="lifting",
        reference="60 ft telescopic boom lift with jib (JLG 600SJ / Genie S-60 J class)",
        description=("4WD, 2WS aerial work platform with a 360° turntable and "
                     "counterweight, a three-section telescopic boom (18.4 m platform height), "
                     "a 1.8 m parallelogram jib and a self-levelling 2.44 m platform basket "
                     "on a 160° rotator."),
        root="chassis",
        links=links,
        joints=joints,
        drive=Drive(type="ackermann", max_speed=1.7, wheelbase=WHEELBASE, max_steer=0.53,
                    steer_rate=0.35, fixed_axle_x=X_FIXED, steer_axle="rear"),
        tools=[
            Tool("slew_joint", "slew", positive="left", negative="right"),
            Tool("boom_joint", "boom", positive="raise", negative="lower"),
            Tool("telescope_joint", "telescope", positive="extend", negative="retract"),
            Tool("jib_joint", "jib", positive="raise", negative="lower"),
            Tool("platform_rotate_joint", "platform_rotate", positive="left", negative="right"),
        ],
        cameras=[
            # operator standing behind the platform console, looking forward and down
            # (on the platform link, so it turns with the basket like the operator does)
            Camera("front", "platform", (PLAT_X0 - ROT_X + 0.30, 0.0, floor_cam_z), (0.0, 0.40, 0.0)),
            # front corners of the turntable covers, looking outward-forward
            Camera("left", "turntable", (1.05, 1.05, 0.90), (0.0, 0.30, 0.70)),
            Camera("right", "turntable", (1.05, -1.05, 0.90), (0.0, 0.30, -0.70)),
        ],
        envelope=(9.45, 2.49, 2.57),
        operating_mass=10400.0,
        materials={**DEFAULT_MATERIALS, "body": JLG_ORANGE},
        notes=("Platform levelling = boom-following bracket (k=-1) + parallelogram jib (rotator "
               "housing follows the jib with k=-1): the floor stays level for every pose; "
               "platform_rotate yaws the basket ±80° about the vertical rotator axis. The fly "
               "section follows the telescope (mid) joint 1:1, so the tool position is half the "
               "total extension. Boom/telescope/jib rates are ~3x real ANSI-limited speeds so every "
               "tool covers its range within the 15 s the tests allow. The unrestricted reach "
               "envelope (17.7 m vs 15.4 m datasheet) is statically unstable if dynamics are "
               "enabled."),
    )
