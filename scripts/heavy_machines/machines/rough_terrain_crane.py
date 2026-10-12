"""Rough-terrain crane (RTC), 30-36 t class — modelled on a Grove RT540E
(cross-checked against the Tadano GR-300EX / GR-550EX family).

Datasheet numbers (Grove RT540E product guide, rounded):
    max capacity 36 t (40 USt) · gross vehicle mass ≈ 29.7 t with the standard
    4.3 t counterweight · overall length (travel, boom over the front) ≈ 12.2 m ·
    overall width 2.54 m · overall height ≈ 3.41 m · wheelbase 3.35 m ·
    2 axles, 4 × 20.5-25 tyres (OD ≈ 1.49 m, section 0.52 m), 4WD with
    front / rear / crab steer · ground clearance ≈ 0.45 m · max travel speed
    40 km/h (11.1 m/s), 6 forward / 6 reverse gears · 4-section full-power
    boom 9.4-31.0 m (three telescoping sections, 7.2 m stroke each) · boom
    elevation -3°..+78° · max tip height ≈ 33.5 m · 360° continuous slew,
    0-2.5 rpm · main hoist line speed up to 152 m/min single line ·
    tail-swing radius ≈ 3.9 m · cab on the LEFT of the boom (Tadano GR
    machines put it on the right).  The GR-550EX is a larger 55 t /
    5-section (10-43 m) sibling; the model follows the 30-36 t dimensions.
Assumed / typical for the class:
    tread (tyre centre to centre) 2.02 m → width over tyres 2.54 m · carrier
    deck 1.65 m above ground, slewing-ring top 1.85 m · boom foot pin 3.0 m above
    ground, 1.55 m behind the slew axis · base section 0.72 × 0.86 m ·
    single lift cylinder (≈ 3.6 m closed, 5.0 m extended) · 2-part hoist
    reeving, hook block ≈ 0.4 t · max virtual steer 0.55 rad (inner wheel 37°,
    2-wheel-steer only — crab / 4-wheel steer are not modelled) · steer rate
    0.3 rad/s · counterweight 4.3 t in the superstructure link · optional
    bi-fold lattice jib shown stowed on the right of the base section (visual).

Joint ranges / rates (rad, m; per second at full command):
    slew           continuous, 0.26 rad/s (2.5 rpm, datasheet)
    boom_luff      -0.05..+1.35 rad (-3°..+77°); 0.10 rad/s
    boom_telescope 0..7.2 m per section (three sections in lock-step →
                   9.4-31.0 m boom); 0.50 m/s per section (1.5 m/s at the tip)
    hoist          rope length 0.8..14.0 m below the head sheave; 1.0 m/s
                   (2-part reeving at ≈ 120 m/min line speed)
    Luff and telescope rates are 2-5 × the real machine's (luff ≈ 35 s,
    full telescope ≈ 75 s) because the environment tests require every tool
    to traverse its full range within 15 s.

Init pose = travel pose: superstructure facing forward, boom fully retracted
and lowered to -3° over the front of the carrier, hook block hoisted up to
0.8 m below the head sheave (hook 0.3 m above ground, 3.9 m ahead of the
front bumper).  Model check (offline FK): L × W × H 12.22 × 2.58 × 3.45 m
(width over the hub caps), head-sheave height 33.2 m at 77° fully extended
(datasheet tip height ≈ 33.5 m), tail-swing radius 3.85 m, mass 29.7 t.

Cosmetic followers: telescope sections 3 and 4 follow section 2 (k = 1); the
hook swivel counter-rotates the luff angle (k = -1) so the hook block and
the rope hang vertically; nine rope segments slide at fractions of the rope
length so the two falls stay continuous from the head sheave to the block;
the lift-cylinder barrel and rod follow linear fits of their true angles
(< 2° error).

Root frame: carrier, origin midway between the axles at axle height
(z = 0 is 0.745 m above the ground), +x towards the steered front axle.
"""

from __future__ import annotations

import math

from ..spec import (Camera, Drive, Joint, Link, MachineSpec, Shape, Tool, beam, box, cyl,
                    sphere, wheel_joint)

# --- Carrier ------------------------------------------------------------------------
WHEEL_R, TYRE_W = 0.745, 0.52        # 20.5-25
TREAD = 2.02                         # tyre centre to tyre centre
WHEELBASE = 3.35
X_FRONT, X_REAR = WHEELBASE / 2, -WHEELBASE / 2
DECK_BOT, DECK_TOP = 0.80, 0.90      # carrier deck (1.545-1.645 m above ground)
SLEW_Z = 1.10                        # top of the slewing ring (1.845 m above ground)

# --- Superstructure (z' = 0 at the top of the slewing ring) ---------------------------
FOOT = (-1.55, 0.0, 1.15)            # boom foot pin
LIFT_BASE = (-0.95, 0.25)            # lift-cylinder base pin (x', z')
CAB = (-0.75, 0.95, 0.45, 1.25, 0.02, 1.42)   # x0, x1, y0, y1, z0, z1 (left of the boom)

# --- Boom (boom frame: origin at the foot pin, x along the boom) -------------------------
LUFF_LO, LUFF_HI = 0.0, 1.35         # datasheet -3°..+78°; see "Deviations" above
TELE_STROKE = 7.20                   # per section
LIFT_EYE = (4.20, -0.50)             # lift-cylinder eye under the base section
HEAD_PIN = (9.40, -0.24)             # head sheave pin (on the fly section, retracted): drop nose,
SHEAVE_R = 0.34                      # sheave hangs 0.32 m below the fly section like the real head
JIB_X = (2.30, 8.45)                 # stowed jib extent along the base section
# (inner end x, outer end x, width, depth) of each section in its own frame
SECTIONS = [(-0.45, 8.85, 0.72, 0.86), (0.65, 8.95, 0.62, 0.74),
            (0.75, 9.05, 0.53, 0.63), (0.85, 9.15, 0.45, 0.53)]

# --- Hoist ---------------------------------------------------------------------------
ROPE_MIN, ROPE_MAX = 0.80, 14.0      # head-sheave pin height → hook-block top
HOOK_X = SHEAVE_R                    # falls leave the FRONT of the head sheave (level frame), so the
                                     # block hangs ahead of the boom nose at every luff angle
FALL_Y, ROPE_R = 0.06, 0.022         # the two falls side by side at y = ±FALL_Y
SEG_LEN = 1.45                       # each rope piece (both falls)
SEG_MID = SEG_LEN / 2                # piece centre depth at ROPE_MIN: sheave tangent → into the block
N_SEG = 9                            # sliding rope segments: gaps stay closed up to ROPE_MAX


# ---------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------

def rod(p0: tuple[float, float], p1: tuple[float, float], radius: float, y: float = 0.0,
        mat: str = "steel", collision: bool = False) -> Shape:
    """Cylinder from p0 to p1 in the link's x-z plane."""
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    return cyl(radius, math.hypot(dx, dz), at=((p0[0] + p1[0]) / 2, y, (p0[1] + p1[1]) / 2),
               rpy=(0.0, math.atan2(dx, dz), 0.0), mat=mat, collision=collision)


def _rot(p: tuple[float, float], q: float) -> tuple[float, float]:
    """Rotate an x-z point by a joint with axis (0, -1, 0) at angle q (raises +x)."""
    c, s = math.cos(q), math.sin(q)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


def _linear_fit(qs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares slope, minimax offset; returns (k, c, max |error|)."""
    n = len(qs)
    mq, my = sum(qs) / n, sum(ys) / n
    k = sum((q - mq) * (y - my) for q, y in zip(qs, ys)) / sum((q - mq) ** 2 for q in qs)
    err = [y - k * q for q, y in zip(qs, ys)]
    return k, (max(err) + min(err)) / 2, (max(err) - min(err)) / 2


def _lift_cylinder_fits() -> dict:
    """Barrel elevation (superstructure frame) and rod elevation (boom frame) vs luff."""
    qs = [LUFF_LO + (LUFF_HI - LUFF_LO) * i / 140 for i in range(141)]
    phis, lengths = [], []
    for q in qs:
        e = _rot(LIFT_EYE, q)
        dx, dz = FOOT[0] + e[0] - LIFT_BASE[0], FOOT[2] + e[1] - LIFT_BASE[1]
        phis.append(math.atan2(dz, dx))
        lengths.append(math.hypot(dx, dz))
    return {"barrel": _linear_fit(qs, phis),
            "rod": _linear_fit(qs, [p - q for p, q in zip(phis, qs)]),
            "length": (min(lengths), max(lengths))}


def _follower(name: str, jtype: str, parent: str, child: str, xyz, axis, src: Joint,
              k: float, c: float, margin: float = 0.01) -> Joint:
    ends = (k * src.lower + c, k * src.upper + c)
    return Joint(name, jtype, parent, child, xyz=tuple(xyz), axis=axis,
                 lower=min(ends) - margin, upper=max(ends) + margin, init=k * src.init + c,
                 velocity=abs(k) * src.velocity + 0.05, role="follower",
                 follows=(src.name, k, c))


def _tyre(name: str, mass: float) -> Link:
    """20.5-25 tyre: collision cylinder at the rolling radius, tread lugs, rim, hub."""
    shapes = [
        cyl(WHEEL_R, TYRE_W, axis="y", mat="rubber", visual=False),
        cyl(WHEEL_R - 0.03, TYRE_W, axis="y", mat="rubber", collision=False),
        cyl(0.36, TYRE_W + 0.01, axis="y", mat="hub", collision=False),          # 25" rim
        cyl(0.20, TYRE_W + 0.02, axis="y", mat="dark", collision=False),         # hub
        box(0.62, TYRE_W + 0.03, 0.10, mat="accent", collision=False),           # spoke bar
    ]
    n = 16
    for row, dy in enumerate((-0.13, 0.13)):
        for i in range(n):
            th = 2 * math.pi * (i + 0.5 * row) / n
            r = WHEEL_R - 0.03
            shapes.append(box(0.15, 0.23, 0.06, at=(r * math.cos(th), dy, r * math.sin(th)),
                              rpy=(0.0, math.pi / 2 - th, 0.0), mat="rubber", collision=False))
    return Link(name, mass, shapes)


def _cab() -> list[Shape]:
    """Operator cab on the left of the boom (superstructure frame)."""
    x0, x1, y0, y1, z0, z1 = CAB
    cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    lx, ly, lz = x1 - x0, y1 - y0, z1 - z0
    return [
        box(lx, ly, lz, at=(cx, cy, cz)),                                             # shell
        box(lx + 0.06, ly + 0.04, 0.06, at=(cx, cy, z1 + 0.03), mat="accent"),         # roof
        box(0.10, ly + 0.02, 0.08, at=(x1 + 0.04, cy, z1 - 0.02), mat="accent", collision=False),  # visor
        # glass (visual only), slightly proud of the shell
        box(0.03, ly - 0.10, 0.92, at=(x1 + 0.015, cy, z0 + 0.88), mat="glass", collision=False),
        box(0.03, ly - 0.10, 0.30, at=(x1 + 0.015, cy, z0 + 0.22), mat="glass", collision=False),
        box(0.85, 0.03, 0.95, at=(x1 - 0.55, y1 + 0.015, z0 + 0.85), mat="glass", collision=False),
        box(0.50, 0.03, 0.62, at=(x0 + 0.35, y1 + 0.015, z0 + 0.98), mat="glass", collision=False),
        box(1.35, 0.03, 0.72, at=(cx + 0.08, y0 - 0.015, z0 + 0.98), mat="glass", collision=False),
        box(0.03, ly - 0.20, 0.55, at=(x0 - 0.015, cy, z0 + 1.02), mat="glass", collision=False),
        box(0.95, ly - 0.16, 0.03, at=(cx + 0.25, cy, z1 + 0.065), mat="glass", collision=False),  # roof window
        # pillars, door frame, handle, wiper
        box(0.05, 0.035, lz - 0.08, at=(x1 - 0.02, y1 + 0.02, cz), mat="accent", collision=False),
        box(0.05, 0.035, lz - 0.08, at=(x1 - 0.02, y0 - 0.02, cz), mat="accent", collision=False),
        box(0.05, 0.035, lz - 0.20, at=(x1 - 1.00, y1 + 0.02, cz), mat="accent", collision=False),
        box(0.10, 0.04, 0.03, at=(x1 - 0.90, y1 + 0.03, z0 + 0.62), mat="accent", collision=False),
        box(0.02, 0.03, 0.55, at=(x1 + 0.035, cy + 0.05, z0 + 0.72), rpy=(0.45, 0.0, 0.0),
            mat="accent", collision=False),
        # mirror arm + mirror on the left front corner
        box(0.18, 0.03, 0.03, at=(x1 + 0.08, y1 - 0.02, z0 + 1.05), mat="accent", collision=False),
        box(0.03, 0.12, 0.26, at=(x1 + 0.17, y1 - 0.01, z0 + 1.00), mat="accent", collision=False),
        # work lights on the roof front edge
        box(0.08, 0.14, 0.09, at=(x1 + 0.02, cy - 0.22, z1 + 0.10), mat="white", collision=False),
        box(0.08, 0.14, 0.09, at=(x1 + 0.02, cy + 0.22, z1 + 0.10), mat="white", collision=False),
        # step under the door
        box(0.40, 0.18, 0.04, at=(x1 - 0.55, y1 - 0.08, z0 - 0.02), mat="steel", collision=False),
    ]


def _section(i: int) -> list[Shape]:
    """Telescopic boom section ``i`` (0 = base) in its own frame: octagon-ish box + end collar."""
    x0, x1, w, d = SECTIONS[i]
    cx, lx = (x0 + x1) / 2, x1 - x0
    return [
        box(lx, w, d - 0.16, at=(cx, 0.0, 0.0)),                                       # webs
        box(lx, w - 0.16, d, at=(cx, 0.0, 0.0), collision=False),                      # flanges
        box(0.22, w + 0.06, d + 0.06, at=(x1 - 0.11, 0.0, 0.0), collision=False),      # wear-pad collar
        box(0.12, w + 0.07, 0.04, at=(x1 - 0.11, 0.0, d / 2 + 0.01), mat="accent", collision=False),
    ]


def _stowed_jib() -> list[Shape]:
    """Bi-fold lattice swingaway extension stowed on the right of the base section (visual)."""
    x0, x1 = JIB_X
    y_in, y_out, zt, zb = -0.43, -0.79, 0.20, -0.20
    lx, n = x1 - x0, 10
    shapes = [cyl(0.035, lx, at=((x0 + x1) / 2, y, z), axis="x", collision=False)
              for y in (y_in, y_out) for z in (zt, zb)]
    pitch = lx / n
    for i in range(n):
        xa, xb = x0 + i * pitch, x0 + (i + 1) * pitch
        za, zb_ = (zt, zb) if i % 2 == 0 else (zb, zt)
        shapes.append(rod((xa, za), (xb, zb_), 0.022, y=y_out, mat="body"))            # side lacing
        ya, yb = (y_in, y_out) if i % 2 == 0 else (y_out, y_in)
        shapes.append(box(math.hypot(pitch, yb - ya), 0.03, 0.03, at=((xa + xb) / 2, (ya + yb) / 2, zt),
                          rpy=(0.0, 0.0, math.atan2(yb - ya, pitch)), collision=False))  # top lacing
    shapes += [
        box(0.30, abs(y_out - y_in) + 0.08, zt - zb + 0.08, at=(x1 - 0.15, (y_in + y_out) / 2, 0.0),
            collision=False),                                                           # jib head
        cyl(0.14, 0.10, at=(x1 - 0.10, (y_in + y_out) / 2, 0.05), axis="y", mat="steel",
            collision=False),
        *[box(0.16, 0.10, 0.30, at=(x, -0.39, 0.0), mat="dark", collision=False)        # stow brackets
          for x in (x0 + 0.40, x1 - 0.70)],
    ]
    return shapes


def build() -> MachineSpec:
    y_t = TREAD / 2
    z_deck = (DECK_BOT + DECK_TOP) / 2

    # ------------------------------------------------------------------ carrier
    out_x = 3.60                                                     # outrigger box centres (±)
    carrier_shapes = [
        box(6.50, 0.84, DECK_BOT + 0.12, at=(0.0, 0.0, (DECK_BOT - 0.12) / 2), mat="dark"),  # frame
        box(6.50, 2.54, DECK_TOP - DECK_BOT, at=(0.0, 0.0, z_deck)),                        # deck
        cyl(0.95, 0.20, at=(0.0, 0.0, SLEW_Z - 0.10), mat="dark"),                          # slewing ring
        cyl(0.85, 0.04, at=(0.0, 0.0, SLEW_Z - 0.02), mat="steel", collision=False),
        # axles + differentials
        *[s for x in (X_FRONT, X_REAR) for s in (
            cyl(0.16, 1.45, at=(x, 0.0, 0.0), axis="y", mat="dark"),
            sphere(0.27, at=(x, 0.0, -0.04), mat="dark"),
            box(0.55, 0.84, 0.30, at=(x, 0.0, 0.20), mat="dark"))],
        cyl(0.07, 2.9, at=(0.0, 0.0, -0.06), axis="x", mat="steel", collision=False),       # drive shaft
        # outrigger boxes (front / rear) with jack housings, stowed jacks and pads
        *[s for sx in (1.0, -1.0) for s in (
            box(0.70, 2.40, 0.70, at=(sx * out_x, 0.0, 0.55)),
            *[t for sy in (1.0, -1.0) for t in (
                box(0.38, 0.20, 0.95, at=(sx * out_x, sy * 1.16, 0.40)),
                cyl(0.08, 0.30, at=(sx * out_x, sy * 1.16, -0.20), mat="steel"),
                box(0.36, 0.22, 0.05, at=(sx * out_x, sy * 1.16, -0.37), mat="dark"),
                box(0.02, 0.18, 0.40, at=(sx * (out_x + 0.20), sy * 1.16, 0.55), mat="accent",
                    collision=False))])],
        # bumpers, lights, hook-block tie-down
        box(0.12, 2.00, 0.28, at=(3.99, 0.0, 0.30), mat="dark"),
        box(0.12, 2.00, 0.28, at=(-3.99, 0.0, 0.30), mat="dark"),
        *[box(0.04, 0.22, 0.14, at=(3.97, sy * 0.80, 0.70), mat="white", collision=False)
          for sy in (1.0, -1.0)],
        *[box(0.04, 0.10, 0.10, at=(3.97, sy * 0.55, 0.70), mat="orange", collision=False)
          for sy in (1.0, -1.0)],
        *[box(0.04, 0.22, 0.12, at=(-3.97, sy * 0.80, 0.70), mat="red", collision=False)
          for sy in (1.0, -1.0)],
        *[box(0.04, 0.10, 0.10, at=(-3.97, sy * 0.55, 0.70), mat="orange", collision=False)
          for sy in (1.0, -1.0)],
        box(0.10, 0.20, 0.20, at=(4.07, 0.0, 0.30), mat="steel", collision=False),
        # side tanks between the axles: fuel (left), hydraulic oil / batteries (right)
        box(1.60, 0.70, 0.62, at=(0.0, 0.86, 0.46), mat="dark"),
        box(1.60, 0.70, 0.62, at=(0.0, -0.86, 0.46)),
        box(1.20, 0.02, 0.30, at=(0.0, -1.22, 0.50), mat="accent", collision=False),
        *[box(0.45, 0.10, 0.04, at=(0.45, 1.22, z), mat="steel", collision=False)
          for z in (0.25, 0.55)],
        cyl(0.07, 0.06, at=(-0.45, 0.86, 0.79), mat="steel", collision=False),             # filler cap
        # engine bay at the rear: side grilles under the deck, raised hood on the right
        box(0.70, 2.44, 0.60, at=(-2.85, 0.0, 0.50)),
        *[box(0.50, 0.02, 0.40, at=(-2.85, sy * 1.225, 0.50), mat="accent", collision=False)
          for sy in (1.0, -1.0)],
        box(1.00, 0.90, 0.16, at=(-2.70, -0.75, DECK_TOP + 0.08)),
        box(0.80, 0.70, 0.02, at=(-2.70, -0.75, DECK_TOP + 0.17), mat="accent", collision=False),
        cyl(0.06, 0.08, at=(-2.40, -0.35, DECK_TOP + 0.04), mat="dark", collision=False),
        # anti-skid strips on the front deck
        *[box(0.50, 0.35, 0.01, at=(2.80, sy * 0.80, DECK_TOP + 0.005), mat="accent",
              collision=False) for sy in (1.0, -1.0)],
    ]
    carrier = Link("carrier", 11450.0, carrier_shapes)

    # ---------------------------------------------------------- superstructure
    fx, _, fz = FOOT
    superstructure = Link("superstructure", 8700.0, [
        box(3.35, 1.90, 0.30, at=(-0.675, 0.0, 0.15)),                                 # turntable
        box(0.60, 1.60, 0.30, at=(-2.65, 0.0, 0.15)),                                  # rear extension
        # boom-foot towers, pin bosses, foot pin
        *[s for sy in (1.0, -1.0) for s in (
            box(1.00, 0.08, 1.00, at=(fx, sy * 0.42, 0.80)),
            beam((fx + 0.45, 1.05), (fx + 1.35, 0.30), 0.08, 0.30, y=sy * 0.42),
            cyl(0.30, 0.08, at=(fx, sy * 0.42, fz), axis="y"))],
        cyl(0.10, 1.00, at=FOOT, axis="y", mat="steel", collision=False),
        # lift-cylinder base bracket
        box(0.45, 0.50, 0.30, at=(LIFT_BASE[0], 0.0, LIFT_BASE[1]), mat="dark", collision=False),
        # counterweight (4.3 t) with rounded rear corners and a black band
        box(0.80, 2.00, 1.00, at=(-3.30, 0.0, 0.55)),
        cyl(0.30, 1.00, at=(-3.40, 0.95, 0.55)),
        cyl(0.30, 1.00, at=(-3.40, -0.95, 0.55)),
        box(0.03, 1.60, 0.22, at=(-3.715, 0.0, 0.70), mat="accent", collision=False),
        *[box(0.03, 0.14, 0.10, at=(-3.715, sy * 0.90, 0.20), mat="red", collision=False)
          for sy in (1.0, -1.0)],
        # main hoist winch behind the boom foot
        cyl(0.30, 0.78, at=(-2.58, 0.0, 0.92), axis="y", mat="dark"),
        *[cyl(0.38, 0.04, at=(-2.58, sy * 0.41, 0.92), axis="y", collision=False)
          for sy in (1.0, -1.0)],
        *[box(0.70, 0.06, 0.55, at=(-2.58, sy * 0.47, 0.62)) for sy in (1.0, -1.0)],
        cyl(0.13, 0.30, at=(-2.58, -0.65, 0.92), axis="y", mat="dark", collision=False),  # hoist motor
        *[cyl(0.305, 0.03, at=(-2.58, y, 0.92), axis="y", mat="steel", collision=False)
          for y in (-0.25, -0.05, 0.15, 0.35)],                                          # rope wraps
        # right-hand hydraulic tank / valve housing with grille and handrail
        box(2.30, 0.60, 0.80, at=(-0.75, -0.85, 0.40)),
        box(1.20, 0.02, 0.40, at=(-0.90, -1.16, 0.45), mat="accent", collision=False),
        box(1.80, 0.04, 0.04, at=(-0.75, -1.12, 1.10), mat="accent", collision=False),
        *[box(0.04, 0.04, 0.30, at=(x, -1.12, 0.95), mat="accent", collision=False)
          for x in (-1.60, 0.10)],
        cyl(0.08, 0.10, at=(-1.60, -0.85, 0.85), mat="orange", collision=False),           # beacon
        box(0.06, 0.16, 0.10, at=(0.42, -0.85, 0.75), mat="white", collision=False),       # work light
        *_cab(),
    ])

    # ---------------------------------------------------------------- joints
    slew_j = Joint("slew_joint", "continuous", "carrier", "superstructure",
                   xyz=(0.0, 0.0, SLEW_Z), axis=(0, 0, 1), velocity=0.26, role="tool")
    luff_j = Joint("boom_luff_joint", "revolute", "superstructure", "boom_base",
                   xyz=FOOT, axis=(0, -1, 0), lower=LUFF_LO, upper=LUFF_HI, init=LUFF_LO,
                   velocity=0.10, role="tool")
    tele_j = Joint("boom_telescope_joint", "prismatic", "boom_base", "boom_sec2",
                   axis=(1, 0, 0), lower=0.0, upper=TELE_STROKE, init=0.0, velocity=0.50,
                   role="tool")
    hoist_j = Joint("hoist_joint", "prismatic", "hook_swivel", "hook_block",
                    xyz=(HOOK_X, 0.0, 0.0), axis=(0, 0, -1), lower=ROPE_MIN, upper=ROPE_MAX,
                    init=ROPE_MIN, velocity=1.0, role="tool")
    fits = _lift_cylinder_fits()
    l_min, l_max = fits["length"]
    barrel_len = l_min - 0.25
    rod_len = l_max - barrel_len + 0.30

    # ------------------------------------------------------------------- boom
    boom_base = Link("boom_base", 3350.0, [
        *_section(0),
        *_stowed_jib(),
        cyl(0.22, 0.76, axis="y", mat="steel", collision=False),                      # foot boss
        box(0.40, 0.34, 0.18, at=(LIFT_EYE[0], 0.0, LIFT_EYE[1] + 0.12), mat="dark",
            collision=False),                                                            # eye lugs
        cyl(0.20, 0.10, at=(1.10, -0.42, 0.05), axis="y", mat="dark", collision=False),  # A2B cable reel
        box(0.10, 0.06, 0.08, at=(7.8, 0.40, 0.25), mat="white", collision=False),       # boom light
    ])
    hx, hz = HEAD_PIN
    top, bot = SECTIONS[3][3] / 2 + 0.04, hz - SHEAVE_R - 0.04       # head cheek plates (z)
    sec4_shapes = _section(3) + [
        *[box(0.70, 0.06, top - bot, at=(hx - 0.05, sy * 0.19, (top + bot) / 2)) for sy in (1.0, -1.0)],
        box(0.64, 0.44, 0.06, at=(hx - 0.07, 0.0, top), collision=False),                # head cover
        cyl(SHEAVE_R, 0.24, at=(hx, 0.0, hz), axis="y", mat="steel", collision=False),  # head sheave
        cyl(0.07, 0.52, at=(hx, 0.0, hz), axis="y", mat="dark", collision=False),        # sheave pin
        cyl(0.02, 0.30, at=(hx - 0.20, 0.0, top + 0.18), mat="accent", collision=False),  # anemometer
        sphere(0.05, at=(hx - 0.20, 0.0, top + 0.34), mat="accent", collision=False),
        box(0.04, 0.10, 0.10, at=(hx + 0.32, 0.15, top - 0.10), mat="red", collision=False),
    ]
    boom_links = [
        boom_base,
        Link("boom_sec2", 1250.0, _section(1)),
        Link("boom_sec3", 1000.0, _section(2)),
        Link("boom_sec4", 850.0, sec4_shapes),
    ]

    # ------------------------------------------------- lift cylinder followers
    lift_barrel = Link("lift_cylinder", 420.0, [
        cyl(0.17, barrel_len, at=(barrel_len / 2, 0.0, 0.0), axis="x", collision=False),
        cyl(0.20, 0.14, at=(barrel_len - 0.07, 0.0, 0.0), axis="x", mat="dark", collision=False),
        cyl(0.14, 0.42, axis="y", mat="steel", collision=False),                         # base eye
    ])
    lift_rod = Link("lift_cylinder_rod", 230.0, [
        cyl(0.11, rod_len, at=(-rod_len / 2, 0.0, 0.0), axis="x", mat="steel", collision=False),
        cyl(0.13, 0.36, axis="y", mat="steel", collision=False),                         # rod eye
    ])

    # ------------------------------------------------------ hook, swivel, rope
    def falls(z: float, x: float = 0.0) -> list[Shape]:
        return [cyl(ROPE_R, SEG_LEN, at=(x, sy * FALL_Y, z), mat="steel", collision=False)
                for sy in (1.0, -1.0)]

    # rope pieces must overlap at full pay-out: spacing (ROPE_MAX - ROPE_MIN) / (N_SEG + 1)
    assert (ROPE_MAX - ROPE_MIN) / (N_SEG + 1) <= SEG_LEN - 0.05, "add rope segments"
    swivel = Link("hook_swivel", 20.0, falls(-SEG_MID, HOOK_X))
    # The block is visual only (no collision): positions are written kinematically and
    # there is no load physics, so a block paid out into the ground must not fight it.
    hook_block = Link("hook_block", 380.0, [
        *[box(0.62, 0.06, 0.70, at=(0.0, sy * 0.14, -0.38), collision=False)             # side plates
          for sy in (1.0, -1.0)],
        *[box(0.50, 0.065, 0.10, at=(0.0, sy * 0.14, -0.38), mat="accent", collision=False)
          for sy in (1.0, -1.0)],
        cyl(0.25, 0.18, at=(0.0, 0.0, -0.35), axis="y", mat="steel", collision=False),  # sheave
        cyl(0.06, 0.40, at=(0.0, 0.0, -0.35), axis="y", mat="dark", collision=False),
        box(0.40, 0.34, 0.14, at=(0.0, 0.0, -0.80), mat="dark", collision=False),        # trunnion
        cyl(0.075, 0.18, at=(0.0, 0.0, -0.95), mat="steel", collision=False),           # hook nut / shank
        box(0.10, 0.12, 0.30, at=(-0.13, 0.0, -1.12), mat="orange", collision=False),   # hook back
        box(0.34, 0.12, 0.10, at=(0.0, 0.0, -1.24), mat="orange", collision=False),     # hook bowl
        box(0.08, 0.11, 0.16, at=(0.13, 0.0, -1.15), mat="orange", collision=False),    # hook tip
        box(0.20, 0.10, 0.08, at=(-0.06, 0.0, -1.01), mat="orange", collision=False),
        box(0.02, 0.04, 0.24, at=(0.06, 0.0, -1.03), rpy=(0.0, 0.5, 0.0), mat="steel",
            collision=False),                                                              # latch
        *falls(ROPE_MIN - SEG_MID),
    ])
    rope_links, rope_joints = [], []
    for i in range(1, N_SEG + 1):
        b = i / (N_SEG + 1)
        name = f"rope_seg_{i}"
        rope_links.append(Link(name, 6.0, falls(0.0)))
        rope_joints.append(_follower(f"{name}_joint", "prismatic", "hook_swivel", name,
                                     (HOOK_X, 0.0, 0.0), (0, 0, -1), hoist_j, b,
                                     SEG_MID - b * ROPE_MIN))

    # ------------------------------------------------------------------ wheels
    wheel_links, wheel_joints, knuckles, steer_joints = [], [], [], []
    for side, sy in (("left", 1.0), ("right", -1.0)):
        kn = f"front_{side}_knuckle"
        knuckles.append(Link(kn, 120.0, [
            cyl(0.15, 0.24, at=(0.0, -sy * 0.39, 0.0), axis="y", mat="dark"),
            cyl(0.09, 0.40, at=(0.0, -sy * 0.42, 0.0), mat="dark", collision=False),     # king pin
        ]))
        steer_joints.append(Joint(f"front_{side}_steer_joint", "revolute", "carrier", kn,
                                  xyz=(X_FRONT, sy * y_t, 0.0), axis=(0, 0, 1),
                                  lower=-0.70, upper=0.70, velocity=0.6, role="steer"))
        for pos, parent, xyz in (("front", kn, (0.0, 0.0, 0.0)),
                                 ("rear", "carrier", (X_REAR, sy * y_t, 0.0))):
            name = f"{pos}_{side}_wheel"
            wheel_links.append(_tyre(name, 450.0))
            wheel_joints.append(wheel_joint(f"{name}_joint", parent, name, xyz, WHEEL_R))

    links = [carrier, superstructure, *boom_links, lift_barrel, lift_rod, swivel, hook_block,
             *rope_links, *knuckles, *wheel_links]

    kb, cb, _ = fits["barrel"]
    kr, cr, _ = fits["rod"]
    joints = [
        slew_j,
        luff_j,
        tele_j,
        _follower("boom_sec3_joint", "prismatic", "boom_sec2", "boom_sec3", (0.0, 0.0, 0.0),
                  (1, 0, 0), tele_j, 1.0, 0.0),
        _follower("boom_sec4_joint", "prismatic", "boom_sec3", "boom_sec4", (0.0, 0.0, 0.0),
                  (1, 0, 0), tele_j, 1.0, 0.0),
        _follower("hook_swivel_joint", "revolute", "boom_sec4", "hook_swivel",
                  (hx, 0.0, hz), (0, -1, 0), luff_j, -1.0, 0.0),
        hoist_j,
        *rope_joints,
        _follower("lift_cylinder_joint", "revolute", "superstructure", "lift_cylinder",
                  (LIFT_BASE[0], 0.0, LIFT_BASE[1]), (0, -1, 0), luff_j, kb, cb, 0.05),
        _follower("lift_cylinder_rod_joint", "revolute", "boom_base", "lift_cylinder_rod",
                  (LIFT_EYE[0], 0.0, LIFT_EYE[1]), (0, -1, 0), luff_j, kr, cr, 0.05),
        *steer_joints,
        *wheel_joints,
    ]

    return MachineSpec(
        name="rough_terrain_crane",
        abbrev="RTC",
        display_name="Rough Terrain Crane",
        category="lifting",
        reference="36 t rough-terrain crane (Grove RT540E / Tadano GR-300EX class)",
        description=("Two-axle rough-terrain mobile crane with large equal tyres, front-wheel "
                     "Ackermann steering, outrigger boxes front and rear, a 360° superstructure "
                     "with the cab left of the boom, a 4-section 9.4-31 m telescopic boom and "
                     "a 2-fall hook block."),
        root="carrier",
        links=links,
        joints=joints,
        drive=Drive(type="ackermann", max_speed=11.1, wheelbase=WHEELBASE, max_steer=0.55,
                    steer_rate=0.30, fixed_axle_x=X_REAR, steer_axle="front"),
        tools=[
            Tool("slew_joint", "slew", positive="left", negative="right"),
            Tool("boom_luff_joint", "boom_luff", positive="raise", negative="lower"),
            Tool("boom_telescope_joint", "boom_telescope", positive="extend", negative="retract"),
            Tool("hoist_joint", "hoist", positive="lower", negative="raise"),
        ],
        cameras=[
            # operator eye height just ahead of the windscreen, looking along the boom; only
            # slightly down so the hook stays in view when it is lifted (up to ≈ 40° elevation)
            Camera("front", "superstructure", (CAB[1] + 0.04, 0.85, 1.05), (0.0, 0.08, -0.10)),
            # front-left corner of the cab roof / front-right corner of the hydraulic tank
            Camera("left", "superstructure", (CAB[1] + 0.03, CAB[3] + 0.02, CAB[5] + 0.08),
                   (0.0, 0.30, 0.65)),
            Camera("right", "superstructure", (0.42, -1.17, 0.95), (0.0, 0.30, -0.65)),
        ],
        envelope=(12.2, 2.54, 3.41),
        operating_mass=29700.0,
        # Telescopic boom sections nest inside each other when retracted (the base
        # section surrounds sections 3 and 4, section 2 surrounds section 4).
        allowed_overlaps=[("boom_base", "boom_sec3"), ("boom_base", "boom_sec4"),
                          ("boom_sec2", "boom_sec4")],
        notes=("Hoist +1 pays out rope (hook down). Telescope sections 3/4, the hook swivel, "
               "rope segments and the lift cylinder are cosmetic followers. Outriggers are "
               "shown stowed and are not actuated."),
    )
