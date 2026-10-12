"""Wheel tractor-scraper (WTS), 15 m³ class — modelled on a CAT 621K.

Reference numbers (CAT 621K class datasheet values, rounded; "≈" = estimated
from drawings / photos where no datasheet figure was to hand):
    operating mass (empty) 33.2 t · weight split empty ≈ 67 % front / 33 % rear ·
    rated load ≈ 22 t · bowl 15.3 m³ heaped / 10.7 m³ struck · single engine
    (Cat C13, ≈ 300 kW) in a single-axle tractor · 33.25R29 tyres on both axles
    (Ø 2.06 m, section 0.85 m) · overall length ≈ 14.0 m · overall width
    ≈ 3.75 m (over the draft arms) · height to the top of the cab ROPS ≈ 3.85 m ·
    wheelbase ≈ 8.9 m · front axle → hitch ≈ 1.45 m · cutting width 3.02 m ·
    max depth of cut 0.32 m · ground clearance under the cutting edge ≈ 0.45 m
    (bowl raised) · apron opening ≈ 1.8 m at the lip · ejector stroke ≈ 2.1 m ·
    steering 90° each way at the hitch · top speed 55 km/h (15.3 m/s), reverse
    ≈ 10 km/h.

Model (offline FK): L × W × H 13.63 × 3.94 × 3.91 m (length to the push block,
width over the trunnion caps, height to the beacon) · 33.4 t, 64 % on the front
axle empty (link-inertia COM 3.23 m behind the front axle; the datasheet's ≈ 67 %
would need ≈ 1.2 t moved from the bowl to the tractor, not modelled — the env is
kinematic) · wheelbase 8.90 m (hinge → front axle 1.45 m, hinge → rear axle
7.45 m) · rear gauge 2.60 m (3.45 m over the rear tyres), front gauge 2.45 m ·
bowl inside 3.02 m wide, floor 2.9 m long, sides 2.4 m high · cutting edge 0.45 m
above grade in travel, 0.32 m below at full lower · apron lip 2.33 m above
grade (1.9 m above the edge) fully open.

Joint ranges / rates used:
    steering hitch  ±0.90 rad (the real hitch turns ±90°; the model caps it at
                    0.9 rad = 52° → front-axle turning radius 10.7 m at full
                    lock), 0.40 rad/s (lock to lock in 4.5 s)
    bowl            -0.194 (cutting edge 0.32 m below grade) … 0 (travel, edge
                    0.45 m above grade) about the rear axle; 0.09 rad/s
                    (full raise ≈ 2.2 s)
    apron           0 (closed on the cutting edge) … 0.55 rad (open);
                    0.30 rad/s (≈ 1.8 s)
    ejector         0 … 2.10 m forward (towards the cutting edge); 0.35 m/s
                    (full eject in 6 s)

Init pose = travel pose: frames straight, bowl raised, apron closed, ejector
back.

Layout: the root link is the TRACTOR (engine hood, centre cab, front axle).  The
steering hitch (vertical king pin 1.45 m behind the front axle) carries a small
hitch casting; from it hang the gooseneck/draft frame (swan neck, yoke cross
beam over the apron, draft arms down the bowl sides to trunnions at the rear of
the bowl) and the bowl, which carries the rear axle.  The tractor frame rails
stop 0.30 m ahead of the king pin and the hydraulic tank 0.30 m ahead of it, so
the casting (king-pin boss r 0.24 m, box corners ≤ 0.20 m forward at full lock)
turns freely between the upper and lower bearing plates.

Cameras: front = operator eye height ahead of the windscreen, looking down the
hood (driving view).  left / right = rear-view cameras on short posts at the
rear outer corners of the front fenders (agl 2.65 m, 1.62 m off centre), looking
back and 17° outward: they show the bowl front and apron (the apron lip closes
onto the cutting edge, so lowering the bowl shows as the lip dropping to grade),
the yoke and draft arms, and, with the apron open, the cutting edge and the
ejector through the bowl mouth.  At full lock the rear unit swings out of the
camera on the outside of the turn and fills the one on the inside.  Mirror-arm
and cab-roof positions were tried: they see about half as much tool motion,
because the yoke hides the bowl mouth from higher up.

Bowl linkage: on the real machine the bowl (with the rear axle) is pinned to the
draft arms at low rear trunnions and the lift cylinders between the yoke and
the bowl front tilt it; with both axles on the ground the bowl turns about the
rear axle and the tractor + draft frame pitch slightly.  The model keeps the
tractor level, turns the bowl exactly about the rear axle (the rear wheels stay
put) and lets the draft frame pitch at the hitch as a follower of the bowl
(2.7° over the full range) so the draft arms stay on the trunnions (≤ 4.5 cm
off, inside the trunnion caps).  The lift cylinders are barrel / rod follower
pairs pinned under the yoke ends and on lugs at the bowl front (minimax linear
fit, rod eye ≤ 1.6 cm off the lug); the ejector cylinder is telescopic (stage
follower at half the ejector stroke).  Cylinders and pin details are visual
only.

Livery: CAT yellow, so DEFAULT_MATERIALS are used.

Root frame: front axle centre (z = 0 is the axle line, 1.03 m above the ground),
+x forward.
"""

from __future__ import annotations

import math

from ..spec import (Camera, Drive, Joint, Link, MachineSpec, Shape, Tool, beam, box, cyl,
                    sphere, wheel_joint, wheel_link)

# --- running gear (33.25R29) -------------------------------------------------------
WHEEL_R, TYRE_W = 1.03, 0.85
FRONT_GAUGE, REAR_GAUGE = 2.45, 2.60
WHEELBASE = 8.90
HITCH = (-1.45, 0.22)                 # king pin (x, z) in the tractor frame
FRONT_LEN = -HITCH[0]                 # hinge -> front axle
REAR_LEN = WHEELBASE - FRONT_LEN      # hinge -> rear axle (7.45 m)
Z0 = WHEEL_R                          # ground is at z = -Z0 in the root frame


def agl(h: float) -> float:
    """Height above ground (design pose) -> z relative to the axle line."""
    return h - Z0


# --- tractor -------------------------------------------------------------------------
BUMPER_X = 2.85
CAB_X0, CAB_X1, CAB_HW = -1.00, 0.25, 0.82
CAB_Z0, CAB_Z1 = agl(2.09), agl(3.74)
MIRROR = (0.18, 1.12, agl(3.05))       # left mirror head centre (right one mirrored)
REAR_CAM = (-0.70, 1.62, agl(2.65))    # left rear-view camera lens (right one mirrored)
FENDER_TOP = agl(2.20)                 # top of the flat front-fender plate (x -0.66 … 0.82)
HITCH_CLEAR = 0.30                     # frame rails / tank stop this far ahead of the king pin

# --- scraper: frames ----------------------------------------------------------------
# Hitch-casting frame = king pin; the gooseneck pitches about a cross pin behind it.
PD = (-0.62, 0.0)                      # gooseneck pitch pin (hitch frame)
RH = (-REAR_LEN, -HITCH[1])            # rear axle (= bowl pivot) in the hitch frame

# --- bowl (bowl frame: origin on the rear axle, x forward, z rel. axle line) ----------
CUT_X = 4.15                           # front of the cutting edge (4.75 m behind the front axle)
FLOOR_X0 = 1.25                        # rear end of the floor = ejector face when retracted
WALL_X0 = FLOOR_X0 - 0.13              # rear face of the fixed rear wall
FLOOR_BOT = agl(0.47)
FLOOR_TOP = FLOOR_BOT + 0.12
BOWL_TOP = agl(2.95)
IN_HW = 1.51                           # inside half width (cutting width 3.02 m)
WALL_Y = IN_HW + 0.03                  # side-wall plate centre (6 cm plate)
TAIL_X1 = -1.55                        # front face of the push block
PUSH_X1 = -1.85                        # rear face of the push block (overall length)
TRUNNION = (FLOOR_X0 + 0.12, agl(1.02))  # draft-arm trunnion on the bowl side
LIFT_LUG = (CUT_X - 0.35, agl(1.00))     # lift-cylinder rod eye on the bowl side
APRON_PIVOT = (CUT_X - 2.75, agl(2.70))  # apron-arm pivot on the bowl side
DRAFT_Y = 1.82                         # draft arm / lift cylinder plane
APRON_ARM_Y = 1.655

# --- gooseneck (gooseneck frame: origin on the pitch pin PD) --------------------------
KNEE = (-0.40, 1.28)
CROSS = (-1.20, 1.62)                  # yoke cross-beam centre
CROSS_HW = 1.91
LIFT_TOP = (-1.62, 1.55)               # lift-cylinder barrel pin (bracket on the draft-arm front)

# --- tools ------------------------------------------------------------------------------
DEPTH_OF_CUT = 0.32
APRON_MAX = 0.55
EJECT_STROKE = 2.10
EJ_Z = agl(1.70)                       # ejector-cylinder axis height


# ------------------------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------------------------

def _rot(p: tuple[float, float], q: float) -> tuple[float, float]:
    """Rotate an x-z point by a joint with axis (0, -1, 0) at angle q (+q tips +x up)."""
    c, s = math.cos(q), math.sin(q)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


def _angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Signed angle (axis (0, -1, 0) sense) that turns direction a onto direction b."""
    return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])


def _linear_fit(qs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares slope, minimax offset; returns (k, c, max |error|)."""
    n = len(qs)
    mq, my = sum(qs) / n, sum(ys) / n
    k = sum((q - mq) * (y - my) for q, y in zip(qs, ys)) / sum((q - mq) ** 2 for q in qs)
    err = [y - k * q for q, y in zip(qs, ys)]
    return k, (max(err) + min(err)) / 2, (max(err) - min(err)) / 2


def _bowl_lower() -> float:
    """Bowl angle that puts the cutting edge DEPTH_OF_CUT below grade."""
    edge = (CUT_X, FLOOR_BOT)
    target = -Z0 - DEPTH_OF_CUT
    lo, hi = -0.6, 0.0
    for _ in range(60):                      # bisection: edge height is monotonic in q
        mid = (lo + hi) / 2
        if _rot(edge, mid)[1] > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


BOWL_LO = _bowl_lower()


def _bowl_pt_hitch(p: tuple[float, float], q: float) -> tuple[float, float]:
    r = _rot(p, q)
    return (RH[0] + r[0], RH[1] + r[1])


def _goose_pt_hitch(p: tuple[float, float], a: float) -> tuple[float, float]:
    r = _rot(p, a)
    return (PD[0] + r[0], PD[1] + r[1])


def _to_goose(p_hitch: tuple[float, float], a: float) -> tuple[float, float]:
    return _rot((p_hitch[0] - PD[0], p_hitch[1] - PD[1]), -a)


# Trunnion in the gooseneck frame (design pose, bowl and gooseneck at 0).
TRUNNION_G = _to_goose(_bowl_pt_hitch(TRUNNION, 0.0), 0.0)


def _linkage_fit() -> dict:
    """Linear follower fits (vs. bowl angle) for the draft frame and lift cylinders.

    * draft frame pitch: the angle that keeps the draft-arm end on the bowl
      trunnion (the residual is the change of the hinge-trunnion distance);
    * lift cylinder: barrel angle (relative to the draft frame) and rod
      extension along the barrel so the rod eye follows the bowl lug.
    """
    qs = [BOWL_LO * i / 40 for i in range(41)]
    t0 = (TRUNNION_G[0], TRUNNION_G[1])
    ang = []
    for q in qs:
        th = _bowl_pt_hitch(TRUNNION, q)
        ang.append(_angle(t0, (th[0] - PD[0], th[1] - PD[1])))
    kd, cd, _ = _linear_fit(qs, ang)

    lug0 = _to_goose(_bowl_pt_hitch(LIFT_LUG, 0.0), cd)
    v0 = (lug0[0] - LIFT_TOP[0], lug0[1] - LIFT_TOP[1])
    psi, dist = [], []
    for q in qs:
        a = kd * q + cd
        lug = _to_goose(_bowl_pt_hitch(LIFT_LUG, q), a)
        v = (lug[0] - LIFT_TOP[0], lug[1] - LIFT_TOP[1])
        psi.append(_angle(v0, v))
        dist.append(math.hypot(*v))
    kp, cp, ep = _linear_fit(qs, psi)
    ks, cs, es = _linear_fit(qs, dist)
    # trunnion miss with the fitted draft-frame pitch
    miss = 0.0
    for q in qs:
        th = _bowl_pt_hitch(TRUNNION, q)
        tg = _goose_pt_hitch(TRUNNION_G, kd * q + cd)
        miss = max(miss, math.hypot(th[0] - tg[0], th[1] - tg[1]))
    return {"kd": kd, "cd": cd, "trunnion_miss": miss,
            "dir0": v0, "kp": kp, "cp": cp, "ep": ep, "ks": ks, "cs": cs, "es": es,
            "lift_err": max(ep * cs, es)}


# ------------------------------------------------------------------------------------
# Links
# ------------------------------------------------------------------------------------

def _cab() -> list[Shape]:
    """Centre-mounted ROPS cab above the front axle (tractor frame)."""
    x0, x1, y1, z0, z1 = CAB_X0, CAB_X1, CAB_HW, CAB_Z0, CAB_Z1
    cx, cz, lx, lz = (x0 + x1) / 2, (z0 + z1) / 2, x1 - x0, z1 - z0
    arm = (MIRROR[0] - (x1 - 0.02), MIRROR[1] - 0.04 - (y1 + 0.03))    # pillar -> mirror head
    return [
        box(lx, 2 * y1, lz, at=(cx, 0.0, cz)),                                          # shell
        box(lx + 0.14, 2 * y1 + 0.10, 0.09, at=(cx - 0.02, 0.0, z1 + 0.045)),           # roof
        cyl(0.06, 0.08, at=(x0 + 0.25, 0.0, z1 + 0.13), mat="orange", collision=False),  # beacon
        # glass (visual only), slightly proud of the shell
        box(0.03, 2 * y1 - 0.16, 1.05, at=(x1 + 0.01, 0.0, z0 + 1.02), mat="glass", collision=False),
        *[box(lx - 0.22, 0.03, 1.00, at=(cx, s * (y1 + 0.01), z0 + 1.02), mat="glass",
              collision=False) for s in (1.0, -1.0)],
        box(0.03, 2 * y1 - 0.30, 0.80, at=(x0 - 0.01, 0.0, z0 + 1.10), mat="glass", collision=False),
        # ROPS pillars / door frame
        *[box(0.07, 0.04, lz - 0.06, at=(x, s * (y1 + 0.02), cz), mat="accent", collision=False)
          for x in (x1 - 0.04, x0 + 0.04, cx - 0.05) for s in (1.0, -1.0)],
        # roof work lights; mirror arms + heads at the front corners
        *[box(0.08, 0.18, 0.09, at=(x1 + 0.03, s * 0.55, z1 - 0.05), mat="white", collision=False)
          for s in (1.0, -1.0)],
        *[box(math.hypot(*arm), 0.03, 0.03, at=(x1 - 0.02 + arm[0] / 2, s * (y1 + 0.03 + arm[1] / 2),
              MIRROR[2] + 0.17), rpy=(0.0, 0.0, s * math.atan2(arm[1], arm[0])), mat="accent",
              collision=False) for s in (1.0, -1.0)],
        *[box(0.05, 0.16, 0.30, at=(MIRROR[0], s * MIRROR[1], MIRROR[2]), mat="accent",
              collision=False) for s in (1.0, -1.0)],
    ]


def _tractor() -> Link:
    y_w = FRONT_GAUGE / 2
    hx = HITCH[0]
    rail_x0, rail_x1 = hx + HITCH_CLEAR, 2.525       # rails end in the hitch bearing plates
    tank_x0 = hx + HITCH_CLEAR                        # rear face of the hydraulic tank
    cx, cy, cz = REAR_CAM
    post_z1 = cz - 0.05                               # camera housing underside
    return Link("tractor", 13600.0, [
        box(rail_x1 - rail_x0, 0.90, 0.50, at=((rail_x0 + rail_x1) / 2, 0.0, agl(1.00))),  # frame rails
        cyl(0.24, 1.58, axis="y", mat="dark"),                                          # front axle
        sphere(0.34, mat="dark"),                                                       # differential
        # engine hood: lower box + sloping top, radiator grille, bumper
        box(2.20, 1.56, 0.75, at=(1.35, 0.0, agl(1.62))),
        beam((0.25, agl(2.06)), (2.47, agl(1.88)), 1.48, 0.28),
        box(0.05, 1.20, 0.55, at=(2.47, 0.0, agl(1.60)), mat="accent", collision=False),  # grille
        *[box(0.04, 1.20, 0.04, at=(2.50, 0.0, agl(h)), mat="dark", collision=False)
          for h in (1.42, 1.55, 1.68, 1.81)],
        box(0.36, 2.10, 0.45, at=(BUMPER_X - 0.18, 0.0, agl(1.06)), mat="dark"),          # bumper
        box(0.20, 1.30, 0.30, at=(BUMPER_X - 0.30, 0.0, agl(1.38)), mat="dark"),          # radiator guard
        *[box(0.05, 0.22, 0.13, at=(2.48, s * 0.55, agl(1.95)), mat="white", collision=False)
          for s in (1.0, -1.0)],                                                        # headlights
        # engine side grilles
        *[box(1.30, 0.02, 0.30, at=(1.45, s * 0.785, agl(1.62)), mat="accent", collision=False)
          for s in (1.0, -1.0)],
        # front fenders over the tyres (wide flat tops, as on the 6xxK scrapers)
        *[s for y in (y_w, -y_w) for s in (
            beam((-0.98, agl(1.72)), (-0.66, agl(2.16)), 0.98, 0.08, y=y),
            beam((-0.66, agl(2.16)), (0.82, agl(2.16)), 0.98, 0.08, y=y),
            beam((0.82, agl(2.16)), (1.32, agl(1.66)), 0.98, 0.08, y=y),
            box(0.05, 0.16, 0.10, at=(1.28, y, agl(1.76)), mat="orange", collision=False))],
        # cab base / transmission housing, rear tank and hitch plates
        box(1.25, 1.56, 0.86, at=(-0.375, 0.0, agl(1.66))),
        box(0.25, 1.40, 0.95, at=(tank_x0 + 0.125, 0.0, agl(1.52)), mat="dark"),         # hydraulic tank
        # upper / lower hitch bearings round the king pin
        *[s for h in (1.72, 0.86) for s in (
            cyl(0.36, 0.14, at=(hx, 0.0, agl(h)), mat="dark"),
            box(0.40, 0.72, 0.14, at=(hx + 0.20, 0.0, agl(h)), mat="dark"))],
        cyl(0.10, 1.10, at=(hx, 0.0, agl(1.29)), mat="steel", collision=False),          # king pin
        # hood details: air pre-cleaner (left), exhaust stack (right)
        cyl(0.12, 0.30, at=(0.70, 0.50, agl(2.18)), mat="accent", collision=False),
        cyl(0.075, 0.80, at=(0.55, -0.52, agl(2.40)), mat="accent", collision=False),
        # access ladder behind the left front tyre + handrail beside the cab door
        *[box(0.30, 0.10, 0.04, at=(-1.22, 0.86, agl(h)), mat="steel", collision=False)
          for h in (1.20, 1.50, 1.80)],
        box(0.04, 0.04, 0.95, at=(-1.08, 0.86, agl(2.55)), mat="accent", collision=False),
        # rear-view camera posts on the rear outer corners of the front fenders; the
        # housing sits just ahead of the lens (the camera looks back)
        *[s for sg in (1.0, -1.0) for s in (
            box(0.05, 0.05, post_z1 - FENDER_TOP, at=(cx + 0.08, sg * cy, (FENDER_TOP + post_z1) / 2),
                mat="accent", collision=False),
            box(0.12, 0.11, 0.10, at=(cx + 0.075, sg * cy, cz), mat="dark", collision=False))],
        *_cab(),
    ])


def _wheel(name: str, sg: float) -> Link:
    """33.25R29 tyre on a yellow rim, with the hub cap on the outer face."""
    lk = wheel_link(name, WHEEL_R, TYRE_W, 1100.0)
    lk.shapes += [
        cyl(0.40, 0.03, at=(0.0, sg * (TYRE_W / 2 + 0.02), 0.0), axis="y", mat="hub", collision=False),
        cyl(0.20, 0.08, at=(0.0, sg * (TYRE_W / 2 + 0.05), 0.0), axis="y", mat="dark", collision=False),
    ]
    return lk


def _hitch() -> Link:
    """Hitch casting between the tractor's hitch plates (king pin + gooseneck cross pin).

    The box runs from 0.12 m behind the king pin back to the cross pin, so the round
    king-pin boss is its front: the swept radius ahead of the pin stays 0.24 m.
    """
    x0, x1 = -0.625, -0.12
    return Link("hitch", 700.0, [
        box(x1 - x0, 0.70, 0.55, at=((x0 + x1) / 2, 0.0, 0.0), mat="dark"),
        cyl(0.24, 0.60, mat="dark"),                                                    # king-pin boss
        cyl(0.17, 0.92, at=(PD[0], 0.0, PD[1]), axis="y", mat="steel", collision=False),  # cross pin
    ])


def _gooseneck() -> Link:
    """Swan neck + yoke cross beam + draft arms (gooseneck frame)."""
    tg = TRUNNION_G
    arm_front = (CROSS[0] - 0.05, CROSS[1] - 0.08)
    return Link("gooseneck", 4200.0, [
        beam((0.0, 0.10), KNEE, 0.80, 0.62),                                            # neck
        cyl(0.33, 0.80, at=(KNEE[0], 0.0, KNEE[1]), axis="y"),                           # bend
        beam(KNEE, CROSS, 0.80, 0.52),                                                  # top beam
        box(0.46, 2 * CROSS_HW, 0.50, at=(CROSS[0], 0.0, CROSS[1])),                    # yoke cross beam
        *[beam(arm_front, tg, 0.18, 0.36, y=s * DRAFT_Y) for s in (1.0, -1.0)],          # draft arms
        *[cyl(0.17, 0.30, at=(tg[0], s * DRAFT_Y, tg[1]), axis="y", mat="steel", collision=False)
          for s in (1.0, -1.0)],                                                        # trunnion caps
        *[box(0.30, 0.24, 0.30, at=(LIFT_TOP[0], s * DRAFT_Y, LIFT_TOP[1] + 0.05), mat="dark",
              collision=False) for s in (1.0, -1.0)],                                   # lift brackets
        *[box(0.06, 0.20, 0.10, at=(CROSS[0] + 0.24, s * (CROSS_HW - 0.12), CROSS[1] + 0.10),
              mat="orange", collision=False) for s in (1.0, -1.0)],                     # marker lights
    ])


def _bowl() -> Link:
    xm = (CUT_X + WALL_X0) / 2
    wall_l = CUT_X - WALL_X0
    wall_h = BOWL_TOP - FLOOR_BOT
    zm = (BOWL_TOP + FLOOR_BOT) / 2
    tz0, tz1 = agl(1.30), agl(2.02)                    # tail (rear frame) box
    return Link("bowl", 7000.0, [
        box(CUT_X - FLOOR_X0, 2 * IN_HW, 0.12, at=((CUT_X + FLOOR_X0) / 2, 0.0, FLOOR_BOT + 0.06),
            mat="dark"),                                                                # floor
        box(0.24, 2 * IN_HW + 0.10, 0.10, at=(CUT_X - 0.12, 0.0, FLOOR_BOT + 0.03), mat="steel"),
        *[s for sg in (1.0, -1.0) for s in (
            box(wall_l, 0.06, wall_h, at=(xm, sg * WALL_Y, zm)),                        # side wall
            box(wall_l, 0.11, 0.14, at=(xm, sg * (IN_HW + 0.055), BOWL_TOP - 0.07)),    # top rail
            box(wall_l, 0.11, 0.20, at=(xm, sg * (IN_HW + 0.055), FLOOR_BOT + 0.12)),   # bottom rail
            box(0.12, 0.11, wall_h, at=(CUT_X - 0.06, sg * (IN_HW + 0.055), zm)),        # front post
            *[box(0.10, 0.05, wall_h - 0.30, at=(x, sg * (WALL_Y + 0.05), zm), collision=False)
              for x in (1.90, 2.65, 3.40)],                                             # side ribs
            box(0.36, DRAFT_Y - WALL_Y - 0.13, 0.40, at=(TRUNNION[0], sg * (WALL_Y + DRAFT_Y - 0.07) / 2,
                TRUNNION[1]), mat="dark", collision=False),                             # trunnion bracket
            box(0.26, DRAFT_Y - WALL_Y + 0.08, 0.26, at=(LIFT_LUG[0], sg * (WALL_Y + DRAFT_Y) / 2,
                LIFT_LUG[1]), mat="dark", collision=False),                             # lift lug
            cyl(0.12, 0.16, at=(APRON_PIVOT[0], sg * (WALL_Y + 0.10), APRON_PIVOT[1]), axis="y",
                mat="steel", collision=False),                                          # apron pivot boss
        )],
        box(0.13, 2 * IN_HW + 0.12, wall_h, at=(WALL_X0 + 0.065, 0.0, zm)),            # rear wall
        # rear frame: tail over the axle to the push block
        box(WALL_X0 - TAIL_X1, 1.52, tz1 - tz0, at=((WALL_X0 + TAIL_X1) / 2, 0.0, (tz0 + tz1) / 2)),
        box(1.20, 1.30, 0.36, at=(0.0, 0.0, agl(1.20)), mat="dark"),                    # axle mount
        cyl(0.25, 1.70, axis="y", mat="dark"),                                           # rear axle
        sphere(0.34, mat="dark"),                                                       # differential
        box(TAIL_X1 - PUSH_X1, 1.90, 0.62, at=((TAIL_X1 + PUSH_X1) / 2, 0.0, agl(1.30)),
            mat="dark"),                                                                # push block
        *[box(0.04, 0.16, 0.10, at=(PUSH_X1 - 0.01, s * 0.70, agl(1.75)), mat="red", collision=False)
          for s in (1.0, -1.0)],                                                        # tail lights
        cyl(0.17, 1.55, at=(WALL_X0 - 0.75, 0.0, EJ_Z), axis="x", mat="dark",
            collision=False),                                                           # ejector barrel
    ])


def _apron() -> Link:
    """Apron (front gate) in the apron frame: origin on the arm pivots."""
    px, pz = APRON_PIVOT
    lip = (CUT_X + 0.10 - px, agl(0.52) - pz)
    mid = (CUT_X + 0.46 - px, agl(1.55) - pz)
    top = (CUT_X + 0.28 - px, agl(2.86) - pz)
    arm_end = (CUT_X + 0.24 - px, agl(2.66) - pz)
    return Link("apron", 1500.0, [
        beam(lip, mid, 2 * IN_HW + 0.14, 0.08),                                         # lower plate
        beam(mid, top, 2 * IN_HW + 0.14, 0.08),                                         # upper plate
        box(0.10, 2 * IN_HW + 0.18, 0.12, at=(top[0] - 0.02, 0.0, top[1] - 0.04)),      # top stiffener
        box(0.08, 2 * IN_HW + 0.14, 0.10, at=(mid[0] + 0.02, 0.0, mid[1]), mat="dark",
            collision=False),                                                           # mid stiffener
        *[s for sg in (1.0, -1.0) for s in (
            # side plates: bands behind the curved front plate, outside the bowl walls
            beam((lip[0] - 0.17, lip[1] + 0.04), (mid[0] - 0.17, mid[1]), 0.05, 0.32,
                 y=sg * (WALL_Y + 0.06)),
            beam((mid[0] - 0.17, mid[1]), (top[0] - 0.17, top[1] - 0.04), 0.05, 0.32,
                 y=sg * (WALL_Y + 0.06)),
            beam((0.0, 0.0), arm_end, 0.08, 0.26, y=sg * APRON_ARM_Y),                  # arm
            cyl(0.10, 0.12, at=(0.0, sg * APRON_ARM_Y, 0.0), axis="y", mat="steel",
                collision=False),                                                       # pivot cap
        )],
    ])


def _ejector() -> Link:
    """Ejector (moving rear wall) in its frame: origin on the plate's rear face, at EJ_Z."""
    z0 = FLOOR_TOP + 0.03 - EJ_Z
    z1 = BOWL_TOP - 0.06 - EJ_Z
    rod_l = 1.35
    return Link("ejector", 1400.0, [
        box(0.12, 2 * IN_HW - 0.04, z1 - z0, at=(0.06, 0.0, (z0 + z1) / 2)),             # plate
        *[box(0.10, 0.10, z1 - z0 - 0.10, at=(-0.05, y, (z0 + z1) / 2), mat="dark",
              collision=False) for y in (-1.0, 0.0, 1.0)],                              # back ribs
        box(0.10, 2 * IN_HW - 0.10, 0.10, at=(-0.05, 0.0, z1 - 0.08), mat="dark",
            collision=False),                                                           # top rib
        cyl(0.10, rod_l, at=(-rod_l / 2, 0.0, 0.0), axis="x", mat="steel", collision=False),  # rod
    ])


def _ejector_stage() -> Link:
    """Middle stage of the telescopic ejector cylinder (frame: rear-wall face, at EJ_Z)."""
    st_l = 1.30
    return Link("ejector_stage", 150.0, [
        cyl(0.135, st_l, at=(-st_l / 2 + 0.05, 0.0, 0.0), axis="x", mat="steel", collision=False),
        cyl(0.15, 0.06, at=(0.03, 0.0, 0.0), axis="x", mat="dark", collision=False),  # gland
    ])


# ------------------------------------------------------------------------------------
# Machine
# ------------------------------------------------------------------------------------

def build() -> MachineSpec:
    fit = _linkage_fit()

    tractor = _tractor()
    hitch = _hitch()
    goose = _gooseneck()
    bowl = _bowl()
    apron = _apron()
    ejector = _ejector()
    stage = _ejector_stage()

    bowl_j = Joint("bowl_joint", "revolute", "hitch", "bowl", xyz=(RH[0], 0.0, RH[1]),
                   axis=(0, -1, 0), lower=BOWL_LO, upper=0.0, init=0.0, velocity=0.09,
                   role="tool")
    apron_j = Joint("apron_joint", "revolute", "bowl", "apron",
                    xyz=(APRON_PIVOT[0], 0.0, APRON_PIVOT[1]), axis=(0, -1, 0),
                    lower=0.0, upper=APRON_MAX, init=0.0, velocity=0.30, role="tool")
    ej_j = Joint("ejector_joint", "prismatic", "bowl", "ejector", xyz=(FLOOR_X0, 0.0, EJ_Z),
                 axis=(1, 0, 0), lower=0.0, upper=EJECT_STROKE, init=0.0, velocity=0.35,
                 role="tool")
    stage_j = Joint("ejector_stage_joint", "prismatic", "bowl", "ejector_stage",
                    xyz=(WALL_X0, 0.0, EJ_Z), axis=(1, 0, 0), lower=-0.02,
                    upper=EJECT_STROKE / 2 + 0.02, init=0.0, velocity=ej_j.velocity / 2 + 0.05,
                    role="follower", follows=(ej_j.name, 0.5, 0.0))

    kd, cd = fit["kd"], fit["cd"]
    a_lo, a_hi = sorted((cd, kd * BOWL_LO + cd))
    goose_j = Joint("gooseneck_pitch_joint", "revolute", "hitch", "gooseneck",
                    xyz=(PD[0], 0.0, PD[1]), axis=(0, -1, 0), lower=a_lo - 0.02, upper=a_hi + 0.02,
                    init=cd, velocity=abs(kd) * bowl_j.velocity + 0.05, role="follower",
                    follows=(bowl_j.name, kd, cd))

    # --- lift cylinders (barrel pinned on the draft-arm front, rod eye on the bowl lug) -
    d0x, d0z = fit["dir0"]
    pitch0 = math.atan2(-d0z, d0x)          # rpy pitch that points barrel +x at the lug
    eye0 = fit["cs"]                        # pin-to-pin distance at the bowl init pose
    barrel_l = 1.45
    rod_l = eye0 - 0.10
    kp, cp, ks = fit["kp"], fit["cp"], fit["ks"]
    p_lo, p_hi = sorted((cp, kp * BOWL_LO + cp))
    s_lo, s_hi = sorted((0.0, ks * BOWL_LO))
    lift_links, lift_joints = [], []
    for side, sg in (("left", 1.0), ("right", -1.0)):
        y = sg * DRAFT_Y
        lift_links += [
            Link(f"{side}_lift_barrel", 160.0, [
                cyl(0.11, barrel_l, at=(barrel_l / 2 - 0.05, 0.0, 0.0), axis="x", mat="dark",
                    collision=False),
                cyl(0.13, 0.10, at=(barrel_l - 0.10, 0.0, 0.0), axis="x", mat="steel",
                    collision=False),
                cyl(0.10, 0.26, axis="y", mat="steel", collision=False),               # top eye
            ]),
            Link(f"{side}_lift_rod", 70.0, [
                cyl(0.065, rod_l, at=(eye0 - rod_l / 2, 0.0, 0.0), axis="x", mat="steel",
                    collision=False),
                cyl(0.09, 0.22, at=(eye0, 0.0, 0.0), axis="y", mat="steel", collision=False),
            ]),
        ]
        lift_joints += [
            Joint(f"{side}_lift_barrel_joint", "revolute", "gooseneck", f"{side}_lift_barrel",
                  xyz=(LIFT_TOP[0], y, LIFT_TOP[1]), rpy=(0.0, pitch0, 0.0), axis=(0, -1, 0),
                  lower=p_lo - 0.05, upper=p_hi + 0.05, init=cp,
                  velocity=abs(kp) * bowl_j.velocity + 0.05, role="follower",
                  follows=(bowl_j.name, kp, cp)),
            Joint(f"{side}_lift_rod_joint", "prismatic", f"{side}_lift_barrel", f"{side}_lift_rod",
                  axis=(1, 0, 0), lower=s_lo - 0.05, upper=s_hi + 0.05, init=0.0,
                  velocity=abs(ks) * bowl_j.velocity + 0.05, role="follower",
                  follows=(bowl_j.name, ks, 0.0)),
        ]

    # --- wheels --------------------------------------------------------------------------
    wheels, wheel_joints = [], []
    for axle, parent, x, gauge in (("front", "tractor", 0.0, FRONT_GAUGE),
                                   ("rear", "bowl", 0.0, REAR_GAUGE)):
        for side, sg in (("left", 1.0), ("right", -1.0)):
            name = f"{axle}_{side}_wheel"
            wheels.append(_wheel(name, sg))
            wheel_joints.append(wheel_joint(f"{name}_joint", parent, name,
                                            (x, sg * gauge / 2, 0.0), WHEEL_R))

    links = [tractor, hitch, goose, bowl, apron, ejector, stage, *lift_links, *wheels]
    joints = [
        Joint("steering_hitch_joint", "revolute", "tractor", "hitch",
              xyz=(HITCH[0], 0.0, HITCH[1]), axis=(0, 0, 1), lower=-0.90, upper=0.90,
              velocity=0.40, role="articulation"),
        goose_j, bowl_j, apron_j, ej_j, stage_j,
        *lift_joints,
        *wheel_joints,
    ]

    cam_z = agl(3.15)
    return MachineSpec(
        name="wheel_tractor_scraper",
        abbrev="WTS",
        display_name="Wheel Tractor-Scraper",
        category="grading",
        reference="15 m³ single-engine wheel tractor-scraper (CAT 621K class)",
        description=("Two-axle open-bowl scraper: single-axle tractor with engine hood and "
                     "centre cab, steering hitch (90° each way on the real machine, ±0.9 rad "
                     "here), gooseneck and draft arms carrying a "
                     "15 m³ bowl on the rear axle; the bowl lowers to cut 0.32 m deep, the "
                     "apron (front gate) raises to open and the ejector (rear wall) pushes the "
                     "load out forward."),
        root="tractor",
        links=links,
        joints=joints,
        drive=Drive(type="articulated", max_speed=15.3, max_reverse_speed=2.8,
                    articulation_joint="steering_hitch_joint", front_length=FRONT_LEN,
                    rear_length=REAR_LEN, articulation_rate=0.40),
        tools=[
            Tool("bowl_joint", "bowl", positive="raise", negative="lower"),
            Tool("apron_joint", "apron", positive="raise", negative="lower"),
            Tool("ejector_joint", "ejector", positive="eject", negative="return"),
        ],
        cameras=[
            # operator eye height on the centre line, just ahead of the windscreen
            # (the glass panes are opaque boxes), looking down the hood
            Camera("front", "tractor", (CAB_X1 + 0.07, 0.0, cam_z), (0.0, 0.25, 0.0)),
            # rear-view cameras on the front-fender posts, looking back (17° outward) at
            # the apron, cutting edge, bowl mouth and draft arms
            Camera("left", "tractor", REAR_CAM, (0.0, 0.18, 2.85)),
            Camera("right", "tractor", (REAR_CAM[0], -REAR_CAM[1], REAR_CAM[2]), (0.0, 0.18, -2.85)),
        ],
        envelope=(14.0, 3.75, 3.85),
        operating_mass=33200.0,
        notes=("Bowl turns about the rear axle; the gooseneck pitches at the hitch as a "
               "follower so the draft arms stay on the bowl trunnions, and the lift / ejector "
               "cylinders are cosmetic followers.  apron raise = open.  Ejector +1 = eject "
               "(forward, towards the tractor), not auto-verifiable because 'out' is measured "
               "from the root origin, which lies ahead of the ejector.  left/right cameras "
               "are rear-view cameras on the front fenders. Between them they see the bowl "
               "front and apron in every pose, but the ejector only through the open apron. "
               "At full lock the rear unit leaves the camera on the outside of the turn."),
    )
