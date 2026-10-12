"""Asphalt paver (PAV), 19 t tracked class — modelled on a Vögele Super 1800-3 / CAT AP655F.

Reference numbers (typical datasheet values for the class):
    operating mass ≈ 19 t with a 2.55–5.0 m extending screed (CAT AP655 with
    SE60 screed 19 119 kg, paver alone 15 292 kg, i.e. screed ≈ 3.8 t;
    Vögele Super 1800-3i with AB 500-3 TV: 19.65 t operating weight in dealer
    listings, 19.4–19.95 t gross depending on year; ≈ 20.1 t with an AB 600) ·
    basic paving width 2.55 m, hydraulically variable 2.55–5.0 m (AB 500),
    i.e. 1.225 m stroke per side · 13 t material hopper, 3.265 m wide with
    the wings open, infeed height ≈ 0.6 m · crawler ground contact 3.06 m,
    360 mm rubber tracks (CAT Mobil-trac / Vögele 305 mm pads), gauge 1.90 m ·
    transport length ≈ 6.4–6.5 m (CAT 6.4 m, Vögele 6.5 m) ·
    height to the top of the roof (roof up) ≈ 3.7–3.9 m (CAT 3.7 m; 2.9–3.1 m
    with the roof folded for low-loader transport — not modelled) ·
    paving speed up to 24 m/min, travel up to 4.5 km/h (1.25 m/s) ·
    counter-rotating tracks, ≈ 0.3 rad/s pivot turn.

The model's hopper is smaller than the real 13 t one: 1.64 m of usable length
between the bulkhead and the front walls, 0.55 m wing walls, ≈ 3.2 m³
level-full (≈ 7 t).  The walls cannot get much deeper: folded, the wall top
reaches y = 0.62 − h·sin 66°, i.e. the centreline at h ≈ 0.68 m.  The hopper
cannot get longer either: the open wings would run into the tow-point
brackets and levelling cylinders behind the bulkhead.

Model joints:
    left_wing_joint    0 … 1.15 rad, 0.25 rad/s (≈ 4.6 s to close).  Hinge along x
                       at y = ±0.62 m, 0.72 m above the ground; 0 = wing floor
                       flat (open, receiving mix: hopper 3.27 m wide), 1.15 rad
                       (66°) = folded up, tipping the mix onto the conveyors.
                       right_wing_joint follows it (mirror image).
    screed_lift_joint  0 … 0.28 m, 0.08 m/s (≈ 3.5 s).  Stroke of the two screed
                       lift cylinders at the rear corners of the tractor
                       (+ = cylinders retract = screed rises).  The cylinder is
                       the operator-controlled DOF; everything else in the
                       linkage follows it linearly (rod eye stays within
                       0.2 mm of the tow-arm lug):
      tow_arm_joint    0 … 0.09 rad.  The tow arms pivot at the tow points on the
                       tractor sides (0.78 m high, 0.9 m ahead of the track
                       centre, ≈ 3.5 m ahead of the screed).  0 = screed
                       floating on the ground (paving); 0.09 = raised for
                       travel: screed plate 0.32 m (front) to 0.37 m (rear)
                       above the ground.
      lift_cylinder_joint  barrel angle.
                       The tool is the stroke (m), not the tow-arm angle: the
                       task controller's fixed revolute tolerance of 1° would
                       be 19 % of the 0.09 rad range (goals stopping far short),
                       its prismatic tolerance of 0.01 m is 3.6 % of the stroke.
    left_extension_joint  0 … 1.225 m, 0.15 m/s.  Rear-mounted extending units
                       slide out sideways; right_extension_joint mirrors it.
                       Paving width 2.55 m retracted, 5.0 m fully extended.
                       Each unit carries a guide tube that stays 0.15 m inside
                       the basic screed at full stroke.

Cameras: ``front`` = operator eye at the front of the platform, looking into
the hopper (both wings).  ``left`` = screed camera under the rear edge of the
roof (left of centre), looking straight back and down over the auger at the
whole screed and both extending units / end gates.  ``right`` = side camera
under the right rear corner of the platform, looking back along the tractor
side at the right screed-lift cylinder, the tow-arm end, the screed end and
the right extending unit — from the side, so the screed lift (a vertical
motion the roof camera sees almost end-on) shows clearly.

Layout as on the real machine: push rollers and hopper at the front, the
slat conveyors running back through the tractor, engine compartment behind
the hopper (hood at platform level, so the operator sees into the hopper),
raised operator platform with two seats, a central console and a roof on four
posts with a windscreen and side glass, the spreading auger across the rear,
and the floating screed towed behind on two long tow arms running along the
outside of the tracks (levelling cylinder above each tow point).

Init pose = site travel pose: hopper wings folded up (closed, within the
2.55 m basic width), screed raised, extensions retracted.  Model envelope in
that pose 6.45 × 2.55 × 3.75 m; model mass 19.1 t (tractor 13.44 t + 0.12 t
sprockets, wings 2 × 0.42 t, tow arms + basic screed 3.6 t, extending units
2 × 0.5 t, lift cylinders 0.1 t — screed and extending units ≈ 3.8 t without
the ≈ 0.8 t of tow arms, as in the CAT figures).

Root frame: centre of the undercarriage at track mid-height (0.30 m above the
ground).  Geometry below is written in ground coordinates (x, z above ground).
"""

from __future__ import annotations

import math

from ..spec import (Camera, Drive, Joint, Link, MachineSpec, Tool, beam, box, cyl,
                    sprocket_link)

# Undercarriage
TRACK_L, TRACK_H, SHOE_W, GAUGE = 3.66, 0.60, 0.36, 1.90   # 3.06 m ground contact
Z0 = TRACK_H / 2                      # root origin height above the ground
SPROCKET_R = 0.27
# Tractor (ground coordinates)
X_BUMPER = 2.78                       # front bumper face (push rollers ahead of it)
X_HOP_F, X_HOP_R = 2.70, 1.00         # hopper front wall / bulkhead
X_HOOD_R = -0.10                      # rear edge of the engine hood = front of the platform
X_REAR = -1.95                        # tractor rear wall (auger behind it)
X_PLAT_R = -2.12                      # rear edge of the operator platform
BODY_Y, PLAT_Y = 1.12, 1.25
Z_BODY, Z_PLAT, Z_HOOD = 1.62, 1.72, 1.74    # hood ≈ platform level: view into the hopper
ROOF_Z = (3.53, 3.63)
# Hopper wings
HINGE_Y, HINGE_Z = 0.62, 0.72
WING_W, WING_WALL = 1.00, 0.55
WING_LO, WING_HI = 0.0, 1.15
# Tow arms / screed (ground coordinates in the paving pose, q = 0)
TOW = (0.90, 0.78)                    # tow point (tow-arm pivot)
ARM_Y = 1.20
ARM_END = (-2.55, 0.58)
SCR_F, SCR_R = -2.42, -3.05           # basic screed body
EXT_F, EXT_R = -3.08, -3.40           # extending units (behind the basic screed)
WALK = (-3.22, -3.52)                 # footboard
SCREED_Z0 = 0.01                      # screed plate bottom when paving
EXT_STROKE = 1.225
EXT_TUBE_IN = 0.15                    # guide tube still inside the basic screed at full stroke
LIFT_LO, LIFT_HI = 0.0, 0.09          # tow-arm angle (rad): paving … raised for travel
LIFT_TIME = 3.5                       # s for the full screed lift
# Screed lift cylinders: barrel pin on the tractor, rod eye on the tow arm (paving pose)
LIFT_BASE = (-2.14, 1.50)
LIFT_EYE = (-2.22, 0.72)


def _bx(x0, x1, y0, y1, z0, z1, mat="body", collision=True, origin=(0.0, Z0)):
    """Box from extents; x/z relative to ``origin`` (x, z) — ground coords by default."""
    ox, oz = origin
    return box(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0),
               at=((x0 + x1) / 2 - ox, (y0 + y1) / 2, (z0 + z1) / 2 - oz),
               mat=mat, collision=collision)


def _track(y: float) -> list:
    """Rubber track loop + track frame, bogie rollers and front idler, centred at (0, y, 0)."""
    r = TRACK_H / 2
    straight = TRACK_L - TRACK_H
    belt = 0.08
    return [
        box(straight, SHOE_W, belt, at=(0.0, y, -r + belt / 2), mat="rubber"),        # bottom run
        box(straight, SHOE_W, belt, at=(0.0, y, r - belt / 2), mat="rubber"),         # top run
        cyl(r, SHOE_W, at=(straight / 2, y, 0.0), axis="y", mat="rubber"),           # front bend
        cyl(r, SHOE_W, at=(-straight / 2, y, 0.0), axis="y", mat="rubber"),          # rear bend
        box(straight - 0.20, SHOE_W * 0.7, 0.30, at=(0.0, y, 0.02), mat="dark"),      # track frame
        cyl(0.25, SHOE_W + 0.02, at=(straight / 2, y, 0.0), axis="y", mat="steel",
            collision=False),                                                        # front idler
        cyl(0.11, SHOE_W + 0.03, at=(straight / 2, y, 0.0), axis="y", mat="dark",
            collision=False),                                                        # idler hub
        *[cyl(0.10, SHOE_W * 0.9, at=(x, y, -r + belt + 0.10), axis="y", mat="steel",
              collision=False) for x in (-1.05, -0.63, -0.21, 0.21, 0.63, 1.05)],    # bogie rollers
    ]


def _wing(sy: float) -> Link:
    """Hopper wing; frame on the hinge line (rear end), +x forward, outward = sy * y."""
    o = (X_HOP_R, HINGE_Z)
    L = X_HOP_F - X_HOP_R
    w, t, h = WING_W, 0.03, WING_WALL

    def yy(a, b):
        return sy * a, sy * b

    return Link(f"{'left' if sy > 0 else 'right'}_wing", 420.0, [
        _bx(X_HOP_R + t, X_HOP_F - t, *yy(0.0, w), HINGE_Z - t, HINGE_Z, origin=o),       # floor
        _bx(X_HOP_R + t, X_HOP_F - t, *yy(w - 0.015, w + 0.015), HINGE_Z - t, HINGE_Z + h,
            origin=o),                                                                     # outer wall
        _bx(X_HOP_F - t, X_HOP_F, *yy(0.0, w + 0.015), HINGE_Z - t, HINGE_Z + h, origin=o),  # front wall
        _bx(X_HOP_R, X_HOP_R + t, *yy(0.0, w + 0.015), HINGE_Z - t, HINGE_Z + h, origin=o),  # rear wall
        _bx(X_HOP_R + t, X_HOP_F - t, *yy(w - 0.03, w + 0.03), HINGE_Z + h - 0.05,
            HINGE_Z + h + 0.01, mat="dark", collision=False, origin=o),                   # top rim
        _bx(X_HOP_F - 0.05, X_HOP_F + 0.01, *yy(0.05, w + 0.03), HINGE_Z + h - 0.05,
            HINGE_Z + h + 0.01, mat="dark", collision=False, origin=o),                   # front rim
        *[_bx(x - 0.04, x + 0.04, *yy(0.05, w - 0.05), HINGE_Z - 0.10, HINGE_Z - t,
              mat="dark", collision=False, origin=o)
          for x in (X_HOP_R + 0.45, X_HOP_R + 0.45 + L / 3, X_HOP_R + 0.45 + 2 * L / 3)],  # ribs
        cyl(0.05, L - 0.1, at=(L / 2, 0.0, 0.0), axis="x", mat="steel", collision=False),  # hinge
    ])


def _lift_linkage(base, pivot, eye0, q_hi, n=61):
    """Screed-lift cylinder linkage parametrised by the cylinder stroke s.

    ``base`` (barrel pin) is fixed on the tractor; ``eye0`` (rod eye at q = 0)
    rotates about ``pivot`` by the tow-arm angle q about +y (axis (0, 1, 0):
    +q lifts points behind the pivot).  s = 0 at q = 0 (longest cylinder) and
    grows as the cylinder retracts.  Returns a dict with the cylinder length at
    s = 0 (``L0``), the full ``stroke`` (q = q_hi), the tow-arm line q = kq·s,
    the barrel elevation line kb·s + cb ((0, -1, 0) convention: +angle raises
    +x) and ``err`` = worst distance (m) between the modelled rod eye and the
    tow-arm lug over the stroke.
    """
    rx, rz = eye0[0] - pivot[0], eye0[1] - pivot[1]

    def eye(q):
        return (pivot[0] + rx * math.cos(q) + rz * math.sin(q),
                pivot[1] - rx * math.sin(q) + rz * math.cos(q))

    def length(q):
        e = eye(q)
        return math.hypot(e[0] - base[0], e[1] - base[1])

    def q_of_s(s):                      # length(q) decreases monotonically in q here
        lo, hi = 0.0, q_hi
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if length(mid) > L0 - s else (lo, mid)
        return (lo + hi) / 2

    L0 = length(0.0)
    stroke = L0 - length(q_hi)
    kq = q_hi / stroke                  # through both ends: q(0) = 0, q(stroke) = q_hi exactly
    ss = [stroke * i / (n - 1) for i in range(n)]
    ang = []
    for s in ss:
        e = eye(q_of_s(s))
        ang.append(math.atan2(e[1] - base[1], e[0] - base[0]))
    ms, ma = sum(ss) / n, sum(ang) / n
    kb = sum((s - ms) * (a - ma) for s, a in zip(ss, ang)) / sum((s - ms) ** 2 for s in ss)
    cb = ma - kb * ms
    err = 0.0
    for s in ss:
        th = kb * s + cb
        rod = (base[0] + (L0 - s) * math.cos(th), base[1] + (L0 - s) * math.sin(th))
        lug = eye(kq * s)
        err = max(err, math.hypot(rod[0] - lug[0], rod[1] - lug[1]))
    return {"L0": L0, "stroke": stroke, "kq": kq, "kb": kb, "cb": cb, "err": err}


def build() -> MachineSpec:
    y_t = GAUGE / 2
    zr0, zr1 = ROOF_Z

    # ------------------------------------------------------------------ tractor
    tr = [
        *_track(+y_t),
        *_track(-y_t),
        _bx(X_REAR, 2.20, -0.74, 0.74, 0.20, 0.62, mat="dark"),               # main frame between tracks
        _bx(2.20, X_BUMPER, -0.74, 0.74, 0.34, 0.56, mat="dark"),             # nose under the hopper
        _bx(X_HOP_F, X_BUMPER, -1.00, 1.00, 0.36, 0.60, mat="dark"),          # push-roller beam
        _bx(X_REAR, X_HOP_R, -BODY_Y, BODY_Y, 0.62, Z_BODY),                  # body over the tracks
        _bx(X_HOOD_R, X_HOP_R, -1.10, 1.10, Z_BODY, Z_HOOD),                  # engine hood
        _bx(X_HOP_R - 0.06, X_HOP_R, -1.10, 1.10, Z_HOOD - 0.02, Z_HOOD + 0.05,
            mat="accent", collision=False),                                   # bulkhead top edge
        # hopper centre: conveyor floor, front apron (just ahead of the wings' front
        # walls, which swing across it when they fold), hinge rails
        _bx(X_HOP_R, X_HOP_F, -HINGE_Y + 0.02, HINGE_Y - 0.02, 0.52, 0.58, mat="dark"),
        _bx(X_HOP_F, X_HOP_F + 0.03, -HINGE_Y + 0.02, HINGE_Y - 0.02, 0.52, 0.92),
        *[_bx(x - 0.02, x + 0.02, -0.56, 0.56, 0.58, 0.60, mat="steel", collision=False)
          for x in [X_HOP_R + 0.10 + 0.20 * i for i in range(8)]],          # conveyor slats
        _bx(X_HOP_R, X_HOP_F, -0.02, 0.02, 0.58, 0.66, mat="dark", collision=False),  # conveyor divider
        # rear: tractor rear wall, auger gearbox
        _bx(X_REAR - 0.06, X_REAR, -1.00, 1.00, 0.55, Z_BODY),
        _bx(-2.32, X_REAR - 0.06, -1.15, 1.15, 0.53, 0.57, mat="dark"),       # auger tunnel cover
        _bx(-2.25, -2.05, -0.12, 0.12, 0.22, 0.62, mat="dark"),               # auger centre drive
        _bx(-2.08, X_REAR - 0.06, -0.12, 0.12, 0.45, 0.62, mat="dark"),       # auger drive bracket
        cyl(0.07, 2.30, at=(-2.15, 0.0, 0.33 - Z0), axis="y", mat="steel"),   # auger shaft
        # operator platform
        _bx(X_PLAT_R, X_HOOD_R, -PLAT_Y, PLAT_Y, Z_BODY, Z_PLAT, mat="dark"),  # platform floor
        _bx(-0.60, -0.25, -0.45, 0.45, Z_PLAT, 2.30, mat="dark"),             # central console
        _bx(-0.60, -0.28, -0.40, 0.40, 2.30, 2.34, mat="glass", collision=False),  # console display
        # roof, posts, windscreen, beacon, work lights
        _bx(-2.22, 0.30, -1.27, 1.27, zr0, zr1),                              # roof
        _bx(-2.20, 0.28, -1.25, 1.25, zr0 - 0.06, zr0, mat="accent", collision=False),
        _bx(-0.18, -0.15, -1.12, 1.12, 2.70, zr0 - 0.06, mat="glass", collision=False),  # windscreen
        beam((-0.15, 2.85 - Z0), (-0.15, 3.30 - Z0), 0.03, 0.02, y=0.30, mat="dark",
             collision=False),                                                # wiper
        cyl(0.08, 0.12, at=(-1.95, 0.95, zr1 + 0.06 - Z0), mat="orange", collision=False),  # beacon
        cyl(0.05, 0.55, at=(0.12, -1.02, Z_HOOD + 0.27 - Z0), mat="accent", collision=False),  # exhaust
        # front push rollers
        *[cyl(0.12, 0.40, at=(X_BUMPER + 0.02, s * 0.72, 0.49 - Z0), axis="y", mat="steel")
          for s in (1.0, -1.0)],
    ]
    # auger flights (helix suggested by alternately tilted discs)
    for i in range(12):
        y = -1.10 + i * 0.20
        tr.append(cyl(0.19, 0.035, at=(-2.15, y, 0.33 - Z0), axis="y",
                      rpy=(math.pi / 2, 0.0, 0.25 if i % 2 else -0.25), mat="dark", collision=False))
    for sy in (1.0, -1.0):
        tr += [
            # seats with arm consoles
            _bx(-1.40, -0.90, sy * 0.45, sy * 0.95, Z_PLAT, 2.12, mat="dark"),
            _bx(-1.40, -0.90, sy * 0.45, sy * 0.95, 2.12, 2.22, mat="accent"),
            _bx(-1.50, -1.38, sy * 0.47, sy * 0.93, 2.22, 2.80, mat="accent"),
            _bx(-1.30, -0.85, sy * 0.95, sy * 1.05, 2.12, 2.32, mat="dark", collision=False),
            # side glass of the roof and railings
            _bx(-0.95, -0.18, sy * 1.21, sy * 1.23, 2.70, zr0 - 0.06, mat="glass", collision=False),
            _bx(X_PLAT_R + 0.05, X_HOOD_R, sy * 1.20, sy * 1.24, 2.48, 2.53, mat="accent",
                collision=False),
            # roof posts
            cyl(0.045, zr0 - Z_PLAT, at=(-0.17, sy * 1.18, (zr0 + Z_PLAT) / 2 - Z0), mat="accent"),
            cyl(0.045, zr0 - Z_PLAT, at=(-2.05, sy * 1.18, (zr0 + Z_PLAT) / 2 - Z0), mat="accent"),
            _bx(0.22, 0.30, sy * 0.85, sy * 1.05, zr0 - 0.10, zr0 - 0.02, mat="white",
                collision=False),                                             # work light
            # engine air-intake grille on the body sides
            _bx(-0.05, 0.75, sy * BODY_Y, sy * (BODY_Y + 0.015), 1.05, 1.55, mat="accent",
                collision=False),
            *[_bx(-0.03, 0.73, sy * (BODY_Y + 0.015), sy * (BODY_Y + 0.025), z, z + 0.03,
                  mat="dark", collision=False) for z in (1.10, 1.20, 1.30, 1.40, 1.50)],
            # hopper hinge rails
            _bx(X_HOP_R, X_HOP_F, sy * (HINGE_Y - 0.04), sy * (HINGE_Y + 0.02), 0.58, 0.70,
                mat="dark"),
            # tow point bracket with the levelling cylinder above it
            _bx(0.78, 1.00, sy * BODY_Y, sy * 1.25, 0.88, 1.00, mat="dark", collision=False),
            cyl(0.07, 0.62, at=(TOW[0], sy * ARM_Y, 1.30 - Z0), mat="dark", collision=False),
            cyl(0.035, 0.20, at=(TOW[0], sy * ARM_Y, 0.92 - Z0), mat="steel", collision=False),
            # lift-cylinder bracket under the platform corner
            _bx(LIFT_BASE[0] - 0.07, X_REAR, sy * (ARM_Y - 0.06), sy * (ARM_Y + 0.06),
                LIFT_BASE[1] - 0.06, Z_BODY, mat="dark", collision=False),
            # tail light
            _bx(X_REAR - 0.08, X_REAR - 0.06, sy * 0.80, sy * 0.98, 1.40, 1.52, mat="red",
                collision=False),
        ]
    tractor = Link("tractor", 13440.0, tr)

    # ------------------------------------------------------------------ screed + tow arms
    def a(p):
        return (p[0] - TOW[0], p[1] - TOW[1])

    so = TOW
    z0 = SCREED_Z0
    scr = [
        _bx(SCR_R, SCR_F, -1.24, 1.24, z0 + 0.05, 0.52, origin=so),                     # screed body
        _bx(SCR_R, SCR_F - 0.08, -1.24, 1.24, z0, z0 + 0.05, mat="steel", origin=so),   # screed plate
        _bx(SCR_F - 0.08, SCR_F, -1.24, 1.24, z0 + 0.01, 0.24, mat="dark", origin=so),  # tamper
        _bx(SCR_R + 0.05, SCR_F - 0.05, -1.20, 1.20, 0.52, 0.56, mat="accent", collision=False,
            origin=so),                                                               # top cover
        # footboard above the extending units, on three brackets
        _bx(WALK[1], WALK[0], -1.22, 1.22, 0.50, 0.54, mat="dark", origin=so),
        *[_bx(SCR_R - 0.20, SCR_R, y - 0.04, y + 0.04, 0.46, 0.54, mat="dark", origin=so)
          for y in (-1.10, 0.0, 1.10)],
    ]
    for sy in (1.0, -1.0):
        y = sy * ARM_Y
        scr += [
            beam(a(TOW), a(ARM_END), 0.09, 0.20, y=y),                                  # tow arm
            _bx(-2.95, -2.40, sy * 1.155, sy * 1.245, 0.20, 0.66, origin=so),           # side bracket
            cyl(0.10, 0.12, at=(0.0, y, 0.0), axis="y", mat="steel", collision=False),  # tow-point pin
            _bx(LIFT_EYE[0] - 0.07, LIFT_EYE[0] + 0.07, sy * (ARM_Y - 0.05), sy * (ARM_Y + 0.05),
                LIFT_EYE[1] - 0.10, LIFT_EYE[1] + 0.03, mat="dark", collision=False,
                origin=so),                                                           # lift lug
            # screed operator consoles at the footboard ends
            _bx(-3.48, -3.30, sy * 0.92, sy * 1.18, 0.54, 1.05, mat="dark", collision=False,
                origin=so),
            _bx(-3.46, -3.32, sy * 0.95, sy * 1.15, 1.05, 1.08, mat="glass", collision=False,
                origin=so),
        ]
    screed = Link("screed", 3600.0, scr)

    # ------------------------------------------------------------------ extending units
    ext_x = (EXT_F + EXT_R) / 2
    eo = (ext_x, 0.0)
    ext_in, ext_out = 0.03, 1.235              # unit body |y| when retracted
    # Guide tube in the gap between the basic screed (rear face SCR_R) and the unit
    # (front face EXT_F), half sunk into each, reaching EXT_TUBE_IN inside the basic
    # screed's end at full stroke.  Left/right at different heights: when retracted
    # both tubes cross the centreline.
    tube_x, tube_r = (SCR_R + EXT_F) / 2, 0.04
    tube_y0 = 1.24 - EXT_TUBE_IN - EXT_STROKE  # inner end, unit frame (|y|)
    tube_y1 = ext_out - 0.04

    def extension(sy):
        name = "left_extension" if sy > 0 else "right_extension"
        tube_z = 0.22 if sy > 0 else 0.34
        return Link(name, 500.0, [
            _bx(EXT_R, EXT_F, sy * ext_in, sy * ext_out, z0 + 0.05, 0.45, origin=eo),   # body
            _bx(EXT_R, EXT_F, sy * ext_in, sy * ext_out, z0, z0 + 0.05, mat="steel", origin=eo),
            _bx(EXT_R - 0.05, SCR_R + 0.15, sy * 1.245, sy * 1.275, z0, 0.62, mat="dark",
                origin=eo),                                                           # end gate
            cyl(tube_r, tube_y1 - tube_y0, at=(tube_x - ext_x, sy * (tube_y0 + tube_y1) / 2, tube_z),
                axis="y", mat="steel", collision=False),                              # guide tube
        ])

    # ------------------------------------------------------------------ lift cylinders
    # The operator drives the cylinder stroke s (screed_lift_joint, prismatic): the
    # barrel angle and the tow-arm angle follow it linearly (_lift_linkage).
    lk = _lift_linkage(LIFT_BASE, TOW, LIFT_EYE, LIFT_HI)
    if lk["err"] > 0.005:
        raise ValueError(f"lift cylinder rod eye misses the tow-arm lug by {lk['err']:.3f} m")
    L0, stroke = lk["L0"], lk["stroke"]
    barrel_len = (L0 - stroke) - 0.06
    rod_len = L0 + 0.08 - barrel_len
    barrels = Link("lift_cylinders", 60.0, [
        s for sy in (1.0, -1.0) for s in (
            cyl(0.06, barrel_len, at=(barrel_len / 2, sy * ARM_Y, 0.0), axis="x", mat="dark",
                collision=False),
            cyl(0.05, 0.12, at=(0.0, sy * ARM_Y, 0.0), axis="y", mat="steel", collision=False),
        )])
    # Rod frame at the rod eye, +x pointing back along the barrel to the base pin.
    rods = Link("lift_rods", 40.0, [
        s for sy in (1.0, -1.0) for s in (
            cyl(0.03, rod_len, at=(rod_len / 2, sy * ARM_Y, 0.0), axis="x", mat="steel",
                collision=False),
            cyl(0.045, 0.12, at=(0.0, sy * ARM_Y, 0.0), axis="y", mat="steel", collision=False),
        )])
    barrel_ang = (lk["cb"], lk["kb"] * stroke + lk["cb"])

    sprocket_x = -(TRACK_L - TRACK_H) / 2
    links = [tractor, _wing(1.0), _wing(-1.0), screed, extension(1.0), extension(-1.0),
             barrels, rods,
             sprocket_link("left_sprocket", SPROCKET_R, SHOE_W + 0.03, 60.0),
             sprocket_link("right_sprocket", SPROCKET_R, SHOE_W + 0.03, 60.0)]

    joints = [
        Joint("left_wing_joint", "revolute", "tractor", "left_wing",
              xyz=(X_HOP_R, HINGE_Y, HINGE_Z - Z0), axis=(1, 0, 0),
              lower=WING_LO, upper=WING_HI, init=WING_HI, velocity=0.25, role="tool"),
        Joint("right_wing_joint", "revolute", "tractor", "right_wing",
              xyz=(X_HOP_R, -HINGE_Y, HINGE_Z - Z0), axis=(-1, 0, 0),
              lower=WING_LO, upper=WING_HI, init=WING_HI, velocity=0.25, role="follower",
              follows=("left_wing_joint", 1.0, 0.0)),
        # screed lift: cylinder stroke (+ = retract = raise), rod slides into the barrel
        Joint("screed_lift_joint", "prismatic", "lift_cylinders", "lift_rods",
              xyz=(L0, 0.0, 0.0), rpy=(0.0, math.pi, 0.0), axis=(1, 0, 0),
              lower=0.0, upper=stroke, init=stroke, velocity=stroke / LIFT_TIME, role="tool"),
        Joint("lift_cylinder_joint", "revolute", "tractor", "lift_cylinders",
              xyz=(LIFT_BASE[0], 0.0, LIFT_BASE[1] - Z0), axis=(0, -1, 0),
              lower=min(barrel_ang) - 0.05, upper=max(barrel_ang) + 0.05, init=barrel_ang[1],
              velocity=1.0, role="follower", follows=("screed_lift_joint", lk["kb"], lk["cb"])),
        Joint("tow_arm_joint", "revolute", "tractor", "screed",
              xyz=(TOW[0], 0.0, TOW[1] - Z0), axis=(0, 1, 0),
              lower=LIFT_LO, upper=LIFT_HI, init=LIFT_HI, velocity=LIFT_HI / LIFT_TIME,
              role="follower", follows=("screed_lift_joint", lk["kq"], 0.0)),
        Joint("left_extension_joint", "prismatic", "screed", "left_extension",
              xyz=(ext_x - TOW[0], 0.0, -TOW[1]), axis=(0, 1, 0),
              lower=0.0, upper=EXT_STROKE, init=0.0, velocity=0.15, role="tool"),
        Joint("right_extension_joint", "prismatic", "screed", "right_extension",
              xyz=(ext_x - TOW[0], 0.0, -TOW[1]), axis=(0, -1, 0),
              lower=0.0, upper=EXT_STROKE, init=0.0, velocity=0.15, role="follower",
              follows=("left_extension_joint", 1.0, 0.0)),
        Joint("left_sprocket_joint", "continuous", "tractor", "left_sprocket",
              xyz=(sprocket_x, y_t, 0.0), axis=(0, 1, 0), velocity=20.0, role="roll_left",
              wheel_radius=SPROCKET_R),
        Joint("right_sprocket_joint", "continuous", "tractor", "right_sprocket",
              xyz=(sprocket_x, -y_t, 0.0), axis=(0, 1, 0), velocity=20.0, role="roll_right",
              wheel_radius=SPROCKET_R),
    ]

    return MachineSpec(
        name="asphalt_paver",
        abbrev="PAV",
        display_name="Asphalt Paver",
        category="paving",
        reference="19 t tracked asphalt paver (Vögele Super 1800-3 / CAT AP655F class)",
        description="Rubber-tracked asphalt paver with a front hopper with folding wings, "
                    "operator platform under a roof, and a floating 2.55–5.0 m extending "
                    "screed towed on two tow arms.",
        root="tractor",
        links=links,
        joints=joints,
        drive=Drive(type="skid", max_speed=1.25, max_yaw_rate=0.3, track_width=GAUGE),
        tools=[
            Tool("left_wing_joint", "hopper_wings", positive="raise", negative="lower"),
            Tool("screed_lift_joint", "screed", positive="raise", negative="lower"),
            Tool("left_extension_joint", "screed_extensions", positive="extend", negative="retract"),
        ],
        cameras=[
            # operator eye at the front of the platform, looking down into the hopper
            Camera("front", "tractor", (0.0, 0.0, 2.95 - Z0), (0.0, 0.65, 0.0)),
            # screed camera under the rear lip of the roof, left of centre, looking straight
            # back and down over the auger at the whole screed: both extending units and
            # end gates, the screed top and the lift-cylinder rods (seen from above)
            Camera("left", "tractor", (X_PLAT_R - 0.13, 0.60, 3.45 - Z0), (0.0, 0.95, math.pi)),
            # side camera at the right rear corner of the platform, flush with its edge and
            # just ahead of the lift-cylinder bracket, looking back along the tractor side:
            # right lift cylinder from the side, tow-arm end, screed end and the right
            # extending unit / end gate.  The screed lift is a vertical motion, which the
            # roof camera sees nearly end-on.
            Camera("right", "tractor", (X_REAR + 0.15, -(PLAT_Y + 0.05), Z_BODY - 0.12 - Z0),
                   (0.0, 0.50, -3.05)),
        ],
        envelope=(6.45, 2.55, 3.75),
        operating_mass=19100.0,
        ground_contact=["tractor"],
        notes="Hopper wings: +1 folds them up (closes the hopper). The screed tool is the lift "
              "cylinder stroke in m (+1 = retract = raise); the tow arms (tow_arm_joint) and "
              "the cylinder barrels follow it. Its auto-check sees the rods rise; the screed "
              "rises with them. Extension is auto-verified.",
    )
