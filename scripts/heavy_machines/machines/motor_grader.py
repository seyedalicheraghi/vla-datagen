"""Motor grader (MG), 16-18 t class — modelled on a CAT 140 / John Deere 672G.

Datasheet numbers (rounded; JD 672G/GP and CAT 140 / 140 GC / 140G sheets):
    wheelbase (front axle → tandem centre) 6.16 m (JD 672G) · tandem axle
    spacing 1.53 m · blade base (front axle → cutting edge) 2.57 m (CAT 140G) ·
    overall length 8.89 m without ripper (JD 672G); the rear ripper adds
    ≈ 0.97 m (beam raised) → 9.9 m · transport width 2.49 m (width over
    tyres) · height over ROPS cab 3.18 m (JD) / ≈ 3.3 m (CAT) · operating mass
    17.57 t typically equipped (CAT 140 GC; base machines ≈ 16 t) ·
    14.00-24 tyres (OD 1.34 m, section 0.37 m) on all six wheels, tread 2.12 m ·
    moldboard 3.66 × 0.61 m (12 ft × 24 in), 360° hydraulic circle ·
    moldboard side shift ±0.683 m (JD 672G) · blade lift above ground 0.48-0.49 m ·
    max depth of cut 0.75 m (CAT; needs the blade banked) · height to the top
    of the lift cylinders 3.02 m (CAT 140M) · front-wheel steer ±47.5° · frame
    articulation ±20° · turning radius ≈ 7.3 m (with articulation) · top speed
    ≈ 44-47 km/h forward, ≈ 37 km/h reverse · rear ripper: 5 shank pockets
    (3 shanks fitted), ripping depth ≈ 0.45 m.
Assumed / typical for the class:
    front frame box beam 0.42 × 0.46 m, top 1.97-2.18 m above ground · drawbar ball
    0.41 m behind the front axle · circle Ø 1.56 m, its centre 0.48 m behind
    the cutting edge · lift cylinders hung from saddle trunnions 2.65 m above
    ground (barrel tops ≈ 2.96 m) and splayed out to ball joints on the drawbar
    0.80 m either side of the centreline · operator cab centred over the
    articulation, floor 1.55 m above ground · max virtual steer 0.72 rad (inner
    wheel 46° of the 47.5° lock, turning radius 7.0 m at the tandem centre and
    10.3 m at the outer front tyre without articulation) · ≈ 3.5 s lock to
    lock → steer rate 0.41 rad/s.

Joint ranges / rates (rad, m; per second at full command):
    blade_lift  -0.14..+0.213 rad drawbar pitch about the ball → cutting edge
                ≈ 0.3 m below grade .. 0.48 m above ground (model caps the cut at
                ≈ 0.3 m: the circle ring stays 0.26 m above grade, the drop-arm
                pads reach ≈ 0.16 m below it); 0.10 rad/s (full stroke 3.5 s)
    blade_rotate -1.40..+1.40 rad circle turn (±80°; the real circle turns 360°,
                but past ±90° the moldboard faces backwards and blade_shift
                "left" would move it right), 0.25 rad/s (14°/s)
    blade_shift -0.68..+0.68 m along the moldboard, 0.22 m/s
    ripper      -0.52..+0.675 rad parallelogram arm → shank tips 0.47 m below
                .. 0.43 m above ground; 0.35 rad/s (full stroke 3.4 s)
Drive: Ackermann front steer about the rear tandem centre, 12.5 m/s (45 km/h)
forward, 10 m/s reverse.  Frame articulation is not modelled (fixed hinge).

Init pose = roading / transport pose: blade raised (blade_lift 0.17), circle
turned 57° clockwise (left toe forward) and the moldboard shifted 0.32 m
towards the toe so both blade ends stay inside the tyre width, ripper raised.
Because the circle tilts with the drawbar, the raised cutting edge runs from
0.11 m above ground at the toe to 0.63 m at the heel (0.37 m at its centre).

Model check (offline FK): cutting edge (blade square) 0.48 m above ground at
full lift, 0.30-0.32 m below grade at full cut; L × W × H 9.92 × 2.52 × 3.30 m;
mass 17.65 t with ≈ 27 % on the front axle (typical graders 25-30 %); the
steered tyres clear the (shortened) front axle beam out to the ±0.83 rad
joint limits.

Cosmetic followers (all driven by their tool joint) keep the hydraulics
attached.  Each blade-lift cylinder rolls about x at its saddle trunnion, and
at the drawbar a ball link counter-rotates with the drawbar (k = -1) and carries
the rod, which rolls like the barrel; the roll is a linear fit (see
:func:`_lift_fit`) and the eye stays within 0.03 m of the barrel axis over
the whole stroke.  The upper ripper link moves with the ripper arm (k = +1)
and the ripper tool-bar counter-rotates (k = -1) so the shanks stay vertical
(parallelogram).

Frames: all geometry below is written in "machine coordinates" — x forward
from the rear tandem centre, y left, h = height above ground with every
joint at q = 0 — and converted to each link's frame with :func:`_loc`.
Root link = rear frame, origin at the tandem centre on the axle line
(z = 0 is 0.67 m above the ground).
"""

from __future__ import annotations

import math

from ..spec import (Camera, Drive, Joint, Link, MachineSpec, Shape, Tool, beam, box, cyl,
                    sphere, wheel_joint, wheel_link)

# --- Running gear ------------------------------------------------------------------
WHEEL_R, TYRE_W = 0.67, 0.37         # 14.00-24
TREAD = 2.12                         # tyre centre to tyre centre (front and tandem)
WHEELBASE = 6.16                     # front axle → tandem centre
TANDEM = 1.53                        # tandem axle spacing
X_FRONT = WHEELBASE

# --- Link origins in machine coordinates (x, y, height above ground) -------------------
O_REAR = (0.0, 0.0, WHEEL_R)         # root: tandem centre, axle line
O_HINGE = (1.45, 0.0, 1.20)          # articulation hinge (fixed) = front-frame origin
O_BALL = (5.75, 0.0, 0.98)           # drawbar ball joint = drawbar origin
O_CIRCLE = (3.11, 0.0, 0.78)         # circle centre (ring mid-plane)
O_BLADE = (3.11, 0.0, 0.00)          # moldboard frame: under the circle centre, edge level
O_RIP_ARM = (-2.12, 0.0, 0.80)       # ripper lower-arm pivot
O_RIP_LINK = (-2.12, 0.0, 1.18)      # ripper upper-link pivot
RIP_ARM_L = 0.80
O_RIP_BAR = (O_RIP_ARM[0] - RIP_ARM_L, 0.0, O_RIP_ARM[2])   # tool-bar pivot

# --- Blade ---------------------------------------------------------------------------
MB_L, MB_H = 3.66, 0.61              # 12 ft × 24 in moldboard
EDGE_X = 0.48                        # cutting edge ahead of the circle centre
CIRCLE_R = 0.78
# Lift cylinders hang from the saddle and splay out to the drawbar (an "A" seen
# from the front); trunnion → eye distance runs 1.23 .. 2.17 m over the stroke.
LIFT_EYE = (3.00, 0.80, 0.90)        # lift-cylinder eyes (ball joints) on the drawbar (±y)
LIFT_TOP = (3.03, 0.40, 2.65)        # lift-cylinder trunnions on the saddle (±y)
LIFT_CAP, LIFT_BARREL, LIFT_ROD = 0.30, 1.12, 1.15   # barrel above / below trunnion, rod
LIFT_LO, LIFT_HI, LIFT_INIT = -0.14, 0.213, 0.17
ROTATE_LIM = 1.40                    # ±80°: blade_shift left/right stays true in the machine frame
ROTATE_INIT = -1.00                  # 57° clockwise: left toe forward for roading
SHIFT_INIT = 0.32

# --- Ripper ---------------------------------------------------------------------------
RIP_LO, RIP_HI = -0.52, 0.675


# ---------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------

def _loc(o, x: float, y: float, h: float) -> tuple[float, float, float]:
    """Machine coordinates → frame with origin ``o`` (all frames axis-aligned at q = 0)."""
    return (x - o[0], y - o[1], h - o[2])


def _xz(o, x: float, h: float) -> tuple[float, float]:
    return (x - o[0], h - o[2])


def _rel(parent, child) -> tuple[float, float, float]:
    """Joint origin of ``child`` in the ``parent`` frame."""
    return tuple(c - p for c, p in zip(child, parent))  # type: ignore[return-value]


def _rod(p0, p1, radius: float, mat: str = "steel", length: float | None = None) -> Shape:
    """Visual-only cylinder from 3-D point p0 towards p1 (full length unless given)."""
    d = [b - a for a, b in zip(p0, p1)]
    n = math.sqrt(sum(v * v for v in d))
    u = [v / n for v in d]
    ln = n if length is None else length
    centre = tuple(a + v * ln / 2 for a, v in zip(p0, u))
    rpy = (-math.asin(u[1]), math.atan2(u[0], u[2]), 0.0)
    return cyl(radius, ln, at=centre, rpy=rpy, mat=mat, collision=False)


def _yaw_bar(p0, p1, width: float, depth: float, h: float, mat: str = "body") -> Shape:
    """Box from p0 to p1 in the x-y plane (top view) at height h."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    return box(math.hypot(dx, dy), width, depth, at=((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, h),
               rpy=(0.0, 0.0, math.atan2(dy, dx)), mat=mat)


def _drawbar_point(p, q: float) -> tuple[float, float, float]:
    """Machine-coordinate point p carried by the drawbar, after blade_lift = q.

    The drawbar joint axis is (0, 1, 0): +q tips the drawbar's -x end (the
    circle) up about the ball.
    """
    x, y, z = _loc(O_BALL, *p)
    c, s = math.cos(q), math.sin(q)
    return (O_BALL[0] + x * c + z * s, y, O_BALL[2] - x * s + z * c)


def _lift_fit() -> tuple[tuple[float, float, float], float, float]:
    """Left lift-cylinder axis (trunnion → eye) as ``Rx(k * blade_lift + c) · d0``.

    The eye swings about the drawbar ball (almost straight up and down), so the
    cylinder mainly rolls about x at the trunnion.  Followers are linear, so
    fit the roll angle to a line in blade_lift (least-squares slope, offset
    centred on the residuals) and give d0 the fixed fore-aft lean that
    minimises the worst eye offset from the axis.  Over the whole stroke the
    eye stays within ≈ 0.03 m of the axis (rod r 0.05 inside barrel r 0.085).
    The right cylinder is the mirror image (roll -(k q + c)).
    """
    qs = [LIFT_LO + (LIFT_HI - LIFT_LO) * i / 40 for i in range(41)]
    vs = [tuple(e - t for e, t in zip(_drawbar_point(LIFT_EYE, q), LIFT_TOP)) for q in qs]
    rolls = [math.atan2(v[1], -v[2]) for v in vs]
    qm, rm = sum(qs) / len(qs), sum(rolls) / len(rolls)
    k = (sum((q - qm) * (r - rm) for q, r in zip(qs, rolls))
         / sum((q - qm) ** 2 for q in qs))
    res = [r - k * q for q, r in zip(qs, rolls)]
    leans = [(math.atan2(v[0], math.hypot(v[1], v[2])), math.sqrt(sum(a * a for a in v))) for v in vs]
    a0, a1 = min(a for a, _ in leans), max(a for a, _ in leans)
    lean = min((a0 + (a1 - a0) * i / 100 for i in range(101)),
               key=lambda a: max(n * abs(math.sin(a - ae)) for ae, n in leans))
    return (math.sin(lean), 0.0, -math.cos(lean)), k, (max(res) + min(res)) / 2


# ---------------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------------

def _cab() -> list[Shape]:
    """Centre-mounted operator cab over the articulation (rear-frame link)."""
    o = O_REAR
    x0, x1, y1, h0, h1 = -0.10, 1.45, 0.80, 1.55, 3.10
    cx, ch = (x0 + x1) / 2, (h0 + h1) / 2
    gx = x1 + 0.012                                   # front glass plane
    return [
        box(1.75, 1.70, 0.15, at=_loc(o, 0.70, 0.0, 1.475), mat="dark"),          # cab platform
        box(x1 - x0, 2 * y1, h1 - h0, at=_loc(o, cx, 0.0, ch)),                   # cab shell
        box(x1 - x0 + 0.14, 2 * y1 + 0.10, 0.10, at=_loc(o, cx, 0.0, h1 + 0.05)),  # roof
        box(x1 - x0 + 0.10, 2 * y1 + 0.06, 0.05, at=_loc(o, cx, 0.0, h1 - 0.02),
            mat="accent", collision=False),                                       # roof seam
        # glass (visual only), slightly proud of the shell
        box(0.025, 1.40, 0.82, at=_loc(o, gx, 0.0, 2.62), mat="glass", collision=False),
        box(0.025, 1.10, 0.52, at=_loc(o, gx, 0.0, 1.92), mat="glass", collision=False),
        *[box(1.30, 0.025, 1.30, at=_loc(o, 0.70, s * (y1 + 0.012), 2.38), mat="glass",
              collision=False) for s in (1.0, -1.0)],
        box(0.025, 1.20, 0.70, at=_loc(o, x0 - 0.012, 0.0, 2.65), mat="glass", collision=False),
        # pillars / door frames
        *[box(0.07, 0.05, h1 - h0, at=_loc(o, x, s * (y1 + 0.015), ch), mat="accent",
              collision=False) for x in (x1 - 0.03, x0 + 0.03) for s in (1.0, -1.0)],
        box(0.03, 1.42, 0.05, at=_loc(o, gx + 0.005, 0.0, 2.19), mat="accent", collision=False),
        # rear-view mirrors on the rear corners (clear of the cameras), roof lights, beacon
        *[_rod(_loc(o, x0 + 0.02, s * y1, 2.75), _loc(o, x0 - 0.10, s * (y1 + 0.22), 2.75), 0.02,
               mat="accent") for s in (1.0, -1.0)],
        *[box(0.04, 0.14, 0.26, at=_loc(o, x0 - 0.10, s * (y1 + 0.22), 2.63), mat="accent",
              collision=False) for s in (1.0, -1.0)],
        *[box(0.08, 0.16, 0.09, at=_loc(o, x, s * 0.52, h1 + 0.06), mat="white", collision=False)
          for x in (x1 + 0.08, x0 - 0.08) for s in (1.0, -1.0)],
        cyl(0.07, 0.10, at=_loc(o, 0.20, -0.45, h1 + 0.15), mat="orange", collision=False),
        # grab rails along the cab sides
        *[box(1.10, 0.04, 0.04, at=_loc(o, 0.65, s * (y1 + 0.06), 1.80), mat="accent",
              collision=False) for s in (1.0, -1.0)],
    ]


def _rear_frame() -> Link:
    o = O_REAR
    tandem_x = (TANDEM / 2, -TANDEM / 2)
    y_t = TREAD / 2 - TYRE_W / 2 - 0.125         # tandem housing centre, just inboard of the tyres
    return Link("rear_frame", 8000.0, [
        # main frame / transmission under the cab and engine
        box(3.30, 0.90, 0.55, at=_loc(o, -0.30, 0.0, 1.125), mat="dark"),
        box(0.25, 0.70, 0.45, at=_loc(o, 1.30, 0.0, 1.20), mat="dark"),          # hinge boss
        # tandem drive housings (oscillating bogie boxes) + pivots
        *[box(2.10, 0.22, 0.54, at=_loc(o, 0.0, s * y_t, WHEEL_R), mat="dark") for s in (1.0, -1.0)],
        *[cyl(0.30, 0.23, at=_loc(o, x, s * y_t, WHEEL_R), axis="y", mat="dark", collision=False)
          for x in tandem_x for s in (1.0, -1.0)],
        *[cyl(0.20, 0.22, at=_loc(o, 0.0, s * 0.55, WHEEL_R + 0.10), axis="y", mat="dark")
          for s in (1.0, -1.0)],
        # engine enclosure: hood, sloping top, side doors, rear grille
        box(1.85, 1.30, 1.05, at=_loc(o, -1.075, 0.0, 1.625)),
        beam(_xz(o, -0.15, 2.19), _xz(o, -2.00, 2.10), 1.26, 0.10),
        *[box(1.30, 0.025, 0.55, at=_loc(o, -1.05, s * 0.66, 1.68), mat="accent", collision=False)
          for s in (1.0, -1.0)],
        *[box(0.04, 0.025, 0.55, at=_loc(o, x, s * 0.667, 1.68), mat="body", collision=False)
          for x in (-0.62, -1.05, -1.48) for s in (1.0, -1.0)],
        box(0.025, 1.10, 0.75, at=_loc(o, -2.012, 0.0, 1.66), mat="accent", collision=False),
        *[box(0.03, 1.12, 0.04, at=_loc(o, -2.02, 0.0, h), mat="dark", collision=False)
          for h in (1.42, 1.58, 1.74, 1.90)],
        # rear bumper / counterweight with ripper mounting bracket and lights
        box(0.30, 1.40, 0.40, at=_loc(o, -1.90, 0.0, 0.95), mat="dark"),
        box(0.18, 1.30, 0.60, at=_loc(o, -2.04, 0.0, 1.00), mat="dark"),
        *[box(0.03, 0.16, 0.10, at=_loc(o, -2.14, s * 0.55, 1.20), mat="red", collision=False)
          for s in (1.0, -1.0)],
        *[box(0.03, 0.14, 0.10, at=_loc(o, -2.025, s * 0.50, 2.02), mat="white", collision=False)
          for s in (1.0, -1.0)],
        # exhaust stack (right), air pre-cleaner (left), hydraulic tank behind the cab
        cyl(0.075, 0.80, at=_loc(o, -0.55, -0.40, 2.58), mat="accent", collision=False),
        cyl(0.12, 0.10, at=_loc(o, -0.55, -0.40, 2.22), mat="dark", collision=False),
        cyl(0.13, 0.32, at=_loc(o, -0.55, 0.40, 2.34), mat="dark", collision=False),
        box(0.30, 1.20, 0.30, at=_loc(o, -0.30, 0.0, 2.20), mat="body", collision=False),
        *_cab(),
    ])


def _front_frame() -> Link:
    o = O_HINGE
    xz = lambda x, h: _xz(o, x, h)  # noqa: E731
    shapes = [
        box(0.40, 0.60, 0.50, at=_loc(o, 1.50, 0.0, 1.20), mat="dark"),            # hinge yoke
        cyl(0.08, 0.66, at=_loc(o, 1.45, 0.0, 1.20), mat="steel", collision=False),  # hinge pin
        beam(xz(1.55, 1.30), xz(2.05, 1.95), 0.42, 0.46),                          # gooseneck
        beam(xz(2.00, 1.95), xz(5.85, 1.74), 0.42, 0.46),                          # main box beam
        # front nose over the axle, push plate, front axle (oscillation pivot)
        box(1.05, 0.76, 0.80, at=_loc(o, 6.175, 0.0, 1.50)),
        box(0.25, 1.20, 0.70, at=_loc(o, 6.825, 0.0, 1.15)),
        box(0.06, 1.24, 0.14, at=_loc(o, 6.97, 0.0, 0.87), mat="dark", collision=False),
        # axle beam stops short of the tyres' steering sweep (knuckles bridge the gap)
        box(0.26, 1.25, 0.24, at=_loc(o, X_FRONT, 0.0, WHEEL_R), mat="dark"),
        box(0.30, 0.40, 0.33, at=_loc(o, X_FRONT, 0.0, 0.955), mat="dark"),         # axle pivot
        cyl(0.04, 1.20, at=_loc(o, X_FRONT + 0.20, 0.0, 1.00), axis="y", mat="dark",
            collision=False),                                                      # wheel-lean bar
        # drawbar ball socket under the nose
        box(0.40, 0.40, 0.12, at=_loc(o, O_BALL[0], 0.0, 1.14), mat="dark"),
        # headlights on the nose, front grille slots on the push plate
        *[box(0.10, 0.16, 0.12, at=_loc(o, 6.62, s * 0.28, 1.96), mat="white", collision=False)
          for s in (1.0, -1.0)],
        *[box(0.025, 0.90, 0.04, at=_loc(o, 6.96, 0.0, h), mat="accent", collision=False)
          for h in (1.20, 1.32)],
        # lift-cylinder saddle: clamp around the beam, a short tower with a
        # cross-head, and slim ears either side of each barrel holding the
        # trunnion pin (along x)
        box(0.40, 0.62, 0.56, at=_loc(o, LIFT_TOP[0], 0.0, 1.92)),
        box(0.36, 0.36, 0.25, at=_loc(o, LIFT_TOP[0], 0.0, 2.325)),
        box(0.30, 0.66, 0.10, at=_loc(o, LIFT_TOP[0], 0.0, 2.50)),
        *[box(0.04, 0.14, 0.30, at=_loc(o, LIFT_TOP[0] + dx, s * LIFT_TOP[1], 2.60))
          for dx in (0.115, -0.115) for s in (1.0, -1.0)],
        *[cyl(0.055, 0.30, at=_loc(o, LIFT_TOP[0], s * LIFT_TOP[1], LIFT_TOP[2]), axis="x",
              mat="steel", collision=False) for s in (1.0, -1.0)],
    ]
    return Link("front_frame", 3850.0, shapes)


def _drawbar() -> Link:
    o = O_BALL
    xe, ye, he = LIFT_EYE
    return Link("drawbar", 750.0, [
        sphere(0.12, mat="steel"),                                                  # ball
        box(0.60, 0.30, 0.20, at=_loc(o, 5.47, 0.0, 0.96)),                         # tongue
        *[_yaw_bar(_loc(o, 5.25, s * 0.12, 0)[:2], _loc(o, xe + 0.05, s * 0.70, 0)[:2], 0.16, 0.22,
                   h=-0.04) for s in (1.0, -1.0)],                                  # A-frame arms
        box(1.30, 1.40, 0.08, at=_loc(o, O_CIRCLE[0], 0.0, 0.865)),                 # circle yoke plate
        box(0.18, 1.70, 0.20, at=_loc(o, xe, 0.0, 0.95)),                           # rear cross bar
        *[box(0.20, 0.06, 0.16, at=_loc(o, xe, s * (ye + dy), he), mat="dark", collision=False)
          for s in (1.0, -1.0) for dy in (0.10, -0.10)],                            # eye clevises
        # circle drive motor + gearbox (visual)
        cyl(0.11, 0.36, at=_loc(o, 3.80, 0.32, 1.06), mat="dark", collision=False),
        box(0.30, 0.30, 0.12, at=_loc(o, 3.80, 0.32, 0.92), mat="dark", collision=False),
        # circle guide shoes
        *[box(0.18, 0.16, 0.12, at=_loc(o, O_CIRCLE[0] + CIRCLE_R * math.cos(a),
                                        CIRCLE_R * math.sin(a), 0.82), mat="dark", collision=False)
          for a in (0.8, -0.8, 2.4, -2.4)],
    ])


def _circle() -> Link:
    o = O_CIRCLE
    n, r_mid, w = 12, CIRCLE_R - 0.08, 0.16
    seg = 2 * (r_mid + w / 2) * math.tan(math.pi / n)
    return Link("circle", 650.0, [
        # ring gear as 12 flat segments (open centre, tight collision bounds)
        *[box(w, seg, 0.09, at=(r_mid * math.cos(a), r_mid * math.sin(a), 0.0), rpy=(0.0, 0.0, a),
              mat="steel") for a in (2 * math.pi * (i + 0.5) / n for i in range(n))],
        # drop arms carrying the moldboard slide rails and the tip pivots
        *[beam(_xz(o, O_CIRCLE[0] + 0.05, 0.76), _xz(o, O_CIRCLE[0] + EDGE_X - 0.24, 0.22), 0.10,
               0.20, y=s * 0.62) for s in (1.0, -1.0)],
        *[box(0.16, 0.18, 0.40, at=_loc(o, O_CIRCLE[0] + EDGE_X - 0.24, s * 0.62, 0.38), mat="dark")
          for s in (1.0, -1.0)],
        box(0.20, 1.40, 0.10, at=_loc(o, O_CIRCLE[0] + 0.12, 0.0, 0.73), mat="dark"),  # arm cross tie
        # blade tip cylinder (visual)
        _rod(_loc(o, O_CIRCLE[0] - 0.30, 0.0, 0.80), _loc(o, O_CIRCLE[0] + EDGE_X - 0.18, 0.0, 0.60),
             0.06, mat="dark"),
    ])


def _moldboard() -> Link:
    o = O_BLADE
    xe = O_BLADE[0] + EDGE_X
    xz = lambda dx, h: _xz(o, xe + dx, h)  # noqa: E731
    return Link("moldboard", 1100.0, [
        # curved moldboard: three facets (yellow back), polished face (visual)
        beam(xz(0.00, 0.06), xz(-0.10, 0.22), MB_L, 0.05),
        beam(xz(-0.10, 0.22), xz(-0.12, 0.44), MB_L, 0.05),
        beam(xz(-0.12, 0.44), xz(-0.04, MB_H), MB_L, 0.05),
        beam(xz(0.025, 0.07), xz(-0.075, 0.23), MB_L - 0.04, 0.02, mat="steel", collision=False),
        beam(xz(-0.075, 0.23), xz(-0.095, 0.44), MB_L - 0.04, 0.02, mat="steel", collision=False),
        beam(xz(-0.095, 0.44), xz(-0.02, 0.60), MB_L - 0.04, 0.02, mat="steel", collision=False),
        # cutting edge and end bits
        box(0.05, MB_L, 0.15, at=_loc(o, xe - 0.005, 0.0, 0.075), mat="steel"),
        *[box(0.06, 0.03, 0.30, at=_loc(o, xe - 0.01, s * (MB_L / 2 - 0.10), 0.15), mat="steel",
              collision=False) for s in (1.0, -1.0)],
        # side-shift slide rails on the back, side-shift cylinder, end plates
        box(0.10, MB_L - 0.30, 0.10, at=_loc(o, xe - 0.20, 0.0, 0.24), mat="dark"),
        box(0.10, MB_L - 0.30, 0.10, at=_loc(o, xe - 0.19, 0.0, 0.46), mat="dark"),
        cyl(0.05, 1.70, at=_loc(o, xe - 0.22, 0.45, 0.35), axis="y", mat="dark", collision=False),
        cyl(0.03, 1.30, at=_loc(o, xe - 0.22, -0.50, 0.35), axis="y", mat="steel", collision=False),
        *[beam(xz(-0.02, 0.0), xz(-0.17, 0.42), 0.04, 0.10, y=s * (MB_L / 2 - 0.02))
          for s in (1.0, -1.0)],
    ])


def _ripper() -> list[Link]:
    arm = Link("ripper_arm", 380.0, [
        *[beam((0.0, 0.0), (-RIP_ARM_L, 0.0), 0.10, 0.20, y=s * 0.55) for s in (1.0, -1.0)],
        cyl(0.07, 1.20, axis="y", mat="steel", collision=False),                   # pivot pin
    ])
    link = Link("ripper_upper_link", 80.0, [
        *[beam((0.0, 0.0), (-RIP_ARM_L, 0.0), 0.08, 0.12, y=s * 0.55, collision=False)
          for s in (1.0, -1.0)],
        cyl(0.05, 1.20, axis="y", mat="steel", collision=False),
    ])
    o = O_RIP_BAR
    shanks = []
    for y in (0.0, 0.80, -0.80):
        shanks += [
            box(0.30, 0.20, 0.28, at=_loc(o, o[0] - 0.02, y, 0.75), mat="dark"),        # shank pocket
            box(0.08, 0.06, 0.66, at=_loc(o, o[0] - 0.04, y, 0.36), mat="dark"),        # shank
            beam(_xz(o, o[0] - 0.04, 0.07), _xz(o, o[0] + 0.10, -0.04), 0.07, 0.07, y=y,
                 mat="steel"),                                                          # tip
        ]
    bar = Link("ripper_toolbar", 620.0, [
        box(0.26, 2.10, 0.26, at=_loc(o, o[0] - 0.05, 0.0, 0.78)),                    # tool bar
        *[box(0.12, 0.10, 0.40, at=_loc(o, o[0], s * 0.55, 1.00)) for s in (1.0, -1.0)],  # masts
        *[cyl(0.06, 0.16, at=_loc(o, o[0], s * 0.55, h), axis="y", mat="steel", collision=False)
          for s in (1.0, -1.0) for h in (O_RIP_ARM[2], O_RIP_LINK[2])],
        *shanks,
    ])
    return [arm, link, bar]


# ---------------------------------------------------------------------------------
# Machine
# ---------------------------------------------------------------------------------

def build() -> MachineSpec:
    y_w = TREAD / 2

    lift_j = Joint("blade_lift_joint", "revolute", "front_frame", "drawbar",
                   xyz=_rel(O_HINGE, O_BALL), axis=(0, 1, 0), lower=LIFT_LO, upper=LIFT_HI,
                   init=LIFT_INIT, velocity=0.10, role="tool")
    rip_j = Joint("ripper_joint", "revolute", "rear_frame", "ripper_arm",
                  xyz=_rel(O_REAR, O_RIP_ARM), axis=(0, 1, 0), lower=RIP_LO, upper=RIP_HI,
                  init=RIP_HI, velocity=0.35, role="tool")

    # --- blade-lift cylinders (cosmetic followers of blade_lift).  The barrel rolls
    # about x at its trunnion; at the eye, a ball link cancels the drawbar pitch
    # (k = -1, so it stays aligned with the frame) and the rod rolls like the barrel.
    d0, k_roll, c_roll = _lift_fit()
    rolls = (k_roll * LIFT_LO + c_roll, k_roll * LIFT_HI + c_roll)
    cyl_links, cyl_joints = [], []
    for side, s in (("left", 1.0), ("right", -1.0)):
        top = (LIFT_TOP[0], s * LIFT_TOP[1], LIFT_TOP[2])
        eye0 = (LIFT_EYE[0], s * LIFT_EYE[1], LIFT_EYE[2])
        lo, hi = sorted(s * r for r in rolls)
        roll = dict(axis=(1, 0, 0), lower=lo - 0.05, upper=hi + 0.05,
                    init=s * (k_roll * LIFT_INIT + c_roll), velocity=lift_j.velocity + 0.1,
                    role="follower", follows=(lift_j.name, s * k_roll, s * c_roll))
        cyl_links += [
            Link(f"{side}_lift_barrel", 80.0, [
                _rod(tuple(-LIFT_CAP * v for v in d0), tuple(LIFT_BARREL * v for v in d0), 0.085,
                     mat="dark"),
                _rod(tuple((LIFT_BARREL - 0.06) * v for v in d0), tuple(LIFT_BARREL * v for v in d0),
                     0.095),                                                     # gland
            ]),
            Link(f"{side}_lift_eye", 10.0, [sphere(0.07, mat="steel", collision=False)]),
            Link(f"{side}_lift_rod", 30.0, [
                _rod((0.0, 0.0, 0.0), tuple(-v for v in d0), 0.05, length=LIFT_ROD)]),
        ]
        cyl_joints += [
            Joint(f"{side}_lift_barrel_joint", "revolute", "front_frame", f"{side}_lift_barrel",
                  xyz=_rel(O_HINGE, top), **roll),
            Joint(f"{side}_lift_eye_joint", "revolute", "drawbar", f"{side}_lift_eye",
                  xyz=_rel(O_BALL, eye0), axis=(0, 1, 0), lower=-LIFT_HI - 0.05,
                  upper=-LIFT_LO + 0.05, init=-LIFT_INIT, velocity=lift_j.velocity + 0.1,
                  role="follower", follows=(lift_j.name, -1.0, 0.0)),
            Joint(f"{side}_lift_rod_joint", "revolute", f"{side}_lift_eye", f"{side}_lift_rod",
                  **roll),
        ]

    # --- wheels: steered front pair on knuckles, four tandem wheels on the rear frame
    wheels, wheel_joints, knuckles, steer_joints = [], [], [], []
    for side, s in (("left", 1.0), ("right", -1.0)):
        kn = f"front_{side}_knuckle"
        knuckles.append(Link(kn, 120.0, [
            cyl(0.11, 0.24, at=(0.0, -s * 0.30, 0.0), axis="y", mat="dark"),           # spindle
            cyl(0.09, 0.50, at=(0.0, -s * 0.38, 0.05), mat="dark", collision=False),    # king pin
            box(0.12, 0.08, 0.50, at=(0.0, -s * 0.30, 0.30), mat="dark", collision=False),
            # fender over the tyre (turns with the wheel)
            beam((-0.62, 0.42), (-0.28, 0.76), 0.38, 0.04, mat="body", collision=False),
            beam((-0.28, 0.76), (0.28, 0.76), 0.38, 0.04, mat="body", collision=False),
            beam((0.28, 0.76), (0.62, 0.42), 0.38, 0.04, mat="body", collision=False),
            box(0.06, 0.06, 0.40, at=(0.0, -s * 0.24, 0.58), mat="dark", collision=False),
        ]))
        steer_joints.append(Joint(f"front_{side}_steer_joint", "revolute", "front_frame", kn,
                                  xyz=_rel(O_HINGE, (X_FRONT, s * y_w, WHEEL_R)), axis=(0, 0, 1),
                                  lower=-0.83, upper=0.83, velocity=0.8, role="steer"))
        name = f"front_{side}_wheel"
        wheels.append(wheel_link(name, WHEEL_R, TYRE_W, 290.0))
        wheel_joints.append(wheel_joint(f"{name}_joint", kn, name, (0.0, 0.0, 0.0), WHEEL_R))
        for pos, x in (("front", TANDEM / 2), ("rear", -TANDEM / 2)):
            name = f"tandem_{pos}_{side}_wheel"
            wheels.append(wheel_link(name, WHEEL_R, TYRE_W, 290.0))
            wheel_joints.append(wheel_joint(f"{name}_joint", "rear_frame", name,
                                            (x, s * y_w, 0.0), WHEEL_R))

    links = [_rear_frame(), _front_frame(), _drawbar(), _circle(), _moldboard(), *_ripper(),
             *cyl_links, *knuckles, *wheels]

    joints = [
        Joint("articulation_hinge", "fixed", "rear_frame", "front_frame",
              xyz=_rel(O_REAR, O_HINGE)),
        lift_j,
        Joint("blade_rotate_joint", "revolute", "drawbar", "circle",
              xyz=_rel(O_BALL, O_CIRCLE), axis=(0, 0, 1), lower=-ROTATE_LIM, upper=ROTATE_LIM,
              init=ROTATE_INIT, velocity=0.25, role="tool"),
        Joint("blade_shift_joint", "prismatic", "circle", "moldboard",
              xyz=_rel(O_CIRCLE, O_BLADE), axis=(0, 1, 0), lower=-0.68, upper=0.68,
              init=SHIFT_INIT, velocity=0.22, role="tool"),
        rip_j,
        Joint("ripper_upper_link_joint", "revolute", "rear_frame", "ripper_upper_link",
              xyz=_rel(O_REAR, O_RIP_LINK), axis=(0, 1, 0), lower=RIP_LO - 0.05,
              upper=RIP_HI + 0.05, init=RIP_HI, velocity=rip_j.velocity + 0.1, role="follower",
              follows=(rip_j.name, 1.0, 0.0)),
        Joint("ripper_toolbar_joint", "revolute", "ripper_arm", "ripper_toolbar",
              xyz=_rel(O_RIP_ARM, O_RIP_BAR), axis=(0, 1, 0), lower=-RIP_HI - 0.05,
              upper=-RIP_LO + 0.05, init=-RIP_HI, velocity=rip_j.velocity + 0.1, role="follower",
              follows=(rip_j.name, -1.0, 0.0)),
        *cyl_joints,
        *steer_joints,
        *wheel_joints,
    ]

    cab_h = lambda h: h - WHEEL_R  # noqa: E731  (height above ground → root z)
    return MachineSpec(
        name="motor_grader",
        abbrev="MG",
        display_name="Motor Grader",
        category="grading",
        reference="16-18 t motor grader (CAT 140 / John Deere 672G class)",
        description=("Six-wheel motor grader: long front box frame on a steered front axle, "
                     "cab over the articulation and engine on the rear tandem bogies, a 3.66 m "
                     "moldboard on a turntable circle under the front frame (lift, rotate, side "
                     "shift) and a rear parallelogram ripper."),
        root="rear_frame",
        links=links,
        joints=joints,
        drive=Drive(type="ackermann", max_speed=12.5, max_reverse_speed=10.0,
                    wheelbase=WHEELBASE, max_steer=0.72, steer_rate=0.41, fixed_axle_x=0.0,
                    steer_axle="front"),
        tools=[
            Tool("blade_lift_joint", "blade_lift", positive="raise", negative="lower"),
            Tool("blade_rotate_joint", "blade_rotate", positive="left", negative="right"),
            Tool("blade_shift_joint", "blade_shift", positive="left", negative="right"),
            Tool("ripper_joint", "ripper", positive="raise", negative="lower"),
        ],
        cameras=[
            # operator eye height, just ahead of the windscreen (outside the solid cab
            # box), looking down past the frame at the blade
            Camera("front", "rear_frame", (1.50, 0.0, cab_h(2.75)), (0.0, 0.55, 0.0)),
            # under the front corners of the roof overhang (outside the cab shell),
            # looking outward-forward at the blade toe / heel, lift cylinder and front wheel
            Camera("left", "rear_frame", (1.53, 0.86, cab_h(3.05)), (0.0, 0.45, 0.60)),
            Camera("right", "rear_frame", (1.53, -0.86, cab_h(3.05)), (0.0, 0.45, -0.60)),
        ],
        envelope=(9.90, 2.49, 3.25),
        operating_mass=17500.0,
        notes=("blade_rotate turns the circle (+1 = counter-clockwise from above). The real "
               "circle turns 360°; it is limited to ±80° so the moldboard never faces backwards "
               "and blade_shift left/right stays true in the machine frame. The lift cylinders "
               "(barrel, eye ball, rod) and the ripper parallelogram are cosmetic followers of "
               "blade_lift / ripper. Not modelled: frame "
               "articulation (fixed; front-steer Ackermann only), blade bank (independent "
               "left/right lift), moldboard tip, circle centre-shift and front-wheel lean."),
    )
