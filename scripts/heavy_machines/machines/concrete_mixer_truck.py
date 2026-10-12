"""Transit concrete mixer truck (CMT), 9 m³ drum on a 3-axle 6x4 chassis — modelled on a
Volvo FMX 6x4 cab-over chassis (Mercedes-Benz Arocs / MAN TGS 6x4 are the same class;
US conventional-cab mixers such as the Mack Granite have a bonnet and a different
silhouette) carrying a Liebherr HTM 905 class rear-discharge drum (McNeilus
11.5-12 yd³ class).

Datasheet / listing numbers used:
    chassis Volvo FMX 6x4, wheelbase 3.90 m + 1.37 m tandem → front axle to tandem
    centre 4.60 m · front overhang 1.45 m · tyres 315/80 R22.5 all round (OD 1.08 m,
    section 0.315 m), single front, dual rear · front track 2.05 m, rear dual centres
    1.82 m (twin spacing 0.35 m) → 2.49 m over the rear tyres · day cab 2.30 m wide,
    roof 3.1 m above the ground · drum nominal 9 m³ (11.8 yd³), geometric ≈ 14 m³,
    max diameter 2.36 m, drum inclination 12°, drum speed 0-14 rpm (charge/mix
    ≈ 14 rpm max) · loading (hopper) height 3.95 m · discharge chute 180° swivel
    (±90°), main chute 1.5 m + folding extension · empty (operating) mass ≈ 13.0 t
    (chassis-cab ≈ 9.0 t + mixer superstructure ≈ 4.0 t) · overall 9.5 × 2.5 × 3.9 m ·
    top speed 80 km/h (governed) · turning circle ≈ 18 m.
Assumed / typical for the class (not in the sources available here):
    drum profile: head cone 0.75 m (Ø1.6 → Ø2.36), cylinder 1.10 m, rear cone 2.70 m
    (Ø2.36 → Ø1.10), neck 0.25 m, lowest point 1.24 m above the ground, opening rim
    centre 3.24 m above the ground · roller track Ø1.34 m on the rear cone 0.45 m ahead
    of the opening · frame top 1.02 m, mixer subframe top 1.17 m · swivel pivot 2.30 m
    above the ground, 1.95 m behind the tandem centre · folding extension 0.85 m, lying
    inverted on the main chute · chute lift −0.55 … +0.35 rad
    (31° down … 20° up) by a hydraulic cylinder, 0.25 rad/s · hydraulic swivel
    0.30 rad/s (180° in ≈ 10 s) · max virtual steer 0.60 rad (inner wheel 39°, outer
    31°) at 0.30 rad/s (2 s centre → full lock, 4 s lock to lock: ≈ 4.5 steering-wheel
    turns at ≈ 1.1 rev/s, a normal manoeuvring pace) · reverse 4 m/s · mass split:
    chassis (frame, engine, axles, suspension, tanks, mixer subframe, pedestals, hopper,
    bumper) 7.45 t, cab (day cab with interior, mirrors, steps) 1.20 t on a fixed
    joint, drum 2.80 t, ten wheels 105 kg each, front hubs/knuckles 120 kg each,
    chute + swivel + lift cylinder 0.26 t.  The cab is its own link so its large,
    mostly hollow shell does not drag the volume-weighted chassis centre of mass
    forward and up (Isaac's merge_fixed_joints folds it back into the chassis body).
Envelope = (9.5, 2.5, 3.9) m.  Drive max_speed = 11.1 m/s (40 km/h, a site / yard speed
as for the backhoe loader and the crane).  The chassis is governed at 80 km/h on the
road, but the environment is a construction site and the shared scripted drivers
command fixed fractions of max_speed (turn goals at 0.5, the demo at 0.6): at the road
speed a scripted 90° turn became a 30 m arc peaking at 13 m/s² lateral acceleration
(now 16 m at 4.6 m/s², in line with the backhoe loader and the crane).

Tools:
    drum        continuous about the drum axis, 1.45 rad/s (13.9 rpm).  +1 = "charge"
                (charging / mixing direction: clockwise seen from the rear), −1 =
                "discharge".  The internal mixing spiral is a LEFT-handed helix about
                the forward-pointing drum axis, so turning it clockwise (seen from the
                rear) screws the concrete forward into the drum; the two orange
                decorative spirals painted on the drum follow the fins, so with +1 they
                visibly travel forward (toward the cab) — checked numerically and in the
                sweep render.
    chute_swing swivel of the discharge chute about the vertical axis under the
                collecting funnel, ±1.55 rad.  +1 = "left" = counter-clockwise seen from
                above, named as seen looking along the chute (the rear-control operator's
                view, same convention as the backhoe loader's rear-pointing swing): the
                rear-pointing chute end moves to the truck's RIGHT side.  −1 = "right".
    chute_lift  main chute pitch about its hinge on the swivel, −0.55 … +0.35 rad,
                +1 = "raise".  The lift cylinder under the chute is cosmetic (followers:
                barrel angle and rod stroke are line fits of the true geometry).

Layout: left-hand-drive cab-over (cab on the front axle), water tank behind the cab,
front pedestal with the drum gearbox, drum tilted up toward the rear, rear pedestal with
two support rollers under the roller track, charging hopper over the drum opening,
collecting funnel under it, swivelling main chute with a folded extension, ladder and
platform on the left rear, fuel tank right, battery box left, side guards, tandem
fenders and mud flaps.

Cameras: front = driver's eye (left seat) just ahead of the windscreen, looking forward
and down at the road (a cab-over shows no part of the truck from there); left/right =
blind-spot side cameras on brackets at the upper rear corners of the cab, looking back
along the truck and slightly outward: the drum (with its painted spirals, so the drum
rotation is visible), the rear wheels and the ground beside the truck.  The 1.5 m main
chute stays tucked behind the drum and tandem from any cab-mounted viewpoint, so it is
not seen by any camera (on the real truck it is worked from the rear controls).

Init pose = travel pose: wheels straight, chute pointing rearward and raised to its
stop, drum at its reference angle.
Root frame: centre of the rear tandem (midway between the two rear axles) at axle
height, +x towards the steered front axle; the ground is at z = -WHEEL_R.
"""

from __future__ import annotations

import math

from ..spec import (DEFAULT_MATERIALS, Camera, Drive, Joint, Link, MachineSpec, Shape, Tool,
                    beam, box, cyl, sphere, wheel_joint, wheel_link)

# --- Wheels / axles ---------------------------------------------------------------------
WHEEL_R, TYRE_W = 0.538, 0.315          # 315/80 R22.5
WHEELBASE = 4.60                        # front axle → tandem centre
TANDEM = 1.37                           # rear axle spread
X_FRONT, X_R1, X_R2 = WHEELBASE, TANDEM / 2, -TANDEM / 2
Y_FRONT = 1.025                         # front track 2.05 m
Y_DUAL, TWIN = 0.91, 0.35               # rear dual centre, inner/outer tyre spacing

# --- Frame / cab (x in the chassis frame, z as height above the GROUND) -----------------
RAIL_Y0, RAIL_Y1 = 0.36, 0.44           # frame rails span |y| in [RAIL_Y0, RAIL_Y1]
FRAME_Z0, FRAME_Z1 = 0.72, 1.02         # truck frame rails
SUB_Z1 = 1.17                           # top of the mixer subframe
FRAME_X0, FRAME_X1 = -2.55, 5.80        # rear end / front end of the rails
CAB = (3.98, 5.93, 1.15, 1.10, 3.08)    # x0, x1, half-width, bottom, roof
CAB_MASS = 1200.0                       # day cab incl. interior, glass, mirrors, steps
DOOR = (4.05, 5.22)                     # door rear / front edge: over the axle and the steps
BUMPER_X = 6.05
SIDE_CAM = (3.86, 1.24, 3.00)           # blind-spot camera behind the upper rear cab corner
SIDE_CAM_RPY = (0.0, 0.30, math.pi - 0.20)   # left camera: 17° down, 11° outboard of dead aft

# --- Drum (chassis frame; s = distance along the axis from the head centre) -------------
DRUM_TILT = 0.21                        # 12° axis inclination (rises toward the rear)
HEAD = (3.25, 2.238)                    # head centre: x, height above the ground
R_MAX = 1.18
S_CYL0, S_CYL1, S_NECK, S_END = 0.75, 1.85, 4.55, 4.80
S_ROLLER = 4.35                         # roller track position
DRUM_RATE = 1.45                        # rad/s (13.9 rpm)

# --- Discharge chute ---------------------------------------------------------------------
SWIVEL = (-1.95, 2.30)                  # swivel axis x, top of the swivel ring (ground height)
HINGE_Z = -0.30                         # chute hinge below the ring top (swivel frame)
CHUTE_LEN, CHUTE_W = 1.50, 0.50         # main chute length from the hinge, outer width
EXT_LEN = 0.85                          # folding extension chute (folded on top)
LIFT_LO, LIFT_HI = -0.55, 0.35
SWING_LIM = 1.55
CYL_BASE = (-0.30, -0.75)               # lift-cylinder base pin (swivel frame x, z)
CYL_EYE = (0.62, -0.08)                 # rod-eye pin under the chute (chute frame x, z)


def Z(zg: float) -> float:
    """Height above the ground → chassis-frame z."""
    return zg - WHEEL_R


# ----------------------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------------------

def gbox(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, mat: str = "body",
         col: bool = True) -> Shape:
    """Axis-aligned box from its extents; z0/z1 are heights above the ground."""
    return box(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0),
               at=((x0 + x1) / 2, (y0 + y1) / 2, Z((z0 + z1) / 2)), mat=mat, collision=col)


def gbeam(p0: tuple[float, float], p1: tuple[float, float], width: float, depth: float,
          y: float = 0.0, mat: str = "body", col: bool = True) -> Shape:
    """beam() with (x, height above the ground) end points."""
    return beam((p0[0], Z(p0[1])), (p1[0], Z(p1[1])), width, depth, y=y, mat=mat, collision=col)


def rpy_from_axes(ex, ey, ez) -> tuple[float, float, float]:
    """URDF rpy of the rotation whose columns are the unit vectors ex, ey, ez."""
    r20, r21, r22 = ex[2], ey[2], ez[2]
    r10, r00 = ex[1], ex[0]
    pitch = -math.asin(max(-1.0, min(1.0, r20)))
    return (math.atan2(r21, r22), pitch, math.atan2(r10, r00))


def _unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def drum_r(s: float) -> float:
    """Outer radius of the drum profile at axial station s."""
    if s < S_CYL0:
        return 0.80 + (R_MAX - 0.80) * s / S_CYL0
    if s < S_CYL1:
        return R_MAX
    if s < S_NECK:
        return R_MAX - (R_MAX - 0.55) * (s - S_CYL1) / (S_NECK - S_CYL1)
    return 0.53


def drum_to_chassis(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """Drum-frame point (x toward the head, z ⟂ axis) → (x, y, height above ground)."""
    c, s = math.cos(DRUM_TILT), math.sin(DRUM_TILT)
    return (HEAD[0] + p[0] * c + p[2] * s, p[1], HEAD[1] - p[0] * s + p[2] * c)


# Visual/collision profile: (s0, s1, radius) stepped cylinders.
def _drum_segments() -> list[tuple[float, float, float]]:
    segs = [(0.0, 0.06, 0.74)]                                         # head plate
    n = 3
    for i in range(n):                                                 # head cone
        a, b = 0.06 + (S_CYL0 - 0.06) * i / n, 0.06 + (S_CYL0 - 0.06) * (i + 1) / n
        segs.append((a, b, drum_r((a + b) / 2)))
    segs.append((S_CYL0, S_CYL1, R_MAX))                               # cylinder
    n = 9
    for i in range(n):                                                 # rear cone
        a = S_CYL1 + (S_NECK - S_CYL1) * i / n
        b = S_CYL1 + (S_NECK - S_CYL1) * (i + 1) / n
        segs.append((a, b, drum_r((a + b) / 2)))
    segs.append((S_NECK, S_END, 0.53))                                 # neck
    return segs


DRUM_SEGS = _drum_segments()


# ----------------------------------------------------------------------------------------
# Parts
# ----------------------------------------------------------------------------------------

def _drum_shapes() -> list[Shape]:
    """Drum in its own frame: origin = head centre, +x along the axis toward the head."""
    shapes = [cyl(r, b - a, at=(-(a + b) / 2, 0.0, 0.0), axis="x", mat="drum")
              for a, b, r in DRUM_SEGS]
    shapes += [
        cyl(0.40, 0.08, at=(0.04, 0.0, 0.0), axis="x", mat="steel", collision=False),  # drive flange
        cyl(0.57, 0.06, at=(-S_END + 0.03, 0.0, 0.0), axis="x", mat="steel"),         # opening rim
        cyl(0.47, 0.01, at=(-S_END - 0.005, 0.0, 0.0), axis="x", mat="accent", collision=False),
        cyl(drum_r(S_ROLLER) + 0.07, 0.10, at=(-S_ROLLER, 0.0, 0.0), axis="x", mat="steel"),
        # weld seams between the cones and the cylinder
        cyl(R_MAX + 0.01, 0.03, at=(-S_CYL0, 0.0, 0.0), axis="x", mat="steel", collision=False),
        cyl(R_MAX + 0.01, 0.03, at=(-S_CYL1, 0.0, 0.0), axis="x", mat="steel", collision=False),
    ]
    # Inspection hatch on the cylinder (asymmetric detail, shows the drum angle).
    shapes.append(box(0.50, 0.40, 0.04, at=(-(S_CYL0 + S_CYL1) / 2, 0.0, R_MAX + 0.01),
                      mat="dark", collision=False))
    # Mixing blades visible in the opening: two flights crossing the neck.
    for ang in (0.0, math.pi / 2):
        shapes.append(box(0.24, 0.90, 0.04, at=(-S_END + 0.10, 0.0, 0.0), rpy=(ang + 0.5, 0.0, 0.0),
                          mat="dark", collision=False))
    # Two decorative spirals following the internal fins: LEFT-handed about +x
    # (phi increases with s, i.e. toward the opening), 1.25 turns from s = 0.3 to 4.5.
    k = 2.0 * math.pi * 1.25 / 4.2
    for phi0 in (0.0, math.pi):
        for a, b, r in DRUM_SEGS[1:-1]:
            n = max(1, math.ceil((b - a) / 0.16 - 1e-9))
            for i in range(n):
                s0, s1 = a + (b - a) * i / n, a + (b - a) * (i + 1) / n
                sm = (s0 + s1) / 2
                if not 0.30 <= sm <= 4.50:
                    continue
                phi = phi0 + k * (sm - 0.30)
                rr = r + 0.011
                ds = s1 - s0
                tangent = _unit((-ds, -rr * k * ds * math.sin(phi), rr * k * ds * math.cos(phi)))
                radial = (0.0, math.cos(phi), math.sin(phi))
                ey = _unit(_cross(radial, tangent))
                ez = _cross(tangent, ey)
                length = math.hypot(ds, rr * k * ds) * 1.04
                shapes.append(Shape(box(length, 0.22, 0.02).geom,
                                    (-sm, rr * math.cos(phi), rr * math.sin(phi)),
                                    rpy_from_axes(tangent, ey, ez), "orange", False, True))
    return shapes


def _cab() -> list[Shape]:
    """Left-hand-drive cab-over day cab on the front axle (tilting cab body only; the
    bumper and skid plate are frame-mounted, see _bumper)."""
    x0, x1, hw, z0, z1 = CAB
    dx0, dx1 = DOOR
    shapes = [
        gbox(x0, x1, -hw, hw, z0, z1),                                               # shell
        gbox(x0 + 0.03, x1 - 0.08, -hw + 0.05, hw - 0.05, z1, z1 + 0.06),          # roof cap
        # glass (visual only), slightly proud of the shell
        gbox(x1, x1 + 0.025, -1.02, 1.02, 2.08, 2.86, "glass", False),              # windscreen
        gbox(dx0 + 0.10, dx1 - 0.08, hw, hw + 0.02, 2.06, 2.82, "glass", False),    # left door
        gbox(dx0 + 0.10, dx1 - 0.08, -hw - 0.02, -hw, 2.06, 2.82, "glass", False),  # right door
        gbox(x0 - 0.02, x0, -0.45, 0.45, 2.35, 2.75, "glass", False),               # rear window
        # windscreen frame, sun visor, grille, lower grille
        gbox(x1, x1 + 0.03, -hw, -1.02, 2.03, 2.90, "accent", False),
        gbox(x1, x1 + 0.03, 1.02, hw, 2.03, 2.90, "accent", False),
        gbox(x1, x1 + 0.03, -1.02, 1.02, 2.02, 2.08, "accent", False),
        gbox(x1 - 0.12, x1 + 0.10, -1.10, 1.10, 2.88, 2.95, "accent", False),       # sun visor
        gbox(x1, x1 + 0.02, -0.80, 0.80, 1.30, 1.98, "dark", False),                # upper grille
        *[gbox(x1 + 0.02, x1 + 0.035, -0.76, 0.76, z, z + 0.05, "steel", False)
          for z in (1.38, 1.53, 1.68, 1.83)],
        gbox(5.72, x1, -hw, hw, 0.95, z0, "dark"),                                  # lower grille panel
        # black plastic wheel-arch corners ahead of the front wheels
        gbox(5.20, 5.75, 0.90, hw, 0.80, z0, "accent", False),
        gbox(5.20, 5.75, -hw, -0.90, 0.80, z0, "accent", False),
        # cab-corner deflectors and the tall air intake on the right rear corner
        gbox(x1 - 0.20, x1 + 0.01, hw, hw + 0.03, 1.15, 2.95, "accent", False),
        gbox(x1 - 0.20, x1 + 0.01, -hw - 0.03, -hw, 1.15, 2.95, "accent", False),
        gbox(x0 - 0.10, x0, -hw, -0.88, 1.70, 3.00, "accent", False),
        # roof beacons
        cyl(0.07, 0.12, at=(x0 + 0.20, 0.85, Z(z1 + 0.12)), mat="orange", collision=False),
        cyl(0.07, 0.12, at=(x0 + 0.20, -0.85, Z(z1 + 0.12)), mat="orange", collision=False),
    ]
    sx, sy, sz = SIDE_CAM
    for s in (1.0, -1.0):
        shapes += [
            # door (over the front axle, directly above the entry steps): seams and handle
            gbox(dx0, dx0 + 0.03, s * hw, s * (hw + 0.012), 1.15, 2.95, "accent", False),
            gbox(dx1 - 0.03, dx1, s * hw, s * (hw + 0.012), 1.15, 2.95, "accent", False),
            gbox(dx0 + 0.08, dx0 + 0.26, s * hw, s * (hw + 0.025), 1.88, 1.92, "steel", False),
            # rear-view mirror: arm, main head, wide-angle head
            gbox(5.80, 5.84, s * hw, s * 1.26, 2.64, 2.68, "accent", False),
            gbox(5.78, 5.86, s * 1.20, s * 1.27, 2.20, 2.64, "accent", False),
            gbox(5.78, 5.86, s * 1.19, s * 1.26, 1.98, 2.16, "accent", False),
            # entry steps behind the front wheel, under the rear edge of the door
            gbox(3.86, 4.03, s * 0.98, s * 1.17, 0.50, 0.54, "dark", False),
            gbox(3.86, 4.03, s * 0.98, s * 1.17, 0.80, 0.84, "dark", False),
            gbox(3.86, 3.89, s * 0.98, s * 1.17, 0.50, z0, "dark", False),
            gbox(4.00, 4.03, s * 0.98, s * 1.17, 0.50, z0, "dark", False),
            # grab handle on the B-pillar, beside the door opening
            gbox(x0 + 0.01, x0 + 0.04, s * (hw + 0.02), s * (hw + 0.05), 1.30, 2.40, "steel", False),
            # blind-spot side camera: bracket off the upper rear cab corner + housing
            gbox(sx + 0.06, x0 + 0.06, s * (hw - 0.02), s * (sy - 0.03), sz - 0.03, sz + 0.02,
                 "accent", False),
            gbox(sx + 0.01, sx + 0.11, s * (sy - 0.05), s * (sy + 0.05), sz - 0.06, sz + 0.06,
                 "dark", False),
        ]
    return shapes


def _bumper() -> list[Shape]:
    """Frame-mounted front bumper with headlights, and the skid plate under it."""
    return [
        gbox(5.80, BUMPER_X, -1.20, 1.20, 0.62, 0.98, "steel"),
        gbox(BUMPER_X, BUMPER_X + 0.02, 0.70, 1.05, 0.80, 0.92, "white", False),
        gbox(BUMPER_X, BUMPER_X + 0.02, -1.05, -0.70, 0.80, 0.92, "white", False),
        gbox(BUMPER_X, BUMPER_X + 0.02, -0.30, 0.30, 0.70, 0.74, "accent", False),
        gbox(5.70, 6.00, -0.75, 0.75, 0.48, 0.62, "dark"),
    ]


def _chassis_frame() -> list[Shape]:
    """Frame rails, engine, axles, suspension, tanks, guards, fenders, rear end."""
    shapes = [
        gbox(4.60 - 1.30, 5.30, -0.34, 0.34, 0.55, 1.08, "dark"),                  # engine
        gbox(2.50, 3.30, -0.26, 0.26, 0.62, 0.95, "dark"),                          # gearbox
        cyl(0.06, 1.70, at=(1.65, 0.0, Z(0.66)), axis="x", mat="steel", collision=False),  # prop shaft
        gbox(5.45, 5.72, -0.55, 0.55, 0.62, 1.08, "dark"),                          # radiator
        gbox(X_FRONT - 0.07, X_FRONT + 0.07, -0.84, 0.84, 0.42, 0.56, "dark"),     # front axle beam
        cyl(0.03, 1.70, at=(X_FRONT - 0.18, 0.0, Z(0.50)), axis="y", mat="steel", collision=False),
        # rear axles: housings + differentials
        *[s for x in (X_R1, X_R2) for s in (
            cyl(0.11, 1.50, at=(x, 0.0, 0.0), axis="y", mat="dark"),
            sphere(0.27, at=(x, 0.0, 0.0), mat="dark"))],
        # rear tandem: walking-beam springs, trunnion
        gbox(-1.05, 1.05, 0.46, 0.56, 0.60, 0.72, "dark", False),
        gbox(-1.05, 1.05, -0.56, -0.46, 0.60, 0.72, "dark", False),
        gbox(-0.16, 0.16, 0.44, 0.56, 0.70, FRAME_Z0, "dark", False),
        gbox(-0.16, 0.16, -0.56, -0.44, 0.70, FRAME_Z0, "dark", False),
        # rear end: cross member, underrun bar, tail lights
        gbox(FRAME_X0 - 0.03, FRAME_X0 + 0.05, -0.45, 0.45, FRAME_Z0, SUB_Z1, "dark"),
        gbox(FRAME_X0 - 0.08, FRAME_X0 + 0.04, -1.05, 1.05, 0.42, 0.58, "accent"),
        gbox(FRAME_X0 - 0.10, FRAME_X0 - 0.08, -1.00, -0.70, 0.45, 0.55, "red", False),
        gbox(FRAME_X0 - 0.10, FRAME_X0 - 0.08, 0.70, 1.00, 0.45, 0.55, "red", False),
        gbox(FRAME_X0 - 0.10, FRAME_X0 - 0.08, -0.30, 0.30, 0.47, 0.53, "white", False),
        # fuel tank (right) and exhaust silencer, battery box + air tanks (left)
        cyl(0.31, 1.30, at=(2.60, -0.80, Z(0.86)), axis="x", mat="steel"),
        gbox(3.30, 3.82, -1.12, -0.55, 0.60, 1.05, "steel"),
        gbox(2.10, 2.95, 0.50, 1.10, 0.60, 1.02, "dark"),
        cyl(0.13, 0.75, at=(3.45, 0.80, Z(0.74)), axis="x", mat="steel", collision=False),
        cyl(0.13, 0.75, at=(3.45, 0.80, Z(1.02)), axis="x", mat="steel", collision=False),
        # tandem fenders and mud flaps
        gbox(-1.32, 1.32, 0.62, 1.26, 1.12, 1.16, "accent"),
        gbox(-1.32, 1.32, -1.26, -0.62, 1.12, 1.16, "accent"),
        gbox(-1.34, -1.31, 0.62, 1.24, 0.40, 1.12, "rubber", False),
        gbox(-1.34, -1.31, -1.24, -0.62, 0.40, 1.12, "rubber", False),
        gbox(1.31, 1.34, 0.62, 1.24, 0.75, 1.12, "rubber", False),
        gbox(1.31, 1.34, -1.24, -0.62, 0.75, 1.12, "rubber", False),
    ]
    for s in (1.0, -1.0):
        shapes += [
            gbox(FRAME_X0, FRAME_X1, s * RAIL_Y0, s * RAIL_Y1, FRAME_Z0, FRAME_Z1, "dark"),   # rail
            gbox(FRAME_X0, 3.62, s * RAIL_Y0, s * (RAIL_Y1 + 0.01), FRAME_Z1, SUB_Z1, "drum"),  # subframe
            gbox(3.95, 5.25, s * 0.34, s * 0.46, 0.60, 0.70, "dark", False),          # front spring
            # side underrun guards
            gbox(1.45, 3.80, s * 1.16, s * 1.21, 0.55, 0.65, "steel", False),
            gbox(1.45, 3.80, s * 1.16, s * 1.21, 0.88, 0.98, "steel", False),
            *[gbox(x, x + 0.06, s * 0.44, s * 1.16, 0.88, 0.94, "dark", False) for x in (1.55, 3.70)],
            # underrun-bar brackets
            gbox(FRAME_X0 - 0.03, FRAME_X0 + 0.05, s * 0.38, s * 0.46, 0.55, FRAME_Z0, "dark", False),
        ]
    return shapes


def _mixer_static() -> list[Shape]:
    """Pedestals, gearbox, water tank, hopper, collecting funnel, ladder (chassis frame)."""
    gear = drum_to_chassis((0.19, 0.0, 0.0))
    motor = drum_to_chassis((0.37, 0.0, 0.0))
    axis_rpy = (0.0, math.pi / 2 + DRUM_TILT, 0.0)            # cylinder along the drum axis
    shapes = [
        # front pedestal + gearbox + hydraulic motor
        gbox(3.30, 3.62, -0.50, 0.50, SUB_Z1, 1.95, "drum"),
        gbox(3.26, 3.66, -0.56, 0.56, SUB_Z1, SUB_Z1 + 0.10, "drum"),
        cyl(0.27, 0.30, at=(gear[0], 0.0, Z(gear[2])), rpy=axis_rpy, mat="dark"),
        cyl(0.11, 0.10, at=(motor[0], 0.0, Z(motor[2])), rpy=axis_rpy, mat="steel", collision=False),
        # water tank behind the cab
        gbox(3.66, 3.94, -0.80, 0.80, 1.20, 2.55, "white"),
        cyl(0.07, 0.06, at=(3.80, 0.50, Z(2.58)), mat="accent", collision=False),
    ]
    # rear pedestal: two columns, cross beam, bearing blocks and support rollers
    for sgn in (1.0, -1.0):
        roller = drum_to_chassis((-S_ROLLER, sgn * 0.79 * math.sin(0.70), -0.79 * math.cos(0.70)))
        shapes += [
            gbox(-1.30, -0.98, sgn * 0.38, sgn * 0.60, SUB_Z1, 2.38, "drum"),
            gbox(roller[0] - 0.12, roller[0] + 0.12, sgn * (abs(roller[1]) - 0.09),
                 sgn * (abs(roller[1]) + 0.09), 2.38, roller[2] - 0.08, "dark"),
            cyl(0.12, 0.16, at=(roller[0], roller[1], Z(roller[2])), rpy=axis_rpy, mat="steel"),
            # diagonal brace to the subframe
            gbeam((-0.20, SUB_Z1), (-1.00, 2.20), 0.10, 0.10, y=sgn * 0.49, mat="drum", col=False),
        ]
    shapes.append(gbox(-1.30, -0.98, -0.60, 0.60, 2.20, 2.38, "drum"))

    # charging hopper over the opening: rear wall A→D slopes into the opening, short front
    # wall resting on the drum, side walls = rectangle + a strip along the rear wall
    rim_z = 3.95
    a, d = (-2.24, rim_z - 0.08), (-1.60, 2.98)
    ln = math.hypot(d[0] - a[0], d[1] - a[1])
    u = ((d[0] - a[0]) / ln, (d[1] - a[1]) / ln)
    n = (-u[1], u[0])                                         # in-plane normal, forward-up
    a2 = (a[0] + 0.20 * u[0], a[1] + 0.20 * u[1])
    off = 0.14                                                # half of the 0.28 m strip
    strip = ((a2[0] + off * n[0], a2[1] + off * n[1]), (d[0] + off * n[0], d[1] + off * n[1]))
    throat = drum_to_chassis((-S_END - 0.06, 0.0, 0.0))
    shapes += [
        gbox(-2.30, -1.25, 0.52, 0.58, rim_z - 0.10, rim_z, "drum"),                # rim sides
        gbox(-2.30, -1.25, -0.58, -0.52, rim_z - 0.10, rim_z, "drum"),
        gbox(-2.30, -2.22, -0.58, 0.58, rim_z - 0.10, rim_z, "drum"),                # rim back
        gbox(-1.33, -1.25, -0.58, 0.58, rim_z - 0.10, rim_z, "drum"),                # rim front
        gbeam(a, d, 1.04, 0.05, mat="drum"),                                          # rear wall
        gbeam((-1.29, rim_z - 0.08), (-1.36, 3.62), 1.04, 0.05, mat="drum"),          # front wall
        gbox(-1.36, -1.25, -0.57, 0.57, 3.58, 3.64, "drum", False),
        # collar around the drum opening
        cyl(0.62, 0.08, at=(throat[0], 0.0, Z(throat[2])), rpy=axis_rpy, mat="drum", collision=False),
    ]
    for sgn in (1.0, -1.0):
        shapes += [
            gbox(-1.90, -1.27, sgn * 0.52, sgn * 0.57, 3.40, rim_z - 0.08, "drum"),  # side walls
            gbeam(*strip, 0.05, 0.28, y=sgn * 0.545, mat="drum"),
            # hopper struts down to the rear pedestal
            gbeam((-1.10, 2.38), (-1.80, 3.80), 0.08, 0.10, y=sgn * 0.62, mat="drum", col=False),
        ]

    # collecting funnel under the opening: sloped floor down to the swivel outlet, back
    # plate above the (raised) main chute, side cheeks
    sx, sz = SWIVEL
    shapes += [
        gbeam((-1.42, 2.66), (sx + 0.14, sz + 0.14), 0.60, 0.04, mat="steel"),       # floor
        gbox(sx - 0.20, sx - 0.16, -0.32, 0.32, 2.47, 2.95, "steel"),                 # back plate
        gbeam((sx - 0.18, 2.49), (sx - 0.02, sz + 0.12), 0.56, 0.03, mat="steel", col=False),
        cyl(0.22, 0.12, at=(sx, 0.0, Z(sz + 0.06)), mat="steel", collision=False),   # outlet
    ]
    # side cheeks start just behind the drum's opening rim (s > S_END), clear of the drum
    for sgn in (1.0, -1.0):
        shapes.append(gbeam((-1.58, 2.80), (sx - 0.10, sz + 0.42), 0.04, 0.34,
                            y=sgn * 0.30, mat="steel"))

    # ladder and platform on the left rear
    lx0, lx1, ly = -1.64, -1.38, 1.08
    shapes += [gbox(x - 0.02, x + 0.02, ly - 0.02, ly + 0.02, 0.50, 3.22, "steel", False)
               for x in (lx0, lx1)]
    shapes += [gbox(lx0, lx1, ly - 0.02, ly + 0.02, z, z + 0.03, "steel", False)
               for z in [0.58 + 0.30 * i for i in range(9)]]
    shapes += [
        gbox(lx1 - 0.02, lx1 + 0.02, 0.60, ly, 1.95, 1.99, "steel", False),             # stay
        gbox(-2.10, -1.34, 0.62, 1.12, 3.18, 3.22, "steel", False),                    # platform
    ]
    # platform guard rail: posts + top and knee rails along the back and the outer side
    for x, y in ((-2.08, 0.64), (-2.08, 1.10), (-1.74, 1.10), (-1.40, 1.10)):
        shapes.append(gbox(x - 0.02, x + 0.02, y - 0.02, y + 0.02, 3.22, 3.84, "steel", False))
    for z in (3.52, 3.80):
        shapes.append(gbox(-2.10, -2.06, 0.62, 1.12, z, z + 0.04, "steel", False))
        shapes.append(gbox(-2.10, -1.38, 1.08, 1.12, z, z + 0.04, "steel", False))
    return shapes


def _swivel_shapes() -> list[Shape]:
    """Swivel ring + hinge lugs + cylinder bracket.  Frame: +x along the chute (rearward
    at swing 0), origin = top centre of the swivel ring."""
    bx, bz = CYL_BASE
    lug_z0, lug_z1 = HINGE_Z - 0.06, -0.04
    return [
        # swivel ring (fits between the chute walls) and the yoke carrying the hinge lugs
        cyl(0.20, 0.08, at=(0.0, 0.0, -0.04), mat="steel"),
        box(0.16, 0.66, 0.04, at=(0.0, 0.0, -0.06), mat="dark"),
        box(0.16, 0.05, lug_z1 - lug_z0, at=(0.0, 0.305, (lug_z0 + lug_z1) / 2), mat="dark"),
        box(0.16, 0.05, lug_z1 - lug_z0, at=(0.0, -0.305, (lug_z0 + lug_z1) / 2), mat="dark"),
        cyl(0.04, 0.68, at=(0.0, 0.0, HINGE_Z), axis="y", mat="steel", collision=False),
        # cylinder bracket hanging forward-down from the yoke
        beam((-0.05, -0.08), (bx, bz), 0.10, 0.10, y=0.30, mat="dark"),
        beam((-0.05, -0.08), (bx, bz), 0.10, 0.10, y=-0.30, mat="dark"),
        box(0.10, 0.68, 0.10, at=(bx, 0.0, bz), mat="dark", collision=False),
        cyl(0.05, 0.16, at=(bx, 0.0, bz), axis="y", mat="steel", collision=False),
    ]


def _chute_shapes() -> list[Shape]:
    """Main chute (U trough) with the folded extension; frame origin = hinge, +x along it."""
    hw = CHUTE_W / 2
    x0, x1 = -0.12, CHUTE_LEN
    w0 = 0.0                                           # walls start at the hinge
    ext_x = x1 - 0.03 - EXT_LEN / 2                    # extension hinge at x1 - 0.03
    shapes = [
        box(x1 - x0, CHUTE_W, 0.04, at=((x0 + x1) / 2, 0.0, -0.02), mat="steel"),         # floor
        box(x1 - w0, 0.03, 0.22, at=((w0 + x1) / 2, hw - 0.015, 0.09), mat="steel"),       # walls
        box(x1 - w0, 0.03, 0.22, at=((w0 + x1) / 2, -hw + 0.015, 0.09), mat="steel"),
        box(0.04, CHUTE_W, 0.10, at=(x1 - 0.02, 0.0, 0.03), mat="dark", collision=False),  # end lip
        # stiffener ribs, hinge boss, rod-eye lug
        *[box(0.05, CHUTE_W + 0.04, 0.05, at=(x, 0.0, -0.045), mat="dark", collision=False)
          for x in (0.10, 0.70, 1.30)],
        cyl(0.06, 0.48, axis="y", mat="dark", collision=False),
        box(0.14, 0.10, 0.10, at=(CYL_EYE[0], 0.0, CYL_EYE[1] + 0.03), mat="dark", collision=False),
        # folded extension chute lying inverted on top, hinged at the end; its free end
        # stays 0.62 m out from the hinge so it clears the collecting funnel's back plate
        # and cheeks while the raised chute swivels
        box(EXT_LEN, CHUTE_W - 0.04, 0.03, at=(ext_x, 0.0, 0.40), mat="steel", collision=False),
        box(EXT_LEN, 0.03, 0.18, at=(ext_x, hw - 0.035, 0.31), mat="steel", collision=False),
        box(EXT_LEN, 0.03, 0.18, at=(ext_x, -hw + 0.035, 0.31), mat="steel", collision=False),
        cyl(0.03, CHUTE_W + 0.06, at=(x1 - 0.03, 0.0, 0.24), axis="y", mat="dark", collision=False),
        # operator handle at the end
        box(0.30, 0.03, 0.03, at=(x1 + 0.12, 0.0, 0.18), mat="accent", collision=False),
    ]
    return shapes


# ----------------------------------------------------------------------------------------
# Lift-cylinder follower fits
# ----------------------------------------------------------------------------------------

def _cyl_vec(q: float) -> tuple[float, float]:
    """Base pin → rod eye in the swivel frame (x, z) for chute lift angle q."""
    c, s = math.cos(q), math.sin(q)
    ex, ez = CYL_EYE
    eye = (ex * c - ez * s, HINGE_Z + ex * s + ez * c)          # rotation about −y by q
    return eye[0] - CYL_BASE[0], eye[1] - CYL_BASE[1]


def _lift_fits() -> dict:
    """Minimax line fits of the barrel angle and the rod stroke vs the lift angle."""
    qs = [LIFT_LO + (LIFT_HI - LIFT_LO) * i / 60 for i in range(61)]
    vec = [_cyl_vec(q) for q in qs]
    ang = [math.atan2(v[1], v[0]) for v in vec]
    dist = [math.hypot(*v) for v in vec]

    def minimax(ys):
        k = (ys[-1] - ys[0]) / (qs[-1] - qs[0])
        res = [y - k * q for q, y in zip(qs, ys)]
        return k, (max(res) + min(res)) / 2, (max(res) - min(res)) / 2

    ka, ca, ea = minimax(ang)
    d_init = math.hypot(*_cyl_vec(LIFT_HI))
    kl, cl, el = minimax([d - d_init for d in dist])
    return {"angle": (ka, ca), "angle_err_m": ea * max(dist), "ext": (kl, cl), "ext_err_m": el,
            "d_init": d_init, "d_min": min(dist)}


# ----------------------------------------------------------------------------------------
# Machine
# ----------------------------------------------------------------------------------------

def build() -> MachineSpec:
    chassis = Link("chassis", 8650.0 - CAB_MASS, [*_chassis_frame(), *_bumper(), *_mixer_static()])
    cab = Link("cab", CAB_MASS, _cab())
    drum = Link("drum", 2800.0, _drum_shapes())
    swivel = Link("chute_swivel", 70.0, _swivel_shapes())
    chute = Link("chute", 150.0, _chute_shapes())

    fits = _lift_fits()
    ka, ca = fits["angle"]
    kl, cl = fits["ext"]
    d0 = fits["d_init"]
    barrel_len = fits["d_min"] - 0.10
    lift_barrel = Link("chute_lift_barrel", 25.0, [
        cyl(0.055, barrel_len, at=(barrel_len / 2, 0.0, 0.0), axis="x", mat="dark", collision=False),
        cyl(0.06, 0.05, at=(barrel_len - 0.03, 0.0, 0.0), axis="x", mat="accent", collision=False),
        cyl(0.05, 0.12, axis="y", mat="steel", collision=False),
    ])
    rod0 = d0 - fits["d_min"] + 0.06          # inner rod end stays inside the barrel when retracted
    lift_rod = Link("chute_lift_rod", 12.0, [
        cyl(0.03, d0 - rod0, at=((d0 + rod0) / 2, 0.0, 0.0), axis="x", mat="steel", collision=False),
        cyl(0.045, 0.10, at=(d0, 0.0, 0.0), axis="y", mat="steel", collision=False),
    ])

    sx, sz = SWIVEL
    drum_j = Joint("drum_joint", "continuous", "chassis", "drum",
                   xyz=(HEAD[0], 0.0, Z(HEAD[1])), rpy=(0.0, DRUM_TILT, 0.0), axis=(1, 0, 0),
                   velocity=DRUM_RATE, role="tool")
    swing_j = Joint("chute_swing_joint", "revolute", "chassis", "chute_swivel",
                    xyz=(sx, 0.0, Z(sz)), rpy=(0.0, 0.0, math.pi), axis=(0, 0, 1),
                    lower=-SWING_LIM, upper=SWING_LIM, init=0.0, velocity=0.30, role="tool")
    lift_j = Joint("chute_lift_joint", "revolute", "chute_swivel", "chute",
                   xyz=(0.0, 0.0, HINGE_Z), axis=(0, -1, 0), lower=LIFT_LO, upper=LIFT_HI,
                   init=LIFT_HI, velocity=0.25, role="tool")

    def follower(name, jtype, parent, child, xyz, axis, k, c) -> Joint:
        ends = (k * lift_j.lower + c, k * lift_j.upper + c)
        return Joint(name, jtype, parent, child, xyz=xyz, axis=axis,
                     lower=min(ends) - 0.02, upper=max(ends) + 0.02, init=k * lift_j.init + c,
                     velocity=abs(k) * lift_j.velocity + 0.05, role="follower",
                     follows=(lift_j.name, k, c))

    # ---------------------------------------------------------------------- wheels
    wheel_links, wheel_joints, knuckles, steer_joints = [], [], [], []
    for side, s in (("left", 1.0), ("right", -1.0)):
        kn = f"front_{side}_knuckle"
        knuckles.append(Link(kn, 120.0, [
            cyl(0.12, 0.10, at=(0.0, -s * 0.21, 0.0), axis="y", mat="dark"),          # hub / spindle
            cyl(0.06, 0.32, at=(0.0, -s * 0.21, 0.0), mat="steel", collision=False),   # king pin
            cyl(0.03, 0.30, at=(-0.18, -s * 0.30, 0.0), axis="x", mat="steel", collision=False),
        ]))
        steer_joints.append(Joint(f"front_{side}_steer_joint", "revolute", "chassis", kn,
                                  xyz=(X_FRONT, s * Y_FRONT, 0.0), axis=(0, 0, 1),
                                  lower=-0.80, upper=0.80, velocity=0.8, role="steer"))
        name = f"front_{side}_wheel"
        wheel_links.append(wheel_link(name, WHEEL_R, TYRE_W, 105.0))
        wheel_joints.append(wheel_joint(f"{name}_joint", kn, name, (0.0, 0.0, 0.0), WHEEL_R))
        for axle, x in (("mid", X_R1), ("rear", X_R2)):
            for tag, dy in (("inner", -TWIN / 2), ("outer", TWIN / 2)):
                name = f"{axle}_{side}_{tag}_wheel"
                wheel_links.append(wheel_link(name, WHEEL_R, TYRE_W, 105.0))
                wheel_joints.append(wheel_joint(f"{name}_joint", "chassis", name,
                                                (x, s * (Y_DUAL + dy), 0.0), WHEEL_R))

    links = [chassis, cab, drum, swivel, chute, lift_barrel, lift_rod, *knuckles, *wheel_links]
    joints = [
        Joint("cab_mount", "fixed", "chassis", "cab"),
        drum_j, swing_j, lift_j,
        follower("chute_lift_barrel_joint", "revolute", "chute_swivel", "chute_lift_barrel",
                 (CYL_BASE[0], 0.0, CYL_BASE[1]), (0, -1, 0), ka, ca),
        follower("chute_lift_rod_joint", "prismatic", "chute_lift_barrel", "chute_lift_rod",
                 (0.0, 0.0, 0.0), (1, 0, 0), kl, cl),
        *steer_joints,
        *wheel_joints,
    ]

    x1 = CAB[1]
    eye_z = Z(2.55)
    cx, cy, cz = SIDE_CAM
    cr, cp, cyaw = SIDE_CAM_RPY
    return MachineSpec(
        name="concrete_mixer_truck",
        abbrev="CMT",
        display_name="Concrete Mixer Truck",
        category="hauling",
        reference="9 m³ transit mixer on a 3-axle 6x4 cab-over chassis (Volvo FMX / "
                  "Mercedes-Benz Arocs class with a Liebherr HTM 905 class drum)",
        description=("Three-axle 6x4 transit concrete mixer: cab-over chassis with Ackermann "
                     "front steering and dual-tyred rear tandem, 9 m³ drum inclined 12° that "
                     "rotates continuously to charge/mix or discharge, charging hopper, and a "
                     "rear discharge chute that swivels ±89° and lifts/lowers hydraulically."),
        root="chassis",
        links=links,
        joints=joints,
        drive=Drive(type="ackermann", max_speed=11.1, max_reverse_speed=4.0,
                    wheelbase=WHEELBASE, max_steer=0.60, steer_rate=0.30,
                    fixed_axle_x=0.0, steer_axle="front"),
        tools=[
            Tool("drum_joint", "drum", positive="charge", negative="discharge"),
            Tool("chute_swing_joint", "chute_swing", positive="left", negative="right"),
            Tool("chute_lift_joint", "chute_lift", positive="raise", negative="lower"),
        ],
        cameras=[
            # driver's eye (left seat), just ahead of the windscreen
            Camera("front", "chassis", (x1 + 0.06, 0.55, eye_z), (0.0, 0.25, 0.0)),
            # blind-spot cameras behind the upper rear cab corners, looking back along the
            # drum and slightly outward
            Camera("left", "chassis", (cx, cy, Z(cz)), (cr, cp, cyaw)),
            Camera("right", "chassis", (cx, -cy, Z(cz)), (-cr, cp, -cyaw)),
        ],
        envelope=(9.5, 2.5, 3.9),
        operating_mass=13000.0,
        materials={**DEFAULT_MATERIALS,
                   "body": (0.94, 0.94, 0.92, 1.0),      # white cab
                   "drum": (0.78, 0.80, 0.83, 1.0),      # light grey mixer superstructure
                   "hub": (0.80, 0.80, 0.82, 1.0)},      # silver steel rims
        notes=("drum charge/discharge is not auto-verifiable (rotation about its own axis); "
               "the painted spirals follow the left-handed fins, so +1 visibly screws them "
               "toward the cab. chute_swing 'left' = CCW from above, named looking along "
               "the rear-pointing chute (as for the backhoe swing), so 'left' moves the "
               "chute end to the truck's right side. Lift cylinder = cosmetic "
               f"followers (rod eye within {fits['angle_err_m'] + fits['ext_err_m']:.3f} m)."),
    )
