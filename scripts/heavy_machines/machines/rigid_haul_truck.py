"""Rigid-frame off-highway haul truck (RHT), 64 t class — modelled on a CAT 775G
(Komatsu HD605-8 is the same class).

Datasheet / listing numbers used (CAT 775G unless noted):
    nominal payload 70.5 US t (64 t) · target gross machine weight ≈ 106.6 t
    (235,000 lb) → empty operating mass ≈ 42.6 t (model 42.7 t) · Cat C27
    engine, 615 kW (825 hp) gross · top speed loaded 67 km/h (41.6 mph) ·
    tyres 24.00R35 (E4) on all six wheels, rear duals · struck / heaped body
    32.6 / 42.4 m³ (42.7 / 55.5 yd³, dual-slope body) · body raise 10 s,
    float / power-down lower ≈ 14 s · overall size (shipping listing)
    10.16 × 5.66 × 4.45 m (33'4" × 18'7" × 14'7"); Komatsu HD605-8 listing
    10.34 × 5.54 × 4.39 m · wheelbase ≈ 4.2 m (14 ft, 775F listing).
Assumed / typical for the class (not in the sources available here):
    24.00R35 OD 2.34 m, section 0.66 m, dual spacing 0.78 m · front tread
    3.95 m · rear dual centres 3.50 m → width over tyres 4.94 m · body outside
    width 5.22 m, 5.44 m over the top rails · loading height (top rail) 4.04 m ·
    deck 2.55 m, cab roof 4.28 m, canopy top 4.47 m above the ground · rear
    axle → tail 3.2 m, front axle → bumper 2.78 m · ground clearance under the
    rear differential 0.67 m · dump angle 55° about a hinge 0.78 m behind the
    rear tyres, 2.02 m above the ground → body-raised height ≈ 10.0 m, tail
    1.5 m above the ground at full dump · front suspension struts double as
    the steering kingpins (CAT layout), 0.515 m inboard of the wheel centre →
    kingpin track 2.92 m · max virtual steer 0.52 rad (inner wheel 35.6°,
    outer 25.5°), turning circle ≈ 21 m · steering ≈ 4 s lock to lock
    (0.25 rad/s virtual) · reverse ≈ 11 km/h · mass split: chassis (frame,
    engine, deck, tanks, front end) 15.1 t, rear axle group (housing,
    differential, final drives, brakes) 7.5 t, cab 2.0 t, body 9.1 t, six
    wheels 1.25 t each, knuckles 0.7 t, hoist cylinders 0.8 t.
Envelope = (10.2, 5.6, 4.45) m from the two listings; the model is
10.18 × 5.68 × 4.49 m in the travel pose (width over the long right mirror arm).
Drive max_speed = 15 m/s (54 km/h, a typical haul-road limit) rather than the
67 km/h top gear, so the scripted episodes stay inside the site.

Joint ranges / rates: dump body 0 .. 0.96 rad (55°) at 0.096 rad/s (full
raise in 10 s, the datasheet raise time; one rate is used for both
directions) · front steer joints ±0.70 rad about the kingpins (Ackermann,
driven by the env).

Layout: cab on the LEFT of the upper deck, engine hood / radiator in the
centre, air cleaners and exhaust on the right deck, diagonal access stairway
across the radiator grille, canopy (part of the body) over the cab and deck.
Twin two-stage hoist cylinders sit between the frame rails and the inner
rear duals, just ahead of the rear axle.  They are cosmetic followers: the
barrel pivots on its frame pin and the two stages extend, each with a line
fit (chord slope, offset centred on the residuals) of the true cylinder
angle / pin distance, so the rod eye sits 0.09 m from its body pin in the
travel pose and within 0.21 m over the stroke (a follower is linear in the
dump angle; the real cylinder is not).

Mass: the cab and the rear axle group are separate links on fixed joints
(merged on import) so their datasheet-class masses, not the volume of the
solid chassis boxes, set the centre of mass: empty axle split ≈ 55 / 45 %
front / rear (775G ≈ 47 / 53 %), COM 2.0 m above the ground.

Cameras: front = operator eye in the left-hand cab, just ahead of the
windscreen, looking forward/down over the deck; left = above the left mirror
head at the front deck corner, looking outward-forward; right = rear-view
camera above the head of the long right mirror arm, looking back along the
right side so the dump body and right hoist cylinder are in view at every
dump angle.

Init pose = travel pose: body down on the frame, wheels straight.
Root frame: centre of the rear axle (axle centres at z = 0), +x towards the
steered front axle; the ground is at z = -WHEEL_R.
"""

from __future__ import annotations

import math

from ..spec import (Camera, Drive, Joint, Link, MachineSpec, Shape, Tool, beam, box, cyl,
                    sphere, wheel_joint, wheel_link)

# --- Wheels / axles -----------------------------------------------------------------
WHEEL_R, TYRE_W = 1.17, 0.66        # 24.00R35
WHEELBASE = 4.20
X_FRONT, X_REAR = WHEELBASE, 0.0
Y_FRONT = 1.975                     # front tread 3.95 m
KINGPIN = 0.515                     # wheel centre → steering axis (suspension strut), inboard
Y_DUAL, TWIN = 1.75, 0.78           # rear dual centre, inner/outer tyre spacing

# --- Frame / deck (chassis frame, z relative to the axle line) ----------------------
RAIL_Y0, RAIL_Y1 = 0.40, 0.68       # frame rails span |y| in [RAIL_Y0, RAIL_Y1]
DECK_Z0, DECK_Z1 = 1.23, 1.38       # upper deck (2.55 m above the ground)
CAB = (3.45, 5.15, 0.85, 2.30, DECK_Z1, 3.06)   # x0, x1, y0, y1, z0, z1 (left side)
MIRROR_X = 6.43                     # mirror heads (centre x)
MIRROR_Y = (2.58, 2.88)             # head centre |y|: left (short arm), right (long arm)
RAIL_Z = DECK_Z1 + 1.0              # deck handrail top

# --- Dump body (chassis frame) ------------------------------------------------------
HINGE = (-1.95, 0.0, 0.85)          # body pivot at the rear of the frame
FLOOR_Z = 1.27                      # underside of the body floor (main section)
TAIL_X = -3.12                      # floor tail
SIDE_Y = 2.56                       # body side-wall centre (outside width 5.22 m)
TOP_Z = 2.79                        # top of the side walls
WALL_X = 3.05                       # body front wall (rear face)
CANOPY_X, CANOPY_Z = 6.10, 3.16     # canopy front edge, canopy underside
DUMP_MAX = 0.96                     # 55°
DUMP_RATE = DUMP_MAX / 10.0         # 10 s raise

# --- Hoist cylinders (chassis frame, x-z) -------------------------------------------
HOIST_Y = 0.86
HOIST_BASE = (0.60, -0.25)          # frame pin
HOIST_EYE = (1.90, 1.00)            # body pin (body down)


# ------------------------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------------------------

def ext(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, mat: str = "body",
        col: bool = True, off: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Shape:
    """Axis-aligned box from its extents, expressed in a frame whose origin is ``off``."""
    return box(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0),
               at=((x0 + x1) / 2 - off[0], (y0 + y1) / 2 - off[1], (z0 + z1) / 2 - off[2]),
               mat=mat, collision=col)


def span_yz(x: float, p0: tuple[float, float], p1: tuple[float, float], sx: float, sz: float,
            mat: str = "accent", off: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Shape:
    """Visual box from (y, z) p0 to p1 at constant x (stair stringers, rails)."""
    dy, dz = p1[0] - p0[0], p1[1] - p0[1]
    return box(sx, math.hypot(dy, dz), sz,
               at=(x - off[0], (p0[0] + p1[0]) / 2 - off[1], (p0[1] + p1[1]) / 2 - off[2]),
               rpy=(math.atan2(dz, dy), 0.0, 0.0), mat=mat, collision=False)


def rod_x(x0: float, x1: float, r: float, y: float = 0.0, mat: str = "steel") -> Shape:
    """Visual cylinder along the link's x axis from x0 to x1."""
    return cyl(r, x1 - x0, at=((x0 + x1) / 2, y, 0.0), axis="x", mat=mat, collision=False)


def _hoist_fits() -> tuple[tuple[float, float], tuple[float, float], float]:
    """Line fits (k, c) vs the dump angle of the hoist barrel angle and the stage
    extension, plus the pin distance in the travel pose.

    Each fit uses the chord slope and centres the offset between the extreme
    residuals (the minimax line for a convex curve).
    """
    qs = [DUMP_MAX * i / 48 for i in range(49)]
    dx, dz = HOIST_EYE[0] - HINGE[0], HOIST_EYE[1] - HINGE[2]
    vec = [(HINGE[0] + dx * math.cos(q) - dz * math.sin(q) - HOIST_BASE[0],     # body pin rotated
            HINGE[2] + dx * math.sin(q) + dz * math.cos(q) - HOIST_BASE[1])     # as axis (0,-1,0)
           for q in qs]
    d0 = math.hypot(*vec[0])

    def fit(ys: list[float]) -> tuple[float, float]:
        k = (ys[-1] - ys[0]) / DUMP_MAX
        res = [y - k * q for q, y in zip(qs, ys)]
        return k, (max(res) + min(res)) / 2

    return fit([math.atan2(v[1], v[0]) for v in vec]), fit([math.hypot(*v) - d0 for v in vec]), d0


# ------------------------------------------------------------------------------------
# Parts
# ------------------------------------------------------------------------------------

def _cab() -> list[Shape]:
    """Operator cab on the left of the upper deck (chassis frame)."""
    x0, x1, y0, y1, z0, z1 = CAB
    cy = (y0 + y1) / 2
    return [
        ext(x0, x1, y0, y1, z0, z1),                                                   # shell
        ext(x0 - 0.03, x1 + 0.03, y0 - 0.03, y1 + 0.03, z1, z1 + 0.05, "accent"),      # roof
        # glass (visual only), slightly proud of the shell
        ext(x1, x1 + 0.03, y0 + 0.10, y1 - 0.10, z0 + 0.75, z1 - 0.12, "glass", False),  # windscreen
        ext(x0 + 0.15, x1 - 0.12, y1, y1 + 0.03, z0 + 0.80, z1 - 0.14, "glass", False),  # left side
        ext(x0 + 0.20, x1 - 0.15, y0 - 0.03, y0, z0 + 0.85, z1 - 0.14, "glass", False),  # right side
        ext(x0 - 0.03, x0, y0 + 0.15, y1 - 0.15, z0 + 0.95, z1 - 0.16, "glass", False),  # rear
        # pillars / door frame (left door)
        ext(x1 - 0.06, x1 + 0.035, y1 - 0.06, y1 + 0.035, z0, z1, "accent", False),
        ext(x1 - 0.06, x1 + 0.035, y0 - 0.035, y0 + 0.06, z0, z1, "accent", False),
        ext(x0 + 0.70, x0 + 0.76, y1, y1 + 0.035, z0 + 0.05, z1 - 0.05, "accent", False),
        ext(x0 + 0.76, x1 - 0.10, y1, y1 + 0.035, z0 + 0.72, z0 + 0.78, "accent", False),
        # wiper, door handle, roof work lights
        ext(x1 + 0.03, x1 + 0.05, cy - 0.05, cy + 0.05, z0 + 0.80, z0 + 1.30, "accent", False),
        ext(x0 + 0.85, x0 + 0.95, y1 + 0.03, y1 + 0.06, z0 + 0.55, z0 + 0.58, "steel", False),
        ext(x1 - 0.02, x1 + 0.08, y0 + 0.15, y0 + 0.35, z1 - 0.10, z1, "white", False),
        ext(x1 - 0.02, x1 + 0.08, y1 - 0.35, y1 - 0.15, z1 - 0.10, z1, "white", False),
    ]


def _front_end() -> list[Shape]:
    """Radiator grille, bumper, diagonal stairway, lights (chassis frame)."""
    shapes = [
        ext(6.30, 6.50, -1.05, 1.05, -0.05, 1.95, "dark"),                             # grille
        *[ext(6.50, 6.53, -1.0, 1.0, z, z + 0.06, "accent", False)                     # grille bars
          for z in (0.05, 0.33, 0.61, 0.89, 1.17, 1.45, 1.73)],
        ext(6.50, 6.95, -1.45, 1.45, -0.35, 0.25),                                     # bumper
        ext(6.95, 6.98, -1.35, -1.05, -0.02, 0.12, "white", False),                    # headlights
        ext(6.95, 6.98, 1.05, 1.35, -0.02, 0.12, "white", False),
        ext(6.55, 6.58, 1.80, 2.10, DECK_Z0, DECK_Z1, "white", False),                 # deck lights
        ext(6.55, 6.58, -2.10, -1.80, DECK_Z0, DECK_Z1, "white", False),
    ]
    # diagonal stairway across the grille: bumper (right) → left deck, next to the cab
    lo, hi = (-1.30, 0.25), (0.78, DECK_Z1)
    for x in (6.63, 6.91):
        shapes.append(span_yz(x, lo, hi, 0.04, 0.16))                                  # stringers
    for i in range(1, 8):
        t = i / 8
        y, z = lo[0] + t * (hi[0] - lo[0]), lo[1] + t * (hi[1] - lo[1])
        shapes.append(ext(6.64, 6.90, y - 0.13, y + 0.13, z - 0.02, z + 0.02, "steel", False))
    shapes.append(span_yz(6.93, (lo[0], lo[1] + 1.0), (hi[0], hi[1] + 1.0), 0.04, 0.04))  # handrail
    for y, z in ((lo[0], lo[1]), (hi[0], hi[1])):
        shapes.append(ext(6.91, 6.95, y - 0.02, y + 0.02, z, z + 1.0, "accent", False))
    return shapes


def _deck() -> list[Shape]:
    """Upper deck left and right of the hood, hood, air cleaners, handrails, mirrors."""
    shapes = [
        ext(3.25, 6.55, 0.78, 2.35, DECK_Z0, DECK_Z1),                                 # left deck
        ext(3.25, 6.55, -2.35, -0.78, DECK_Z0, DECK_Z1),                               # right deck
        ext(3.30, 6.30, -0.78, 0.78, DECK_Z1, 1.95),                                   # engine hood
        ext(4.40, 6.10, -0.55, 0.55, 1.95, 1.97, "accent", False),                     # hood grille
        # air cleaners + exhaust on the right deck
        cyl(0.22, 0.95, at=(5.45, -1.30, DECK_Z1 + 0.26), axis="x", mat="dark", collision=False),
        cyl(0.22, 0.95, at=(5.45, -1.85, DECK_Z1 + 0.26), axis="x", mat="dark", collision=False),
        cyl(0.16, 0.55, at=(3.75, -1.55, DECK_Z1 + 0.30), mat="steel", collision=False),  # muffler
        cyl(0.08, 0.80, at=(3.75, -1.55, DECK_Z1 + 0.95), mat="accent", collision=False),  # stack
        # handrails: left front + side (in front of the cab), right front + side
        ext(6.48, 6.52, 0.80, 2.32, RAIL_Z - 0.04, RAIL_Z, "accent", False),
        ext(5.20, 6.52, 2.28, 2.32, RAIL_Z - 0.04, RAIL_Z, "accent", False),
        ext(6.48, 6.52, -2.32, -0.80, RAIL_Z - 0.04, RAIL_Z, "accent", False),
        ext(3.30, 6.52, -2.32, -2.28, RAIL_Z - 0.04, RAIL_Z, "accent", False),
    ]
    # deck skirts ahead of the front tyres only: the steered tyre sweeps out to |y| 2.47
    # under the deck edge between x 3.65 and 4.75 (5.05 for its bounding box)
    for s in (1.0, -1.0):
        shapes.append(ext(5.40, 6.55, s * 2.31, s * 2.35, DECK_Z0 - 0.20, DECK_Z0, "body", False))
    for x, y in ((6.50, 1.0), (6.50, 2.30), (5.85, 2.30), (5.22, 2.30),
                 (6.50, -1.0), (6.50, -2.30), (5.40, -2.30), (4.35, -2.30), (3.32, -2.30)):
        shapes.append(ext(x - 0.02, x + 0.02, y - 0.02, y + 0.02, DECK_Z1, RAIL_Z, "accent", False))
    # rear-view mirrors on arms off the front deck corners (the right arm is long and
    # carries the rear-view camera)
    for s, ym in ((1.0, MIRROR_Y[0]), (-1.0, MIRROR_Y[1])):
        shapes.append(ext(MIRROR_X + 0.01, MIRROR_X + 0.05, s * 2.30, s * ym, RAIL_Z + 0.10,
                          RAIL_Z + 0.14, "accent", False))
        shapes.append(ext(MIRROR_X - 0.03, MIRROR_X + 0.03, s * (ym - 0.08), s * (ym + 0.08),
                          RAIL_Z - 0.05, RAIL_Z + 0.40, "accent", False))
    return shapes


def _chassis_frame() -> list[Shape]:
    """Frame rails, engine, tanks, suspension mounts (chassis frame)."""
    shapes = [
        ext(3.40, 6.25, -0.62, 0.62, -0.20, DECK_Z0, "dark"),                          # engine
        ext(-2.05, -1.85, -RAIL_Y1, RAIL_Y1, 0.40, 0.90, "dark"),                      # rear cross member
        ext(2.40, 2.60, -RAIL_Y1, RAIL_Y1, -0.20, 0.40, "dark", False),                # mid cross member
        beam((0.30, -0.05), (2.50, -0.12), 0.45, 0.22, mat="dark", collision=False),  # A-frame
        cyl(0.11, 1.80, at=(HINGE[0], 0.0, HINGE[2]), axis="y", mat="steel", collision=False),
        # fuel tank (left) and hydraulic tank (right) between the axles, ending 0.06 m
        # behind the front tyre's swept path at full lock
        ext(1.45, 2.70, 1.05, 1.75, -0.20, 0.70),
        ext(1.45, 2.70, -1.75, -1.05, -0.20, 0.70),
        cyl(0.09, 0.08, at=(2.30, 1.45, 0.74), mat="accent", collision=False),        # filler caps
        cyl(0.09, 0.08, at=(2.30, -1.45, 0.74), mat="accent", collision=False),
    ]
    for s in (1.0, -1.0):
        yc = s * (RAIL_Y0 + RAIL_Y1) / 2
        w = RAIL_Y1 - RAIL_Y0
        shapes += [
            ext(-2.05, 0.45, s * RAIL_Y0, s * RAIL_Y1, 0.25, 1.00),                    # rear rail
            beam((0.35, 0.625), (2.30, 0.10), w, 0.72, y=yc),                           # kick-down
            ext(2.20, 6.55, s * RAIL_Y0, s * RAIL_Y1, -0.26, 0.46),                    # front rail
            # tank straps, front strut (kingpin) mounts — inside 0.18 m of the kingpin axis,
            # so the tyre (inner face 0.185 m outboard of it) clears them at any steer
            # angle — hoist pin lugs, tail lights
            ext(1.80, 1.86, s * 1.04, s * 1.76, -0.21, 0.71, "accent", False),
            ext(2.40, 2.46, s * 1.04, s * 1.76, -0.21, 0.71, "accent", False),
            ext(X_FRONT - 0.17, X_FRONT + 0.17, s * RAIL_Y1, s * (Y_FRONT - KINGPIN + 0.06),
                0.90, 1.20, "dark", False),
            ext(HOIST_BASE[0] - 0.20, HOIST_BASE[0] + 0.20, s * RAIL_Y1, s * 0.74,
                HOIST_BASE[1] - 0.14, 0.30, "dark", False),
            ext(-2.08, -2.05, s * 0.44, s * 0.64, 0.72, 0.88, "red", False),
        ]
    return shapes


def _body_shapes() -> list[Shape]:
    """Dual-slope dump body with canopy; coordinates given in the chassis frame."""
    o = HINGE
    hx, hz = HINGE[0], HINGE[2]
    t_out, t_in = SIDE_Y + 0.05, SIDE_Y - 0.05
    cw = SIDE_Y + 0.10                                        # canopy half-width
    slope_z = FLOOR_Z + 0.05 + (-0.70 - TAIL_X) * 0.19        # rear-floor centre at the tail
    shapes = [
        ext(-0.70, WALL_X, -t_out, t_out, FLOOR_Z, FLOOR_Z + 0.10, off=o),             # floor
        beam((-0.70 - hx, FLOOR_Z + 0.05 - hz), (TAIL_X - hx, slope_z - hz),           # rear floor
             2 * t_out, 0.10),                                                          # (rises 11°)
        ext(WALL_X, WALL_X + 0.15, -t_out, t_out, FLOOR_Z, CANOPY_Z, off=o),           # front wall
        ext(WALL_X, CANOPY_X, -cw, cw, CANOPY_Z, CANOPY_Z + 0.14, off=o),              # canopy
        # canopy edge beams, front lip, gussets (visual)
        ext(CANOPY_X - 0.10, CANOPY_X + 0.02, -cw, cw, CANOPY_Z - 0.10, CANOPY_Z + 0.16,
            "body", False, o),
        *[ext(WALL_X + 0.15, CANOPY_X, s * (cw - 0.08), s * cw, CANOPY_Z - 0.10, CANOPY_Z,
              "body", False, o) for s in (1.0, -1.0)],
        *[beam((WALL_X + 0.15 - hx, TOP_Z - 0.15 - hz), (WALL_X + 1.05 - hx, CANOPY_Z - hz),
               0.08, 0.14, y=y, collision=False) for y in (-2.45, -0.40, 2.45)],
        ext(WALL_X + 0.15, WALL_X + 0.20, -2.40, 2.40, TOP_Z - 0.10, TOP_Z, "body", False, o),
    ]
    for s in (1.0, -1.0):
        y0, y1 = s * t_in, s * t_out
        shapes += [
            # side walls: stepped bottom edge follows the rising rear floor
            ext(-0.70, WALL_X, y0, y1, FLOOR_Z, TOP_Z, off=o),
            ext(-1.90, -0.70, y0, y1, FLOOR_Z + 0.15, TOP_Z, off=o),
            ext(TAIL_X + 0.05, -1.90, y0, y1, FLOOR_Z + 0.36, FLOOR_Z + 1.18, off=o),
            beam((-1.85 - hx, TOP_Z - 0.17 - hz), (TAIL_X - hx, 2.30 - hz), 0.10, 0.36, y=s * SIDE_Y),
            # top rail (main + sloped tail), lower side rail
            ext(-1.85, WALL_X + 0.15, s * (SIDE_Y - 0.02), s * (SIDE_Y + 0.16), TOP_Z - 0.08, TOP_Z + 0.08,
                off=o),
            beam((-1.85 - hx, TOP_Z - hz), (TAIL_X - 0.03 - hx, 2.42 - hz), 0.18, 0.16,
                 y=s * (SIDE_Y + 0.07)),
            ext(-1.90, WALL_X, s * t_out, s * (SIDE_Y + 0.13), FLOOR_Z, FLOOR_Z + 0.15, "body", False, o),
            # longitudinal sills on the frame rails, hinge brackets, hoist eye brackets
            ext(-0.70, 3.00, s * RAIL_Y0, s * RAIL_Y1, 1.02, FLOOR_Z, off=o),
            beam((-0.70 - hx, 1.20 - hz), (-2.05 - hx, 1.45 - hz), RAIL_Y1 - RAIL_Y0, 0.36,
                 y=s * (RAIL_Y0 + RAIL_Y1) / 2),
            ext(-2.15, -1.75, s * 0.72, s * 0.92, 0.72, 1.40, "dark", off=o),
            ext(HOIST_EYE[0] - 0.24, HOIST_EYE[0] + 0.24, s * 0.74, s * 0.98,
                HOIST_EYE[1] - 0.14, FLOOR_Z, "dark", False, o),
        ]
        # outside ribs (visual)
        for x in (-1.30, -0.10, 1.10, 2.30):
            z0 = FLOOR_Z + (0.15 if x < -0.70 else 0.0)
            shapes.append(ext(x - 0.07, x + 0.07, s * t_out, s * (SIDE_Y + 0.12), z0, TOP_Z - 0.08,
                              "body", False, o))
        shapes.append(ext(-2.57, -2.43, s * t_out, s * (SIDE_Y + 0.12), FLOOR_Z + 0.40, 2.45,
                          "body", False, o))
    # floor cross members ahead of the rear tyres (visual), tail lip
    for x in (1.55, 2.55):
        shapes.append(ext(x - 0.08, x + 0.08, -2.40, 2.40, FLOOR_Z - 0.14, FLOOR_Z, "body", False, o))
    shapes.append(ext(TAIL_X - 0.08, TAIL_X + 0.04, -t_out, t_out, slope_z - 0.10, slope_z + 0.12,
                      "body", False, o))
    shapes.append(cyl(0.13, 0.30, at=(0.0, 0.82, 0.0), axis="y", mat="steel", collision=False))
    shapes.append(cyl(0.13, 0.30, at=(0.0, -0.82, 0.0), axis="y", mat="steel", collision=False))
    return shapes


# ------------------------------------------------------------------------------------
# Machine
# ------------------------------------------------------------------------------------

def build() -> MachineSpec:
    chassis = Link("chassis", 15100.0, [*_chassis_frame(), *_deck(), *_front_end()])
    # Sub-assemblies on fixed joints (merged into the chassis on import) so their own
    # masses, not the volume of the solid cab / engine boxes, place the centre of mass.
    cab = Link("cab", 2000.0, _cab())
    rear_axle = Link("rear_axle", 7500.0, [
        cyl(0.32, 2.05, axis="y", mat="dark"),                                         # housing
        sphere(0.50, mat="dark"),                                                      # differential
        cyl(0.42, 0.10, at=(0.0, 0.98, 0.0), axis="y", mat="dark", collision=False),   # final drives
        cyl(0.42, 0.10, at=(0.0, -0.98, 0.0), axis="y", mat="dark", collision=False),
    ])
    body = Link("dump_body", 9100.0, _body_shapes())

    # ---------------------------------------------------------------- hoist cylinders
    (ka, ca), (kl, cl), d0 = _hoist_fits()
    barrel_len = d0 - 0.22
    hoist_barrels = Link("hoist_barrels", 450.0, [
        *[s for y in (-HOIST_Y, HOIST_Y) for s in (
            rod_x(0.0, barrel_len, 0.16, y, "dark"),
            rod_x(barrel_len - 0.08, barrel_len, 0.18, y, "accent"),                    # gland
            cyl(0.12, 0.30, at=(0.0, y, 0.0), axis="y", mat="steel", collision=False))],  # base eye
    ])
    hoist_stages = Link("hoist_stages", 200.0, [
        *[s for y in (-HOIST_Y, HOIST_Y) for s in (
            rod_x(0.15, d0 - 0.12, 0.125, y, "steel"),
            rod_x(d0 - 0.18, d0 - 0.12, 0.14, y, "accent"))],
    ])
    hoist_rods = Link("hoist_rods", 150.0, [
        *[s for y in (-HOIST_Y, HOIST_Y) for s in (
            rod_x(0.20, d0, 0.09, y, "steel"),
            cyl(0.11, 0.24, at=(d0, y, 0.0), axis="y", mat="steel", collision=False))],   # rod eye
    ])

    dump_j = Joint("dump_body_joint", "revolute", "chassis", "dump_body", xyz=HINGE,
                   axis=(0, -1, 0), lower=0.0, upper=DUMP_MAX, init=0.0,
                   velocity=DUMP_RATE, role="tool")

    def follower(name, jtype, parent, child, xyz, axis, k, c) -> Joint:
        ends = (k * dump_j.lower + c, k * dump_j.upper + c)
        return Joint(name, jtype, parent, child, xyz=xyz, axis=axis,
                     lower=min(ends) - 0.02, upper=max(ends) + 0.02, init=k * dump_j.init + c,
                     velocity=abs(k) * dump_j.velocity + 0.05, role="follower",
                     follows=(dump_j.name, k, c))

    # ------------------------------------------------------------------- wheels
    # Knuckle frame = steering axis: the front suspension strut is the kingpin, KINGPIN
    # inboard of the wheel centre, so the strut turns in place in its frame mount.
    wheel_links, wheel_joints, knuckles, steer_joints = [], [], [], []
    for side, s in (("left", 1.0), ("right", -1.0)):
        kn = f"front_{side}_knuckle"
        knuckles.append(Link(kn, 350.0, [
            cyl(0.26, 0.20, at=(0.0, s * (KINGPIN - 0.46), 0.0), axis="y", mat="dark"),  # spindle housing
            cyl(0.13, 0.95, at=(0.0, 0.0, 0.62), mat="steel", collision=False),            # suspension strut
            cyl(0.17, 0.30, at=(0.0, 0.0, 0.95), mat="dark", collision=False),
        ]))
        steer_joints.append(Joint(f"front_{side}_steer_joint", "revolute", "chassis", kn,
                                  xyz=(X_FRONT, s * (Y_FRONT - KINGPIN), 0.0), axis=(0, 0, 1),
                                  lower=-0.70, upper=0.70, velocity=0.8, role="steer"))
        name = f"front_{side}_wheel"
        wheel_links.append(wheel_link(name, WHEEL_R, TYRE_W, 1250.0))
        wheel_joints.append(wheel_joint(f"{name}_joint", kn, name, (0.0, s * KINGPIN, 0.0), WHEEL_R))
        for tag, dy in (("inner", -TWIN / 2), ("outer", TWIN / 2)):
            name = f"rear_{side}_{tag}_wheel"
            wheel_links.append(wheel_link(name, WHEEL_R, TYRE_W, 1250.0))
            wheel_joints.append(wheel_joint(f"{name}_joint", "chassis", name,
                                            (X_REAR, s * (Y_DUAL + dy), 0.0), WHEEL_R))

    links = [chassis, cab, rear_axle, body, hoist_barrels, hoist_stages, hoist_rods,
             *knuckles, *wheel_links]
    joints = [
        Joint("cab_mount", "fixed", "chassis", "cab"),
        Joint("rear_axle_mount", "fixed", "chassis", "rear_axle"),
        dump_j,
        follower("hoist_barrel_joint", "revolute", "chassis", "hoist_barrels",
                 (HOIST_BASE[0], 0.0, HOIST_BASE[1]), (0, -1, 0), ka, ca),
        follower("hoist_stage_joint", "prismatic", "hoist_barrels", "hoist_stages",
                 (0.0, 0.0, 0.0), (1, 0, 0), kl / 2, cl / 2),
        follower("hoist_rod_joint", "prismatic", "hoist_stages", "hoist_rods",
                 (0.0, 0.0, 0.0), (1, 0, 0), kl / 2, cl / 2),
        *steer_joints,
        *wheel_joints,
    ]

    return MachineSpec(
        name="rigid_haul_truck",
        abbrev="RHT",
        display_name="Rigid Haul Truck",
        category="hauling",
        reference="64 t rigid-frame off-highway truck (CAT 775G / Komatsu HD605 class)",
        description=("Two-axle rigid-frame mining/quarry haul truck: 24.00R35 tyres, single "
                     "front and dual rear, Ackermann front steering, cab on the left of the "
                     "upper deck, dual-slope dump body with a canopy over the cab, tipped "
                     "55° about a rear hinge by twin two-stage hoist cylinders."),
        root="chassis",
        links=links,
        joints=joints,
        drive=Drive(type="ackermann", max_speed=15.0, max_reverse_speed=3.0,
                    wheelbase=WHEELBASE, max_steer=0.52, steer_rate=0.25,
                    fixed_axle_x=X_REAR, steer_axle="front"),
        tools=[Tool("dump_body_joint", "dump_body", positive="raise", negative="lower")],
        cameras=[
            # operator eye height, just ahead of the windscreen of the left-hand cab
            Camera("front", "chassis", (CAB[1] + 0.07, 1.55, 2.72), (0.0, 0.28, -0.04)),
            # above the left mirror head (clear of it), looking outward-forward
            Camera("left", "chassis", (MIRROR_X, MIRROR_Y[0], RAIL_Z + 0.67), (0.0, 0.35, 0.70)),
            # rear-view camera above the long right mirror arm: dump body + right hoist
            Camera("right", "chassis", (MIRROR_X, -MIRROR_Y[1], RAIL_Z + 0.62), (0.0, 0.25, -2.90)),
        ],
        envelope=(10.2, 5.6, 4.45),
        operating_mass=42700.0,
        notes=("Hoist cylinders are cosmetic followers of dump_body_joint (line fits; the rod "
               "eye is 0.09 m from its body pin in the travel pose, < 0.21 m over the stroke). "
               "Inertials are volume-weighted per link; the cab and rear axle group are "
               "separate fixed links so the empty axle split is ≈ 55/45 front/rear "
               "(775G ≈ 47/53). No payload is simulated."),
    )
