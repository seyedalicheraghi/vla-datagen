"""Articulated dump truck (ADT), 28 t payload class — modelled on a CAT 730 / Volvo A30G.

Reference numbers (CAT 730 / Volvo A30G class datasheets, rounded):
    operating mass (empty) 23.5 t · rated payload 28 t · heaped body 17.5 m³ ·
    6x6, three axles · 23.5R25 tyres (OD 1.61 m, section 0.60 m) · gauge
    2.32 m · width over tyres 2.92 m (body 2.93 m) · overall length 10.4 m
    (A30G 10.3 m, 730 10.5 m) · height over cab ROPS / beacon 3.43 m ·
    loading height (body top rail) 3.0 m · front axle to bumper 1.50 m ·
    front axle to hitch 2.15 m · hitch to first bogie axle 3.00 m · bogie
    axle spacing 1.67 m · front axle to tandem centre 5.99 m · ground
    clearance 0.50 m (axle differentials) · articulation ±45° (outer turning
    radius ≈ 9.0 m in the model) · tipping angle 70° · tipping time 12 s ·
    height with the body raised ≈ 6.5 m (model 6.73 m, set by the canopy lip) ·
    top speed 53 km/h (14.7 m/s) · reverse ≈ 15 km/h · lock-to-lock
    steering ≈ 4.5 s · empty weight ≈ 55-58 % on the front axle (model 53.5 %:
    link inertials spread mass by shape volume, so the large cab box pulls the
    tractor's centre of mass rearward).

Joint ranges / rates used: articulation ±0.785 rad at 0.35 rad/s; dump body
0..1.22 rad (70°) at 0.10 rad/s (full tip in ≈ 12 s).

Init pose = travel pose: frames straight, body down.

Layout: the root link is the FRONT tractor frame (engine hood, centre-mounted
cab, front axle).  The articulation joint (vertical hinge behind the cab)
carries the REAR frame (two-axle tandem bogie, hoist cylinders, body pivot)
and the dump body tips about a pin at the rear end of the rear frame, behind
the last axle; the frame rails stop just ahead of the pin so the body's sills
and tail never swing into them.  The two single-stage hoist cylinders (base
just behind the hitch, eye under the body) are cosmetic follower links, see
``_hoist_fit``.  At full articulation the short canopy lip clears the cab's
rear corners by ≈ 5 cm.
Livery: CAT and Volvo ADTs are both yellow, so DEFAULT_MATERIALS are used.

Root frame: front axle centre (axle line, z = 0 is 0.805 m above the ground),
+x forward.
"""

from __future__ import annotations

import math

from ..spec import (Camera, Drive, Joint, Link, MachineSpec, Tool, beam, box, cyl, sphere,
                    wheel_joint, wheel_link)

# --- Running gear ----------------------------------------------------------------
WHEEL_R, TYRE_W = 0.805, 0.60        # 23.5R25
GAUGE = 2.32                         # tyre centre to tyre centre
HITCH_X = -2.15                      # articulation hinge behind the front axle
BOGIE_X = (-3.00, -4.67)             # rear-frame axles, relative to the hinge
TANDEM_X = sum(BOGIE_X) / 2          # -3.835
BUMPER_X = 1.50                      # front of the bumper / grille
CAB = (-1.95, -0.30, 0.90, 1.15, 2.45)   # cab shell: rear x, front x, half width, floor z, roof z
MIRROR = (-0.17, 1.16, 1.87)         # left mirror head centre (right one mirrored)
MIRROR_ARM_Z = MIRROR[2] + 0.32      # mirror-arm end = left-camera mount; the head hangs on a
                                     # short drop below it, out of the camera frustum
# Right camera = rear-view / body-watch camera on the cab roof's rear right corner (3.39 m AGL,
# 2 cm ahead of the roof's rear edge, above the canopy lip), looking back along the body.
REAR_CAM = (CAB[0] - 0.05, -(CAB[2] - 0.04), CAB[4] + 0.13)

# --- Dump body ---------------------------------------------------------------------
PIVOT = (-5.55, 0.745)               # body pivot pin (x rel. hinge, z rel. axle line; 1.55 m AGL)
BODY_FRONT = 4.85                    # headboard, ahead of the pivot (body frame)
BODY_TAIL = -1.20                    # tail end of the body behind the pivot
DUMP_MAX = 1.22                      # 70 deg tipping angle

# --- Hoist cylinders (cosmetic followers) -------------------------------------------
HOIST_BASE = (-0.45, 0.145)          # on the rear frame (x rel. hinge, z), just behind the hitch
HOIST_EYE = (2.20, 0.0)              # on the body underside (body frame)
HOIST_Y = 0.68
BARREL_LEN = 2.55                    # rod length follows from the fit (see build)


def _hoist_fit() -> dict:
    """Linear follower fits (vs. dump angle) for one hoist cylinder.

    The barrel pivots on the rear frame (angle ``k * q + c``) and the rod slides
    out of it (``ks * q``), so the cylinder is always straight and exactly
    pinned at its base.  Planar follower chains can only give link angles that
    are linear in q while the true pin-to-pin angle is not (±4° over a 70° tip),
    so the rod eye wanders a little: the fit keeps it on the body underside
    (small body-z error) and lets it slide along a longitudinal hoist bracket
    (body-x error), whose extent is returned in ``eye_x`` / ``eye_z``.

    Derived from the geometry constants above (≈ 9 ms), so it stays correct if
    PIVOT / HOIST_* / DUMP_MAX are edited.
    """
    def _rot(p, q):          # x-z point turned by a (0, -1, 0) joint at angle q (+q tips +x up)
        c, s = math.cos(q), math.sin(q)
        return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)

    def _linear_fit(xs, ys):  # least-squares slope, minimax offset
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        err = [y - k * x for x, y in zip(xs, ys)]
        return k, (max(err) + min(err)) / 2

    qs = [DUMP_MAX * i / 40 for i in range(41)]
    ang, dist = [], []
    for q in qs:
        r = _rot(HOIST_EYE, q)
        dx, dz = PIVOT[0] + r[0] - HOIST_BASE[0], PIVOT[1] + r[1] - HOIST_BASE[1]
        ang.append(math.atan2(dz, -dx))                        # barrel frame is yawed 180 deg
        dist.append(math.hypot(dx, dz))

    def eye_miss(p):
        k, c, ks, cs = p
        xs, zs = [], []
        for q in qs:
            a, s = k * q + c, ks * q + cs
            eye = (HOIST_BASE[0] - s * math.cos(a), HOIST_BASE[1] + s * math.sin(a))
            b = _rot((eye[0] - PIVOT[0], eye[1] - PIVOT[1]), -q)     # into the body frame
            xs.append(b[0] - HOIST_EYE[0])
            zs.append(b[1] - HOIST_EYE[1])
        return xs, zs

    def cost(p):
        xs, zs = eye_miss(p)
        return max(6.0 * max(abs(z) for z in zs), max(xs) - min(xs))

    p = [*_linear_fit(qs, ang), *_linear_fit(qs, dist)]
    steps = [0.04, 0.04, 0.16, 0.16]
    best = cost(p)
    for _ in range(12):                                         # coordinate descent, halving steps
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for sgn in (1.0, -1.0):
                    trial = list(p)
                    trial[i] += sgn * steps[i]
                    c_t = cost(trial)
                    if c_t < best - 1e-9:
                        p, best, improved = trial, c_t, True
        steps = [s / 2 for s in steps]
    xs, zs = eye_miss(p)
    return {"k": p[0], "c": p[1], "ks": p[2], "cs": p[3],
            "eye_x": (min(xs), max(xs)), "eye_z": (min(zs), max(zs))}


def build() -> MachineSpec:
    y_w = GAUGE / 2

    # ------------------------------------------------------------- cab (front frame)
    x0, x1, y1, z0, z1 = CAB
    cx, cz, lx, lz = (x0 + x1) / 2, (z0 + z1) / 2, x1 - x0, z1 - z0
    arm = (MIRROR[0] - (x1 - 0.02), MIRROR[1] - 0.04 - (y1 + 0.03))    # pillar -> arm end
    cab = [
        box(lx, 2 * y1, lz, at=(cx, 0.0, cz)),                                          # cab shell
        box(lx + 0.10, 2 * y1 + 0.08, 0.08, at=(cx - 0.02, 0.0, z1 + 0.04)),            # roof
        cyl(0.07, 0.09, at=(x0 + 0.30, 0.0, z1 + 0.125), mat="orange", collision=False),  # beacon
        # glass (visual only), slightly proud of the shell
        box(0.03, 2 * y1 - 0.20, 0.86, at=(x1 + 0.01, 0.0, z0 + 0.82), mat="glass", collision=False),
        *[box(lx - 0.30, 0.03, 0.80, at=(cx + 0.05, s * (y1 + 0.01), z0 + 0.83), mat="glass",
              collision=False) for s in (1.0, -1.0)],
        box(0.03, 2 * y1 - 0.40, 0.55, at=(x0 - 0.01, 0.0, z0 + 0.95), mat="glass", collision=False),
        # ROPS pillars / door frame
        *[box(0.07, 0.04, lz - 0.08, at=(x, s * (y1 + 0.02), cz), mat="accent", collision=False)
          for x in (x1 - 0.04, x0 + 0.04, cx - 0.10) for s in (1.0, -1.0)],
        # roof work lights; mirror arms from the front pillars, mirror heads hanging below
        # the arm ends on short drops (the left camera sits on the left arm end)
        *[box(0.08, 0.18, 0.09, at=(x1 + 0.03, s * 0.55, z1 - 0.05), mat="white", collision=False)
          for s in (1.0, -1.0)],
        *[box(math.hypot(*arm), 0.03, 0.03, at=(x1 - 0.02 + arm[0] / 2, s * (y1 + 0.03 + arm[1] / 2),
              MIRROR_ARM_Z), rpy=(0.0, 0.0, s * math.atan2(arm[1], arm[0])), mat="accent",
              collision=False) for s in (1.0, -1.0)],
        *[box(0.03, 0.03, MIRROR_ARM_Z - MIRROR[2] - 0.15, at=(MIRROR[0], s * (MIRROR[1] - 0.04),
              (MIRROR_ARM_Z + MIRROR[2] + 0.15) / 2), mat="accent", collision=False)
          for s in (1.0, -1.0)],
        *[box(0.05, 0.16, 0.32, at=(MIRROR[0], s * MIRROR[1], MIRROR[2]), mat="accent",
              collision=False) for s in (1.0, -1.0)],
    ]

    # ------------------------------------------------------------- front frame
    # Masses: real ADTs carry ≈ 55-58 % of the empty weight on the front axle (engine,
    # transmission and cab are all on the tractor), hence a heavy front frame.
    front = Link("front_frame", 11600.0, [
        box(3.40, 1.00, 0.55, at=(-0.55, 0.0, 0.125), mat="dark"),                    # frame rails
        cyl(0.20, 1.70, axis="y", mat="dark"),                                        # front axle
        sphere(0.30, mat="dark"),                                                     # differential
        # engine hood: lower box + sloping top, grille, bumper
        box(1.70, 1.56, 0.75, at=(0.55, 0.0, 0.775)),
        beam((-0.30, 1.26), (1.38, 1.07), 1.48, 0.26),
        box(0.04, 1.10, 0.55, at=(1.42, 0.0, 0.82), mat="accent", collision=False),   # grille
        *[box(0.04, 1.10, 0.04, at=(1.445, 0.0, z), mat="dark", collision=False)
          for z in (0.65, 0.78, 0.91)],
        box(0.20, 2.00, 0.30, at=(BUMPER_X - 0.10, 0.0, 0.25), mat="dark"),            # bumper
        *[box(0.05, 0.22, 0.12, at=(1.43, s * 0.60, 1.06), mat="white", collision=False)
          for s in (1.0, -1.0)],                                                      # headlights
        # front fenders over the tyres
        *[s for y in (y_w, -y_w) for s in (
            beam((-0.95, 0.60), (-0.70, 0.97), 0.62, 0.08, y=y),
            beam((-0.70, 0.97), (0.70, 0.97), 0.62, 0.08, y=y),
            beam((0.70, 0.97), (1.15, 0.62), 0.62, 0.08, y=y),
            box(0.05, 0.16, 0.10, at=(1.10, y, 0.72), mat="orange", collision=False))],  # indicators
        # cab base (transmission / step well), fuel tank, steps
        box(1.65, 1.60, 0.75, at=(-1.125, 0.0, 0.775)),
        box(0.85, 0.45, 0.50, at=(-1.42, -1.02, 0.45)),                              # fuel tank
        *[box(0.40, 0.08, 0.04, at=(-1.20, 0.84, z), mat="steel", collision=False)
          for z in (0.25, 0.55, 0.85)],                                               # steps
        box(0.04, 0.04, 0.95, at=(-1.42, 0.86, 1.30), mat="accent", collision=False),  # handrail
        # hood details: air pre-cleaner, exhaust stack (right)
        cyl(0.08, 0.28, at=(0.05, 0.66, 1.42), mat="accent", collision=False),
        cyl(0.075, 0.85, at=(-0.12, -0.60, 1.62), mat="accent", collision=False),
        # hitch housing behind the cab + king pin
        box(0.45, 0.80, 0.60, at=(-2.125, 0.0, 0.25), mat="dark"),
        cyl(0.11, 0.85, at=(HITCH_X, 0.0, 0.25), mat="steel", collision=False),
        *cab,
    ])

    # -------------------------------------------------------------- rear frame
    # Origin = articulation hinge; x relative to the hinge, z relative to the axle line.
    # The rails and crossmembers stop ahead of the body pivot and the pin sits on a
    # bracket between the body lugs, so nothing the body swings down while tipping
    # (sill ends, tail kick-up) can reach the frame.
    rail_x0, rail_x1 = -0.80, PIVOT[0] + 0.10
    rear = Link("rear_frame", 3600.0, [
        box(1.05, 0.70, 0.55, at=(-0.575, 0.0, 0.325)),                             # hitch neck
        *[box(rail_x0 - rail_x1, 0.24, 0.38, at=((rail_x0 + rail_x1) / 2, s * 0.42, 0.49))
          for s in (1.0, -1.0)],                                                    # rails
        *[box(0.25, 1.08, 0.30, at=(x, 0.0, 0.49), mat="dark") for x in (-1.95, PIVOT[0] + 0.25)],
        # tandem bogie: axle housings, differentials, bogie beams and pivot brackets
        *[cyl(0.19, 1.66, at=(x, 0.0, 0.0), axis="y", mat="dark") for x in BOGIE_X],
        *[sphere(0.28, at=(x, 0.0, 0.0), mat="dark") for x in BOGIE_X],
        *[box(BOGIE_X[0] - BOGIE_X[1] + 0.56, 0.18, 0.36, at=(TANDEM_X, s * 0.68, 0.0), mat="dark")
          for s in (1.0, -1.0)],
        *[box(0.40, 0.30, 0.32, at=(TANDEM_X, s * 0.56, 0.24), mat="dark") for s in (1.0, -1.0)],
        # hoist-cylinder base brackets; body pivot bracket (between the body lugs) + boss;
        # tail lights on the bracket's rear face, below the pin
        *[box(0.30, 0.20, 0.30, at=(HOIST_BASE[0], s * (HOIST_Y - 0.05), HOIST_BASE[1]), mat="dark")
          for s in (1.0, -1.0)],
        box(0.40, 0.56, 0.42, at=(PIVOT[0] + 0.08, 0.0, PIVOT[1] - 0.21), mat="dark"),
        cyl(0.13, 1.10, at=(PIVOT[0], 0.0, PIVOT[1]), axis="y", mat="steel"),
        *[box(0.04, 0.16, 0.10, at=(PIVOT[0] - 0.14, s * 0.18, PIVOT[1] - 0.30), mat="red",
              collision=False) for s in (1.0, -1.0)],
    ])

    # --------------------------------------------------------------- dump body
    # Origin = body pivot pin; +x forward towards the headboard at q = 0.
    hf = _hoist_fit()
    ex0, ex1 = (HOIST_EYE[0] + v for v in hf["eye_x"])
    ez0 = HOIST_EYE[1] + hf["eye_z"][0]
    xm = (BODY_FRONT + BODY_TAIL) / 2
    body = Link("dump_body", 4300.0, [
        *[box(0.30, 0.12, 0.24, at=(0.0, s * 0.42, 0.04), mat="dark") for s in (1.0, -1.0)],  # pivot lugs
        # sills start ahead of the pin (they would swing down into the rails otherwise)
        *[box(4.45, 0.22, 0.15, at=(2.375, s * 0.42, 0.075), mat="dark") for s in (1.0, -1.0)],
        # longitudinal hoist brackets: cover the locus of the cylinder eye (see _hoist_fit)
        *[box(ex1 - ex0 + 0.30, 0.16, 0.15 - ez0 + 0.08, at=((ex0 + ex1) / 2, s * HOIST_Y,
              (0.15 + ez0 - 0.08) / 2), mat="dark", collision=False) for s in (1.0, -1.0)],
        box(BODY_FRONT + 0.05, 2.50, 0.08, at=(BODY_FRONT / 2 - 0.025, 0.0, 0.19)),     # floor
        beam((0.0, 0.19), (BODY_TAIL, 0.52), 2.62, 0.08),                            # tail kick-up
        box(0.10, 2.66, 1.28, at=(BODY_FRONT - 0.03, 0.0, 0.79)),                    # headboard
        # short canopy lip: a longer one reaches the cab's rear corners at full articulation
        beam((BODY_FRONT - 0.05, 1.43), (BODY_FRONT + 0.12, 1.47), 2.76, 0.10),
        # sloping side walls (main + raised tail section) and top rails
        *[s for sg in (1.0, -1.0) for s in (
            box(BODY_FRONT + 0.10, 0.07, 1.30, at=(BODY_FRONT / 2 - 0.05, sg * 1.31, 0.78),
                rpy=(-sg * 0.14, 0.0, 0.0)),
            box(-BODY_TAIL, 0.07, 1.00, at=(BODY_TAIL / 2, sg * 1.33, 0.93),
                rpy=(-sg * 0.14, 0.0, 0.0)),
            box(BODY_FRONT - BODY_TAIL, 0.14, 0.12, at=(xm, sg * 1.39, 1.40)),
            *[box(0.10, 0.05, 1.05, at=(x, sg * 1.37, 0.80), rpy=(-sg * 0.14, 0.0, 0.0),
                  collision=False) for x in (0.4, 1.6, 2.8, 4.0)],                    # side ribs
        )],
    ])

    # -------------------------------------------------------------- hoist cylinders
    dump_j = Joint("dump_joint", "revolute", "rear_frame", "dump_body",
                   xyz=(PIVOT[0], 0.0, PIVOT[1]), axis=(0, -1, 0), lower=0.0, upper=DUMP_MAX,
                   init=0.0, velocity=0.10, role="tool")
    k, c, ks, eye0 = hf["k"], hf["c"], hf["ks"], hf["cs"]
    rod_len = eye0 - 0.05            # retracted: rod end just short of the base pin
    if BARREL_LEN - (eye0 + ks * DUMP_MAX - rod_len) <= 0.15:
        raise ValueError("hoist rod leaves the barrel at full tip; lengthen BARREL_LEN")
    hoist_links, hoist_joints = [], []
    for side, sg in (("left", 1.0), ("right", -1.0)):
        y = sg * HOIST_Y
        hoist_links += [
            Link(f"{side}_hoist_barrel", 120.0, [
                cyl(0.10, BARREL_LEN, at=(BARREL_LEN / 2, 0.0, 0.0), axis="x", mat="dark",
                    collision=False),
                cyl(0.12, 0.10, at=(BARREL_LEN - 0.05, 0.0, 0.0), axis="x", mat="steel",
                    collision=False),
                cyl(0.09, 0.22, axis="y", mat="steel", collision=False),           # base eye
            ]),
            # rod frame = barrel frame; the rod eye sits at x = eye0 when retracted
            Link(f"{side}_hoist_rod", 60.0, [
                cyl(0.06, rod_len, at=(eye0 - rod_len / 2, 0.0, 0.0), axis="x", mat="steel",
                    collision=False),
                cyl(0.08, 0.20, at=(eye0, 0.0, 0.0), axis="y", mat="steel", collision=False),
            ]),
        ]
        hoist_joints += [
            # barrel frame yawed 180 deg: +x points rearwards along the cylinder
            Joint(f"{side}_hoist_barrel_joint", "revolute", "rear_frame", f"{side}_hoist_barrel",
                  xyz=(HOIST_BASE[0], y, HOIST_BASE[1]), rpy=(0.0, 0.0, math.pi), axis=(0, -1, 0),
                  lower=c - 0.1, upper=k * DUMP_MAX + c + 0.1, init=c,
                  velocity=abs(k) * dump_j.velocity + 0.1, role="follower",
                  follows=(dump_j.name, k, c)),
            Joint(f"{side}_hoist_rod_joint", "prismatic", f"{side}_hoist_barrel", f"{side}_hoist_rod",
                  axis=(1, 0, 0), lower=-0.05, upper=ks * DUMP_MAX + 0.05, init=0.0,
                  velocity=abs(ks) * dump_j.velocity + 0.1, role="follower",
                  follows=(dump_j.name, ks, 0.0)),
        ]

    # ------------------------------------------------------------------ wheels
    wheels, wheel_joints = [], []
    for axle, parent, x in (("front", "front_frame", 0.0), ("mid", "rear_frame", BOGIE_X[0]),
                            ("rear", "rear_frame", BOGIE_X[1])):
        for side, sg in (("left", 1.0), ("right", -1.0)):
            name = f"{axle}_{side}_wheel"
            wheels.append(wheel_link(name, WHEEL_R, TYRE_W, 600.0))       # tyre + 25" rim
            wheel_joints.append(wheel_joint(f"{name}_joint", parent, name, (x, sg * y_w, 0.0), WHEEL_R))

    links = [front, rear, body, *hoist_links, *wheels]
    joints = [
        Joint("articulation_joint", "revolute", "front_frame", "rear_frame",
              xyz=(HITCH_X, 0.0, 0.0), axis=(0, 0, 1), lower=-0.785, upper=0.785,
              velocity=0.35, role="articulation"),
        dump_j,
        *hoist_joints,
        *wheel_joints,
    ]

    return MachineSpec(
        name="articulated_dump_truck",
        abbrev="ADT",
        display_name="Articulated Dump Truck",
        category="hauling",
        reference="28 t articulated dump truck (CAT 730 / Volvo A30G class)",
        description=("6x6 articulated hauler: front tractor with engine hood and centre-mounted "
                     "cab, centre hinge (±45°), rear frame on a two-axle tandem bogie carrying a "
                     "17.5 m³ dump body that tips 70° about a rear pivot on two hoist cylinders."),
        root="front_frame",
        links=links,
        joints=joints,
        drive=Drive(type="articulated", max_speed=14.7, max_reverse_speed=4.2,
                    articulation_joint="articulation_joint", front_length=-HITCH_X,
                    rear_length=-TANDEM_X, articulation_rate=0.35),
        tools=[Tool("dump_joint", "dump_body", positive="raise", negative="lower")],
        cameras=[
            # operator eye height (2.95 m AGL) on the centre line, just ahead of the
            # windscreen (the glass panes are opaque boxes), looking down the hood
            Camera("front", "front_frame", (-0.24, 0.0, 2.15), (0.0, 0.22, 0.0)),
            # on the left mirror-arm end (3.07 m AGL), looking outward-forward; the mirror head
            # hangs 0.15 m lower, out of the frustum
            Camera("left", "front_frame", (MIRROR[0], MIRROR[1], MIRROR_ARM_Z + 0.07), (0.0, 0.30, 0.60)),
            # rear-view camera on the cab roof's rear right corner, looking back (20° outboard,
            # 17° down) over the canopy along the body's right side: the body tips in view
            # (headboard rising, hoist cylinders, bogie), with the ground right of the truck.
            # The roof plate stays inside the 0.1 m near clip.
            Camera("right", "front_frame", REAR_CAM, (0.0, 0.30, -2.80)),
        ],
        envelope=(10.40, 2.92, 3.43),
        operating_mass=23500.0,
        notes=("Hoist cylinders are cosmetic followers of dump_joint: barrel pinned on the rear "
               "frame + sliding rod, always straight; because follower angles are linear in q "
               "the rod eye slides up to 0.6 m along a bracket under the body while tipping. "
               "Cameras: front over the hood (operator view), left outward-forward from the "
               "left mirror arm, right is a rear-view camera on the cab roof's rear right "
               "corner looking back along the body, so raise/lower is visible in rgb_right."),
    )
