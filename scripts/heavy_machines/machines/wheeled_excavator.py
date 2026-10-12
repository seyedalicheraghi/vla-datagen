"""Wheeled excavator (WEX), 18 t class — modelled on a CAT M318 / Liebherr A918.

Datasheet numbers (CAT M318, 2.5 m stick, dual 10.00-20 tyres, rear blade):
    operating mass 17.9-19.95 t (model: 18.5 t) · wheelbase 2.70 m ·
    ground clearance 0.325 m · counterweight clearance 1.30 m ·
    tail-swing radius 2.35 m · shipping height (top of cab) 3.36 m ·
    shipping length 8.36-8.47 m · max reach at ground 9.21 m ·
    max dig depth 5.89 m · max loading height 7.30 m · travel 35 km/h.
    Liebherr A918 EW (2.65 m stick): 8.65 × 2.75 × 3.35 m, ~18 t, swing 0-9 rpm.
Assumed / typical for the class:
    10.00-20 tyre OD 1.04 m, section 0.28 m, dual spacing 0.32 m · gauge 1.94 m
    (twin centre to twin centre) → width over tyres 2.54 m (transport width
    2.55 m) · upper-structure width 2.50 m · mono boom 5.1 m pin to pin ·
    0.8 m³ bucket, 1.05 m wide · rear blade 2.50 m wide, lifts ≈ 0.45 m ·
    max virtual steer 0.55 rad (inner wheel 38°, outer turning radius ≈ 6.3 m) ·
    Orbitrol steering ≈ 5 s lock to lock → 0.22 rad/s virtual steer rate.
Envelope = CAT M318: L 8.45 m (shipping length), W 2.55 m, H 3.36 m.  The model
is 9.13 m long (+8 %) in the roading pose: the boom nose alone gives 8.48 m, but
the stick-cylinder eye on the stick's dog-leg tail sits 0.8 m ahead of the
boom-tip pin.  That 0.71 m lever, bent forward, is what keeps the stick cylinder
above the boom nose at both ends of the 120° stick stroke.
Model check (offline FK): reach at ground 9.1 m · dig depth 5.8 m ·
loading (bucket-pin) height 7.7 m · max cutting height 9.0 m · tail swing
2.35 m (counterweight facets inscribed in that arc; asserted in build()).

Joint ranges: boom -57°..+60° (foot-pin to stick-pin line -49°..+69°),
stick 120°, bucket 172°, blade -37°..+9°.  Rates at full command: swing
0.94 rad/s (9 rpm), boom 0.55 rad/s (full stroke ≈ 3.7 s), stick 0.65 rad/s,
bucket 1.0 rad/s, blade 0.32 rad/s.

Init pose = roading/transport pose: upper structure facing forward (boom over
the steered front axle), boom lowered, stick 0.15 rad short of fully retracted
and bucket curled to 1.80 rad (of 1.90), so the bucket clears the boom-cylinder
barrels and the boom and hangs 0.37 m above the road; rear blade raised.

The boom-cylinder barrels and the stick-cylinder rod are follower links (a
linear fit of the true pin-to-pin angle, < 4° error) so the cylinders stay
attached to both pins; the other cylinder halves are fixed to their links.
The stick cylinder sits on a bracket over the boom knee and build() asserts
it clears the boom (knee boss, top faces) over the whole stick range.
Front steering: the kingpins are at the twin centres; the axle housing stops
at y = ±0.48 and the knuckles carry the stub axles, so the inner tyre clears
the housing at full lock.

Root frame: swing centre at wheel-axle height (axle centres at z = 0),
+x towards the steered front axle.
"""

from __future__ import annotations

import math

from ..kinematics import Tf, shape_local_points
from ..spec import (Camera, Drive, Joint, Link, MachineSpec, Shape, Tool, beam, box, cyl,
                    sphere, wheel_joint, wheel_link)

# --- Undercarriage --------------------------------------------------------------
WHEEL_R, TYRE_W = 0.52, 0.28         # 10.00-20 tyre
TWIN = 0.32                          # dual-tyre centre spacing
GAUGE = 1.94                         # twin centre to twin centre
WHEELBASE = 2.70
X_FRONT, X_REAR = WHEELBASE / 2, -WHEELBASE / 2
FRONT_HOUSING_Y = 0.48               # steer-axle housing ends; the knuckles carry the stubs from here
SWING_Z = 0.73                       # top of the swing bearing above the axle line (1.25 m AGL)
BLADE_PIVOT = (-1.70, 0.0, -0.20)    # rear stabiliser blade arm pivot (0.32 m AGL)

# --- Upper structure -----------------------------------------------------------------
TAIL_R = 2.35                        # tail-swing radius: the counterweight is inscribed in this arc

# --- Front attachment ----------------------------------------------------------------
# Boom frame: origin = foot pin, x forward along the boom at q = 0 (x-z pairs below).
BOOM_Y = -0.12                       # boom sits right of centre, cab on the left
BOOM_FOOT = (0.90, BOOM_Y, 0.70)     # in the upper-structure frame (1.95 m AGL)
BOOM_KNEE = (2.45, 1.30)
BOOM_TIP = (5.05, 0.75)              # stick pin; 5.11 m from the foot pin
BOOM_DEPTH = (0.62, 0.50)            # foot->knee and knee->tip box depths
KNEE_R = 0.31                        # knee boss radius
STICK_LEN = 2.50
STICK_PITCH = 1.45                   # stick hangs 83° below the boom x axis at q = 0
BUCKET_W, BUCKET_R = 1.05, 1.25      # width, pin-to-tooth radius
# Hydraulic cylinder pins (x, z)
BCYL_BASE = (1.10, 0.25)             # boom cylinders, on the upper structure
BCYL_EYE = (2.20, 0.62)              # boom cylinders, under the boom (boom frame)
BCYL_DY = 0.33                       # boom cylinders at BOOM_Y ± BCYL_DY
SCYL_BASE = (2.30, 2.08)             # stick cylinder, on a 0.5 m bracket over the knee (boom frame)
# Stick-cylinder eye on the stick's dog-leg tail (stick frame): 0.71 m from the boom-tip pin,
# bent 10° towards the stick's top (bucket-cylinder) side.  The eye sweeps a 120° arc over
# the stick range; this lever/offset keeps it (and the rod) above the boom nose at both ends.
SCYL_EYE = (-0.70, 0.12)
SCYL_R, SCYL_ROD_R = 0.105, 0.065    # stick-cylinder barrel / rod radii


# ---------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------

def rod(p0: tuple[float, float], p1: tuple[float, float], radius: float, y: float = 0.0,
        mat: str = "steel") -> Shape:
    """Visual-only cylinder from p0 to p1 in the link's x-z plane (hydraulic cylinders)."""
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    return cyl(radius, math.hypot(dx, dz), at=((p0[0] + p1[0]) / 2, y, (p0[1] + p1[1]) / 2),
               rpy=(0.0, math.atan2(dx, dz), 0.0), mat=mat, collision=False)


def _elev(v: tuple[float, float]) -> float:
    """Elevation of an x-z vector (positive = +x tipped towards +z)."""
    return math.atan2(v[1], v[0])


def _raise(p: tuple[float, float], q: float) -> tuple[float, float]:
    """Rotate an x-z point by a joint with axis (0, -1, 0) at angle q."""
    c, s = math.cos(q), math.sin(q)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


def _unwrap(ys: list[float]) -> list[float]:
    out = list(ys)
    for i in range(1, len(out)):
        while out[i] - out[i - 1] > math.pi:
            out[i] -= 2 * math.pi
        while out[i] - out[i - 1] < -math.pi:
            out[i] += 2 * math.pi
    return out


def _mid_angle(ys: list[float]) -> float:
    ys = _unwrap(ys)
    return (max(ys) + min(ys)) / 2


def _linear_fit(qs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares slope, minimax offset; returns (k, c, max |error|)."""
    ys = _unwrap(ys)
    n = len(qs)
    mq, my = sum(qs) / n, sum(ys) / n
    k = sum((q - mq) * (y - my) for q, y in zip(qs, ys)) / sum((q - mq) ** 2 for q in qs)
    err = [y - k * q for q, y in zip(qs, ys)]
    c = (max(err) + min(err)) / 2
    c = math.atan2(math.sin(c + k * mq), math.cos(c + k * mq)) - k * mq   # mid-range in (-pi, pi]
    return k, c, (max(err) - min(err)) / 2


def _cylinder_fits(boom_lim: tuple[float, float], stick_lim: tuple[float, float]) -> dict:
    """Follower fits for the boom-cylinder barrels and the stick-cylinder rod."""
    out = {}
    qs = [boom_lim[0] + (boom_lim[1] - boom_lim[0]) * i / 120 for i in range(121)]
    foot = (BOOM_FOOT[0], BOOM_FOOT[2])
    eyes = [tuple(f + r for f, r in zip(foot, _raise(BCYL_EYE, q))) for q in qs]
    out["boom_barrel"] = _linear_fit(qs, [_elev((e[0] - BCYL_BASE[0], e[1] - BCYL_BASE[1]))
                                          for e in eyes])
    rod_ang = [_elev((BCYL_BASE[0] - e[0], BCYL_BASE[1] - e[1])) - q for e, q in zip(eyes, qs)]
    out["boom_rod_angle"] = _mid_angle(rod_ang)                            # fixed in the boom frame

    qs = [stick_lim[0] + (stick_lim[1] - stick_lim[0]) * i / 120 for i in range(121)]
    eyes = [_stick_eye(q) for q in qs]
    bar = [_elev((e[0] - SCYL_BASE[0], e[1] - SCYL_BASE[1])) for e in eyes]
    out["stick_barrel_angle"] = _mid_angle(bar)                            # fixed in the boom frame
    out["stick_rod"] = _linear_fit(qs, [_elev((SCYL_BASE[0] - e[0], SCYL_BASE[1] - e[1]))
                                        + STICK_PITCH - q for e, q in zip(eyes, qs)])
    span = [math.hypot(e[0] - SCYL_BASE[0], e[1] - SCYL_BASE[1]) for e in eyes]
    out["stick_span"] = (min(span), max(span))                             # base -> eye distance
    return out


def _stick_eye(q: float) -> tuple[float, float]:
    """Stick-cylinder eye in the boom x-z frame at stick angle q."""
    th = STICK_PITCH - q                         # stick frame = boom frame · R_y(th)
    c, s = math.cos(th), math.sin(th)
    return (BOOM_TIP[0] + SCYL_EYE[0] * c + SCYL_EYE[1] * s,
            BOOM_TIP[1] - SCYL_EYE[0] * s + SCYL_EYE[1] * c)


def _boom_clearance(p0: tuple[float, float], p1: tuple[float, float], r: float) -> float:
    """Clearance (m) of a cylinder with axis p0 -> p1 (boom x-z frame) and radius r above the
    boom body: the knee boss and the top faces of the two boom boxes.  < 0 = buried."""
    worst = math.inf
    for i in range(41):
        t = i / 40
        x, z = p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t
        worst = min(worst, math.hypot(x - BOOM_KNEE[0], z - BOOM_KNEE[1]) - KNEE_R - r)
        for a, b, depth in (((0.0, 0.0), BOOM_KNEE, BOOM_DEPTH[0]), (BOOM_KNEE, BOOM_TIP, BOOM_DEPTH[1])):
            dx, dz = b[0] - a[0], b[1] - a[1]
            length = math.hypot(dx, dz)
            if 0.0 <= ((x - a[0]) * dx + (z - a[1]) * dz) / length <= length:
                worst = min(worst, ((z - a[1]) * dx - (x - a[0]) * dz) / length - depth / 2 - r)
    return worst


def _follower(name: str, parent: str, child: str, xyz, src: Joint, fit) -> Joint:
    k, c, _ = fit
    ends = (k * src.lower + c, k * src.upper + c)
    return Joint(name, "revolute", parent, child, xyz=xyz, axis=(0, -1, 0),
                 lower=min(ends) - 0.1, upper=max(ends) + 0.1, init=k * src.init + c,
                 velocity=abs(k) * src.velocity + 0.1, role="follower", follows=(src.name, k, c))


def _cab() -> list[Shape]:
    """Operator cab on the left front of the upper structure (upper frame)."""
    x0, x1, y0, y1, z0, z1 = -0.45, 1.20, 0.32, 1.24, 0.30, 2.00
    cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    lx, ly, lz = x1 - x0, y1 - y0, z1 - z0
    return [
        box(lx, ly, lz, at=(cx, cy, cz)),                                            # cab shell
        box(lx + 0.06, ly + 0.04, 0.06, at=(cx, cy, z1 + 0.03), mat="accent"),        # roof
        # glass (visual only), slightly proud of the shell
        box(0.03, ly - 0.12, 1.05, at=(x1 + 0.01, cy, z0 + 1.10), mat="glass", collision=False),
        box(0.03, ly - 0.12, 0.40, at=(x1 + 0.01, cy, z0 + 0.32), mat="glass", collision=False),
        box(lx - 0.30, 0.03, 1.00, at=(cx + 0.10, y1 + 0.01, z0 + 1.08), mat="glass", collision=False),
        box(lx - 0.40, 0.03, 0.80, at=(cx + 0.15, y0 - 0.01, z0 + 1.18), mat="glass", collision=False),
        box(0.03, ly - 0.20, 0.70, at=(x0 - 0.01, cy, z0 + 1.20), mat="glass", collision=False),
        # door frame / pillars
        box(0.06, 0.03, lz - 0.10, at=(x1 - 0.05, y1 + 0.02, cz), mat="accent", collision=False),
        box(0.05, 0.03, lz - 0.10, at=(x0 + 0.62, y1 + 0.02, cz), mat="accent", collision=False),
        box(0.06, 0.03, lz - 0.10, at=(x1 - 0.05, y0 - 0.02, cz), mat="accent", collision=False),
        # mirror arm + mirror on the left front pillar
        box(0.16, 0.03, 0.03, at=(x1 + 0.07, y1 - 0.02, z0 + 1.05), mat="accent", collision=False),
        box(0.03, 0.10, 0.24, at=(x1 + 0.15, y1 + 0.01, z0 + 1.00), mat="accent", collision=False),
        # work lights on the front edge of the roof
        box(0.08, 0.16, 0.09, at=(x1 + 0.02, cy - 0.28, z1 - 0.06), mat="white", collision=False),
        box(0.08, 0.16, 0.09, at=(x1 + 0.02, cy + 0.28, z1 - 0.06), mat="white", collision=False),
    ]


def _counterweight() -> list[Shape]:
    """Counterweight as three facets inscribed in the tail-swing arc (upper frame).

    Each facet's outer corners lie on the circle of radius TAIL_R about the swing
    axis, so nothing reaches past the datasheet tail-swing radius; the inner halves
    are buried in the engine hood.  Tail lights and grille sit on the facets.
    """
    depth, height, zc = 0.55, 1.00, 0.55          # 1.30 m counterweight clearance
    shapes = []
    for a0, a1 in ((-32.0, -11.0), (-11.0, 11.0), (11.0, 32.0)):   # degrees from -x towards +y
        mid, half = math.radians((a0 + a1) / 2), math.radians((a1 - a0) / 2)
        face = TAIL_R * math.cos(half)            # outer face distance from the swing axis
        c, s = -math.cos(mid), math.sin(mid)      # outward normal
        r = face - depth / 2
        shapes.append(box(depth, 2 * TAIL_R * math.sin(half), height, at=(r * c, r * s, zc),
                          rpy=(0.0, 0.0, -mid)))
        if a0 > 0 or a1 < 0:                      # tail lights, low on the side facets
            r = face + 0.01
            shapes.append(box(0.03, 0.16, 0.10, at=(r * c, r * s, 0.30), rpy=(0.0, 0.0, -mid),
                              mat="red", collision=False))
    shapes.append(box(0.03, 0.66, 0.30, at=(-(TAIL_R * math.cos(math.radians(11.0)) + 0.01), 0.0, 0.80),
                      mat="accent", collision=False))   # grille on the centre facet
    return shapes


def _plan_radius(shapes: list[Shape]) -> float:
    """Largest distance from the swing axis (upper-frame z axis) over the given shapes."""
    worst = 0.0
    for sh in shapes:
        tf = Tf.from_xyz_rpy(sh.xyz, sh.rpy)
        worst = max([worst] + [math.hypot(p[0], p[1]) for p in
                               (tf.apply(v) for v in shape_local_points(sh, 72))])
    return worst


def build() -> MachineSpec:
    y_t = GAUGE / 2

    # ------------------------------------------------------------------ chassis
    undercarriage = Link("undercarriage", 5900.0, [
        box(3.90, 0.84, 0.36, at=(0.0, 0.0, 0.28)),                                # longitudinal frame
        box(1.40, 1.90, 0.42, at=(0.0, 0.0, 0.39)),                                # centre box (tanks/steps)
        cyl(0.80, 0.13, at=(0.0, 0.0, SWING_Z - 0.065), mat="dark"),              # swing bearing
        cyl(0.12, 2 * FRONT_HOUSING_Y, at=(X_FRONT, 0.0, 0.0), axis="y", mat="dark"),  # steer axle
        cyl(0.12, 1.30, at=(X_REAR, 0.0, 0.0), axis="y", mat="dark"),              # rear (rigid) axle
        sphere(0.195, at=(X_FRONT, 0.0, 0.0), mat="dark"),                         # differentials
        sphere(0.195, at=(X_REAR, 0.0, 0.0), mat="dark"),
        box(0.42, 0.70, 0.16, at=(X_FRONT, 0.0, 0.13), mat="dark"),                # axle mounts
        box(0.42, 0.70, 0.16, at=(X_REAR, 0.0, 0.13), mat="dark"),
        box(0.30, 0.50, 0.40, at=(-1.75, 0.0, -0.02), mat="dark"),                 # blade pivot bracket
        # access steps / toolbox lids on the centre box (visual)
        box(0.50, 0.06, 0.04, at=(0.0, 0.97, 0.25), mat="steel", collision=False),
        box(0.50, 0.06, 0.04, at=(0.0, 0.97, 0.45), mat="steel", collision=False),
        box(0.90, 0.02, 0.26, at=(0.0, -0.96, 0.39), mat="accent", collision=False),
        # blade lift cylinder brackets (visual)
        *[box(0.20, 0.10, 0.25, at=(-1.72, y, 0.30), mat="dark", collision=False) for y in (-0.35, 0.35)],
    ])

    # --------------------------------------------------------- upper structure
    # z = 0 is the top of the swing bearing (1.25 m above the ground).
    house = Link("upper_structure", 7650.0, [
        box(3.05, 2.50, 0.30, at=(-0.325, 0.0, 0.15)),                              # deck
        box(1.45, 2.50, 0.75, at=(-1.075, 0.0, 0.675)),                             # engine hood
        *_counterweight(),
        box(1.50, 0.65, 0.75, at=(0.45, -0.925, 0.675)),                            # tanks / toolbox
        box(0.75, 0.80, 0.55, at=(0.88, BOOM_Y, 0.575), mat="dark"),                # boom foot bracket
        *_cab(),
        # engine grilles (visual)
        box(0.90, 0.03, 0.42, at=(-1.10, -1.26, 0.70), mat="accent", collision=False),
        box(0.90, 0.03, 0.42, at=(-1.10, 1.26, 0.70), mat="accent", collision=False),
        box(1.00, 0.70, 0.03, at=(-1.10, 0.0, 1.06), mat="accent", collision=False),  # hood grille
        cyl(0.07, 0.45, at=(-0.80, -0.80, 1.27), mat="accent", collision=False),    # exhaust stack
        cyl(0.08, 0.10, at=(-1.60, 0.70, 1.10), mat="orange", collision=False),     # beacon
        # handrails on the right-hand tank (visual)
        box(1.30, 0.04, 0.04, at=(0.45, -1.22, 1.35), mat="accent", collision=False),
        box(0.04, 0.04, 0.30, at=(-0.15, -1.22, 1.20), mat="accent", collision=False),
        box(0.04, 0.04, 0.30, at=(1.05, -1.22, 1.20), mat="accent", collision=False),
    ])
    tail = _plan_radius(house.shapes)
    assert tail <= TAIL_R + 0.01, f"upper structure reaches {tail:.3f} m > tail-swing radius {TAIL_R}"

    boom_j = Joint("boom_joint", "revolute", "upper_structure", "boom",
                   xyz=BOOM_FOOT, axis=(0, -1, 0), lower=-1.00, upper=1.05, init=-0.35,
                   velocity=0.55, role="tool")
    stick_j = Joint("stick_joint", "revolute", "boom", "stick",
                    xyz=(BOOM_TIP[0], 0.0, BOOM_TIP[1]), rpy=(0.0, STICK_PITCH, 0.0),
                    axis=(0, -1, 0), lower=-1.30, upper=0.80, init=-1.15, velocity=0.65, role="tool")
    fits = _cylinder_fits((boom_j.lower, boom_j.upper), (stick_j.lower, stick_j.upper))

    # ------------------------------------------------------------------- boom
    ra = fits["boom_rod_angle"]
    rod_end = (BCYL_EYE[0] + 1.35 * math.cos(ra), BCYL_EYE[1] + 1.35 * math.sin(ra))
    # Stick cylinder: barrel fixed on the boom at the mid-stroke angle, rod on a follower
    # from the stick eye.  Barrel + rod overlap >= 0.2 m at full extension and neither
    # pokes past the opposite pin at full retraction.
    sa = fits["stick_barrel_angle"]
    span_lo, span_hi = fits["stick_span"]
    s_bar = round(0.55 * (span_hi + 0.2), 2)
    s_rod = round(span_hi + 0.2 - s_bar, 2)
    assert max(s_bar, s_rod) <= span_lo - 0.1, (s_bar, s_rod, span_lo)
    s_end = (SCYL_BASE[0] + s_bar * math.cos(sa), SCYL_BASE[1] + s_bar * math.sin(sa))
    # ... and both halves ride over the boom (knee boss and top faces) over the whole stick range.
    k_rod, c_rod, _ = fits["stick_rod"]
    clear = _boom_clearance(SCYL_BASE, s_end, SCYL_R)
    for i in range(41):
        q = stick_j.lower + (stick_j.upper - stick_j.lower) * i / 40
        eye = _stick_eye(q)
        el = k_rod * q + c_rod - (STICK_PITCH - q)          # rod elevation in the boom frame
        clear = min(clear, _boom_clearance(eye, (eye[0] + s_rod * math.cos(el),
                                                 eye[1] + s_rod * math.sin(el)), SCYL_ROD_R),
                    _boom_clearance(eye, eye, 0.10))         # eye boss
    assert clear >= 0.02, f"stick cylinder clears the boom by only {clear:.3f} m"
    boom = Link("boom", 1100.0, [
        beam((0.0, 0.0), BOOM_KNEE, 0.52, BOOM_DEPTH[0]),
        beam(BOOM_KNEE, BOOM_TIP, 0.46, BOOM_DEPTH[1]),
        cyl(KNEE_R, 0.52, at=(BOOM_KNEE[0], 0.0, BOOM_KNEE[1]), axis="y"),          # knee
        cyl(0.20, 0.64, axis="y", mat="steel", collision=False),                     # foot pin boss
        cyl(0.16, 0.56, at=(BOOM_TIP[0], 0.0, BOOM_TIP[1]), axis="y", mat="steel", collision=False),
        # boom-cylinder rods + eye cross-tube (the barrels are a follower link)
        *[rod(BCYL_EYE, rod_end, 0.055, y=y) for y in (-BCYL_DY, BCYL_DY)],
        cyl(0.07, 2 * BCYL_DY + 0.12, at=(BCYL_EYE[0], 0.0, BCYL_EYE[1]), axis="y",
            mat="dark", collision=False),
        box(0.30, 0.40, 0.30, at=(BCYL_EYE[0], 0.0, BCYL_EYE[1] + 0.12), collision=False),
        # stick-cylinder barrel on its bracket over the knee (the rod is a follower link)
        box(0.40, 0.30, 0.56, at=(SCYL_BASE[0], 0.0, SCYL_BASE[1] - 0.24), collision=False),
        cyl(0.09, 0.36, at=(SCYL_BASE[0], 0.0, SCYL_BASE[1]), axis="y", mat="steel", collision=False),
        rod(SCYL_BASE, s_end, SCYL_R, mat="dark"),
        cyl(0.12, 0.08, at=(s_end[0], 0.0, s_end[1]), rpy=(0.0, math.pi / 2 - sa, 0.0),
            mat="steel", collision=False),
        # work light on the left side of the boom
        box(0.12, 0.05, 0.12, at=(1.6, 0.27, 0.95), mat="white", collision=False),
    ])

    # ------------------------------------------------------------------ stick
    stick = Link("stick", 600.0, [
        beam((0.0, 0.0), (STICK_LEN, 0.0), 0.40, 0.46),
        beam((0.20, 0.0), SCYL_EYE, 0.36, 0.36),                                     # dog-leg tail
        box(0.22, 0.30, 0.20, at=(-0.28, 0.0, 0.28), collision=False),               # bucket-cyl lug
        cyl(0.17, 0.52, axis="y", mat="steel", collision=False),                     # pin boss
        cyl(0.10, 0.44, at=(SCYL_EYE[0], 0.0, SCYL_EYE[1]), axis="y", mat="steel", collision=False),
        cyl(0.13, 0.48, at=(STICK_LEN, 0.0, 0.0), axis="y", mat="steel", collision=False),
        # bucket cylinder on the outer side of the stick + side links (visual)
        rod((-0.30, 0.38), (0.95, 0.36), 0.09, mat="dark"),
        rod((0.90, 0.36), (2.00, 0.32), 0.055),
        *[rod((2.00, 0.32), (2.38, 0.08), 0.04, y=y, mat="dark") for y in (-0.16, 0.16)],
    ])

    # Bucket frame: origin at the bucket pin, +x toward the cutting edge at q = 0
    # (along the stick); the mouth faces -z (towards the machine when the stick
    # hangs down), the curved back faces +z.
    bucket = Link("bucket", 600.0, [
        beam((-0.02, 0.05), (0.30, 0.44), BUCKET_W, 0.06, mat="dark"),            # back plate
        beam((0.30, 0.44), (0.82, 0.48), BUCKET_W, 0.06, mat="dark"),
        beam((0.82, 0.48), (1.16, 0.22), BUCKET_W, 0.06, mat="dark"),
        beam((1.16, 0.22), (BUCKET_R, -0.08), BUCKET_W, 0.06, mat="dark"),         # bottom / lip
        *[s for y in (-BUCKET_W / 2, BUCKET_W / 2) for s in (                       # side plates
            beam((0.02, 0.13), (1.20, 0.05), 0.04, 0.24, y=y, mat="dark"),           # (stepped
            beam((0.14, 0.31), (1.04, 0.27), 0.04, 0.16, y=y, mat="dark"),           #  "D" profile)
            beam((0.34, 0.42), (0.84, 0.42), 0.04, 0.10, y=y, mat="dark"))],
        box(0.24, 0.34, 0.22, at=(0.04, 0.0, 0.10), mat="dark"),                    # pin bosses
        box(0.16, 0.30, 0.16, at=(0.10, 0.0, 0.36), mat="dark", collision=False),  # link lugs
        *[box(0.14, 0.08, 0.06, at=(BUCKET_R + 0.05, y, -0.12), mat="steel", collision=False)
          for y in (-0.39, -0.13, 0.13, 0.39)],                                     # teeth
    ])

    # ------------------------------------------------- hydraulic cylinder followers
    boom_barrels = Link("boom_cylinders", 180.0, [
        *[s for y in (-BCYL_DY, BCYL_DY) for s in (
            cyl(0.095, 1.45, at=(0.78, y, 0.0), axis="x", mat="dark", collision=False),
            cyl(0.11, 0.10, at=(1.48, y, 0.0), axis="x", mat="steel", collision=False))],
        cyl(0.07, 2 * BCYL_DY + 0.12, axis="y", mat="dark", collision=False),        # base pin
    ])
    stick_rod = Link("stick_cylinder_rod", 50.0, [
        cyl(SCYL_ROD_R, s_rod, at=(s_rod / 2, 0.0, 0.0), axis="x", mat="steel", collision=False),
        cyl(0.09, 0.20, axis="y", mat="steel", collision=False),                     # rod eye
    ])

    # --------------------------------------------------------- rear blade
    # Blade frame: origin at the arm pivot; at q = 0 the cutting edge is on the ground.
    blade = Link("stabilizer_blade", 650.0, [
        cyl(0.08, 1.20, axis="y", mat="dark"),                                       # pivot tube
        *[beam((0.0, 0.0), (-0.56, -0.02), 0.12, 0.20, y=y, mat="dark") for y in (-0.55, 0.55)],
        box(0.12, 2.50, 0.60, at=(-0.58, 0.0, -0.02)),                               # moldboard
        box(0.08, 2.50, 0.05, at=(-0.62, 0.0, -0.30), mat="steel"),                  # cutting edge
        box(0.10, 2.40, 0.08, at=(-0.50, 0.0, 0.26), mat="dark"),                    # top rib
        *[rod((-0.50, 0.25), (0.05, 0.62), 0.06, y=y, mat="steel") for y in (-0.35, 0.35)],
    ])

    # ------------------------------------------------------------ wheels
    def twin(prefix: str, parent: str, x: float, y_c: float, sign: float):
        """Inner/outer twin wheels about a twin centre at (x, y_c) of ``parent``."""
        lks, jts = [], []
        for tag, dy in (("inner", -sign * TWIN / 2), ("outer", sign * TWIN / 2)):
            name = f"{prefix}_{tag}_wheel"
            lks.append(wheel_link(name, WHEEL_R, TYRE_W, 180.0))
            jts.append(wheel_joint(f"{name}_joint", parent, name, (x, y_c + dy, 0.0), WHEEL_R))
        return lks, jts

    wheel_links, wheel_joints, knuckles, steer_joints = [], [], [], []
    for side, sign in (("left", 1.0), ("right", -1.0)):
        kn = f"front_{side}_knuckle"
        stub = y_t - FRONT_HOUSING_Y               # housing end -> twin centre (kingpin)
        knuckles.append(Link(kn, 150.0, [
            cyl(0.12, stub, at=(0.0, -sign * stub / 2, 0.0), axis="y", mat="dark"),  # stub axle
            cyl(0.10, 0.30, axis="z", mat="dark", collision=False),                  # kingpin
        ]))
        steer_joints.append(Joint(f"front_{side}_steer_joint", "revolute", "undercarriage", kn,
                                  xyz=(X_FRONT, sign * y_t, 0.0), axis=(0, 0, 1),
                                  lower=-0.70, upper=0.70, velocity=0.8, role="steer"))
        for lk_list, jt_list in (twin(f"front_{side}", kn, 0.0, 0.0, sign),
                                 twin(f"rear_{side}", "undercarriage", X_REAR, sign * y_t, sign)):
            wheel_links += lk_list
            wheel_joints += jt_list

    links = [undercarriage, house, boom, stick, bucket, blade, boom_barrels, stick_rod,
             *knuckles, *wheel_links]

    joints = [
        Joint("swing_joint", "continuous", "undercarriage", "upper_structure",
              xyz=(0.0, 0.0, SWING_Z), axis=(0, 0, 1), velocity=0.94, role="tool"),
        boom_j,
        stick_j,
        Joint("bucket_joint", "revolute", "stick", "bucket",
              xyz=(STICK_LEN, 0.0, 0.0), axis=(0, 1, 0), lower=-1.10, upper=1.90, init=1.80,
              velocity=1.0, role="tool"),
        Joint("blade_joint", "revolute", "undercarriage", "stabilizer_blade",
              xyz=BLADE_PIVOT, axis=(0, -1, 0), lower=-0.65, upper=0.15, init=-0.65,
              velocity=0.32, role="tool"),
        _follower("boom_cylinder_joint", "upper_structure", "boom_cylinders",
                  (BCYL_BASE[0], BOOM_Y, BCYL_BASE[1]), boom_j, fits["boom_barrel"]),
        _follower("stick_cylinder_joint", "stick", "stick_cylinder_rod",
                  (SCYL_EYE[0], 0.0, SCYL_EYE[1]), stick_j, fits["stick_rod"]),
        *steer_joints,
        *wheel_joints,
    ]

    return MachineSpec(
        name="wheeled_excavator",
        abbrev="WEX",
        display_name="Wheeled Excavator",
        category="excavation",
        reference="18 t wheeled excavator (CAT M318 / Liebherr A918 class)",
        description=("Two-axle wheeled excavator with twin tyres, Ackermann front steering, "
                     "a rear stabiliser blade, a 360° upper structure with the cab on the "
                     "left and a mono boom / stick / 0.8 m³ bucket."),
        root="undercarriage",
        links=links,
        joints=joints,
        drive=Drive(type="ackermann", max_speed=9.7, wheelbase=WHEELBASE, max_steer=0.55,
                    steer_rate=0.22, fixed_axle_x=X_REAR, steer_axle="front"),
        tools=[
            Tool("swing_joint", "swing", positive="left", negative="right"),
            Tool("boom_joint", "boom", positive="raise", negative="lower"),
            Tool("stick_joint", "stick", positive="out", negative="in"),
            Tool("bucket_joint", "bucket", positive="curl", negative="dump"),
            Tool("blade_joint", "stabilizer_blade", positive="lower", negative="raise"),
        ],
        cameras=[
            # operator eye height, just ahead of the windscreen, looking down the boom
            Camera("front", "upper_structure", (1.26, 0.78, 1.45), (0.0, 0.30, -0.12)),
            # front-left corner of the cab roof / front-right corner of the tank
            Camera("left", "upper_structure", (1.18, 1.20, 2.12), (0.0, 0.35, 0.65)),
            Camera("right", "upper_structure", (1.24, -1.24, 1.25), (0.0, 0.30, -0.60)),
        ],
        envelope=(8.45, 2.55, 3.36),        # CAT M318; model L is +8 % (see module docstring)
        operating_mass=18500.0,
        notes=("Bucket curl is not auto-verified (no up/out/left semantics): +1 swings the "
               "teeth towards the machine and turns the mouth up (checked in the sweep render). "
               "Cylinder followers are cosmetic."),
    )
