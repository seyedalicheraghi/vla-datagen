"""Bulldozer (DOZ), 20 t medium crawler dozer — modelled on a CAT D6 / Komatsu D65EX.

Reference numbers (rounded datasheet values for the class; CAT D6 (2019) and
Komatsu D65EX-18 sheets):
    operating mass ≈ 20.5 t with semi-U blade, single-shank ripper and ROPS cab
    (CAT D6 ≈ 20.6 t, Komatsu D65EX-18 ≈ 20.3 t) · engine ≈ 160 kW ·
    elevated-sprocket ("high drive") steel tracks, track gauge 1.88 m, 560 mm
    shoes → width over tracks 2.44 m · track on ground ≈ 2.8 m, overall track
    length ≈ 3.5 m · ground clearance ≈ 0.40 m · height over the ROPS cab 3.20 m ·
    semi-U blade 3.42 m wide × 1.42 m high (≈ 5.6 m³; D6 SU 3.26 m, D65EX
    semi-U 3.46 m), lift above ground ≈ 1.15 m, digging depth ≈ 0.50 m, max
    tilt ≈ 0.66-0.70 m (corner height difference) · single-shank ripper: max
    penetration ≈ 0.60 m, clearance under the raised tip ≈ 0.55 m · overall
    length with blade and ripper ≈ 6.6 m (basic tractor ≈ 4.2 m) · travel
    0-10 km/h forward (2.8 m/s), 0-12 km/h reverse (3.3 m/s) · counter-rotation
    ≈ 0.5 rad/s (assumed).

Joints (rates per second at full command):
    blade_lift_joint  push arms about the trunnions on the track roller frames
                      (arms outboard of the tracks): -0.161 … +0.359 rad →
                      cutting edge 0.50 m below grade (0.54 m at the wing tips)
                      … 1.12 m above ground; 0.18 rad/s (≈ 0.55 m/s at the edge,
                      full stroke ≈ 2.9 s).  +1 = raise.
    blade_tilt_joint  blade roll about its fore-aft axis through the blade centre
                      at the push-arm ball joints: ±0.20 rad (±11.5°, corner
                      height difference 0.66 m), 0.15 rad/s.  +1 = tilt_left:
                      the LEFT (+y) end of the blade goes DOWN and the right end
                      up, as with the operator's joystick moved left on CAT /
                      Komatsu dozers.  The two push arms follow the tilt
                      (k = ∓0.50: one up, one down) so the ball joints stay on
                      the blade.
    ripper_joint      parallelogram ripper arms on the rear mount (pivots 1.00 /
                      1.46 m above ground, arms 1.10 m): -0.545 … +0.555 rad →
                      shank tip 0.59 m below grade … 0.56 m above ground;
                      0.30 rad/s.  +1 = raise.  The tool beam (k = -1) keeps the
                      shank vertical and the upper link (k = +1) closes the
                      parallelogram.  At full depth only the shank is in the
                      ground: beam, pocket and arms stay ≥ 0.27 m above grade.
    Hydraulic cylinders are visual-only followers:
      * lift cylinders: barrels pivot on brackets on the hood sides, rods on
        lugs high on the blade back (0.80 m above the cutting edge, as on D6 /
        D65).  Their angles are fitted linearly in the lift angle (max error
        1.4° → rod ≤ 4.5 cm off the barrel axis at the extremes).  The lugs are
        on a virtual C-frame ("lift_frame") that does not roll, so at full tilt
        the blade ears slide ≤ 0.18 m along them (pinning them to the blade would
        need a follower of both lift and tilt).
      * tilt cylinder (right) and tilt brace (left): barrel / outer tube on the
        push arm, rod on the blade's upper corner lug.  Each half is aimed at the
        other's pivot by its own follower of blade_tilt_joint (axis fitted to the
        cone the direction sweeps; ≤ 1.3° pointing error → ≤ 1.6 cm off the other
        pivot), so the rod stays pinned to the lug and slides in the barrel
        (0.72-0.79 m pin to pin) while the barrel swings ±14° on the push arm.
      * ripper cylinders: barrels on the rear mount, rods on the ripper beam
        (max error 1.5° → rod ≤ 3 cm off the barrel axis).

Layout as on the real machine: high-drive track triangles with the sprocket
under the rear of the cab, engine hood sloping from 2.06 m at the cab down to
1.72 m at the radiator (the D6's sloped hood / D65's "slant nose" for blade
visibility) with the exhaust stack and pre-cleaner outboard on it, radiator
guard with grille and corner lamps behind the blade, push arms outboard of the
tracks, lift cylinders from towers on the hood sides down to the blade back,
ROPS cab centred over the rear half, fuel tank behind the cab and the
single-shank ripper on the rear.

Cameras: "front" sits on the front edge of the cab roof on the centreline
(above the operator's eye, like an autonomy-retrofit camera), pitched 0.55 rad
down; from the seated eye (2.62 m) the hood and radiator guard hide most of the
blade at or below grade (even with the lowered hood only 1.4 % of the image at
grade, 0.3 % at full digging depth).  From the roof it sees the whole blade top
edge and both corners in every pose (pinhole render with the env intrinsics:
blade 6.0 % of the image in the init pose, 4.1 % at grade, 1.8 % at full
digging depth, tilt ±0.2 clearly visible).  "left" / "right" sit on the front corners of the roof
and look out over the blade ends (blade 4.7 % at grade, 2.4 % at full depth).

Init pose = travel/transport pose: blade raised (lowest point 0.38 m above
ground), tilt level, ripper fully raised.  Model envelope in that pose
6.64 × 3.43 × 3.20 m (blade top 1.84 m; 2.53 m at full lift).

Frames: geometry is written in "ground coordinates" — x forward from the
middle of the track on the ground, y left, z above the ground, all joints at
q = 0 (blade on the ground, ripper arms horizontal) — and converted to each
link frame by subtracting that link's origin.  Root link = tractor, origin at
the idler axle height (0.33 m above the ground).
"""

from __future__ import annotations

import math

from .. import kinematics as K
from ..spec import (Box, Camera, Drive, Joint, Link, MachineSpec, Shape, Tool, beam, box, cyl,
                    sphere, sprocket_link)

# ----------------------------------------------------------------------------- undercarriage
GAUGE, SHOE_W, BELT = 1.88, 0.56, 0.09
IDLER_R = 0.33                         # outer radius of the track around the idlers
FRONT_IDLER = (1.40, IDLER_R)
REAR_IDLER = (-1.40, IDLER_R)
SPROCKET = (-0.95, 0.98)               # elevated drive sprocket centre
SPROCKET_WRAP_R = 0.40                 # outer radius of the track around the sprocket
SPROCKET_R = round(SPROCKET_WRAP_R - BELT / 2, 3)  # pitch radius = belt centre line (0.355)
Z0 = IDLER_R                           # root origin height above the ground
ROOT_O = (0.0, 0.0, Z0)

# ----------------------------------------------------------------------------- tractor body
CAB_X, CAB_Y, CAB_Z = (-1.45, 0.22), 0.86, (1.50, 3.04)
ROOF_TOP = 3.20
HOOD_X, HOOD_Y = (0.22, 1.90), 0.56
HOOD_TOP = (2.06, 1.72)                # hood top at the cab / at the radiator (sloped)
FENDER_Z = (1.42, 1.50)
EXHAUST = (0.95, -0.43)                # exhaust stack (x, y) — outboard, clear of the
PRECLEANER = (1.30, 0.42)              # pre-cleaner        — front camera's centre line

# ----------------------------------------------------------------------------- blade
TRUNNION = (-0.10, 0.62)               # push-arm trunnion (x, z) on the track roller frames
ARM_Y = 1.36                           # push arms run outboard of the tracks
BALL = (2.60, 0.38)                    # push-arm ball joint on the blade (x, z)
TILT_O = (2.70, 0.0, 0.40)             # blade frame origin = tilt axis (fore-aft)
PROFILE = [(3.06, 0.00), (2.90, 0.38), (2.84, 0.80), (2.88, 1.18), (3.00, 1.42)]
BLADE_YC = 1.22                        # half-width of the straight centre section
WING_A, WING_L = 0.42, 0.49            # semi-U wings: 24° forward, 0.49 m long
BLADE_T = 0.07                         # moldboard plate thickness
BLADE_BACK = 2.66                      # back face of the blade's upper box beam (x)
LIFT_BASE = (0.95, 2.22)               # lift-cylinder pivot on the hood-side towers
LIFT_EYE = (2.60, 0.80)                # lift-cylinder eye on the blade back (blade on ground)
LIFT_CYL_Y = 0.80
TILT_CYL_BASE = (2.08, 0.66)           # tilt cylinder / brace pivot on the push arm (x, z)
TILT_LUG = (2.61, 1.18)                # tilt rod eye on the blade's upper back corner (x, z)
TILT_CYL_Y = 1.30
EDGE = (3.06, 0.0)                     # cutting edge (lowest point of the centre section)
LIFT_UP, LIFT_DOWN, LIFT_TRAVEL = 1.15, -0.50, 0.40   # edge heights: max lift, max cut, init
TILT_MAX = 0.20

# ----------------------------------------------------------------------------- ripper
RIP_LO = (-2.12, 1.00)                 # lower arm pivot on the rear mount
RIP_HI = (-2.12, 1.46)                 # upper link pivot
RIP_ARM = 1.10
RIP_TIP = (-3.00, -0.03)               # lowest point of the shank tip at q = 0
RIP_CYL_BASE = (-2.17, 1.58)
RIP_CYL_EYE_OFF = (-0.10, 0.0)         # eye relative to the beam's lower pivot
RIP_CYL_Y = 0.28
RIP_CLEAR, RIP_DEPTH = 0.55, -0.60


# ============================================================================= helpers
def _bx(x0, x1, y0, y1, z0, z1, mat="body", collision=True, o=ROOT_O):
    """Box from ground-coordinate extents, in the frame of a link whose origin is ``o``."""
    return box(abs(x1 - x0), abs(y1 - y0), abs(z1 - z0),
               at=((x0 + x1) / 2 - o[0], (y0 + y1) / 2 - o[1], (z0 + z1) / 2 - o[2]),
               mat=mat, collision=collision)


def _bm(p0, p1, width, depth, y=0.0, mat="body", collision=True, o=ROOT_O):
    """beam() between ground-coordinate (x, z) points."""
    return beam((p0[0] - o[0], p0[1] - o[2]), (p1[0] - o[0], p1[1] - o[2]), width, depth,
                y=y - o[1], mat=mat, collision=collision)


def _cy(r, length, at, axis="y", mat="steel", collision=False, o=ROOT_O):
    return cyl(r, length, at=(at[0] - o[0], at[1] - o[1], at[2] - o[2]), axis=axis, mat=mat,
               collision=collision)


def _hood_z(x):
    """Height of the sloped hood top at x (ground coordinates)."""
    (x0, x1), (z0, z1) = HOOD_X, HOOD_TOP
    return z0 + (z1 - z0) * (x - x0) / (x1 - x0)


def _rot(p, q, s):
    """Rotate (x, z) about the origin by joint angle q of a joint with axis (0, s, 0)."""
    a = s * q
    return (p[0] * math.cos(a) + p[1] * math.sin(a), -p[0] * math.sin(a) + p[1] * math.cos(a))


def _about(pivot, p, q, s):
    r = _rot((p[0] - pivot[0], p[1] - pivot[1]), q, s)
    return (pivot[0] + r[0], pivot[1] + r[1])


def _angle_for_height(pivot, p, h):
    """Joint angle (axis (0, -1, 0)) that brings point p (at q = 0) to height h."""
    dx, dz = p[0] - pivot[0], p[1] - pivot[1]
    return math.asin((h - pivot[1]) / math.hypot(dx, dz)) - math.atan2(dz, dx)


def _linfit(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    c = my - k * mx
    return k, c, max(abs(k * x + c - y) for x, y in zip(xs, ys))


def _cylinder_fit(base, eye_fn, lo, hi, backward=False, n=61):
    """Linear fit of a cylinder's elevation angle (axis (0, -1, 0) convention) vs. q.

    Returns (k, c, max_err, (len_min, len_max)); ``eye_fn(q)`` gives the eye (x, z).
    ``backward``: measure the elevation from the -x direction (for a joint frame
    yawed by pi, so a rearward-pointing cylinder keeps its angle near 0).
    """
    qs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    ang, ln, prev = [], [], None
    for q in qs:
        e = eye_fn(q)
        dx, dz = e[0] - base[0], e[1] - base[1]
        a = math.atan2(dz, -dx if backward else dx)
        if prev is not None:                      # unwrap
            a += 2 * math.pi * round((prev - a) / (2 * math.pi))
        ang.append(a)
        ln.append(math.hypot(dx, dz))
        prev = a
    k, c, err = _linfit(qs, ang)
    return k, c, err, (min(ln), max(ln))


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _tangent(c1, r1, c2, r2, ref):
    """Outer common tangent of two circles (x, z) on the side of ``ref``: (t1, t2, normal)."""
    dx, dz = c2[0] - c1[0], c2[1] - c1[1]
    d = math.hypot(dx, dz)
    ux, uz, mx, mz = dx / d, dz / d, -dz / d, dx / d
    a = (r1 - r2) / d
    b = math.sqrt(1.0 - a * a)
    cands = [(a * ux + sb * b * mx, a * uz + sb * b * mz) for sb in (1.0, -1.0)]
    nx, nz = max(cands, key=lambda n: n[0] * ref[0] + n[1] * ref[1])
    return (c1[0] + r1 * nx, c1[1] + r1 * nz), (c2[0] + r2 * nx, c2[1] + r2 * nz), (nx, nz)


def _belt(c1, r1, c2, r2, ref, y):
    t1, t2, n = _tangent(c1, r1, c2, r2, ref)
    h = BELT / 2
    return _bm((t1[0] - n[0] * h, t1[1] - n[1] * h), (t2[0] - n[0] * h, t2[1] - n[1] * h),
               SHOE_W, BELT, y=y, mat="dark")


def _track(y: float) -> list[Shape]:
    """High-drive steel track (triangle: front idler, rear idler, elevated sprocket)."""
    fx, rx = FRONT_IDLER[0], REAR_IDLER[0]
    shapes = [
        _bx(rx, fx, y - SHOE_W / 2, y + SHOE_W / 2, 0.0, BELT, mat="dark"),        # ground run
        _cy(IDLER_R, SHOE_W, (fx, y, IDLER_R), mat="dark", collision=True),          # front bend
        _cy(IDLER_R, SHOE_W, (rx, y, IDLER_R), mat="dark", collision=True),          # rear bend
        _cy(SPROCKET_WRAP_R, SHOE_W, (SPROCKET[0], y, SPROCKET[1]), mat="dark", collision=True),
        _belt(FRONT_IDLER, IDLER_R, SPROCKET, SPROCKET_WRAP_R, (0.0, 1.0), y),      # upper run
        _belt(SPROCKET, SPROCKET_WRAP_R, REAR_IDLER, IDLER_R, (-1.0, 0.3), y),      # rear run
        # track roller frame, rollers, idlers (inside the loop)
        _bx(-1.22, 1.22, y - 0.17, y + 0.17, 0.27, 0.52, mat="accent"),
        _cy(0.25, SHOE_W + 0.02, (fx, y, IDLER_R), mat="steel"),
        _cy(0.25, SHOE_W + 0.02, (rx, y, IDLER_R), mat="steel"),
        _cy(0.09, SHOE_W + 0.03, (fx, y, IDLER_R), mat="dark"),
        _cy(0.09, SHOE_W + 0.03, (rx, y, IDLER_R), mat="dark"),
        _bm((-1.00, 0.50), (SPROCKET[0] + 0.02, SPROCKET[1] - 0.20), 0.30, 0.20, y=y,
            mat="accent", collision=False),                                          # frame → final drive
    ]
    for x in (-1.04, -0.62, -0.21, 0.21, 0.62, 1.04):                               # track rollers
        shapes.append(_cy(0.115, SHOE_W * 0.62, (x, y, BELT + 0.115), mat="steel"))
    # grouser bars across the upper run (visual texture)
    t1, t2, n = _tangent(FRONT_IDLER, IDLER_R, SPROCKET, SPROCKET_WRAP_R, (0.0, 1.0))
    pitch = -math.atan2(t2[1] - t1[1], t2[0] - t1[0])
    for i in range(1, 12):
        f = i / 12
        px, pz = t1[0] + f * (t2[0] - t1[0]), t1[1] + f * (t2[1] - t1[1])
        shapes.append(Shape(Box((0.05, SHOE_W, 0.03)),
                            (px + n[0] * 0.012, y, pz + n[1] * 0.012 - Z0), (0.0, pitch, 0.0),
                            "accent", False, True))
    return shapes


def _wing(pa, pb, side, depth, mat="body", collision=True, o=TILT_O):
    """Moldboard segment of a semi-U wing: the centre-section segment pa→pb swept forward."""
    dx, dz = pb[0] - pa[0], pb[1] - pa[1]
    xm, zm = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
    cx = xm + WING_L / 2 * math.sin(WING_A)
    cy = side * (BLADE_YC + WING_L / 2 * math.cos(WING_A))
    return Shape(Box((math.hypot(dx, dz), WING_L, depth)),
                 (cx - o[0], cy - o[1], zm - o[2]),
                 (0.0, -math.atan2(dz, dx), -side * WING_A), mat, collision, True)


def _follower(name, parent, child, xyz, source, k, c, lo, hi, axis=(0, -1, 0), init_src=0.0,
              rpy=(0.0, 0.0, 0.0)):
    vals = (k * lo + c, k * hi + c)
    return Joint(name, "revolute", parent, child, xyz=xyz, rpy=rpy, axis=axis,
                 lower=min(vals) - 0.05, upper=max(vals) + 0.05, init=k * init_src + c,
                 velocity=2.0, role="follower", follows=(source, k, c))


def _aim_follower(name, parent, child, xyz, aim, source, lo, hi, n=41):
    """Follower whose child +x keeps pointing along the unit vector ``aim(q)`` (parent frame).

    The joint axis is the normal of the plane through the aim directions at lo, 0
    and hi, so the child's +x sweeps the cone through all three; the angle about it
    is fitted linearly in q.  Returns (joint, max pointing error in rad).
    """
    d0 = aim(0.0)
    a = K.vunit(_cross(K.vsub(aim(hi), d0), K.vsub(aim(lo), d0)))

    def perp(d):
        return K.vsub(d, K.vscale(a, _dot(d, a)))

    p0 = perp(d0)
    qs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    angs = [math.atan2(_dot(a, _cross(p0, perp(aim(q)))), _dot(p0, perp(aim(q)))) for q in qs]
    k, c, _ = _linfit(qs, angs)
    err = max(math.acos(max(-1.0, min(1.0, _dot(K.mat_vec(K.axis_angle_to_mat(a, k * q + c), d0),
                                                 aim(q))))) for q in qs)
    rpy = (0.0, -math.atan2(d0[2], math.hypot(d0[0], d0[1])), math.atan2(d0[1], d0[0]))
    axis = tuple(round(v, 6) for v in K.mat_vec(K.transpose(K.rpy_to_mat(rpy)), a))
    return _follower(name, parent, child, xyz, source, k, c, lo, hi, axis=axis, rpy=rpy), err


def _barrel_shapes(front, back, r, ys, mat="dark") -> list[Shape]:
    """Cylinder barrels along +x from their pivot (toward the eye), visual only."""
    return [s for y in ys for s in (
        cyl(r, front + back, at=((front - back) / 2, y, 0.0), axis="x", mat=mat, collision=False),
        cyl(r * 1.25, 0.06, at=(front - 0.03, y, 0.0), axis="x", mat=mat, collision=False),
        cyl(r * 0.9, 0.16, at=(0.0, y, 0.0), axis="y", mat="steel", collision=False),
    )]


def _rod_shapes(length, r, ys) -> list[Shape]:
    """Piston rods along +x from their eye (toward the barrel), visual only."""
    return [s for y in ys for s in (
        cyl(r, length, at=(length / 2, y, 0.0), axis="x", mat="steel", collision=False),
        cyl(r * 1.5, 0.12, at=(0.0, y, 0.0), axis="y", mat="steel", collision=False),
    )]


# ============================================================================= build
def build() -> MachineSpec:
    y_t = GAUGE / 2

    # ------------------------------------------------------------------ tractor (root)
    cx0, cx1 = CAB_X
    cz0, cz1 = CAB_Z
    hx0, hx1 = HOOD_X
    zf = HOOD_TOP[1]
    hood_t = HOOD_TOP[0] - HOOD_TOP[1] + 0.12     # sloped hood top, thick enough to fill the wedge
    tr = [*_track(+y_t), *_track(-y_t)]
    tr += [
        _bx(-1.90, 1.90, -0.60, 0.60, 0.42, 1.20, mat="dark"),                    # main frame / cases
        _bx(1.70, 2.10, -0.55, 0.55, 0.42, 0.62, mat="dark"),                     # front bumper / guard
        _bx(hx0, hx1, -HOOD_Y, HOOD_Y, 1.05, zf),                                 # engine hood / side panels
        _bm((hx0, HOOD_TOP[0] - hood_t / 2), (hx1, zf - hood_t / 2), 2 * HOOD_Y, hood_t),  # sloped top
        _bx(1.90, 2.04, -0.64, 0.64, 0.55, zf + 0.02),                            # radiator guard
        _bx(2.04, 2.055, -0.50, 0.50, 0.98, zf - 0.06, mat="accent", collision=False),  # grille
        *[_bx(2.055, 2.065, -0.50, 0.50, z, z + 0.035, mat="body", collision=False)
          for z in (1.06, 1.20, 1.34, 1.48, 1.62, 1.76) if z + 0.035 < zf - 0.08],  # grille bars
        _bx(cx0 - 0.05, cx1, -0.60, 0.60, 1.20, cz0),                             # cab base
        _bx(cx0, cx1, -CAB_Y, CAB_Y, cz0, cz1),                                   # ROPS cab
        _bx(cx0 - 0.07, cx1 + 0.08, -CAB_Y - 0.06, CAB_Y + 0.06, cz1, ROOF_TOP),  # roof
        _bx(cx1, cx1 + 0.02, -0.74, 0.74, 1.70, 2.94, mat="glass", collision=False),     # windscreen
        _bx(cx0 - 0.02, cx0, -0.70, 0.70, 2.05, 2.94, mat="glass", collision=False),     # rear window
        _bx(-2.05, -1.45, -0.92, 0.92, 1.25, 1.98),                               # fuel tank
        _bx(-2.25, -1.88, -0.50, 0.50, 0.48, RIP_CYL_BASE[1] + 0.10, mat="dark"),  # ripper mount
        cyl(0.075, 0.70, at=(EXHAUST[0], EXHAUST[1], _hood_z(EXHAUST[0]) - 0.02 + 0.35 - Z0),
            mat="accent", collision=False),                                       # exhaust stack
        cyl(0.13, 0.26, at=(PRECLEANER[0], PRECLEANER[1], _hood_z(PRECLEANER[0]) + 0.11 - Z0),
            mat="dark", collision=False),                                         # pre-cleaner
        cyl(0.15, 0.04, at=(PRECLEANER[0], PRECLEANER[1], _hood_z(PRECLEANER[0]) + 0.26 - Z0),
            mat="accent", collision=False),
    ]
    for sy in (1.0, -1.0):
        tr += [
            _bx(-1.50, 1.30, sy * 0.60, sy * 1.24, *FENDER_Z),                    # fender
            _bm((1.30, 1.46), (1.66, 1.14), 0.64, 0.08, y=sy * 0.92),             # fender nose
            _bx(-1.30, 0.10, sy * CAB_Y, sy * (CAB_Y + 0.02), 1.75, 2.94, mat="glass",
                collision=False),                                                 # side window
            _bx(0.80, 1.70, sy * HOOD_Y, sy * (HOOD_Y + 0.012), 1.30, zf - 0.06, mat="accent",
                collision=False),                                                 # hood louvres
            *[_bx(0.80, 1.70, sy * (HOOD_Y + 0.012), sy * (HOOD_Y + 0.02), z, z + 0.03,
                  collision=False) for z in (1.38, 1.48, 1.58)],
            _bx(cx1 + 0.08, cx1 + 0.12, sy * 0.50, sy * 0.70, 3.07, 3.15, mat="white",
                collision=False),                                                 # front work lights
            _bx(-2.07, -2.05, sy * 0.70, sy * 0.84, 1.80, 1.90, mat="red", collision=False),  # tail light
            _bx(2.04, 2.08, sy * 0.51, sy * 0.63, zf - 0.15, zf - 0.03, mat="accent",
                collision=False),                                                 # front lamp housing
            _bx(2.08, 2.09, sy * 0.53, sy * 0.61, zf - 0.13, zf - 0.05, mat="white",
                collision=False),                                                 # front lamp
            _bx(-2.07, -2.05, sy * 0.46, sy * 0.60, 1.80, 1.90, mat="white", collision=False),
            _bx(LIFT_BASE[0] - 0.07, LIFT_BASE[0] + 0.07, sy * HOOD_Y, sy * (LIFT_CYL_Y - 0.09),
                _hood_z(LIFT_BASE[0]) - 0.25, LIFT_BASE[1] + 0.07, mat="dark",
                collision=False),                                                 # lift-cyl tower
            _cy(0.12, 0.30, (TRUNNION[0], sy * 1.11, TRUNNION[1]), mat="dark"),  # trunnion
            # handrail along the fender beside the cab door
            _bm((-1.40, 1.90), (0.15, 1.90), 0.03, 0.03, y=sy * 1.20, mat="steel", collision=False),
            *[_bm((x, 1.50), (x, 1.90), 0.03, 0.03, y=sy * 1.20, mat="steel", collision=False)
              for x in (-1.40, 0.15)],
        ]
    tractor = Link("tractor", 14000.0, tr)

    # ------------------------------------------------------------------ blade lift / tilt
    lift_lo = _angle_for_height(TRUNNION, EDGE, LIFT_DOWN)
    lift_hi = _angle_for_height(TRUNNION, EDGE, LIFT_UP)
    lift_init = _angle_for_height(TRUNNION, EDGE, LIFT_TRAVEL)
    lift_o = (TRUNNION[0], 0.0, TRUNNION[1])

    # Virtual C-frame: carries the trunnion caps and the lift-cylinder lugs, which
    # must not roll with the blade tilt (the lift-cylinder rods hang from it too).
    lift_frame = Link("lift_frame", 120.0, [
        s for sy in (1.0, -1.0) for s in (
            _cy(0.09, 0.05, (TRUNNION[0], sy * (ARM_Y + 0.12), TRUNNION[1]), o=lift_o),  # trunnion cap
            _bx(LIFT_EYE[0] - 0.06, BLADE_BACK, sy * (LIFT_CYL_Y - 0.09), sy * (LIFT_CYL_Y + 0.09),
                LIFT_EYE[1] - 0.11, LIFT_EYE[1] + 0.11, mat="dark", collision=False, o=lift_o),  # lug
        )])

    def push_arm(side: float, name: str) -> Link:
        y = side * ARM_Y
        return Link(name, 600.0, [
            _bm(TRUNNION, (BALL[0] - 0.10, BALL[1] + 0.01), 0.16, 0.30, y=y, o=lift_o),
            _bm((0.40, 0.60), (BALL[0] - 0.12, BALL[1] + 0.12), 0.14, 0.14, y=y, o=lift_o),
            _cy(0.17, 0.20, (TRUNNION[0], y, TRUNNION[1]), mat="dark", o=lift_o),
            sphere(0.10, at=(BALL[0] - lift_o[0], y, BALL[1] - lift_o[2]), mat="steel",
                   collision=False),
            _bx(2.00, 2.16, side * 1.28, side * 1.40, 0.52, 0.70, mat="dark", collision=False,
                o=lift_o),                                       # tilt cylinder / brace bracket
        ])

    # blade (frame origin on the tilt axis)
    o = TILT_O
    bl = []
    for pa, pb in zip(PROFILE[:-1], PROFILE[1:]):
        bl.append(_bm(pa, pb, 2 * BLADE_YC, BLADE_T, o=o))
        for sy in (1.0, -1.0):
            bl.append(_wing(pa, pb, sy, BLADE_T))
    edge0, edge1 = (3.07, 0.035), (3.03, 0.16)
    bl.append(_bm(edge0, edge1, 2 * BLADE_YC, BLADE_T + 0.01, mat="steel", o=o))     # cutting edge
    bl += [
        _bx(2.64, 2.90, -1.30, 1.30, 0.14, 0.46, o=o),                            # lower box beam
        _bx(BLADE_BACK, 2.88, -1.30, 1.30, 1.00, 1.30, o=o),                      # upper box beam
        *[_bx(2.68, 2.86, y - 0.04, y + 0.04, 0.46, 1.00, collision=False, o=o)
          for y in (-1.10, -0.45, 0.45, 1.10)],                                   # ribs
    ]
    tip_dx, tip_y = WING_L * math.sin(WING_A), BLADE_YC + WING_L * math.cos(WING_A)
    for sy in (1.0, -1.0):
        bl += [
            _wing(edge0, edge1, sy, BLADE_T + 0.01, mat="steel"),
            _bx(2.70 + tip_dx, 3.04 + tip_dx, sy * (tip_y - 0.01), sy * (tip_y + 0.04), 0.04, 1.40, o=o),  # end plate
            _bx(2.92 + tip_dx, 3.10 + tip_dx, sy * (tip_y - 0.015), sy * (tip_y + 0.045), 0.0, 0.20,
                mat="steel", collision=False, o=o),                               # end bit
            _bx(BALL[0] - 0.08, 2.66, sy * (ARM_Y - 0.10), sy * (ARM_Y + 0.10), 0.22, 0.52,
                mat="dark", collision=False, o=o),                                # ball socket
            _bx(TILT_LUG[0] - 0.05, BLADE_BACK + 0.02, sy * (TILT_CYL_Y - 0.08),
                sy * (TILT_CYL_Y + 0.08), TILT_LUG[1] - 0.10, TILT_LUG[1] + 0.10, mat="dark",
                collision=False, o=o),                                            # tilt rod lug
            _bx(BLADE_BACK, 2.83, sy * (LIFT_CYL_Y - 0.10), sy * (LIFT_CYL_Y + 0.10),
                LIFT_EYE[1] - 0.16, 1.00, collision=False, o=o),                  # lift-cylinder ears
        ]
    blade = Link("blade", 2600.0, bl)

    # push arms follow the tilt so their ball joints stay on the blade
    ball_rel = (BALL[0] - TILT_O[0], ARM_Y, BALL[1] - TILT_O[2])
    r_arm = math.hypot(BALL[0] - TRUNNION[0], BALL[1] - TRUNNION[1])
    phi0 = math.atan2(BALL[1] - TRUNNION[1], BALL[0] - TRUNNION[0])
    ths = [-TILT_MAX + 2 * TILT_MAX * i / 40 for i in range(41)]
    deltas = []
    for th in ths:
        p = K.mat_vec(K.axis_angle_to_mat((-1.0, 0.0, 0.0), th), ball_rel)
        z_rel = TILT_O[2] + p[2] - TRUNNION[1]
        deltas.append(math.asin(z_rel / r_arm) - phi0)
    k_arm, c_arm, err_arm = _linfit(ths, deltas)
    if err_arm > 0.01:
        raise ValueError(f"push-arm tilt follower not linear enough ({err_arm:.3f} rad)")

    # Tilt cylinder (right) and tilt brace (left): the barrel / outer tube rides on the
    # push arm, the rod on the blade lug.  Their relative motion depends on the tilt
    # only (both hang from lift_frame), so each half is a follower of the tilt that
    # keeps aiming at the other half's pivot.
    blade_o = K.vsub(TILT_O, lift_o)                  # blade frame origin in lift_frame

    def tilt_ends(side, th):
        """(arm rotation, blade rotation, barrel pivot, rod eye) in lift_frame at tilt th."""
        ra = K.axis_angle_to_mat((0.0, -1.0, 0.0), side * (k_arm * th + c_arm))
        rb = K.axis_angle_to_mat((-1.0, 0.0, 0.0), th)
        b = K.mat_vec(ra, (TILT_CYL_BASE[0] - lift_o[0], side * TILT_CYL_Y,
                           TILT_CYL_BASE[1] - lift_o[2]))
        e = K.vadd(blade_o, K.mat_vec(rb, (TILT_LUG[0] - TILT_O[0], side * TILT_CYL_Y,
                                           TILT_LUG[1] - TILT_O[2])))
        return ra, rb, b, e

    tilt_links, tilt_joints, miss_tilt = [], [], 0.0
    for side, arm, barrel, rod, r_b, r_r, mat, mass in (
            (-1.0, "right_push_arm", "right_tilt_cylinder", "right_tilt_rod", 0.065, 0.035,
             "dark", (70.0, 30.0)),
            (1.0, "left_push_arm", "left_tilt_brace", "left_brace_rod", 0.055, 0.035,
             "body", (60.0, 30.0))):
        def aim_barrel(th, side=side):
            ra, _, b, e = tilt_ends(side, th)
            return K.vunit(K.mat_vec(K.transpose(ra), K.vsub(e, b)))

        def aim_rod(th, side=side):
            _, rb, b, e = tilt_ends(side, th)
            return K.vunit(K.mat_vec(K.transpose(rb), K.vsub(b, e)))

        lens = [K.vnorm(K.vsub(tilt_ends(side, th)[3], tilt_ends(side, th)[2])) for th in ths]
        jb, eb = _aim_follower(f"{barrel}_joint", arm, barrel,
                               (TILT_CYL_BASE[0] - lift_o[0], side * TILT_CYL_Y,
                                TILT_CYL_BASE[1] - lift_o[2]),
                               aim_barrel, "blade_tilt_joint", -TILT_MAX, TILT_MAX)
        jr, er = _aim_follower(f"{rod}_joint", "blade", rod,
                               (TILT_LUG[0] - TILT_O[0], side * TILT_CYL_Y, TILT_LUG[1] - TILT_O[2]),
                               aim_rod, "blade_tilt_joint", -TILT_MAX, TILT_MAX)
        miss_tilt = max(miss_tilt, max(eb, er) * max(lens))   # axis miss at the other pivot
        t_front = min(lens) - 0.30                    # barrel stops 0.30 m short of the eye
        t_rod = max(lens) - t_front + 0.08            # rod 8 cm inside at full extension
        tilt_links += [Link(barrel, mass[0], _barrel_shapes(t_front, 0.08, r_b, (0.0,), mat=mat)),
                       Link(rod, mass[1], _rod_shapes(t_rod, r_r, (0.0,)))]
        tilt_joints += [jb, jr]
    if miss_tilt > 0.025:
        raise ValueError(f"tilt cylinder followers not accurate enough ({miss_tilt:.3f} m)")

    # lift cylinders: barrel on the hood-side tower, rod on the lift frame
    def lift_eye(q):
        return _about(TRUNNION, LIFT_EYE, q, -1.0)
    k_l, c_l, err_l, (lmin, lmax) = _cylinder_fit(LIFT_BASE, lift_eye, lift_lo, lift_hi)
    if err_l > 0.035:
        raise ValueError(f"lift cylinder angle not linear enough ({err_l:.3f} rad)")
    front = lmin - 0.08                           # barrel never reaches the eye
    rod_len = lmax - front + 0.10                 # rod still 10 cm inside at full extension
    lift_cyl = Link("lift_cylinders", 220.0,
                    _barrel_shapes(front, 0.12, 0.085, (LIFT_CYL_Y, -LIFT_CYL_Y)))
    lift_rods = Link("lift_rods", 120.0, _rod_shapes(rod_len, 0.05, (LIFT_CYL_Y, -LIFT_CYL_Y)))

    # ------------------------------------------------------------------ ripper
    def rip_beam_pos(q):
        return _about(RIP_LO, (RIP_LO[0] - RIP_ARM, RIP_LO[1]), q, 1.0)
    rip_hi = math.asin((RIP_CLEAR - RIP_TIP[1]) / RIP_ARM)
    rip_lo = math.asin((RIP_DEPTH - RIP_TIP[1]) / RIP_ARM)
    bl0 = (RIP_LO[0] - RIP_ARM, RIP_LO[1])
    bu0 = (RIP_HI[0] - RIP_ARM, RIP_HI[1])
    arm_o = (RIP_LO[0], 0.0, RIP_LO[1])
    beam_o = (bl0[0], 0.0, bl0[1])
    link_o = (RIP_HI[0], 0.0, RIP_HI[1])

    rip_arm_sh = []
    for sy in (1.0, -1.0):
        rip_arm_sh += [
            _bm(RIP_LO, bl0, 0.08, 0.24, y=sy * 0.42, o=arm_o),
            _cy(0.11, 0.12, (RIP_LO[0], sy * 0.42, RIP_LO[1]), mat="dark", o=arm_o),
        ]
    ripper_arm = Link("ripper_arm", 450.0, rip_arm_sh)

    sx, bz = -3.30, bl0[1]                        # shank x, beam lower-pivot height at q = 0
    rip_beam_sh = [
        _cy(0.08, 0.96, (bl0[0], 0.0, bl0[1]), mat="dark", collision=True, o=beam_o),     # lower tube
        _cy(0.07, 0.30, (bu0[0], 0.0, bu0[1]), mat="dark", o=beam_o),                     # upper pin
        _bx(sx - 0.12, sx + 0.12, -0.14, 0.14, bz - 0.16, bz + 0.66, o=beam_o),   # shank pocket
        _bm((sx, bz + 1.00), (sx, 0.42), 0.09, 0.30, mat="dark", o=beam_o),       # shank
        _bm((sx, 0.42), (-3.14, 0.10), 0.09, 0.26, mat="dark", o=beam_o),
        _bm((-3.14, 0.10), (RIP_TIP[0], RIP_TIP[1] + 0.06), 0.10, 0.12, mat="steel",
            o=beam_o),                                                            # tip
        _bx(sx - 0.05, sx + 0.05, -0.05, 0.05, bz + 0.76, bz + 0.84, mat="steel", collision=False,
            o=beam_o),                                                            # pin
    ]
    for sy in (1.0, -1.0):
        rip_beam_sh.append(_bx(bl0[0] - 0.09, bl0[0] + 0.09, sy * 0.36, sy * 0.48, bl0[1] - 0.08,
                               bu0[1] + 0.08, o=beam_o))                          # side plates
        rip_beam_sh.append(_bx(sx - 0.10, sx + 0.10, sy * 0.14, sy * 0.36, bz + 0.15,
                               bz + 0.35, collision=False, o=beam_o))             # web to the pocket
    ripper_beam = Link("ripper_beam", 900.0, rip_beam_sh)
    ripper_link = Link("ripper_link", 180.0, [
        _bm(RIP_HI, bu0, 0.20, 0.16, o=link_o, collision=False),
        _cy(0.08, 0.24, (RIP_HI[0], 0.0, RIP_HI[1]), mat="dark", o=link_o),
    ])

    eye_off = RIP_CYL_EYE_OFF

    def rip_eye(q):
        b = rip_beam_pos(q)
        return (b[0] + eye_off[0], b[1] + eye_off[1])
    # barrel joint frame yawed by pi: elevation measured from the rear horizontal
    k_r, c_r, err_r, (rmin, rmax) = _cylinder_fit(RIP_CYL_BASE, rip_eye, rip_lo, rip_hi,
                                                  backward=True)
    if err_r > 0.03:
        raise ValueError(f"ripper cylinder angle not linear enough ({err_r:.3f} rad)")
    r_front = rmin - 0.08
    r_rod = rmax - r_front + 0.08
    rip_cyl = Link("ripper_cylinders", 140.0,
                   _barrel_shapes(r_front, 0.10, 0.075, (RIP_CYL_Y, -RIP_CYL_Y)))
    rip_rods = Link("ripper_rods", 60.0, _rod_shapes(r_rod, 0.045, (RIP_CYL_Y, -RIP_CYL_Y)))

    # ------------------------------------------------------------------ joints
    sp_x, sp_z = SPROCKET[0], SPROCKET[1] - Z0
    k_rod_l, c_rod_l = k_l - 1.0, _wrap(c_l + math.pi)
    joints = [
        Joint("blade_lift_joint", "revolute", "tractor", "lift_frame",
              xyz=(TRUNNION[0], 0.0, TRUNNION[1] - Z0), axis=(0, -1, 0),
              lower=lift_lo, upper=lift_hi, init=lift_init, velocity=0.18, role="tool"),
        Joint("blade_tilt_joint", "revolute", "lift_frame", "blade",
              xyz=(TILT_O[0] - TRUNNION[0], 0.0, TILT_O[2] - TRUNNION[1]), axis=(-1, 0, 0),
              lower=-TILT_MAX, upper=TILT_MAX, init=0.0, velocity=0.15, role="tool"),
        _follower("left_push_arm_joint", "lift_frame", "left_push_arm", (0.0, 0.0, 0.0),
                  "blade_tilt_joint", k_arm, c_arm, -TILT_MAX, TILT_MAX),
        _follower("right_push_arm_joint", "lift_frame", "right_push_arm", (0.0, 0.0, 0.0),
                  "blade_tilt_joint", -k_arm, -c_arm, -TILT_MAX, TILT_MAX),
        *tilt_joints,
        _follower("lift_cylinder_joint", "tractor", "lift_cylinders",
                  (LIFT_BASE[0], 0.0, LIFT_BASE[1] - Z0), "blade_lift_joint", k_l, c_l,
                  lift_lo, lift_hi, init_src=lift_init),
        _follower("lift_rod_joint", "lift_frame", "lift_rods",
                  (LIFT_EYE[0] - TRUNNION[0], 0.0, LIFT_EYE[1] - TRUNNION[1]), "blade_lift_joint",
                  k_rod_l, c_rod_l, lift_lo, lift_hi, init_src=lift_init),
        Joint("ripper_joint", "revolute", "tractor", "ripper_arm",
              xyz=(RIP_LO[0], 0.0, RIP_LO[1] - Z0), axis=(0, 1, 0),
              lower=rip_lo, upper=rip_hi, init=rip_hi, velocity=0.30, role="tool"),
        _follower("ripper_beam_joint", "ripper_arm", "ripper_beam", (-RIP_ARM, 0.0, 0.0),
                  "ripper_joint", -1.0, 0.0, rip_lo, rip_hi, axis=(0, 1, 0), init_src=rip_hi),
        _follower("ripper_link_joint", "tractor", "ripper_link",
                  (RIP_HI[0], 0.0, RIP_HI[1] - Z0), "ripper_joint", 1.0, 0.0, rip_lo, rip_hi,
                  axis=(0, 1, 0), init_src=rip_hi),
        _follower("ripper_cylinder_joint", "tractor", "ripper_cylinders",
                  (RIP_CYL_BASE[0], 0.0, RIP_CYL_BASE[1] - Z0), "ripper_joint", k_r, c_r,
                  rip_lo, rip_hi, init_src=rip_hi, rpy=(0.0, 0.0, math.pi)),
        # the beam does not rotate, so the rod (pointing forward at the barrel) is at -angle
        _follower("ripper_rod_joint", "ripper_beam", "ripper_rods",
                  (eye_off[0], 0.0, eye_off[1]), "ripper_joint", -k_r, -c_r,
                  rip_lo, rip_hi, init_src=rip_hi),
        Joint("left_sprocket_joint", "continuous", "tractor", "left_sprocket",
              xyz=(sp_x, y_t, sp_z), axis=(0, 1, 0), velocity=20.0, role="roll_left",
              wheel_radius=SPROCKET_R),
        Joint("right_sprocket_joint", "continuous", "tractor", "right_sprocket",
              xyz=(sp_x, -y_t, sp_z), axis=(0, 1, 0), velocity=20.0, role="roll_right",
              wheel_radius=SPROCKET_R),
    ]

    links = [tractor, lift_frame, push_arm(1.0, "left_push_arm"), push_arm(-1.0, "right_push_arm"),
             blade, *tilt_links, lift_cyl, lift_rods, ripper_arm, ripper_beam, ripper_link,
             rip_cyl, rip_rods,
             sprocket_link("left_sprocket", SPROCKET_R, SHOE_W + 0.04, 150.0),
             sprocket_link("right_sprocket", SPROCKET_R, SHOE_W + 0.04, 150.0)]

    cam_x = cx1 + 0.05
    return MachineSpec(
        name="bulldozer",
        abbrev="DOZ",
        display_name="Bulldozer",
        category="dozing",
        reference="20 t medium crawler dozer (CAT D6 / Komatsu D65EX class)",
        description="High-drive crawler dozer with a 3.4 m semi-U tilt blade on outboard push "
                    "arms and a rear single-shank ripper.",
        root="tractor",
        links=links,
        joints=joints,
        drive=Drive(type="skid", max_speed=2.8, max_reverse_speed=3.3, max_yaw_rate=0.5,
                    track_width=GAUGE),
        tools=[
            Tool("blade_lift_joint", "blade_lift", positive="raise", negative="lower"),
            Tool("blade_tilt_joint", "blade_tilt", positive="tilt_left", negative="tilt_right"),
            Tool("ripper_joint", "ripper", positive="raise", negative="lower"),
        ],
        cameras=[
            # front edge of the cab roof on the centre line, looking over the sloped hood:
            # the blade top edge and both corners stay in view down to full digging depth
            Camera("front", "tractor", (cx1 + 0.14, 0.0, ROOF_TOP + 0.06 - Z0), (0.0, 0.55, 0.0)),
            # front corners of the cab roof, looking out over the blade ends
            Camera("left", "tractor", (cam_x + 0.05, CAB_Y + 0.08, ROOF_TOP - 0.05 - Z0),
                   (0.0, 0.45, 0.40)),
            Camera("right", "tractor", (cam_x + 0.05, -CAB_Y - 0.08, ROOF_TOP - 0.05 - Z0),
                   (0.0, 0.45, -0.40)),
        ],
        envelope=(6.60, 3.42, 3.20),
        operating_mass=20500.0,
        ground_contact=["tractor"],
        notes="blade_tilt +1 = tilt_left: the left (+y) end of the blade goes DOWN (operator "
              "convention); not auto-verified.  FK at +0.2: blade corners left 0.08 m / right "
              "0.74 m above ground in the init pose (left 0.33 m below / right 0.32 m above at "
              "grade) — visible in the 3-D panels of the pose render and in the front camera "
              "(the ortho 'front' panels are drawn back-to-front by render.py).  Push arms, "
              "hydraulic cylinders and the ripper parallelogram are followers.",
    )
