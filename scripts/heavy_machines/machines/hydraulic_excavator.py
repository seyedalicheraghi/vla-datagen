"""Hydraulic excavator (HEX), 20 t class — modelled on a CAT 320 / Komatsu PC210.

Reference numbers (typical 20 t crawler excavator datasheet):
    operating mass 20.5 t · track length 4.45 m · gauge 2.38 m · 600 mm shoes
    (width 2.98 m) · ground clearance 0.47 m · tail-swing radius 2.83 m ·
    cab height 2.96 m · counterweight clearance 1.05 m · boom 5.7 m ·
    stick 2.9 m · 1.0 m³ bucket · swing 11 rpm · travel 5.4 km/h ·
    shipping length ≈ 9.5 m.

Init pose = shipping/transport pose: boom lowered, stick folded under the
boom, bucket curled clear of the ground.

Root frame: centre of the undercarriage at track mid-height.

Layout (as on the real machine class): the cab is on the LEFT of the house, the
boom foot is on the front frame just right of the centre line, the hydraulic
tank / tool box is on the front right, the engine hood spans the rear and the
counterweight is a faceted casting inscribed in the 2.83 m tail-swing circle
(asserted in build()).  The cab is 1.45 m long so the left boom cylinder passes
between its front glass and the boom foot.

Hydraulics: both boom cylinders (one on each side of the boom foot), the stick
cylinder (over the boom), the bucket cylinder (over the stick) and the bucket
linkage (idler link on the stick, H-link on the bucket) are visual follower
links (no collision, no tool of their own).  Each cylinder is a barrel pinned on
one link and a rod pinned on the other, both revolute followers whose angle is
a minimax line fit of the true pin-to-pin direction (init pose weighted 5x), so
both pins are always exact and the only error is a sideways offset between
barrel and rod.  Measured over each tool's whole range (offset = distance of the
rod eye from the barrel axis; init pose / worst case at an end stop):

    boom cylinders 1.4 / 11 cm · stick cylinder 2.5 / 11 cm · bucket cylinder
    1.8 / 8 cm · bucket linkage (idler link vs H-link at their common pin)
    1.7 / 11 cm

The offsets come from the nonlinearity of the pin-to-pin angle (the stick
cylinder's eye is only 0.3 m from the stick pin, so its direction swings
sinusoidally over the 129 deg stick range); build() asserts they stay below
these bounds.  Strokes are 1.4x (boom), 1.2x (stick) and 1.3x (bucket) the
closed length.  The stick cylinder sits on an A-frame bracket over the boom knee
and rides over the boom nose (which tapers to 0.38 m depth): at the stick-in stop
(the init pose) the rod just grazes it (axis 6.5 cm from the nose, rod radius
5.8 cm, asserted in build()); the left boom cylinder passes between the cab
front and the boom-foot ear.

Because the init pose lays the stick horizontally 0.2 m above the ground with
its cylinder side facing down, the stick is only 0.38 m deep (top face at
z = +0.19 in its frame, root block 0.49 m) and the bucket cylinder, linkage and
their pins stay within 0.2 m of that face so nothing hangs into the ground.

Visual detail on top of the unchanged datasheet envelope: track rollers (7
bottom, 2 carrier), idlers, 36 tread lugs per side, nine-tooth sprockets; cab
with door, hinges, roof, vent, work lights and mirrors; handrails, engine
louvres, exhaust, beacon, counterweight lamps and grille; boom foot ears with
pin retainers; stick-tail ears; five bucket teeth on adapters.
"""

from __future__ import annotations

import math

from .. import kinematics as K
from ..kinematics import Tf, shape_local_points
from ..spec import Box, Camera, Drive, Joint, Link, MachineSpec, Shape, Tool, beam, box, cyl

# Undercarriage
TRACK_L, TRACK_H, SHOE_W, GAUGE = 4.45, 0.95, 0.60, 2.38
SWING_Z = 0.745                 # top of the swing bearing above the root origin
# Upper structure (house frame: z = 0 is the top of the swing bearing)
TAIL_R = 2.83                   # tail-swing radius: the counterweight is inscribed in this arc
CAB = (-0.45, 1.00, 0.27, 1.27)  # cab x0, x1, y0, y1 (left side of the house)
CAB_Z = (0.35, 1.63, 1.74)      # cab floor, roof underside, roof top (= datasheet cab height)
# Boom geometry in the boom frame (origin = foot pin, x forward when q = 0)
BOOM_FOOT = (1.10, -0.09, 0.75)  # in the upper-structure frame
BOOM_KNEE = (2.75, 1.55)
BOOM_TIP = (5.60, 1.05)          # stick pin; 5.70 m from the foot pin
STICK_LEN = 2.90
STICK_PITCH = 1.37               # stick hangs 78.5° below the boom frame x at q = 0
BUCKET_INIT = 1.75               # fully curled for transport
BOOM_Y = BOOM_FOOT[1]
BOOM_MID = (BOOM_KNEE[0] + 0.58 * (BOOM_TIP[0] - BOOM_KNEE[0]),
            BOOM_KNEE[1] + 0.58 * (BOOM_TIP[1] - BOOM_KNEE[1]))   # nose taper starts here
NOSE_DEPTH = 0.38                # boom depth over the last 42 % of the second segment
# Hydraulic cylinder pins: (base pin on the parent-side link, rod-eye pin on the driven link)
BOOM_CYL_Y = 0.46                # the two boom cylinders sit at BOOM_Y +- this
BOOM_CYL = ((1.20, BOOM_Y, 0.40), (1.54, 0.0, 0.53))   # front frame -> lug under the boom (boom frame)
ARM_CYL = ((3.70, 0.0, 2.65), (-0.30, 0.0, 0.06))      # bracket over the boom knee -> stick tail ears
BKT_CYL_BASE = (0.17, 0.0, 0.30)                       # stick top (rod eye on the idler link)
# Bucket four-bar: stick - idler link - H-link - bucket (x, z in the stick / bucket frame)
IDLER_PIN = (2.36, 0.04)         # idler-link pin through the stick, 0.54 m behind the bucket pin
IDLER_LEN, HLINK_LEN = 0.47, 0.65
LINK_PIN = (0.07, 0.29)          # H-link pin on the bucket ears (bucket frame)
STICK_TOP = 0.19                 # stick top face (+z side); the init pose lays the stick 0.2 m above the
#                                  ground with that face down, so everything on it must stay low
W_INIT, SIGMA_INIT = 4.0, 0.12   # extra weight / width (rad) of the init pose in the follower fits
# Bounds on the barrel / rod offset (m): (init pose, whole range); see the module docstring
MAX_OFFSET = {"boom_cyl": (0.03, 0.12), "arm_cyl": (0.04, 0.13), "bucket_cyl": (0.04, 0.10)}
MAX_LINKAGE_GAP = (0.03, 0.13)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def rod(p0: tuple[float, float], p1: tuple[float, float], radius: float, y: float = 0.0,
        mat: str = "steel") -> Shape:
    """Visual-only cylinder from p0 to p1 in the link's x-z plane (pipes, hoses, struts)."""
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    return cyl(radius, math.hypot(dx, dz), at=((p0[0] + p1[0]) / 2, y, (p0[1] + p1[1]) / 2),
               rpy=(0.0, math.atan2(dx, dz), 0.0), mat=mat, collision=False)


def _boom_top(s: float, h: float = 0.04) -> tuple[float, float]:
    """Point ``h`` above the top face of the boom's first segment, ``s`` along it (boom frame)."""
    kx, kz = BOOM_KNEE
    n = math.hypot(kx, kz)
    ux, uz = kx / n, kz / n
    return (ux * s - uz * (0.35 + h), uz * s + ux * (0.35 + h))


def _plan_radius(shapes: list[Shape]) -> float:
    """Largest distance from the swing axis (house z axis) over the given shapes."""
    worst = 0.0
    for sh in shapes:
        tf = Tf.from_xyz_rpy(sh.xyz, sh.rpy)
        worst = max([worst] + [math.hypot(p[0], p[1]) for p in
                               (tf.apply(v) for v in shape_local_points(sh, 72))])
    return worst


def _boom_clearance(spec: MachineSpec, p: tuple[float, float]) -> float:
    """Signed distance (m, boom x-z frame) from a point to the boom's collision shapes."""
    best = math.inf
    for sh in spec.link("boom").shapes:
        if not sh.collision:
            continue
        tf = Tf.from_xyz_rpy(sh.xyz, sh.rpy)
        loc = K.mat_vec(K.transpose(tf.R), K.vsub((p[0], 0.0, p[1]), tf.t))
        g = sh.geom
        if isinstance(g, Box):
            dx, dz = abs(loc[0]) - g.size[0] / 2, abs(loc[2]) - g.size[2] / 2
            d = math.hypot(max(dx, 0.0), max(dz, 0.0)) if dx > 0 or dz > 0 else max(dx, dz)
        else:                                          # knee boss: cylinder about the boom y axis
            d = math.hypot(loc[0], loc[1]) - g.radius
        best = min(best, d)
    return best


def _arm_cyl_clearance(spec: MachineSpec) -> float:
    """Smallest distance (m) from the stick-cylinder axis to the boom over the stick range.

    The last 0.25 m at the stick eye (which lies beyond the boom nose) are excluded.
    """
    jt = spec.joint("stick_joint")
    (ax, _, az), eye = ARM_CYL
    worst = math.inf
    for q in _samples(jt):
        tfs = K.link_transforms(spec, {"stick_joint": q})
        tb = tfs["boom"]
        v = K.mat_vec(K.transpose(tb.R), K.vsub(tfs["stick"].apply(eye), tb.t))
        length = math.hypot(v[0] - ax, v[2] - az)
        for i in range(31):
            t = i / 30 * (1.0 - 0.25 / length)
            worst = min(worst, _boom_clearance(spec, (ax + (v[0] - ax) * t, az + (v[2] - az) * t)))
    return worst


# ---------------------------------------------------------------------------
# Follower fitting (hydraulic cylinders and the bucket linkage)
# ---------------------------------------------------------------------------

def _unwrap(a: list[float]) -> list[float]:
    out = list(a)
    for i in range(1, len(out)):
        out[i] += 2 * math.pi * round((out[i - 1] - out[i]) / (2 * math.pi))
    return out


def _samples(jt: Joint, n: int = 40) -> list[float]:
    return sorted({jt.lower + (jt.upper - jt.lower) * i / n for i in range(n + 1)} | {jt.init})


def _fit(qs: list[float], ys: list[float], q_init: float) -> tuple[float, float]:
    """Line k*q + c minimising the largest weighted residual (init pose weighs 1 + W_INIT)."""
    ys = _unwrap(ys)
    ws = [1.0 + W_INIT * math.exp(-((q - q_init) / SIGMA_INIT) ** 2) for q in qs]

    def worst(k: float) -> tuple[float, float]:
        r = [y - k * q for q, y in zip(qs, ys)]
        lo, hi = min(r), max(r)
        for _ in range(40):          # balance the worst weighted over- and undershoot
            c = (lo + hi) / 2
            over = max(w * (x - c) for w, x in zip(ws, r))
            under = max(w * (c - x) for w, x in zip(ws, r))
            lo, hi = (c, hi) if over > under else (lo, c)
        return max(over, under), c
    k0 = (ys[-1] - ys[0]) / (qs[-1] - qs[0])
    lo, hi = k0 - 1.0, k0 + 1.0
    for _ in range(50):              # the weighted worst residual is convex in k
        k1, k2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        lo, hi = (k1, hi) if worst(k1)[0] > worst(k2)[0] else (lo, k2)
    k = (lo + hi) / 2
    return k, worst(k)[1]


def _follower(name: str, parent: str, child: str, xyz, src: Joint, k: float, c: float) -> Joint:
    """Revolute follower about -y: child +x elevation in the parent = k * q_src + c.

    The mid-range angle goes into the joint origin so the joint stays centred on 0.
    """
    base = k * (src.lower + src.upper) / 2 + c
    lo, hi = sorted((k * src.lower + c - base, k * src.upper + c - base))
    return Joint(name, "revolute", parent, child, xyz=tuple(xyz), rpy=(0.0, -base, 0.0),
                 axis=(0, -1, 0), lower=lo - 0.02, upper=hi + 0.02, init=k * src.init + c - base,
                 velocity=abs(k) * src.velocity + 0.05, role="follower",
                 follows=(src.name, k, c - base))


def _ram(spec: MachineSpec, name: str, link_a: str, pa, link_b: str, pb, driver: str,
         r: float, masses: tuple[float, float], ys: tuple[float, ...] = (0.0,),
         barrel_mat: str = "body") -> tuple[float, float]:
    """Hydraulic cylinder(s) from pin ``pa`` on ``link_a`` to pin ``pb`` on ``link_b`` (x-z plane).

    ``<name>_barrel`` pivots about pa and ``<name>_rod`` about pb, both followers
    of ``driver``; ``ys`` places parallel cylinders at those lateral offsets (the
    two boom cylinders).  The barrel follows a line fit of the true pa -> pb
    direction and the rod points back along the same line, so the pins are exact
    and the fit error only shows as a sideways offset between barrel and rod.
    Returns that offset (m) in the init pose and its worst case over the range.
    """
    jt = spec.joint(driver)
    qs, th, psi, ln = _samples(jt), [], [], []
    for q in qs:
        tfs = K.link_transforms(spec, {driver: q})
        ta, tb = tfs[link_a], tfs[link_b]
        rt = K.transpose(ta.R)
        v = K.vsub(K.mat_vec(rt, K.vsub(tb.apply(pb), ta.t)), pa)
        rab = K.mat_mul(rt, tb.R)
        th.append(math.atan2(v[2], v[0]))
        psi.append(math.atan2(rab[2][0], rab[0][0]))   # link_b's +x elevation in link_a (linear)
        ln.append(math.hypot(v[0], v[2]))
    k, c = _fit(qs, th, jt.init)
    psi = _unwrap(psi)
    kp = (psi[-1] - psi[0]) / (qs[-1] - qs[0])
    cp = psi[0] - kp * qs[0]
    l_lo, l_hi = min(ln), max(ln)
    b_len = l_lo - 0.08
    r_len = l_hi - b_len + 0.08                        # rod still 8 cm inside at full stroke
    assert r_len < l_lo - 0.02, f"{name}: stroke too long for its closed length"

    barrel_shapes, rod_shapes = [], []
    for y in ys:
        barrel_shapes += [
            cyl(r, b_len, at=(b_len / 2, y, 0.0), axis="x", mat=barrel_mat, collision=False),
            cyl(r * 1.25, 0.06, at=(b_len - 0.03, y, 0.0), axis="x", mat="accent",
                collision=False),                                                   # gland
            cyl(r * 1.12, 0.07, at=(0.035, y, 0.0), axis="x", mat="accent",
                collision=False),                                                   # base cap
            cyl(r * 0.9, r * 2.2, at=(0.0, y, 0.0), axis="y", mat="steel",
                collision=False),                                                   # base eye
            box(r * 1.2, r * 1.0, r * 0.8, at=(b_len * 0.12, y, r * 1.0), mat="steel",
                collision=False),                                                   # ports
            box(r * 1.2, r * 1.0, r * 0.8, at=(b_len - 0.16, y, r * 1.0), mat="steel",
                collision=False),
        ]
        rod_shapes += [
            cyl(r * 0.55, r_len, at=(r_len / 2, y, 0.0), axis="x", mat="steel", collision=False),
            cyl(r * 0.75, r * 2.0, at=(0.0, y, 0.0), axis="y", mat="steel", collision=False),  # eye
        ]
    barrel = Link(f"{name}_barrel", masses[0], barrel_shapes)
    rod_link = Link(f"{name}_rod", masses[1], rod_shapes)
    spec.links += [barrel, rod_link]
    spec.joints += [_follower(f"{name}_barrel_joint", link_a, barrel.name, pa, jt, k, c),
                    _follower(f"{name}_rod_joint", link_b, rod_link.name, pb, jt, k - kp,
                              c + math.pi - cp)]
    off = [l * math.sin(t - (k * q + c)) for l, t, q in zip(ln, _unwrap(th), qs)]
    return abs(off[qs.index(jt.init)]), max(abs(o) for o in off)


def _bucket_linkage(spec: MachineSpec) -> tuple[float, float]:
    """Idler link (pivoting on the stick) and H-link (pivoting on the bucket) followers.

    The four-bar is solved exactly over the bucket range; each link's angle is
    then line-fitted like the cylinders, so the two links meet at their common
    pin (the bucket-cylinder rod eye) within the fit error.  Returns that gap (m)
    in the init pose and its worst case.
    """
    bj = spec.joint("bucket_joint")
    (sx, sz), (lx, lz) = IDLER_PIN, LINK_PIN
    r_l, a_l = math.hypot(lx, lz), math.atan2(lz, lx)
    qs, th_d, th_h = _samples(bj), [], []
    for q in qs:
        px, pz = STICK_LEN + r_l * math.cos(a_l - q), r_l * math.sin(a_l - q)   # link pin
        d = math.hypot(px - sx, pz - sz)
        ex, ez = (px - sx) / d, (pz - sz) / d
        a = (IDLER_LEN ** 2 - HLINK_LEN ** 2 + d * d) / (2 * d)
        h = math.sqrt(IDLER_LEN ** 2 - a * a)
        jx, jz = sx + a * ex - h * ez, sz + a * ez + h * ex                     # common pin
        th_d.append(math.atan2(jz - sz, jx - sx))
        th_h.append(math.atan2(jz - pz, jx - px) + q)                            # bucket frame
    fit_d, fit_h = _fit(qs, th_d, bj.init), _fit(qs, th_h, bj.init)
    idler = Link("bucket_idler", 45.0, [
        *[box(IDLER_LEN + 0.09, 0.03, 0.09, at=(IDLER_LEN / 2, y, 0.0), mat="accent",
              collision=False) for y in (0.30, -0.30)],                          # side plates
        cyl(0.05, 0.66, at=(IDLER_LEN, 0.0, 0.0), axis="y", mat="steel", collision=False),
    ])
    hlink = Link("bucket_hlink", 50.0, [
        *[box(HLINK_LEN + 0.09, 0.03, 0.11, at=(HLINK_LEN / 2, y, 0.0), mat="dark",
              collision=False) for y in (0.14, -0.14)],
        box(0.12, 0.28, 0.07, at=(HLINK_LEN * 0.55, 0.0, 0.0), mat="dark", collision=False),
        cyl(0.05, 0.40, axis="y", mat="steel", collision=False),
    ])
    spec.links += [idler, hlink]
    spec.joints += [
        _follower("bucket_idler_joint", "stick", idler.name, (sx, 0.0, sz), bj, *fit_d),
        _follower("bucket_hlink_joint", "bucket", hlink.name, (lx, 0.0, lz), bj, *fit_h),
    ]
    gaps = []
    for q, ang_d, ang_h in zip(qs, (fit_d[0] * q + fit_d[1] for q in qs),
                               (fit_h[0] * q + fit_h[1] - q for q in qs)):
        px, pz = STICK_LEN + r_l * math.cos(a_l - q), r_l * math.sin(a_l - q)
        gaps.append(math.hypot(sx + IDLER_LEN * math.cos(ang_d) - (px + HLINK_LEN * math.cos(ang_h)),
                               sz + IDLER_LEN * math.sin(ang_d) - (pz + HLINK_LEN * math.sin(ang_h))))
    return gaps[qs.index(bj.init)], max(gaps)


# ---------------------------------------------------------------------------
# Visual detail blocks
# ---------------------------------------------------------------------------

def _track(y: float) -> list[Shape]:
    """Crawler track: belt runs + end wraps + frame (collide), rollers, idler, lugs (visual)."""
    straight = TRACK_L - TRACK_H
    r, belt = TRACK_H / 2, 0.10
    shapes = [
        box(straight, SHOE_W, belt, at=(0.0, y, -r + belt / 2), mat="rubber"),       # ground run
        box(straight, SHOE_W, belt, at=(0.0, y, r - belt / 2), mat="rubber"),        # top run
        cyl(r, SHOE_W, at=(straight / 2, y, 0.0), axis="y", mat="rubber"),            # idler wrap
        cyl(r, SHOE_W, at=(-straight / 2, y, 0.0), axis="y", mat="rubber"),           # sprocket wrap
        box(straight * 0.92, 0.34, TRACK_H - 2 * belt - 0.04, at=(0.0, y, 0.0), mat="dark"),  # track frame
        cyl(r - 0.07, SHOE_W + 0.02, at=(straight / 2, y, 0.0), axis="y", mat="steel",
            collision=False),                                                         # front idler
        cyl(0.15, SHOE_W + 0.06, at=(straight / 2, y, 0.0), axis="y", mat="dark",
            collision=False),                                                         # idler hub
        # final-drive housing hidden inside the sprocket (keeps the track's mass centred)
        cyl(r - 0.07, SHOE_W - 0.02, at=(-straight / 2, y, 0.0), axis="y", mat="dark",
            collision=False),
        cyl(0.15, SHOE_W + 0.02, at=(-straight / 2, y, 0.0), axis="y", mat="dark",
            collision=False),
    ]
    for i in range(7):                                                                # bottom rollers
        x = -1.35 + 0.45 * i
        shapes += [cyl(0.095, 0.50, at=(x, y, -r + belt + 0.095), axis="y", mat="steel",
                       collision=False),
                   cyl(0.12, 0.05, at=(x, y + 0.255, -r + belt + 0.095), axis="y", mat="dark",
                       collision=False),                                              # outer flange
                   cyl(0.12, 0.05, at=(x, y - 0.255, -r + belt + 0.095), axis="y", mat="dark",
                       collision=False)]
    shapes += [cyl(0.075, 0.44, at=(x, y, r - belt - 0.075), axis="y", mat="steel",
                   collision=False) for x in (-0.55, 0.55)]                           # carrier rollers
    for i in range(18):                                                               # tread lugs
        x = -1.70 + 0.20 * i
        shapes += [box(0.07, SHOE_W + 0.02, 0.06, at=(x, y, -r + 0.03), mat="dark",
                       collision=False),
                   box(0.07, SHOE_W + 0.02, 0.06, at=(x, y, r - 0.03), mat="dark",
                       collision=False)]
    return shapes


def _sprocket(name: str) -> Link:
    """Rotating drive sprocket: rim, hub, spokes and nine teeth (visual; the track collides)."""
    r, w = TRACK_H / 2, 0.62
    shapes = [
        cyl(r, w, axis="y", mat="dark", collision=False),
        cyl(0.30, w + 0.02, axis="y", mat="steel", collision=False),
        cyl(0.13, w + 0.04, axis="y", mat="dark", collision=False),
        box(r * 1.7, w + 0.02, 0.10, mat="steel", collision=False),
        box(0.10, w + 0.02, r * 1.7, mat="steel", collision=False),
    ]
    for i in range(9):
        a = 2 * math.pi * i / 9
        rho = r - 0.03
        shapes.append(box(0.12, w + 0.02, 0.05, at=(rho * math.cos(a), 0.0, rho * math.sin(a)),
                          rpy=(0.0, math.pi / 2 - a, 0.0), mat="steel", collision=False))
    return Link(name, 25.0, shapes)


def _counterweight() -> list[Shape]:
    """Two-tier counterweight as facets inscribed in the tail-swing arc (house frame).

    Each facet's outer corners lie on a circle about the swing axis (2 cm inside
    TAIL_R for the lower tier, 0.11 m less for the upper one), so nothing reaches
    past the datasheet tail-swing radius; the inner halves are buried in the hood.
    Tail lamps (outer facets) and the radiator grille (centre facet) sit flush
    on the lower tier.
    """
    shapes = []
    n = 5
    r_cw = TAIL_R - 0.02                                         # lamps / grille stand 1 cm proud
    span = math.degrees(math.asin(1.27 / r_cw))                 # the facets cover the 2.54 m deck width
    edges = [-span + 2 * span * i / n for i in range(n + 1)]    # degrees from -x towards +y
    for rr, z0, z1 in ((r_cw, -0.17, 0.85), (r_cw - 0.11, 0.85, 1.20)):
        for i, (a0, a1) in enumerate(zip(edges, edges[1:])):
            mid, half = math.radians((a0 + a1) / 2), math.radians((a1 - a0) / 2)
            face = rr * math.cos(half)                   # outer face distance from the swing axis
            depth = 0.65
            r_c = face - depth / 2
            c, s_ = -math.cos(mid), math.sin(mid)        # outward normal
            shapes.append(box(depth, 2 * rr * math.sin(half), z1 - z0,
                              at=(r_c * c, r_c * s_, (z0 + z1) / 2), rpy=(0.0, 0.0, -mid),
                              mat="dark"))
            if rr == r_cw and i in (0, n - 1):         # tail lamps, low on the outer facets
                shapes.append(box(0.03, 0.14, 0.10, at=((face - 0.005) * c, (face - 0.005) * s_, 0.28),
                                  rpy=(0.0, 0.0, -mid), mat="red", collision=False))
            if rr == r_cw and i == n // 2:             # radiator grille on the centre facet
                shapes.append(box(0.03, 0.60, 0.34, at=((face - 0.005) * c, 0.0, 0.52), rpy=(0.0, 0.0, -mid),
                                  mat="accent", collision=False))
                shapes += [box(0.02, 0.58, 0.02, at=((face - 0.008) * c, 0.0, 0.40 + 0.06 * k),
                               rpy=(0.0, 0.0, -mid), mat="steel", collision=False)
                           for k in range(5)]
    return shapes


def _cab() -> list[Shape]:
    """Operator cab on the left front of the house: black frame, glass all round, yellow roof."""
    x0, x1, y0, y1 = CAB
    z0, zr, z1 = CAB_Z
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    lx, ly = x1 - x0, y1 - y0
    return [
        box(lx, ly, zr - z0, at=(cx, cy, (z0 + zr) / 2), mat="accent"),               # frame (collides)
        box(lx + 0.16, ly + 0.04, 0.07, at=(cx + 0.04, cy, zr + 0.035)),               # roof
        box(0.55, 0.45, 0.04, at=(cx - 0.40, cy, z1 - 0.02), mat="dark", collision=False),  # roof vent
        # windscreen (upper + lower pane) and mid rail
        box(0.03, ly - 0.10, 0.80, at=(x1 + 0.01, cy, 1.18), mat="glass", collision=False),
        box(0.03, ly - 0.10, 0.34, at=(x1 + 0.01, cy, 0.58), mat="glass", collision=False),
        box(0.04, ly - 0.04, 0.04, at=(x1 + 0.015, cy, 0.765), mat="accent", collision=False),
        *[box(0.05, 0.06, 1.28, at=(x1 + 0.025, y, 1.00), mat="accent", collision=False)
          for y in (y0 + 0.03, y1 - 0.03)],                                            # A pillars
        # left side: fixed pane, door glass, rear pane, door frame, skirt panels, handle
        box(0.37, 0.03, 0.80, at=(0.765, y1 + 0.01, 1.18), mat="glass", collision=False),
        box(0.57, 0.03, 0.80, at=(0.235, y1 + 0.01, 1.18), mat="glass", collision=False),
        box(0.28, 0.03, 0.80, at=(-0.26, y1 + 0.01, 1.18), mat="glass", collision=False),
        box(0.05, 0.04, 1.26, at=(0.555, y1 + 0.02, 1.00), mat="accent", collision=False),
        box(0.04, 0.04, 1.26, at=(-0.08, y1 + 0.02, 1.00), mat="accent", collision=False),
        box(0.63, 0.04, 0.04, at=(0.235, y1 + 0.02, 0.775), mat="accent", collision=False),
        box(0.63, 0.02, 0.38, at=(0.235, y1 + 0.01, 0.575), collision=False),           # door panel
        box(0.42, 0.02, 0.40, at=(0.79, y1 + 0.01, 0.565), collision=False),
        box(0.12, 0.04, 0.03, at=(-0.02, y1 + 0.04, 0.84), mat="steel", collision=False),  # handle
        *[box(0.03, 0.04, 0.09, at=(0.57, y1 + 0.04, z), mat="steel", collision=False)
          for z in (0.46, 1.50)],                                                      # hinges
        cyl(0.015, 0.42, at=(0.64, y1 + 0.075, 1.00), axis="z", mat="accent", collision=False),
        *[box(0.03, 0.05, 0.03, at=(0.64, y1 + 0.045, z), mat="accent", collision=False)
          for z in (0.82, 1.18)],                                                      # grab bar
        # right side
        box(0.95, 0.03, 0.80, at=(0.22, y0 - 0.01, 1.18), mat="glass", collision=False),
        box(1.35, 0.02, 0.40, at=(0.275, y0 - 0.01, 0.565), collision=False),
        # rear wall with a small window
        box(0.02, ly - 0.04, 1.20, at=(x0 - 0.01, cy, 1.00), collision=False),
        box(0.03, 0.50, 0.40, at=(x0 - 0.01, cy, 1.25), mat="glass", collision=False),
        # work lights hanging under the roof edge (front and rear), mirrors
        *[box(0.08, 0.16, 0.08, at=(x1 + 0.09, y, 1.60), mat="white", collision=False)
          for y in (0.50, 1.04)],
        *[box(0.08, 0.16, 0.08, at=(x0 - 0.07, y, 1.60), mat="white", collision=False)
          for y in (0.50, 1.04)],
        box(0.03, 0.16, 0.03, at=(x1 + 0.03, y1 + 0.08, 1.45), mat="accent", collision=False),
        box(0.04, 0.04, 0.22, at=(x1 + 0.03, y1 + 0.17, 1.40), mat="accent", collision=False),
        box(0.03, 0.14, 0.03, at=(x1 + 0.03, y0 - 0.07, 1.45), mat="accent", collision=False),
        box(0.04, 0.04, 0.20, at=(x1 + 0.03, y0 - 0.15, 1.40), mat="accent", collision=False),
        # cab step on the left of the deck
        box(0.55, 0.22, 0.04, at=(0.30, 1.38, 0.14), mat="steel", collision=False),
        *[box(0.04, 0.12, 0.12, at=(x, 1.31, 0.08), mat="dark", collision=False)
          for x in (0.08, 0.52)],
    ]


# ---------------------------------------------------------------------------
# Machine
# ---------------------------------------------------------------------------

def build() -> MachineSpec:
    y_t = GAUGE / 2
    cab_x0, cab_x1, cab_y0, cab_y1 = CAB
    undercarriage = Link("undercarriage", 7740.0, [
        *_track(+y_t),
        *_track(-y_t),
        box(2.2, GAUGE - SHOE_W, 0.50, at=(0.0, 0.0, 0.245), mat="dark"),      # car body
        cyl(0.90, 0.25, at=(0.0, 0.0, 0.62), mat="steel"),                       # swing bearing
        cyl(0.97, 0.05, at=(0.0, 0.0, 0.69), mat="dark", collision=False),       # slew-ring flange
        box(1.80, 1.50, 0.04, at=(0.0, 0.0, -0.02), mat="steel", collision=False),  # belly plate
        *[box(0.34, 0.30, 0.46, at=(x, s * 0.74, 0.0), mat="dark", collision=False)
          for x in (-1.75, 1.75) for s in (1, -1)],                              # final drives / idler yokes
        *[box(0.45, 0.10, 0.12, at=(0.0, s * 0.95, 0.46), mat="steel", collision=False)
          for s in (1, -1)],                                                     # track guides
    ])

    # Upper structure: z = 0 is the top of the swing bearing.
    ear_y = [BOOM_Y + s * 0.31 for s in (1, -1)]
    house = Link("upper_structure", 8760.0, [
        box(3.50, 2.54, 0.35, at=(-0.45, 0.0, 0.175)),                               # deck
        box(1.80, 2.54, 1.00, at=(-1.40, 0.0, 0.85)),                                # engine hood
        *_counterweight(),
        box(1.50, 0.82, 0.80, at=(0.25, -0.86, 0.75)),                               # tank / toolbox
        *_cab(),
        # boom foot: twin ears carry the foot pin, base block under it
        *[box(0.28, 0.05, 0.52, at=(0.94, y, 0.49), mat="dark") for y in ear_y],
        *[cyl(0.26, 0.05, at=(BOOM_FOOT[0], y, BOOM_FOOT[2]), axis="y", mat="dark")
          for y in ear_y],
        box(0.62, 0.62, 0.34, at=(0.89, BOOM_Y, 0.30), mat="dark"),
        box(0.14, 0.62, 0.30, at=(1.05, BOOM_Y, 0.55), mat="dark", collision=False),
        # boom-cylinder base pins on pedestals at the front of the frame
        *[box(0.24, 0.26, 0.18, at=(BOOM_CYL[0][0], BOOM_Y + s * BOOM_CYL_Y, 0.30), mat="dark",
              collision=False) for s in (1, -1)],
        cyl(0.045, 2 * BOOM_CYL_Y + 0.22, at=(BOOM_CYL[0][0], BOOM_Y, BOOM_CYL[0][2]), axis="y",
            mat="steel", collision=False),
        # engine hood: top grille with louvres, side doors with louvres and handles
        box(0.90, 0.90, 0.02, at=(-1.40, 0.0, 1.36), mat="accent", collision=False),
        *[box(0.03, 0.88, 0.02, at=(-1.40 + 0.16 * i, 0.0, 1.375), mat="steel", collision=False)
          for i in range(-2, 3)],
        *[s for y in (-1.285, 1.285) for s in (
            box(1.30, 0.02, 0.62, at=(-1.40, y, 0.85), mat="accent", collision=False),
            *[box(1.16, 0.025, 0.025, at=(-1.40, y * 1.004, 0.66 + 0.075 * i), mat="steel",
                  collision=False) for i in range(6)],
            box(0.10, 0.03, 0.04, at=(-0.85, y * 1.007, 0.85), mat="steel", collision=False))],
        # exhaust: muffler + stack, air cleaner, beacon
        cyl(0.12, 0.32, at=(-1.90, -0.85, 1.51), mat="dark", collision=False),
        cyl(0.05, 0.10, at=(-1.90, -0.85, 1.69), mat="accent", collision=False),
        cyl(0.10, 0.30, at=(-0.80, -1.05, 1.50), mat="accent", collision=False),
        cyl(0.07, 0.10, at=(-2.00, 0.90, 1.40), mat="orange", collision=False),
        # tank: access door, filler cap, sight gauge
        box(1.00, 0.02, 0.50, at=(0.25, -1.28, 0.75), mat="accent", collision=False),
        box(0.10, 0.03, 0.04, at=(-0.20, -1.295, 0.75), mat="steel", collision=False),
        cyl(0.07, 0.04, at=(-0.20, -0.90, 1.17), mat="accent", collision=False),
        box(0.03, 0.03, 0.30, at=(0.75, -1.285, 0.80), mat="glass", collision=False),
        # handrails on the right-hand tank and along the hood edge
        *[cyl(0.02, 0.45, at=(x, -1.22, 1.375), axis="z", mat="accent", collision=False)
          for x in (-0.40, 0.30, 0.95)],
        cyl(0.02, 1.35, at=(0.275, -1.22, 1.60), axis="x", mat="accent", collision=False),
        cyl(0.02, 1.35, at=(0.275, -1.22, 1.40), axis="x", mat="accent", collision=False),
        *[cyl(0.02, 0.34, at=(x, -1.22, 1.53), axis="z", mat="accent", collision=False)
          for x in (-0.80, -2.00)],
        cyl(0.02, 1.20, at=(-1.40, -1.22, 1.70), axis="x", mat="accent", collision=False),
        # camera housings (the lenses sit just ahead of them)
        box(0.09, 0.06, 0.07, at=(cab_x1 + 0.07, cab_y1 + 0.07, 1.595), mat="dark", collision=False),
        box(0.09, 0.06, 0.07, at=(0.86, -1.22, 1.68), mat="dark", collision=False),
        cyl(0.02, 0.10, at=(0.95, -1.22, 1.62), axis="z", mat="accent", collision=False),
    ])
    tail = _plan_radius(house.shapes)
    assert tail <= TAIL_R + 0.005, f"upper structure reaches {tail:.3f} m > tail-swing radius {TAIL_R}"

    bx, bz = BOOM_CYL[1][0], BOOM_CYL[1][2]
    ax, az = ARM_CYL[0][0], ARM_CYL[0][2]
    kx, kz = BOOM_KNEE
    mx, mz = BOOM_MID
    boom = Link("boom", 1630.0, [
        beam((0.0, 0.0), BOOM_KNEE, 0.55, 0.70),
        beam(BOOM_KNEE, BOOM_MID, 0.55, 0.60),
        beam(BOOM_MID, BOOM_TIP, 0.46, NOSE_DEPTH),
        cyl(0.35, 0.56, at=(kx, 0.0, kz), axis="y"),                               # knee
        cyl(0.22, 0.70, axis="y", mat="steel", collision=False),                     # foot pin boss
        *[cyl(0.12, 0.04, at=(0.0, s * 0.355, 0.0), axis="y", mat="dark", collision=False)
          for s in (1, -1)],                                                         # pin retainers
        cyl(0.19, 0.50, at=(BOOM_TIP[0], 0.0, BOOM_TIP[1]), axis="y", mat="steel",
            collision=False),                                                        # nose boss
        # boom-cylinder lug block under the boom with ear plates, the rod-eye pin through them
        box(0.36, 0.64, 0.16, at=(bx, 0.0, bz + 0.10), rpy=(0.0, -0.5136, 0.0),
            collision=False),
        *[beam((bx - 0.13, bz + 0.12), (bx, bz), 0.05, 0.20, y=s * 0.31, collision=False)
          for s in (1, -1)],
        cyl(0.055, 2 * BOOM_CYL_Y + 0.22, at=(bx, 0.0, bz), axis="y", mat="steel", collision=False),
        # stick-cylinder bracket over the knee (two A-frame plates) and pin
        *[s for y in (0.12, -0.12) for s in (
            beam((2.95, 1.84), (ax, az), 0.05, 0.18, y=y, collision=False),
            beam((4.15, 1.62), (ax, az), 0.05, 0.18, y=y, collision=False))],
        cyl(0.07, 0.40, at=(ax, 0.0, az), axis="y", mat="steel", collision=False),
        # hydraulic lines along the top of the boom, work light on the left
        *[rod(_boom_top(0.6), _boom_top(3.0), 0.025, y=y, mat="accent") for y in (0.17, -0.17)],
        box(0.14, 0.10, 0.12, at=(1.28, 0.16, 1.23), rpy=(0.0, -0.51, 0.0), mat="white",
            collision=False),
    ])

    ex, ez = ARM_CYL[1][0], ARM_CYL[1][2]
    bcx, bcz = BKT_CYL_BASE[0], BKT_CYL_BASE[2]
    stick = Link("stick", 780.0, [
        box(1.00, 0.45, STICK_TOP + 0.30, at=(0.40, 0.0, (STICK_TOP - 0.30) / 2)),  # deep root
        box(STICK_LEN - 0.90, 0.45, 2 * STICK_TOP, at=((STICK_LEN + 0.90) / 2, 0.0, 0.0)),
        cyl(0.18, 0.62, axis="y", mat="steel", collision=False),                    # pin boss
        cyl(0.14, 0.50, at=(STICK_LEN, 0.0, 0.0), axis="y", mat="steel", collision=False),
        # tail ears carrying the stick-cylinder rod eye
        *[beam((0.05, 0.0), (ex, ez), 0.05, 0.26, y=s * 0.15, collision=False) for s in (1, -1)],
        cyl(0.08, 0.40, at=(ex, 0.0, ez), axis="y", mat="steel", collision=False),
        # bucket-cylinder base gussets on top of the stick and the idler-link pin through it
        *[s for y in (0.14, -0.14) for s in (
            beam((-0.05, STICK_TOP), (bcx, bcz), 0.05, 0.10, y=y, collision=False),
            beam((0.55, STICK_TOP), (bcx, bcz), 0.05, 0.10, y=y, collision=False))],
        cyl(0.06, 0.34, at=(bcx, 0.0, bcz), axis="y", mat="steel", collision=False),
        cyl(0.065, 0.66, at=(IDLER_PIN[0], 0.0, IDLER_PIN[1]), axis="y", mat="steel",
            collision=False),
        # ribs along the top and the belly of the stick
        beam((0.8, STICK_TOP + 0.01), (2.2, STICK_TOP + 0.01), 0.10, 0.02, collision=False),
        beam((0.9, -STICK_TOP - 0.01), (2.5, -STICK_TOP - 0.01), 0.10, 0.02, collision=False),
    ])

    # Bucket frame: origin at the bucket pin, +x toward the cutting edge.  With the
    # stick hanging down, +z points away from the machine: the bucket's curved
    # back bulges that way and its mouth faces the machine (backhoe layout), so
    # curling pulls the teeth toward the cab and turns the mouth up.
    lx, lz = LINK_PIN
    bucket = Link("bucket", 700.0, [
        beam((-0.02, 0.06), (0.32, 0.44), 1.20, 0.06, mat="dark"),                  # back plate
        beam((0.32, 0.44), (0.86, 0.48), 1.20, 0.06, mat="dark"),
        beam((0.86, 0.48), (1.20, 0.22), 1.20, 0.06, mat="dark"),                    # heel
        beam((1.20, 0.22), (1.30, -0.10), 1.20, 0.06, mat="dark"),                   # lip
        *[box(1.22, 0.05, 0.50, at=(0.64, y, 0.22), mat="dark") for y in (0.60, -0.60)],  # sides
        box(0.30, 0.50, 0.16, at=(0.05, 0.0, 0.10), mat="steel", collision=False),  # pin lugs
        # H-link ears over the back, back-plate ribs, side cutters, five teeth on adapters
        *[beam((0.02, 0.10), (lx, lz), 0.05, 0.14, y=y, mat="dark", collision=False)
          for y in (0.20, -0.20)],
        cyl(0.05, 0.50, at=(lx, 0.0, lz), axis="y", mat="steel", collision=False),
        *[beam((0.36, 0.47), (0.84, 0.51), 0.04, 0.05, y=y, mat="steel", collision=False)
          for y in (-0.40, 0.0, 0.40)],
        *[box(0.08, 0.08, 0.20, at=(1.31, y, -0.04), mat="steel", collision=False)
          for y in (0.62, -0.62)],
        *[box(0.14, 0.12, 0.12, at=(1.28, y, -0.07), mat="dark", collision=False)
          for y in (-0.48, -0.24, 0.0, 0.24, 0.48)],
        *[box(0.08, 0.10, 0.16, at=(1.33, y, -0.18), mat="steel", collision=False)  # teeth
          for y in (-0.48, -0.24, 0.0, 0.24, 0.48)],
    ])

    sprocket_x = -(TRACK_L - TRACK_H) / 2
    links = [undercarriage, house, boom, stick, bucket,
             _sprocket("left_sprocket"), _sprocket("right_sprocket")]

    joints = [
        Joint("swing_joint", "continuous", "undercarriage", "upper_structure",
              xyz=(0.0, 0.0, SWING_Z), axis=(0, 0, 1), velocity=1.15, role="tool"),
        Joint("boom_joint", "revolute", "upper_structure", "boom",
              xyz=BOOM_FOOT, axis=(0, -1, 0), lower=-0.90, upper=1.00, init=-0.45,
              velocity=0.55, role="tool"),
        Joint("stick_joint", "revolute", "boom", "stick",
              xyz=(BOOM_TIP[0], 0.0, BOOM_TIP[1]), rpy=(0.0, STICK_PITCH, 0.0),
              axis=(0, -1, 0), lower=-1.35, upper=0.90, init=-1.35, velocity=0.65, role="tool"),
        Joint("bucket_joint", "revolute", "stick", "bucket",
              xyz=(STICK_LEN, 0.0, 0.0), axis=(0, 1, 0), lower=-1.20, upper=1.75, init=BUCKET_INIT,
              velocity=1.0, role="tool"),
        Joint("left_sprocket_joint", "continuous", "undercarriage", "left_sprocket",
              xyz=(sprocket_x, y_t, 0.0), axis=(0, 1, 0), velocity=20.0, role="roll_left",
              wheel_radius=TRACK_H / 2),
        Joint("right_sprocket_joint", "continuous", "undercarriage", "right_sprocket",
              xyz=(sprocket_x, -y_t, 0.0), axis=(0, 1, 0), velocity=20.0, role="roll_right",
              wheel_radius=TRACK_H / 2),
    ]

    spec = MachineSpec(
        name="hydraulic_excavator",
        abbrev="HEX",
        display_name="Hydraulic Excavator",
        category="excavation",
        reference="20 t crawler excavator (CAT 320 / Komatsu PC210 class)",
        description="Tracked excavator with 360° swing, two-piece boom/stick arm and a 1 m³ bucket.",
        root="undercarriage",
        links=links,
        joints=joints,
        drive=Drive(type="skid", max_speed=1.5, max_yaw_rate=0.5, track_width=GAUGE),
        tools=[
            Tool("swing_joint", "swing", positive="left", negative="right"),
            Tool("boom_joint", "boom", positive="raise", negative="lower"),
            Tool("stick_joint", "stick", positive="out", negative="in"),
            Tool("bucket_joint", "bucket", positive="curl", negative="dump"),
        ],
        cameras=[
            # operator eye point, just ahead of the windscreen (URDF glass is opaque)
            Camera("front", "upper_structure", (cab_x1 + 0.12, 0.77, 1.45), (0.0, 0.35, 0.0)),
            # front-left corner of the cab roof, looking across the front at the arm
            Camera("left", "upper_structure", (cab_x1 + 0.14, cab_y1 + 0.03, 1.58), (0.0, 0.30, -0.28)),
            # front post of the tank handrail (above the top rail), looking across at the arm
            Camera("right", "upper_structure", (0.95, -1.22, 1.68), (0.0, 0.30, 0.28)),
        ],
        envelope=(9.53, 2.98, 2.96),
        operating_mass=20500.0,
        ground_contact=["undercarriage"],
        # The stick folds tight under the boom in the shipping pose and the
        # bucket tucks in front of the tracks; real machines do the same.
        allowed_overlaps=[("boom", "bucket")],
        notes=("Bucket curl is not auto-verified (no up/out/left semantics): +1 pulls the "
               "teeth toward the machine and turns the mouth up (checked in the sweep render). "
               "Cylinder and linkage followers are cosmetic."),
    )

    # Hydraulics and bucket linkage (visual followers; see the module docstring).
    stats = {
        "boom_cyl": _ram(spec, "boom_cyl", "upper_structure", BOOM_CYL[0], "boom", BOOM_CYL[1],
                         "boom_joint", 0.10, (240.0, 110.0),
                         ys=(BOOM_CYL_Y, -BOOM_CYL_Y)),
        "arm_cyl": _ram(spec, "arm_cyl", "boom", ARM_CYL[0], "stick", ARM_CYL[1], "stick_joint",
                        0.105, (150.0, 80.0)),
    }
    gap = _bucket_linkage(spec)
    stats["bucket_cyl"] = _ram(spec, "bucket_cyl", "stick", BKT_CYL_BASE, "bucket_idler",
                               (IDLER_LEN, 0.0, 0.0), "bucket_joint", 0.095, (110.0, 55.0))
    for name, (at_init, worst) in stats.items():
        lim_init, lim_worst = MAX_OFFSET[name]
        assert at_init <= lim_init and worst <= lim_worst, \
            f"{name}: barrel/rod offset {at_init:.3f} m (init) / {worst:.3f} m (worst)"
    assert gap[0] <= MAX_LINKAGE_GAP[0] and gap[1] <= MAX_LINKAGE_GAP[1], \
        f"bucket linkage gap {gap[0]:.3f} m (init) / {gap[1]:.3f} m (worst)"
    clear = _arm_cyl_clearance(spec)
    assert clear >= 0.058, f"stick cylinder axis is only {clear:.3f} m from the boom"   # rod radius
    low = K.bounding_box(spec)[0][2] + K.root_height(spec)
    assert low > -0.005, f"visual shapes reach {low:.3f} m below the ground in the init pose"
    assert abs(spec.total_mass - spec.operating_mass) < 1e-6, spec.total_mass
    return spec
