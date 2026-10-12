"""Mini excavator (MEX), 3.5-4 t class — modelled on a CAT 303.5 / Kubota KX040-4.

Reference numbers (typical 3.5-4 t rubber-track mini excavator datasheet;
values in brackets are what this model measures):
    operating mass 3.8 t with cab and rubber tracks [3.82 t] · transport
    length 4.85 m [4.83] · width 1.78 m over 300 mm rubber tracks, gauge
    1.48 m [1.82 incl. sprocket hubs] · height 2.50 m to the cab roof [2.50] ·
    crawler length 2.13 m, track height 0.46 m · ground clearance 0.29 m ·
    counterweight clearance 0.56 m · short tail swing, tail-swing radius
    0.98 m [0.98] · dozer blade 1.78 x 0.36 m, lift 0.39 m above / 0.40 m
    below ground · boom 2.43 m pin to pin on a boom-swing bracket, boom swing
    75 deg left / 60 deg right · stick 1.35 m · 0.12 m3 600 mm bucket, tip
    radius 0.70 m · max dig depth 3.1 m [3.06] · max reach at ground 5.2 m
    [5.18] · max dig height 4.9 m [4.78] · max dump clearance 3.5 m [4.0,
    lowest point of the fully dumped bucket at full boom raise and stick out,
    where the dump stop (relative to the stick) leaves the cutting edge
    pointing up rather than down] · swing 9.5 rpm · travel 4.6 km/h high
    range (1.28 m/s) · counter-rotation about 0.6 rad/s.

House front and blade: the deck's front corners are chamfered 0.20 m and the
lower swing-post ear sits on top of the deck, so below the cab floor (0.70 m
above ground) nothing of the house reaches beyond r = 1.03 m of the swing
axis (front swing radius above it: cab / tank corners 1.08 m, roof 1.12 m).
The blade's cutting edge sits 1.25 m ahead of the swing axis (0.19 m ahead
of the idlers); over the whole blade range the blade keeps >= 3.7 cm radial
clearance from the house, so the house slews freely over the raised blade.

Joint ranges and full-command rates (q = 0 is the geometric reference):
    swing continuous, 1.0 rad/s · boom_swing -1.05..+1.31 rad (-60..+75 deg),
    0.45 rad/s · boom -1.16..+1.12 rad (boom pin line -57..+74 deg from
    horizontal), 0.75 rad/s · stick -1.05..+0.85 rad (boom/stick included
    angle 54..163 deg), 0.85 rad/s · bucket -1.05 (dump)..+2.15 (curl) rad,
    1.3 rad/s · blade -0.39..+0.35 rad (cutting edge 0.40 m below to 0.39 m
    above ground), 0.35 rad/s.

Hydraulics and bucket linkage are visual follower links (no collision).
Each cylinder is a barrel pinned on one link and a rod pinned on the other,
both revolute followers whose angle is a minimax line fit of the true
pin-to-pin direction with the init pose weighted 5x, so both pins are
always exact and the only error is a sideways offset between barrel and rod:
<= 1.4 cm in the init pose; worst case boom 6.9, stick 6.3, bucket 4.9,
blade 2.6 cm at the range ends.  The bucket is driven through a dog-bone
(on the stick) / H-link (on the bucket) four-bar whose two links are fitted
the same way and meet within 1.4 cm in the init pose, 7.4 cm at full dump.

Init pose = travel/transport pose: blade fully raised, boom slightly above
horizontal, stick fully in and bucket fully curled under the boom knee.

Root frame: centre of the undercarriage at track mid-height (0.23 m above
ground).  The cab sits on the LEFT of the house, the boom is on the centre
line on a swing bracket at the front, the engine hood / counterweight form a
rounded short-tail-swing rear and the hydraulic tank compartment is on the
front right — as on the real machine class.
"""

from __future__ import annotations

import math

from .. import kinematics as K
from ..spec import (Camera, Drive, Joint, Link, MachineSpec, Shape, Tool, beam, box, cyl,
                    sprocket_link)

# Undercarriage (root frame: origin at the track mid-height, ground at -TRACK_H / 2)
TRACK_L, TRACK_H, SHOE_W, GAUGE = 2.13, 0.46, 0.30, 1.48
GROUND_Z = -TRACK_H / 2
SWING_Z = 0.33                     # top of the swing bearing = house floor (0.56 m above ground)
# Dozer blade: pivot on the front of the car body, bottom edge 1.25 m ahead of the swing axis
BLADE_PIVOT = (0.15, 0.0, 0.02)
BLADE_EDGE = (1.10, -0.25)         # bottom cutting edge in the blade frame at q = 0 (on the ground)
BLADE_W, BLADE_H = 1.78, 0.36
BLADE_LIFT, BLADE_DROP = 0.39, 0.40
# Upper structure (house frame: z = 0 is the house floor)
TAIL_R = 0.95                      # counterweight radius about the swing axis
DECK = (-0.40, 0.70, 0.86, 0.20)   # house floor plate: rear x, front x, half width, front chamfer
SWING_POST = (0.86, 0.0, 0.26)     # boom-swing king pin in the house frame
BOOM_FOOT = (0.20, 0.0, 0.20)      # boom foot pin in the swing-bracket frame
# Boom geometry in the boom frame (origin = foot pin, x forward when q = 0)
BOOM_KNEE = (1.25, 0.70)
BOOM_TIP = (2.40, 0.40)            # stick pin; 2.43 m from the foot pin
STICK_LEN = 1.35
STICK_PITCH = 1.40                 # stick points 80 deg below the boom frame x at q = 0
BUCKET_TIP = (0.70, -0.04)         # cutting edge in the bucket frame (tip radius 0.70 m)
BUCKET_W = 0.60
# Hydraulic cylinder pins: (base pin on the parent-side link, rod-eye pin on the driven link)
BOOM_CYL = ((0.14, 0.0, 0.0), (0.99, 0.0, 0.32))       # swing bracket -> lug under the boom
ARM_CYL = ((0.785, 0.0, 0.646), (-0.25, 0.0, 0.12))    # bracket on top of the boom -> stick top
BLADE_CYL_BASE = (0.50, 0.0, 0.20)                     # car-body front (rod eye on the blade)
BKT_CYL_BASE = (0.05, 0.0, 0.20)                       # stick top (rod eye on the dog-bone)
# Bucket four-bar: stick - dog-bone - H-link - bucket (x, z in the stick / bucket frame)
DOGBONE_PIN = (1.17, 0.02)         # dog-bone pin through the stick, 0.18 m behind the bucket pin
DOGBONE_LEN, HLINK_LEN = 0.31, 0.21
LINK_PIN = (0.08, 0.20)            # H-link pin on the bucket ears (bucket frame)
W_INIT = 4.0                       # extra weight of the init pose in the follower fits


def _unwrap(a: list[float]) -> list[float]:
    out = list(a)
    for i in range(1, len(out)):
        out[i] += 2 * math.pi * round((out[i - 1] - out[i]) / (2 * math.pi))
    return out


def _samples(jt: Joint, n: int = 40) -> list[float]:
    return sorted({jt.lower + (jt.upper - jt.lower) * i / n for i in range(n + 1)} | {jt.init})


def _fit(qs: list[float], ys: list[float], q_init: float) -> tuple[float, float]:
    """Line k*q + c minimising the largest residual; residuals at q_init weigh 1 + W_INIT."""
    ys = _unwrap(ys)
    ws = [1.0 + W_INIT * math.exp(-((q - q_init) / 0.05) ** 2) for q in qs]

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
         r: float, masses: tuple[float, float]) -> None:
    """Hydraulic cylinder from pin ``pa`` on ``link_a`` to pin ``pb`` on ``link_b`` (x-z plane).

    ``<name>_barrel`` pivots about pa and ``<name>_rod`` about pb, both followers
    of ``driver``.  The barrel follows a line fit of the true pa -> pb direction
    and the rod points back along the same line, so the pins are exact and the
    fit error only shows as a sideways offset between barrel and rod.
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
    barrel = Link(f"{name}_barrel", masses[0], [
        cyl(r, b_len, at=(b_len / 2, 0.0, 0.0), axis="x", mat="body", collision=False),
        cyl(r * 1.2, 0.05, at=(b_len - 0.025, 0.0, 0.0), axis="x", mat="accent",
            collision=False),                                                      # gland
        cyl(r * 0.9, r * 2.2, axis="y", mat="steel", collision=False),             # base eye
    ])
    rod = Link(f"{name}_rod", masses[1], [
        cyl(r * 0.55, r_len, at=(r_len / 2, 0.0, 0.0), axis="x", mat="steel", collision=False),
        cyl(r * 0.75, r * 2.0, axis="y", mat="steel", collision=False),            # rod eye
    ])
    spec.links += [barrel, rod]
    spec.joints += [_follower(f"{name}_barrel_joint", link_a, barrel.name, pa, jt, k, c),
                    _follower(f"{name}_rod_joint", link_b, rod.name, pb, jt, k - kp,
                              c + math.pi - cp)]


def _bucket_linkage(spec: MachineSpec) -> None:
    """Dog-bone (pivoting on the stick) and H-link (pivoting on the bucket) followers.

    The four-bar is solved exactly over the bucket range; each link's angle is
    then line-fitted like the cylinders, so the two links meet at their common
    pin (the bucket-cylinder rod eye) within the fit error.
    """
    bj = spec.joint("bucket_joint")
    (sx, sz), (lx, lz) = DOGBONE_PIN, LINK_PIN
    r_l, a_l = math.hypot(lx, lz), math.atan2(lz, lx)
    qs, th_d, th_h = _samples(bj), [], []
    for q in qs:
        px, pz = STICK_LEN + r_l * math.cos(a_l - q), r_l * math.sin(a_l - q)   # link pin
        d = math.hypot(px - sx, pz - sz)
        ex, ez = (px - sx) / d, (pz - sz) / d
        a = (DOGBONE_LEN ** 2 - HLINK_LEN ** 2 + d * d) / (2 * d)
        h = math.sqrt(DOGBONE_LEN ** 2 - a * a)
        jx, jz = sx + a * ex - h * ez, sz + a * ez + h * ex                     # common pin
        th_d.append(math.atan2(jz - sz, jx - sx))
        th_h.append(math.atan2(jz - pz, jx - px) + q)                            # bucket frame
    dog = Link("bucket_dogbone", 8.0, [
        *[box(DOGBONE_LEN + 0.07, 0.02, 0.07, at=(DOGBONE_LEN / 2, y, 0.0), mat="accent",
              collision=False) for y in (0.10, -0.10)],                          # side links
        cyl(0.035, 0.24, axis="y", mat="steel", collision=False),
        cyl(0.035, 0.24, at=(DOGBONE_LEN, 0.0, 0.0), axis="y", mat="steel", collision=False),
    ])
    hlink = Link("bucket_hlink", 9.0, [
        *[box(HLINK_LEN + 0.07, 0.02, 0.08, at=(HLINK_LEN / 2, y, 0.0), mat="dark",
              collision=False) for y in (0.05, -0.05)],
        box(0.07, 0.10, 0.05, at=(HLINK_LEN * 0.55, 0.0, 0.0), mat="dark", collision=False),
        cyl(0.03, 0.17, axis="y", mat="steel", collision=False),
    ])
    spec.links += [dog, hlink]
    spec.joints += [
        _follower("bucket_dogbone_joint", "stick", dog.name, (sx, 0.0, sz), bj,
                  *_fit(qs, th_d, bj.init)),
        _follower("bucket_hlink_joint", "bucket", hlink.name, (lx, 0.0, lz), bj,
                  *_fit(qs, th_h, bj.init)),
    ]


def _track(y: float) -> list[Shape]:
    """Rubber track loop: open sides showing the track frame, idler and rollers."""
    straight = TRACK_L - TRACK_H
    r, belt = TRACK_H / 2, 0.08
    shapes = [
        box(straight, SHOE_W, belt, at=(0.0, y, -r + belt / 2), mat="rubber"),     # ground run
        box(straight, SHOE_W, belt, at=(0.0, y, r - belt / 2), mat="rubber"),      # top run
        cyl(r, SHOE_W, at=(straight / 2, y, 0.0), axis="y", mat="rubber"),         # idler wrap
        cyl(r, SHOE_W, at=(-straight / 2, y, 0.0), axis="y", mat="rubber"),        # sprocket wrap
        box(straight * 0.9, SHOE_W * 0.66, TRACK_H - 2 * belt - 0.06, at=(0.0, y, 0.0),
            mat="dark"),                                                           # track frame
        cyl(r - 0.06, SHOE_W + 0.02, at=(straight / 2, y, 0.0), axis="y", mat="steel",
            collision=False),                                                      # front idler
    ]
    shapes += [cyl(0.065, SHOE_W * 0.8, at=(x, y, -r + belt + 0.05), axis="y", mat="steel",
                   collision=False) for x in (-0.42, 0.0, 0.42)]                   # bottom rollers
    shapes += [box(0.05, SHOE_W + 0.01, 0.025, at=(x, y, -r + 0.012), mat="rubber",
                   collision=False) for x in (-0.7, -0.35, 0.0, 0.35, 0.7)]       # tread lugs
    return shapes


def build() -> MachineSpec:
    y_t = GAUGE / 2

    # ------------------------------------------------------------------ undercarriage
    undercarriage = Link("undercarriage", 1360.0, [
        *_track(+y_t),
        *_track(-y_t),
        box(0.95, GAUGE - SHOE_W, 0.20, at=(0.0, 0.0, 0.16), mat="dark"),         # car body
        cyl(0.48, 0.07, at=(0.0, 0.0, 0.295), mat="steel"),                        # swing bearing
        box(0.16, 1.00, 0.10, at=(0.15, 0.0, 0.03), mat="dark"),                   # blade lugs
    ])

    # ------------------------------------------------------------------ dozer blade
    ex, ez = BLADE_EDGE
    blade = Link("blade", 170.0, [
        beam((ex + 0.02, ez), (ex - 0.03, ez + 0.13), BLADE_W, 0.05),               # moldboard
        beam((ex - 0.03, ez + 0.13), (ex - 0.03, ez + 0.25), BLADE_W, 0.05),
        beam((ex - 0.03, ez + 0.25), (ex + 0.02, ez + BLADE_H), BLADE_W, 0.05),
        box(0.05, BLADE_W, 0.04, at=(ex + 0.02, 0.0, ez + 0.03), mat="steel"),      # cutting edge
        box(0.10, 1.30, 0.22, at=(ex - 0.10, 0.0, ez + 0.17)),                     # back frame
        beam((0.0, 0.0), (ex - 0.12, ez + 0.16), 0.10, 0.14, y=+0.42),             # push arms
        beam((0.0, 0.0), (ex - 0.12, ez + 0.16), 0.10, 0.14, y=-0.42),
        box(0.10, 0.74, 0.08, at=(ex - 0.22, 0.0, ez + 0.20)),                     # cross tube
        cyl(0.07, 0.12, at=(0.0, +0.42, 0.0), axis="y", mat="steel", collision=False),
        cyl(0.07, 0.12, at=(0.0, -0.42, 0.0), axis="y", mat="steel", collision=False),
        box(0.02, 0.06, 0.30, at=(ex + 0.03, BLADE_W / 2 - 0.03, ez + 0.18), mat="accent",
            collision=False),                                                      # end plates
        box(0.02, 0.06, 0.30, at=(ex + 0.03, -BLADE_W / 2 + 0.03, ez + 0.18), mat="accent",
            collision=False),
    ])

    # ------------------------------------------------------------------ upper structure
    cw = []
    for a_deg in (128.0, 154.0, 180.0, 206.0, 232.0):                            # rounded tail
        a = math.radians(a_deg)
        rc = TAIL_R - 0.15
        cw.append(box(0.30, 0.44, 0.70, at=(rc * math.cos(a), rc * math.sin(a), 0.35),
                      rpy=(0.0, 0.0, a)))
    cw += [box(0.03, 0.10, 0.05, at=(-TAIL_R - 0.005, y, 0.56), mat="red", collision=False)
           for y in (0.13, -0.13)]                                                 # tail lamps
    # Deck: rectangle with its two front corners cut at 45 deg (each cut filled by a
    # rotated box whose outer face lies on the chamfer line).
    dx0, dx1, dy, ch = DECK
    deck = [box(dx1 - ch - dx0, 2 * dy, 0.14, at=((dx0 + dx1 - ch) / 2, 0.0, 0.07)),
            box(ch, 2 * (dy - ch), 0.14, at=(dx1 - ch / 2, 0.0, 0.07))]
    deck += [box(ch * math.sqrt(2), ch / math.sqrt(2), 0.14,
                 at=(dx1 - 0.75 * ch, s * (dy - 0.75 * ch), 0.07), rpy=(0.0, 0.0, -s * math.pi / 4))
             for s in (1, -1)]
    cab_x0, cab_x1, cab_y0, cab_y1, cab_z0, cab_z1 = -0.45, 0.66, -0.04, 0.86, 0.14, 1.88
    cab_cx, cab_cy = (cab_x0 + cab_x1) / 2, (cab_y0 + cab_y1) / 2
    house = Link("upper_structure", 1680.0, [
        *deck,
        *cw,                                                                       # counterweight
        box(0.28, 1.24, 0.60, at=(-0.54, 0.0, 0.44)),                              # engine hood
        box(0.40, 0.82, 0.60, at=(-0.20, -0.45, 0.44)),                            # (behind cab, right)
        box(0.66, 0.74, 0.50, at=(0.33, -0.49, 0.39)),                             # tank / valves
        box(0.36, 0.02, 0.30, at=(-0.20, -0.87, 0.46), mat="accent", collision=False),  # grille
        *[box(0.34, 0.025, 0.025, at=(-0.20, -0.875, z), mat="steel", collision=False)
          for z in (0.36, 0.43, 0.50, 0.57)],
        box(0.30, 0.02, 0.16, at=(0.30, -0.865, 0.46), mat="accent", collision=False),  # tank door
        cyl(0.035, 0.24, at=(-0.45, -0.52, 0.86), mat="accent", collision=False),   # exhaust
        # cab: black ROPS frame, glass all round, yellow roof
        box(cab_x1 - cab_x0, cab_y1 - cab_y0, cab_z1 - cab_z0,
            at=(cab_cx, cab_cy, (cab_z0 + cab_z1) / 2), mat="accent"),
        box(1.17, 0.94, 0.06, at=(cab_cx, cab_cy, cab_z1 + 0.03)),                 # roof
        box(0.02, 0.80, 1.05, at=(cab_x1 + 0.01, cab_cy, 1.27), mat="glass", collision=False),
        box(0.02, 0.80, 0.38, at=(cab_x1 + 0.01, cab_cy, 0.48), mat="glass", collision=False),
        box(0.98, 0.02, 1.25, at=(cab_cx, cab_y1 + 0.01, 1.17), mat="glass", collision=False),
        box(0.98, 0.02, 1.05, at=(cab_cx, cab_y0 - 0.01, 1.27), mat="glass", collision=False),
        box(0.02, 0.80, 0.95, at=(cab_x0 - 0.01, cab_cy, 1.32), mat="glass", collision=False),
        box(0.04, 0.03, 1.30, at=(0.18, cab_y1 + 0.015, 1.17), mat="accent", collision=False),
        box(0.40, 0.03, 0.03, at=(0.36, cab_y1 + 0.02, 1.05), mat="steel", collision=False),
        box(0.06, 0.12, 0.06, at=(cab_x1 + 0.02, 0.72, 1.82), mat="white", collision=False),
        box(0.25, 0.20, 0.08, at=(0.35, cab_y1 - 0.10, 0.10), mat="accent", collision=False),
        # swing-post ears (the boom-swing bracket pivots between them); the lower
        # one sits on top of the deck so it clears the raised blade
        box(0.26, 0.36, 0.07, at=(0.73, 0.0, 0.105), mat="dark"),
        cyl(0.17, 0.07, at=(SWING_POST[0], 0.0, 0.105), mat="dark"),
        box(0.26, 0.36, 0.07, at=(0.73, 0.0, 0.455), mat="dark"),
        cyl(0.17, 0.07, at=(SWING_POST[0], 0.0, 0.455), mat="dark"),
        box(0.08, 0.36, 0.46, at=(0.64, 0.0, 0.26), mat="dark"),
    ])

    # Boom-swing bracket (origin = king pin centre)
    fx, _, fz = BOOM_FOOT
    bracket = Link("swing_bracket", 90.0, [
        cyl(0.095, 0.28, at=(0.0, 0.0, 0.02), mat="body"),                         # king-pin boss
        beam((0.0, -0.02), (fx, fz), 0.26, 0.22),                                  # casting
        cyl(0.08, 0.30, at=BOOM_FOOT, axis="y", mat="steel", collision=False),     # foot pin boss
        beam((0.02, -0.06), (BOOM_CYL[0][0], BOOM_CYL[0][2]), 0.16, 0.12,
             collision=False),                                                     # cylinder lug
    ])

    boom = Link("boom", 180.0, [
        beam((0.0, 0.0), BOOM_KNEE, 0.22, 0.28),
        beam(BOOM_KNEE, BOOM_TIP, 0.20, 0.24),
        cyl(0.14, 0.22, at=(BOOM_KNEE[0], 0.0, BOOM_KNEE[1]), axis="y"),          # knee
        cyl(0.09, 0.30, axis="y", mat="steel", collision=False),                     # foot boss
        cyl(0.08, 0.26, at=(BOOM_TIP[0], 0.0, BOOM_TIP[1]), axis="y", mat="steel",
            collision=False),
        box(0.08, 0.10, 0.07, at=(0.40, 0.0, 0.37), rpy=(0.0, -0.51, 0.0), mat="white",
            collision=False),                                                      # boom light
        beam((BOOM_CYL[1][0] - 0.08, BOOM_CYL[1][2] + 0.14), (BOOM_CYL[1][0], BOOM_CYL[1][2]),
             0.14, 0.10, collision=False),                                         # cylinder lug
        beam((ARM_CYL[0][0] + 0.06, ARM_CYL[0][2] - 0.13), (ARM_CYL[0][0], ARM_CYL[0][2]),
             0.14, 0.10, collision=False),                                         # cylinder lug
        cyl(0.05, 0.16, at=BOOM_CYL[1], axis="y", mat="steel", collision=False),
        cyl(0.05, 0.16, at=ARM_CYL[0], axis="y", mat="steel", collision=False),
    ])

    stick = Link("stick", 100.0, [
        beam((-0.24, 0.06), (0.40, 0.0), 0.17, 0.26),
        beam((-0.28, 0.12), (-0.05, 0.10), 0.15, 0.12),                            # arm-cyl boss
        beam((0.40, 0.0), (STICK_LEN, 0.0), 0.15, 0.17),
        cyl(0.08, 0.24, axis="y", mat="steel", collision=False),                     # pin boss
        cyl(0.06, 0.22, at=(STICK_LEN, 0.0, 0.0), axis="y", mat="steel", collision=False),
        cyl(0.06, 0.19, at=ARM_CYL[1], axis="y", mat="steel", collision=False),
        box(0.14, 0.10, 0.08, at=(BKT_CYL_BASE[0], 0.0, BKT_CYL_BASE[2] - 0.04),
            collision=False),                                                      # bucket-cyl lug
        cyl(0.05, 0.18, at=(DOGBONE_PIN[0], 0.0, DOGBONE_PIN[1]), axis="y", mat="steel",
            collision=False),                                                      # dog-bone pin boss
    ])

    # Bucket frame: origin at the bucket pin, +x toward the cutting edge, the
    # mouth faces -z (toward the machine when the stick hangs straight down).
    tx, tz = BUCKET_TIP
    lx, lz = LINK_PIN
    shell = [(0.10, -0.15), (-0.04, 0.08), (0.20, 0.25), (0.48, 0.20), (tx, tz)]
    tooth_dir = K.vunit((tx - 0.48, 0.0, tz - 0.20))
    side = [((0.04, 0.03), (0.50, 0.05), 0.26),                                    # side plates
            ((0.45, 0.02), (0.64, -0.03), 0.12)]
    bucket = Link("bucket", 100.0, [
        *[beam(shell[i], shell[i + 1], BUCKET_W, 0.035, mat="dark") for i in range(4)],
        *[beam(p0, p1, 0.02, d, y=s * (BUCKET_W / 2 + 0.01), mat="dark")
          for p0, p1, d in side for s in (1, -1)],
        *[beam((0.0, -0.04), (lx, lz + 0.04), 0.035, 0.12, y=s * 0.09, mat="dark",
               collision=False) for s in (1, -1)],                                 # pin ears
        cyl(0.035, 0.21, at=(lx, 0.0, lz), axis="y", mat="steel", collision=False),  # link pin
        box(0.04, BUCKET_W, 0.03, at=(tx - 0.02, 0.0, tz + 0.02), mat="steel",
            collision=False),                                                      # cutting edge
        *[beam((tx - 0.02, tz + 0.01), (tx + 0.08 * tooth_dir[0], tz + 0.08 * tooth_dir[2]),
               0.05, 0.035, y=y, mat="steel", collision=False)
          for y in (-0.22, -0.075, 0.075, 0.22)],                                  # teeth
    ])

    sprocket_x = -(TRACK_L - TRACK_H) / 2
    links = [undercarriage, blade, house, bracket, boom, stick, bucket,
             sprocket_link("left_sprocket", 0.20, SHOE_W + 0.02, 15.0),
             sprocket_link("right_sprocket", 0.20, SHOE_W + 0.02, 15.0)]

    # Blade limits from the datasheet lift / drop of the cutting edge (which is
    # on the ground at q = 0: BLADE_EDGE z = ground in the pivot frame).
    assert abs(BLADE_PIVOT[2] + ez - GROUND_Z) < 1e-9
    r_e, a_e = math.hypot(ex, ez), math.atan2(ez, ex)
    blade_up = math.asin((ez + BLADE_LIFT) / r_e) - a_e
    blade_down = math.asin((ez - BLADE_DROP) / r_e) - a_e

    joints = [
        Joint("blade_joint", "revolute", "undercarriage", "blade", xyz=BLADE_PIVOT,
              axis=(0, -1, 0), lower=blade_down, upper=blade_up, init=blade_up,
              velocity=0.35, role="tool"),
        Joint("swing_joint", "continuous", "undercarriage", "upper_structure",
              xyz=(0.0, 0.0, SWING_Z), axis=(0, 0, 1), velocity=1.0, role="tool"),
        Joint("boom_swing_joint", "revolute", "upper_structure", "swing_bracket",
              xyz=SWING_POST, axis=(0, 0, 1), lower=-1.05, upper=1.31, init=0.0,
              velocity=0.45, role="tool"),
        Joint("boom_joint", "revolute", "swing_bracket", "boom", xyz=BOOM_FOOT,
              axis=(0, -1, 0), lower=-1.16, upper=1.12, init=0.05, velocity=0.75, role="tool"),
        Joint("stick_joint", "revolute", "boom", "stick",
              xyz=(BOOM_TIP[0], 0.0, BOOM_TIP[1]), rpy=(0.0, STICK_PITCH, 0.0),
              axis=(0, -1, 0), lower=-1.05, upper=0.85, init=-1.05, velocity=0.85, role="tool"),
        Joint("bucket_joint", "revolute", "stick", "bucket", xyz=(STICK_LEN, 0.0, 0.0),
              axis=(0, 1, 0), lower=-1.05, upper=2.15, init=2.15, velocity=1.3, role="tool"),
        Joint("left_sprocket_joint", "continuous", "undercarriage", "left_sprocket",
              xyz=(sprocket_x, y_t, 0.0), axis=(0, 1, 0), velocity=20.0, role="roll_left",
              wheel_radius=0.20),
        Joint("right_sprocket_joint", "continuous", "undercarriage", "right_sprocket",
              xyz=(sprocket_x, -y_t, 0.0), axis=(0, 1, 0), velocity=20.0, role="roll_right",
              wheel_radius=0.20),
    ]

    spec = MachineSpec(
        name="mini_excavator",
        abbrev="MEX",
        display_name="Mini Excavator",
        category="excavation",
        reference="3.5-4 t rubber-track mini excavator (CAT 303.5 / Kubota KX040-4 class)",
        description="Compact tracked excavator with cab, 360 deg swing, boom-swing bracket for "
                    "offset digging, 2.4 m boom / 1.35 m stick, 600 mm bucket and a front "
                    "dozer blade.",
        root="undercarriage",
        links=links,
        joints=joints,
        drive=Drive(type="skid", max_speed=1.28, max_yaw_rate=0.6, track_width=GAUGE),
        tools=[
            Tool("swing_joint", "swing", positive="left", negative="right"),
            Tool("boom_swing_joint", "boom_swing", positive="left", negative="right"),
            Tool("boom_joint", "boom", positive="raise", negative="lower"),
            Tool("stick_joint", "stick", positive="out", negative="in"),
            Tool("bucket_joint", "bucket", positive="curl", negative="dump"),
            Tool("blade_joint", "blade", positive="raise", negative="lower"),
        ],
        cameras=[
            # operator eye point, just ahead of the windscreen (URDF glass is opaque)
            Camera("front", "upper_structure", (0.71, 0.42, 1.32), (0.0, 0.45, -0.10)),
            Camera("left", "upper_structure", (0.62, 0.90, 1.84), (0.0, 0.45, 0.55)),
            Camera("right", "upper_structure", (0.60, -0.86, 0.95), (0.0, 0.35, -0.55)),
        ],
        envelope=(4.85, 1.78, 2.50),
        operating_mass=3800.0,
        ground_contact=["undercarriage"],
        notes="Bucket curl is not auto-verified (no up/out/left semantics); +1 = curl "
              "(cutting edge toward the machine), checked in the sweep render.",
    )

    # Hydraulics and bucket linkage (visual followers; see the module docstring).
    _ram(spec, "boom_cyl", "swing_bracket", BOOM_CYL[0], "boom", BOOM_CYL[1], "boom_joint",
         0.055, (22.0, 12.0))
    _ram(spec, "arm_cyl", "boom", ARM_CYL[0], "stick", ARM_CYL[1], "stick_joint",
         0.05, (18.0, 10.0))
    _ram(spec, "blade_cyl", "undercarriage", BLADE_CYL_BASE, "blade",
         (ex - 0.14, 0.0, ez + 0.22), "blade_joint", 0.045, (10.0, 6.0))
    _bucket_linkage(spec)
    _ram(spec, "bucket_cyl", "stick", BKT_CYL_BASE, "bucket_dogbone", (DOGBONE_LEN, 0.0, 0.0),
         "bucket_joint", 0.05, (12.0, 6.0))
    return spec
