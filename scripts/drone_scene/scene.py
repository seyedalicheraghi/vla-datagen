"""Layout and USD authoring of the PBR scenery around the hovering Crazyflie.

A sunlit backyard: the drone hovers 0.5 m above a herringbone clay-paver patio edged with a concrete
curb; a lawn (photo-scanned texture plus 3D grass clumps) surrounds it; behind the patio a mulch bed
holds boxwood and hydrangea shrubs with a Japanese maple further back; a few fallen leaves lie on the
pavers.  Lighting is the noon_grass HDR sky with its sun cut out and replaced by a DistantLight at a
more flattering elevation, both calibrated against the sky Isaac Lab exposes well (see hdr.calibrate).

``build(stage, cfg, manifest)`` authors everything under ``/World/Scenery`` (plus cameras under
``/World/Cameras``) on any pxr stage: Isaac's live stage or a plain usd-core stage for the preview.
``apply_drone_materials(stage, root)`` gives the Crazyflie real PBR materials; the asset ships ten
OmniPBR materials but binds none, so it renders flat grey.

World frame: Z up, metres, drone at the origin (x, y).  Uses pxr only inside functions so the config
can be imported before Isaac Sim starts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import assets

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CameraCfg:
    name: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    focal_length: float = 18.147          # mm, Isaac's default viewport lens
    horizontal_aperture: float = 20.955   # mm
    width: int = 1280
    height: int = 720
    f_stop: float = 0.0                   # 0 = pinhole
    focus_distance: float = 0.0           # m; 0 = distance to target


@dataclass
class SunCfg:
    elevation_deg: float = 55.0
    azimuth_deg: float = 60.0             # direction TO the sun, counter-clockwise from +X
    angle_deg: float = 0.53               # angular diameter (sharp, real sun)


@dataclass
class SceneCfg:
    drone_pos: tuple[float, float, float] = (0.0, 0.0, 0.5)
    patio: tuple[float, float, float, float] = (-1.6, 2.0, -2.2, 2.0)      # x0, x1, y0, y1
    patio_top: float = 0.0
    lawn_z: float = -0.025                # lawn sits a little below the pavers
    curb_width: float = 0.10
    curb_top: float = 0.008
    bed: tuple[float, float, float, float] = (-3.8, -2.4, -3.6, 3.2)
    lawn_half: float = 80.0               # far lawn extent; the HDR horizon takes over beyond
    grass_area: tuple[float, float, float, float] = (-10.5, 8.5, -10.5, 9.5)
    grass_feather: float = 3.0            # clumps thin out over this band inside grass_area: no hard edge
    grass_height: float = 0.55            # z scale of the clumps: ~7-9 cm, a mown lawn
    leaves: int = 40                      # fallen leaves on the patio
    sun: SunCfg = field(default_factory=SunCfg)
    sky_rotation_deg: float = 0.0
    seed: int = 7
    cameras: list[CameraCfg] = field(default_factory=lambda: [
        CameraCfg("demo", eye=(0.5, 0.5, 1.0), target=(0.0, 0.0, 0.5)),                   # quadcopter.py view
        # 70 mm from ~1 m, a little below the drone, sun behind the camera's right shoulder: the drone is
        # front-lit against the sunlit garden and the trees on the horizon, sharp at f/8, the rest soft
        CameraCfg("hero", eye=(0.16, 1.02, 0.43), target=(0.0, 0.0, 0.505), focal_length=70.0, f_stop=8.0),
        # the same lens looking past the drone into the shade of the maple (moodier)
        CameraCfg("garden", eye=(1.02, -0.16, 0.43), target=(0.0, 0.0, 0.505), focal_length=70.0, f_stop=8.0),
        CameraCfg("top", eye=(0.01, 0.0, 9.0), target=(-0.8, 0.0, 0.0), focal_length=14.0),
    ])

    def camera(self, name: str) -> CameraCfg:
        for c in self.cameras:
            if c.name == name:
                return c
        raise KeyError(f"no camera {name!r}; available: {[c.name for c in self.cameras]}")


# props placed by hand: (prop, x, y, yaw_deg, scale)
PLANTING = [
    ("boxwood", -3.15, -2.6, 20.0, 0.80),
    ("hydrangea", -3.10, -1.3, 75.0, 0.70),
    ("boxwood", -3.20, -0.1, 140.0, 0.90),
    ("hydrangea", -3.05, 1.2, 200.0, 0.62),
    ("boxwood", -3.15, 2.5, 300.0, 0.75),
    ("maple", -5.6, 0.8, 30.0, 1.0),
]

# Grass clumps, largest first: (prototype, half-width at scale 1 (m), Poisson spacing (m),
# only within this distance of a hard edge (m) or None).  The small clumps fill the strip along the
# curb that a big clump could not reach without poking through the pavers.
GRASS_TIERS = [
    ("grass_a", 0.65, 0.62, None),        # overlapping clumps: a lawn, not tufts on a carpet
    ("grass_b", 0.34, 0.34, 1.15),        # reaches the big clumps (which start ~1.06 m out)
    ("grass_c", 0.15, 0.15, 0.62),        # reaches the medium clumps (which start ~0.55 m out)
]
# The clump texture mixes green and straw-coloured blades; this diffuse_tint (an OmniPBR input both RTX
# and the preview honour) turns it into a watered lawn
GRASS_TINT = (0.62, 0.90, 0.50)

# The Crazyflie's meshes and what they are on a real Crazyflie 2.1
DRONE_PARTS = {
    "cf_body_001": "pcb",              # the frame is the PCB: black solder mask
    "motor_mount": "mount",            # white plastic motor mounts and legs
    "Cylinder": "motor",               # brushed coreless motors
    "battery": "battery",              # LiPo pouch in silver foil
    "battery_holder": "plastic_dark",
    "pin_headers": "gold",
    "ccw_prop": "prop",
    "cw_prop": "prop",
}

# UsdPreviewSurface constants: (base colour linear RGB, roughness, metallic, clearcoat)
DRONE_MATERIALS = {
    "pcb": ((0.018, 0.020, 0.022), 0.32, 0.0, 0.6),
    "mount": ((0.62, 0.62, 0.60), 0.38, 0.0, 0.0),        # off-white injection-moulded plastic
    "motor": ((0.62, 0.62, 0.63), 0.28, 1.0, 0.0),
    "battery": ((0.80, 0.80, 0.82), 0.30, 0.85, 0.0),
    "plastic_dark": ((0.035, 0.035, 0.035), 0.5, 0.0, 0.0),
    "gold": ((1.0, 0.77, 0.34), 0.22, 1.0, 0.0),
    "prop": ((0.03, 0.03, 0.035), 0.38, 0.0, 0.0),
}


# ---------------------------------------------------------------------------
# Geometry helpers (numpy)
# ---------------------------------------------------------------------------

def sun_direction(sun: SunCfg) -> np.ndarray:
    """Unit vector from the ground to the sun."""
    el, az = math.radians(sun.elevation_deg), math.radians(sun.azimuth_deg)
    return np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


def box(x0, x1, y0, y1, z0, z1, tile: float, top_only: bool = False):
    """Axis-aligned box as (points, counts, indices, normals, st), split per face, UVs in metres/tile."""
    faces = [((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (0, 0, 1))]
    if not top_only:
        faces += [
            ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (0, -1, 0)),
            ((x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (0, 1, 0)),
            ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1), (1, 0, 0)),
            ((x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (-1, 0, 0)),
        ]
    pts, nrm, st = [], [], []
    for *corners, n in faces:
        for c in corners:
            pts.append(c)
            nrm.append(n)
            if n[2]:                                   # top: world xy
                st.append((c[0] / tile, c[1] / tile))
            elif n[0]:                                 # x-facing side: (y, z)
                st.append((c[1] / tile, c[2] / tile))
            else:                                      # y-facing side: (x, z)
                st.append((c[0] / tile, c[2] / tile))
    k = len(faces)
    return (np.array(pts, float), np.full(k, 4), np.arange(4 * k), np.array(nrm, float), np.array(st, float))


def merge(*meshes):
    pts, cnt, idx, nrm, st, off = [], [], [], [], [], 0
    for p, c, i, n, s in meshes:
        pts.append(p); cnt.append(c); idx.append(i + off); nrm.append(n); st.append(s)
        off += len(p)
    return tuple(np.concatenate(a) for a in (pts, cnt, idx, nrm, st))


def curb_boxes(cfg: SceneCfg, tile: float):
    x0, x1, y0, y1 = cfg.patio
    w, top, bot = cfg.curb_width, cfg.curb_top, cfg.lawn_z - 0.04
    return merge(
        box(x0 - w, x1 + w, y0 - w, y0, bot, top, tile),
        box(x0 - w, x1 + w, y1, y1 + w, bot, top, tile),
        box(x0 - w, x0, y0, y1, bot, top, tile),
        box(x1, x1 + w, y0, y1, bot, top, tile),
    )


def poisson_disk(rect, spacing: float, rng: np.random.Generator, keep=None, tries: int = 30) -> np.ndarray:
    """Bridson Poisson-disk samples in rect = (x0, x1, y0, y1), optionally filtered by keep(x, y)."""
    x0, x1, y0, y1 = rect
    cell = spacing / math.sqrt(2)
    gw, gh = int(math.ceil((x1 - x0) / cell)), int(math.ceil((y1 - y0) / cell))
    grid = -np.ones((gw, gh), int)
    pts = [np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])]
    grid[int((pts[0][0] - x0) / cell), int((pts[0][1] - y0) / cell)] = 0
    active = [0]
    while active:
        i = active[rng.integers(len(active))]
        for _ in range(tries):
            r, a = spacing * (1 + rng.random()), rng.uniform(0, 2 * math.pi)
            q = pts[i] + r * np.array([math.cos(a), math.sin(a)])
            if not (x0 <= q[0] < x1 and y0 <= q[1] < y1):
                continue
            gx, gy = int((q[0] - x0) / cell), int((q[1] - y0) / cell)
            near = grid[max(gx - 2, 0):gx + 3, max(gy - 2, 0):gy + 3]
            if all(np.linalg.norm(pts[j] - q) >= spacing for j in near[near >= 0]):
                grid[gx, gy] = len(pts)
                pts.append(q)
                active.append(len(pts) - 1)
                break
        else:
            active.remove(i)
    out = np.array(pts)
    if keep is not None:
        out = out[np.array([keep(x, y) for x, y in out], bool)]
    return out


def hard_rects(cfg: SceneCfg) -> list[tuple[float, float, float, float]]:
    """Patio + curb and the bed: grass must stay off these."""
    x0, x1, y0, y1 = cfg.patio
    w = cfg.curb_width
    return [(x0 - w, x1 + w, y0 - w, y1 + w), cfg.bed]


def rect_distance(x: float, y: float, rect) -> float:
    """Euclidean distance from (x, y) to an axis-aligned rect (0 inside)."""
    x0, x1, y0, y1 = rect
    return math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))


GRASS_MAX_SCALE = 1.15                    # random clump scale is 0.85 .. 1.15


def grass_instances(cfg: SceneCfg) -> list[tuple[str, float, float]]:
    """(prototype, x, y) for every clump: each clump's footprint (rotated square) stays off hard surfaces."""
    rng = np.random.default_rng(cfg.seed)
    rects = hard_rects(cfg)
    out = []
    for proto, half, spacing, near in GRASS_TIERS:
        reach = half * math.sqrt(2) * GRASS_MAX_SCALE          # worst case over yaw

        gx0, gx1, gy0, gy1 = cfg.grass_area

        def keep(x, y, reach=reach, near=near):
            d = min(rect_distance(x, y, r) for r in rects)
            if d < reach or (near is not None and d > near):
                return False
            edge = min(x - gx0, gx1 - x, y - gy0, gy1 - y)          # feathered outer edge
            return edge >= cfg.grass_feather or rng.random() < edge / cfg.grass_feather

        out += [(proto, float(x), float(y)) for x, y in poisson_disk(cfg.grass_area, spacing, rng, keep)]
    return out


def leaf_positions(cfg: SceneCfg) -> np.ndarray:
    """Fallen leaves on the patio, gathered along the curb and in the corners like wind leaves them."""
    rng = np.random.default_rng(cfg.seed + 2)
    x0, x1, y0, y1 = cfg.patio
    pts = []
    while len(pts) < cfg.leaves:
        x, y = rng.uniform(x0 + 0.06, x1 - 0.06), rng.uniform(y0 + 0.06, y1 - 0.06)
        edge = min(x - x0, x1 - x, y - y0, y1 - y)
        if rng.random() < math.exp(-edge / 0.25) and math.hypot(x, y) > 0.35:     # not right under the drone
            pts.append((x, y))
    return np.array(pts)


def look_at_matrix(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """4x4 row-vector (USD) transform of a camera at eye looking at target (camera looks down -Z, +Y up)."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:                  # looking straight down
        right = np.cross(fwd, (0.0, 1.0, 0.0))
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3], m[3, :3] = right, cam_up, -fwd, eye
    return m


def quat_z_to(d) -> tuple[float, float, float, float]:
    """(w, x, y, z) quaternion rotating +Z onto unit vector d."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, d))
    if c < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)
    axis = np.cross(z, d)
    w = math.sqrt((1 + c) / 2)
    s = axis / (2 * w)
    return (w, float(s[0]), float(s[1]), float(s[2]))


# ---------------------------------------------------------------------------
# USD authoring
# ---------------------------------------------------------------------------

def _mesh(stage, path: str, data, material=None, double_sided: bool = False):
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    pts, cnt, idx, nrm, st = data
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([int(c) for c in cnt]))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in idx]))
    m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*n) for n in nrm]))
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateDoubleSidedAttr(double_sided)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))
    UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec2fArray([Gf.Vec2f(*s) for s in st]))
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    return m


def preview_surface(stage, path: str, *, albedo=None, orm=None, normal=None, roughness_tex=None,
                    color=(0.5, 0.5, 0.5), roughness=0.6, metallic=0.0, clearcoat=0.0,
                    flip_green=True, tint=(1.0, 1.0, 1.0), roughness_scale=1.0):
    """UsdPreviewSurface; textures are read through primvar 'st' with repeat wrapping."""
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, f"{path}/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    pbr.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(float(clearcoat))
    pbr.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.08)
    pbr.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    if not (albedo or orm or normal or roughness_tex):
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
        return mat

    reader = UsdShade.Shader.Define(stage, f"{path}/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    st = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    def tex(name, file, raw, scale=(1, 1, 1, 1), bias=(0, 0, 0, 0)):
        t = UsdShade.Shader.Define(stage, f"{path}/{name}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(file).replace("\\", "/"))
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw" if raw else "sRGB")
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
        if scale != (1, 1, 1, 1):
            t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
        if bias != (0, 0, 0, 0):
            t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
        return t

    if albedo:
        a = tex("albedoTex", albedo, raw=False, scale=(*tint, 1.0))
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            a.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    else:
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    if orm:
        o = tex("ormTex", orm, raw=True, scale=(1.0, roughness_scale, 1.0, 1.0))
        pbr.CreateInput("occlusion", Sdf.ValueTypeNames.Float).ConnectToSource(o.CreateOutput("r", Sdf.ValueTypeNames.Float))
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(o.CreateOutput("g", Sdf.ValueTypeNames.Float))
        pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(o.CreateOutput("b", Sdf.ValueTypeNames.Float))
    else:
        if roughness_tex:
            r = tex("roughTex", roughness_tex, raw=True, scale=(roughness_scale,) * 3 + (1.0,))
            pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(r.CreateOutput("r", Sdf.ValueTypeNames.Float))
        else:
            pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    if normal:
        g = -2.0 if flip_green else 2.0
        n = tex("normalTex", normal, raw=True, scale=(2.0, g, 2.0, 1.0), bias=(-1.0, -g / 2.0, -1.0, 0.0))
        pbr.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(n.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    return mat


def author_ground_materials(stage, root: str, textures: dict[str, dict[str, str]]) -> dict:
    mats = {}
    for name, tex in textures.items():
        spec = assets.MATERIALS[name]
        mats[name] = preview_surface(
            stage, f"{root}/Looks/{name}", albedo=tex.get("albedo"), orm=tex.get("orm"),
            normal=tex.get("normal"), roughness_tex=tex.get("roughness"), flip_green=spec.flip_green,
            tint=spec.tint, roughness_scale=spec.roughness_scale)
    return mats


def author_ground(stage, root: str, cfg: SceneCfg, mats: dict) -> None:
    from pxr import UsdGeom

    UsdGeom.Xform.Define(stage, f"{root}/Ground")
    T = {k: assets.MATERIALS[k].tile_m for k in assets.MATERIALS}
    x0, x1, y0, y1 = cfg.patio
    h = cfg.lawn_half
    _mesh(stage, f"{root}/Ground/Lawn", box(-h, h, -h, h, cfg.lawn_z - 0.05, cfg.lawn_z, T["lawn"], top_only=True),
          mats.get("lawn"))
    _mesh(stage, f"{root}/Ground/Patio", box(x0, x1, y0, y1, cfg.lawn_z - 0.04, cfg.patio_top, T["pavers"]),
          mats.get("pavers"))
    _mesh(stage, f"{root}/Ground/Curb", curb_boxes(cfg, T["curb"]), mats.get("curb"))
    bx0, bx1, by0, by1 = cfg.bed
    _mesh(stage, f"{root}/Ground/Bed", box(bx0, bx1, by0, by1, cfg.lawn_z - 0.04, cfg.lawn_z + 0.012, T["mulch"]),
          mats.get("mulch"))


def _reference(stage, path: str, asset_path: str, units: float, xyz, yaw_deg: float, scale: float):
    """Place an asset: our transform on a wrapper Xform, the reference on its child ``asset``.

    NVIDIA's assets carry their own xformOps on their root prim, so the reference must not share a prim
    with our placement ops.
    """
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    x.AddRotateZOp().Set(float(yaw_deg))
    x.AddScaleOp().Set(Gf.Vec3f(units * scale, units * scale, units * scale))
    UsdGeom.Xform.Define(stage, f"{path}/asset").GetPrim().GetReferences().AddReference(asset_path)
    return x


def author_props(stage, root: str, cfg: SceneCfg, prop_paths: dict[str, str]) -> None:
    from pxr import UsdGeom

    UsdGeom.Xform.Define(stage, f"{root}/Props")
    counts: dict[str, int] = {}
    for name, x, y, yaw, scale in PLANTING:
        if name not in prop_paths:
            continue
        counts[name] = counts.get(name, 0) + 1
        on_patio = cfg.patio[0] < x < cfg.patio[1] and cfg.patio[2] < y < cfg.patio[3]
        in_bed = cfg.bed[0] < x < cfg.bed[1] and cfg.bed[2] < y < cfg.bed[3]
        z = cfg.patio_top if on_patio else (cfg.lawn_z + 0.012 if in_bed else cfg.lawn_z)
        _reference(stage, f"{root}/Props/{name}_{counts[name]}", prop_paths[name], assets.PROPS[name].units,
                   (x, y, z), yaw, scale)


def author_grass(stage, root: str, cfg: SceneCfg, prop_paths: dict[str, str]) -> int:
    """Grass clumps as one PointInstancer (instanced in both RTX and the preview)."""

    protos = [t[0] for t in GRASS_TIERS if t[0] in prop_paths]
    if not protos:
        return 0
    clumps = [c for c in grass_instances(cfg) if c[0] in protos]
    rng = np.random.default_rng(cfg.seed + 1)
    n = len(clumps)
    yaws = rng.uniform(0, 2 * math.pi, n)
    scales = rng.uniform(2 - GRASS_MAX_SCALE, GRASS_MAX_SCALE, n)
    inst = _instancer(stage, f"{root}/Lawn/Grass", protos, prop_paths,
                      [protos.index(p) for p, _, _ in clumps], [(x, y, cfg.lawn_z) for _, x, y in clumps],
                      [(math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)) for a in yaws],
                      [(s, s, s * cfg.grass_height) for s in scales])
    tint_materials(stage, str(inst.GetPath()) + "/Prototypes", GRASS_TINT)
    return n


def tint_materials(stage, root: str, tint) -> int:
    """Override ``diffuse_tint`` on every MDL shader under ``root`` (inside referenced assets too)."""
    from pxr import Gf, Sdf, Usd, UsdShade

    n = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.IsA(UsdShade.Shader) and UsdShade.Shader(prim).GetImplementationSource() == "sourceAsset":
            UsdShade.Shader(prim).CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*tint))
            n += 1
    return n


def _default_cache() -> str:
    import os
    return os.path.join(os.path.expanduser("~"), ".cache", "drone_scene")


def author_leaves(stage, root: str, cfg: SceneCfg, prop_paths: dict[str, str], on_bricks: bool = False) -> int:
    protos = [n for n in ("leaves_maple", "leaves_oak") if n in prop_paths]
    if not protos or cfg.leaves <= 0:
        return 0
    pts = leaf_positions(cfg)
    z0 = cfg.patio_top + 0.002
    if on_bricks:                                  # rest on the 3D paver tops, not inside them
        from .pavers import PaverSpec
        z0 += PaverSpec().top_rise
    rng = np.random.default_rng(cfg.seed + 3)
    n = len(pts)
    quats = []
    for yaw, tilt, axis in zip(rng.uniform(0, 2 * math.pi, n), rng.normal(0, 0.12, n), rng.uniform(0, 2 * math.pi, n)):
        q = quat_yaw_tilt(yaw, tilt, axis)
        quats.append(q)
    scales = rng.uniform(0.8, 1.25, n)
    _instancer(stage, f"{root}/Patio/Leaves", protos, prop_paths, list(rng.integers(len(protos), size=n)),
               [(x, y, z0) for x, y in pts], quats, [(s, s, s) for s in scales])
    return n


def quat_yaw_tilt(yaw: float, tilt: float, axis: float) -> tuple[float, float, float, float]:
    """(w, x, y, z): yaw about Z, then a small tilt about a horizontal axis (leaves never lie flat)."""
    qy = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    qt = np.array([math.cos(tilt / 2), math.cos(axis) * math.sin(tilt / 2), math.sin(axis) * math.sin(tilt / 2), 0.0])
    w1, x1, y1, z1 = qt
    w2, x2, y2, z2 = qy
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def _instancer(stage, path: str, protos: list[str], prop_paths: dict[str, str], indices, positions, quats, scales):
    from pxr import Gf, UsdGeom, Vt

    parent = path.rsplit("/", 1)[0]
    if not stage.GetPrimAtPath(parent):
        UsdGeom.Xform.Define(stage, parent)
    inst = UsdGeom.PointInstancer.Define(stage, path)
    UsdGeom.Scope.Define(stage, f"{path}/Prototypes")
    rels = [_reference(stage, f"{path}/Prototypes/{n}", prop_paths[n], assets.PROPS[n].units, (0, 0, 0), 0, 1.0).GetPath()
            for n in protos]
    inst.CreatePrototypesRel().SetTargets(rels)
    inst.CreateProtoIndicesAttr(Vt.IntArray([int(i) for i in indices]))
    inst.CreatePositionsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in positions]))
    inst.CreateOrientationsAttr(Vt.QuathArray([Gf.Quath(*map(float, q)) for q in quats]))
    inst.CreateScalesAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, s)) for s in scales]))
    return inst


def author_lights(stage, root: str, cfg: SceneCfg, sky: dict) -> None:
    """sky: {'file', 'dome_intensity', 'sun_intensity', 'sun_color'} from prepare."""
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    dome = UsdLux.DomeLight.Define(stage, f"{root}/Sky")
    dome.CreateTextureFileAttr().Set(Sdf.AssetPath(str(sky["file"]).replace("\\", "/")))
    dome.CreateTextureFormatAttr().Set(UsdLux.Tokens.latlong)
    dome.CreateIntensityAttr().Set(float(sky["dome_intensity"]))
    if cfg.sky_rotation_deg:
        UsdGeom.Xformable(dome).AddRotateZOp().Set(float(cfg.sky_rotation_deg))
    sun = UsdLux.DistantLight.Define(stage, f"{root}/Sun")
    sun.CreateIntensityAttr().Set(float(sky["sun_intensity"]))
    sun.CreateColorAttr().Set(Gf.Vec3f(*sky.get("sun_color", (1.0, 0.97, 0.92))))
    sun.CreateAngleAttr().Set(float(cfg.sun.angle_deg))
    # a DistantLight shines along its local -Z: rotate +Z onto the direction towards the sun
    w, x, y, z = quat_z_to(sun_direction(cfg.sun))
    UsdGeom.Xformable(sun).AddOrientOp().Set(Gf.Quatf(w, x, y, z))


def author_cameras(stage, cfg: SceneCfg, root: str = "/World/Cameras") -> None:
    from pxr import Gf, UsdGeom

    UsdGeom.Xform.Define(stage, root)
    for c in cfg.cameras:
        cam = UsdGeom.Camera.Define(stage, f"{root}/{c.name}")
        cam.CreateFocalLengthAttr(c.focal_length)
        cam.CreateHorizontalApertureAttr(c.horizontal_aperture)
        cam.CreateVerticalApertureAttr(c.horizontal_aperture * c.height / c.width)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10000.0))
        if c.f_stop > 0:
            cam.CreateFStopAttr(c.f_stop)
            focus = c.focus_distance or float(np.linalg.norm(np.subtract(c.target, c.eye)))
            cam.CreateFocusDistanceAttr(focus)
        cam.AddTransformOp().Set(Gf.Matrix4d(look_at_matrix(c.eye, c.target).tolist()))


def apply_drone_materials(stage, drone_root: str, looks_root: str = "/World/Looks/Drone") -> int:
    """Bind real PBR materials to the Crazyflie's meshes (matched by mesh name).  Returns meshes bound."""
    from pxr import Usd, UsdGeom, UsdShade

    mats = {k: preview_surface(stage, f"{looks_root}/{k}", color=c, roughness=r, metallic=m, clearcoat=cc)
            for k, (c, r, m, cc) in DRONE_MATERIALS.items()}
    n = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(drone_root)):
        if prim.IsA(UsdGeom.Mesh) and prim.GetName() in DRONE_PARTS:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                mats[DRONE_PARTS[prim.GetName()]], UsdShade.Tokens.strongerThanDescendants)
            n += 1
    return n


def add_ground_collision(stage, root: str = "/World/Scenery") -> int:
    """Make the visible ground meshes static colliders (what you see is what the drone lands on)."""
    from pxr import UsdGeom, UsdPhysics

    n = 0
    for name in ("Lawn", "Patio", "Curb", "Bed"):
        prim = stage.GetPrimAtPath(f"{root}/Ground/{name}")
        if prim and prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(UsdPhysics.Tokens.none)
            n += 1
    return n


def build(stage, cfg: SceneCfg, manifest: dict, root: str = "/World/Scenery", physics: bool = False,
          bricks: bool = True) -> dict:
    """Author the whole scenery from a prepare.py manifest; returns a summary."""
    from pxr import UsdGeom

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, root)
    mats = author_ground_materials(stage, root, manifest["textures"])
    author_ground(stage, root, cfg, mats)
    paving = None
    if bricks:
        # real 3D herringbone pavers on top; the flat slab stays as the (invisible) collider
        from . import pavers
        pavers.hide_flat_patio(stage, root)
        paving = pavers.author_patio(stage, root, cfg.patio, cfg.patio_top,
                                     manifest.get("cache_dir") or _default_cache(), seed=cfg.seed,
                                     textures=manifest["textures"])
    author_props(stage, root, cfg, manifest["props"])
    clumps = author_grass(stage, root, cfg, manifest["props"])
    leaves = author_leaves(stage, root, cfg, manifest["props"], on_bricks=bricks)
    author_lights(stage, root, cfg, manifest["sky"])
    author_cameras(stage, cfg)
    colliders = add_ground_collision(stage, root) if physics else 0
    return {"materials": len(mats), "grass_clumps": clumps, "leaves": leaves, "colliders": colliders,
            "paving": {k: paving[k] for k in ("bricks", "cut_bricks", "crops", "triangles")} if paving else None,
            "props": sum(1 for p in PLANTING if p[0] in manifest["props"])}
