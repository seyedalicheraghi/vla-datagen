"""The drones the backyard scene can fly, and how two LUCID cameras are mounted on them.

``lmf2`` (default): NTNU Autonomous Robots Lab's LMF2 quadrotor, the asset behind Isaac Lab's ``ARL_ROBOT_1_CFG``
(``Isaac/Robots/NTNU/ARL-Robot-1``): 1.2 kg, 0.52 m across its four propeller guards, 4 x 10 N of thrust.  Two
LUCID Triton cameras with C-mount lenses (~110 g each) are a normal payload for it.

``crazyflie``: the Bitcraze Crazyflie of Isaac Lab's quadcopter demo: 27 g with a ~15 g payload.  It cannot
lift even one LUCID camera; the option is there to keep the original demo drone in the same scene.

What this module adds on top of the assets:

- ``apply_lmf2_materials``: the LMF2 mesh's CAD appearance groups (GeomSubsets named ``matte_aluminum``,
  ``blue_anodized_aluminum``, ``clear_glass``, ...) are bound to an OmniPBR material with a pure black
  (0, 0, 0) diffuse colour and no metal/roughness values, i.e. 52 % of the drone renders as an unlit
  silhouette.  They get physically based metals, plastics and glass instead, chosen by those names;
- ``mount_cameras``: front and back camera poses from the drone's own mesh: each camera's housing sits just
  clear of the body surface on the centre line, at the vehicle's mount height, tilted ``pitch_down_deg``;
- ``frame_scene``: the third-person cameras (``demo``, ``hero``, ``garden``) pulled back to fit the larger drone.

Pure numpy except the functions that take a ``stage``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace

import numpy as np

from . import assets


@dataclass(frozen=True)
class Vehicle:
    key: str
    name: str
    usd: str                    # Nucleus path (relative to the asset root, like assets.CRAZYFLIE)
    prim: str                   # where it is spawned
    body: str                   # the rigid body the cameras ride on (child of prim)
    prop_bodies: str            # regex of the bodies the hover thrust acts on
    mass_kg: float
    payload_kg: float           # what it can carry and still have a thrust-to-weight ratio of ~2
    view_scale: float           # third-person cameras stand this much further away than for the Crazyflie
    camera_z: float             # mount height of the cameras' optical axis in the body frame (m)


VEHICLES: dict[str, Vehicle] = {
    "lmf2": Vehicle("lmf2", "NTNU ARL LMF2 quadrotor (Isaac Lab ARL_ROBOT_1)",
                    "Isaac/Robots/NTNU/ARL-Robot-1/arl_robot_1.usd", "/World/Drone", "base_link", ".*_prop",
                    mass_kg=1.2, payload_kg=40.0 / (2 * 9.81) - 1.2, view_scale=3.0, camera_z=0.06),
    "crazyflie": Vehicle("crazyflie", "Bitcraze Crazyflie 2.x", assets.CRAZYFLIE, "/World/Crazyflie", "body",
                         "m.*_prop", mass_kg=0.027, payload_kg=0.015, view_scale=1.0, camera_z=0.02),
}
DEFAULT_VEHICLE = "lmf2"

# ---------------------------------------------------------------------------
# LMF2 materials, by CAD appearance name (UsdPreviewSurface: base colour linear RGB, roughness, metallic,
# opacity).  Metals use their measured F0 as base colour (aluminium 0.91, steel 0.56, brass 0.91/0.78/0.42)
# ---------------------------------------------------------------------------

LMF2_MATERIALS = {
    "matte_aluminum": ((0.91, 0.92, 0.92), 0.45, 1.0, 1.0),
    "blue_anodized_aluminum": ((0.05, 0.14, 0.42), 0.32, 0.7, 1.0),    # dyed oxide over aluminium
    "brushed_steel": ((0.56, 0.57, 0.58), 0.30, 1.0, 1.0),
    "matte_brass": ((0.91, 0.78, 0.42), 0.45, 1.0, 1.0),
    "clear_glass": ((0.95, 0.95, 0.95), 0.02, 0.0, 0.12),
    "black_low_gloss_plastic": ((0.035, 0.035, 0.035), 0.55, 0.0, 1.0),
    "black_soft_touch_plastic": ((0.03, 0.03, 0.03), 0.75, 0.0, 1.0),
    "dark_grey_satin_finish_plastic": ((0.09, 0.09, 0.09), 0.45, 0.0, 1.0),
    "dark_grey_medium_gloss_plastic": ((0.09, 0.09, 0.09), 0.35, 0.0, 1.0),
    "white_low_gloss_plastic": ((0.78, 0.78, 0.76), 0.50, 0.0, 1.0),
    # the subsets left on the pure-black material without a telling name: black plastic / carbon parts
    "black_unnamed": ((0.03, 0.03, 0.032), 0.45, 0.0, 1.0),
}
# green plastics keep their CAD colour, with a plastic's roughness
LMF2_KEEP_COLOUR = {"green_low_gloss_plastic": 0.55, "green_high_gloss_plastic": 0.20}


def subset_stem(name: str) -> str:
    """CAD appearance name of a GeomSubset: 'blue_anodized_aluminum3' -> 'blue_anodized_aluminum'."""
    return re.sub(r"_?\d+$", "", name)


def _diffuse_of(material) -> tuple[float, float, float] | None:
    """The constant diffuse colour of an OmniPBR or UsdPreviewSurface material, if it has one."""
    from pxr import UsdShade

    for prim in material.GetPrim().GetChildren():
        sh = UsdShade.Shader(prim)
        if not sh:
            continue
        for name in ("diffuse_color_constant", "diffuseColor"):
            inp = sh.GetInput(name)
            if inp and inp.Get() is not None:
                return tuple(float(c) for c in inp.Get())
    return None


def apply_lmf2_materials(stage, root: str, looks_root: str = "/World/Looks/LMF2") -> dict:
    """Bind physically based materials to the LMF2's CAD appearance groups; returns {material: faces}."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    from .scene import preview_surface

    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.IsInstanceable() and prim.GetName() == "visuals":
            prim.SetInstanceable(False)               # the subsets must be editable
    mats: dict[str, object] = {}

    def material(key, colour=None, roughness=None):
        if key not in mats:
            c, r, m, op = LMF2_MATERIALS.get(key, ((0.5, 0.5, 0.5), 0.5, 0.0, 1.0))
            mat = preview_surface(stage, f"{looks_root}/{key}", color=colour or c,
                                  roughness=r if roughness is None else roughness, metallic=m)
            if op < 1.0:
                pbr = UsdShade.Shader(stage.GetPrimAtPath(f"{looks_root}/{key}/PBR"))
                pbr.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(op))
                pbr.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)
            mats[key] = mat
        return mats[key]

    faces: dict[str, int] = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        for subset in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Mesh(prim)):
            sp = subset.GetPrim()
            stem = subset_stem(sp.GetName())
            bound = UsdShade.MaterialBindingAPI(sp).GetDirectBinding().GetMaterial()
            colour = _diffuse_of(bound) if bound else None
            if stem in LMF2_MATERIALS:
                key, mat = stem, material(stem)
            elif stem in LMF2_KEEP_COLOUR and colour is not None:
                key = stem + "_" + "".join(f"{min(255, int(round(c * 255))):02x}" for c in colour)
                mat = material(key, colour, LMF2_KEEP_COLOUR[stem])
            elif colour is not None and max(colour) < 0.01:
                key, mat = "black_unnamed", material("black_unnamed")
            else:
                continue                              # a real CAD colour (connectors, LEDs, labels): keep it
            UsdShade.MaterialBindingAPI.Apply(sp).Bind(mat)
            faces[key] = faces.get(key, 0) + len(subset.GetIndicesAttr().Get() or [])
    return faces


# ---------------------------------------------------------------------------
# Camera mounts
# ---------------------------------------------------------------------------

def body_points(stage, body_path: str, max_points: int = 400000) -> np.ndarray:
    """Vertices of every mesh under ``body_path`` (instances included), in the body's frame, (N, 3)."""
    from pxr import Usd, UsdGeom

    body = stage.GetPrimAtPath(body_path)
    if not body:
        return np.zeros((0, 3))
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    to_body = np.array(cache.GetLocalToWorldTransform(body).GetInverse())
    out = []
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not pts:
            continue
        p = np.asarray(pts, float)
        m = np.array(cache.GetLocalToWorldTransform(prim)) @ to_body      # USD row-vector matrices
        out.append(p @ m[:3, :3] + m[3, :3])
    if not out:
        return np.zeros((0, 3))
    pts = np.concatenate(out)
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
    return pts


@dataclass(frozen=True)
class Mount:
    name: str
    look: float                     # +1 forward (+X), -1 backward
    position: tuple[float, float, float]      # lens front (the camera prim) in the body frame
    surface_x: float                # where the body ends behind the camera (signed, body frame)
    bracket_mm: float


def mount_cameras(points: np.ndarray, rig, camera_z: float, *, pitch_down_deg: float = 10.0,
                  clearance_m: float = 0.016, margin_m: float = 0.006,
                  z_options=(0.0, 0.02, 0.04, 0.06, -0.02)) -> dict[str, Mount]:
    """Front and back mounts on the centre line: the housing's rear face ``clearance_m`` (room for the M12
    connector) beyond the outermost body vertex inside the space the tilted housing sweeps (+ ``margin_m``).
    Each camera tries the heights ``camera_z + z_options`` and takes the one that sticks out least, as you would
    when fitting a bracket (a part sticking out low at the tail pushes a tilted camera up, not out)."""
    w, h, d = (v * 1e-3 for v in rig.body.size_mm)
    length = rig.length_mm() * 1e-3
    half_w, half_h = max(w, rig.lens.diameter_mm * 1e-3) / 2 + margin_m, h / 2 + margin_m
    pitch = math.radians(pitch_down_deg)
    mounts = {}
    for name, look in (("front", 1.0), ("back", -1.0)):
        best = None
        for dz in sorted(z_options, key=abs):
            z = camera_z + dz
            sel = (np.abs(points[:, 1]) <= half_w) & (np.abs(points[:, 2] - z) <= half_h + length * math.sin(pitch))
            ahead = look * points[sel, 0]
            surf = max(float(ahead.max()) if ahead.size else 0.0, 0.0)
            if best is None or surf < best[0] - 1e-3:
                best = (surf, z)
        surface, z_mount = best
        rear = np.array([look * (surface + clearance_m), 0.0, z_mount])
        fwd = np.array([look * math.cos(pitch), 0.0, -math.sin(pitch)])
        lens_front = rear + fwd * length
        mounts[name] = Mount(name, look, tuple(float(v) for v in lens_front), look * surface,
                             bracket_mm=(clearance_m + 0.010) * 1e3)
    return mounts


def payload_report(vehicle: Vehicle, rig, n_cameras: int = 2) -> str:
    grams = n_cameras * (rig.mass_g() + 15.0)          # + bracket and cable
    ok = grams * 1e-3 <= vehicle.payload_kg
    return (f"{n_cameras} x {rig.body.model} + {rig.lens.name} ~ {grams:.0f} g with brackets; {vehicle.name} payload "
            f"~{vehicle.payload_kg * 1e3:.0f} g: " + ("OK" if ok else "TOO HEAVY for the real drone (it could not take off)"))


# ---------------------------------------------------------------------------
# Third-person framing
# ---------------------------------------------------------------------------

def frame_scene(cfg, vehicle: Vehicle, names=("demo", "hero", "garden")):
    """A copy of ``scene.SceneCfg`` whose third-person cameras keep their framing around a larger drone."""
    if vehicle.view_scale == 1.0:
        return cfg
    cams = []
    for c in cfg.cameras:
        if c.name in names:
            t = np.asarray(c.target, float)
            eye = t + (np.asarray(c.eye, float) - t) * vehicle.view_scale
            eye[2] = max(eye[2], 0.2)                  # stay above the lawn
            c = replace(c, eye=tuple(float(v) for v in eye))
        cams.append(c)
    return replace(cfg, cameras=cams)
