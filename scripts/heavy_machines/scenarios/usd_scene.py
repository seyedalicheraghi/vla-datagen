"""USD authoring for scenario scenes (pxr + numpy only — testable without Isaac Sim with ``usd-core``).

``build_scene(stage, sc, tex_dir, opts)`` authors terrain (seamless split meshes with baked colour,
normal and roughness maps), boulders/gravel (PointInstancers) and render twins of the machines (one
Xform per link holding bevelled, textured meshes).  ``pose_machines(ops, sc, t)`` sets the link
transforms for time ``t``.  Lights, cameras and rendering are added by ``render_scenario.py``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

from .. import kinematics as K
from .pbr_textures import (detail_px_per_m, machine_textures, rock_textures, save_texture_set,
                           terrain_detail_cavity, terrain_detail_maps)
from .rocks import contact_map, scatter
from .scenario import Scenario
from .terrain import Mesh, bake_albedo, split_terrain_meshes
from .visual_meshes import machine_visual_meshes


@dataclass
class SceneOptions:
    terrain_step: float = 0.15          # detailed terrain mesh spacing (m)
    texture_ppm: float = 24.0           # colour px/m in the detailed area
    texture_ppm_near: float = 48.0      # colour px/m along the camera path (scenario.near_rect)
    far_ppm: float = 1.0
    detail_factor: float = 2.5          # relief maps are baked this much denser than the colour
    boulders: int = 160
    gravel_per_m2: float = 1.2
    rock_tex_size: int = 512
    machine_tex_size: int = 1024
    bevel: float = 0.025


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def xform(stage, path: str) -> UsdGeom.Xform:
    return UsdGeom.Xform.Define(stage, path)


def _texture(stage, path: str, file: str, st_out, raw: bool, wrap: str):
    tex = UsdShade.Shader.Define(stage, path)
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(file)
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw" if raw else "sRGB")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(wrap)
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set(wrap)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
    return tex


def pbr_material(stage, path: str, albedo: str | None = None, roughness_tex: str | None = None,
                 normal_tex: str | None = None, color=(0.6, 0.6, 0.6), roughness: float = 0.9,
                 metallic: float = 0.0, opacity: float = 1.0, clearcoat: float = 0.0,
                 wrap: str = "repeat") -> UsdShade.Material:
    """UsdPreviewSurface with optional albedo / roughness / normal textures read through primvar 'st'."""
    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, f"{path}/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    pbr.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    pbr.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(float(clearcoat))
    pbr.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.25)
    st_out = None
    if albedo or roughness_tex or normal_tex:
        st = UsdShade.Shader.Define(stage, f"{path}/stReader")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st_out = st.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    if albedo:
        tex = _texture(stage, f"{path}/Albedo", albedo, st_out, raw=False, wrap=wrap)
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    else:
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    if roughness_tex:
        tex = _texture(stage, f"{path}/Roughness", roughness_tex, st_out, raw=True, wrap=wrap)
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
            tex.CreateOutput("r", Sdf.ValueTypeNames.Float))
    else:
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    if normal_tex:
        tex = _texture(stage, f"{path}/Normal", normal_tex, st_out, raw=True, wrap=wrap)
        tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2.0, 2.0, 2.0, 1.0))
        tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1.0, -1.0, -1.0, 0.0))
        pbr.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat


def define_mesh(stage, path: str, mesh: Mesh, material: UsdShade.Material | None, uv_scale: float = 1.0):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(mesh.points, dtype=np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.triangles), 3, dtype=np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(mesh.triangles.reshape(-1), dtype=np.int32)))
    m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(mesh.normals, dtype=np.float32)))
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    lo, hi = mesh.points.min(axis=0), mesh.points.max(axis=0)
    m.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
    st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(mesh.uvs * uv_scale, dtype=np.float32)))
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    return m


def tf_to_matrix(tf: K.Tf) -> Gf.Matrix4d:
    """Column-vector rigid transform -> USD row-vector matrix."""
    R, t = tf.R, tf.t
    return Gf.Matrix4d(R[0][0], R[1][0], R[2][0], 0.0,
                       R[0][1], R[1][1], R[2][1], 0.0,
                       R[0][2], R[1][2], R[2][2], 0.0,
                       t[0], t[1], t[2], 1.0)


def _png(arr: np.ndarray, path: str) -> str:
    Image.fromarray(arr).save(path)
    return path


# ---------------------------------------------------------------------------
# Scene parts
# ---------------------------------------------------------------------------

def _terrain_material(stage, sc: Scenario, tex_dir: str, name: str, rect, ppm: float, rocks, o: SceneOptions,
                      log=print) -> UsdShade.Material:
    """Colour (with baked cavity darkening and boulder contact shadows) + denser normal/roughness maps."""
    albedo = bake_albedo(sc.terrain, rect, ppm, seed=1).astype(np.float32)
    albedo *= (terrain_detail_cavity(sc.terrain, rect, ppm, seed=1).astype(np.float32) / 255.0)[..., None] ** 0.8
    if rocks is not None and len(rocks):
        albedo *= contact_map(rocks, rect, ppm)
    a_png = _png(np.clip(albedo + 0.5, 0, 255).astype(np.uint8), os.path.join(tex_dir, f"terrain_{name}_albedo.png"))
    dppm = detail_px_per_m(ppm, rect, factor=o.detail_factor)
    nrm, rough = terrain_detail_maps(sc.terrain, rect, dppm, seed=1)
    n_png = _png(nrm, os.path.join(tex_dir, f"terrain_{name}_normal.png"))
    r_png = _png(rough, os.path.join(tex_dir, f"terrain_{name}_roughness.png"))
    log(f"[scenario]   {name}: colour {ppm:.0f} px/m, relief {dppm:.0f} px/m")
    return pbr_material(stage, f"/World/Looks/Terrain_{name}", a_png, r_png, n_png, wrap="clamp")


def build_terrain(stage, sc: Scenario, tex_dir: str, o: SceneOptions, log=print):
    xform(stage, "/World/Terrain")
    log("[scenario] scattering boulders and gravel ...")
    rocks = scatter(sc.terrain, sc.local_rect, seed=6, keep_clear=sc.keep_clear, boulders=o.boulders,
                    gravel_per_m2=o.gravel_per_m2, grid_step=o.terrain_step, focus=sc.near_rect)
    log(f"[scenario] {len(rocks)} rocks ({rocks.n_boulders} boulders)")

    log("[scenario] baking terrain textures ...")
    regions = []
    if sc.near_rect is not None:
        regions.append(("near", sc.near_rect, o.texture_ppm_near))
    regions.append(("local", sc.local_rect, o.texture_ppm))
    mats = [_terrain_material(stage, sc, tex_dir, n, r, ppm, rocks, o, log) for n, r, ppm in regions]
    far_png = _png(bake_albedo(sc.terrain, sc.far_rect, o.far_ppm, seed=2, far=True),
                   os.path.join(tex_dir, "terrain_far_albedo.png"))
    mats.append(pbr_material(stage, "/World/Looks/Terrain_far", far_png, roughness=0.97, wrap="clamp"))

    log("[scenario] building terrain meshes ...")
    # one graded grid cut into parts that share their border vertices exactly (no seams / z-fighting)
    parts = split_terrain_meshes(sc.terrain, sc.local_rect, sc.far_rect, fine=o.terrain_step, cap=3.0,
                                 inner_rect=sc.near_rect)
    names = [n for n, _, _ in regions] + ["far"]
    for name, mesh, mat in zip(names, parts, mats):
        define_mesh(stage, f"/World/Terrain/{name.capitalize()}", mesh, mat)

    # rocks: three shade variants of the stone material, one PointInstancer each
    protos = rocks.prototypes()
    shade = rocks.shade if rocks.shade is not None else np.ones(len(rocks), np.float32)
    edges = np.quantile(shade, [1 / 3, 2 / 3]) if len(rocks) else np.array([0.9, 1.0])
    bucket = np.digitize(shade, edges)
    for b, tint in enumerate((0.86, 0.97, 1.08)):
        sel = np.nonzero(bucket == b)[0]
        if len(sel) == 0:
            continue
        ts = rock_textures(seed=11 + b, size=o.rock_tex_size, shade=tint)
        paths = save_texture_set(ts, tex_dir, f"rock_{b}")
        mat = pbr_material(stage, f"/World/Looks/Rock_{b}", paths["albedo"], paths["roughness"], paths["normal"])
        root = f"/World/Terrain/Rocks_{b}"
        inst = UsdGeom.PointInstancer.Define(stage, root)
        targets = []
        for i, pm in enumerate(protos):
            path = f"{root}/Prototypes/rock_{i}"
            define_mesh(stage, path, pm, mat, uv_scale=1.0 / ts.metres_per_tile)
            targets.append(Sdf.Path(path))
        inst.CreatePrototypesRel().SetTargets(targets)
        inst.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(rocks.proto[sel], dtype=np.int32)))
        inst.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(rocks.positions[sel], dtype=np.float32)))
        inst.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(rocks.scales[sel], dtype=np.float32)))
        half = rocks.yaws[sel].astype(np.float64) / 2.0
        inst.CreateOrientationsAttr(Vt.QuathArray([Gf.Quath(float(math.cos(h)), 0.0, 0.0, float(math.sin(h)))
                                                   for h in half]))
    return rocks


def build_machine_twins(stage, sc: Scenario, tex_dir: str, o: SceneOptions) -> dict[str, dict[str, object]]:
    """Render twins: one Xform per link (transform op set by ``pose_machines``), bevelled textured meshes."""
    materials: dict[tuple[str, str], tuple[UsdShade.Material, float]] = {}
    ops: dict[str, dict[str, object]] = {}
    for actor in sc.actors:
        spec = actor.spec
        xform(stage, f"/World/{actor.name}")
        ops[actor.name] = {}
        for link, items in machine_visual_meshes(spec, bevel=o.bevel).items():
            x = xform(stage, f"/World/{actor.name}/{link}")
            ops[actor.name][link] = x.AddTransformOp()
            for k, (mname, mesh) in enumerate(items):
                key = (spec.name, mname)
                if key not in materials:
                    rgb = spec.materials.get(mname, (0.6, 0.6, 0.6, 1.0))[:3]
                    ts = machine_textures(mname, tuple(rgb), seed=17 + len(materials), size=o.machine_tex_size)
                    paths = save_texture_set(ts, tex_dir, f"{spec.name}_{mname}")
                    mat = pbr_material(stage, f"/World/Looks/{spec.name}_{mname}", paths["albedo"],
                                       paths["roughness"], paths["normal"], metallic=ts.metallic,
                                       opacity=ts.opacity, clearcoat=ts.clearcoat)
                    materials[key] = (mat, ts.metres_per_tile)
                mat, mpt = materials[key]
                define_mesh(stage, f"/World/{actor.name}/{link}/{mname}_{k}", mesh, mat, uv_scale=1.0 / mpt)
    return ops


def pose_machines(ops: dict[str, dict[str, object]], sc: Scenario, t: float) -> None:
    for actor in sc.actors:
        base, q = actor.root_tf(t)
        tfs = K.link_transforms(actor.spec, q)
        for link, op in ops[actor.name].items():
            op.Set(tf_to_matrix(base @ tfs[link]))


def build_scene(stage, sc: Scenario, tex_dir: str, o: SceneOptions | None = None, log=print):
    """Terrain, rocks and machine twins.  Returns the link transform ops for ``pose_machines``."""
    o = o or SceneOptions()
    os.makedirs(tex_dir, exist_ok=True)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    xform(stage, "/World")
    stage.DefinePrim("/World/Looks", "Scope")
    build_terrain(stage, sc, tex_dir, o, log)
    log("[scenario] building machine render twins ...")
    ops = build_machine_twins(stage, sc, tex_dir, o)
    pose_machines(ops, sc, 0.0)
    return ops
