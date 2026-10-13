"""USD authoring for scenario scenes (pxr + numpy only — testable without Isaac Sim with ``usd-core``).

Textures are baked (and cached) by ``bake.py``.  ``build_scene(stage, sc, tex_dir, opts)`` authors terrain (seamless split meshes with baked colour,
normal and roughness maps), boulders/gravel (PointInstancers) and render twins of the machines (one
Xform per link holding bevelled, textured meshes).  ``pose_machines(ops, sc, t)`` sets the link
transforms for time ``t``.  Lights, cameras and rendering are added by ``render_scenario.py``.
"""

from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

from .. import kinematics as K
from .bake import ROCK_TINTS, SceneOptions, bake_assets, load_scatter, regions
from .scenario import Scenario
from .terrain import Mesh, split_terrain_meshes
from .visual_meshes import machine_visual_meshes


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
        st.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
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


# ---------------------------------------------------------------------------
# Scene parts
# ---------------------------------------------------------------------------

def build_terrain(stage, sc: Scenario, man: dict, o: SceneOptions, log=print):
    """Terrain parts (near / local / far) with their baked texture sets, and the rock instancers."""
    xform(stage, "/World/Terrain")
    files = man["files"]
    regs = regions(sc, o)
    mats = []
    for name, _, _ in regs:
        f = files[f"terrain_{name}"]
        mats.append(pbr_material(stage, f"/World/Looks/Terrain_{name}", f["albedo"], f["roughness"], f["normal"],
                                 wrap="clamp"))
    mats.append(pbr_material(stage, "/World/Looks/Terrain_far", files["terrain_far"]["albedo"], roughness=0.97,
                             wrap="clamp"))

    log("[scenario] building terrain meshes ...")
    # one graded grid cut into parts that share their border vertices exactly (no seams / z-fighting)
    parts = split_terrain_meshes(sc.terrain, sc.local_rect, sc.far_rect, fine=o.terrain_step, cap=3.0,
                                 inner_rect=sc.near_rect)
    for name, mesh, mat in zip([n for n, _, _ in regs] + ["far"], parts, mats):
        define_mesh(stage, f"/World/Terrain/{name.capitalize()}", mesh, mat)

    # rocks: three shade variants of the stone material, one PointInstancer each
    rocks = load_scatter(man["rocks"])
    protos = rocks.prototypes()
    shade = rocks.shade if rocks.shade is not None else np.ones(len(rocks), np.float32)
    edges = np.quantile(shade, [1 / 3, 2 / 3]) if len(rocks) else np.array([0.9, 1.0])
    bucket = np.digitize(shade, edges)
    for b in range(len(ROCK_TINTS)):
        sel = np.nonzero(bucket == b)[0]
        if len(sel) == 0:
            continue
        f = files[f"rock_{b}"]
        mat = pbr_material(stage, f"/World/Looks/Rock_{b}", f["albedo"], f["roughness"], f["normal"])
        root = f"/World/Terrain/Rocks_{b}"
        inst = UsdGeom.PointInstancer.Define(stage, root)
        targets = []
        for i, pm in enumerate(protos):
            path = f"{root}/Prototypes/rock_{i}"
            define_mesh(stage, path, pm, mat, uv_scale=1.0 / f["metres_per_tile"])
            targets.append(Sdf.Path(path))
        inst.CreatePrototypesRel().SetTargets(targets)
        inst.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(rocks.proto[sel], dtype=np.int32)))
        inst.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(rocks.positions[sel], dtype=np.float32)))
        inst.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(rocks.scales[sel], dtype=np.float32)))
        half = rocks.yaws[sel].astype(np.float64) / 2.0
        inst.CreateOrientationsAttr(Vt.QuathArray([Gf.Quath(float(math.cos(h)), 0.0, 0.0, float(math.sin(h)))
                                                   for h in half]))
    log(f"[scenario] {len(rocks)} rocks instanced")
    return rocks


def build_machine_twins(stage, sc: Scenario, man: dict, o: SceneOptions) -> dict[str, dict[str, object]]:
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
                    f = man["files"][f"machine:{spec.name}:{mname}"]
                    mat = pbr_material(stage, f"/World/Looks/{spec.name}_{mname}", f["albedo"], f["roughness"],
                                       f["normal"], metallic=f["metallic"], opacity=f["opacity"],
                                       clearcoat=f["clearcoat"])
                    materials[key] = (mat, f["metres_per_tile"])
                mat, mpt = materials[key]
                define_mesh(stage, f"/World/{actor.name}/{link}/{mname}_{k}", mesh, mat, uv_scale=1.0 / mpt)
    return ops


def pose_machines(ops: dict[str, dict[str, object]], sc: Scenario, t: float) -> None:
    for actor in sc.actors:
        base, q = actor.root_tf(t)
        tfs = K.link_transforms(actor.spec, q)
        for link, op in ops[actor.name].items():
            op.Set(tf_to_matrix(base @ tfs[link]))


def build_scene(stage, sc: Scenario, tex_dir: str, o: SceneOptions | None = None, log=print, workers: int = 1):
    """Terrain, rocks and machine twins (baking textures first unless cached).  Returns the link ops."""
    o = o or SceneOptions()
    man = bake_assets(sc, tex_dir, o, workers=workers, log=log)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    xform(stage, "/World")
    stage.DefinePrim("/World/Looks", "Scope")
    build_terrain(stage, sc, man, o, log)
    log("[scenario] building machine render twins ...")
    ops = build_machine_twins(stage, sc, man, o)
    pose_machines(ops, sc, 0.0)
    return ops
