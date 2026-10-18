"""Writing world geometry to USD: big triangle meshes with one GeomSubset per material, and the PBR material
library they bind to (pxr; numpy arrays go straight into Vt arrays).

A ``MaterialSpec`` is a UsdPreviewSurface recipe (textures or constants, real-world texture repeat, tint): the
same kind of material the backyard scene uses (``scene.preview_surface``), so Isaac Sim renders the world and the
backyard with one consistent shading model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .terrain import MeshData


@dataclass(frozen=True)
class MaterialSpec:
    name: str
    textures: dict = field(default_factory=dict)       # {'albedo', 'orm', 'normal', 'roughness'}: path or URL
    tile_m: float = 1.0                                 # real-world size of one texture repeat
    tint: tuple = (1.0, 1.0, 1.0)
    color: tuple = (0.5, 0.5, 0.5)                      # when there is no albedo texture
    roughness: float = 0.7
    metallic: float = 0.0
    opacity: float = 1.0
    flip_green: bool = True                             # NVIDIA Base normal maps are DirectX-style
    roughness_scale: float = 1.0

    def with_tint(self, name: str, tint) -> "MaterialSpec":
        t = tuple(float(a) * float(b) for a, b in zip(self.tint, tint))
        return MaterialSpec(name, dict(self.textures), self.tile_m, t, self.color, self.roughness, self.metallic,
                            self.opacity, self.flip_green, self.roughness_scale)


def define_material(stage, path: str, spec: MaterialSpec):
    """Author (or reuse) the UsdPreviewSurface material for a spec at ``path``."""
    from pxr import Sdf, UsdShade

    from ..scene import preview_surface

    prim = stage.GetPrimAtPath(path)
    if prim:
        return UsdShade.Material(prim)
    tex = spec.textures
    color = spec.color if not tex.get("albedo") else (1.0, 1.0, 1.0)
    mat = preview_surface(stage, path, albedo=tex.get("albedo"), orm=tex.get("orm"), normal=tex.get("normal"),
                          roughness_tex=tex.get("roughness"), color=tuple(c * t for c, t in zip(color, spec.tint))
                          if not tex.get("albedo") else color, roughness=spec.roughness, metallic=spec.metallic,
                          flip_green=spec.flip_green, tint=spec.tint, roughness_scale=spec.roughness_scale)
    if spec.opacity < 1.0:
        pbr = UsdShade.Shader(stage.GetPrimAtPath(f"{path}/PBR"))
        pbr.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(spec.opacity))
    return mat


def write_mesh(stage, path: str, mesh: MeshData, materials: list | None = None, looks_root: str = "/World/Looks/Map",
               double_sided: bool = True, subdivision: bool = False):
    """Author ``mesh`` at ``path``.  ``materials[i]`` is the MaterialSpec for material id i (mesh.material);
    each used id becomes a GeomSubset bound to its material.  Returns the UsdGeom.Mesh."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    m = UsdGeom.Mesh.Define(stage, path)
    pts = np.ascontiguousarray(mesh.points, dtype=np.float32)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(mesh.faces.reshape(-1), dtype=np.int32)))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.catmullClark if subdivision else UsdGeom.Tokens.none)
    m.CreateDoubleSidedAttr(bool(double_sided))
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, pts.min(0))), Gf.Vec3f(*map(float, pts.max(0)))]))
    if mesh.normals is not None:
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(mesh.normals, dtype=np.float32)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if mesh.uv is not None:
        st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                  UsdGeom.Tokens.faceVarying)
        st.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(mesh.uv.reshape(-1, 2), dtype=np.float32)))
    if materials and mesh.material is not None:
        ids = np.asarray(mesh.material)
        used = np.unique(ids)
        if len(used) == 1:
            UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(
                define_material(stage, f"{looks_root}/{materials[used[0]].name}", materials[used[0]]))
        else:
            for k in used:
                spec = materials[int(k)]
                faces = np.nonzero(ids == k)[0].astype(np.int32)
                sub = UsdGeom.Subset.CreateGeomSubset(m, spec.name, UsdGeom.Tokens.face,
                                                      Vt.IntArray.FromNumpy(faces), "materialBind")
                UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(
                    define_material(stage, f"{looks_root}/{spec.name}", spec))
            UsdGeom.Subset.SetFamilyType(m, "materialBind", UsdGeom.Tokens.partition)
    return m


def merge(meshes: list[MeshData]) -> MeshData:
    """One MeshData from several (points offset, all per-face/per-vertex arrays concatenated)."""
    meshes = [m for m in meshes if m is not None and len(m.faces)]
    if not meshes:
        return MeshData(np.zeros((0, 3)), np.zeros((0, 3), np.int32))
    off, faces = 0, []
    for m in meshes:
        faces.append(m.faces + off)
        off += len(m.points)
    has_n = all(m.normals is not None for m in meshes)
    has_uv = all(m.uv is not None for m in meshes)
    has_mat = all(m.material is not None for m in meshes)
    return MeshData(np.concatenate([m.points for m in meshes]), np.concatenate(faces).astype(np.int32),
                    np.concatenate([m.normals for m in meshes]) if has_n else None,
                    np.concatenate([m.uv for m in meshes]) if has_uv else None,
                    np.concatenate([m.material for m in meshes]) if has_mat else None)
