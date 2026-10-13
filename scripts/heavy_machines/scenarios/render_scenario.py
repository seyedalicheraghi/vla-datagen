"""Render a scenario in Isaac Sim and record its cameras to video.

    ./isaaclab.sh -p scripts/heavy_machines/scenarios/render_scenario.py --scenario adt_approach_hex --headless
    ./isaaclab.sh -p scripts/heavy_machines/scenarios/render_scenario.py --scenario adt_approach_hex \
        --cameras front chase --out ~/videos/adt_hex

Output: <out>/<camera>/frame_NNNNN.png and <out>/<scenario>_<camera>.mp4 (ffmpeg or
imageio-ffmpeg, if available — otherwise the ffmpeg command to run is printed).

The scene is built from the pure-Python scenario description: terrain meshes with
baked albedo textures (UsdPreviewSurface), scattered rocks, a sun (DistantLight)
and an HDR sky (DomeLight), the machines from the heavy_machines registry, and
camera rigs that follow the machines through forward kinematics.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Render a heavy-machine scenario to video.")
parser.add_argument("--scenario", default="adt_approach_hex")
parser.add_argument("--out", default="heavy_machine_videos/{scenario}")
parser.add_argument("--cameras", nargs="*", help="camera rigs to record (default: all)")
parser.add_argument("--warmup", type=int, default=60, help="rendered frames before recording (textures, RTX)")
parser.add_argument("--subframes", type=int, default=1, help="extra RTX renders per frame (smoother lighting)")
parser.add_argument("--texture_ppm", type=float, default=24.0, help="terrain texture pixels per metre")
parser.add_argument("--no_sky_hdr", action="store_true", help="plain sky colour instead of the Nucleus HDR")
parser.add_argument("--no_machine_material_tweaks", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402

from heavy_machines import kinematics as K  # noqa: E402
from heavy_machines.machine_env import articulation_cfg  # noqa: E402
from heavy_machines.scenarios.preview import load  # noqa: E402
from heavy_machines.scenarios.scenario import Scenario, sun_direction  # noqa: E402
from heavy_machines.scenarios.terrain import (Mesh, bake_albedo, grid_mesh,  # noqa: E402
                                              scatter_rocks)


# ---------------------------------------------------------------------------
# USD helpers
# ---------------------------------------------------------------------------

def _preview_material(stage, path: str, color=(0.6, 0.6, 0.6), roughness=0.9, texture: str | None = None,
                      metallic: float = 0.0) -> UsdShade.Material:
    """UsdPreviewSurface material, optionally with an albedo texture read through the 'st' primvar."""
    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, f"{path}/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    if texture:
        st = UsdShade.Shader.Define(stage, f"{path}/stReader")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, f"{path}/Albedo")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
    else:
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    return mat


def _define_mesh(stage, path: str, mesh: Mesh, material: UsdShade.Material) -> None:
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
    st.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(mesh.uvs, dtype=np.float32)))
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)


def _quat_from_mat(R) -> tuple[float, float, float, float]:
    return K.mat_to_quat_wxyz(R)


def _rotation_taking(a, b):
    """Rotation matrix taking unit vector a onto unit vector b."""
    a, b = K.vunit(a), K.vunit(b)
    axis = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    s = K.vnorm(axis)
    c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    if s < 1e-9:
        return K.IDENTITY if c > 0 else K.axis_angle_to_mat((1.0, 0.0, 0.0), math.pi)
    return K.axis_angle_to_mat(axis, math.atan2(s, c))


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

def build_terrain(sc: Scenario, stage, tex_dir: str) -> None:
    sim_utils.create_prim("/World/Terrain", "Xform")
    print("[scenario] baking terrain textures ...", flush=True)
    local_png = os.path.join(tex_dir, "terrain_local.png")
    far_png = os.path.join(tex_dir, "terrain_far.png")
    Image.fromarray(bake_albedo(sc.terrain, sc.local_rect, args_cli.texture_ppm, seed=1)).save(local_png)
    Image.fromarray(bake_albedo(sc.terrain, sc.far_rect, 1.0, seed=2, far=True)).save(far_png)
    local_mat = _preview_material(stage, "/World/Looks/TerrainLocal", texture=local_png, roughness=0.95)
    far_mat = _preview_material(stage, "/World/Looks/TerrainFar", texture=far_png, roughness=0.97)
    print("[scenario] building terrain meshes ...", flush=True)
    _define_mesh(stage, "/World/Terrain/Local", grid_mesh(sc.terrain, sc.local_rect, 0.25), local_mat)
    # The far mesh dips 0.3 m under the detailed area so the two never z-fight.
    _define_mesh(stage, "/World/Terrain/Far",
                 grid_mesh(sc.terrain, sc.far_rect, 6.0, sink_inside=sc.local_rect, sink=0.3), far_mat)

    rocks = scatter_rocks(sc.terrain, sc.local_rect, sc.rocks, keep_clear=sc.keep_clear)
    shades = [_preview_material(stage, f"/World/Looks/Rock{i}", color=(0.52 * s, 0.48 * s, 0.43 * s), roughness=0.85)
              for i, s in enumerate((0.75, 0.85, 0.95, 1.05, 1.15))]
    sim_utils.create_prim("/World/Terrain/Rocks", "Xform")
    for i, r in enumerate(rocks):
        sph = UsdGeom.Sphere.Define(stage, f"/World/Terrain/Rocks/rock_{i}")
        sph.CreateRadiusAttr(1.0)
        xf = UsdGeom.Xformable(sph)
        xf.AddTranslateOp().Set(Gf.Vec3d(r.x, r.y, r.z))
        xf.AddRotateZOp().Set(math.degrees(r.yaw))
        xf.AddScaleOp().Set(Gf.Vec3f(*r.size))
        mat = shades[min(int((r.shade - 0.75) / 0.4 * 5), 4)]
        UsdShade.MaterialBindingAPI.Apply(sph.GetPrim()).Bind(mat)


def build_lights(sc: Scenario) -> None:
    L = sc.lighting
    if args_cli.no_sky_hdr:
        dome = sim_utils.DomeLightCfg(intensity=L.sky_intensity, color=L.sky_color)
    else:
        dome = sim_utils.DomeLightCfg(intensity=L.sky_intensity,
                                      texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/{L.sky_hdr}")
    dome.func("/World/Sky", dome)
    sun = sim_utils.DistantLightCfg(intensity=L.sun_intensity, color=L.sun_color, angle=0.53)
    # a DistantLight shines along its local -Z axis
    R = _rotation_taking((0.0, 0.0, -1.0), sun_direction(L))
    sun.func("/World/Sun", sun, orientation=_quat_from_mat(R))


def tweak_machine_materials(stage, root: str) -> int:
    """Give the imported flat URDF colours more realistic paint / rubber / steel / glass responses."""
    changed = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if not prim.IsA(UsdShade.Shader):
            continue
        sh = UsdShade.Shader(prim)
        color_in = sh.GetInput("diffuseColor") or sh.GetInput("diffuse_color_constant")
        if not color_in or color_in.Get() is None:
            continue
        r, g, b = color_in.Get()
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        sat = max(r, g, b) - min(r, g, b)
        if b > r and b > 0.3 and lum < 0.5:          # glass
            rough, metal = 0.05, 0.0
        elif lum < 0.12:                             # rubber, black trim
            rough, metal = 0.85, 0.0
        elif sat < 0.08:                             # bare steel
            rough, metal = 0.35, 0.85
        else:                                        # painted body
            rough, metal = 0.38, 0.0
        for name, val in (("roughness", rough), ("metallic", metal),
                          ("reflection_roughness_constant", rough), ("metallic_constant", metal)):
            inp = sh.GetInput(name)
            if inp:
                inp.Set(float(val))
            elif name in ("roughness", "metallic") and sh.GetIdAttr().Get() == "UsdPreviewSurface":
                sh.CreateInput(name, Sdf.ValueTypeNames.Float).Set(float(val))
        changed += 1
    return changed


def write_video(frame_dir: str, fps: int, path: str) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            ffmpeg = None
    cmd = [ffmpeg or "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
           "-i", os.path.join(frame_dir, "frame_%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", path]
    if ffmpeg is None:
        print("[scenario] ffmpeg not found — encode with:\n  " + " ".join(cmd), flush=True)
        return
    subprocess.run(cmd, check=False)
    print(f"[scenario] wrote {path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sc = load(args_cli.scenario)
    out = os.path.abspath(os.path.expanduser(args_cli.out.format(scenario=sc.name)))
    tex_dir = os.path.join(out, "textures")
    os.makedirs(tex_dir, exist_ok=True)
    rigs = [c for c in sc.cameras if not args_cli.cameras or c.name in args_cli.cameras]
    dt = 1.0 / sc.fps

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=dt, render_interval=1, device=args_cli.device,
        render=sim_utils.RenderCfg(rendering_mode=args_cli.rendering_mode or "quality",
                                   antialiasing_mode="DLAA", enable_shadows=True,
                                   enable_ambient_occlusion=True, enable_reflections=True)))
    stage = sim.stage
    sim_utils.create_prim("/World/Looks", "Scope")
    build_terrain(sc, stage, tex_dir)
    build_lights(sc)

    robots: dict[str, Articulation] = {}
    for actor in sc.actors:
        base, q = actor.root_tf(0.0)
        cfg = articulation_cfg(actor.spec, f"/World/{actor.name}", fix_base=actor.static)
        cfg.init_state.pos = tuple(base.t)
        cfg.init_state.rot = _quat_from_mat(base.R)
        cfg.init_state.joint_pos = {j.name: q[j.name] for j in actor.spec.movable_joints}
        robots[actor.name] = Articulation(cfg)

    sim_utils.create_prim("/World/Cameras", "Xform")
    cams: dict[str, Camera] = {}
    for rig in rigs:
        cams[rig.name] = Camera(CameraCfg(
            prim_path=f"/World/Cameras/{rig.name}", update_period=0.0,
            height=rig.height, width=rig.width, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=rig.focal_length,
                                             horizontal_aperture=rig.horizontal_aperture,
                                             clipping_range=(0.05, 3000.0)),
            offset=CameraCfg.OffsetCfg(convention="world")))

    sim.reset()
    if not args_cli.no_machine_material_tweaks:
        n = sum(tweak_machine_materials(stage, f"/World/{a.name}") for a in sc.actors)
        print(f"[scenario] adjusted {n} machine shader(s)" + ("" if n else
              " — none editable (instanced asset); machines keep their flat URDF colours"), flush=True)
    if not args_cli.headless and rigs:
        try:
            from omni.kit.viewport.utility import get_active_viewport
            get_active_viewport().set_active_camera(f"/World/Cameras/{rigs[0].name}")
        except Exception as exc:  # noqa: BLE001
            print(f"[scenario] viewport camera not switched: {exc}", flush=True)

    def apply_state(t: float) -> None:
        for actor in sc.actors:
            robot = robots[actor.name]
            base, q = actor.root_tf(t)
            if not actor.static:
                pose = torch.tensor([[*base.t, *_quat_from_mat(base.R)]], device=sim.device, dtype=torch.float32)
                robot.write_root_pose_to_sim(pose)
                robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=sim.device))
            jp = torch.tensor([[q[n] for n in robot.joint_names]], device=sim.device, dtype=torch.float32)
            robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
            robot.set_joint_position_target(jp)
            robot.write_data_to_sim()
        for rig in rigs:
            tf = sc.camera_tf(rig, t)
            cams[rig.name].set_world_poses(
                torch.tensor([list(tf.t)], device=sim.device, dtype=torch.float32),
                torch.tensor([list(_quat_from_mat(tf.R))], device=sim.device, dtype=torch.float32),
                convention="world")

    def advance() -> None:
        sim.step(render=True)
        for _ in range(max(0, args_cli.subframes)):
            sim.render()
        for robot in robots.values():
            robot.update(dt)
        for cam in cams.values():
            cam.update(dt)

    print(f"[scenario] warming up the renderer ({args_cli.warmup} frames) ...", flush=True)
    for _ in range(args_cli.warmup):
        apply_state(0.0)
        advance()

    frame_dirs = {}
    for rig in rigs:
        frame_dirs[rig.name] = os.path.join(out, rig.name)
        os.makedirs(frame_dirs[rig.name], exist_ok=True)
    print(f"[scenario] recording {sc.n_frames} frames at {sc.fps} fps → {out}", flush=True)
    for k in range(sc.n_frames):
        if not simulation_app.is_running():
            break
        apply_state(k * dt)
        advance()
        for rig in rigs:
            rgb = cams[rig.name].data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
            Image.fromarray(rgb).save(os.path.join(frame_dirs[rig.name], f"frame_{k:05d}.png"))
        if k % sc.fps == 0:
            print(f"[scenario] t = {k * dt:4.1f} s", flush=True)

    for rig in rigs:
        write_video(frame_dirs[rig.name], sc.fps, os.path.join(out, f"{sc.name}_{rig.name}.mp4"))
    print("[scenario] done", flush=True)
    # Isaac Sim's close() can hang on background threads; everything is written already.
    os._exit(0)


if __name__ == "__main__":
    main()
