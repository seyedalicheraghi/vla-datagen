"""Render a scenario in Isaac Sim and record its cameras to video.

    ./isaaclab.sh -p scripts/heavy_machines/scenarios/render_scenario.py --scenario adt_approach_hex --headless
    ./isaaclab.sh -p scripts/heavy_machines/scenarios/render_scenario.py --scenario adt_approach_hex \
        --renderer rtx --cameras front            # fast real-time RTX instead of path tracing

Output: <out>/<camera>/frame_NNNNN.png (with the camera look), <out>/<camera>_raw/ (with --save_raw),
<out>/<scenario>_<camera>.mp4 (ffmpeg or imageio-ffmpeg if available, BT.709-tagged) and
<out>/textures/ (baked maps, cached: a re-render with the same scenario and options skips the bake).
Exit code 0 only if every frame and video was written.

Textures are baked by ``bake.py`` in a separate process *before* Isaac Sim starts (parallel, cached;
worker processes inside Kit are not safe).  The scene is authored by ``usd_scene.build_scene`` (pure pxr — validated offline by
test/test_usd_scene.py): terrain split into seamless parts with baked colour / normal / roughness maps,
boulders and gravel instanced from irregular rock meshes, and render twins of the machines (bevelled
meshes with dusty-paint / rubber / steel / glass PBR texture sets) posed every frame from forward
kinematics.  This script adds the HDR sky and sun, path tracing (or real-time RTX), the cameras, and the
camera look (depth haze, tone curve, lens, grain) applied to every recorded frame.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import math
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from isaaclab.app import AppLauncher  # noqa: E402

from heavy_machines.scenarios import bake  # noqa: E402  (numpy + Pillow only)

parser = argparse.ArgumentParser(description="Render a heavy-machine scenario to video.")
parser.add_argument("--scenario", default="adt_approach_hex")
parser.add_argument("--out", default="heavy_machine_videos/{scenario}")
parser.add_argument("--cameras", nargs="*", help="camera rigs to record (default: all)")
parser.add_argument("--renderer", choices=["pathtraced", "rtx"], default="pathtraced",
                    help="pathtraced = offline quality (slow); rtx = real-time ray tracing")
parser.add_argument("--spp", type=int, default=32, help="path-tracing samples per render call")
parser.add_argument("--subframes", type=int, default=4, help="render calls accumulated per video frame")
parser.add_argument("--warmup", type=int, default=40, help="rendered frames before recording")
parser.add_argument("--bake_workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="processes for the texture bake (runs before Isaac Sim starts)")
bake.add_cli_options(parser)
parser.add_argument("--look", choices=["camera", "none"], default="camera", help="post-processed camera look")
parser.add_argument("--save_raw", action="store_true", help="also keep the frames before the camera look")
parser.add_argument("--no_sky_hdr", action="store_true", help="plain sky colour instead of the Nucleus HDR")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

OUT = os.path.abspath(os.path.expanduser(args_cli.out.format(scenario=args_cli.scenario)))
TEX_DIR = os.path.join(OUT, "textures")
OPTS = bake.options_from_args(args_cli)


def run_bake() -> None:
    """Bake (or reuse) the textures in a plain Python process before Kit starts."""
    cmd = [sys.executable, os.path.abspath(bake.__file__), "--scenario", args_cli.scenario, "--tex_dir", TEX_DIR,
           "--workers", str(args_cli.bake_workers)]
    for f in dataclasses.fields(bake.SceneOptions):
        if hasattr(args_cli, f.name):
            cmd += [f"--{f.name}", str(getattr(args_cli, f.name))]
    print("[scenario] baking textures:", " ".join(cmd), flush=True)
    if subprocess.run(cmd, check=False).returncode != 0:
        print("[scenario] texture bake FAILED", flush=True)
        sys.exit(3)


run_bake()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, check_file_path  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402

from heavy_machines import kinematics as K  # noqa: E402
from heavy_machines.scenarios import camera_look  # noqa: E402
from heavy_machines.scenarios.preview import load  # noqa: E402
from heavy_machines.scenarios.scenario import Scenario, sun_direction  # noqa: E402
from heavy_machines.scenarios.usd_scene import build_scene, pose_machines  # noqa: E402


def _rotation_taking(a, b):
    """Rotation matrix taking unit vector a onto unit vector b."""
    a, b = K.vunit(a), K.vunit(b)
    axis = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    s = K.vnorm(axis)
    c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    if s < 1e-9:
        return K.IDENTITY if c > 0 else K.axis_angle_to_mat((1.0, 0.0, 0.0), math.pi)
    return K.axis_angle_to_mat(axis, math.atan2(s, c))


def build_lights(sc: Scenario) -> None:
    L = sc.lighting
    hdr = f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/{L.sky_hdr}"
    use_hdr = (not args_cli.no_sky_hdr) and check_file_path(hdr) != 0
    if not args_cli.no_sky_hdr and not use_hdr:
        print("[scenario] sky HDR unreachable -> plain sky colour", flush=True)
    if use_hdr:
        dome = sim_utils.DomeLightCfg(intensity=L.sky_intensity, texture_file=hdr)
    else:
        dome = sim_utils.DomeLightCfg(intensity=L.sky_intensity, color=L.sky_color)
    dome.func("/World/Sky", dome)
    sun = sim_utils.DistantLightCfg(intensity=L.sun_intensity, color=L.sun_color, angle=0.53)
    # a UsdLux DistantLight shines along its local -Z axis
    R = _rotation_taking((0.0, 0.0, -1.0), sun_direction(L))
    sun.func("/World/Sun", sun, orientation=K.mat_to_quat_wxyz(R))


def write_video(frame_dir: str, fps: int, path: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            ffmpeg = None
    # tag BT.709 so players decode the colours the way they were encoded
    cmd = [ffmpeg or "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
           "-i", os.path.join(frame_dir, "frame_%05d.png"),
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-c:v", "libx264", "-crf", "16", path]
    if ffmpeg is None:
        print("[scenario] ffmpeg not found — encode with:\n  " + " ".join(cmd), flush=True)
        return False
    ok = subprocess.run(cmd, check=False).returncode == 0
    print(f"[scenario] wrote {path}" if ok else f"[scenario] ffmpeg FAILED for {path}", flush=True)
    return ok


def main():
    sc = load(args_cli.scenario)
    out = OUT
    rigs = [c for c in sc.cameras if not args_cli.cameras or c.name in args_cli.cameras]
    if not rigs:
        print(f"[scenario] no camera matches {args_cli.cameras}; available: {[c.name for c in sc.cameras]}",
              flush=True)
        os._exit(2)
    dt = 1.0 / sc.fps

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=dt, render_interval=1, device=args_cli.device,
        render=sim_utils.RenderCfg(rendering_mode=args_cli.rendering_mode or "quality",
                                   antialiasing_mode="DLAA", enable_translucency=True,
                                   enable_shadows=True, enable_ambient_occlusion=True,
                                   enable_reflections=True, enable_global_illumination=True)))
    if args_cli.renderer == "pathtraced":
        for key, value in (("/rtx/rendermode", "PathTracing"), ("/rtx/pathtracing/spp", args_cli.spp),
                           ("/rtx/pathtracing/totalSpp", args_cli.spp * max(1, args_cli.subframes)),
                           ("/rtx/pathtracing/optixDenoiser/enabled", True),
                           ("/rtx/pathtracing/maxBounces", 6)):
            # set_setting never raises for an unknown key — it silently creates it, so check first
            if sim.get_setting(key) is None:
                print(f"[scenario] WARNING: {key} is not a registered renderer setting in this Isaac Sim "
                      "version; it may have no effect", flush=True)
            sim.set_setting(key, value)

    # finds the cache written by run_bake(); bakes in-process only if it is missing
    link_ops = build_scene(sim.stage, sc, TEX_DIR, OPTS, log=lambda m: print(m, flush=True))
    build_lights(sc)

    sim_utils.create_prim("/World/Cameras", "Xform")
    cams: dict[str, Camera] = {}
    for rig in rigs:
        cams[rig.name] = Camera(CameraCfg(
            prim_path=f"/World/Cameras/{rig.name}", update_period=0.0,
            height=rig.height, width=rig.width, data_types=["rgb", "distance_to_camera"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=rig.focal_length,
                                             horizontal_aperture=rig.horizontal_aperture,
                                             clipping_range=(0.05, 3000.0)),
            offset=CameraCfg.OffsetCfg(convention="world")))

    sim.reset()
    if not args_cli.headless:
        try:
            from omni.kit.viewport.utility import get_active_viewport
            get_active_viewport().set_active_camera(f"/World/Cameras/{rigs[0].name}")
        except Exception as exc:  # noqa: BLE001
            print(f"[scenario] viewport camera not switched: {exc}", flush=True)

    def apply_state(t: float) -> None:
        pose_machines(link_ops, sc, t)
        for rig in rigs:
            tf = sc.camera_tf(rig, t)
            cams[rig.name].set_world_poses(
                torch.tensor([list(tf.t)], device=sim.device, dtype=torch.float32),
                torch.tensor([list(K.mat_to_quat_wxyz(tf.R))], device=sim.device, dtype=torch.float32),
                convention="world")

    def advance() -> None:
        sim.step(render=True)
        for _ in range(max(0, args_cli.subframes - 1)):
            sim.render()
        for cam in cams.values():
            cam.update(dt)

    print(f"[scenario] warming up the renderer ({args_cli.warmup} frames) ...", flush=True)
    for _ in range(args_cli.warmup):
        apply_state(0.0)
        advance()
    for _ in range(3000):                       # textures / sky are streamed asynchronously
        if not SimulationManager.assets_loading():
            break
        apply_state(0.0)
        advance()

    dirs, looks = {}, {}
    for rig in rigs:
        dirs[rig.name] = os.path.join(out, rig.name)
        for d in (dirs[rig.name], dirs[rig.name] + "_raw"):
            os.makedirs(d, exist_ok=True)
            for old in glob.glob(os.path.join(d, "frame_*.png")):   # no stale frames from older runs
                os.remove(old)
        if args_cli.look == "camera":
            # a fixed horizon line keeps the sky glow from flickering when machines cross it
            f_px = rig.width * rig.focal_length / rig.horizontal_aperture
            row, tilt = camera_look.horizon_from_camera(rig.height, f_px, pitch_down_deg=math.degrees(rig.rpy[1]))
            looks[rig.name] = camera_look.LookParams(horizon_row=row, horizon_tilt=tilt)
            camera_look.warm_up(rig.height, rig.width, looks[rig.name])

    print(f"[scenario] recording {sc.n_frames} frames at {sc.fps} fps ({args_cli.renderer}) → {out}", flush=True)
    completed = True
    for k in range(sc.n_frames):
        if not simulation_app.is_running():
            completed = False
            break
        apply_state(k * dt)
        advance()
        for rig in rigs:
            data = cams[rig.name].data.output
            rgb = data["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
            if args_cli.save_raw:
                Image.fromarray(rgb).save(os.path.join(dirs[rig.name] + "_raw", f"frame_{k:05d}.png"))
            if rig.name in looks:
                depth = data["distance_to_camera"][0, ..., 0].detach().cpu().numpy()
                rgb = camera_look.apply(rgb, depth, looks[rig.name], seed=k)
            Image.fromarray(rgb).save(os.path.join(dirs[rig.name], f"frame_{k:05d}.png"))
        if k % sc.fps == 0:
            print(f"[scenario] t = {k * dt:4.1f} s", flush=True)

    ok = [write_video(dirs[r.name], sc.fps, os.path.join(out, f"{sc.name}_{r.name}.mp4")) for r in rigs]
    print("[scenario] done", flush=True)
    # Isaac Sim's close() can hang on background threads; everything is written already.
    os._exit(0 if (completed and all(ok)) else 1)


if __name__ == "__main__":
    main()
