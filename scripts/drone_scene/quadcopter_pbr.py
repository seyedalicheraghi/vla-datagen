"""The Crazyflie quadcopter demo (scripts/demos/quadcopter.py) in a photorealistic PBR backyard.

Same behaviour as the original demo (the drone hovers in place: each propeller pushes a quarter of its
weight, reset every 10 s), plus:

- scenery from ``scene.py``: photo-scanned PBR ground (clay pavers, lawn with 3D grass, mulch bed,
  concrete curb), NVIDIA shrubs and a maple, fallen leaves, an HDR sky with its sun replaced by a
  calibrated DistantLight, and real PBR materials on the drone;
- ``--view demo|hero|top`` picks the viewport camera (``demo`` = the original view);
- ``--pathtracing`` switches the renderer to path tracing (offline quality);
- ``--record SECONDS`` renders the chosen cameras to PNG frames and an mp4 (use with --headless).

    ./isaaclab.sh -p scripts/drone_scene/quadcopter_pbr.py                         # GUI
    ./isaaclab.sh -p scripts/drone_scene/quadcopter_pbr.py --headless --record 4 --cameras hero demo --pathtracing

Assets are downloaded and pre-processed by ``prepare.py`` before Isaac Sim starts (cached in
~/.cache/drone_scene).  Output and log: <out>/ (default ./drone_scene_videos).
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_scene import prepare as prep  # noqa: E402  (no pxr, no Isaac)
from drone_scene import scene  # noqa: E402

parser = argparse.ArgumentParser(description="Crazyflie hover demo in a PBR backyard.")
CAMERA_NAMES = [c.name for c in scene.SceneCfg().cameras]
parser.add_argument("--view", default="demo", choices=CAMERA_NAMES, help="viewport camera")
parser.add_argument("--pathtracing", action="store_true", help="path-traced rendering (slower, best quality)")
parser.add_argument("--spp", type=int, default=64, help="path-tracing samples per pixel per frame")
parser.add_argument("--record", type=float, default=0.0, help="seconds of video to render (0 = interactive)")
parser.add_argument("--cameras", nargs="+", default=["hero", "demo"], choices=CAMERA_NAMES, help="cameras to record")
parser.add_argument("--fps", type=int, default=25, help="video frame rate (25 = exactly every 8th 200 Hz physics step)")
parser.add_argument("--out", default="drone_scene_videos")
parser.add_argument("--cache", default=prep.DEFAULT_CACHE)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
OUT = os.path.abspath(os.path.expanduser(args_cli.out))
os.makedirs(OUT, exist_ok=True)
LOG_PATH = os.path.join(OUT, "drone_scene.log")
_log = open(LOG_PATH, "w", encoding="utf-8")


def say(msg: str) -> None:
    _log.write(msg + "\n")
    _log.flush()
    print(msg, flush=True)


def fail(code: int) -> None:
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__      # AppLauncher silences stdout while starting
    if sys.exc_info()[0] is not None:
        say(traceback.format_exc().rstrip())
    say(f"[drone] FAILED (exit code {code}) - details in {LOG_PATH}")
    _log.close()
    os._exit(code)


try:
    MANIFEST = prep.prepare(os.path.expanduser(args_cli.cache), mirror=False, log=say)
except BaseException:  # noqa: BLE001
    say("[drone] could not download/prepare the scene assets")
    fail(3)

if args_cli.record > 0:
    args_cli.enable_cameras = True
try:
    say("[drone] starting Isaac Sim ...")
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import numpy as np
    import torch
    from PIL import Image

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.sensors import Camera, CameraCfg

    from isaaclab_assets import CRAZYFLIE_CFG  # isort:skip
    from isaacsim.core.simulation_manager import SimulationManager
except BaseException:  # noqa: BLE001
    fail(4)


def write_video(frame_dir: str, fps: float, path: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            say(f"[drone] no ffmpeg: frames are in {frame_dir}")
            return False
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-i", os.path.join(frame_dir, "frame_%05d.png"),
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-colorspace", "bt709",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-c:v", "libx264", "-crf", "16", path]
    res = subprocess.run(cmd, check=False, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in res.stdout.splitlines():
        say("[ffmpeg] " + line)
    say(f"[drone] wrote {path}" if res.returncode == 0 else f"[drone] ffmpeg FAILED for {path}")
    return res.returncode == 0


def main() -> None:
    cfg = scene.SceneCfg()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=0.005, device=args_cli.device,
        render=sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="DLAA", enable_translucency=True,
                                   enable_shadows=True, enable_ambient_occlusion=True, enable_reflections=True,
                                   enable_global_illumination=True)))
    if args_cli.pathtracing:
        for key, value in (("/rtx/rendermode", "PathTracing"), ("/rtx/pathtracing/spp", args_cli.spp),
                           ("/rtx/pathtracing/totalSpp", args_cli.spp), ("/rtx/pathtracing/maxBounces", 8),
                           ("/rtx/pathtracing/optixDenoiser/enabled", True)):
            if sim.get_setting(key) is None:
                say(f"[drone] WARNING: {key} is not a registered renderer setting here; it may have no effect")
            sim.set_setting(key, value)

    stage = sim.stage
    summary = scene.build(stage, cfg, MANIFEST, physics=True)
    say(f"[drone] scenery: {summary}")

    robot_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)
    say(f"[drone] PBR materials on {scene.apply_drone_materials(stage, '/World/Crazyflie')} drone meshes")
    robot = Articulation(robot_cfg)

    view = cfg.camera(args_cli.view)
    sim.set_camera_view(eye=list(view.eye), target=list(view.target))
    if not args_cli.headless:
        # look through the authored camera itself so the view also gets its lens and depth of field
        try:
            from omni.kit.viewport.utility import get_active_viewport
            get_active_viewport().camera_path = f"/World/Cameras/{args_cli.view}"
        except Exception as exc:  # noqa: BLE001
            say(f"[drone] viewport keeps Isaac's default lens ({exc})")

    cams: dict[str, Camera] = {}
    if args_cli.record > 0:
        for name in args_cli.cameras:
            c = cfg.camera(name)
            spawn = sim_utils.PinholeCameraCfg(
                focal_length=c.focal_length, horizontal_aperture=c.horizontal_aperture,
                clipping_range=(0.01, 10000.0), f_stop=c.f_stop,
                focus_distance=c.focus_distance or float(np.linalg.norm(np.subtract(c.target, c.eye))))
            cams[name] = Camera(CameraCfg(prim_path=f"/World/RecordCameras/{name}", update_period=0.0,
                                          height=c.height, width=c.width, data_types=["rgb"], spawn=spawn,
                                          offset=CameraCfg.OffsetCfg(convention="world")))

    sim.reset()
    for name, cam in cams.items():
        c = cfg.camera(name)
        cam.set_world_poses_from_view(torch.tensor([c.eye], device=sim.device, dtype=torch.float32),
                                      torch.tensor([c.target], device=sim.device, dtype=torch.float32))

    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    robot_mass = robot.root_physx_view.get_masses().sum()
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()
    sim_dt = sim.get_physics_dt()
    say("[drone] setup complete")

    record_steps = int(round(args_cli.record / sim_dt))
    every = max(1, int(round(1.0 / (args_cli.fps * sim_dt))))
    video_fps = 1.0 / (every * sim_dt)               # exact, so the video plays in real time
    dirs = {}
    for name in cams:
        dirs[name] = os.path.join(OUT, name)
        os.makedirs(dirs[name], exist_ok=True)
        for old in glob.glob(os.path.join(dirs[name], "frame_*.png")):
            try:
                os.remove(old)
            except OSError:
                pass
    if cams:
        say("[drone] warming up the renderer and waiting for textures/assets to stream in ...")
        for _ in range(30):
            sim.render()
        for _ in range(3000):
            if not SimulationManager.assets_loading():
                break
            sim.render()

    count, frame = 0, 0
    while simulation_app.is_running():
        if count % 2000 == 0:                      # reset, as in the original demo
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
            robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()
        forces = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
        forces[..., 2] = robot_mass * gravity / 4.0
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces, torques=torch.zeros_like(forces), body_ids=prop_body_ids)
        robot.write_data_to_sim()
        capture = bool(cams) and count % every == 0
        sim.step(render=capture or not cams)
        robot.update(sim_dt)
        if capture:
            for name, cam in cams.items():
                cam.update(sim_dt)
                rgb = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
                Image.fromarray(rgb).save(os.path.join(dirs[name], f"frame_{frame:05d}.png"))
            frame += 1
            if frame % round(video_fps) == 0:
                say(f"[drone] recorded {frame / video_fps:.0f} s")
        count += 1
        if cams and count >= record_steps:
            break

    if cams:
        ok = [write_video(dirs[n], video_fps, os.path.join(OUT, f"drone_{n}.mp4")) for n in cams]
        say("[drone] done" if all(ok) else "[drone] finished, but some videos were not written")
        _log.close()
        os._exit(0 if all(ok) else 1)


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001
        fail(1)
    _log.close()
    simulation_app.close()
