"""Isaac Lab's quadcopter demo (scripts/demos/quadcopter.py) in a photorealistic PBR backyard, with two LUCID
Vision Labs cameras on the drone.

Same behaviour as the original demo (the drone hovers in place: each propeller pushes a quarter of its
weight, reset every 10 s), plus:

- scenery from ``scene.py``: photo-scanned PBR ground (clay pavers, lawn with 3D grass, mulch bed,
  concrete curb), NVIDIA shrubs and a maple, fallen leaves, an HDR sky with its sun replaced by a
  calibrated DistantLight;
- ``--vehicle lmf2`` (default): NTNU ARL's 1.2 kg LMF2 quadrotor with physically based metals, plastics and
  glass (``vehicles.py``), which can carry the cameras; ``--vehicle crazyflie``: the original 27 g demo drone
  (which in reality could not lift them);
- two LUCID cameras (``lucid.py``, default Triton TRI032S-CC + 4 mm lens): ``front`` looking forward and
  ``back`` looking backward, mounted on the drone's centre line just clear of its body and visible on it.
  Recording them gives what those cameras would deliver: Isaac Sim renders the linear HDR image
  (``HdrColor``) and ``sensor_sim.py`` adds, on the same GPU, the lens distortion, vignetting and lateral
  colour, auto exposure, photon and read noise of the Sony Pregius sensor, the 12-bit RGGB Bayer raw frame,
  demosaicing, white balance and the output gamma.  Each video gets a ``*_camera_info.json`` (intrinsics
  ``K``, distortion ``D``, mount pose, exposure per frame);
- ``--view demo|hero|garden|top|front|back`` picks the viewport camera (``demo`` = the original view);
- ``--pathtracing`` switches the renderer to path tracing (offline quality);
- ``--record SECONDS`` renders the chosen cameras to PNG frames and an mp4 (use with --headless);
- ``--fly`` flies the drone along ``flight.Flight`` (a 20 s garden tour) instead of hovering.

    ./isaaclab.sh -p scripts/drone_scene/quadcopter_pbr.py                         # GUI
    ./isaaclab.sh -p scripts/drone_scene/quadcopter_pbr.py --fly --view front     # GUI, onboard view
    ./isaaclab.sh -p scripts/drone_scene/quadcopter_pbr.py --fly --headless --record 20 --cameras front back
    ./isaaclab.sh -p scripts/drone_scene/quadcopter_pbr.py --headless --record 4 --cameras hero demo --pathtracing

Everything renders on the local GPU.  Assets are downloaded and pre-processed by ``prepare.py`` before Isaac Sim
starts (cached in ~/.cache/drone_scene).  Output and log: <out>/ (default ./drone_scene_videos).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_scene import flight  # noqa: E402  (numpy only)
from drone_scene import lucid  # noqa: E402  (numpy only)
from drone_scene import prepare as prep  # noqa: E402  (no pxr, no Isaac)
from drone_scene import scene  # noqa: E402
from drone_scene import vehicles  # noqa: E402

parser = argparse.ArgumentParser(description="Quadcopter demo with two LUCID cameras in a PBR backyard.")
ONBOARD = sorted(flight.CAMERAS)                    # cameras mounted on the drone (with --fly)
CAMERA_NAMES = [c.name for c in scene.SceneCfg().cameras] + ONBOARD + ["chase"]
parser.add_argument("--view", default="demo", choices=CAMERA_NAMES, help="viewport camera")
parser.add_argument("--pathtracing", action="store_true", help="path-traced rendering (slower, best quality)")
parser.add_argument("--spp", type=int, default=64, help="path-tracing samples per pixel per frame")
parser.add_argument("--record", type=float, default=0.0,
                    help="seconds of video to render (0 = interactive, -1 = the whole flight)")
parser.add_argument("--cameras", nargs="+", default=["hero", "demo"], choices=CAMERA_NAMES, help="cameras to record")
parser.add_argument("--fps", type=int, default=25, help="video frame rate (25 = exactly every 8th 200 Hz physics step)")
parser.add_argument("--fly", action="store_true", help="fly the 20 s garden tour (flight.py) instead of hovering")
parser.add_argument("--route", default="garden", choices=["garden", "straight"],
                    help="garden: the 20 s backyard tour; straight: take off from a real backyard, fly --distance m "
                         "straight over a real map built from open geodata (world/), land on grass")
parser.add_argument("--region", default=None, help="lat,lon to look for a straight route around (default: Ames, Iowa)")
parser.add_argument("--takeoff", default=None, help="lat,lon of the take-off spot (straight route)")
parser.add_argument("--landing", default=None, help="lat,lon of the landing spot (straight route)")
parser.add_argument("--distance", type=float, default=2000.0, help="straight route length (m)")
parser.add_argument("--when", default="2025-07-15T15:30:00Z", help="UTC date and time of the flight (sets the sun)")
parser.add_argument("--cruise_agl", type=float, default=40.0, help="cruise height above the take-off ground (m)")
parser.add_argument("--speed", type=float, default=10.0, help="cruise speed (m/s)")
parser.add_argument("--camera_pitch", type=float, default=None,
                    help="LUCID cameras tilted down by this (deg; default 10 garden, 20 straight)")
parser.add_argument("--save_frames", action="store_true", help="also keep every frame as a PNG (large)")
parser.add_argument("--vehicle", default=vehicles.DEFAULT_VEHICLE, choices=sorted(vehicles.VEHICLES),
                    help="the drone: lmf2 (1.2 kg, carries the cameras) or crazyflie (the original 27 g demo drone)")
parser.add_argument("--lucid", default=lucid.DEFAULT_CAMERA, choices=sorted(lucid.CAMERAS), help="LUCID camera model")
parser.add_argument("--lucid_lens", default=None, choices=sorted(lucid.LENSES),
                    help="lens on the LUCID cameras (default: the widest one that covers the sensor)")
parser.add_argument("--resolution_scale", type=float, default=1.0,
                    help="render the onboard cameras smaller for a quick test, e.g. 0.5 (not a camera mode)")
parser.add_argument("--supersample", type=float, default=1.0,
                    help="render the onboard cameras this much finer and filter down (sharper, slower)")
parser.add_argument("--exposure", default="auto", choices=["auto", "fixed"], help="LUCID exposure mode")
parser.add_argument("--exposure_us", type=float, default=500.0, help="exposure time with --exposure fixed")
parser.add_argument("--gain_db", type=float, default=0.0, help="analog gain with --exposure fixed")
parser.add_argument("--gamma", default="srgb", help="output gamma: srgb, or the camera's Gamma value (1 = linear)")
parser.add_argument("--save_raw", action="store_true", help="also save the 12-bit Bayer raw frames (16-bit PNG)")
parser.add_argument("--isaac_rgb", action="store_true",
                    help="record the onboard cameras as Isaac's tone-mapped pinhole image (no sensor simulation)")
parser.add_argument("--lens", action="store_true", help="third-person cameras: bloom, vignette, grain, colour fringe")
parser.add_argument("--motion_blur", action="store_true", help="real-time motion blur (not with --pathtracing)")
parser.add_argument("--aces", action="store_true", help="force the ACES tonemapper (enum unverified; see rtx_look)")
parser.add_argument("--out", default="drone_scene_videos")
parser.add_argument("--cache", default=prep.DEFAULT_CACHE)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.route == "straight":
    args_cli.fly = True
if "chase" in ({args_cli.view} | set(args_cli.cameras)) and args_cli.route != "straight":
    parser.error("the chase camera follows the straight route (--route straight)")
if not args_cli.fly and ({args_cli.view} | set(args_cli.cameras)) & set(ONBOARD):
    parser.error(f"the onboard cameras {ONBOARD} need --fly")
try:
    RIG = lucid.make_rig(args_cli.lucid, args_cli.lucid_lens, args_cli.resolution_scale)
    PLAN = RIG.render_plan(args_cli.supersample)
    GAMMA = args_cli.gamma if args_cli.gamma == "srgb" else float(args_cli.gamma)
except (KeyError, ValueError) as exc:
    parser.error(str(exc))
if args_cli.supersample < 1.0:
    parser.error("--supersample must be >= 1")
VEHICLE = vehicles.VEHICLES[args_cli.vehicle]
CAMERA_PITCH = args_cli.camera_pitch if args_cli.camera_pitch is not None else (20.0 if args_cli.route == "straight" else 10.0)
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


WORLD = None
if args_cli.route == "straight":
    try:
        from drone_scene.world import runner
        wargs = ["--distance", f"{args_cli.distance:g}", "--when", args_cli.when, "--cruise-agl", f"{args_cli.cruise_agl:g}"]
        for flag, val in (("--region", args_cli.region), ("--takeoff", args_cli.takeoff), ("--landing", args_cli.landing)):
            if val:
                wargs += [flag, val]
        WORLD = runner.ensure_world(os.path.expanduser(args_cli.cache), wargs, log=say)
        say(f"[drone] map: take-off {WORLD['takeoff']['lat']:.5f},{WORLD['takeoff']['lon']:.5f} -> landing "
            f"{WORLD['landing']['lat']:.5f},{WORLD['landing']['lon']:.5f} ({WORLD['distance_m']:.0f} m, heading "
            f"{WORLD['heading_deg']:.0f} deg), cruise at {WORLD['cruise_z']:.0f} m; sun {WORLD['sun']['elevation_deg']:.0f} deg up")
        say(f"[drone] {WORLD['attribution']}")
    except BaseException:  # noqa: BLE001
        say("[drone] could not build the map for the straight route")
        fail(3)
try:
    from drone_scene import assets
    if WORLD:
        SUN_EL = round(max(5.0, float(WORLD["sun"]["elevation_deg"])), 1)
        MANIFEST = prep.prepare(os.path.expanduser(args_cli.cache), mirror=False, sun_elevation_deg=SUN_EL, log=say,
                                sky=assets.WORLD_SKY)
    else:
        MANIFEST = prep.prepare(os.path.expanduser(args_cli.cache), mirror=False,
                                sun_elevation_deg=scene.SceneCfg().sun.elevation_deg, log=say)
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
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.utils.array import convert_to_torch

    from isaaclab_assets import CRAZYFLIE_CFG  # isort:skip
    from isaacsim.core.simulation_manager import SimulationManager

    from drone_scene import nucleus, sensor_sim
except BaseException:  # noqa: BLE001
    fail(4)


def find_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            return None
    return ffmpeg


class VideoWriter:
    """Frames go straight into ffmpeg (H.264, BT.709) through a pipe: a 4-minute 3 MP flight would be tens of GB
    as PNG frames.  Without ffmpeg (or with keep_png) every frame is also written as a PNG."""

    def __init__(self, path: str, frame_dir: str, fps: float, keep_png: bool = False):
        self.path, self.frame_dir, self.fps, self.n = path, frame_dir, fps, 0
        self.ffmpeg, self.proc, self.keep_png = find_ffmpeg(), None, keep_png
        if self.ffmpeg is None:
            self.keep_png = True
            say(f"[drone] no ffmpeg: {os.path.basename(path)} is written as PNG frames in {frame_dir}")

    def add(self, rgb) -> None:
        h, w = rgb.shape[:2]
        if self.ffmpeg and self.proc is None:
            cmd = [self.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                   "-framerate", f"{self.fps:g}", "-i", "-",
                   "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-colorspace", "bt709",
                   "-color_primaries", "bt709", "-color_trc", "bt709", "-c:v", "libx264", "-crf", "16",
                   "-preset", "medium", self.path]
            self.log = open(self.path + ".ffmpeg.log", "w")
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT)
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(rgb[..., :3], dtype=np.uint8).tobytes())
        if self.keep_png:
            Image.fromarray(rgb[..., :3]).save(os.path.join(self.frame_dir, f"frame_{self.n:05d}.png"))
        self.n += 1

    def close(self) -> bool:
        if self.proc is None:
            return self.keep_png and self.n > 0
        self.proc.stdin.close()
        ok = self.proc.wait() == 0
        self.log.close()
        with open(self.path + ".ffmpeg.log") as fh:
            for line in fh.read().splitlines():
                say("[ffmpeg] " + line)
        say(f"[drone] wrote {self.path} ({self.n} frames)" if ok else f"[drone] ffmpeg FAILED for {self.path}")
        if ok:
            os.remove(self.path + ".ffmpeg.log")
        return ok


def author_onboard_cameras(stage, body: str, geom: flight.Flight) -> None:
    """Plain USD cameras on the drone body for the GUI viewport (--view front/back): the pinhole render of the
    LUCID lens (the viewport shows it undistorted, without the sensor simulation)."""
    from pxr import Gf, UsdGeom

    for name, oc in geom.cameras.items():
        pos, (w, x, y, z) = geom.camera_local(name)
        cam = UsdGeom.Camera.Define(stage, f"{body}/{name}_view")
        cam.CreateFocalLengthAttr(oc.focal_length)
        cam.CreateHorizontalApertureAttr(oc.horizontal_aperture)
        cam.CreateVerticalApertureAttr(oc.vertical_aperture or oc.horizontal_aperture * 9 / 16)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10000.0))
        cam.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        cam.AddOrientOp().Set(Gf.Quatf(w, x, y, z))


def author_lucid_bodies(stage, body: str, geom: flight.Flight, mounts: dict) -> None:
    """The two cameras as visible PBR geometry on the drone, at the poses of the camera prims."""
    from pxr import Gf, UsdGeom

    for name in geom.cameras:
        pos, (w, x, y, z) = geom.camera_local(name)
        path = lucid.author_body(stage, f"{body}/{name}_lucid", RIG, bracket_mm=mounts[name].bracket_mm)
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        xf.AddOrientOp().Set(Gf.Quatf(w, x, y, z))


def spawn_vehicle(stage, cfg: scene.SceneCfg):
    """Spawn the drone, give it its PBR look and return its Articulation config."""
    if VEHICLE.key == "crazyflie":
        robot_cfg = CRAZYFLIE_CFG.replace(prim_path=VEHICLE.prim)
        robot_cfg.spawn.func(VEHICLE.prim, robot_cfg.spawn, translation=robot_cfg.init_state.pos)
        say(f"[drone] PBR materials on {scene.apply_drone_materials(stage, VEHICLE.prim)} drone meshes")
        from drone_scene import drone_look
        from drone_scene.preview import PROP_BLUR_GAIN
        look = drone_look.apply_drone_look(stage, VEHICLE.prim, MANIFEST.get("cache_dir") or prep.DEFAULT_CACHE,
                                           blur_gain=PROP_BLUR_GAIN)
        say(f"[drone] close-up detail: PCB/battery/motor textures, {look.get('leds')} LEDs, "
            f"{look.get('prop_discs')} propeller blur discs")
        return robot_cfg
    robot_cfg = ArticulationCfg(
        prim_path=VEHICLE.prim,
        spawn=sim_utils.UsdFileCfg(
            usd_path=nucleus.url(VEHICLE.usd),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0)),
        # the propellers are fixed joints: no degrees of freedom, so no joint defaults and no actuators
        init_state=ArticulationCfg.InitialStateCfg(pos=cfg.drone_pos, joint_pos={}, joint_vel={}),
        actuators={},
    )
    robot_cfg.spawn.func(VEHICLE.prim, robot_cfg.spawn, translation=robot_cfg.init_state.pos)
    faces = vehicles.apply_lmf2_materials(stage, VEHICLE.prim)
    say(f"[drone] {VEHICLE.name}: PBR materials on {sum(faces.values())} faces ({', '.join(sorted(faces))})")
    return robot_cfg


class ChaseCamera:
    """A third-person camera following the drone like a chase drone: behind and above it, lagging smoothly."""

    def __init__(self, back: float = 10.0, up: float = 3.5, lag_s: float = 0.8):
        self.back, self.up, self.lag = back, up, lag_s
        self.eye = None

    def update(self, p, direction, dt: float):
        p = np.asarray(p, float)
        want = p - np.asarray(direction, float) * self.back + np.array([0.0, 0.0, self.up])
        if self.eye is None:
            self.eye = want
        else:
            a = 1.0 - math.exp(-dt / self.lag)
            self.eye = self.eye + (want - self.eye) * a
        return [float(v) for v in self.eye], [float(v) for v in p]


class HdrReader:
    """Replicator's HdrColor (linear radiance before tone mapping) on a camera's render product."""

    def __init__(self, cam, device: str):
        import omni.replicator.core as rep

        self.device, self.shape = device, (PLAN["height"], PLAN["width"])
        self.annot = rep.AnnotatorRegistry.get_annotator("HdrColor", device="cuda" if "cuda" in device else "cpu")
        self.annot.attach(cam.render_product_paths[0])             # as Isaac Lab's Camera attaches its own

    def read(self):
        data = self.annot.get_data()
        if isinstance(data, dict):
            data = data.get("data")
        if data is None:
            return None
        t = convert_to_torch(data, device=self.device)
        if t.numel() < self.shape[0] * self.shape[1] * 3:
            return None
        return t.reshape(self.shape[0], self.shape[1], -1)


def scene_cfg() -> scene.SceneCfg:
    """The backyard's configuration; on the straight route it is the take-off backyard of the real map: the real sun,
    the pure sky (aimed so its glow sits around the sun) and only a small lawn of its own (the map's terrain is
    the ground beyond it)."""
    from dataclasses import replace

    from drone_scene import hdr
    cfg = scene.SceneCfg()
    if WORLD:
        az = float(WORLD["sun"]["azimuth_ccw_from_east_deg"])
        hs = MANIFEST["sky"].get("hdr_sun", {})
        cfg = replace(cfg, sun=scene.SunCfg(elevation_deg=SUN_EL, azimuth_deg=az), lawn_half=12.0,
                      sky_rotation_deg=hdr.dome_rotation_deg(hs["azimuth_deg"], az) if "azimuth_deg" in hs else 0.0)
    return vehicles.frame_scene(cfg, VEHICLE)


def make_flight(onboard: dict, robot_z0: float):
    """The flight (or just the camera geometry when hovering)."""
    if not WORLD:
        return flight.Flight(cameras=onboard)
    L = WORLD["landing"]
    start_z = robot_z0
    clearance = start_z - WORLD["takeoff"]["z"]
    cruise = max(WORLD["cruise_z"], start_z + args_cli.cruise_agl, L["z"] + clearance + 10.0)
    return flight.StraightFlight(start=(0.0, 0.0, start_z), end_ground_z=L["z"], distance=WORLD["distance_m"],
                                 heading_deg=WORLD["heading_deg"], cruise_z=cruise, speed=args_cli.speed,
                                 ground_clearance=clearance, cameras=onboard)


def main() -> None:
    cfg = scene_cfg()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=0.005, device=args_cli.device,
        render=sim_utils.RenderCfg(rendering_mode="quality", antialiasing_mode="DLAA", enable_translucency=True,
                                   enable_shadows=True, enable_ambient_occlusion=True, enable_reflections=True,
                                   enable_global_illumination=True)))
    # physically based camera pipeline (rtx_look.py): no artificial ambient fill (the "quality" preset turns it on,
    # so this must come after SimulationContext), fixed exposure, sun/sky sampling, GI bounces, path-tracer samples
    from drone_scene import rtx_look
    optional = [g for g, on in (("lens", args_cli.lens), ("tonemap_aces", args_cli.aces),
                                ("motion_blur", args_cli.motion_blur and not args_cli.pathtracing)) if on]
    if args_cli.pathtracing:
        pt, _ = rtx_look.path_tracing_samples(args_cli.spp)
        rtx_look.apply(sim.set_setting, sim.get_setting, "pathtracing", say, optional=optional, overrides=pt)
    else:
        rtx_look.apply(sim.set_setting, sim.get_setting, "realtime", say, optional=optional)

    stage = sim.stage
    summary = scene.build(stage, cfg, MANIFEST, physics=True)
    say(f"[drone] scenery: {summary}")
    if WORLD:
        from pxr import UsdGeom
        UsdGeom.Xform.Define(stage, "/World/Map").GetPrim().GetReferences().AddReference(WORLD["usd"])
        say(f"[drone] map: {WORLD['usd']} ({WORLD['stats']})")

    robot_cfg = spawn_vehicle(stage, cfg)
    robot = Articulation(robot_cfg)
    body = f"{VEHICLE.prim}/{VEHICLE.body}"

    # the LUCID cameras: mounted from the drone's own mesh, with the lens of the chosen rig
    pts = vehicles.body_points(stage, body)
    if len(pts) == 0:
        say(f"[drone] WARNING: no mesh found under {body}; the cameras are mounted at the body origin")
    mounts = vehicles.mount_cameras(pts, RIG, VEHICLE.camera_z, pitch_down_deg=CAMERA_PITCH)
    onboard = {n: flight.OnboardCamera(n, m.position, m.look, pitch_down_deg=CAMERA_PITCH,
                                       focal_length=PLAN["focal_length"],
                                       horizontal_aperture=PLAN["horizontal_aperture"],
                                       vertical_aperture=PLAN["vertical_aperture"]) for n, m in mounts.items()}
    # body origin above its landing gear, standing on the patio (backyard) or the map's take-off spot
    z_feet = float(pts[:, 2].min()) if len(pts) else 0.0
    from drone_scene.pavers import PaverSpec
    robot_z0 = cfg.patio_top + PaverSpec().top_rise - z_feet + 0.005
    geom = make_flight(onboard, robot_z0)           # camera poses (and the flight path, with --fly)
    fl = geom if args_cli.fly else None
    if WORLD:
        say(f"[drone] flight: {fl.duration:.0f} s - take-off, climb to {fl.cruise_z:.0f} m, {fl.distance:.0f} m at "
            f"{fl.speed:g} m/s heading {fl.heading_deg:.0f} deg, descend, land on grass")
    author_lucid_bodies(stage, body, geom, mounts)
    author_onboard_cameras(stage, body, geom)
    say(f"[drone] cameras: {lucid.describe(RIG)}")
    for n, m in mounts.items():
        say(f"[drone]   {n}: lens at {tuple(round(v, 3) for v in m.position)} m in the {VEHICLE.body} frame, "
            f"{CAMERA_PITCH:g} deg down; render {PLAN['width']}x{PLAN['height']} (x{PLAN['overscan']:.2f} for the distortion)")
    say(f"[drone] {vehicles.payload_report(VEHICLE, RIG)}")

    view_path = f"/World/Cameras/{args_cli.view}"
    if args_cli.view in ONBOARD:
        view_path = f"{body}/{args_cli.view}_view"
    elif args_cli.view == "chase":
        view_path = None                            # the GUI camera is moved every frame instead
    else:
        view = cfg.camera(args_cli.view)
        sim.set_camera_view(eye=list(view.eye), target=list(view.target))
    if not args_cli.headless and view_path:
        # look through the authored camera itself so the view also gets its lens and depth of field
        try:
            from omni.kit.viewport.utility import get_active_viewport
            get_active_viewport().camera_path = view_path
        except Exception as exc:  # noqa: BLE001
            say(f"[drone] viewport keeps Isaac's default lens ({exc})")

    cams: dict[str, Camera] = {}
    if args_cli.record > 0:
        for name in [n for n in args_cli.cameras if n in ONBOARD]:
            pos, quat = geom.camera_local(name)
            cams[name] = Camera(CameraCfg(
                prim_path=f"{body}/{name}_cam", update_period=0.0, height=PLAN["height"], width=PLAN["width"],
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=PLAN["focal_length"],
                                                 horizontal_aperture=PLAN["horizontal_aperture"],
                                                 vertical_aperture=PLAN["vertical_aperture"],
                                                 clipping_range=(0.01, 10000.0)),
                offset=CameraCfg.OffsetCfg(pos=tuple(float(v) for v in pos), rot=quat, convention="opengl")))
        if "chase" in args_cli.cameras:
            cams["chase"] = Camera(CameraCfg(prim_path="/World/RecordCameras/chase", update_period=0.0, height=720,
                                             width=1280, data_types=["rgb"],
                                             spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 20000.0)),
                                             offset=CameraCfg.OffsetCfg(convention="world")))
        for name in [n for n in args_cli.cameras if n not in ONBOARD and n != "chase"]:
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
        if name in ONBOARD or name == "chase":
            continue                                # mounted on the drone / following it: they move with it
        c = cfg.camera(name)
        cam.set_world_poses_from_view(torch.tensor([c.eye], device=sim.device, dtype=torch.float32),
                                      torch.tensor([c.target], device=sim.device, dtype=torch.float32))

    sim_dt = sim.get_physics_dt()
    every = max(1, int(round(1.0 / (args_cli.fps * sim_dt))))
    video_fps = 1.0 / (every * sim_dt)               # exact, so the video plays in real time

    # the LUCID sensor simulation of each recorded onboard camera, fed by Isaac's linear HDR render
    sensors, readers, records = {}, {}, {}
    if not args_cli.isaac_rgb:
        settings = sensor_sim.SensorSettings(exposure=args_cli.exposure, exposure_us=args_cli.exposure_us,
                                             gain_db=args_cli.gain_db, gamma=GAMMA)
        for i, name in enumerate(n for n in cams if n in ONBOARD):
            try:
                readers[name] = HdrReader(cams[name], sim.device)
            except Exception as exc:  # noqa: BLE001
                say(f"[drone] WARNING: no HdrColor for {name} ({exc}); its sensor simulation starts from Isaac's "
                    f"tone-mapped 8-bit image instead (clipped highlights, Isaac's tone curve)")
            sensors[name] = sensor_sim.LucidSensor(
                RIG, PLAN, device=sim.device, seed=17 + i, settings=settings, fps=video_fps,
                reference_illuminance=MANIFEST["sky"].get("reference_illuminance"))
            records[name] = []
        for w in RIG.warnings(video_fps):
            say(f"[drone] note: {w}")

    prop_body_ids = robot.find_bodies(VEHICLE.prop_bodies)[0]
    robot_mass = robot.root_physx_view.get_masses().sum()
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()
    say("[drone] setup complete")

    seconds = fl.duration if (args_cli.record < 0 and fl is not None) else args_cli.record
    record_steps = int(round(seconds / sim_dt))
    if cams:
        say(f"[drone] recording {seconds:.0f} s at {video_fps:g} fps = {int(seconds * video_fps)} frames per camera")
    dirs, writers = {}, {}
    for name in cams:
        dirs[name] = os.path.join(OUT, name)
        os.makedirs(dirs[name], exist_ok=True)
        for old in glob.glob(os.path.join(dirs[name], "frame_*.png")) + glob.glob(os.path.join(dirs[name], "raw_*.png")):
            try:
                os.remove(old)
            except OSError:
                pass
        writers[name] = VideoWriter(os.path.join(OUT, f"drone_{name}.mp4"), dirs[name], video_fps, args_cli.save_frames)
    chase = ChaseCamera() if ("chase" in cams or args_cli.view == "chase") and fl is not None else None
    if cams:
        say("[drone] warming up the renderer and waiting for textures/assets to stream in ...")
        for _ in range(30):
            sim.render()
        for _ in range(3000):
            if not SimulationManager.assets_loading():
                break
            sim.render()

    grey = 0.18 * MANIFEST["sky"].get("reference_illuminance", 0.0) / math.pi    # an 18 % grey card, in render units
    hdr_checked, ldr_noted = set(), set()
    count, frame = 0, 0
    while simulation_app.is_running():
        if fl is not None:
            # kinematic flight: the pose and velocity of the garden tour (looped, 2 s hover between laps)
            t = (count * sim_dt) % (fl.duration + 2.0) if not cams else count * sim_dt
            t = min(t, fl.duration)
            p, q, v = fl.position(t), fl.quat_wxyz(t), fl.velocity(t)
            robot.write_root_pose_to_sim(torch.tensor([[*p, *q]], device=sim.device, dtype=torch.float32))
            robot.write_root_velocity_to_sim(torch.tensor([[*v, 0.0, 0.0, 0.0]], device=sim.device, dtype=torch.float32))
        elif count % 2000 == 0:                    # reset, as in the original demo
            if robot.num_joints > 0:               # the LMF2's propellers are fixed: no joints
                robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
            robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()
        forces = torch.zeros(robot.num_instances, len(prop_body_ids), 3, device=sim.device)
        if fl is None:
            forces[..., 2] = robot_mass * gravity / max(1, len(prop_body_ids))
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces, torques=torch.zeros_like(forces), body_ids=prop_body_ids)
        robot.write_data_to_sim()
        capture = bool(cams) and count % every == 0
        if chase is not None and (capture or not cams):
            t = min(count * sim_dt, fl.duration)
            eye, target = chase.update(fl.position(t), fl.direction, sim_dt * (every if cams else 1))
            if "chase" in cams:
                cams["chase"].set_world_poses_from_view(torch.tensor([eye], device=sim.device, dtype=torch.float32),
                                                        torch.tensor([target], device=sim.device, dtype=torch.float32))
            if args_cli.view == "chase" and not args_cli.headless:
                sim.set_camera_view(eye=list(eye), target=list(target))
        sim.step(render=capture or not cams)
        robot.update(sim_dt)
        if capture:
            for name, cam in cams.items():
                cam.update(sim_dt)
                if name not in sensors:
                    writers[name].add(cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8))
                    continue
                hdr, source = (readers[name].read() if name in readers else None), "HdrColor"
                if hdr is None or not bool(hdr[..., :3].abs().amax() > 0):
                    if name not in ldr_noted:
                        say(f"[drone] {name}: no HdrColor image at frame {frame}; the sensor simulation uses the "
                            f"tone-mapped image (this is only expected for the first frame or two)")
                        ldr_noted.add(name)
                    hdr, source = sensor_sim.ldr_to_linear(cam.data.output["rgb"][0]), "rgb"
                if records[name] and records[name][-1]["source"] != source:
                    sensors[name].reset_exposure()      # HdrColor and the 8-bit image differ in scale by ~1000x
                out = sensors[name].process(hdr)
                if source == "HdrColor" and name not in hdr_checked and grey > 0:
                    hdr_checked.add(name)
                    ratio = out["mean_radiance"] / grey
                    say(f"[drone] {name}: mean scene radiance = {ratio:.2f} x an 18 % grey card under the calibrated "
                        f"sun + sky" + ("" if 0.03 < ratio < 30 else " - HdrColor is not in the scene's light "
                                        "units, so exposure_us in camera_info is not physical (the images are fine)"))
                writers[name].add(out["rgb"].cpu().numpy())
                if args_cli.save_raw:
                    Image.fromarray(out["raw"].cpu().numpy().astype(np.uint16)).save(
                        os.path.join(dirs[name], f"raw_{frame:05d}.png"))
                records[name].append({"frame": frame, "time_s": round(count * sim_dt, 4), "source": source,
                                      "exposure_us": None if math.isnan(out["exposure_us"]) else round(out["exposure_us"], 2),
                                      "gain_db": out["gain_db"], "saturated": round(out["saturated"], 5)})
            frame += 1
            if frame % round(video_fps * (1 if seconds <= 30 else 10)) == 0:
                extra = "".join(f", {n} exposure {records[n][-1]['exposure_us']} us" for n in records if records[n])
                phase = f" ({fl.phase_name(count * sim_dt)})" if hasattr(fl, "phase_name") else ""
                say(f"[drone] recorded {frame / video_fps:.0f} of {seconds:.0f} s{phase}{extra}")
        count += 1
        if cams and count >= record_steps:
            break

    if cams:
        for name in records:
            pos, quat = geom.camera_local(name)
            info = RIG.camera_info(name)
            info.update(vehicle=VEHICLE.name, mount={"body": VEHICLE.body, "position_m": [float(v) for v in pos],
                                                     "orientation_wxyz": list(quat), "convention": "USD camera (-Z forward, +Y up)",
                                                     "pitch_down_deg": geom.cameras[name].pitch_down_deg},
                        route=({k: WORLD[k] for k in ("takeoff", "landing", "distance_m", "heading_deg", "cruise_z", "sun")}
                               if WORLD else "garden"),
                        render_plan=PLAN, fps=video_fps, settings=sensor_sim.SensorSettings(
                            exposure=args_cli.exposure, exposure_us=args_cli.exposure_us, gain_db=args_cli.gain_db,
                            gamma=GAMMA).__dict__, notes=RIG.warnings(video_fps), frames=records[name])
            with open(os.path.join(OUT, f"drone_{name}_camera_info.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, indent=1)
        ok = [writers[n].close() for n in cams]
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
