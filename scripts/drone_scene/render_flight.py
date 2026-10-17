"""Render the drone's onboard front/back cameras along ``flight.Flight`` to two videos (Mitsuba; no Isaac).

The garden is built once (``mitsuba_preview.Session``); every frame only creates the two cameras at their
pose on the flying drone.  Frames are path traced, denoised with Intel Open Image Denoise (colour only:
Mitsuba's AOV integrator, which would give albedo/normal guides, is ~300x slower than the path tracer on
this scene) when it is installed, exposed with the same calibrated exposure as ``preview.py``, given a
photographic camera look (``heavy_machines.scenarios.camera_look``: filmic curve, bloom, lens softness,
vignette, chromatic aberration, sensor grain) and encoded to H.264.  The cameras look away from the drone,
so the drone itself is not in these renders.  Already rendered frames are kept, so an interrupted run
resumes.

    python scripts/drone_scene/render_flight.py --out . --seconds 20 --fps 24 --spp 16
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "drone_scene"

from drone_scene import denoise, flight  # noqa: E402
from drone_scene import prepare as prep  # noqa: E402
from drone_scene import preview, scene  # noqa: E402


def garden_look():
    from heavy_machines.scenarios import camera_look as CL

    return CL.LookParams(haze_strength=0.0, exposure=1.0, contrast=0.55, saturation=1.0, white_balance=(1.0, 1.0, 1.0),
                         highlight_rolloff=0.6, bloom_strength=0.04, bloom_radius_px=14.0, vignette=0.18,
                         chromatic_aberration_px=0.6, sharpen=0.2, grain=0.012, lens_blur_px=0.4)


def encode(frame_dir: str, fps: int, path: str, log) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            log(f"[flight] no ffmpeg: frames are in {frame_dir}")
            return False
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(frame_dir, "frame_%05d.png"),
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-colorspace", "bt709",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-c:v", "libx264", "-crf", "17", "-preset", "slow",
           "-movflags", "+faststart", path]
    ok = subprocess.run(cmd, check=False).returncode == 0
    log(f"[flight] wrote {path}" if ok else f"[flight] ffmpeg FAILED for {path}")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=".", help="where the two mp4 files go")
    p.add_argument("--frames", default=None, help="frame folder (default <out>/drone_flight_frames)")
    p.add_argument("--cache", default=prep.DEFAULT_CACHE)
    p.add_argument("--cameras", nargs="+", default=["front", "back"], choices=sorted(flight.CAMERAS))
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--width", type=int, default=854)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--spp", type=int, default=8)
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--no_denoise", action="store_true")
    p.add_argument("--no_look", action="store_true", help="skip the camera-look post-processing")
    p.add_argument("--only_frames", type=str, default=None, help="e.g. 0,120,240 (tests)")
    a = p.parse_args()
    log = lambda m: print(m, flush=True)  # noqa: E731

    from drone_scene import mitsuba_preview as MP

    out = os.path.abspath(a.out)
    frames_root = os.path.abspath(a.frames or os.path.join(out, "drone_flight_frames"))
    cfg = scene.SceneCfg()
    man = prep.prepare(os.path.expanduser(a.cache), mirror=True, sun_elevation_deg=cfg.sun.elevation_deg, log=log)
    stage, _ = preview.build_preview_stage(os.path.join(frames_root, "garden.usda"), cfg, man, drone=False)
    ses = MP.Session(stage, aovs=False, max_depth=a.max_depth, log=log)
    ev = preview.exposure_ev(man)
    use_oidn = not a.no_denoise and denoise.available()
    log(f"[flight] denoiser: {'Open Image Denoise' if use_oidn else 'none'}")
    look = None
    if not a.no_look:
        from heavy_machines.scenarios import camera_look as CL
        look = garden_look()
        CL.warm_up(a.height, a.width, look)

    fl = flight.Flight()
    n = int(round(a.seconds * a.fps))
    # first and last frames first: they are the start/end images for the generative comparison
    frames = [int(k) for k in a.only_frames.split(",")] if a.only_frames else [0, n - 1] + list(range(1, n - 1))
    dirs = {c: os.path.join(frames_root, c) for c in a.cameras}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    t_start, done = time.time(), 0
    for k in frames:
        t = min(k / a.fps, fl.duration)
        for c in a.cameras:
            path = os.path.join(dirs[c], f"frame_{k:05d}.png")
            if os.path.exists(path):
                continue
            oc = flight.CAMERAS[c]
            cam = MP.camera_from_matrix(fl.camera_matrix(t, c), focal_length=oc.focal_length,
                                        horizontal_aperture=oc.horizontal_aperture, name=c)
            r = ses.render(cam, a.width, a.height, a.spp, seed=k)
            img = denoise.denoise(r["color"]) if use_oidn else r["color"]
            rgb = MP.tonemap_image(img, ev, "aces")
            if look is not None:
                from heavy_machines.scenarios import camera_look as CL
                rgb = CL.apply(rgb, np.full(rgb.shape[:2], np.inf, np.float32), look, seed=k)
            Image.fromarray(rgb).save(path + ".part.png")
            os.replace(path + ".part.png", path)
        done += 1
        if done % 12 == 0 or done == len(frames):
            el = time.time() - t_start
            log(f"[flight] {done}/{len(frames)} frames, {el / done:.1f} s/frame, ~{el / done * (len(frames) - done) / 60:.0f} min left")
    if not a.only_frames:
        for c in a.cameras:
            encode(dirs[c], a.fps, os.path.join(out, f"drone_{c}_camera.mp4"), log)


if __name__ == "__main__":
    main()
