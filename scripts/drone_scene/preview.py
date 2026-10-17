"""Offline path-traced preview of the drone scene (Mitsuba 3; no Isaac Sim).

Builds the same USD stage Isaac Sim gets (``scene.build`` + the Crazyflie with ``apply_drone_materials``)
from locally mirrored assets, saves it as ``<out>/drone_scene.usda`` and renders the chosen cameras with
``mitsuba_preview.render``.  Exposure is fixed, not automatic: an 18 % grey card lit by the reference
illuminance that Isaac Lab's default exposure is tuned for renders mid-grey, so brightness differences
between iterations are real.

    python scripts/drone_scene/preview.py --out /tmp/drone_preview [--cameras demo hero] [--spp 64]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "drone_scene"

from drone_scene import prepare as prep  # noqa: E402
from drone_scene import scene  # noqa: E402


def build_preview_stage(path: str, cfg: scene.SceneCfg, manifest: dict, drone: bool = True, drone_look: bool = True):
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(path)
    man = dict(manifest)
    man["sky"] = dict(manifest["sky"], file=manifest["sky"]["file_preview"])     # smaller sky for Mitsuba
    summary = scene.build(stage, cfg, man)
    if not drone:
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
        stage.GetRootLayer().Save()
        return stage, summary
    drone = UsdGeom.Xform.Define(stage, "/World/Crazyflie")
    drone.GetPrim().GetReferences().AddReference(manifest["crazyflie"])
    # the asset brings its own translate op (Isaac Lab's spawner sets it the same way)
    ops = {op.GetOpName(): op for op in drone.GetOrderedXformOps()}
    (ops.get("xformOp:translate") or drone.AddTranslateOp()).Set(Gf.Vec3d(*cfg.drone_pos))
    summary["drone_meshes_bound"] = scene.apply_drone_materials(stage, "/World/Crazyflie")
    if drone_look:
        from drone_scene import drone_look as DL
        summary["drone_look"] = {k: v for k, v in DL.apply_drone_look(
            stage, "/World/Crazyflie", manifest.get("cache_dir") or prep.DEFAULT_CACHE).items()
            if k in ("prop_discs", "leds", "meshes_bound", "blades_hidden")}
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    stage.GetRootLayer().Save()
    return stage, summary


def exposure_ev(manifest: dict) -> float:
    """EV so an 18 % grey card under the reference illuminance renders as 0.18 (before tonemapping)."""
    e_ref = manifest["sky"].get("reference_illuminance")
    return math.log2(math.pi / e_ref) if e_ref else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default=prep.DEFAULT_CACHE)
    p.add_argument("--cameras", nargs="*", default=["demo", "hero"])
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--spp", type=int, default=64)
    p.add_argument("--ev", type=float, default=0.0, help="extra exposure on top of the calibrated one")
    a = p.parse_args()
    log = lambda m: print(m, flush=True)  # noqa: E731

    from drone_scene import mitsuba_preview

    os.makedirs(a.out, exist_ok=True)
    cfg = scene.SceneCfg()
    man = prep.prepare(os.path.expanduser(a.cache), mirror=True, sun_elevation_deg=cfg.sun.elevation_deg, log=log)
    stage, summary = build_preview_stage(os.path.join(a.out, "drone_scene.usda"), cfg, man)
    log(f"[preview] stage: {summary}")
    ev = exposure_ev(man) + a.ev
    t0 = time.time()
    outs = {f"/World/Cameras/{name}": os.path.join(a.out, f"{name}.png") for name in a.cameras}
    # volpath: the translucent propeller-blur discs need it (path treats any opacity < 1 as nearly opaque)
    stats = mitsuba_preview.render_many(stage, outs, width=a.width, height=a.height, spp=a.spp, exposure_ev=ev,
                                        log=log, integrator="volpath")
    for (cam, out), st in zip(outs.items(), stats):
        log(f"[preview] {cam}: {out} {st}")
    log(f"[preview] {len(outs)} camera(s) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
