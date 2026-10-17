"""Download and pre-process everything the drone scene needs; write a manifest (no pxr, no Isaac Sim).

Runs before Isaac Sim starts (``quadcopter_pbr.py``) and before the offline preview (``preview.py``):

- the ground materials' textures (NVIDIA Base materials, local copies);
- the HDR sky with its sun cut out (``hdr.remove_sun``), plus a sun light of the same irradiance and
  colour, both calibrated against the sky Isaac Lab exposes well (``hdr.calibrate``);
- props: URLs for Isaac (it streams them), or local mirrors for the preview (``mirror=True``).

Everything is cached under ``cache_dir`` (default ``~/.cache/drone_scene``); a second run only reads
the manifest.

    python scripts/drone_scene/prepare.py [--cache ~/.cache/drone_scene] [--mirror]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

if __package__ in (None, ""):                     # run as a script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "drone_scene"

from drone_scene import assets, hdr, nucleus  # noqa: E402
from drone_scene.scene import SunCfg  # noqa: E402  (numpy only)

PREPARE_VERSION = "3"
DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "drone_scene")
GROUND = ("pavers", "lawn", "mulch", "curb", "sand")


def _key(mirror: bool, preview_sky_factor: int, sun_elevation_deg: float) -> str:
    blob = repr((PREPARE_VERSION, assets.SKY, assets.REFERENCE_SKY, assets.REFERENCE_INTENSITY, GROUND, sun_elevation_deg,
                 sorted(assets.PROPS), assets.CRAZYFLIE, mirror, preview_sky_factor,
                 {k: (m.mdl, m.tile_m) for k, m in assets.MATERIALS.items()}))
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def prepare_sky(cache_dir: str, preview_sky_factor: int = 4, sun_elevation_deg: float = SunCfg.elevation_deg,
                log=print) -> dict:
    """Sunless sky HDR (+ a smaller copy for the preview) and the calibrated dome/sun intensities.

    The sun light is placed at ``sun_elevation_deg`` (scene.SunCfg), not where the HDR has it, so the
    calibration uses that elevation for the sun's share of the horizontal illuminance.
    """
    out_dir = os.path.join(cache_dir, "sky")
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(assets.SKY))[0]
    full = os.path.join(out_dir, f"{name}_nosun.hdr")
    small = os.path.join(out_dir, f"{name}_nosun_x{preview_sky_factor}.hdr")
    info_path = os.path.join(out_dir, f"{name}_nosun_sun{sun_elevation_deg:g}.json")
    if all(os.path.exists(p) for p in (full, small, info_path)):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        info.update(file=full, file_preview=small)
        return info

    t0 = time.time()
    log(f"[prepare] sky: {assets.SKY}")
    sky = hdr.read_hdr(nucleus.fetch(assets.SKY, cache_dir, log=log))
    sun = hdr.find_sun(sky)
    if sun is None:
        raise RuntimeError(f"no clear sun found in {assets.SKY}")
    sunless = hdr.remove_sun(sky, sun)
    reference = hdr.read_hdr(nucleus.fetch(assets.REFERENCE_SKY, cache_dir, log=log))
    cal = hdr.calibrate(sunless, sun, reference, assets.REFERENCE_INTENSITY,
                        sun_cos_zenith=math.sin(math.radians(sun_elevation_deg)))
    irr = np.asarray(sun.irradiance, float)
    color = irr / max(float(hdr.luminance(irr)), 1e-9)
    color = color / max(color.max(), 1e-9)                      # UsdLux colour: max channel 1
    # sun_intensity x color must equal dome_intensity x the HDR sun's irradiance
    sun_intensity = float(cal["dome_intensity"]) * float(hdr.luminance(irr)) / float(hdr.luminance(color))
    hdr.write_hdr(full, sunless)
    hdr.write_hdr(small, hdr.downsample(sunless, preview_sky_factor))
    info = {
        "dome_intensity": float(cal["dome_intensity"]),
        "sun_intensity": sun_intensity,
        "sun_color": [float(c) for c in color],
        "sun_elevation_deg": float(sun_elevation_deg),
        # where the HDR had its sun (hdr.py's azimuth convention, the mirror of USD's dome mapping)
        "hdr_sun": {"elevation_deg": float(sun.elevation_deg), "azimuth_deg": float(sun.azimuth_deg),
                    "solid_angle_sr": float(sun.solid_angle), "irradiance": [float(c) for c in irr]},
        "reference": {"sky": assets.REFERENCE_SKY, "intensity": assets.REFERENCE_INTENSITY},
        # horizontal illuminance of the reference setup: what Isaac's default exposure is tuned for
        "reference_illuminance": float(assets.REFERENCE_INTENSITY * hdr.luminance(hdr.irradiance(reference))),
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    log(f"[prepare] sky done in {time.time() - t0:.0f} s: dome {info['dome_intensity']:.0f}, "
        f"sun {sun_intensity:.0f} (HDR sun at {sun.elevation_deg:.1f} deg elevation)")
    info.update(file=full, file_preview=small)
    return info


def prepare(cache_dir: str = DEFAULT_CACHE, *, mirror: bool = False, preview_sky_factor: int = 4,
            sun_elevation_deg: float = SunCfg.elevation_deg, log=print) -> dict:
    """Return the manifest: {'textures', 'sky', 'props', 'crazyflie'} (paths local, or URLs for props)."""
    os.makedirs(cache_dir, exist_ok=True)
    key = _key(mirror, preview_sky_factor, sun_elevation_deg)
    manifest_path = os.path.join(cache_dir, f"manifest_{'mirror' if mirror else 'isaac'}.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            man = json.load(f)
        files = [p for t in man["textures"].values() for p in t.values()] + [man["sky"]["file"]]
        if mirror:
            files += list(man["props"].values()) + [man["crazyflie"]]
        if man.get("key") == key and all(os.path.exists(p) for p in files):
            log(f"[prepare] cache hit ({key}) in {cache_dir}")
            return man

    t0 = time.time()
    textures = {}
    for name in GROUND:
        log(f"[prepare] material {name}: {assets.MATERIALS[name].mdl}")
        textures[name] = assets.material_textures(name, cache_dir, log=log)
    sky = prepare_sky(cache_dir, preview_sky_factor, sun_elevation_deg, log=log)
    if mirror:
        props = {}
        for name, prop in assets.PROPS.items():
            log(f"[prepare] prop {name}: {prop.usd}")
            props[name] = nucleus.mirror_usd(prop.usd, cache_dir, log=log)
        crazyflie = nucleus.mirror_usd(assets.CRAZYFLIE, cache_dir, log=log)
    else:
        props = {name: nucleus.url(prop.usd) for name, prop in assets.PROPS.items()}
        crazyflie = nucleus.url(assets.CRAZYFLIE)
    man = {"key": key, "textures": textures, "sky": sky, "props": props, "crazyflie": crazyflie,
           "cache_dir": os.path.abspath(cache_dir)}
    with open(manifest_path + ".part", "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
    os.replace(manifest_path + ".part", manifest_path)
    log(f"[prepare] done in {time.time() - t0:.0f} s -> {manifest_path}")
    return man


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--mirror", action="store_true", help="also mirror props locally (offline preview)")
    a = p.parse_args()
    prepare(os.path.expanduser(a.cache), mirror=a.mirror, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
