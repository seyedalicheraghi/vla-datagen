"""Run the world builder from Isaac Sim's Python without touching Isaac's packages (standard library only).

The builder needs rasterio (GDAL), pyproj, pyarrow, shapely, scipy and usd-core.  Installing those into the
Isaac Sim environment could upgrade numpy under Isaac, so they go into their own folder instead
(``pip install --target <cache>/world_libs``) and the builder runs as a separate process with that folder first
on PYTHONPATH.  Same Python, so the wheels match; Isaac's own process never imports them.

    from drone_scene.world import runner
    man = runner.ensure_world(cache_dir, ["--region", "42.025,-93.690"], log=print)   # world.json as a dict
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

PACKAGES = ["numpy>=1.26", "scipy>=1.11", "rasterio>=1.4", "pyproj>=3.6", "pyarrow>=15", "shapely>=2.0",
            "requests>=2.31", "Pillow>=10", "usd-core>=24.8", "apsw>=3.45"]
# apsw reads USGS's 1 m lidar project index (a GeoPackage) over HTTP; without it the 10 m 3DEP DEM is used
CHECK = "import numpy, scipy, rasterio, pyproj, pyarrow, shapely, requests, PIL, pxr, apsw"
SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _env(libs: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([libs, SCRIPTS] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    env.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.tiff")
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("PXR_USDC_EMIT_DEPRECATION_WARNINGS", "0")     # NVIDIA's older tree files trigger it
    return env


def ensure_libs(cache_dir: str, python: str = sys.executable, log=print) -> str:
    """The folder with the builder's packages, installed on first use."""
    libs = os.path.join(cache_dir, "world_libs")
    stamp = os.path.join(libs, ".installed")
    want = hashlib.sha1(repr((PACKAGES, sys.version_info[:2])).encode()).hexdigest()[:10]
    if os.path.exists(stamp) and open(stamp).read().strip() == want:
        return libs
    if os.path.isdir(libs) and subprocess.run([python, "-c", CHECK], env=_env(libs), capture_output=True).returncode == 0:
        with open(stamp, "w") as f:
            f.write(want)
        return libs
    os.makedirs(libs, exist_ok=True)
    log(f"[world] installing the map tools ({', '.join(p.split('>')[0] for p in PACKAGES)}) into {libs} - once, "
        f"about 250 MB; Isaac Sim's own packages are not changed")
    cmds = [[python, "-m", "pip", "install", "--upgrade", "--target", libs] + PACKAGES]
    if shutil.which("uv"):
        cmds.append(["uv", "pip", "install", "--python", python, "--target", libs] + PACKAGES)
    for cmd in cmds:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            break
        log(f"[world] {' '.join(cmd[:4])} ... failed:\n{res.stdout[-1500:]}{res.stderr[-1500:]}")
    else:
        raise RuntimeError("could not install the map tools (pip and uv both failed); see the messages above")
    chk = subprocess.run([python, "-c", CHECK], env=_env(libs), capture_output=True, text=True)
    if chk.returncode != 0:
        raise RuntimeError(f"map tools installed but not importable:\n{chk.stderr[-2000:]}")
    with open(stamp, "w") as f:
        f.write(want)
    return libs


def ensure_world(cache_dir: str, build_args: list[str], python: str = sys.executable, log=print) -> dict:
    """Build the world (or reuse the cached one) and return its manifest (world.json)."""
    libs = ensure_libs(cache_dir, python, log)
    out = os.path.join(cache_dir, "worlds")
    os.makedirs(out, exist_ok=True)
    result = os.path.join(out, "last_manifest_path.txt")
    if os.path.exists(result):
        os.remove(result)
    cmd = [python, "-u", "-m", "drone_scene.world.build", "--out", out, "--cache", cache_dir,
           "--write-path", result] + list(build_args)
    proc = subprocess.Popen(cmd, env=_env(libs), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    remote_refs = 0
    for line in proc.stdout:
        # usd-core cannot open https:// references (Isaac Sim's resolver can): expected, so one summary line
        if "Could not open asset @http" in line:
            remote_refs += 1
            continue
        log(line.rstrip())
    if remote_refs:
        log(f"[world] {remote_refs} trees/grass models are referenced from NVIDIA's server; Isaac Sim loads them "
            "(the map tools themselves cannot open web links, which is expected)")
    if proc.wait() != 0 or not os.path.exists(result):
        raise RuntimeError(f"the world builder failed (exit code {proc.returncode}); see the [world] messages above")
    with open(result, encoding="utf-8") as f:
        path = f.read().strip()
    with open(path, encoding="utf-8") as f:
        return json.load(f)
