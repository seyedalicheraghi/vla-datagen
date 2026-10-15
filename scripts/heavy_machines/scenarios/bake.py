"""Texture/asset baking for scenario scenes (numpy + Pillow only — no pxr, no Isaac Sim).

Everything slow happens here: terrain colour (with cavity darkening and boulder contact shadows),
terrain normal/roughness relief maps, the far terrain, rock and machine PBR texture sets, and the
rock scatter.  Results are cached in ``tex_dir`` under a hash of every input, so a re-render with the
same scenario and options skips straight to rendering; independent jobs run in parallel processes.

    python scripts/heavy_machines/scenarios/bake.py --scenario adt_approach_hex --tex_dir out/textures --workers 4

``render_scenario.py`` runs this as a subprocess *before* starting Isaac Sim (forking or spawning
workers inside the Kit process is not safe), then ``usd_scene.build_scene`` finds the cache.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from PIL import Image

if __package__ in (None, ""):                      # run as a script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    __package__ = "heavy_machines.scenarios"

from heavy_machines.scenarios.pbr_textures import (detail_px_per_m, machine_textures, rock_textures,  # noqa: E402
                                                   save_texture_set, terrain_detail_cavity,
                                                   terrain_detail_maps)
from heavy_machines.scenarios.rocks import Scatter, contact_map, scatter  # noqa: E402
from heavy_machines.scenarios.terrain import bake_albedo  # noqa: E402

BAKE_VERSION = "2"
ROCK_TINTS = (0.86, 0.97, 1.08)


@dataclasses.dataclass
class SceneOptions:
    terrain_step: float = 0.15          # detailed terrain mesh spacing (m)
    texture_ppm: float = 24.0           # colour px/m in the detailed area
    texture_ppm_near: float = 48.0      # colour px/m along the camera path (scenario.near_rect)
    far_ppm: float = 1.0
    detail_factor: float = 2.5          # relief maps are baked this much denser than the colour
    boulders: int = 160
    gravel_per_m2: float = 1.2
    rock_tex_size: int = 512
    machine_tex_size: int = 1024
    bevel: float = 0.025


def load_scenario(name: str):
    return importlib.import_module(f"heavy_machines.scenarios.{name}").build()


def machine_material_jobs(sc) -> list[tuple[str, str, tuple[float, float, float], int]]:
    """(machine, material, rgb, seed) for every visual material used by the scenario's machines."""
    seen, out = set(), []
    for actor in sc.actors:
        spec = actor.spec
        for lk in spec.links:
            for s in lk.shapes:
                if s.visual and (spec.name, s.material) not in seen:
                    seen.add((spec.name, s.material))
                    rgb = tuple(spec.materials.get(s.material, (0.6, 0.6, 0.6, 1.0))[:3])
                    out.append((spec.name, s.material, rgb, 17 + len(out)))
    return out


def regions(sc, o: SceneOptions) -> list[tuple[str, tuple, float]]:
    out = []
    if sc.near_rect is not None:
        out.append(("near", tuple(sc.near_rect), o.texture_ppm_near))
    out.append(("local", tuple(sc.local_rect), o.texture_ppm))
    return out


def cache_key(sc, o: SceneOptions) -> str:
    blob = repr((BAKE_VERSION, sc.name, sc.local_rect, sc.near_rect, sc.far_rect, sc.keep_clear,
                 dataclasses.asdict(o), repr(sc.terrain), [a.machine for a in sc.actors]))
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Jobs (top-level so a spawn-based process pool can pickle them)
# ---------------------------------------------------------------------------

def _png(arr, path):
    Image.fromarray(arr).save(path, compress_level=1)
    return path


def _job_terrain(terrain, o: SceneOptions, tex_dir: str, name: str, rect, ppm: float, rocks_npz: str):
    rocks = load_scatter(rocks_npz)
    albedo = bake_albedo(terrain, rect, ppm, seed=1).astype(np.float32)
    albedo *= (terrain_detail_cavity(terrain, rect, ppm, seed=1).astype(np.float32) / 255.0)[..., None] ** 0.8
    if len(rocks):
        albedo *= contact_map(rocks, rect, ppm)
    files = {"albedo": _png(np.clip(albedo + 0.5, 0, 255).astype(np.uint8),
                            os.path.join(tex_dir, f"terrain_{name}_albedo.png"))}
    dppm = detail_px_per_m(ppm, rect, factor=o.detail_factor)
    nrm, rough = terrain_detail_maps(terrain, rect, dppm, seed=1)
    files["normal"] = _png(nrm, os.path.join(tex_dir, f"terrain_{name}_normal.png"))
    files["roughness"] = _png(rough, os.path.join(tex_dir, f"terrain_{name}_roughness.png"))
    return f"terrain_{name}", files, f"{name}: colour {ppm:.0f} px/m, relief {dppm:.0f} px/m"


def _job_far(terrain, far_rect, o: SceneOptions, tex_dir: str):
    path = _png(bake_albedo(terrain, tuple(far_rect), o.far_ppm, seed=2, far=True),
                os.path.join(tex_dir, "terrain_far_albedo.png"))
    return "terrain_far", {"albedo": path}, f"far: colour {o.far_ppm:g} px/m"


def _job_rock(o: SceneOptions, tex_dir: str, b: int, tint: float):
    ts = rock_textures(seed=11 + b, size=o.rock_tex_size, shade=tint)
    files = save_texture_set(ts, tex_dir, f"rock_{b}")
    files["metres_per_tile"] = ts.metres_per_tile
    return f"rock_{b}", files, f"rock material {b}"


def _job_machine(o: SceneOptions, tex_dir: str, machine: str, material: str, rgb, seed: int):
    ts = machine_textures(material, tuple(rgb), seed=seed, size=o.machine_tex_size)
    files = save_texture_set(ts, tex_dir, f"{machine}_{material}")
    files.update(metallic=ts.metallic, opacity=ts.opacity, clearcoat=ts.clearcoat, metres_per_tile=ts.metres_per_tile)
    return f"machine:{machine}:{material}", files, f"{machine} / {material}"


# ---------------------------------------------------------------------------
# Scatter persistence
# ---------------------------------------------------------------------------

def save_scatter(sc_rocks: Scatter, path: str) -> str:
    np.savez(path, positions=sc_rocks.positions, scales=sc_rocks.scales, yaws=sc_rocks.yaws, proto=sc_rocks.proto,
             shade=sc_rocks.shade if sc_rocks.shade is not None else np.ones(len(sc_rocks), np.float32),
             meta=np.array([sc_rocks.n_boulders, sc_rocks.n_protos, sc_rocks.proto_seed, sc_rocks.proto_detail]))
    return path


def load_scatter(path: str) -> Scatter:
    with np.load(path) as d:                       # closed at once: Windows cannot overwrite open files
        nb, npr, ps, pd = (int(v) for v in d["meta"])
        return Scatter(d["positions"], d["scales"], d["yaws"], d["proto"], n_boulders=nb, n_protos=npr,
                       proto_seed=ps, shade=d["shade"], proto_detail=pd)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def bake_assets(sc, tex_dir: str, o: SceneOptions | None = None, workers: int = 1, log=print) -> dict:
    """Bake (or reuse) every texture and the rock scatter.  Returns the manifest dict."""
    o = o or SceneOptions()
    os.makedirs(tex_dir, exist_ok=True)
    key = cache_key(sc, o)
    manifest_path = os.path.join(tex_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            man = json.load(f)
        paths = [p for files in man.get("files", {}).values() for k, p in files.items() if isinstance(p, str)]
        if man.get("key") == key and os.path.exists(man.get("rocks", "")) and all(os.path.exists(p) for p in paths):
            log(f"[bake] cache hit ({key}) in {tex_dir}")
            return man

    t0 = time.time()
    log("[bake] scattering boulders and gravel ...")
    rocks = scatter(sc.terrain, sc.local_rect, seed=6, keep_clear=sc.keep_clear, boulders=o.boulders,
                    gravel_per_m2=o.gravel_per_m2, grid_step=o.terrain_step, focus=sc.near_rect)
    rocks_npz = save_scatter(rocks, os.path.join(tex_dir, "rocks.npz"))
    log(f"[bake] {len(rocks)} rocks ({rocks.n_boulders} boulders)")

    jobs = [(_job_terrain, (sc.terrain, o, tex_dir, n, r, ppm, rocks_npz)) for n, r, ppm in regions(sc, o)]
    jobs.append((_job_far, (sc.terrain, sc.far_rect, o, tex_dir)))
    jobs += [(_job_rock, (o, tex_dir, b, t)) for b, t in enumerate(ROCK_TINTS)]
    jobs += [(_job_machine, (o, tex_dir) + j) for j in machine_material_jobs(sc)]
    workers = max(1, min(workers, len(jobs)))
    if sys.platform == "win32":
        workers = min(workers, 61)                 # ProcessPoolExecutor's limit on Windows
    log(f"[bake] {len(jobs)} jobs on {workers} worker(s) ...")

    files: dict[str, dict] = {}
    if workers > 1:
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futs = [pool.submit(fn, *a) for fn, a in jobs]
            for fut in cf.as_completed(futs):
                name, f, msg = fut.result()
                files[name] = f
                log(f"[bake]   {msg}  ({time.time() - t0:.0f} s)")
    else:
        for fn, a in jobs:
            name, f, msg = fn(*a)
            files[name] = f
            log(f"[bake]   {msg}  ({time.time() - t0:.0f} s)")

    man = {"key": key, "scenario": sc.name, "rocks": rocks_npz, "files": files}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
    log(f"[bake] done in {time.time() - t0:.0f} s -> {tex_dir}")
    return man


def add_cli_options(p: argparse.ArgumentParser) -> None:
    d = SceneOptions()
    p.add_argument("--terrain_step", type=float, default=d.terrain_step, help="detailed terrain mesh spacing (m)")
    p.add_argument("--texture_ppm", type=float, default=d.texture_ppm, help="terrain colour px/m (detailed area)")
    p.add_argument("--texture_ppm_near", type=float, default=d.texture_ppm_near,
                   help="terrain colour px/m along the camera path")
    p.add_argument("--boulders", type=int, default=d.boulders)
    p.add_argument("--gravel_per_m2", type=float, default=d.gravel_per_m2)


def options_from_args(a) -> SceneOptions:
    return SceneOptions(terrain_step=a.terrain_step, texture_ppm=a.texture_ppm, texture_ppm_near=a.texture_ppm_near,
                        boulders=a.boulders, gravel_per_m2=a.gravel_per_m2)


def main():
    from heavy_machines import safe_console
    safe_console()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="adt_approach_hex")
    p.add_argument("--tex_dir", required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    add_cli_options(p)
    a = p.parse_args()
    bake_assets(load_scenario(a.scenario), a.tex_dir, options_from_args(a), a.workers, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
