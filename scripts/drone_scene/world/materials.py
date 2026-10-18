"""The world's PBR material library: ground classes, road surfaces, walls and roofs (``usdio.MaterialSpec``s).

Ground and road materials are NVIDIA's photo-scanned "Base" materials (textures fetched into the cache, read
out of the .mdl the same way ``assets.material_textures`` does); crops, house walls with windows and roofs are
the procedural textures of ``textures.py``.

``calibrate_tints`` matches a class's tiled texture to the real reflectance of the ground: Sentinel-2 L2A
surface reflectance is very nearly albedo, so the ratio of the satellite's mean colour over a class (or a patch
of it) to the texture's mean albedo is the tint that makes the rendered ground the colour it really is.
"""

from __future__ import annotations

import functools
import os

import numpy as np

from .. import assets, nucleus
from . import landcover as lc
from . import textures
from .buildings import SLOT_NAMES as BUILDING_SLOTS
from .roads import SURFACES
from .usdio import MaterialSpec

BASE = "NVIDIA/Materials/Base"

# ground class -> (NVIDIA Base .mdl or "procedural:<kind>", texture repeat m, tint, roughness scale)
GROUND = {
    lc.LAWN: (f"{BASE}/Natural/Grass_Cut.mdl", 1.5, (1.20, 1.15, 0.52), 1.0),          # as the backyard lawn
    lc.MEADOW: (f"{BASE}/Natural/Grass_Countryside.mdl", 2.0, (1.0, 1.0, 0.85), 1.0),
    lc.CROP: ("procedural:crop_rows", 3.04, (1.0, 1.0, 1.0), 1.0),
    lc.FOREST: (f"{BASE}/Natural/Leaves.mdl", 1.5, (0.8, 0.8, 0.8), 1.0),
    lc.SHRUB: (f"{BASE}/Natural/Grass_Countryside.mdl", 2.0, (0.7, 0.75, 0.6), 1.0),
    lc.SOIL: (f"{BASE}/Natural/Dirt.mdl", 1.5, (1.0, 1.0, 1.0), 1.0),
    lc.WATER: (None, 1.0, (1.0, 1.0, 1.0), 1.0),
    lc.WETLAND: (f"{BASE}/Natural/Grass_Winter.mdl", 2.0, (0.75, 0.85, 0.65), 1.0),
    lc.CONCRETE: (f"{BASE}/Masonry/Concrete_Rough.mdl", 1.0, (1.0, 1.0, 1.0), 1.0),
    lc.SAND: (f"{BASE}/Natural/Sand.mdl", 0.8, (1.0, 1.0, 1.0), 1.0),
}
WATER = MaterialSpec("water", color=(0.015, 0.025, 0.03), roughness=0.04)          # dark, mirror-like still water

ROAD = {
    "asphalt": (f"{BASE}/Natural/Asphalt.mdl", 3.0, (1.0, 1.0, 1.0)),
    "concrete": (f"{BASE}/Masonry/Concrete_Rough.mdl", 1.0, (1.05, 1.05, 1.05)),
    "gravel": (f"{BASE}/Natural/Soil_Rocky.mdl", 1.5, (1.0, 0.95, 0.85)),
}
MARKINGS = {"marking_yellow": (0.70, 0.50, 0.08), "marking_white": (0.75, 0.75, 0.73)}

# building slot -> (procedural texture, kind, tint)
BUILDING = {
    "siding_white": ("facade", "siding", (1.0, 1.0, 1.0)),
    "siding_beige": ("facade", "siding", (0.93, 0.86, 0.72)),
    "siding_grey": ("facade", "siding", (0.72, 0.74, 0.76)),
    "siding_blue": ("facade", "siding", (0.66, 0.74, 0.84)),
    "brick_red": ("facade", "brick", (1.0, 1.0, 1.0)),
    "metal_wall": ("facade", "metal", (0.95, 0.95, 0.95)),
    "shingle_dark": ("shingles", "asphalt", (0.55, 0.55, 0.57)),
    "shingle_brown": ("shingles", "asphalt", (0.85, 0.68, 0.52)),
    "shingle_grey": ("shingles", "asphalt", (0.95, 0.95, 0.95)),
    "metal_roof": ("shingles", "metal", (1.0, 1.0, 1.0)),
    "flat_roof": (None, None, (0.55, 0.55, 0.55)),                   # light grey membrane / ballast
}
WALL_TILE = (4.0, 3.0)
TEXTURE_VERSION = "v2"         # in the generated texture file names: bump when textures.py changes them
ROOF_TILE = 2.0


def _base_textures(mdl: str, cache_dir: str, log=None) -> dict:
    local = nucleus.fetch(mdl, cache_dir, log=log)
    with open(local, encoding="utf-8", errors="replace") as f:
        rel = assets.mdl_textures(f.read())
    base = os.path.dirname(mdl)
    return {role: nucleus.fetch(os.path.normpath(os.path.join(base, p)).replace(os.sep, "/"), cache_dir, log=log)
            for role, p in rel.items()}


@functools.lru_cache(maxsize=None)
def mean_albedo(albedo_path: str) -> np.ndarray:
    """Mean linear RGB of an sRGB albedo texture (read once per file)."""
    from PIL import Image
    a = np.asarray(Image.open(albedo_path).convert("RGB").reduce(8), np.float64) / 255.0
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    return lin.reshape(-1, 3).mean(0)


def ground_specs(cache_dir: str, log=None) -> dict[int, MaterialSpec]:
    """{class: MaterialSpec} for every ground class (textures fetched / generated into the cache)."""
    tex_dir = os.path.join(cache_dir, "world_textures")
    out = {}
    for k, (src, tile, tint, rs) in GROUND.items():
        name = lc.CLASS_NAMES[k]
        if k == lc.WATER:
            out[k] = MaterialSpec(name, color=WATER.color, roughness=WATER.roughness)
            continue
        if src.startswith("procedural:"):
            t = textures.crop_rows(tex_dir, f"crop_rows_{TEXTURE_VERSION}", seed=k)   # versioned like the others
            out[k] = MaterialSpec(name, {r: t[r] for r in ("albedo", "normal", "orm")}, t["tile_m"], tint,
                                  flip_green=False, roughness_scale=rs)
        else:
            out[k] = MaterialSpec(name, _base_textures(src, cache_dir, log), tile, tint, roughness_scale=rs)
    return out


def road_specs(cache_dir: str, log=None) -> list[MaterialSpec]:
    """MaterialSpecs in ``roads.SURFACES`` order."""
    out = []
    for s in SURFACES:
        if s in ROAD:
            src, tile, tint = ROAD[s]
            out.append(MaterialSpec(f"road_{s}", _base_textures(src, cache_dir, log), tile, tint))
        else:
            out.append(MaterialSpec(s, color=MARKINGS[s], roughness=0.55))
    return out


def building_specs(cache_dir: str) -> list[MaterialSpec]:
    """MaterialSpecs in ``buildings.SLOT_NAMES`` order (the texture repeat is baked into the mesh UVs, so tile 1)."""
    tex_dir = os.path.join(cache_dir, "world_textures")
    out = []
    for slot in BUILDING_SLOTS:
        gen, kind, tint = BUILDING[slot]
        if gen == "facade":
            t = textures.facade(tex_dir, f"wall_{kind}_{TEXTURE_VERSION}", kind=kind, windows=kind != "metal", tile_w=WALL_TILE[0],
                                tile_h=WALL_TILE[1])
        elif gen == "shingles":
            t = textures.shingles(tex_dir, f"roof_{kind}_{TEXTURE_VERSION}", kind=kind, tile_m=ROOF_TILE)
        else:
            out.append(MaterialSpec(slot, color=tint, roughness=0.8))
            continue
        out.append(MaterialSpec(slot, {r: t[r] for r in ("albedo", "normal", "orm")}, 1.0, tint, flip_green=False))
    return out


def calibrate_tints(spec: MaterialSpec, target_rgb, strength: float = 1.0, clamp=(0.35, 2.5)) -> tuple:
    """Tint that turns the texture's mean albedo into ``target_rgb`` (linear reflectance), blended by strength."""
    alb = spec.textures.get("albedo")
    if not alb:
        return spec.tint
    base = mean_albedo(alb) * np.asarray(spec.tint)
    ratio = np.clip(np.asarray(target_rgb, float) / np.maximum(base, 1e-4), *clamp)
    return tuple(float(t * (1 + strength * (r - 1))) for t, r in zip(spec.tint, ratio))
