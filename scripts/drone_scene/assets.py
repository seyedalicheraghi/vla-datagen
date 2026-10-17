"""What the drone scene uses from NVIDIA's asset server (Nucleus), and how to get the files.

Paths are relative to ``nucleus.NUCLEUS_ROOT``.  Isaac Sim can reference them as URLs; the offline
preview mirrors them locally first.  Ground materials are NVIDIA's photo-scanned "Base" materials: an
``.mdl`` file that calls ``OmniPBR`` with a colour, an ORM (R occlusion, G roughness, B metallic) and a
normal texture.  We read the texture file names out of the ``.mdl`` and author our own
UsdPreviewSurface with them, so Isaac and the Mitsuba preview see exactly the same PBR inputs.

Normal maps: NVIDIA's Base textures are DirectX-style (green = -dh/dv).  Measured on Grass_Cut, which
ships a height map: corr(N.g, dH/drow) = -0.62, corr(N.r, dH/dcol) = -0.67.  UsdPreviewSurface wants
OpenGL-style, so ``flip_green`` is set for them.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from . import nucleus

# ---------------------------------------------------------------------------
# Sky
# ---------------------------------------------------------------------------

SKY = "NVIDIA/Assets/Skies/Clear/noon_grass_4k.hdr"            # sunny park lawn with trees on the horizon
REFERENCE_SKY = "Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
REFERENCE_INTENSITY = 750.0       # Isaac Lab's drone and locomotion tasks expose this sky well at 750


# ---------------------------------------------------------------------------
# Ground materials
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaseMaterial:
    mdl: str                      # relative .mdl path
    tile_m: float                 # real-world size of one texture repeat (m), judged from the content
    flip_green: bool = True       # DirectX normal map
    roughness_scale: float = 1.0  # tweak on top of the ORM roughness
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)


BASE = "NVIDIA/Materials/Base"
MATERIALS: dict[str, BaseMaterial] = {
    # herringbone clay pavers: ~8 paver lengths per repeat, 200 mm pavers
    "pavers": BaseMaterial(f"{BASE}/Masonry/Brick_Pavers.mdl", tile_m=1.6),
    # Grass_Cut's albedo is ~0.06 (too dark for a sunlit lawn, ~0.1); fresh hardwood mulch is darker and
    # warmer than Mulch_Brown's photo
    "lawn": BaseMaterial(f"{BASE}/Natural/Grass_Cut.mdl", tile_m=1.5, tint=(1.45, 1.45, 1.25)),
    "mulch": BaseMaterial(f"{BASE}/Natural/Mulch_Brown.mdl", tile_m=1.2, tint=(0.62, 0.52, 0.42)),
    "curb": BaseMaterial(f"{BASE}/Masonry/Concrete_Rough.mdl", tile_m=1.0),
    "soil": BaseMaterial(f"{BASE}/Natural/Soil_Rocky.mdl", tile_m=1.5),
}

_TEXTURE_PARAMS = {
    "diffuse_texture": "albedo",
    "ORM_texture": "orm",
    "normalmap_texture": "normal",
    "reflectionroughness_texture": "roughness",
}


def mdl_textures(mdl_text: str) -> dict[str, str]:
    """{role: relative texture path} from an NVIDIA Base .mdl (albedo, orm, normal, roughness)."""
    out = {}
    for param, role in _TEXTURE_PARAMS.items():
        m = re.search(rf"\b{param}\s*:\s*texture_2d\(\s*\"([^\"]+)\"", mdl_text)
        if m:
            out[role] = m.group(1)
    return out


def material_textures(name: str, cache_dir: str, log=None) -> dict[str, str]:
    """Fetch one ground material's textures; return {role: local path}."""
    mat = MATERIALS[name]
    mdl_local = nucleus.fetch(mat.mdl, cache_dir, log=log)
    with open(mdl_local, encoding="utf-8", errors="replace") as f:
        rel = mdl_textures(f.read())
    if "albedo" not in rel:
        raise RuntimeError(f"no diffuse texture found in {mat.mdl}")
    base = os.path.dirname(mat.mdl)
    out = {}
    for role, path in rel.items():
        tex_rel = os.path.normpath(os.path.join(base, path)).replace(os.sep, "/")
        out[role] = nucleus.fetch(tex_rel, cache_dir, log=log)
    return out


# ---------------------------------------------------------------------------
# Props (NVIDIA vegetation / ArchVis assets, all modelled in centimetres, Z-up)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Prop:
    usd: str
    units: float = 0.01           # metres per unit of the asset


VEG = "NVIDIA/Assets/Vegetation"
PROPS: dict[str, Prop] = {
    "boxwood": Prop(f"{VEG}/Shrub/Boxwood.usd"),
    "hydrangea": Prop(f"{VEG}/Shrub/Hydrangea.usd"),
    "grass_a": Prop(f"{VEG}/Shrub/Grass_Short_A.usd"),
    "grass_b": Prop(f"{VEG}/Shrub/Grass_Short_B.usd"),
    "grass_c": Prop(f"{VEG}/Shrub/Grass_Short_C.usd"),
    "maple": Prop(f"{VEG}/Trees/Japanese_Maple.usd"),
    "leaves_maple": Prop(f"{VEG}/Debris/maplefall1.usd"),
    "leaves_oak": Prop(f"{VEG}/Debris/oakfall1.usd"),
}

CRAZYFLIE = "Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd"


def prop_path(name: str, cache_dir: str | None) -> str:
    """Local mirror path if mirrored (offline preview), else the URL (Isaac)."""
    return nucleus.local_or_url(PROPS[name].usd, cache_dir)
