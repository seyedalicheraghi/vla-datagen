"""Ground classes of a world: what covers each square metre (lawn, crop field, forest floor, water, ...).

Sources, coarse to fine:

1. ESA WorldCover 2021, 10 m classes, everywhere on Earth (``WORLDCOVER``).  Upsampled to the terrain's finer
   cells by blending the per-class indicator fields (bilinear) and taking the strongest class, with a little
   noise, so boundaries are organic curves instead of 10 m pixel stairs.
2. Overture land use / land cover / water polygons (OpenStreetMap-derived), rasterised on top: field edges,
   lawns, ponds and parking lots with their real outlines (``OVERTURE_LAND_USE``, ``OVERTURE_WATER``).

Roads and buildings are not ground classes: they are their own meshes laid on top.
"""

from __future__ import annotations

import numpy as np

from .geo import Grid

# our ground classes (index = material slot of the base material, see world.materials)
LAWN, MEADOW, CROP, FOREST, SHRUB, SOIL, WATER, WETLAND, CONCRETE, SAND = range(10)
CLASS_NAMES = ["lawn", "meadow", "crop", "forest_floor", "shrub", "soil", "water", "wetland", "concrete", "sand"]

# ESA WorldCover v200 codes -> our classes (built-up 50 is mostly gardens, lawns and roofs: roofs are buildings)
WORLDCOVER = {10: FOREST, 20: SHRUB, 30: MEADOW, 40: CROP, 50: LAWN, 60: SOIL, 70: SOIL, 80: WATER, 90: WETLAND,
              95: FOREST, 100: MEADOW, 0: LAWN}

# Overture base/land_use: (subtype or class) -> our class.  Classes are checked before subtypes.
OVERTURE_LAND_USE = {
    # classes
    "farmland": CROP, "orchard": MEADOW, "vineyard": MEADOW, "meadow": MEADOW, "grass": LAWN, "village_green": LAWN,
    "garden": LAWN, "park": LAWN, "pitch": LAWN, "golf_course": LAWN, "fairway": LAWN, "playground": LAWN,
    "cemetery": LAWN, "grave_yard": LAWN, "recreation_ground": LAWN, "parking": CONCRETE, "parking_space": CONCRETE,
    "construction": SOIL, "brownfield": SOIL, "landfill": SOIL, "quarry": SOIL, "farmyard": SOIL,
    "plant_nursery": MEADOW, "allotments": CROP, "greenhouse_horticulture": CONCRETE, "railway": SOIL,
    # subtypes
    "agriculture": CROP, "horticulture": CROP, "residential": LAWN, "education": LAWN, "religious": LAWN,
    "medical": LAWN, "golf": LAWN, "entertainment": LAWN, "recreation": LAWN, "resource_extraction": SOIL,
}
OVERTURE_LAND_COVER = {"forest": FOREST, "shrub": SHRUB, "grass": MEADOW, "crop": CROP, "barren": SOIL,
                       "wetland": WETLAND, "mangrove": FOREST, "moss": MEADOW, "snow": SOIL, "urban": LAWN}
OVERTURE_WATER = {"water", "lake", "pond", "reservoir", "river", "stream", "canal", "ocean", "human_made", "basin",
                  "swimming_pool", "wastewater"}


def from_worldcover(codes: Grid) -> Grid:
    """WorldCover code grid -> our class grid (same cells)."""
    lut = np.full(256, LAWN, np.uint8)
    for k, v in WORLDCOVER.items():
        lut[k] = v
    return Grid(lut[codes.data.astype(np.uint8)], codes.x0, codes.y1, codes.res)


def upsample_classes(classes: Grid, rect, res: float, seed: int = 0, wobble: float = 0.35) -> Grid:
    """Finer class grid with smooth, slightly irregular boundaries (see the module docstring)."""
    out = Grid.empty(rect, res, dtype=np.uint8)
    X, Y = out.centres()
    present = np.unique(classes.data)
    rng = np.random.default_rng(seed)
    # one smooth noise field shared by all classes: shifts boundaries by up to ~wobble cells of the source
    from scipy.ndimage import gaussian_filter, zoom
    ny, nx = out.shape
    coarse = rng.standard_normal((max(2, ny // 8), max(2, nx // 8)))
    noise = zoom(gaussian_filter(coarse, 1.0), (ny / coarse.shape[0], nx / coarse.shape[1]), order=1)[:ny, :nx]
    noise = noise / (noise.std() + 1e-9) * wobble
    best = np.full(out.shape, -np.inf)
    cls = np.zeros(out.shape, np.uint8)
    for k in present:
        ind = Grid((classes.data == k).astype(np.float32), classes.x0, classes.y1, classes.res)
        v = ind.sample(X, Y) + noise * (1 if k % 2 else -1) * 0.25
        better = v > best
        best = np.where(better, v, best)
        cls = np.where(better, k, cls).astype(np.uint8)
    out.data = cls
    return out


def burn(classes: Grid, polygons, value: int) -> int:
    """Rasterise shapely (Multi)Polygons in the local frame into the class grid; returns cells changed."""
    if not polygons:
        return 0
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    ny, nx = classes.shape
    mask = rasterize([(p, 1) for p in polygons if p is not None and not p.is_empty], out_shape=(ny, nx),
                     transform=from_origin(classes.x0, classes.y1, classes.res, classes.res), fill=0,
                     all_touched=False, dtype="uint8").astype(bool)
    changed = int((mask & (classes.data != value)).sum())
    classes.data[mask] = value
    return changed


def overture_class(subtype: str | None, cls: str | None, table: dict) -> int | None:
    for key in (cls, subtype):
        if key and key in table:
            return table[key]
    return None


def field_angles(classes: Grid, patch: np.ndarray, rng=None) -> dict:
    """Row direction (rad) per crop patch id: US Midwest fields follow the north-south / east-west survey grid,
    so rows run N-S or E-W; which one is drawn per patch."""
    rng = rng or np.random.default_rng(0)
    ids = np.unique(patch[classes.data == CROP]) if patch.shape == classes.shape else np.unique(patch)
    return {int(i): float(rng.choice([0.0, np.pi / 2])) for i in ids}
