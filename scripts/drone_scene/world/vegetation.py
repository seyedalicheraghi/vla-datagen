"""Trees of a world from a canopy height model: where each tree stands, how tall it is, which 3D tree it gets.

Detection (``detect_trees``): the canopy height model (CHM, metres above ground, 1 m cells) is smoothed and each
local maximum higher than ``min_height`` is a tree top.  The search window grows with the tree's height (a 25 m
oak has a wider crown than a 6 m ornamental), the usual variable-window local-maximum method of forestry
lidar.  The crown radius is half the distance to the nearest neighbouring top, capped by an allometric bound.

Placement (``tree_instances``): every top gets a species from ``SPECIES`` (broadleaf hardwoods with a share of
conifers), is scaled so the NVIDIA model's height equals the measured height (a little random squash/stretch of
the crown for variety), turned to a random heading and stood on the terrain.  Tops on buildings, on paved roads
or on the take-off/landing pads are dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geo import Grid

VEG = "NVIDIA/Assets/Vegetation/Trees"


@dataclass(frozen=True)
class Species:
    key: str
    usd: str
    kind: str                  # "broadleaf" or "conifer"
    weight: float              # relative frequency within its kind
    height_m: float | None = None      # height of the model at scale 1 (metres), None = measure it
    units: float = 0.01        # metres per unit of the asset (NVIDIA vegetation is modelled in cm)


# Midwest US mix (Iowa: oaks, maples, ash, honey locust in town; spruce windbreaks and yards)
SPECIES = [
    Species("red_oak", f"{VEG}/Red_Oak.usd", "broadleaf", 3.0),
    Species("sugar_maple", f"{VEG}/Sugar_Maple.usd", "broadleaf", 2.5),
    Species("honey_locust", f"{VEG}/Honey_Locust.usd", "broadleaf", 1.5),
    Species("red_ash", f"{VEG}/Red_Ash.usd", "broadleaf", 1.5),
    Species("shumard_oak", f"{VEG}/Shumard_Oak.usd", "broadleaf", 1.0),
    Species("norway_spruce", f"{VEG}/Norway_Spruce.usd", "conifer", 1.5),
    Species("colorado_spruce", f"{VEG}/Colorado_Spruce.usd", "conifer", 1.0),
]


def detect_trees(chm: Grid, min_height: float = 3.0, smooth_m: float = 1.0) -> np.ndarray:
    """(N, 4) array of tree tops: x, y, height (m), crown radius (m)."""
    from scipy.ndimage import gaussian_filter, maximum_filter
    from scipy.spatial import cKDTree

    res = chm.res
    h = np.nan_to_num(chm.data.astype(np.float32), nan=0.0)
    h = np.clip(h, 0, 80)
    hs = gaussian_filter(h, smooth_m / res) if smooth_m > 0 else h
    peak = np.zeros(h.shape, bool)
    # variable window: tops of trees in each height band must be the maximum over that band's window
    for lo, hi, win_m in ((min_height, 8.0, 3.0), (8.0, 15.0, 5.0), (15.0, 25.0, 7.0), (25.0, 999.0, 9.0)):
        size = max(3, int(round(win_m / res)) | 1)
        m = maximum_filter(hs, size=size, mode="nearest")
        peak |= (hs >= m - 1e-6) & (hs >= lo) & (hs < hi)
    j, i = np.nonzero(peak)
    if len(i) == 0:
        return np.zeros((0, 4))
    x = chm.x0 + (i + 0.5) * res
    y = chm.y1 - (j + 0.5) * res
    ht = h[j, i]
    # non-maximum suppression: flat tops (equal cells) and shoulders give several candidates per tree; keep
    # the tallest and drop candidates within half its search window
    order = np.argsort(-hs[j, i], kind="stable")
    x, y, ht = x[order], y[order], ht[order]
    radius = np.select([ht < 8, ht < 15, ht < 25], [1.5, 2.5, 3.5], 4.5)
    kd = cKDTree(np.column_stack([x, y]))
    taken = np.zeros(len(x), bool)
    keep = []
    for k in range(len(x)):
        if taken[k]:
            continue
        keep.append(k)
        taken[kd.query_ball_point((x[k], y[k]), radius[k])] = True
    x, y, ht = x[keep], y[keep], ht[keep]
    pts = np.column_stack([x, y])
    if len(pts) > 1:
        d, _ = cKDTree(pts).query(pts, k=2)
        nn = d[:, 1]
    else:
        nn = np.full(len(pts), 10.0)
    crown = np.minimum(np.maximum(nn / 2, 1.0), 1.0 + 0.3 * ht)
    return np.column_stack([x, y, ht, crown])


def drop_trees(trees: np.ndarray, buildings=(), roads=(), pads=(), road_buffer: float = 0.5) -> np.ndarray:
    """Remove tops inside building footprints, on paved road surfaces, or within a pad's clear radius.
    ``roads`` are (shapely LineString, width) pairs; ``pads`` are (x, y, clear_radius)."""
    if len(trees) == 0:
        return trees
    import shapely
    keep = np.ones(len(trees), bool)
    for x, y, r in pads:
        keep &= np.hypot(trees[:, 0] - x, trees[:, 1] - y) > r
    geoms = [b for b in buildings if b is not None]
    geoms += [line.buffer(w / 2 + road_buffer, cap_style="flat") for line, w in roads]
    if geoms:
        tree = shapely.STRtree(geoms)
        pts = shapely.points(trees[:, 0], trees[:, 1])
        hit = tree.query(pts, predicate="intersects")
        keep[np.unique(hit[0])] = False
    return trees[keep]


def assign_species(trees: np.ndarray, conifer_share: float = 0.12, seed: int = 0,
                   model_heights: dict | None = None) -> np.ndarray:
    """Index into SPECIES per tree.  With ``model_heights`` a species is likelier the closer its model is to the
    tree's height (a 3.6 m model stretched to 20 m would look like a giant sapling)."""
    rng = np.random.default_rng(seed)
    n = len(trees)
    is_con = rng.random(n) < conifer_share
    out = np.zeros(n, np.int32)
    for kind, sel in (("conifer", is_con), ("broadleaf", ~is_con)):
        ks = [k for k, s in enumerate(SPECIES) if s.kind == kind]
        if not ks or not sel.any():
            continue
        w = np.array([SPECIES[k].weight for k in ks], float)[None, :].repeat(int(sel.sum()), 0)
        if model_heights:
            mh = np.array([float(model_heights.get(SPECIES[k].key) or 15.0) for k in ks])[None, :]
            ratio = np.log(trees[sel, 2][:, None] / mh)
            w = w * np.exp(-0.5 * (ratio / 0.45) ** 2)              # within ~1.6x of the model's own height
            w[w.sum(1) < 1e-12] = 1.0
        p = w / w.sum(1, keepdims=True)
        u = rng.random(int(sel.sum()))[:, None]
        out[sel] = np.asarray(ks)[np.minimum((u > np.cumsum(p, 1)).sum(1), len(ks) - 1)]
    return out


def infill_trees(chm: Grid, trees: np.ndarray, min_height: float = 4.0, cover: float = 0.42, seed: int = 0) -> np.ndarray:
    """Extra trees where the canopy height model shows canopy that no detected tree's crown covers (detection merges
    the touching crowns of closed woods).  A tree of height h covers a disc of radius ``cover * h``; new trees go
    on a jittered grid whose spacing grows with the local canopy height.  Returns (N, 4) like ``detect_trees``."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed + 11)
    out = []
    have = trees[:, :3] if len(trees) else np.zeros((0, 3))
    for lo, hi, spacing in ((min_height, 8.0, 4.0), (8.0, 15.0, 6.5), (15.0, 25.0, 9.0), (25.0, 999.0, 12.0)):
        x0, x1, y0, y1 = chm.rect
        gx, gy = np.meshgrid(np.arange(x0, x1, spacing), np.arange(y0, y1, spacing))
        px = gx.ravel() + rng.uniform(0.15, 0.85, gx.size) * spacing
        py = gy.ravel() + rng.uniform(0.15, 0.85, gy.size) * spacing
        ph = chm.sample(px, py, "nearest")
        ok = (ph >= lo) & (ph < hi)
        px, py, ph = px[ok], py[ok], ph[ok]
        if len(px) == 0:
            continue
        if len(have):
            d, i = cKDTree(have[:, :2]).query(np.column_stack([px, py]))
            ok = d > np.maximum(cover * have[i, 2], 0.6 * spacing)
            px, py, ph = px[ok], py[ok], ph[ok]
        new = np.column_stack([px, py, ph, np.full(len(px), spacing / 2)])
        out.append(new)
        have = np.concatenate([have, new[:, :3]]) if len(have) else new[:, :3]
    return np.concatenate(out) if out else np.zeros((0, 4))


def tree_instances(trees: np.ndarray, species: np.ndarray, model_heights: dict, ground, seed: int = 0) -> dict:
    """Per species key: {'positions' (n, 3), 'orientations' (n, 4) wxyz, 'scales' (n, 3), 'heights' (n,)}.
    ``model_heights[key]`` is the model's height in metres at scale 1 (after its unit conversion)."""
    rng = np.random.default_rng(seed + 1)
    out = {}
    if len(trees) == 0:
        return out
    z = ground(trees[:, 0], trees[:, 1]) - 0.05
    yaw = rng.uniform(0, 2 * math.pi, len(trees))
    squash = rng.uniform(0.9, 1.12, len(trees))
    for k, sp in enumerate(SPECIES):
        sel = species == k
        if not sel.any():
            continue
        mh = float(model_heights.get(sp.key) or 15.0)
        s = trees[sel, 2] / mh
        n = int(sel.sum())
        out[sp.key] = {
            "positions": np.column_stack([trees[sel, 0], trees[sel, 1], z[sel]]),
            "orientations": np.column_stack([np.cos(yaw[sel] / 2), np.zeros(n), np.zeros(n), np.sin(yaw[sel] / 2)]),
            "scales": np.column_stack([s * squash[sel], s * squash[sel], s]) * sp.units,
            "heights": trees[sel, 2],
        }
    return out
