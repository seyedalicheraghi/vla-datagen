"""Procedural PBR textures the NVIDIA library has no equivalent for (numpy + Pillow): row crops seen from above,
house walls with windows (lap siding, brick, ribbed metal) and roofs (asphalt shingles, standing-seam metal).

``crop_rows`` makes a seamlessly tiling texture set (albedo sRGB, OpenGL normal, ORM) of a row-crop field: plant
rows at a real row spacing (0.76 m = the 30-inch rows of US corn and soybean) with leafy canopies over soil.
``growth`` 0..1 sets how much of the inter-row soil the canopy covers (0.95 ~ July corn, 0.4 ~ June soybean).
All noise is FFT-filtered, so it is periodic and the texture repeats without seams.
"""

from __future__ import annotations

import os

import numpy as np


def periodic_noise(n: int, scale_px: float, rng, aspect: float = 1.0, width: int | None = None) -> np.ndarray:
    """Seamless band-limited noise (n rows x ``width`` or n columns), zero mean, unit std; features ~scale_px
    across (x stretched by aspect).  Periodic along both axes, also when rectangular."""
    w = width or n
    white = rng.standard_normal((n, w))
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(w)[None, :] / aspect
    f = np.sqrt(fx * fx + fy * fy) * scale_px
    spec = np.fft.fft2(white) * np.exp(-(f ** 2) * 4.0)
    out = np.real(np.fft.ifft2(spec))
    return (out - out.mean()) / (out.std() + 1e-12)


def _srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def crop_rows(out_dir: str, name: str = "crop_rows", *, size_px: int = 1024, rows: int = 4, spacing_m: float = 0.76,
              growth: float = 0.85, seed: int = 0, overwrite: bool = False) -> dict:
    """Write {name}_albedo.png, _normal.png, _orm.png; return {'albedo', 'normal', 'orm', 'tile_m'}.

    Rows run along u (texture x).  The tile is ``rows * spacing_m`` square."""
    tile_m = rows * spacing_m
    paths = {k: os.path.join(out_dir, f"{name}_{k}.png") for k in ("albedo", "normal", "orm")}
    if not overwrite and all(os.path.exists(p) for p in paths.values()):
        return dict(paths, tile_m=tile_m)
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = size_px
    px_m = tile_m / n
    y = (np.arange(n) + 0.5) * px_m
    # distance to the nearest row centre, per texture row
    d_row = np.abs(((y / spacing_m) % 1.0) - 0.5) * spacing_m
    half_w = 0.5 * spacing_m * np.clip(growth, 0.05, 1.0)
    leaves = periodic_noise(n, 0.06 / px_m, rng, aspect=2.5)        # leaf clusters, elongated along the row
    fine = periodic_noise(n, 0.015 / px_m, rng)
    wobble = periodic_noise(n, 0.25 / px_m, rng, aspect=4.0)        # canopy edge meanders
    edge = half_w * (1.0 + 0.18 * wobble + 0.10 * leaves)
    cover = np.clip((edge - d_row[:, None]) / (0.06 * spacing_m) + 0.5, 0, 1)
    cover *= np.clip(0.75 + 0.5 * leaves, 0, 1) ** 0.3
    # canopy height (m): dome over the row, leafy relief on top
    canopy_h = cover * (0.35 + 0.10 * leaves + 0.03 * fine) * (1 - (d_row[:, None] / (half_w + 1e-6)) ** 2).clip(0, 1) ** 0.5
    soil_h = 0.01 * periodic_noise(n, 0.03 / px_m, rng)
    height = np.maximum(canopy_h, soil_h)
    # albedo (linear): soil brown with clods, leaves mid green varying, darker deep in the canopy
    soil = np.array([0.18, 0.13, 0.09])[None, None, :] * (1 + 0.15 * fine[..., None] + 0.1 * leaves[..., None])
    hue = 0.5 + 0.5 * np.tanh(leaves)[..., None]
    leaf = (np.array([0.055, 0.13, 0.035]) * (1 - hue) + np.array([0.09, 0.17, 0.045]) * hue)[...] \
        * (1 + 0.10 * fine[..., None] + 0.12 * leaves[..., None])
    depth_shade = 0.6 + 0.4 * np.clip(canopy_h / 0.45, 0, 1)[..., None]
    albedo = soil * (1 - cover[..., None]) + leaf * depth_shade * cover[..., None]
    # normals from the height field (OpenGL convention: green = +dh/dv up)
    gy, gx = np.gradient(np.pad(height, 1, mode="wrap"), px_m)
    gx, gy = gx[1:-1, 1:-1], gy[1:-1, 1:-1]
    nrm = np.stack([-gx, gy, np.ones_like(gx)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    # ORM: occlusion lower between and under leaves, rough soil, slightly glossy leaves
    occl = np.clip(1.0 - 0.45 * cover * (1 - np.clip(canopy_h / 0.45, 0, 1)) - 0.25 * (1 - cover) * cover.mean(), 0.3, 1)
    rough = 0.92 * (1 - cover) + 0.6 * cover
    orm = np.stack([occl, rough, np.zeros_like(rough)], -1)
    Image.fromarray((_srgb(albedo) * 255 + 0.5).astype(np.uint8)).save(paths["albedo"])
    Image.fromarray(((nrm * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8)).save(paths["normal"])
    Image.fromarray((orm * 255 + 0.5).astype(np.uint8)).save(paths["orm"])
    return dict(paths, tile_m=tile_m)


def _save_set(paths, albedo, nrm, orm):
    from PIL import Image
    Image.fromarray((_srgb(albedo) * 255 + 0.5).astype(np.uint8)).save(paths["albedo"])
    Image.fromarray(((nrm * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8)).save(paths["normal"])
    Image.fromarray((np.clip(orm, 0, 1) * 255 + 0.5).astype(np.uint8)).save(paths["orm"])


def _normals_from_height(height, px_m):
    gy, gx = np.gradient(np.pad(height, 1, mode="wrap"), px_m)
    gx, gy = gx[1:-1, 1:-1], gy[1:-1, 1:-1]
    n = np.stack([-gx, gy, np.ones_like(gx)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def _paths(out_dir, name):
    return {k: os.path.join(out_dir, f"{name}_{k}.png") for k in ("albedo", "normal", "orm")}


def facade(out_dir: str, name: str, *, kind: str = "siding", windows: bool = True, size_px: int = 1024,
           tile_w: float = 4.0, tile_h: float = 3.0, seed: int = 0, overwrite: bool = False) -> dict:
    """One storey (tile_h) by tile_w of a wall: ``kind`` siding (white lap boards, tint it), brick or metal
    (vertical corrugation), with one window per tile when ``windows``.  Texture v = 0 is the floor level; the
    window sill is 0.9 m above it.  Returns {'albedo', 'normal', 'orm', 'tile_m': (tile_w, tile_h)}."""
    paths = _paths(out_dir, name)
    if not overwrite and all(os.path.exists(p) for p in paths.values()):
        return dict(paths, tile_m=(tile_w, tile_h))
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    W = size_px
    H = int(round(size_px * tile_h / tile_w))
    px = tile_w / W
    x = (np.arange(W) + 0.5) * px
    y = (np.arange(H)[::-1] + 0.5) * px                    # image row 0 is the top: y up from the floor
    X, Y = np.meshgrid(x, y)
    n1 = periodic_noise(H, 0.05 / px, rng, width=W)
    # pattern periods divide the tile exactly (and brick courses come in an even number), so it repeats seamlessly
    if kind == "brick":
        course, brick = tile_h / 46, tile_w / 20            # ~65 x 200 mm: US modular brick with mortar
        row = np.floor(Y / course)
        xs = X + (row % 2) * brick / 2
        mortar = (np.minimum(Y % course, course - Y % course) < 0.006) | (np.minimum(xs % brick, brick - xs % brick) < 0.006)
        tone = 0.85 + 0.3 * (_hash_img(np.floor(xs / brick), row, seed) - 0.5)
        base = np.where(mortar[..., None], np.array([0.42, 0.40, 0.37]), np.array([0.36, 0.13, 0.08]) * tone[..., None])
        height = np.where(mortar, -0.004, 0.0)
        rough = np.where(mortar, 0.95, 0.85)
    elif kind == "metal":
        rib = tile_w / 17                                    # ~9" ribbed steel siding
        prof = np.cos(2 * np.pi * X / rib)
        base = np.ones((H, W, 3)) * np.array([0.62, 0.63, 0.64])
        height = 0.008 * np.clip(prof, 0, 1) ** 4
        rough = np.full((H, W), 0.45)
    else:
        lap = tile_h / 20                                    # 6" lap siding
        t = (Y % lap) / lap
        height = 0.012 * t - 0.012 * (t > 0.97)
        base = np.ones((H, W, 3)) * np.array([0.78, 0.78, 0.76]) * (0.9 + 0.1 * t)[..., None]
        rough = np.full((H, W), 0.6)
    base = base * (1 + 0.04 * n1[..., None])
    metal = np.zeros((H, W))
    if windows:
        cx, ww, wh, sill = tile_w / 2, 1.0, 1.35, 0.9
        trim = 0.09
        inx = np.abs(X - cx) < ww / 2 + trim
        iny = (Y > sill - trim) & (Y < sill + wh + trim)
        frame = inx & iny
        glass = (np.abs(X - cx) < ww / 2) & (Y > sill) & (Y < sill + wh)
        mullion = glass & (np.abs(Y - (sill + wh / 2)) < 0.025)
        base = np.where(frame[..., None], np.array([0.80, 0.80, 0.78]), base)
        glass_c = np.array([0.03, 0.04, 0.05]) * (1 + 0.3 * (Y - sill)[..., None] / wh)
        base = np.where((glass & ~mullion)[..., None], glass_c, base)
        height = np.where(frame, 0.02, height)
        height = np.where(glass & ~mullion, -0.03, height)
        rough = np.where(glass & ~mullion, 0.05, np.where(frame, 0.5, rough))
    occl = np.clip(1.0 - 25.0 * np.maximum(0, -height), 0.5, 1.0)
    orm = np.stack([occl, rough, metal], -1)
    _save_set(paths, base, _normals_from_height(height, px), orm)
    return dict(paths, tile_m=(tile_w, tile_h))


def _hash_img(i, j, seed):
    h = (i.astype(np.int64) * 73856093) ^ (j.astype(np.int64) * 19349663) ^ (seed * 83492791)
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return (h & 0xFFFF) / 65535.0


def shingles(out_dir: str, name: str = "shingles", *, kind: str = "asphalt", size_px: int = 1024, tile_m: float = 2.0,
             seed: int = 0, overwrite: bool = False) -> dict:
    """Roof covering, u along the ridge, v up the slope: ``asphalt`` 3-tab shingles (5.6" exposure, 12" tabs;
    neutral grey, tint it) or ``metal`` standing-seam panels (16" seams)."""
    paths = _paths(out_dir, name)
    if not overwrite and all(os.path.exists(p) for p in paths.values()):
        return dict(paths, tile_m=tile_m)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = size_px
    px = tile_m / n
    c = (np.arange(n) + 0.5) * px
    X, Y = np.meshgrid(c, c[::-1])
    grain = periodic_noise(n, 0.004 / px, rng)
    blotch = periodic_noise(n, 0.4 / px, rng)
    if kind == "metal":
        seam = tile_m / 5                                    # ~16" standing seams
        d = np.minimum(X % seam, seam - X % seam)
        height = 0.02 * (d < 0.012)
        base = np.ones((n, n, 3)) * np.array([0.55, 0.56, 0.57]) * (1 + 0.03 * blotch[..., None])
        orm = np.stack([np.ones((n, n)), np.full((n, n), 0.35), np.full((n, n), 0.9)], -1)
    else:
        expo, tab = tile_m / 14, tile_m / 7                 # ~5.6" exposure, ~11" tabs; 14 courses (even)
        row = np.floor(Y / expo)
        xs = X + (row % 2) * tab / 2
        t = (Y % expo) / expo
        slot = np.minimum(xs % tab, tab - xs % tab) < 0.004
        tone = 0.88 + 0.24 * (_hash_img(np.floor(xs / tab), row, seed) - 0.5)
        height = 0.004 * t + np.where(slot, -0.003, 0.0)
        base = np.ones((n, n, 3)) * np.array([0.30, 0.30, 0.30]) * (tone * (1 + 0.12 * grain + 0.05 * blotch))[..., None]
        base = base * np.where((t < 0.08)[..., None], 0.55, 1.0)        # shadow line under each course
        orm = np.stack([np.clip(0.75 + 0.25 * t, 0, 1), np.full((n, n), 0.88), np.zeros((n, n))], -1)
    _save_set(paths, base, _normals_from_height(height, px), orm)
    return dict(paths, tile_m=tile_m)
