"""Procedural PBR texture sets (numpy only) that make the Isaac Sim render look photographic.

Two entry points feed ``render_scenario.py``:

* :func:`machine_textures` - seamless albedo / roughness / normal sets for the
  ``spec.DEFAULT_MATERIALS`` names (painted steel with a dust film, chips, mud
  spatter and rain streaks; dusty black trim; brushed worn steel; tyre rubber;
  tinted dusty glass; rust ...).  Every map tiles seamlessly, so a UV mapping with
  ``repeat`` wrap and ``metres_per_tile`` of real-world size is enough.  Flat-panel
  paint (body / red / orange / white / hub) uses 3-4 m tiles and its discrete marks
  (chips, drips, spatter) are many, small, faint, brown-grey and uniformly scattered
  rather than a few dark groups: a 6 m panel then shows no repeat.  **Always scale the
  UVs by the returned ``metres_per_tile``**, the sizes differ per material (1.5 - 4 m).
* :func:`terrain_detail_maps` - a micro-relief normal map and a roughness map that
  have exactly the pixel layout of ``terrain.bake_albedo`` (same rect, same
  ``px_per_m``): gravel and cobbles, clods, cracks on the dry lane, tyre-tread
  imprints in the tyre tracks, wandering erosion rills on the earth slopes and angular
  fractured blocks on the blast-cut bench face.  The relief is anchored in world space
  and band-limited to the pixel size, so what you see scales with ``px_per_m``: at the
  albedo's own 24 px/m you get clods, cobbles and tread chevrons, while pebbles, cracks
  and grain need a denser map.  **Bake the detail maps 2-3x denser than the albedo**
  (:func:`detail_px_per_m`; the texture inputs of a material may differ in resolution
  on the same UVs), or a near-camera sub-rect densely (40 x 16 m at 64 px/m takes about
  12 s).  :func:`terrain_detail_cavity` bakes the matching micro-AO: multiply it into the
  albedo to get dark tread grooves, cracks and pebble gaps, which no renderer derives
  from a normal map alone.

Conventions
-----------
* albedo       sRGB uint8 (``sourceColorSpace = sRGB``).
* roughness    uint8, linear (``raw``): value / 255 is the PBR roughness.
* normal       uint8 tangent space, OpenGL (+Y up = towards image row 0), ``raw``;
               flat is (128, 128, 255).  +X is the image's column direction.
* Image rows run top to bottom and **row direction is treated as "down"** for rain
  streaks, drips and dust washing; a mapping that puts V up on a vertical face (the
  USD default) then gets streaks that run down the machine.

Everything is composited in linear light, tiled via FFT-filtered (hence periodic)
noise and wrap-around stamping, and is deterministic for a given seed.
"""

from __future__ import annotations

import math
import os
import zlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

__all__ = ["TextureSet", "machine_textures", "terrain_detail_maps", "terrain_detail_cavity", "detail_px_per_m",
           "save_texture_set", "save_terrain_detail_maps", "srgb_to_linear", "linear_to_srgb"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class TextureSet:
    albedo: np.ndarray        # (H, W, 3) uint8, sRGB
    roughness: np.ndarray     # (H, W) uint8, linear: value / 255 = roughness
    normal: np.ndarray        # (H, W, 3) uint8, tangent space, OpenGL (+Y up), flat = (128, 128, 255)
    metallic: float
    metres_per_tile: float
    opacity: float = 1.0
    clearcoat: float = 0.0


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float32), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055).astype(np.float32)


def _srgb_const(rgb) -> np.ndarray:
    return srgb_to_linear(np.array(rgb, dtype=np.float32))


def _to_u8_srgb(lin: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    s = linear_to_srgb(lin) * 255.0
    if rng is not None:                                   # +-0.5 LSB dither: no banding in dusty gradients
        s = s + rng.random(s.shape, dtype=np.float32) - 0.5
    return np.clip(s + 0.5, 0, 255).astype(np.uint8)


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _mix(a, b, t):
    return a + (b - a) * t


def _encode_normal(nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    """Unnormalised (nx, ny, 1) -> uint8 tangent-space normal map."""
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + 1.0)
    out = np.empty(nx.shape + (3,), np.uint8)
    out[..., 0] = np.floor(nx * inv * 127.5 + 128.0).clip(0, 255)
    out[..., 1] = np.floor(ny * inv * 127.5 + 128.0).clip(0, 255)
    out[..., 2] = np.floor(inv * 127.5 + 128.0).clip(0, 255)
    return out


def _rough_u8(r: np.ndarray) -> np.ndarray:
    return np.clip(r * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Periodic noise toolbox (FFT based: everything wraps seamlessly)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _fgrid(n: int):
    fy = (np.fft.fftfreq(n) * n).astype(np.float32)[:, None]      # cycles per tile
    fx = (np.fft.rfftfreq(n) * n).astype(np.float32)[None, :]
    return fy, fx


def nz(rng, n, lam_hi, lam_lo, beta=2.5, stretch=(1.0, 1.0)):
    """Zero-mean unit-std periodic noise.

    ``lam_hi`` / ``lam_lo`` are the largest / smallest feature wavelengths in pixels of
    a 1024 px tile (they scale with ``n``, so a 256 px tile is the same pattern, coarser).
    Power spectrum ~ f^-beta inside the band.  ``stretch=(sx, sy)`` elongates the
    features along x (columns) / y (rows).
    """
    fy, fx = _fgrid(n)
    f = np.sqrt((fx / stretch[0]) ** 2 + (fy / stretch[1]) ** 2)
    f = np.maximum(f, 1e-3)
    lo, hi = 1024.0 / lam_hi, 1024.0 / lam_lo
    amp = f ** (-beta / 2.0) * (1.0 - np.exp(-(f / lo) ** 2)) * np.exp(-(f / hi) ** 2)
    amp[0, 0] = 0.0
    w = rng.standard_normal((n, n), dtype=np.float32)
    out = np.fft.irfft2(np.fft.rfft2(w) * amp, s=(n, n)).astype(np.float32)
    out -= out.mean()
    out /= out.std() + 1e-9
    return out


def blur(img, sx, sy=None):
    """Periodic gaussian blur, sigma in pixels (of the actual image)."""
    sy = sx if sy is None else sy
    n = img.shape[0]
    fy, fx = _fgrid(n)
    g = np.exp(-2.0 * math.pi ** 2 * ((sx * fx / n) ** 2 + (sy * fy / n) ** 2)).astype(np.float32)
    return np.fft.irfft2(np.fft.rfft2(img) * g, s=img.shape).astype(np.float32)


def streak_down(img, length):
    """Smear ``img`` towards increasing rows with an exponential tail (periodic).

    A single bright pixel becomes a drip: 1.0 at the source decaying as exp(-k/length)."""
    n = img.shape[0]
    q = math.exp(-1.0 / max(length, 1e-3))
    m = np.arange(n, dtype=np.float32)[:, None]
    h = 1.0 / (1.0 - q * np.exp(-2j * np.pi * m / n))
    return np.fft.irfft2(np.fft.rfft2(img) * h, s=img.shape).astype(np.float32)


def periodic_grad(h):
    """Central differences of a periodic field: (d/dcol, d/drow) in units per pixel."""
    return ((np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5,
            (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5)


def height_to_normal(h, metres_per_px, gain=1.0):
    """Periodic height field (metres) -> uint8 tangent-space normal map (OpenGL)."""
    dcol, drow = periodic_grad(h)
    k = gain / metres_per_px
    return _encode_normal(-dcol * k, drow * k)         # +Y = towards row 0 = -drow


def discs(n, cx, cy, r, soft=1.0, val=None):
    """Anti-aliased discs (pixel units, wrap-around), max-composited. Returns (n, n) float32 in 0..1."""
    out = np.zeros(n * n, np.float32)
    cx, cy, r = (np.asarray(a, np.float32) for a in (cx, cy, r))
    if len(r) == 0:
        return out.reshape(n, n)
    val = None if val is None else np.asarray(val, np.float32)
    rr = np.ceil(r + soft).astype(int)
    for R in np.unique(rr):
        idx = np.nonzero(rr == R)[0]
        ox = np.arange(-R, R + 2)
        x0, y0 = np.floor(cx[idx]).astype(int), np.floor(cy[idx]).astype(int)
        px = x0[:, None, None] + ox[None, None, :]
        py = y0[:, None, None] + ox[None, :, None]
        d = np.hypot(px - cx[idx][:, None, None], py - cy[idx][:, None, None])
        a = np.clip((r[idx][:, None, None] - d) / soft + 0.5, 0.0, 1.0)
        if val is not None:
            a = a * val[idx][:, None, None]
        np.maximum.at(out, ((py % n) * n + (px % n)).ravel(), a.ravel().astype(np.float32))
    return out.reshape(n, n)


def polylines(n, rng, count, length_px, width_px=0.9, angle=None, angle_jitter=0.35,
              curve=0.35, intensity=(0.4, 1.0), taper=True):
    """Random wrapped polylines ("scratches") splatted with bilinear weights, then blurred. -> (n, n) float32."""
    flat = []
    wts = []
    for i in range(count):
        L = float(length_px[i]) if hasattr(length_px, "__len__") else float(length_px)
        steps = max(int(L / 0.6), 4)
        t = np.linspace(0.0, 1.0, steps)
        th0 = (rng.uniform(0, 2 * math.pi) if angle is None else angle + rng.normal(0, angle_jitter))
        th = th0 + curve * rng.normal() * (t - 0.5) + 0.08 * np.cumsum(rng.normal(0, 0.15, steps)) / math.sqrt(steps)
        sx, sy = rng.uniform(0, n, 2)
        px = sx + np.cumsum(np.cos(th)) * (L / steps)
        py = sy + np.cumsum(np.sin(th)) * (L / steps)
        w = rng.uniform(*intensity) * (np.sin(math.pi * t) ** 0.6 if taper else 1.0) * (L / steps) / 0.6
        x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
        fx, fy = px - x0, py - y0
        for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            flat.append(((y0 + dy) % n) * n + ((x0 + dx) % n))
            wts.append(wt * w)
    if not flat:
        return np.zeros((n, n), np.float32)
    acc = np.bincount(np.concatenate(flat), weights=np.concatenate(wts), minlength=n * n).reshape(n, n).astype(np.float32)
    return np.clip(blur(acc, width_px * 0.55) * 2.2, 0.0, 1.5)


def voronoi_f12(n, cells, rng, jitter=0.95):
    """Periodic cellular noise: (F1, F2) feature distances in cell units; ``cells`` cells per tile side."""
    G = int(cells)
    jx = rng.random((G, G), dtype=np.float32)
    jy = rng.random((G, G), dtype=np.float32)
    pos = (np.arange(n, dtype=np.float32) + 0.5) * (G / n)
    ci = np.floor(pos).astype(int)
    f1 = np.full((n, n), 9.0, np.float32)
    f2 = np.full((n, n), 9.0, np.float32)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            gx = (ci + ox)[None, :]
            gy = (ci + oy)[:, None]
            cx = gx + 0.5 + jitter * (jx[gy % G, gx % G] - 0.5)
            cy = gy + 0.5 + jitter * (jy[gy % G, gx % G] - 0.5)
            d = np.hypot(pos[None, :] - cx, pos[:, None] - cy)
            f2 = np.minimum(f2, np.maximum(f1, d))
            f1 = np.minimum(f1, d)
    return f1, f2


def voronoi_cells(n, cells, rng, jitter=0.95):
    """Periodic cellular noise with the nearest feature point: (F1, F2, dx, dy, cell_id), distances in cell units
    (dx, dy = offset from the nearest feature point); ``cells`` cells per tile side."""
    G = int(cells)
    jx = rng.random((G, G), dtype=np.float32)
    jy = rng.random((G, G), dtype=np.float32)
    pos = (np.arange(n, dtype=np.float32) + 0.5) * (G / n)
    ci = np.floor(pos).astype(int)
    f1 = np.full((n, n), 9.0, np.float32)
    f2 = np.full((n, n), 9.0, np.float32)
    bx = np.zeros((n, n), np.float32)
    by = np.zeros((n, n), np.float32)
    bid = np.zeros((n, n), np.int32)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            gx = (ci + ox)[None, :]
            gy = (ci + oy)[:, None]
            cx = gx + 0.5 + jitter * (jx[gy % G, gx % G] - 0.5)
            cy = gy + 0.5 + jitter * (jy[gy % G, gx % G] - 0.5)
            ddx = pos[None, :] - cx
            ddy = pos[:, None] - cy
            d = np.hypot(ddx, ddy)
            closer = d < f1
            f2 = np.minimum(f2, np.maximum(f1, d))
            bx = np.where(closer, ddx, bx)
            by = np.where(closer, ddy, by)
            bid = np.where(closer, (gy % G) * G + (gx % G), bid)
            f1 = np.minimum(f1, d)
    return f1, f2, bx, by, bid


# ---------------------------------------------------------------------------
# Machine materials
# ---------------------------------------------------------------------------

DUST_RGB = (0.60, 0.52, 0.40)       # quarry dust film (sRGB), warm tan
DUST_DARK_RGB = (0.46, 0.445, 0.425)  # the same dust on black plastic / rubber: pale, neutral, a touch cool
MUD_RGB = (0.38, 0.29, 0.20)
GRIME_RGB = (0.09, 0.075, 0.06)
DIRT_RGB = (0.27, 0.22, 0.17)     # brown-grey grime for drips, spatter and deep scratches (never near-black)
PRIMER_RGB = (0.30, 0.245, 0.20)  # grey-brown primer / red oxide under chipped paint
RUST_RGB = (0.42, 0.20, 0.08)


@dataclass(frozen=True)
class _Style:
    rough: float = 0.34        # clean paint roughness
    dust: float = 0.16         # mean dust film coverage (0..1)
    grime: float = 0.6         # oil, drips, dark streaks
    chips: float = 1.0
    scratches: float = 1.0
    fade: float = 0.10         # chalking / sun bleaching
    mud: float = 1.0
    rust_bleed: float = 0.5
    grain: float = 0.0         # plastic / cast moulding grain
    metres: float = 4.0        # flat-panel paint: big tile so that nothing repeats on one panel
    dent: float = 1.0
    dust_contrast: float = 0.85


# Paint tiles are 3-4 m: discrete marks (chips, drips, spatter) are deliberately many, small and faint and
# uniformly scattered, so no group of them is a "landmark" and a 6 m panel does not read as wallpaper.
_STYLES = {
    "body":   _Style(dust=0.18, metres=4.0),
    "hub":    _Style(rough=0.42, dust=0.22, grime=0.8, chips=1.3, scratches=1.3, mud=1.3, rust_bleed=0.8, metres=3.0),
    "red":    _Style(rough=0.34, dust=0.09, grime=0.6, fade=0.10, chips=0.9, metres=4.0, dust_contrast=0.60),
    "orange": _Style(rough=0.34, dust=0.15, grime=0.6, fade=0.13, chips=0.9, metres=4.0),
    "white":  _Style(rough=0.32, dust=0.12, grime=0.75, fade=0.05, chips=0.7, mud=0.7, rust_bleed=0.7, metres=4.0),
    "accent": _Style(rough=0.50, dust=0.07, grime=0.30, fade=0.16, chips=0.30, scratches=1.0, grain=1.0, metres=2.5,
                     mud=0.30, dent=0.15, dust_contrast=0.30),
    "dark":   _Style(rough=0.52, dust=0.06, grime=0.35, fade=0.14, chips=0.35, scratches=1.0, grain=0.6, metres=2.5,
                     mud=0.35, dent=0.30, dust_contrast=0.30),
}


def _compress_gamut(rgb) -> np.ndarray:
    """Real pigments never sit on the edge of sRGB: pull pure colours slightly inward."""
    c = np.clip(np.asarray(rgb, np.float32), 0.0, 1.0)
    lum = float(c @ np.array([0.2126, 0.7152, 0.0722], np.float32))
    c = c * 0.955 + lum * 0.045
    return np.maximum(c, 0.018)


def _lum(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _dust_field(rng, n, mean, streak_clean=0.7, lam=(450, 70), contrast=0.7):
    """Dust film coverage 0..1: soft drifts, powdery grain and (mild) rain-washed vertical streaks."""
    d_low = nz(rng, n, lam[0], lam[1], beta=3.0)
    d_mid = nz(rng, n, 90, 14, beta=2.5)
    d_fine = nz(rng, n, 12, 2.4, beta=1.0)
    cover = mean * np.clip(1.0 + contrast * (0.45 * d_low + 0.28 * d_mid), 0.12, 2.6)
    wash = nz(rng, n, 320, 12, beta=2.4, stretch=(1.0, 16.0))
    clean = _smooth(0.7, 2.2, wash) * streak_clean * 0.55
    wash2 = nz(rng, n, 160, 6, beta=2.0, stretch=(1.0, 30.0))
    clean = np.maximum(clean, _smooth(1.4, 2.6, wash2) * streak_clean * 0.5)
    cover = cover * (1.0 - clean)
    grit = _smooth(0.9, 2.4, nz(rng, n, 5, 1.5, beta=0.3))            # individual dust grains
    dust = np.clip(cover * (0.80 + 0.22 * d_fine + 0.45 * grit), 0.0, 0.9)
    return dust, d_fine, d_mid, clean


def _mud_splatter(rng, n, strength, clusters=1.0, blots=1.0, speck=1.0):
    """Kicked-up mud: fine speckle and mid-size droplets scattered uniformly, optionally clustered sprays
    (``clusters`` > 0) and caked blotches (``blots`` > 0).  Returns (alpha, height)."""
    s = n / 1024.0
    cxs, cys, rs = [], [], []
    for _ in range(int(9 * strength * clusters) + (2 if clusters > 0 else 0)):
        cx, cy = rng.uniform(0, n, 2)
        m = int(rng.uniform(25, 110))
        spread = rng.uniform(10, 55) * s
        ang = rng.uniform(0, math.pi)
        rad = rng.exponential(spread, m)
        th = rng.uniform(0, 2 * math.pi, m)
        ex, ey = rad * np.cos(th) * 1.7, rad * np.sin(th)
        cxs.append(cx + ex * math.cos(ang) - ey * math.sin(ang))
        cys.append(cy + ex * math.sin(ang) + ey * math.cos(ang))
        rs.append((0.45 + rng.exponential(1.0, m) * (1.15 - np.clip(rad / (4 * spread), 0, 0.9))) * s)
    # fine speckle everywhere
    m = int(2000 * strength * speck)
    cxs.append(rng.uniform(0, n, m)); cys.append(rng.uniform(0, n, m)); rs.append((0.4 + rng.exponential(0.55, m)) * s)
    # mid-size droplets, uniformly scattered (heavy-tailed sizes, ragged outline below)
    m = int(70 * strength * speck)
    cxs.append(rng.uniform(0, n, m)); cys.append(rng.uniform(0, n, m)); rs.append((0.8 + rng.exponential(1.0, m)) * s)
    cx, cy, r = np.concatenate(cxs), np.concatenate(cys), np.minimum(np.concatenate(rs), 7.0 * s)
    drop = discs(n, cx, cy, r, soft=1.0 * max(s, 0.5))
    if clusters <= 0:                    # uniform spatter: break the perfect roundness of the larger drops
        drop = drop * (0.55 + 0.45 * _smooth(-0.9, 0.3, nz(rng, n, 9, 1.8, beta=0.6)))
    a = drop
    h = drop * 0.5
    if blots > 0:
        blot = nz(rng, n, 140, 18, beta=2.2) + 0.8 * nz(rng, n, 500, 100, beta=3.0)
        blot = _smooth(3.4 - 0.25 * strength, 3.9 - 0.25 * strength, blot) * 0.7 * blots
        a = np.maximum(drop, blot)
        h = drop * 0.5 + blot
    return a.astype(np.float32), h.astype(np.float32)


def _paint_chips(rng, n, count, s):
    """Small ragged paint chips scattered uniformly (a Poisson field has no landmarks).  -> mask 0..1."""
    m = int(count)
    cx, cy = rng.uniform(0, n, m), rng.uniform(0, n, m)
    r = np.exp(rng.uniform(math.log(1.0), math.log(3.4), m)) * s
    disc = discs(n, cx, cy, r, soft=0.8 * max(s, 0.5))
    rag = _smooth(-0.45, 0.30, nz(rng, n, 16, 2.2, beta=0.8))
    return (disc * rag).astype(np.float32)


def _drip_trails(rng, n, count, length, s):
    """Many faint dirt runs: sources are uniformly scattered, lengths vary; strongest trail normalised to 1."""
    pts = np.zeros((n, n), np.float32)
    pts[rng.integers(0, n, count), rng.integers(0, n, count)] = rng.uniform(0.25, 1.0, count)
    t = streak_down(blur(pts, 1.1 * s), length * s)
    t = t / (np.percentile(t, 99.95) + 1e-9)
    return np.clip(blur(t, 0.7 * s) * (0.75 + 0.45 * nz(rng, n, 300, 30, beta=2, stretch=(1, 12))), 0, 1)


def _painted(rng, n, base_rgb, st: _Style, dark_base: bool = False):
    s = n / 1024.0
    ss = max(s, 0.5)
    base = _compress_gamut(base_rgb)
    base_lin = srgb_to_linear(base)
    dust_lin = _srgb_const(DUST_DARK_RGB if dark_base else DUST_RGB)
    mud_lin, dirt_lin, primer_lin, rust_lin = (_srgb_const(c) for c in (MUD_RGB, DIRT_RGB, PRIMER_RGB, RUST_RGB))

    # --- paint: gentle tone drift, warm/cool drift, chalking -----------------------------------
    tone = 1.0 + 0.030 * nz(rng, n, 480, 90, beta=3.0) + 0.016 * nz(rng, n, 120, 20, beta=2.5)
    warm = 0.03 * nz(rng, n, 420, 70, beta=3.0)
    paint = base_lin[None, None, :] * tone[..., None]
    paint[..., 0] *= 1.0 + 0.3 * warm
    paint[..., 1] *= 1.0 - 0.6 * warm
    paint[..., 2] *= 1.0 - 0.3 * warm
    fade_field = nz(rng, n, 480, 70, beta=3.0) + 0.5 * nz(rng, n, 120, 25, beta=2.5)
    fade = _smooth(-0.8, 2.0, fade_field) * st.fade
    chalk_lin = srgb_to_linear(np.clip(_mix(base, np.array([0.80, 0.74, 0.62], np.float32), 0.30), 0, 1))
    if dark_base:                       # black plastic / trim greys out in the sun (neutral, not brown)
        chalk_lin = srgb_to_linear(np.array([0.27, 0.265, 0.26], np.float32))
    alb = _mix(paint, chalk_lin[None, None, :], (fade * 0.55)[..., None])
    rough = np.full((n, n), st.rough, np.float32) + 0.16 * fade
    height = np.zeros((n, n), np.float32)

    # --- sheet-metal waviness, orange peel, moulding grain, dents --------------------------------------
    height += 1.6e-3 * nz(rng, n, 1400, 260, beta=3.5) + 0.22e-3 * nz(rng, n, 260, 60, beta=3.0)
    height += (0.45e-4 + 0.7e-4 * st.grain) * nz(rng, n, 16, 5.0, beta=1.5)
    if st.dent > 0:
        dents = _smooth(1.9, 3.0, nz(rng, n, 220, 50, beta=2.5) + 0.8 * nz(rng, n, 900, 200, beta=3.0))
        height -= 1.1e-3 * st.dent * blur(dents, 8 * s)
    if st.grain > 0:                    # moulded plastic: fine stipple only (cells any bigger read as leather / stone)
        height += 0.6e-4 * st.grain * nz(rng, n, 7, 1.6, beta=0.5)
        rough += 0.04 * st.grain * nz(rng, n, 12, 3, beta=1.0)

    # --- oil and hydraulic grime: soft, dark, glossy, sticky -------------------------------------------------
    oil_f = nz(rng, n, 300, 18, beta=2.4, stretch=(1.0, 2.2)) + 0.25 * nz(rng, n, 1400, 400, beta=3.5)
    oil = _smooth(1.25, 2.5, oil_f) * min(st.grime, 1.4)
    run = blur(np.clip(streak_down(oil * 0.10, 60 * s), 0, 1), 2.0 * s, 1.0 * s)
    oil = np.maximum(oil, run * 0.35) * (0.75 + 0.25 * nz(rng, n, 30, 5, beta=1.5))
    alb = alb * (1.0 - 0.17 * oil)[..., None]
    alb = _mix(alb, dirt_lin[None, None, :], (0.10 * oil)[..., None])
    rough = _mix(rough, np.float32(0.24), oil * 0.7)

    # --- chips: many small ragged chips down to grey-brown primer, a few bare-steel flecks, faint rust weep ----
    chip = _paint_chips(rng, n, 75 * st.chips, s)
    steel_fleck = _smooth(0.9, 1.8, nz(rng, n, 10, 2.5, beta=0.8)) * chip
    primer_col = primer_lin[None, None, :] * (0.8 + 0.4 * nz(rng, n, 14, 3, beta=1.0))[..., None]
    chip_rgb = _mix(primer_col, _srgb_const((0.46, 0.44, 0.42))[None, None, :], (steel_fleck * 0.7)[..., None])
    halo = np.clip(blur(chip, 2.0 * ss, 2.0 * ss) * 2.2, 0, 1) * (1 - chip)
    bleed = np.clip(streak_down(blur(chip, 1.0 * s) * 0.5, 70 * s) * 0.2, 0, 1) * (1 - chip)
    rust_amt = np.clip((halo * 0.45 + bleed) * st.rust_bleed, 0, 0.5)
    alb = _mix(alb, rust_lin[None, None, :] * (0.7 + 0.5 * nz(rng, n, 30, 5, beta=1.5))[..., None], rust_amt[..., None])
    alb = _mix(alb, chip_rgb, (chip * 0.92)[..., None])
    rough = _mix(rough, np.float32(0.62), chip)
    rough = _mix(rough, np.float32(0.75), rust_amt)
    height -= 2.2e-4 * blur(chip, 0.8 * s)

    # --- scratches: light abrasion lines + a few deeper ones (brown-grey primer, not black) --------------------
    master = rng.uniform(0, math.pi)
    n_light = int(120 * st.scratches)
    lens = np.clip(rng.lognormal(math.log(70 * s), 0.8, n_light), 12 * s, 500 * s)
    light = polylines(n, rng, n_light, lens, width_px=1.0 * ss, angle=master, angle_jitter=0.5, intensity=(0.2, 0.6))
    light += polylines(n, rng, int(60 * st.scratches), np.clip(rng.lognormal(math.log(40 * s), 0.6, int(60 * st.scratches)), 8, 300 * s),
                       width_px=1.0 * ss, intensity=(0.15, 0.55))
    n_deep = int(9 * st.scratches)
    deep = polylines(n, rng, n_deep, np.clip(rng.lognormal(math.log(55 * s), 0.7, n_deep), 15 * s, 300 * s),
                     width_px=1.1 * ss, angle=master, angle_jitter=0.6, intensity=(0.4, 0.9))
    light = np.clip(light, 0, 1)
    deep = np.clip(deep, 0, 1)
    light_rgb = _mix(alb, _srgb_const((0.55, 0.52, 0.46))[None, None, :] * (0.4 if dark_base else 1.0), 0.45)
    lum_k = 0.25 + 0.75 * np.clip(_lum(alb) / 0.4, 0.0, 1.0)         # abrasion shows less on deep saturated / dark paint
    alb = _mix(alb, light_rgb, (light * 0.65 * lum_k)[..., None])
    alb = _mix(alb, primer_lin[None, None, :] * 1.15, (deep * 0.55)[..., None])
    height -= 0.8e-4 * light + 1.6e-4 * deep
    rough += 0.10 * light

    # --- dust film (drifts + grain), thicker in oil, washed off in vertical rain lines -----------------------
    D, d_fine, d_mid, clean = _dust_field(rng, n, st.dust, contrast=st.dust_contrast, streak_clean=0.95)
    D = np.clip(D * (1.0 + 0.5 * oil), 0, 0.92)
    dust_col = dust_lin[None, None, :] * (1.0 + 0.10 * d_mid + 0.06 * d_fine)[..., None]
    veil = np.clip(D * (1.25 if dark_base else 1.1), 0.0, 0.85)
    alb = _mix(alb, dust_col, veil[..., None])
    rough = _mix(rough, np.float32(0.82), np.clip(D * 1.35, 0, 1))
    height += 2.0e-4 * D * nz(rng, n, 10, 3.0, beta=0.8) + 3.5e-4 * D * nz(rng, n, 60, 14, beta=2.0)

    # --- drip lines: many faint brown-grey runs with varied lengths, not a few dark trails ---------------------
    k = int(46 * st.grime) + 6
    trails = np.maximum(_drip_trails(rng, n, k, 170, s), _drip_trails(rng, n, k, 60, s) * 0.8)
    alb = alb * (1.0 - 0.10 * st.grime * trails)[..., None]
    alb = _mix(alb, dirt_lin[None, None, :] * 1.1, (0.07 * st.grime * trails)[..., None])
    rough = _mix(rough, np.float32(0.45), trails * 0.25)

    # --- mud spatter: fine uniformly scattered droplets (no clusters = no repeating landmarks) -------------------
    if st.mud > 0:
        a, hmud = _mud_splatter(rng, n, st.mud, clusters=0.0, blots=0.0)
        mud_col = mud_lin[None, None, :] * (0.7 + 0.6 * nz(rng, n, 25, 4, beta=1.5))[..., None]
        alb = _mix(alb, mud_col, (a * 0.62)[..., None])
        rough = _mix(rough, np.float32(0.9), a)
        height += 6e-4 * blur(hmud, 0.7 * s)

    # --- final micro variation of the gloss (hand-wiped, uneven film) ------------------------------------------
    rough += 0.035 * nz(rng, n, 200, 20, beta=2.5) + 0.02 * nz(rng, n, 14, 3, beta=0.8)
    # slight global sparkle in dusty paint grain
    alb = alb * (1.0 + 0.035 * nz(rng, n, 8, 2.0, beta=0.5))[..., None]
    height *= st.metres / 2.0           # heights were tuned for a 2 m tile: keep the same surface slopes
    return alb, np.clip(rough, 0.12, 0.92), height


def _steel(rng, n, base_rgb):
    s = n / 1024.0
    base = np.clip(np.asarray(base_rgb, np.float32), 0, 1)
    # bare steel F0 (linear) sits well above the dark URDF grey; keep the tint of the spec colour
    f0 = srgb_to_linear(np.clip(_mix(base, np.array([0.80, 0.80, 0.81], np.float32), 0.35), 0, 1)) * 0.80
    dust_lin, rust_lin = _srgb_const(DUST_RGB), _srgb_const((0.38, 0.19, 0.08))

    # brushed / rolled grain: long streaks along x of several widths
    brush = 0.55 * nz(rng, n, 300, 2.2, beta=0.4, stretch=(60.0, 1.0)) + 0.45 * nz(rng, n, 260, 4.0, beta=0.8, stretch=(18.0, 1.0))
    brush2 = nz(rng, n, 500, 2.5, beta=0.3, stretch=(120.0, 1.0))
    tone = 1.0 + 0.10 * brush + 0.04 * brush2 + 0.035 * nz(rng, n, 480, 80, beta=3.0) + 0.03 * nz(rng, n, 100, 15, beta=2.5)
    alb = f0[None, None, :] * tone[..., None]
    # heat-tint / polish patches: warm-bluish wander and brighter polished wear
    wander = nz(rng, n, 700, 90, beta=3.0)
    alb[..., 0] *= 1.0 + 0.025 * wander
    alb[..., 2] *= 1.0 - 0.03 * wander
    polish = _smooth(0.6, 1.8, nz(rng, n, 300, 50, beta=2.8))
    alb = alb * (1.0 + 0.12 * polish)[..., None]
    rough = 0.38 - 0.10 * polish + 0.07 * brush + 0.03 * brush2 + 0.03 * nz(rng, n, 250, 30, beta=2.5)
    height = 3.5e-5 * brush + 2.5e-5 * brush2 + 4e-4 * nz(rng, n, 900, 120, beta=3.2) + 0.5e-4 * nz(rng, n, 20, 5, beta=1.5)

    # scratches: mostly along the brushing direction, plus cross scratches and gouges
    sc = polylines(n, rng, 260, np.clip(rng.lognormal(math.log(90 * s), 0.8, 260), 15 * s, 700 * s), width_px=0.9 * max(s, 0.5),
                   angle=0.0, angle_jitter=0.12, curve=0.06, intensity=(0.2, 0.8))
    sc += polylines(n, rng, 70, np.clip(rng.lognormal(math.log(55 * s), 0.7, 70), 10 * s, 400 * s), width_px=0.9 * max(s, 0.5),
                    intensity=(0.2, 0.9), curve=0.2)
    sc = np.clip(sc, 0, 1)
    gouge = polylines(n, rng, 9, np.clip(rng.lognormal(math.log(40 * s), 0.6, 9), 8 * s, 200 * s), width_px=2.0 * max(s, 0.5),
                      intensity=(0.5, 0.9), curve=0.3)
    gouge = np.clip(gouge, 0, 1)
    alb = alb * (1.0 + 0.45 * sc - 0.25 * gouge)[..., None]
    rough = rough - 0.06 * sc + 0.08 * gouge
    height -= 0.9e-4 * sc + 3.5e-4 * gouge

    # oil / hydraulic fluid: dark, wet-looking, collects dust
    oil = _smooth(1.7, 2.7, nz(rng, n, 260, 25, beta=2.6) + 0.4 * nz(rng, n, 1400, 400, beta=3.5))
    alb = alb * (1.0 - 0.35 * oil)[..., None]
    rough = _mix(rough, np.float32(0.28), oil * 0.7)

    # rust blooms and pitting
    rust_f = nz(rng, n, 420, 40, beta=2.8) + 0.6 * nz(rng, n, 1500, 400, beta=3.5)
    rust_cover = _smooth(1.6, 3.0, rust_f)
    pit = nz(rng, n, 24, 3, beta=1.0) + 0.5 * nz(rng, n, 80, 12, beta=1.8)
    rust = np.clip(rust_cover * _smooth(-0.4, 1.0, pit + 0.3), 0, 1) * 0.9
    rust_col = rust_lin[None, None, :] * (0.7 + 0.5 * nz(rng, n, 30, 4, beta=1.5))[..., None]
    alb = _mix(alb, rust_col, (rust * 0.9)[..., None])
    rough = _mix(rough, np.float32(0.80), rust)
    height -= 2.5e-4 * rust * (0.5 + 0.5 * pit)

    # dust and a bit of dried mud
    D, d_fine, d_mid, _ = _dust_field(rng, n, 0.20, contrast=0.35)
    D = np.clip(D * (1.0 + 0.7 * oil), 0, 0.85)
    alb = _mix(alb, dust_lin[None, None, :] * 0.85 * (1 + 0.1 * d_mid)[..., None], (np.clip(D * 1.1, 0, 0.85))[..., None])
    rough = _mix(rough, np.float32(0.70), np.clip(D * 1.2, 0, 1))
    height += 1.6e-4 * D * nz(rng, n, 10, 3.0, beta=0.8)
    a, hmud = _mud_splatter(rng, n, 0.5, clusters=0.3, blots=0.3)
    alb = _mix(alb, _srgb_const(MUD_RGB)[None, None, :] * (0.7 + 0.6 * nz(rng, n, 25, 4, beta=1.5))[..., None], (a * 0.85)[..., None])
    rough = _mix(rough, np.float32(0.9), a)
    height += 4e-4 * blur(hmud, 0.7 * s)
    return alb, np.clip(rough, 0.18, 0.92), height


def _rubber(rng, n, base_rgb):
    s = n / 1024.0
    base = _srgb_const(np.clip(np.maximum(np.asarray(base_rgb, np.float32), 0.06), 0, 1))
    dust_lin, mud_lin = _srgb_const(DUST_DARK_RGB), _srgb_const(MUD_RGB)

    # near-black, matte, fine grain; the large-scale tone terms are kept tiny (blotches read as stone / clay)
    fine = nz(rng, n, 6, 1.5, beta=0.2)
    tone = 1.0 + 0.16 * fine + 0.05 * nz(rng, n, 28, 6, beta=1.5) + 0.035 * nz(rng, n, 400, 70, beta=3.0)
    alb = base[None, None, :] * np.maximum(tone, 0.2)[..., None] * 1.10
    rough = 0.86 + 0.035 * nz(rng, n, 200, 20, beta=2.5) + 0.03 * fine
    height = 1.2e-4 * nz(rng, n, 12, 2.0, beta=0.6) + 2.2e-4 * nz(rng, n, 60, 12, beta=2.0) + 5e-4 * nz(rng, n, 800, 150, beta=3.2)

    # weathering crazing: a fine crack network, dust packed into it
    f1, f2 = voronoi_f12(n, 46, rng, 0.9)
    edge = f2 - f1
    present = _smooth(-0.2, 0.9, nz(rng, n, 200, 30, beta=2.0) + 0.3 * nz(rng, n, 24, 6, beta=1.0))
    cracks = np.exp(-(edge / 0.06) ** 2) * present
    height -= 3.2e-4 * cracks
    alb = alb * (1.0 - 0.30 * cracks)[..., None]
    # moulding stipple
    cellf = voronoi_f12(n, 150, rng, 1.0)[0]
    height += 0.8e-4 * (0.5 - cellf)

    # scuffs: pale abrasion smears along the rolling direction
    scuff = _smooth(1.1, 2.4, nz(rng, n, 500, 5, beta=2.0, stretch=(16.0, 1.0)) + 0.5 * nz(rng, n, 800, 150, beta=3.0))
    alb = _mix(alb, _srgb_const((0.20, 0.195, 0.19))[None, None, :], (scuff * 0.40)[..., None])
    rough = _mix(rough, np.float32(0.66), scuff * 0.6)

    # dust film: pale neutral grey powder, packed into the cracks
    D, d_fine, d_mid, clean = _dust_field(rng, n, 0.09, streak_clean=0.3, contrast=0.35)
    D = np.clip(D * (1.0 + 0.7 * cracks), 0, 0.9)
    alb = _mix(alb, dust_lin[None, None, :] * (0.78 + 0.14 * d_mid + 0.08 * d_fine)[..., None], (np.clip(D * 1.25, 0, 0.9))[..., None])
    rough = _mix(rough, np.float32(0.97), np.clip(D * 1.4, 0, 1))
    height += 3.0e-4 * D * nz(rng, n, 14, 3.0, beta=1.0)

    # caked mud: mostly droplets and a few small sprays (no big clouds)
    a, hmud = _mud_splatter(rng, n, 1.0, clusters=0.4, blots=0.4)
    mud_col = mud_lin[None, None, :] * (0.75 + 0.5 * nz(rng, n, 25, 4, beta=1.5))[..., None]
    alb = _mix(alb, mud_col, (a * 0.75)[..., None])
    rough = _mix(rough, np.float32(0.93), a)
    height += 8e-4 * blur(hmud, 0.8 * s)
    return alb, np.clip(rough, 0.3, 0.98), height


def _glass(rng, n, base_rgb):
    s = n / 1024.0
    base = np.clip(np.asarray(base_rgb, np.float32), 0, 1)
    tint = srgb_to_linear(base) * np.array([0.24, 0.30, 0.31], np.float32) * 0.85
    dust_lin = _srgb_const(DUST_RGB)
    alb = tint[None, None, :] * (1.0 + 0.05 * nz(rng, n, 480, 80, beta=3.0) + 0.04 * nz(rng, n, 80, 15, beta=2.0))[..., None]
    rough = np.full((n, n), 0.045, np.float32) + 0.01 * nz(rng, n, 300, 30, beta=2.5)
    height = 3e-5 * nz(rng, n, 1200, 200, beta=3.5)

    # fine scratches in the glass
    sc = np.clip(polylines(n, rng, 70, np.clip(rng.lognormal(math.log(80 * s), 0.7, 70), 15 * s, 500 * s),
                           width_px=0.8 * max(s, 0.5), intensity=(0.1, 0.5), curve=0.1), 0, 1)
    rough += 0.18 * sc
    alb = alb * (1 + 0.8 * sc)[..., None]

    # dust haze, cleaned by wiper-ish vertical washing, with film-like grain
    D, d_fine, d_mid, clean = _dust_field(rng, n, 0.10, streak_clean=0.85, contrast=0.45)
    D = np.clip(D, 0, 0.8)
    haze = np.clip(D * 1.2, 0, 0.8)
    alb = _mix(alb, dust_lin[None, None, :] * 0.55, (haze * 0.8)[..., None])
    rough = _mix(rough, np.float32(0.42), np.clip(D * 1.5, 0, 1))
    height += 1.2e-4 * D * nz(rng, n, 10, 3, beta=0.8)

    # water spots: domes with a mineral ring, and dust-free trails behind some of them
    m = 1600                              # ~ 400 spots per square metre of the 2 m tile
    cx, cy = rng.uniform(0, n, m), rng.uniform(0, n, m)
    r = (1.4 + rng.exponential(2.0, m)) * s
    r = np.minimum(r, 12 * s)
    inner = discs(n, cx, cy, r * 0.82, soft=1.0 * max(s, 0.5))
    full = discs(n, cx, cy, r, soft=1.0 * max(s, 0.5))
    ring = np.clip(full - inner, 0, 1)
    dome = blur(full, 1.0 * s) * 1.0
    ring = blur(ring, 0.8 * s)
    alb = _mix(alb, dust_lin[None, None, :] * 0.8, (ring * 0.22)[..., None])
    alb = alb * (1.0 + 0.15 * inner)[..., None]
    rough = _mix(rough, np.float32(0.03), inner * 0.8)
    height += 2.2e-4 * dome + 0.6e-4 * ring
    trail = np.clip(streak_down(blur(discs(n, cx[:230], cy[:230], r[:230] * 0.5), 0.8 * s), 120 * s) * 0.4, 0, 1)
    rough = _mix(rough, np.float32(0.05), trail * 0.5)
    alb = _mix(alb, tint[None, None, :], (trail * 0.5)[..., None])
    return alb, np.clip(rough, 0.02, 0.9), height


def _rust(rng, n, base_rgb):
    base = _srgb_const(np.clip(base_rgb, 0.02, 1.0))
    deep = _srgb_const((0.20, 0.09, 0.05))
    orange = _srgb_const((0.50, 0.24, 0.085))
    ochre = _srgb_const((0.62, 0.40, 0.20))
    scale = _srgb_const((0.13, 0.115, 0.11))
    steel = _srgb_const((0.42, 0.40, 0.38))

    macro = nz(rng, n, 420, 70, beta=3.0)
    meso = nz(rng, n, 120, 16, beta=2.4)
    fine = nz(rng, n, 16, 2.5, beta=1.2)
    mix_f = _smooth(-1.2, 1.4, 0.7 * macro + 0.6 * meso)
    alb = _mix(deep[None, None, :], orange[None, None, :], mix_f[..., None])
    ochre_amt = _smooth(0.3, 1.6, meso + 0.6 * nz(rng, n, 300, 40, beta=2.5))
    alb = _mix(alb, ochre[None, None, :], (ochre_amt * 0.55)[..., None])
    alb = _mix(alb, base[None, None, :] * 1.1, 0.15)
    # dark scale and exposed steel under flaking
    f1, f2 = voronoi_f12(n, 36, rng, 1.0)
    flake = _smooth(1.1, 2.0, nz(rng, n, 260, 30, beta=2.4) + 0.4 * nz(rng, n, 30, 5, beta=1.0))
    alb = _mix(alb, scale[None, None, :], (flake * 0.8)[..., None])
    bare = _smooth(1.7, 2.6, nz(rng, n, 120, 12, beta=1.8) + 0.3 * macro)
    alb = _mix(alb, steel[None, None, :], (bare * 0.85)[..., None])
    alb = alb * (1.0 + 0.30 * fine + 0.12 * meso)[..., None]

    pits = voronoi_f12(n, 90, rng, 1.0)[0]
    height = 2.5e-3 * nz(rng, n, 500, 50, beta=3.0) + 1.0e-3 * meso + 3.5e-4 * fine
    height -= 1.8e-3 * np.exp(-(pits / 0.35) ** 2) * (0.4 + 0.6 * _smooth(-0.5, 1.0, meso))
    height += 1.8e-3 * (flake - bare)
    rough = 0.86 + 0.06 * meso + 0.03 * fine - 0.22 * bare - 0.06 * flake
    D, d_fine, d_mid, _ = _dust_field(rng, n, 0.12)
    alb = _mix(alb, _srgb_const(DUST_RGB)[None, None, :] * 0.7, (np.clip(D * 1.0, 0, 0.6))[..., None])
    rough = _mix(rough, np.float32(0.95), np.clip(D * 1.2, 0, 1))
    return alb, np.clip(rough, 0.35, 0.98), height


_GLASS_OPACITY = 0.42


def machine_textures(material_name: str, base_rgb: tuple[float, float, float], seed: int = 0,
                     size: int = 1024) -> TextureSet:
    """Seamless albedo / roughness / normal for one machine material.

    ``material_name`` is a ``spec.DEFAULT_MATERIALS`` key; unknown names are treated as
    painted steel derived from ``base_rgb`` (0..1, or 0..255).  Row direction is "down"
    for rain streaks; see the module docstring.
    """
    n = int(size)
    rgb = np.asarray(base_rgb, np.float32)[:3]
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    rng = np.random.default_rng([int(seed) & 0x7FFFFFFF, zlib.crc32(material_name.encode())])
    key = material_name.lower()

    metallic, opacity, clearcoat = 0.0, 1.0, 0.0
    if key == "steel":
        alb, rough, height = _steel(rng, n, rgb)
        metallic, metres = 1.0, 2.5
    elif key == "rubber":
        alb, rough, height = _rubber(rng, n, rgb)
        metres = 1.5
    elif key == "glass":
        alb, rough, height = _glass(rng, n, rgb)
        opacity, metres = _GLASS_OPACITY, 2.0
    elif key == "rust":
        alb, rough, height = _rust(rng, n, rgb)
        metres = 4.0
    else:
        st = _STYLES.get(key)
        dark = key in ("accent", "dark")
        if st is None:                                  # unknown: painted, derived from the base colour
            lum = float(_lum(rgb))
            dark = lum < 0.10
            st = _Style(rough=0.45 if dark else 0.36, dust=0.09 if dark else 0.16, fade=0.15 if dark else 0.12,
                        grain=0.5 if dark else 0.0, metres=2.5 if dark else 3.0, dust_contrast=0.3 if dark else 0.7,
                        dent=0.3 if dark else 1.0)
        alb, rough, height = _painted(rng, n, rgb, st, dark_base=dark)
        metres = st.metres

    mpp = metres / n
    return TextureSet(
        albedo=_to_u8_srgb(alb, rng),
        roughness=_rough_u8(rough),
        normal=height_to_normal(height, mpp, gain=1.0),
        metallic=metallic, metres_per_tile=float(metres), opacity=opacity, clearcoat=clearcoat)


# ---------------------------------------------------------------------------
# Terrain micro relief
# ---------------------------------------------------------------------------

class _Tile:
    """A periodic scalar tile of ``period`` metres with a box-filter mip chain, sampled in world space."""

    def __init__(self, arr: np.ndarray, period: float):
        self.period = float(period)
        self.levels = [arr.astype(np.float32)]
        while self.levels[-1].shape[0] > 16:
            a = self.levels[-1]
            self.levels.append(0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]))

    def _bilinear(self, lvl: int, u, v):
        a = self.levels[lvl].ravel()
        N = self.levels[lvl].shape[0]
        u = u * N
        v = v * N
        u0 = np.floor(u)
        v0 = np.floor(v)
        fu = u - u0
        fv = v - v0
        iu0 = u0.astype(np.int32) % N
        iv0 = (v0.astype(np.int32) % N) * N
        iu1 = (iu0 + 1) % N
        iv1 = (iv0 + N) % (N * N)
        top = a.take(iv0 + iu0) * (1 - fu) + a.take(iv0 + iu1) * fu
        bot = a.take(iv1 + iu0) * (1 - fu) + a.take(iv1 + iu1) * fu
        return (top * (1 - fv) + bot * fv).astype(np.float32)

    def sample(self, X, Y, px_per_m: float, angle=0.0, offset=(0.0, 0.0), swap: bool = False, scale: float = 1.0):
        """Bilinear lookup at world coordinates (any array shape); the mip level is chosen from the pixel size.
        ``angle`` may be a scalar or a per-pixel array (radians); ``scale`` > 1 zooms the pattern in."""
        if np.ndim(angle) == 0:
            ca, sa = math.cos(angle), math.sin(angle)
        else:
            ca, sa = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
        xr = X * ca - Y * sa + offset[0]
        yr = X * sa + Y * ca + offset[1]
        if swap:
            xr, yr = yr, xr
        per = self.period * scale
        u = (xr / per) % 1.0
        v = (yr / per) % 1.0
        foot = (self.levels[0].shape[0] / per) / px_per_m          # tile pixels per output pixel
        lvl = int(np.clip(round(math.log2(max(foot, 1e-6)) + 0.25), 0, len(self.levels) - 1))
        return self._bilinear(lvl, u.astype(np.float32), v.astype(np.float32))


def _stamp_domes(n, rng, rad_m, cover, hrel, mpp, shape=0.9, aspect=(0.6, 1.0)):
    """Randomly placed pebbles (domes, log-uniform radii ``rad_m`` = (min, max) metres, random orientation /
    aspect), wrap-around, max-composited.  ``cover`` is the summed disc area / tile area (can exceed 1);
    ``hrel`` is height / radius.  Returns a height field in metres (``mpp`` = metres per pixel)."""
    out = np.zeros(n * n, np.float32)
    rmin, rmax = rad_m[0] / mpp, rad_m[1] / mpp
    r_mean = (rmax - rmin) / math.log(rmax / rmin) if rmax > rmin * 1.01 else rmin
    count = int(cover * n * n / (math.pi * r_mean * r_mean * 0.75))
    cx = rng.uniform(0, n, count)
    cy = rng.uniform(0, n, count)
    rr = np.exp(rng.uniform(math.log(rmin), math.log(rmax), count))
    asp = rng.uniform(*aspect, count)
    th = rng.uniform(0, math.pi, count)
    hs = hrel * rng.uniform(0.55, 1.0, count)
    Rw = np.ceil(rr).astype(int) + 1
    for R in np.unique(Rw):
        idx = np.nonzero(Rw == R)[0]
        side = 2 * R + 1
        chunk = max(1, int(1.5e6 // (side * side)))
        o = np.arange(-R, R + 1)
        for i0 in range(0, len(idx), chunk):
            k = idx[i0:i0 + chunk]
            x0 = np.floor(cx[k]).astype(int)
            y0 = np.floor(cy[k]).astype(int)
            px = x0[:, None, None] + o[None, None, :]
            py = y0[:, None, None] + o[None, :, None]
            dx = px - cx[k][:, None, None]
            dy = py - cy[k][:, None, None]
            c, s_ = np.cos(th[k])[:, None, None], np.sin(th[k])[:, None, None]
            u = (dx * c + dy * s_) / rr[k][:, None, None]
            v = (-dx * s_ + dy * c) / (rr[k] * asp[k])[:, None, None]
            d2 = u * u + v * v
            h = np.where(d2 < 1.0, np.power(np.clip(1.0 - d2, 0, 1), shape), 0.0) * (rr[k] * mpp * hs[k])[:, None, None]
            np.maximum.at(out, ((py % n) * n + (px % n)).ravel(), h.ravel().astype(np.float32))
    return out.reshape(n, n)


@lru_cache(maxsize=16)
def _terrain_tile(kind: str, seed: int) -> _Tile:
    rng = np.random.default_rng([seed & 0x7FFFFFFF, zlib.crc32(kind.encode())])
    N = 1024
    if kind == "gravel":              # 3.2 m tile: stacked pebble populations, 4 mm .. 16 cm across
        P = 3.2
        mpp = P / N
        h = _stamp_domes(N, rng, (0.004, 0.012), 1.6, 0.40, mpp)
        h = np.maximum(h, _stamp_domes(N, rng, (0.010, 0.025), 1.3, 0.40, mpp))
        h = np.maximum(h, _stamp_domes(N, rng, (0.020, 0.045), 1.0, 0.38, mpp))
        h = np.maximum(h, _stamp_domes(N, rng, (0.035, 0.080), 0.7, 0.36, mpp))
        return _Tile(h, P)
    if kind == "cobble":              # 11 m tile: loose fist-sized stones, half buried
        P = 11.0
        h = _stamp_domes(N, rng, (0.05, 0.13), 0.20, 0.30, P / N, shape=0.8, aspect=(0.55, 1.0))
        return _Tile(h, P)
    if kind == "clod":                # 5 m tile: lumpy, crusted ground (metres)
        P = 5.0
        h = 0.5 * nz(rng, N, 320, 55, beta=2.8) + 0.45 * nz(rng, N, 90, 18, beta=2.4) + 0.25 * nz(rng, N, 26, 6, beta=1.8)
        # break the clods up with crumble: sharpen peaks
        h = np.sign(h) * np.abs(h) ** 1.15
        return _Tile(h * 0.013, P)
    if kind == "grain":               # 1.6 m tile: sand / dust grain and wind ripples (metres)
        P = 1.6
        h = 0.55 * nz(rng, N, 20, 3, beta=0.6) + 0.5 * nz(rng, N, 70, 8, beta=1.6) + 0.5 * nz(rng, N, 220, 28, beta=2.4, stretch=(2.5, 1.0))
        return _Tile(h * 0.0012, P)
    if kind == "cracks":              # 6.5 m tile: polygonal desiccation cracks (0..1, 1 = crack centre)
        P = 6.5
        out = np.zeros((N, N), np.float32)
        for cells, w, amp in ((14, 0.032, 1.0), (7, 0.048, 0.9), (30, 0.018, 0.6)):
            f1, f2 = voronoi_f12(N, cells, rng, 0.95)
            cell_m = P / cells
            edge = (f2 - f1) * cell_m * 0.5                           # ~distance to the cell wall in metres
            present = _smooth(-0.5, 0.7, nz(rng, N, 500, 60, beta=2.6))
            wmod = w * (0.6 + 0.8 * _smooth(-1, 1.5, nz(rng, N, 120, 20, beta=2.0)))
            out = np.maximum(out, np.exp(-(edge / wmod) ** 2) * present * amp)
        return _Tile(out, P)
    if kind == "rills":               # 9 m tile, rills run along tile y
        P = 9.0
        meander = 40.0 * nz(rng, N, 800, 150, beta=3.0)                  # columns of shift, varies along y
        base = nz(rng, N, 260, 34, beta=2.2, stretch=(1.0, 10.0))
        base2 = nz(rng, N, 120, 16, beta=1.8, stretch=(1.0, 8.0))
        ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        uu = (jj + meander) % N
        u0 = np.floor(uu).astype(int)
        fu = (uu - u0).astype(np.float32)

        def warp(a):
            return a[ii, u0 % N] * (1 - fu) + a[ii, (u0 + 1) % N] * fu

        b = warp(base) + 0.45 * warp(base2)
        chan = _smooth(0.0, 0.9, np.abs(b))                              # 0 at the channel centre, 1 on the interfluve
        depth = 0.35 + 0.65 * _smooth(-0.9, 0.9, nz(rng, N, 500, 60, beta=2.4, stretch=(1.5, 7.0)))   # rills fade in / out
        return _Tile(((chan - 0.6) * 0.026 * depth).astype(np.float32), P)
    if kind == "facets":              # 7 m tile: angular fractured blocks with ledges (blast-cut / dug rock-earth faces)
        P = 7.0
        h = np.zeros((N, N), np.float32)
        for cells, step_m, tilt in ((10, 0.014, 0.09), (22, 0.012, 0.17), (58, 0.006, 0.30)):
            f1, f2, dx, dy, cid = voronoi_cells(N, cells, rng, 1.0)
            cell_m = P / cells
            m = cells * cells
            tx, ty = rng.normal(0, tilt, m).astype(np.float32), rng.normal(0, tilt, m).astype(np.float32)
            off = rng.normal(0, 1.0, m).astype(np.float32)
            hh = (tx[cid] * dx + ty[cid] * dy) * cell_m + off[cid] * step_m           # tilted planar facet + step
            hh -= 0.20 * step_m * np.exp(-((f2 - f1) / 0.14) ** 2)                     # soft chamfer between blocks
            h += hh
        h += 0.012 * np.tanh(1.6 * nz(rng, N, 300, 14, beta=2.0, stretch=(14.0, 1.0)))  # bedding ledges along tile x
        return _Tile(blur(h, 2.0), P)
    if kind == "field":               # 41 m tile of slow world-space variation (0 mean, unit std)
        P = 41.0
        return _Tile(0.65 * nz(rng, N, 520, 80, beta=3.0) + 0.45 * nz(rng, N, 130, 25, beta=2.4), P)
    raise KeyError(kind)


def _coarse_fields(terrain, rect, ppm, W, H):
    """Slope / lane mask / height on a coarse grid, bilinearly upsampled to the pixel grid."""
    x0, x1, y0, y1 = rect
    step = max(0.2, 3.0 / ppm)
    xc = np.arange(x0 - step, x1 + 2 * step, step)
    yc = np.arange(y0 - step, y1 + 2 * step, step)
    Xc, Yc = np.meshgrid(xc, yc, indexing="xy")
    z = np.asarray(terrain.height(Xc, Yc), np.float64)
    lane = np.asarray(terrain.lane_mask(Xc, Yc), np.float64)
    gy, gx = np.gradient(z, step)
    slope = np.clip(np.hypot(gx, gy) / 1.2, 0.0, 1.0)
    lap = (np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4 * z) / (step * step)
    # blast-cut bench face: steep AND part of the loading bench (dozed berms and windrows are loose earth, not rock)
    if hasattr(terrain, "bench") and getattr(terrain, "bench_height", 0.0) > 0:
        bfrac = np.asarray(terrain.bench(Xc, Yc), np.float64) / float(terrain.bench_height)
        rock = _smooth(0.20, 0.55, slope) * _smooth(0.05, 0.25, bfrac)
    else:
        rock = np.zeros_like(slope)
    xs = x0 + (np.arange(W) + 0.5) / ppm
    ys = y1 - (np.arange(H) + 0.5) / ppm

    def up(c):
        fx = np.clip((xs - xc[0]) / step, 0, len(xc) - 1.001)
        fy = np.clip((ys - yc[0]) / step, 0, len(yc) - 1.001)
        i0, j0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
        tx, ty = (fx - i0).astype(np.float32), (fy - j0).astype(np.float32)
        rows = c[:, i0] * (1 - tx) + c[:, i0 + 1] * tx                         # (ny_c, W)
        return (rows[j0] * (1 - ty)[:, None] + rows[j0 + 1] * ty[:, None]).astype(np.float32)

    return dict(lane=up(lane), slope=up(slope), gx=up(gx), gy=up(gy), lap=up(lap), z=up(z), rock=up(rock), xs=xs, ys=ys)


def _height_band(terrain, seed, ppm, X, Y, lane, slope, gxs, gys, rock=None):
    """Micro-relief height (metres) for one band of pixels with world coordinates X, Y (float32)."""
    face = np.clip(slope * 1.3, 0.0, 1.0)
    off = (1.0 - lane) * (1.0 - face)

    T = lambda k: _terrain_tile(k, seed)          # noqa: E731
    field_a = T("field").sample(X, Y, ppm, angle=0.3)
    field_b = T("field").sample(X, Y, ppm, angle=1.9, offset=(17.3, 5.1))

    # --- gravel / cobbles: patchy cover, denser off the lane and on the faces -------------------------
    cover = _smooth(-0.9, 0.7, field_a + 0.4 * field_b)
    g1 = T("gravel").sample(X, Y, ppm, angle=0.0)
    g2 = T("gravel").sample(X, Y, ppm, angle=1.1, offset=(1.37, 0.71))        # second copy decorrelates the repeat
    gravel = np.maximum(g1, g2 * 0.8)
    cob = T("cobble").sample(X, Y, ppm, angle=0.6)
    gamp = (1.0 * off + 0.40 * lane + 1.25 * face) * (0.5 + 0.5 * cover)
    h = gravel * gamp + cob * (0.8 * off + 0.15 * lane + 1.0 * face) * (0.2 + 0.8 * cover)

    # --- clods / crust (stronger on loose ground and faces, nearly absent on the compacted lane) -----------
    clod = T("clod").sample(X, Y, ppm, angle=0.0) + 0.6 * T("clod").sample(X, Y, ppm, angle=2.2, offset=(2.2, 1.3))
    h += clod * (0.9 * off + 0.30 * lane + 1.5 * face)
    # --- sand grain and wind ripples -----------------------------------------------------------------------------
    h += T("grain").sample(X, Y, ppm, angle=0.4) * (1.0 * off + 0.5 * lane + 0.4 * face)

    # --- dry cracks on the compacted lane ------------------------------------------------------------------------
    crack_cover = _smooth(-0.3, 0.9, field_b)
    cr = T("cracks").sample(X, Y, ppm, angle=0.17)
    h -= cr * 0.020 * lane ** 2 * (0.25 + 0.75 * crack_cover)

    # --- erosion rills on slopes: run down the fall line, but wander (slow domain warp), swing in direction and mix
    #     two spacings, so there is no comb-like regularity; the two axis-aligned tiles are blended by direction -----
    gm = np.hypot(gxs, gys) + 1e-6
    wy = (gys / gm) ** 2                                         # fall line along y -> rills along tile y
    Xw = X + 0.20 * field_b                                      # +-0.2 m wander on a 3-20 m scale: rills bend
    Yw = Y + 0.20 * field_a                                      # by up to ~20 degrees and their spacing varies
    Tr = T("rills")
    rill_y = 0.70 * Tr.sample(Xw, Yw, ppm) + 0.50 * Tr.sample(Xw, Yw, ppm, angle=0.13, offset=(5.3, 2.9), scale=1.55)
    rill_x = 0.70 * Tr.sample(Xw, Yw, ppm, swap=True, offset=(3.1, 1.7)) + \
        0.50 * Tr.sample(Xw, Yw, ppm, angle=0.13, swap=True, offset=(1.9, 6.1), scale=1.55)
    rills = rill_y * wy + rill_x * (1.0 - wy)
    rill_amt = _smooth(0.10, 0.45, slope) * (1.0 - lane) * (0.35 + 0.65 * _smooth(-0.8, 0.6, field_b))
    if rock is not None:
        rill_amt = rill_amt * (1.0 - 0.55 * rock)                # faces of fractured rock are blocky, not combed
    h += rills * rill_amt

    # --- fractured blocks with bedding ledges on the blast-cut bench face -------------------------------------------
    if rock is not None and float(rock.max()) > 0.01:
        Tf = T("facets")
        fac_y = Tf.sample(Xw, Yw, ppm)
        fac_x = Tf.sample(Xw, Yw, ppm, swap=True, offset=(2.1, 4.4))
        h += (fac_y * wy + fac_x * (1.0 - wy)) * rock * (0.35 + 0.45 * _smooth(-1.0, 0.8, field_a - 0.3 * field_b))

    # --- haul-lane features: patchy washboard corrugations and tyre-tread imprints --------------------------
    wb_patch = _smooth(0.2, 1.1, field_b - 0.2 * field_a)
    phase = X / 0.62 + 0.18 * np.sin(Y * 0.8 + 3.0 * field_a)
    h += 0.0065 * np.sin(2 * math.pi * phase) * wb_patch * lane ** 2 * _smooth(0.6, 1.0, lane)
    tread_h, track_w = _tread_height(terrain, X, Y, ppm, field_a, field_b, lane)
    h += tread_h
    return h.astype(np.float32), dict(track=track_w, gravel=gravel, cover=cover, cracks=cr * crack_cover,
                                      field=field_a, face=face, off=off)


_INFO_KEYS = ("track", "gravel", "cover", "cracks", "field", "face", "off")


def _height_bands(F, terrain, ppm, seed, W, H, band: int = 384, halo: int = 0):
    """Yield ``(r0, r1, a, b, h, info)``: output rows ``r0:r1``; ``h`` / ``info`` cover rows ``a:b`` = the band
    plus ``halo`` extra rows each side (clipped to the image), so finite differences need no full-size arrays."""
    xs, ys = F["xs"].astype(np.float32), F["ys"].astype(np.float32)
    for r0 in range(0, H, band):
        r1 = min(r0 + band, H)
        a, b = max(r0 - halo, 0), min(r1 + halo, H)
        X = np.broadcast_to(xs[None, :], (b - a, W))
        Y = np.broadcast_to(ys[a:b, None], (b - a, W))
        hb, ib = _height_band(terrain, seed, ppm, X, Y, F["lane"][a:b], F["slope"][a:b], F["gx"][a:b], F["gy"][a:b],
                              rock=F["rock"][a:b])
        ib["lane"] = F["lane"][a:b]
        ib["slope"] = F["slope"][a:b]
        yield r0, r1, a, b, hb, ib


def _terrain_height(terrain, rect, ppm, seed, W, H, band: int = 384):
    """Micro-relief height (metres) and the class weights it was built from (processed in row bands)."""
    F = _coarse_fields(terrain, rect, ppm, W, H)
    h = np.empty((H, W), np.float32)
    info = {k: np.empty((H, W), np.float32) for k in _INFO_KEYS}
    for r0, r1, _a, _b, hb, ib in _height_bands(F, terrain, ppm, seed, W, H, band):
        h[r0:r1] = hb
        for k in _INFO_KEYS:
            info[k][r0:r1] = ib[k]
    info.update(lane=F["lane"], slope=F["slope"])
    return h, info


def _tread_height(terrain, X, Y, ppm, field_a, field_b, lane):
    """Chevron tyre-tread imprint inside the tracks (pitch 0.32 m like bake_albedo's tread band) + compaction.

    X varies along columns and Y along rows; only the rows near a track are evaluated."""
    h = np.zeros(X.shape, np.float32)
    band_all = np.zeros(X.shape, np.float32)
    pitch = 0.32
    tracks = [(t, 1.0) for t in terrain.tracks_y] + [(t, 0.4) for t in terrain.extra_tracks]
    wear = _smooth(-0.5, 0.7, field_a * 0.6 + field_b * 0.8)                      # dusty / worn patches fill the lugs
    ycol = np.asarray(Y[:, 0], np.float64)
    for ty, strength in tracks:
        rows = np.nonzero(np.abs(ycol - ty) < 0.95)[0]
        if len(rows) == 0:
            continue
        r0, r1 = rows[0], rows[-1] + 1
        x = np.asarray(X[r0:r1], np.float64)
        y = np.asarray(Y[r0:r1], np.float64)
        dy = y - ty - 0.06 * np.sin(x * 0.21 + ty * 3.0)
        inside = np.exp(-(dy / 0.34) ** 4)                                        # tyre contact patch, ~0.7 m wide
        band_all[r0:r1] = np.maximum(band_all[r0:r1], inside * strength)
        ph = x / pitch + 0.65 * np.abs(dy) / pitch
        t = 0.5 + 0.5 * np.cos(2 * math.pi * (ph - 0.75))
        lug = _smooth(0.55, 0.9, t)                                              # narrow trough along each lug
        hh = -0.018 * lug * inside * strength * (0.45 + 0.55 * wear[r0:r1])
        hh -= 0.004 * inside * strength                                           # compacted, slightly depressed footprint
        hh += 0.006 * np.exp(-((np.abs(dy) - 0.52) / 0.12) ** 2) * strength       # soft shoulder either side
        h[r0:r1] += hh.astype(np.float32)
    return h * lane, np.clip(band_all * lane, 0, 1)


_RELIEF_GAIN = 1.3          # the renderer shows normal maps with less contrast than a raking-light preview
_FLAT_ROUGH = 0.95


def _grid_size(rect, px_per_m):
    """(W, H) exactly as terrain.bake_albedo computes them."""
    x0, x1, y0, y1 = rect
    return int(round((x1 - x0) * px_per_m)), int(round((y1 - y0) * px_per_m))


def detail_px_per_m(albedo_px_per_m: float, rect=None, factor: float = 2.5, max_pixels: float = 40e6) -> float:
    """Resolution to bake the detail maps (normal / roughness / cavity) at, given the albedo's ``px_per_m``.

    The relief is band-limited to the pixel size, so the detail maps should be baked 2-3x denser than the albedo
    (UsdUVTexture inputs may have different resolutions on the same UVs).  ``max_pixels`` (and 16384 per side)
    caps the result for big rects; for a camera path that stays near the lane pass a near-camera sub-rect instead.
    """
    ppm = float(albedo_px_per_m) * float(factor)
    if rect is not None:
        x0, x1, y0, y1 = rect
        w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        ppm = min(ppm, math.sqrt(max_pixels / (w * h)), 16384.0 / max(w, h))
    return max(ppm, float(albedo_px_per_m))


def _terrain_maps(terrain, rect, px_per_m, seed):
    W, H = _grid_size(rect, px_per_m)
    h, info = _terrain_height(terrain, rect, float(px_per_m), int(seed), W, H)
    return h, info, W, H


def _rough_from_info(info):
    lane, face, off = info["lane"], info["face"], info["off"]
    r = 0.955 * off + 0.88 * lane * (1 - face) + 0.945 * face
    r = _mix(r, 0.80, info["track"] * 0.9)                           # polished by traffic
    pebble_top = _smooth(0.004, 0.02, info["gravel"]) * (0.4 + 0.6 * info["cover"])
    r = r - 0.13 * pebble_top * (off + 0.4 * lane + 0.5 * face)      # stone is smoother than powdery dust
    r = r + 0.04 * info["cracks"] * lane
    r = r + 0.035 * info["field"]
    return _rough_u8(np.clip(r, 0.35, 1.0))


def terrain_detail_maps(terrain, rect, px_per_m: float, seed: int = 0, strength: float = 1.0):
    """(normal RGB uint8 (H, W, 3), roughness uint8 (H, W)) for ``rect = (x0, x1, y0, y1)``.

    Same pixel layout as ``terrain.bake_albedo(terrain, rect, px_per_m)``: row 0 is y1,
    ``W = round((x1 - x0) * px_per_m)``, ``H = round((y1 - y0) * px_per_m)``; the tangent
    frame is (+x, +y) so tangent-space X/Y are world x/y slopes of the micro relief.
    Features are anchored in world space (a crop of a larger rect looks identical) and
    band-limited to the pixel size, so a coarse far texture gets almost flat normals.

    The albedo does not have to be baked at the same ``px_per_m``: UsdUVTexture inputs of one
    material may differ in resolution, and the detail only really appears once a pixel is
    smaller than a pebble.  Use :func:`detail_px_per_m` (2-3x the albedo density) or bake a
    near-camera sub-rect densely; a 40 x 16 m lane strip at 64 px/m is 2560 x 1024 px and
    takes about 10 s.  The image is processed in row bands, so peak memory stays near the
    size of the two output maps.  Degenerate rects (fewer than 2 px a side) work too.
    """
    W, H = _grid_size(rect, px_per_m)
    if W <= 0 or H <= 0:
        return np.empty((max(H, 0), max(W, 0), 3), np.uint8), np.empty((max(H, 0), max(W, 0)), np.uint8)
    Wp, Hp = max(W, 2), max(H, 2)                       # pad 1-px strips: finite differences need two samples
    ppm = float(px_per_m)
    F = _coarse_fields(terrain, rect, ppm, Wp, Hp)
    normal = np.empty((Hp, Wp, 3), np.uint8)
    rough = np.empty((Hp, Wp), np.uint8)
    for r0, r1, a, b, hb, ib in _height_bands(F, terrain, ppm, int(seed), Wp, Hp, halo=1):
        hb = hb * _RELIEF_GAIN
        dcol = np.gradient(hb, axis=1) * ppm            # world gradients (metres per metre); row index runs against +y
        drow = np.gradient(hb, axis=0) * ppm
        sl = slice(r0 - a, r1 - a)
        normal[r0:r1] = _encode_normal(-dcol[sl] * strength, drow[sl] * strength)
        rough[r0:r1] = _rough_from_info({k: v[sl] for k, v in ib.items()})
    return normal[:H, :W], rough[:H, :W]


def terrain_detail_cavity(terrain, rect, px_per_m: float, seed: int = 0) -> np.ndarray:
    """Optional baked cavity / micro-AO (uint8, 255 = unoccluded) from the same relief; multiply it into
    the albedo to get dark cracks, tread lugs and pebble gaps that the renderer cannot derive from a normal map.
    Same pixel layout as :func:`terrain_detail_maps`; processed in row bands (blur halo of 3 sigma)."""
    W, H = _grid_size(rect, px_per_m)
    if W <= 0 or H <= 0:
        return np.empty((max(H, 0), max(W, 0)), np.uint8)
    Wp, Hp = max(W, 2), max(H, 2)
    ppm = float(px_per_m)
    sigma = max(1.0, 0.06 * ppm)
    F = _coarse_fields(terrain, rect, ppm, Wp, Hp)
    out = np.empty((Hp, Wp), np.uint8)
    for r0, r1, a, b, hb, _ib in _height_bands(F, terrain, ppm, int(seed), Wp, Hp, band=512, halo=int(3 * sigma) + 2):
        occ = _blur_nonperiodic(hb, sigma) - hb              # positive in hollows
        cav = 1.0 - np.clip(occ * 11.0, 0.0, 0.36)
        out[r0:r1] = np.clip(cav[r0 - a:r1 - a] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out[:H, :W]


def _blur_nonperiodic(a, sigma):
    pad = int(3 * sigma) + 1
    p = np.pad(a, pad, mode="reflect")
    q = p.astype(np.float32)
    fy = np.fft.fftfreq(p.shape[0])[:, None]
    fx = np.fft.rfftfreq(p.shape[1])[None, :]
    g = np.exp(-2.0 * math.pi ** 2 * sigma ** 2 * (fx ** 2 + fy ** 2))
    out = np.fft.irfft2(np.fft.rfft2(q) * g, s=q.shape)
    return out[pad:-pad, pad:-pad].astype(np.float32)


# ---------------------------------------------------------------------------
# Rock (blasted quarry stone) — used by the rock / gravel instances
# ---------------------------------------------------------------------------

def rock_textures(seed: int = 0, size: int = 512, shade: float = 1.0, metres: float = 1.0) -> TextureSet:
    """Tileable fractured quarry stone: grey-brown rock with mineral speckle, a network of fine
    fractures (darker, slightly rougher), iron-oxide staining and pale dust settled in the cracks.
    ``shade`` scales the base brightness (instance colour variants); ``metres`` = size of one tile."""
    rng = np.random.default_rng(int(seed) * 7 + 3)
    n = int(size)
    mpp = metres / n
    base = np.array([0.50, 0.47, 0.43], np.float32) * float(shade)
    blotch = nz(rng, n, 700, 90, beta=2.6)
    mid = nz(rng, n, 160, 20, beta=2.2)
    speck = nz(rng, n, 10, 2.5, beta=0.6)
    f1, f2 = voronoi_f12(n, 5, rng)
    keep = np.clip((nz(rng, n, 400, 80, beta=2.2) + 0.2) * 1.6, 0.0, 1.0)          # only some cell edges crack
    crack = keep * np.clip(1.0 - (f2 - f1) / 0.035, 0.0, 1.0) ** 2
    vein = np.clip(1.0 - np.abs(nz(rng, n, 300, 40, beta=2.0, stretch=(3.0, 1.0))) / 0.06, 0.0, 1.0) ** 3
    crack = np.maximum(crack, 0.35 * vein)
    oxide = np.clip(nz(rng, n, 500, 60, beta=2.4) - 0.8, 0.0, None)
    grain = nz(rng, n, 4, 1.5, beta=0.2)

    lum = 1.0 + 0.09 * blotch + 0.07 * mid + 0.06 * speck + 0.05 * grain
    col = base[None, None, :] * lum[..., None]
    col = col * (1.0 - 0.32 * crack[..., None])
    col = col + oxide[..., None] * np.array([0.10, 0.045, 0.0], np.float32) * 0.5
    dust = np.clip(blur(crack, 1.5) * 1.2 + 0.25 * np.clip(-mid, 0, None), 0.0, 1.0) * 0.45
    col = col * (1.0 - dust[..., None]) + np.array([0.72, 0.66, 0.56], np.float32) * dust[..., None]
    albedo = (np.clip(col, 0, 1) * 255 + 0.5).astype(np.uint8)

    height = 0.004 * blotch + 0.0018 * mid + 0.0006 * speck + 0.0003 * grain - 0.002 * crack
    normal = height_to_normal(height.astype(np.float32), mpp, gain=1.0)
    rough = np.clip(0.80 + 0.08 * mid + 0.10 * crack - 0.05 * dust, 0.55, 1.0)
    return TextureSet(albedo, (rough * 255 + 0.5).astype(np.uint8), normal, metallic=0.0,
                      metres_per_tile=float(metres))


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_texture_set(ts: TextureSet, directory: str, prefix: str) -> dict[str, str]:
    """Write ``<prefix>_albedo.png`` (sRGB), ``<prefix>_roughness.png`` and ``<prefix>_normal.png`` (raw).
    Returns absolute paths keyed ``albedo`` / ``roughness`` / ``normal``."""
    from PIL import Image

    directory = os.path.abspath(os.path.expanduser(directory))
    os.makedirs(directory, exist_ok=True)
    out = {}
    for key, arr, mode in (("albedo", ts.albedo, "RGB"), ("roughness", ts.roughness, "L"), ("normal", ts.normal, "RGB")):
        path = os.path.join(directory, f"{prefix}_{key}.png")
        Image.fromarray(np.ascontiguousarray(arr), mode).save(path)
        out[key] = path
    return out


def save_terrain_detail_maps(normal: np.ndarray, roughness: np.ndarray, directory: str, prefix: str) -> dict[str, str]:
    """Write the maps from :func:`terrain_detail_maps` as ``<prefix>_normal.png`` / ``<prefix>_roughness.png``
    (both ``raw`` colour space); returns absolute paths keyed ``normal`` / ``roughness``."""
    from PIL import Image

    directory = os.path.abspath(os.path.expanduser(directory))
    os.makedirs(directory, exist_ok=True)
    out = {}
    for key, arr, mode in (("normal", normal, "RGB"), ("roughness", roughness, "L")):
        path = os.path.join(directory, f"{prefix}_{key}.png")
        Image.fromarray(np.ascontiguousarray(arr), mode).save(path)
        out[key] = path
    return out
