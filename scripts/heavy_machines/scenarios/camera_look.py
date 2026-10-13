"""Camera-look post-processing: turns a clean CG render into believable camera footage.

The renderer hands us perfect, noiseless, hazeless pixels.  A real mast camera in a dusty
open-pit quarry delivers something else: aerial perspective (distant benches fade into the
haze), a soft optical transfer function, a little glare around bright areas, a filmic
highlight shoulder, lens distortion / colour fringing / vignetting, in-camera sharpening
and sensor noise.  ``apply`` adds all of that to an sRGB frame::

    from heavy_machines.scenarios import camera_look as CL
    CL.warm_up(720, 1280)                                    # optional: build the per-size lens maps once
    out = CL.apply(rgb, depth_m, CL.LookParams(), seed=frame_index)    # (H, W, 3) uint8

Pipeline (linear light unless noted)::

    sRGB decode (x exposure x white balance) -> aerial-perspective haze / dust with depth
    -> lens softness (MTF) -> bloom + veiling glare -> highlight shoulder + S-curve (one LUT)
    ---- display-referred values ----
    -> saturation -> lens distortion + chromatic aberration -> vignette -> luma sharpen
    -> luma-weighted sensor noise -> uint8

Exposure / white balance are plain linear gains, so they are folded into the decode table (and
into the airlight colours): the result is identical to applying them after the haze.

Pure numpy, float32 planar ``(3, H, W)`` internally, processed in cache-sized bands of rows (the
elementwise passes then work on ~160 KB slabs instead of streaming the whole frame through memory
once per operation; the result does not depend on the band height).  Measured on this 4-core box
with other jobs running: 1280x720 about 0.18 s at best and ~0.2 s typical, worst ~0.3 s over 25
frames; 1920x1080 ~0.4 to 0.5 s; 3840x2160 ~2.5 s (cost is ~linear in pixels).  The first frame of a
size is slower (~0.35 s at 720p: lens maps and allocator warm-up): call ``warm_up(h, w)`` once before
timing or recording.  The geometric maps (distortion / CA / vignette) are cached per frame size under
a memory budget (``_MAP_CACHE_BYTES``; larger sets are rebuilt per frame instead of cached).
``identity_params()`` switches every effect off and ``apply`` then returns the input unchanged
(+-1 LSB).

Depth convention: ``depth_m`` is metres from the camera; ``inf``, ``NaN``, ``0`` and anything
>= 1e5 m is *sky*.  Sky gets no depth haze of its own; instead it is blended toward the haze
colour near the horizon, so the far terrain, the ridge line and the sky meet without a seam,
like in a hazy photograph.  Where the horizon is is either given (``LookParams.horizon_row``,
see ``horizon_from_camera``; fully stable from frame to frame) or estimated from the frame:
the weighted median row where the sky meets *far* geometry (boundaries against machines or
other near surfaces carry no weight, so a boom or truck sweeping through the sky does not make
the glow pump).  Without a depth map the haze/dust stage is skipped (everything else applies).
"""

from __future__ import annotations

import functools
import math
from collections import OrderedDict
from dataclasses import dataclass, replace

import numpy as np

__all__ = ["LookParams", "apply", "identity_params", "quarry_midday", "dusty_afternoon", "overcast",
           "horizon_from_camera", "warm_up"]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class LookParams:
    """Defaults: 4 m mast camera in a dusty open-pit quarry at hazy midday.

    The first sixteen fields are the specified ones, in the specified order, so positional construction
    is safe; the extras after ``dust_in_air`` are meant to be passed by keyword.
    """

    haze_visibility_m: float = 900.0     # transmittance = exp(-haze_strength * depth / visibility)
    haze_color: tuple = (0.82, 0.84, 0.86)   # horizon / airlight colour, sRGB-encoded (what you see at infinity)
    haze_strength: float = 1.0           # 0 = off; scales the extinction (and the sky horizon glow)
    exposure: float = 1.0                # linear multiplier applied before bloom / tonemap
    contrast: float = 1.0                # S-curve strength; 0 = off, 1 = gentle filmic, 2 = punchy
    saturation: float = 0.9
    white_balance: tuple = (1.02, 1.0, 0.97)   # linear per-channel gain (slightly warm)
    highlight_rolloff: float = 0.8       # 0 = hard clip, 1 = long soft shoulder
    bloom_strength: float = 0.06         # also drives the faint veiling glare that lifts the blacks
    bloom_radius_px: float = 18.0
    vignette: float = 0.22               # corner light fall-off (fraction at the corner)
    chromatic_aberration_px: float = 0.7  # red/blue radial displacement at the corner, opposite directions
    distortion_k1: float = -0.025        # radial distortion; < 0 barrel (the frame is re-zoomed to stay filled)
    sharpen: float = 0.25                # in-camera unsharp mask on luma
    grain: float = 0.018                 # sensor noise; 0.018 ~ 1.9/255 luma std at mid-grey (+ soft chroma noise)
    dust_in_air: float = 0.0             # 0 = clear; 1 = thick dust (~150 m visibility), warm tinted
    # --- extras (not in the original spec; appended so positional use of the spec fields is safe) ----
    lens_blur_px: float = 0.55           # sigma of the optical softness / demosaic blur (0 = off)
    dust_color: tuple = (0.80, 0.73, 0.63)   # sRGB-encoded colour of the dust (airlight)
    horizon_row: float | None = None     # image row (0 = top, pre-distortion pixels) of the true horizon; None = estimate
    horizon_tilt: float = 0.0            # rows per column the horizon sinks toward the right (camera roll); only used
    #                                      together with ``horizon_row``

    def replace(self, **kw) -> "LookParams":
        return replace(self, **kw)


def identity_params() -> LookParams:
    """Every effect off: ``apply(rgb, depth, identity_params())`` returns ``rgb[..., :3]`` (+-1 LSB)."""
    return LookParams(haze_strength=0.0, dust_in_air=0.0, exposure=1.0, contrast=0.0, saturation=1.0,
                      white_balance=(1.0, 1.0, 1.0), highlight_rolloff=0.0, bloom_strength=0.0,
                      vignette=0.0, chromatic_aberration_px=0.0, distortion_k1=0.0, lens_blur_px=0.0,
                      sharpen=0.0, grain=0.0)


def quarry_midday() -> LookParams:
    """The default look (hazy, dusty midday sun)."""
    return LookParams()


def dusty_afternoon() -> LookParams:
    """Thicker dust in the air, warmer cast, a bit more glow."""
    return LookParams(haze_visibility_m=700.0, haze_color=(0.84, 0.81, 0.76), dust_in_air=0.22,
                      white_balance=(1.03, 1.0, 0.95), saturation=0.88, bloom_strength=0.07)


def overcast() -> LookParams:
    """Flat light: cooler, less contrast and glow."""
    return LookParams(haze_visibility_m=1400.0, haze_color=(0.80, 0.82, 0.84), contrast=0.7,
                      white_balance=(0.99, 1.0, 1.02), saturation=0.88, bloom_strength=0.03,
                      vignette=0.18)


def horizon_from_camera(height_px: int, focal_px: float, pitch_down_deg: float = 0.0,
                        roll_deg: float = 0.0) -> tuple[float, float]:
    """``(horizon_row, horizon_tilt)`` of the infinite ground plane for a pinhole camera.

    height_px       image height in pixels
    focal_px        focal length in pixels (``focal_length / vertical_aperture * height_px`` for a USD camera)
    pitch_down_deg  positive = camera looks down, so the horizon sits above the image centre
    roll_deg        positive = the horizon sinks toward the right edge (rows per column = tan(roll))

    Feed the result to ``LookParams(horizon_row=..., horizon_tilt=...)``: the sky glow is then
    tied to the camera, not to whatever happens to be in the frame.
    """
    row = 0.5 * (height_px - 1) - float(focal_px) * math.tan(math.radians(pitch_down_deg))
    return float(row), float(math.tan(math.radians(roll_deg)))


# ---------------------------------------------------------------------------
# Constants and tables
# ---------------------------------------------------------------------------

SKY_DEPTH_M = 1.0e5            # depth at / beyond which a pixel counts as sky
_SKY_HAZE_PEAK = 0.80          # fraction of the haze colour mixed into the sky right at the horizon
_SKY_HAZE_FLOOR = 0.03         # ... and far above it (a faint overall whitening of the sky)
_SKY_HAZE_HEIGHT = 0.11        # e-folding height above the horizon, in image heights
_HORIZON_NEAR_M = 60.0         # sky boundaries against surfaces nearer than this carry (almost) no weight ...
_HORIZON_FAR_M = 200.0         # ... and full weight from here on (smoothstep in between)
_HORIZON_MIN_WEIGHT = 0.02     # keeps the estimate defined when only near boundaries exist
_HORIZON_TRIM_LO, _HORIZON_TRIM_HI = 0.10, 0.90   # weighted quantile range averaged by the horizon estimate
_BLOOM_THRESHOLD = 0.58        # linear value where highlights start to glow
_BLOOM_KNEE = 0.22
_VEIL_GAIN = 0.06              # unthresholded wide glare, relative to bloom_strength
_LUMA = (0.2126, 0.7152, 0.0722)
_LUMA32 = np.asarray(_LUMA, np.float32)
_GRAIN_SCALE = 0.42           # grain=0.018 -> ~1.9/255 luma std at mid-grey (plus soft chroma noise)
_GRAIN_CHROMA = 0.6
_DISPLAY_GAMMA = 2.2           # for converting photometric gains (vignette) into display space
_MAP_CACHE_BYTES = 200 * 1024 * 1024   # budget of the per-size lens-map cache
_TONE_XMAX = 8.0               # linear values above this are clamped before the shoulder LUT (it is flat there)


def _build_decode_table() -> np.ndarray:
    i = np.arange(256, dtype=np.float64) / 255.0
    return np.where(i <= 0.04045, i / 12.92, ((i + 0.055) / 1.055) ** 2.4)


_DEC_TABLE = _build_decode_table()                       # float64, sRGB -> linear


def _srgb_to_linear(c) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _oetf(x: np.ndarray) -> np.ndarray:
    """Linear -> sRGB-encoded (float64, x in [0, 1])."""
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.maximum(x, 1e-12), 1 / 2.4) - 0.055)


@functools.lru_cache(maxsize=8)
def _tone_lut(rolloff: float, contrast: float):
    """Lookup table linear -> display for the highlight shoulder, the sRGB OETF and the S-curve.

    Returns ``(table float32, scale)``: index = int(linear * scale), clamped by ``take(mode='clip')``.
    The index is a truncation, so each table entry holds the value at the bin centre.
    """
    if rolloff > 0.0:
        k = float(np.clip(1.0 - 0.55 * rolloff, 0.25, 1.0))
        xmax, n = _TONE_XMAX, 1 << 17
    else:
        k, xmax, n = 1.0, 1.0, 1 << 16
    scale = n / xmax
    x = (np.arange(n, dtype=np.float64) + 0.5) / scale
    if k < 1.0:                                           # C1 soft knee, asymptote 1.0
        s = 1.0 - k
        x = np.where(x > k, k + s * (1.0 - np.exp(-(x - k) / s)), x)
    y = _oetf(np.clip(x, 0.0, 1.0))
    if contrast > 0.0:
        mix = min(0.50 * contrast, 0.95)
        y = y - mix * y * (1.0 - y) * (1.0 - 2.0 * y)
    return y.astype(np.float32), float(scale)


# ---------------------------------------------------------------------------
# Small array helpers (all float32, planar (C, H, W) or (H, W))
# ---------------------------------------------------------------------------

def _gauss_kernel(sigma: float, radius_sigmas: float = 2.5) -> np.ndarray:
    r = max(1, int(round(radius_sigmas * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    w = np.exp(-0.5 * (x / sigma) ** 2)
    return (w / w.sum()).astype(np.float32)


def _sl(ndim: int, axis: int, s: slice) -> tuple:
    return tuple(s if i == axis else slice(None) for i in range(ndim))


def _blur_axis(a: np.ndarray, w: np.ndarray, axis: int) -> np.ndarray:
    """Separable 1-D convolution with a symmetric kernel and edge clamping (new array)."""
    axis %= a.ndim
    r = len(w) // 2
    n = a.shape[axis]
    nd = a.ndim

    def sl(s):
        return _sl(nd, axis, s)

    if n < 2 * r:                                 # tiny arrays: generic gather
        out = np.zeros_like(a)
        for k in range(-r, r + 1):
            out += w[k + r] * np.take(a, np.clip(np.arange(n) + k, 0, n - 1), axis=axis)
        return out
    out = a * w[r]
    for o in range(1, r + 1):
        wo = w[r + o]
        m = n - 2 * o
        if m > 0:                                 # interior: out[i] += wo * (a[i - o] + a[i + o])
            u = a[sl(slice(0, m))] + a[sl(slice(2 * o, n))]
            u *= wo
            out[sl(slice(o, n - o))] += u
        e = a[sl(slice(0, 1))] + a[sl(slice(o, 2 * o))]            # left edge: a[i - o] clamps to a[0]
        e *= wo
        out[sl(slice(0, o))] += e
        e = a[sl(slice(n - 2 * o, n - o))] + a[sl(slice(n - 1, n))]   # right edge: a[i + o] clamps to a[n - 1]
        e *= wo
        out[sl(slice(n - o, n))] += e
    return out


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur over the last two axes."""
    w = _gauss_kernel(sigma)
    return _blur_axis(_blur_axis(a, w, a.ndim - 1), w, a.ndim - 2)


_K3 = np.array([0.25, 0.5, 0.25], np.float32)           # sigma = 0.707


def _blur3(a: np.ndarray) -> np.ndarray:
    return _blur_axis(_blur_axis(a, _K3, a.ndim - 1), _K3, a.ndim - 2)


def _down2(a: np.ndarray) -> np.ndarray:
    """2x2 box downsample over the last two axes (odd trailing row / column dropped)."""
    h, w = a.shape[-2] // 2 * 2, a.shape[-1] // 2 * 2
    if h < 2 or w < 2:
        return a
    return (a[..., 0:h:2, 0:w:2] + a[..., 1:h:2, 0:w:2] + a[..., 0:h:2, 1:w:2] + a[..., 1:h:2, 1:w:2]) * np.float32(0.25)


def _up_axis_int(a: np.ndarray, k: int, axis: int) -> np.ndarray:
    """Linear upsample by the integer ``k`` along ``axis`` (edge clamped); output length is ``k * n``.

    Identical to sampling at ``(o + 0.5) / k - 0.5`` but built from ``k`` phase slices instead of
    gathers: each output phase is ``L + f * (R - L)`` of two shifted views.
    """
    axis %= a.ndim
    nd = a.ndim
    n = a.shape[axis]

    def sl(s):
        return _sl(nd, axis, s)

    p = np.concatenate([a[sl(slice(0, 1))], a, a[sl(slice(n - 1, n))]], axis=axis)     # edge-replicated, n + 2
    lo, mid, hi = p[sl(slice(0, n))], p[sl(slice(1, n + 1))], p[sl(slice(2, n + 2))]
    d_lo, d_hi = mid - lo, hi - mid
    shape = list(a.shape)
    shape[axis] = n * k
    out = np.empty(shape, a.dtype)
    for ph in range(k):
        s = (ph + 0.5) / k - 0.5
        if s < 0.0:
            left, d, f = lo, d_lo, s + 1.0
        else:
            left, d, f = mid, d_hi, s
        view = out[sl(slice(ph, None, k))]
        np.multiply(d, np.float32(f), out=view)
        view += left
    return out


def _fit_axis(a: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """Crop, or edge-pad, ``axis`` to length ``n_out``."""
    axis %= a.ndim
    n = a.shape[axis]
    if n == n_out:
        return a
    if n > n_out:
        return a[_sl(a.ndim, axis, slice(0, n_out))]
    pad = np.repeat(a[_sl(a.ndim, axis, slice(n - 1, n))], n_out - n, axis=axis)
    return np.concatenate([a, pad], axis=axis)


def _upsample_int(a: np.ndarray, k: int, h: int, w: int) -> np.ndarray:
    """Bilinear x``k`` upsample of (..., h', w') to (..., h, w) (edge clamped, cropped / padded to size)."""
    a = _up_axis_int(a, k, a.ndim - 1)                  # cheap pass on the small array first
    a = _up_axis_int(a, k, a.ndim - 2)                  # ... then whole-row slabs on the large one
    return _fit_axis(_fit_axis(a, w, a.ndim - 1), h, a.ndim - 2)


def _luma(d: np.ndarray) -> np.ndarray:
    """Rec.709 luma of planar (3, H, W)."""
    _, h, w = d.shape
    return np.dot(_LUMA32, d.reshape(3, -1)).reshape(h, w)




def _row_weights(n_in: int, k: int, h_out: int):
    """Per output row of a x``k`` bilinear upsample: source rows ``i0``, ``i1`` and the weight ``f`` of ``i1``."""
    s = (np.arange(h_out, dtype=np.float32) + np.float32(0.5)) / np.float32(k) - np.float32(0.5)
    np.clip(s, 0.0, n_in - 1, out=s)
    i0 = np.minimum(s.astype(np.intp), n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    return i0, i1, (s - i0).astype(np.float32)


def _rows_interp(ax: np.ndarray, rw, y0: int, y1: int) -> np.ndarray:
    """Output rows ``y0:y1`` of a vertical bilinear upsample of ``ax`` (..., n_in, W) with weights ``rw``."""
    i0, i1, f = rw
    a = ax.take(i0[y0:y1], axis=-2)
    b = ax.take(i1[y0:y1], axis=-2)
    b -= a
    b *= f[y0:y1, None]
    b += a
    return b


# ---------------------------------------------------------------------------
# Colour space
# ---------------------------------------------------------------------------

def _decode_rows(img: np.ndarray, luts, a0: int, a1: int) -> np.ndarray:
    """Rows ``a0:a1`` of uint8 sRGB (H, W, 3) -> planar linear float32 (3, rows, W) (``luts`` hold the gains)."""
    slab = np.empty((3, a1 - a0, img.shape[1]), np.float32)
    for c in range(3):
        luts[c].take(img[a0:a1, :, c], out=slab[c], mode="clip")
    return slab


def _tone(lin: np.ndarray, rolloff: float, contrast: float) -> np.ndarray:
    """Linear light -> display-referred [0, 1]: highlight shoulder, sRGB OETF and S-curve in one table lookup."""
    lut, scale = _tone_lut(round(float(rolloff), 6), round(float(contrast), 6))
    idx = lin * np.float32(scale)
    return lut.take(idx.astype(np.intp), mode="clip")


def _saturate(disp: np.ndarray, saturation: float) -> np.ndarray:
    """Blend each pixel toward its own luma (``saturation`` < 1 desaturates); ``disp`` is C-contiguous (3, H, W)."""
    s = float(saturation)
    _, h, w = disp.shape
    m = (s * np.eye(3) + (1.0 - s) * np.outer(np.ones(3), _LUMA)).astype(np.float32)
    out = np.dot(m, disp.reshape(3, -1)).reshape(3, h, w)
    if s > 1.0:
        np.clip(out, 0.0, 1.0, out=out)
    return out


# ---------------------------------------------------------------------------
# Atmosphere
# ---------------------------------------------------------------------------

def _prep_depth(depth_m, h: int, w: int):
    if depth_m is None:
        return None
    d = np.asarray(depth_m)
    if d.ndim == 3 and d.shape[2] == 1:
        d = d[..., 0]
    if d.shape != (h, w):
        raise ValueError(f"depth_m shape {d.shape} does not match the image {(h, w)}")
    return d.astype(np.float32, copy=False)


def _horizon_estimate(sky: np.ndarray, depth: np.ndarray):
    """Horizon row estimated from the frame: weighted trimmed mean of where the sky meets *far* geometry.

    For every column that shows sky take its lowest sky pixel (the sky boundary) and look at the
    depth a few pixels below it.  A boundary against a machine, boom or nearby wall says nothing
    about the horizon and gets (almost) no weight; boundaries against far terrain / ridges (and
    columns that are sky down to the bottom edge) count fully.  The estimate is the mean of the
    middle 80 % (by weight) of the boundary rows: a rank-based statistic with smooth weights, so an
    occluder sliding across the frame moves it by well under a pixel per frame.  Returns None
    without sky.
    """
    h, w = sky.shape
    has = sky.any(axis=0)
    if not has.any():
        return None
    cols = np.flatnonzero(has)
    bottom = (h - 1) - np.argmax(sky[::-1][:, cols], axis=0)
    probe = np.minimum(bottom + 3, h - 1)
    dp = depth[probe, cols]
    dp = np.where((dp > 0.0) & (dp < SKY_DEPTH_M), dp, np.float32(np.inf))     # sky-like probes are "infinitely far"
    t = np.clip((dp - _HORIZON_NEAR_M) / (_HORIZON_FAR_M - _HORIZON_NEAR_M), 0.0, 1.0).astype(np.float64)
    wt = _HORIZON_MIN_WEIGHT + (1.0 - _HORIZON_MIN_WEIGHT) * t * t * (3.0 - 2.0 * t)
    order = np.argsort(bottom, kind="stable")
    rows = bottom[order].astype(np.float64)
    wt = wt[order]
    hi_cum = np.cumsum(wt)
    lo_cum = hi_cum - wt
    total = hi_cum[-1]
    inside = np.clip(np.minimum(hi_cum, _HORIZON_TRIM_HI * total) - np.maximum(lo_cum, _HORIZON_TRIM_LO * total), 0.0, None)
    return float((inside * rows).sum() / inside.sum())


def _sky_mix(h: int, w: int, horizon_row: float, tilt: float) -> np.ndarray:
    """Fraction of haze colour mixed into the sky: (h, 1) for a level horizon, (h, w) when tilted."""
    rows = np.arange(h, dtype=np.float32)
    if tilt == 0.0:
        height = (np.float32(horizon_row) - rows) * np.float32(1.0 / h)
        height = height[:, None]
    else:
        hor = np.float32(horizon_row) + np.float32(tilt) * (np.arange(w, dtype=np.float32) - np.float32((w - 1) / 2.0))
        height = (hor[None, :] - rows[:, None]) * np.float32(1.0 / h)
    np.maximum(height, 0.0, out=height)
    m = np.exp(height * np.float32(-1.0 / _SKY_HAZE_HEIGHT))
    return m * np.float32(_SKY_HAZE_PEAK - _SKY_HAZE_FLOOR) + np.float32(_SKY_HAZE_FLOOR)


def _veil(lin: np.ndarray, trans: np.ndarray, color) -> None:
    """lin = lin * trans + color * (1 - trans), per channel, in place."""
    for c in range(3):
        col = np.float32(color[c])
        x = lin[c]
        x -= col
        x *= trans
        x += col


class _Atmosphere:
    """Aerial-perspective haze and dust.  Frame-global quantities (sky mask, horizon glow) are computed once;
    ``apply`` then works on any slab of rows, so it can run inside the banded pipeline.

    ``gains`` is the exposure x white balance already folded into the linear image: the airlight colours
    get the same gains, which leaves the mix unchanged."""

    def __init__(self, depth: np.ndarray, p: LookParams, gains):
        h, w = depth.shape
        self.depth, self.p = depth, p
        self.sky = ~((depth > 0.0) & (depth < SKY_DEPTH_M))        # NaN, inf, <= 0, absurdly far
        self.mix = None
        if p.haze_strength > 0.0 and self.sky.any():
            if p.horizon_row is not None and math.isfinite(float(p.horizon_row)):
                horizon = float(p.horizon_row)
                tilt = float(p.horizon_tilt) if math.isfinite(float(p.horizon_tilt)) else 0.0
            else:
                horizon, tilt = _horizon_estimate(self.sky, depth), 0.0
            if horizon is not None:
                self.mix = _sky_mix(h, w, horizon, tilt) * np.float32(min(p.haze_strength, 1.0))
        g = np.asarray(gains, np.float32)
        self.haze_col = _srgb_to_linear(p.haze_color) * g
        self.dust_col = _srgb_to_linear(p.dust_color) * g

    def apply(self, lin: np.ndarray, a0: int, a1: int, soften: bool) -> None:
        """In place on ``lin`` = rows ``a0:a1`` of the image (planar (3, rows, W))."""
        p = self.p
        sky = self.sky[a0:a1]
        d = np.where(sky, np.float32(0.0), self.depth[a0:a1])
        if p.haze_strength > 0.0:
            vis = max(float(p.haze_visibility_m), 1.0)
            trans = np.exp(d * np.float32(-p.haze_strength / vis))
            if self.mix is not None:
                trans = np.where(sky, np.float32(1.0) - self.mix[a0:a1], trans)
            if soften:                                  # the aliased depth AOV gets the same softness as the picture
                trans = _blur3(trans)
            _veil(lin, trans, self.haze_col)
        if p.dust_in_air > 0.0:
            path = d + np.where(sky, np.float32(70.0), np.float32(0.0))   # dust hangs low: the sky sees ~70 m of it
            trans = np.exp(path * np.float32(-p.dust_in_air / 160.0))
            if soften:
                trans = _blur3(trans)
            _veil(lin, trans, self.dust_col)


# ---------------------------------------------------------------------------
# Optics in linear light
# ---------------------------------------------------------------------------

def _bloom_glow(q: np.ndarray, strength: float, radius_px: float) -> np.ndarray:
    """Glow of bright areas plus a faint unthresholded veiling glare, at the 1/4-resolution grid of ``q``
    (the 4x4 box mean of the linear image).  Multi-scale: three Gaussians at 1/4, 1/8 and 1/16 resolution."""
    thr, knee = np.float32(_BLOOM_THRESHOLD), np.float32(_BLOOM_KNEE)
    soft = np.clip(q - thr + knee, 0.0, 2.0 * knee)
    soft *= soft
    soft *= np.float32(1.0 / (4.0 * knee))
    bright = np.maximum(q - thr, soft)
    bright *= np.float32(1.0 / (1.0 - _BLOOM_THRESHOLD))

    r = max(float(radius_px), 1.0)
    b8 = _down2(bright)
    b16 = _down2(b8)
    q16 = _down2(_down2(q))
    g_a = _blur(bright, max(0.5 * r / 4.0, 0.5))
    g_b = _blur(b8, max(1.2 * r / 8.0, 0.5))
    g_c = _blur(b16, max(3.0 * r / 16.0, 0.5))
    qh, qw = q.shape[1:]
    glow = np.float32(0.50) * g_a
    glow += np.float32(0.30) * _upsample_int(g_b, 2, qh, qw)
    glow += np.float32(0.20) * _upsample_int(g_c, 4, qh, qw)
    glow *= np.float32(strength)
    veil = _blur(q16, 12.0)                                    # sigma ~190 px at full resolution
    glow += np.float32(strength * _VEIL_GAIN) * _upsample_int(veil, 4, qh, qw)
    return glow


# ---------------------------------------------------------------------------
# Geometric lens model (cached per frame size)
# ---------------------------------------------------------------------------

def _build_warp_maps(h: int, w: int, k1: float, ca_px: float):
    """Bilinear sampling maps per channel: list of (idx00, fx, fy), shared when channels coincide."""
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    rhd = math.hypot(cx, cy)
    dx = (np.arange(w, dtype=np.float32) - np.float32(cx))[None, :]
    dy = (np.arange(h, dtype=np.float32) - np.float32(cy))[:, None]
    r2 = (dx * dx + dy * dy) * np.float32(1.0 / (rhd * rhd + 1e-9))
    base = (1.0 + k1) / (1.0 + k1 * r2) if k1 != 0.0 else np.ones_like(r2)
    e = ca_px / max(rhd, 1e-9)
    scales = [(1.0 - e), 1.0, (1.0 + e)] if ca_px != 0.0 else [1.0]
    maps = []
    wm1, hm1 = np.float32(w - 1), np.float32(h - 1)
    wm2, hm2 = np.float32(max(w - 2, 0)), np.float32(max(h - 2, 0))
    for s in scales:
        sc = base * np.float32(s)
        sx = np.multiply(sc, dx)
        sx += np.float32(cx)
        np.clip(sx, 0.0, wm1, out=sx)
        sy = np.multiply(sc, dy)
        sy += np.float32(cy)
        np.clip(sy, 0.0, hm1, out=sy)
        fl_x = np.floor(sx)
        np.minimum(fl_x, wm2, out=fl_x)
        fl_y = np.floor(sy)
        np.minimum(fl_y, hm2, out=fl_y)
        sx -= fl_x                                                    # fractional parts (float32)
        sy -= fl_y
        fl_y *= np.float32(w)                                         # flat indices < 2^24: exact in float32
        fl_y += fl_x
        idx = fl_y.astype(np.intp)
        for a in (idx, sx, sy):
            a.flags.writeable = False
        maps.append((idx, sx, sy))
    return maps


class _BudgetCache:
    """Tiny LRU cache bounded by bytes rather than entries; oversized results are returned uncached."""

    def __init__(self, fn, size_of, budget_bytes):
        self._fn, self._size_of, self._budget = fn, size_of, budget_bytes
        self._items: OrderedDict = OrderedDict()

    def __call__(self, *key):
        hit = self._items.get(key)
        if hit is not None:
            self._items.move_to_end(key)
            return hit[0]
        value = self._fn(*key)
        size = self._size_of(value)
        if size <= self._budget:
            self._items[key] = (value, size)
            while sum(s for _, s in self._items.values()) > self._budget:
                self._items.popitem(last=False)
        return value

    def cache_clear(self):
        self._items.clear()


def _maps_bytes(maps) -> int:
    return sum(a.nbytes for m in maps for a in m)


_warp_maps = _BudgetCache(_build_warp_maps, _maps_bytes, _MAP_CACHE_BYTES)


def _remap_rows(out: np.ndarray, plane: np.ndarray, m, a0: int, a1: int) -> None:
    """Bilinear gather of output rows ``a0:a1`` of one (H, W) plane into ``out`` (rows, W).  The four taps are
    ``take`` calls on shifted views of the flat plane, so no ``idx + 1`` / ``idx + w`` index arrays are built."""
    h, w = plane.shape
    idx, fx, fy = (a[a0:a1] for a in m)
    f = plane.reshape(-1)
    a = f.take(idx, mode="clip")
    top = a
    if w > 1:
        top = f[1:].take(idx, mode="clip")
        top -= a
        top *= fx
        top += a                                # top row: lerp along x
    if h == 1:                                  # fy == 0 everywhere
        out[...] = top
        return
    bot = f[w:].take(idx, mode="clip")
    if w > 1:
        d = f[w + 1:].take(idx, mode="clip")
        d -= bot
        d *= fx
        d += bot
        bot = d                                 # bottom row: lerp along x
    bot -= top
    bot *= fy
    np.add(bot, top, out=out)                   # lerp along y


@functools.lru_cache(maxsize=3)
def _vignette_gain(h: int, w: int, v: float) -> np.ndarray:
    """Display-space multiplier (H, W); 1 at the centre, 1 - v (photometric) at the corners."""
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    dx = (np.arange(w, dtype=np.float32) - np.float32(cx))[None, :]
    dy = (np.arange(h, dtype=np.float32) - np.float32(cy))[:, None]
    t = (dx * dx + dy * dy) * np.float32(1.0 / (cx * cx + cy * cy + 1e-9))
    fall = np.float32(0.55) * t + np.float32(0.45) * t * t
    g = np.power(np.clip(1.0 - np.float32(v) * fall, 0.0, 1.0), np.float32(1.0 / _DISPLAY_GAMMA)).astype(np.float32)
    g.flags.writeable = False
    return g


def warm_up(height: int, width: int, params: LookParams | None = None, run_frame: bool = True) -> None:
    """Pay the one-off costs for a frame size up front (lens maps, vignette, tone table and, with
    ``run_frame``, the allocator / page-fault warm-up of one blank frame through the whole pipeline), so
    that the first real frame is as fast as the rest (~0.4 s -> ~0.18 s at 1280x720)."""
    p = params if params is not None else LookParams()
    h, w = int(height), int(width)
    if p.distortion_k1 != 0.0 or p.chromatic_aberration_px != 0.0:
        _warp_maps(h, w, float(p.distortion_k1), float(p.chromatic_aberration_px))
    if p.vignette > 0.0:
        _vignette_gain(h, w, float(p.vignette))
    _tone_lut(round(float(p.highlight_rolloff), 6), round(float(p.contrast), 6))
    if run_frame:
        apply(np.zeros((h, w, 3), np.uint8), np.full((h, w), 100.0, np.float32), p)


# ---------------------------------------------------------------------------
# Sensor / ISP
# ---------------------------------------------------------------------------

_SHARPEN_SIGMA = 0.9
_SHARPEN_LIMIT = 0.12


def _sharpen_slab(slab: np.ndarray, c0: int, c1: int, amount: float) -> np.ndarray:
    """Luma unsharp mask on the central rows ``c0:c1`` of ``slab`` (in place; the other rows only feed the blur).
    Returns the luma of the sharpened central rows."""
    y = _luma(slab)
    d = y - _blur(y, _SHARPEN_SIGMA)
    d *= np.float32(_SHARPEN_LIMIT) / (np.float32(_SHARPEN_LIMIT) + np.abs(d))   # soft limit: no hard halos
    d *= np.float32(amount)
    d = d[c0:c1]
    slab[:, c0:c1] += d[None]
    y = y[c0:c1]
    y += d                                                         # the same d went to every channel
    return y


class _Grain:
    """Zero-mean sensor noise: slightly correlated luma + soft chroma noise, strongest in the mid-tones.

    The random fields are drawn once per frame (so the result does not depend on the band size);
    ``add`` then applies rows ``y0:y1`` of it."""

    def __init__(self, h: int, w: int, amount: float, seed: int):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0x7FFFFFFFFFFFFFFF, 0x5EED5EED]))
        self.h, self.w, self.amount = h, w, amount
        self.z = rng.standard_normal((h, w), dtype=np.float32)       # luma white noise, softened per band
        hc, wc = (h + 1) // 2, (w + 1) // 2
        nc = rng.standard_normal((3, hc, wc), dtype=np.float32)
        nc -= nc.mean(axis=0, keepdims=True)
        nc *= np.float32(_GRAIN_CHROMA)
        self.ncx = _fit_axis(_up_axis_int(nc, 2, 2), w, 2)           # bilinear: soft colour blotches, no 2x2 blocks
        self.rw = _row_weights(hc, 2, h)

    def add(self, cen: np.ndarray, y0: int, y1: int, y: np.ndarray) -> None:
        """Add the noise of rows ``y0:y1`` to ``cen`` (3, rows, W); ``y`` is the display luma of those rows."""
        z0, z1 = max(0, y0 - 1), min(self.h, y1 + 1)
        nl = _blur3(self.z[z0:z1])[y0 - z0:y1 - z0]
        nl *= np.float32(1.0 / 0.375)                                # sqrt(sum k^2)^2 -> unit variance again
        nc = _rows_interp(self.ncx, self.rw, y0, y1)
        y = np.clip(y, 0.0, 1.0)
        amp = np.float32(1.0) - y
        amp *= y                                                     # y (1 - y): strongest in the mid-tones
        amp *= np.float32(1.8 * self.amount * _GRAIN_SCALE)
        amp += np.float32(0.55 * self.amount * _GRAIN_SCALE)
        for c in range(3):
            n = nc[c]
            n += nl
            n *= amp
            cen[c] += n


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_BAND_TARGET_PX = 40960        # a band of rows holds ~this many pixels per plane (~160 KB float32): stays in cache
_FORCE_BAND_ROWS = None        # tests: override the band height


def _band_rows(w: int) -> int:
    """Rows per band (a multiple of 4).  The pipeline runs band by band so that the elementwise passes work on
    cache-resident slabs instead of streaming the whole frame through memory once per operation."""
    if _FORCE_BAND_ROWS:
        return max(4, int(_FORCE_BAND_ROWS) // 4 * 4)
    return int(np.clip(_BAND_TARGET_PX // max(w, 1) // 4 * 4, 8, 64))


def _as_rgb_uint8(rgb) -> np.ndarray:
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError(f"rgb must be (H, W, 3) or (H, W, 4), got {a.shape}")
    a = a[..., :3]
    if a.dtype == np.uint8:
        return a
    if a.dtype.kind == "f":                      # float images are taken as [0, 1]
        return np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.clip(a, 0, 255).astype(np.uint8)


def _stage_linear(img, depth, p: LookParams, gains, bh: int):
    """Decode -> haze / dust -> lens softness, band by band.  Returns ``(lin, q)``: the linear image
    (3, H, W) and, when bloom is on, its 4x4 box mean ``q`` (3, H//4, W//4)."""
    h, w = img.shape[:2]
    luts = [(_DEC_TABLE * float(g)).astype(np.float32) for g in gains]
    atmo = _Atmosphere(depth, p, gains) if depth is not None and (p.haze_strength > 0.0 or p.dust_in_air > 0.0) else None
    soft = p.lens_blur_px > 0.05
    # rows of context a band needs: the lens blur reaches ``rb`` rows, and the softened transmittance one more
    rb = len(_gauss_kernel(float(p.lens_blur_px))) // 2 if soft else 0
    margin = rb + (1 if (soft and atmo is not None) else 0)
    want_q = p.bloom_strength > 0.0 and h // 4 >= 2 and w // 4 >= 2
    lin = np.empty((3, h, w), np.float32)
    q = np.empty((3, h // 4, w // 4), np.float32) if want_q else None
    for y0 in range(0, h, bh):
        y1 = min(h, y0 + bh)
        a0, a1 = max(0, y0 - margin), min(h, y1 + margin)
        slab = _decode_rows(img, luts, a0, a1)
        if atmo is not None:
            atmo.apply(slab, a0, a1, soft)
        if soft:
            slab = _blur(slab, float(p.lens_blur_px))
        cen = slab[:, y0 - a0:y1 - a0]
        lin[:, y0:y1] = cen
        if want_q:
            g = (y1 - y0) // 4
            if g:
                q[:, y0 // 4:y0 // 4 + g] = _down2(_down2(cen[:, :4 * g, :4 * (w // 4)]))
    return lin, q


def _stage_tone(lin, q, p: LookParams, bh: int) -> np.ndarray:
    """Bloom glow -> highlight shoulder + S-curve (one table) -> saturation, band by band; returns display values."""
    _, h, w = lin.shape
    glow_x = rw = None
    if q is not None:
        glow = _bloom_glow(q, float(p.bloom_strength), float(p.bloom_radius_px))
        glow_x = _fit_axis(_up_axis_int(glow, 4, 2), w, 2)           # horizontal x4 once; rows are interpolated per band
        rw = _row_weights(glow.shape[1], 4, h)
    disp = np.empty_like(lin)
    for y0 in range(0, h, bh):
        y1 = min(h, y0 + bh)
        x = lin[:, y0:y1]
        if glow_x is not None:
            x = x + _rows_interp(glow_x, rw, y0, y1)
        d = _tone(x, p.highlight_rolloff, p.contrast)
        if p.saturation != 1.0:
            d = _saturate(d, p.saturation)
        disp[:, y0:y1] = d
    return disp


def _stage_camera(disp, p: LookParams, seed: int, bh: int) -> np.ndarray:
    """Lens distortion + CA -> vignette -> sharpen -> sensor noise -> uint8, band by band."""
    _, h, w = disp.shape
    warp = p.distortion_k1 != 0.0 or p.chromatic_aberration_px != 0.0
    maps = _warp_maps(h, w, float(p.distortion_k1), float(p.chromatic_aberration_px)) if warp else None
    vig = _vignette_gain(h, w, float(p.vignette)) if p.vignette > 0.0 else None
    margin = len(_gauss_kernel(_SHARPEN_SIGMA)) // 2 if p.sharpen > 0.0 else 0
    grain = _Grain(h, w, float(p.grain), seed) if p.grain > 0.0 else None
    out = np.empty((h, w, 3), np.uint8)
    for y0 in range(0, h, bh):
        y1 = min(h, y0 + bh)
        a0, a1 = max(0, y0 - margin), min(h, y1 + margin)
        c0, c1 = y0 - a0, y1 - a0
        if warp:
            slab = np.empty((3, a1 - a0, w), np.float32)
            for c in range(3):
                _remap_rows(slab[c], disp[c], maps[c] if len(maps) == 3 else maps[0], a0, a1)
        else:
            slab = disp[:, a0:a1].copy()
        if vig is not None:
            slab *= vig[a0:a1][None]
        y = None
        if p.sharpen > 0.0:
            y = _sharpen_slab(slab, c0, c1, float(p.sharpen))
        cen = slab[:, c0:c1]
        if grain is not None:
            grain.add(cen, y0, y1, _luma(cen) if y is None else y)
        cen *= np.float32(255.0)
        cen += np.float32(0.5)
        np.clip(cen, 0.0, 255.0, out=cen)
        for c in range(3):
            out[y0:y1, :, c] = cen[c]
    return out


def apply(rgb: np.ndarray, depth_m: np.ndarray | None = None, params: LookParams = LookParams(),
          seed: int = 0) -> np.ndarray:
    """Turn a clean sRGB render into camera footage.

    rgb      (H, W, 3|4) uint8 sRGB (alpha ignored; float arrays are read as [0, 1])
    depth_m  (H, W) metres from the camera, or None (haze/dust are then skipped); inf/NaN/0 = sky
    seed     per-frame seed of the sensor noise (use the frame index for temporally varying grain)
    returns  (H, W, 3) uint8
    """
    p = params
    img = _as_rgb_uint8(rgb)
    h, w = img.shape[:2]
    depth = _prep_depth(depth_m, h, w)
    bh = _band_rows(w)
    # exposure and white balance are linear gains: folded into the decode table (and the airlight colours)
    gains = [max(float(p.exposure) * float(g), 0.0) for g in p.white_balance]
    lin, q = _stage_linear(img, depth, p, gains, bh)
    disp = _stage_tone(lin, q, p, bh)
    del lin
    return _stage_camera(disp, p, seed, bh)
