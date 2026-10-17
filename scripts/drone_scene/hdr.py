"""HDR sky tools: Radiance .hdr I/O, sun extraction and brightness calibration (numpy only).

Isaac renders the sky from an equirectangular (lat-long) HDR on a DomeLight.  A dome gives soft, noisy
sun shadows in real-time RTX, so the sun is cut out of the HDR (``find_sun`` + ``remove_sun``) and
replaced by a DistantLight with the same direction and irradiance.  ``calibrate`` then picks the dome
and sun intensities so that the scene is as bright as a reference sky that Isaac Lab is known to expose
well.  Runs on Isaac Sim's Python (numpy and the standard library only)::

    python scripts/drone_scene/hdr.py info sky.hdr [--reference ref.hdr] [--sun-elevation DEG]

Direction convention (the single one used by every function here and by the code that places lights)::

    image   u in [0, 1) runs left to right, v in [0, 1] runs top (zenith) to bottom (nadir);
            pixel (row i, column j) is sampled at its centre u = (j + .5) / W, v = (i + .5) / H
    world   Z-up, the zenith is +Z
    azimuth = 2 pi (u - 0.5), measured from +X towards +Y (counter-clockwise seen from above)
    elev.   = pi (0.5 - v)
    direction = (cos el cos az, cos el sin az, sin el)

so u = 0.5 looks along +X, u = 0.75 along +Y and u = 0 (= 1) along -X.  Seen from inside the sphere this
is the mirror image of a photographed panorama (which turns clockwise with growing u).

This is NOT the mapping of a stock USD DomeLight.  The UsdLux schema (following OpenEXR) has the
longitude fall as u grows; on a Z-up stage with poleAxis "scene" that puts u = 0.25 along +X, u = 0.5
along -Y and u = 0.75 along -X, i.e. azimuth_usd = -azimuth_here - 90 deg.  The two differ by a mirror,
so no rotateZ on the dome turns one into the other: to show an image on a DomeLight with the convention
above, flip it left-right and rotate the dome by +90 deg about Z (not verified against RTX; Isaac Sim
is not available where this was written).  Only horizon landmarks are affected, not the sun, whose
direction is set by the caller.

Radiance units: pixels are radiance L in arbitrary linear units, integrals are sums over pixels of
``L * solid_angle``.  ``SunInfo.irradiance`` is the irradiance at normal incidence, ``irradiance()`` the
cosine-weighted one.

Calibration assumption: Isaac's DistantLight ``intensity`` is treated as a (normalised) irradiance and
the DomeLight ``intensity`` as a multiplier of the HDR radiance, so with dome intensity ``D`` and sun
intensity ``S`` the horizontal illuminance of the scene is ``D * E_sky + S * cos(zenith)``.  The zenith
angle is that of the light as placed in the scene: pass it to ``calibrate`` (``sun_cos_zenith``) when it
is not where the HDR's sun was.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

__all__ = ["NUCLEUS_ROOT", "SunInfo", "read_hdr", "write_hdr", "direction_from_uv", "uv_from_direction",
           "solid_angle_map", "luminance", "irradiance", "find_sun", "remove_sun", "calibrate",
           "downsample", "fetch"]

NUCLEUS_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
# The sky (and DomeLight intensity) Isaac Lab's dexsuite task uses, so known to be exposed well.
REFERENCE_SKY = NUCLEUS_ROOT + "/Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
REFERENCE_INTENSITY = 750.0

Log = Callable[[str], None]

_LUMA = (0.2126, 0.7152, 0.0722)        # Rec.709
_CHUNK_ROWS = 256


# ---------------------------------------------------------------------------
# Radiance RGBE files
# ---------------------------------------------------------------------------

# exponent byte -> scale of the 8-bit mantissas (0 means a black pixel)
_EXP_SCALE = np.ldexp(1.0, np.arange(256) - 136).astype(np.float32)
_EXP_SCALE[0] = 0.0
_MIN_RUN = 4
_MAX_RGBE = np.float32(1.6e38)          # below 2**127, the largest value an exponent byte can hold


def _parse_header(data: bytes) -> tuple[int, int, int, bool, bool, float]:
    """(data offset, height, width, flip rows, flip columns, exposure) of a Radiance file."""
    pos = 0

    def line() -> str:
        nonlocal pos
        end = data.find(b"\n", pos)
        if end < 0:
            raise ValueError("truncated HDR header")
        s = data[pos:end].rstrip(b"\r").decode("latin-1")
        pos = end + 1
        return s

    if not line().startswith("#?"):
        raise ValueError("not a Radiance HDR file (missing '#?RADIANCE' / '#?RGBE' magic)")
    exposure = 1.0
    while True:
        s = line()
        if s == "":
            break
        key, _, val = s.partition("=")
        key = key.strip().upper()
        if key == "FORMAT" and val.strip() != "32-bit_rle_rgbe":
            raise ValueError(f"unsupported HDR FORMAT {val.strip()!r} (only 32-bit_rle_rgbe)")
        if key == "EXPOSURE":
            try:
                e = float(val)
            except ValueError:
                continue
            if e > 0:
                exposure *= e       # several EXPOSURE lines are cumulative
    res = line().split()
    if len(res) != 4 or res[0][1:].upper() not in ("Y", "X") or res[2][1:].upper() not in ("Y", "X"):
        raise ValueError(f"bad HDR resolution string {' '.join(res)!r}")
    if res[0][1:].upper() != "Y":
        raise ValueError("transposed HDR (+X/-X first) is not supported")
    h, w = int(res[1]), int(res[3])
    if h <= 0 or w <= 0:
        raise ValueError(f"bad HDR resolution string {' '.join(res)!r}")
    return pos, h, w, res[0][0] == "+", res[2][0] == "-", exposure


def _rle_channel(buf: bytes, pos: int, width: int) -> tuple[bytearray, int]:
    out = bytearray()
    n = 0
    while n < width:
        c = buf[pos]
        pos += 1
        if c > 128:
            c -= 128
            out += buf[pos:pos + 1] * c
            pos += 1
        else:
            if c == 0:
                raise ValueError("corrupt RLE scanline (zero-length literal)")
            out += buf[pos:pos + c]
            pos += c
        n += c
    if n != width:
        raise ValueError("corrupt RLE scanline (run crosses the scanline end)")
    return out, pos


def _truncated(buf: bytes, start: int) -> ValueError:
    """Error for pixel data that ends early; says so when it holds Radiance's old-style (1,1,1,n) run-length code."""
    px = np.frombuffer(buf, np.uint8, (len(buf) - start) // 4 * 4, start).reshape(-1, 4)
    if (px[:, :3] == 1).all(axis=1).any():
        return ValueError("truncated HDR pixel data (it holds old-style (1,1,1,n) run-length pixels, "
                          "which are not supported)")
    return ValueError("truncated HDR pixel data")


def _decode_scanlines(buf: bytes, pos: int, h: int, w: int) -> np.ndarray:
    """RGBE bytes as (h, 4, w) uint8 planes; scanlines are new-style RLE or flat, decided per line."""
    n, start = len(buf), pos
    # A header must not make us allocate more than the file can possibly fill: a scanline takes at least
    # its RLE minimum (4 header bytes + one 2-byte run per 127 pixels and channel), or 4 bytes a pixel if flat.
    min_line = 4 + 8 * -(-w // 127) if 8 <= w <= 0x7FFF else 4 * w
    if h * min_line > n - pos:
        raise _truncated(buf, start)
    planes = np.empty((h, 4, w), np.uint8)
    try:
        for y in range(h):
            if 8 <= w <= 0x7FFF and pos + 4 <= n and buf[pos] == 2 and buf[pos + 1] == 2 and not buf[pos + 2] & 0x80:
                if ((buf[pos + 2] << 8) | buf[pos + 3]) != w:
                    raise ValueError("RLE scanline length does not match the image width")
                pos += 4
                for c in range(4):
                    chan, pos = _rle_channel(buf, pos, w)
                    planes[y, c] = np.frombuffer(chan, np.uint8)
            else:
                if pos + 4 * w > n:
                    raise IndexError
                planes[y] = np.frombuffer(buf, np.uint8, 4 * w, pos).reshape(w, 4).T
                pos += 4 * w
    except IndexError:
        raise _truncated(buf, start) from None
    return planes


def read_hdr(path: str) -> np.ndarray:
    """Read a Radiance RGBE ``.hdr`` as float32 (H, W, 3), row 0 = top.

    Handles ``#?RADIANCE`` / ``#?RGBE``, ``FORMAT=32-bit_rle_rgbe``, new-style RLE and flat scanlines and
    all four ``+-Y H +-X W`` orientations.  ``EXPOSURE=`` (cumulative) is divided out, as the Radiance
    format defines it.  Old-style (1,1,1,n) run-length pixels are not supported.
    """
    with open(path, "rb") as f:
        data = f.read()
    pos, h, w, flip_rows, flip_cols, exposure = _parse_header(data)
    planes = _decode_scanlines(data, pos, h, w)
    out = planes[:, :3, :].transpose(0, 2, 1) * _EXP_SCALE[planes[:, 3, :]][..., None]
    if flip_rows:
        out = out[::-1]
    if flip_cols:
        out = out[:, ::-1]
    out = np.ascontiguousarray(out, dtype=np.float32)
    if exposure != 1.0:
        out /= np.float32(exposure)
    return out


def _to_rgbe(img: np.ndarray) -> np.ndarray:
    """float (h, w, 3) -> (h, w, 4) uint8, mantissas rounded to nearest (decoded as m * 2**(e-136))."""
    a = np.clip(np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=_MAX_RGBE), 0.0, _MAX_RGBE)
    _, e = np.frexp(a.max(axis=-1))
    e = e.astype(np.int32)
    m = np.rint(np.ldexp(a, (8 - e)[..., None]))
    bump = m.max(axis=-1) > 255                     # rounded up to 256: take the next exponent
    if bump.any():
        e = e + bump
        m = np.rint(np.ldexp(a, (8 - e)[..., None]))
    out = np.empty(a.shape[:2] + (4,), np.uint8)
    out[..., :3] = np.minimum(m, 255)
    eb = np.clip(e + 128, 0, 255)
    dead = (a.max(axis=-1) <= 0) | (e + 128 < 1)    # zero, or too small for the exponent byte
    out[..., 3] = np.where(dead, 0, eb)
    out[dead, :3] = 0
    return out


def _emit_literals(out: bytearray, b: bytes, start: int, stop: int) -> None:
    for k in range(start, stop, 128):
        n = min(128, stop - k)
        out.append(n)
        out += b[k:k + n]


def _rle_encode_channel(ch: np.ndarray) -> bytearray:
    b = ch.tobytes()
    n = len(b)
    change = np.flatnonzero(ch[1:] != ch[:-1]) + 1
    starts = np.concatenate(([0], change))
    lens = np.diff(np.concatenate((starts, [n])))
    out = bytearray()
    lit = 0
    for k in np.flatnonzero(lens >= _MIN_RUN).tolist():     # only the long runs need a Python loop
        s, run = int(starts[k]), int(lens[k])
        _emit_literals(out, b, lit, s)
        lit = s + run
        while run > 0:
            c = min(127, run)
            out.append(128 + c)
            out.append(b[s])
            run -= c
    _emit_literals(out, b, lit, n)
    return out


def write_hdr(path: str, img: np.ndarray, *, rle: bool = True) -> None:
    """Write float (H, W, 3) as a Radiance RGBE file (new-style RLE, or flat if ``rle`` is False).

    Mantissas are rounded to nearest, so the round-trip error is at most half an RGBE step: 2**-8 of the
    brightest channel of each pixel (smaller channels share its step).  Negative and NaN values are
    written as 0.  RLE is only valid for 8 <= width <= 32767; other widths are written flat.
    """
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got {img.shape}")
    h, w = img.shape[:2]
    use_rle = rle and 8 <= w <= 0x7FFF
    with open(path, "wb") as f:
        f.write(f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n".encode("ascii"))
        for r0 in range(0, h, _CHUNK_ROWS):
            rgbe = _to_rgbe(img[r0:r0 + _CHUNK_ROWS])
            if not use_rle:
                f.write(rgbe.tobytes())
                continue
            planes = np.ascontiguousarray(rgbe.transpose(0, 2, 1))
            buf = bytearray()
            for row in planes:
                buf += bytes((2, 2, w >> 8, w & 0xFF))
                for c in range(4):
                    buf += _rle_encode_channel(row[c])
            f.write(buf)


# ---------------------------------------------------------------------------
# Geometry of the lat-long map
# ---------------------------------------------------------------------------

def direction_from_uv(u, v) -> np.ndarray:
    """Unit direction(s) (..., 3) for image coordinates ``u``, ``v`` (broadcast together); see the module docstring."""
    az = (np.asarray(u, np.float64) - 0.5) * (2.0 * math.pi)
    el = (0.5 - np.asarray(v, np.float64)) * math.pi
    ce = np.cos(el)
    x, y, z = np.broadcast_arrays(ce * np.cos(az), ce * np.sin(az), np.sin(el))
    return np.stack((x, y, z), axis=-1)


def uv_from_direction(d) -> tuple[np.ndarray, np.ndarray]:
    """Exact inverse of ``direction_from_uv``: (u, v) of direction(s) ``d`` (..., 3), not necessarily normalised.

    u is in [0, 1); at the poles u is arbitrary (0.5 for a direction exactly on the axis).
    """
    d = np.asarray(d, np.float64)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    u = 0.5 + np.arctan2(y, x) / (2.0 * math.pi)
    u = np.where(u >= 1.0, u - 1.0, u)
    v = 0.5 - np.arctan2(z, np.hypot(x, y)) / math.pi
    return u[()], v[()]


def _row_elevation(h: int) -> np.ndarray:
    return math.pi * (0.5 - (np.arange(h) + 0.5) / h)


def _col_azimuth(w: int) -> np.ndarray:
    return 2.0 * math.pi * ((np.arange(w) + 0.5) / w - 0.5)


def _row_solid_angle(h: int, w: int) -> np.ndarray:
    """Solid angle of one pixel of each row: the exact band area of the cell, so the sum over the map is 4 pi."""
    edges = np.cos(math.pi * np.arange(h + 1) / h)
    return (2.0 * math.pi / w) * (edges[:-1] - edges[1:])


def solid_angle_map(h: int, w: int) -> np.ndarray:
    """Per-pixel solid angle (steradians), float64 (h, w); sums to 4 pi."""
    return np.repeat(_row_solid_angle(h, w)[:, None], w, axis=1)


def luminance(rgb) -> np.ndarray:
    """Rec.709 luminance of RGB along the last axis."""
    rgb = np.asarray(rgb)
    if not np.issubdtype(rgb.dtype, np.floating):
        rgb = rgb.astype(np.float64)
    return rgb[..., 0] * _LUMA[0] + rgb[..., 1] * _LUMA[1] + rgb[..., 2] * _LUMA[2]


def _as_rgb(img) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f"expected an (H, W, 3) image, got {a.shape}")
    return a[..., :3]


def _row_chunks(h: int) -> Iterator[tuple[int, int]]:
    for r0 in range(0, h, _CHUNK_ROWS):
        yield r0, min(h, r0 + _CHUNK_ROWS)


def irradiance(img: np.ndarray, normal=(0.0, 0.0, 1.0), *, mask: np.ndarray | None = None) -> np.ndarray:
    """RGB irradiance on a surface with ``normal``: sum of L * max(0, n . w) * d_omega over the pixels.

    With ``mask`` (bool, H x W) only the pixels where it is True contribute, e.g. ``mask=sun.mask`` for
    the sun alone or ``mask=~sun.mask`` for the sky without it.  The default normal is up (+Z), so the
    luminance of the result is the horizontal illuminance.
    """
    img = _as_rgb(img)
    h, w = img.shape[:2]
    n = np.asarray(normal, np.float64).reshape(3)
    norm = float(np.linalg.norm(n))
    if norm == 0.0:
        raise ValueError("normal must be non-zero")
    n = n / norm
    if mask is not None:
        mask = np.asarray(mask, bool)
        if mask.shape != (h, w):
            raise ValueError(f"mask shape {mask.shape} does not match the image {(h, w)}")
    el = _row_elevation(h)
    ce, se = np.cos(el), np.sin(el)
    a_col = n[0] * np.cos(_col_azimuth(w)) + n[1] * np.sin(_col_azimuth(w))
    dw = _row_solid_angle(h, w)
    out = np.zeros(3)
    for r0, r1 in _row_chunks(h):
        wt = np.maximum(ce[r0:r1, None] * a_col[None, :] + (n[2] * se[r0:r1])[:, None], 0.0) * dw[r0:r1, None]
        if mask is not None:
            wt *= mask[r0:r1]
        out += np.einsum("ijc,ij->c", img[r0:r1], wt, dtype=np.float64)
    return out


def downsample(img: np.ndarray, factor: int) -> np.ndarray:
    """Box-filter ``img`` (H, W[, C]) by an integer ``factor``; a ragged edge (H or W % factor) is dropped."""
    f = int(factor)
    if f < 1:
        raise ValueError("factor must be >= 1")
    img = np.asarray(img)
    dtype = img.dtype if np.issubdtype(img.dtype, np.floating) else np.float32
    if f == 1:
        return img.astype(dtype, copy=True)
    h, w = img.shape[:2]
    hh, ww = h // f, w // f
    if hh == 0 or ww == 0:
        raise ValueError(f"cannot downsample {h}x{w} by {f}")
    blocks = img[:hh * f, :ww * f].reshape((hh, f, ww, f) + img.shape[2:])
    return blocks.mean(axis=(1, 3), dtype=dtype)


# ---------------------------------------------------------------------------
# Sun
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class SunInfo:
    """The sun cut out of an HDR sky (see ``find_sun``)."""

    direction: np.ndarray       # unit vector from the ground towards the sun
    elevation_deg: float        # above the horizon
    azimuth_deg: float          # from +X towards +Y, in (-180, 180]
    irradiance: np.ndarray      # RGB, sum of L * d_omega over the mask (normal incidence)
    solid_angle: float          # of the mask, sr
    mask: np.ndarray            # bool (H, W)
    peak_luminance: float = 0.0     # brightest pixel
    sky_luminance: float = 0.0      # median luminance of the upper hemisphere


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    cw = np.cumsum(weights[order], dtype=np.float64)
    return float(values[order][min(int(np.searchsorted(cw, 0.5 * cw[-1])), len(order) - 1)])


def _upper_sky_median(lum: np.ndarray) -> float:
    """Solid-angle weighted median luminance of the upper hemisphere (subsampled to ~0.5 M pixels)."""
    h, w = lum.shape
    rows = np.flatnonzero(_row_elevation(h) >= 0.0)
    step_r = max(1, len(rows) // 512)
    step_c = max(1, w // 1024)
    rows = rows[::step_r]
    vals = lum[rows][:, ::step_c]
    wts = np.repeat(_row_solid_angle(h, w)[rows][:, None], vals.shape[1], axis=1)
    return _weighted_median(vals.ravel(), wts.ravel())


def _box_mean(a: np.ndarray, ry: int, rx: int) -> np.ndarray:
    """Mean over the (2 ry + 1) x (2 rx + 1) box around each pixel, wrapping horizontally and clamping vertically."""
    h, w = a.shape
    rx = min(rx, (w - 1) // 2)
    a = np.asarray(a, np.float32)
    s = a.copy()
    for k in range(1, rx + 1):
        s[:, k:] += a[:, :-k]
        s[:, :k] += a[:, w - k:]
        s[:, :-k] += a[:, k:]
        s[:, w - k:] += a[:, :k]
    p = np.pad(s, ((ry, ry), (0, 0)), mode="edge")
    out = p[:h].copy()
    for k in range(1, 2 * ry + 1):
        out += p[k:k + h]
    return out / np.float32((2 * rx + 1) * (2 * ry + 1))


def _window(h: int, w: int, i0: int, j0: int, radius_deg: float, pad_y: int = 0, pad_x: int = 0):
    """(r0, r1, cols, wrap): rows, wrapped column indices and a wrap flag of a box around pixel (i0, j0).

    The box contains every pixel within ``radius_deg`` of the centre direction, plus ``pad`` pixels.
    ``wrap`` is True when the columns cover the full width (then the box edges wrap around).
    """
    ry = int(math.ceil(radius_deg / 180.0 * h)) + pad_y
    r0, r1 = max(0, i0 - ry), min(h, i0 + ry + 1)
    el0 = abs(float(_row_elevation(h)[i0]))
    rho = math.radians(radius_deg)
    if el0 + rho >= 0.5 * math.pi:
        return r0, r1, np.arange(w), True
    daz = math.asin(min(1.0, math.sin(rho) / math.cos(el0)))
    rx = int(math.ceil(daz / (2.0 * math.pi) * w)) + pad_x
    if 2 * rx + 1 >= w:
        return r0, r1, np.arange(w), True
    return r0, r1, np.arange(j0 - rx, j0 + rx + 1) % w, False


def _shift_rows(m: np.ndarray, dy: int) -> np.ndarray:
    out = np.zeros_like(m)
    if dy > 0:
        out[dy:] = m[:-dy]
    elif dy < 0:
        out[:dy] = m[-dy:]
    else:
        out[:] = m
    return out


def _shift_cols(m: np.ndarray, dx: int, wrap: bool) -> np.ndarray:
    if wrap:
        return np.roll(m, dx, axis=1)
    out = np.zeros_like(m)
    if dx > 0:
        out[:, dx:] = m[:, :-dx]
    elif dx < 0:
        out[:, :dx] = m[:, -dx:]
    else:
        out[:] = m
    return out


def _dilate_ellipse(m: np.ndarray, ry: int, rx: int, wrap: bool) -> np.ndarray:
    out = m.copy()
    for dy in range(-ry, ry + 1):
        half = int(rx * math.sqrt(max(0.0, 1.0 - (dy / (ry + 0.5)) ** 2)))
        rows = _shift_rows(m, dy)
        for dx in range(-half, half + 1):
            out |= _shift_cols(rows, dx, wrap)
    return out


def _grow_component(cand: np.ndarray, seed: tuple[int, int], wrap: bool) -> np.ndarray:
    """8-connected component of ``cand`` containing ``seed`` (geodesic dilation)."""
    cur = np.zeros_like(cand)
    cur[seed] = True
    count = 1
    while True:
        wide = cur | _shift_cols(cur, 1, wrap) | _shift_cols(cur, -1, wrap)
        nxt = (wide | _shift_rows(wide, 1) | _shift_rows(wide, -1)) & cand
        c = int(nxt.sum())
        if c == count:
            return nxt
        cur, count = nxt, c


def _window_directions(h: int, w: int, r0: int, r1: int, cols: np.ndarray) -> np.ndarray:
    el = _row_elevation(h)[r0:r1, None]
    az = _col_azimuth(w)[cols][None, :]
    ce = np.cos(el)
    return np.stack(np.broadcast_arrays(ce * np.cos(az), ce * np.sin(az), np.sin(el)), axis=-1)


def find_sun(img: np.ndarray, *, min_peak_ratio: float = 50.0, grow_ratio: float = 20.0,
             dilate_deg: float = 0.5, max_radius_deg: float = 12.0,
             sky_annulus_deg: tuple[float, float] = (6.0, 25.0), smooth_deg: float = 0.5) -> SunInfo | None:
    """Find the sun in a lat-long HDR (finite values), or None if there is no clear one.

    The peak is the brightest pixel inside the box around the maximum of the box-smoothed luminance above
    the horizon.  The box reaches ``smooth_deg`` (a radius, 3x3 pixels at the least) around each pixel, which
    is wider than the sun, so a lone hot pixel is not mistaken for it unless it is brighter than the
    sun's peak by about the sun's size in pixels.  There is no sun if the peak
    luminance is below ``min_peak_ratio`` times the median luminance of the upper hemisphere.  The sun
    mask is the connected region around the peak whose luminance exceeds ``grow_ratio`` times the local
    sky level (median in the ``sky_annulus_deg`` ring around the peak, so the sun and its inner glow do
    not bias it), capped to ``max_radius_deg``, then dilated by ``dilate_deg`` to take in the halo.  The
    direction is the centroid of the mask weighted by luminance x solid angle; the irradiance is the sum
    of L x solid angle over the mask.  None is also returned if the region around the peak cannot be
    isolated within ``max_radius_deg`` (a bright wall or ground rather than a sun).
    """
    img = _as_rgb(img)
    h, w = img.shape[:2]
    lum = luminance(img).astype(np.float32)
    sky_med = _upper_sky_median(lum)

    el_rows = _row_elevation(h)
    r_hi = int(np.count_nonzero(el_rows > 0.0))             # the peak is searched above the horizon only
    if r_hi == 0:
        return None
    # Rank pixels by the mean over a box a little wider than the sun, so a lone hot pixel (a firefly, a
    # glint) cannot outrank the real sun, which has far more pixels; at low resolution the box is 3x3.
    sy = max(1, int(round(smooth_deg / 180.0 * h)))
    sx = max(1, int(round(smooth_deg / 360.0 * w)))
    sm = _box_mean(lum[:min(h, r_hi + sy)], sy, sx)[:r_hi]
    i, j = np.unravel_index(int(np.argmax(sm)), sm.shape)
    # the true peak pixel inside the box around the smoothed maximum
    ys, xs = np.arange(max(0, i - sy), min(h, i + sy + 1)), np.arange(j - sx, j + sx + 1) % w
    sub = lum[ys][:, xs]
    k = np.unravel_index(int(np.argmax(sub)), sub.shape)
    i0, j0 = int(ys[k[0]]), int(xs[k[1]])
    peak = float(lum[i0, j0])
    if not peak > 0.0 or peak < min_peak_ratio * sky_med:
        return None

    d0 = direction_from_uv((j0 + 0.5) / w, (i0 + 0.5) / h)
    dw_rows = _row_solid_angle(h, w)

    # local sky level from a ring around the peak
    r0, r1, cols, _ = _window(h, w, i0, j0, sky_annulus_deg[1])
    ang = np.degrees(np.arccos(np.clip(_window_directions(h, w, r0, r1, cols) @ d0, -1.0, 1.0)))
    ring = (ang >= sky_annulus_deg[0]) & (ang <= sky_annulus_deg[1])
    if ring.any():
        wts = np.broadcast_to(dw_rows[r0:r1, None], ring.shape)
        sky_local = _weighted_median(lum[r0:r1][:, cols][ring], wts[ring])
    else:                                     # tiny image: the ring is narrower than a pixel
        sky_local = sky_med
    thresh = min(max(grow_ratio * sky_local, 1e-4 * peak), 0.5 * peak)

    ry = max(1, int(round(dilate_deg / 180.0 * h)))
    cos_el = max(math.cos(float(el_rows[i0])), 0.05)
    rx = max(1, int(round(dilate_deg / 360.0 * w / cos_el)))
    # Grow in windows of increasing size (the sun is nearly always small, and near the zenith a window
    # spans every column).  A region that still reaches the largest window edge means the threshold lies
    # inside the sky glow, so it is doubled.
    radii = [r for r in (3.0, 6.0) if r < max_radius_deg] + [max_radius_deg]
    for attempt in range(len(radii) + 6):
        r0, r1, cols, wrap = _window(h, w, i0, j0, radii[min(attempt, len(radii) - 1)], pad_y=ry + 2, pad_x=rx + 2)
        seed = (i0 - r0, int(np.flatnonzero(cols == j0)[0]))
        comp = _grow_component(lum[r0:r1][:, cols] > thresh, seed, wrap)
        edge = ((r0 > 0 and comp[0].any()) or (r1 < h and comp[-1].any())
                or (not wrap and (comp[:, 0].any() or comp[:, -1].any())))
        if not edge:
            break
        if attempt >= len(radii) - 1:
            thresh *= 2.0
            if thresh >= peak:
                return None
    else:
        return None                           # no compact sun: the peak is part of something large and bright
    comp = _dilate_ellipse(comp, ry, rx, wrap)

    rr, cc = np.nonzero(comp)
    rows_abs, cols_abs = rr + r0, cols[cc]
    mask = np.zeros((h, w), bool)
    mask[rows_abs, cols_abs] = True
    dw = dw_rows[rows_abs]
    dirs = direction_from_uv((cols_abs + 0.5) / w, (rows_abs + 0.5) / h)
    weight = lum[rows_abs, cols_abs].astype(np.float64) * dw
    d = (dirs * weight[:, None]).sum(axis=0)
    if not np.linalg.norm(d) > 0.0:
        d = (dirs * dw[:, None]).sum(axis=0)
    d = d / np.linalg.norm(d)
    return SunInfo(direction=d,
                   elevation_deg=math.degrees(math.asin(float(np.clip(d[2], -1.0, 1.0)))),
                   azimuth_deg=math.degrees(math.atan2(d[1], d[0])),
                   irradiance=(img[rows_abs, cols_abs].astype(np.float64) * dw[:, None]).sum(axis=0),
                   solid_angle=float(dw.sum()), mask=mask, peak_luminance=peak, sky_luminance=sky_med)


def _mask_crop(mask: np.ndarray, pad: int) -> tuple[int, int, np.ndarray, bool]:
    """Smallest box (rows r0:r1, wrapped columns) around ``mask`` plus ``pad``; the flag says it spans the full width."""
    h, w = mask.shape
    rows = np.flatnonzero(mask.any(axis=1))
    r0, r1 = max(0, int(rows[0]) - pad), min(h, int(rows[-1]) + 1 + pad)
    occ = np.flatnonzero(mask.any(axis=0))
    gaps = np.diff(np.concatenate((occ, [occ[0] + w])))
    g = int(np.argmax(gaps))                                      # the largest empty arc is outside the box
    start, length = int(occ[(g + 1) % len(occ)]), w - int(gaps[g]) + 1
    if length + 2 * pad >= w:
        return r0, r1, np.arange(w), True
    return r0, r1, np.arange(start - pad, start + length + pad) % w, False


def _neighbour_sum(v: np.ndarray, wrap: bool) -> np.ndarray:
    s = np.zeros_like(v)
    s[1:] += v[:-1]
    s[:-1] += v[1:]
    if wrap:
        s += np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1)
    else:
        s[:, 1:] += v[:, :-1]
        s[:, :-1] += v[:, 1:]
    return s


def _harmonic_fill(x: np.ndarray, unknown: np.ndarray, wrap: bool) -> np.ndarray:
    """Solve the discrete Laplace equation on ``unknown`` with ``x`` outside it as boundary (conjugate gradients)."""
    known = ~unknown
    deg = _neighbour_sum(np.ones(x.shape), wrap)
    b = np.where(unknown, _neighbour_sum(np.where(known, x, 0.0), wrap), 0.0)

    def apply(v: np.ndarray) -> np.ndarray:
        return np.where(unknown, deg * v - _neighbour_sum(v, wrap), 0.0)

    v = np.where(unknown, b / np.maximum(deg, 1.0), 0.0)         # start from the mean of the known neighbours
    r = b - apply(v)
    p = r.copy()
    rs = float((r * r).sum())
    tol = 1e-12 * max(float((b * b).sum()), 1e-30)
    for _ in range(4 * sum(x.shape) + 100):
        if rs <= tol:
            break
        ap = apply(p)
        alpha = rs / float((p * ap).sum())
        v += alpha * p
        r -= alpha * ap
        rs_new = float((r * r).sum())
        p = r + (rs_new / rs) * p
        rs = rs_new
    return np.where(unknown, v, x)


def remove_sun(img: np.ndarray, sun: SunInfo) -> np.ndarray:
    """Copy of ``img`` with the sun mask filled with plausible sky.

    The fill is the smooth (harmonic) interpolation of the log radiance from the pixels around the mask,
    per channel: it is continuous with the surroundings (no seam), cannot exceed the brightest pixel of the
    ring and never goes negative.  ``sun`` must come from an image of the same size.  Channels beyond the
    first three (alpha) are copied unchanged.
    """
    img = np.asarray(img)
    _as_rgb(img)
    out = np.array(img, dtype=img.dtype if np.issubdtype(img.dtype, np.floating) else np.float32)
    mask = np.asarray(sun.mask, bool)
    if mask.shape != out.shape[:2]:
        raise ValueError(f"sun.mask shape {mask.shape} does not match the image {out.shape[:2]}")
    if not mask.any():
        return out
    r0, r1, cols, wrap = _mask_crop(mask, pad=2)
    crop = out[r0:r1][:, cols, :3].astype(np.float64)
    m = mask[r0:r1][:, cols]
    floor = 1e-6 * max(float(crop[~m].max()) if (~m).any() else 1.0, 1e-30)
    logs = np.log(np.maximum(crop, floor))
    rr, cc = np.nonzero(m)
    for c in range(3):
        filled = _harmonic_fill(logs[..., c], m, wrap)
        out[rr + r0, cols[cc], c] = np.exp(filled[rr, cc])
    return out


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(sky_img: np.ndarray, sun: SunInfo | None, reference_img: np.ndarray,
              reference_intensity: float = REFERENCE_INTENSITY, *, sun_cos_zenith: float | None = None) -> dict:
    """Dome and sun intensities that make ``sky_img`` as bright as ``reference_img`` at ``reference_intensity``.

    With E_ref = horizontal illuminance (luminance of ``irradiance``) of the reference HDR, E_sky that of
    ``remove_sun(sky_img, sun)`` (the image the dome will actually show; ``sky_img`` may already be sun-free)
    and S the luminance of ``sun.irradiance``::

        (a) dome * E_sky + sun_intensity * cos(zenith) = reference_intensity * E_ref
        (b) sun_intensity = dome * S                   (the sun/sky ratio of the HDR is kept)

    Isaac's DistantLight intensity is thus treated as a normalised irradiance and the dome intensity as a
    radiance multiplier.  ``cos(zenith)`` is that of the light as placed in the scene: by default the HDR's
    own sun (``sun.direction[2]``); pass ``sun_cos_zenith`` (the sine of the placed light's elevation)
    when the DistantLight sits elsewhere, otherwise the scene misses the target by the cosine ratio.
    Without a sun (None) or with the light below the horizon, the dome alone carries the brightness and
    ``sun_intensity`` is 0.  Returns ``dome_intensity`` and ``sun_intensity`` plus the numbers they came
    from: ``sky_illuminance`` (E_sky), ``reference_illuminance`` (E_ref), ``sun_illuminance`` (S),
    ``sun_cos_zenith`` (the one used in (a), 0 if the light is off) and ``sun_color`` (sun RGB scaled to
    luminance 1).
    """
    e_ref = float(luminance(irradiance(reference_img)))
    if not e_ref > 0.0:
        raise ValueError("reference image has no horizontal illuminance")
    e_sky = float(luminance(irradiance(sky_img if sun is None else remove_sun(sky_img, sun))))
    s_lum, cos_z, color = 0.0, 0.0, (1.0, 1.0, 1.0)
    if sun is not None:
        c = float(sun.direction[2]) if sun_cos_zenith is None else float(sun_cos_zenith)
        if not math.isfinite(c):
            raise ValueError(f"sun_cos_zenith must be finite, got {sun_cos_zenith!r}")
        if c > 0.0:
            s_lum = float(luminance(sun.irradiance))
            cos_z = min(c, 1.0)
            color = tuple(float(x) for x in np.asarray(sun.irradiance) / max(s_lum, 1e-30))
    denom = e_sky + s_lum * cos_z
    if not denom > 0.0:
        raise ValueError("sky image has no horizontal illuminance")
    dome = reference_intensity * e_ref / denom
    return {"dome_intensity": dome, "sun_intensity": dome * s_lum,
            "sky_illuminance": e_sky, "reference_illuminance": e_ref, "sun_illuminance": s_lum,
            "sun_cos_zenith": cos_z, "sun_color": color}


# ---------------------------------------------------------------------------
# Download + CLI
# ---------------------------------------------------------------------------

def fetch(url: str, cache_dir: str, *, log: Log | None = None) -> str:
    """Download ``url`` into ``cache_dir`` (once) and return the local path.

    The cached file is named ``<hash of the url>_<basename>``, so two URLs with the same file name (the
    server has several) do not share an entry.  A download that ends before ``Content-Length`` bytes
    raises ``OSError`` and leaves nothing in the cache.
    """
    os.makedirs(cache_dir, exist_ok=True)
    name = os.path.basename(url.split("?")[0]) or "download"
    path = os.path.join(cache_dir, f"{hashlib.sha1(url.encode()).hexdigest()[:8]}_{name}")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    if log:
        log(f"downloading {url}")
    tmp = f"{path}.{os.getpid()}.part"
    try:
        got = 0
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            expected = (r.headers.get("Content-Length") or "").strip()
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                f.write(block)
                got += len(block)
        if expected.isdigit() and got != int(expected):     # http.client does not raise on a short body
            raise OSError(f"incomplete download of {url}: {got} of {expected} bytes")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def _local(path_or_url: str, cache_dir: str) -> str:
    if path_or_url.startswith(("http://", "https://")):
        return fetch(path_or_url, cache_dir, log=lambda s: print(s, file=sys.stderr))
    return path_or_url


def _load(path_or_url: str, cache_dir: str) -> np.ndarray:
    try:
        return read_hdr(_local(path_or_url, cache_dir))
    except (OSError, ValueError) as e:
        raise SystemExit(f"error: cannot read {path_or_url}: {e}") from None


def _cmd_info(args: argparse.Namespace) -> int:
    img = _load(args.file, args.cache)
    h, w = img.shape[:2]
    lum = luminance(img)
    e_h = luminance(irradiance(img))
    print(f"{args.file}: {w}x{h}, luminance min {lum.min():.4g} median {np.median(lum):.4g} max {lum.max():.4g}")
    print(f"horizontal illuminance (whole HDR): {float(e_h):.5g}")
    sun = find_sun(img)
    if sun is None:
        print("sun: none (no clear peak)")
    else:
        s_lum = float(luminance(sun.irradiance))
        print(f"sun: direction ({sun.direction[0]:.4f}, {sun.direction[1]:.4f}, {sun.direction[2]:.4f}), "
              f"elevation {sun.elevation_deg:.2f} deg, azimuth {sun.azimuth_deg:.2f} deg")
        print(f"     peak luminance {sun.peak_luminance:.4g} (sky median {sun.sky_luminance:.4g}), "
              f"{int(sun.mask.sum())} px, {sun.solid_angle:.4g} sr")
        print(f"     irradiance rgb ({sun.irradiance[0]:.4g}, {sun.irradiance[1]:.4g}, {sun.irradiance[2]:.4g}), "
              f"luminance {s_lum:.4g}")
    ref = _load(args.reference, args.cache)
    cos_z = None if args.sun_elevation is None else math.sin(math.radians(args.sun_elevation))
    cal = calibrate(img, sun, ref, args.reference_intensity, sun_cos_zenith=cos_z)
    sun_over_sky = cal["sun_illuminance"] * cal["sun_cos_zenith"] / cal["sky_illuminance"]
    print(f"calibration vs {os.path.basename(args.reference)} at {args.reference_intensity:g}: "
          f"E_ref {cal['reference_illuminance']:.5g}, E_sky (sunless) {cal['sky_illuminance']:.5g}, "
          f"sun {cal['sun_illuminance']:.5g} x cos(zenith) {cal['sun_cos_zenith']:.4f} "
          f"(sun/sky horizontal {sun_over_sky:.3g})")
    print(f"     dome_intensity {cal['dome_intensity']:.6g}  sun_intensity {cal['sun_intensity']:.6g}  "
          f"sun_color ({cal['sun_color'][0]:.3f}, {cal['sun_color'][1]:.3f}, {cal['sun_color'][2]:.3f})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="HDR sky tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    info = sub.add_parser("info", help="print the sun and the calibration against a reference sky")
    info.add_argument("file", help="path or URL of a Radiance .hdr")
    info.add_argument("--reference", default=REFERENCE_SKY, help="reference sky (path or URL); default: Isaac Lab's dexsuite sky")
    info.add_argument("--reference-intensity", type=float, default=REFERENCE_INTENSITY)
    info.add_argument("--sun-elevation", type=float, default=None, metavar="DEG",
                      help="elevation the DistantLight will have in the scene (default: where the HDR's sun is)")
    info.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "drone_scene_hdr"), help="download cache")
    info.set_defaults(func=_cmd_info)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
