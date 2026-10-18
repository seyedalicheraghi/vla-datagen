"""What a LUCID camera records, simulated on the GPU from Isaac Sim's linear HDR render (torch).

Isaac Sim renders an ideal pinhole image in linear radiance (Replicator's ``HdrColor``, before any tone
mapping), a little wider than the real lens (``lucid.Rig.render_plan``).  ``LucidSensor.process`` turns one such
frame into the 8-bit image the camera would deliver, in the order light goes through a real camera:

1. lens: barrel distortion (OpenCV plumb_bob, the rig's ``D``), lateral colour (red and blue magnified a little
   differently), relative illumination (vignetting) and a small Gaussian blur (lens + pixel aperture);
2. exposure: auto exposure like the camera's ``ExposureAuto = Continuous`` - it meters the clipped raw signal
   and moves a fraction of the way to its target each frame - or a fixed exposure time / gain;
3. sensor: a Sony Pregius global-shutter pixel array behind an RGGB colour filter: photo-electrons with photon
   (Poisson) noise and per-pixel gain non-uniformity (PRNU), the full well, read noise, analog gain and a
   12-bit ADC; ``raw`` is that 12-bit Bayer frame (LUCID's BayerRG12);
4. image pipeline: demosaicing (Malvar-He-Cutler, the gradient-corrected linear interpolation many camera
   ISPs use), white balance, and the output gamma (sRGB by default) to RGB8.

The render's colours are taken as the white-balanced camera colours, i.e. the sensor's raw channels are the
render divided by the white-balance gains: the red and blue channels then collect fewer electrons than green
and are noisier after white balance, as on a real colour sensor in daylight.

Exposure time and gain are physical only through one assumption: the scene's reference illuminance (the
calibrated sun + sky, ``prepare.py``) is ``scene_lux`` lux, 100 000 by default (clear sky, sun at 55 deg).
Auto exposure does not depend on it (it normalises the signal itself), so it only changes the reported
``exposure_us``; a fixed ``exposure_us`` does depend on it.

    rig = lucid.make_rig("TRI032S-CC")
    sensor = LucidSensor(rig, rig.render_plan(), device="cuda:0", reference_illuminance=man["sky"]["reference_illuminance"])
    out = sensor.process(hdr)            # hdr: (H_render, W_render, 3 or 4) linear tensor
    out["rgb"]                           # (H, W, 3) uint8 on the device; out["exposure_us"], out["gain_db"], ...
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .lucid import Rig, distort, undistort

# photo-electrons per (lux x second x um^2) of sensor irradiance for a Pregius colour pixel's green channel:
# 4.09e15 photons/(m^2 s lux) at 555 nm x 1e-12 m^2/um^2 x QE 0.64 x 0.55 colour-filter transmission
ELECTRONS_PER_LUX_S_UM2 = 4.09e15 * 1e-12 * 0.64 * 0.55
LENS_TRANSMISSION = 0.9
# the ADC's full scale sits a little below the full well, so a saturated pixel reads a clean 4095
ADC_HEADROOM = 0.98


@dataclass
class SensorSettings:
    exposure: str = "auto"                  # "auto" or "fixed"
    exposure_us: float = 500.0              # fixed mode
    gain_db: float = 0.0                    # fixed mode
    ae_target: float = 0.22                 # mean of the clipped linear raw signal auto exposure aims at
    ae_speed: float = 0.35                  # fraction of the way (in log exposure) per frame
    wb_gains: tuple[float, float, float] = (1.9, 1.0, 1.6)     # daylight R, G, B gains of a Pregius colour sensor
    gamma: str | float = "srgb"             # "srgb", or LUCID's Gamma value (out = in ** gamma); 1.0 = linear
    prnu: float = 0.006                     # per-pixel gain non-uniformity (std)
    scene_lux: float = 100000.0             # what the scene's reference illuminance stands for
    noise: bool = True


# Malvar, He, Cutler (2004), "High-quality linear interpolation for demosaicing of Bayer-patterned colour
# images", the four 5x5 filters (x 1/8)
_G_AT_RB = [[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]]
_RB_AT_G_ROW = [[0, 0, 0.5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0], [0, 0, 0.5, 0, 0]]
_RB_AT_G_COL = [list(r) for r in zip(*_RB_AT_G_ROW)]
_RB_AT_BR = [[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]]


def bayer_masks(height: int, width: int, pattern: str = "RGGB", device=None) -> torch.Tensor:
    """(3, H, W) 0/1 masks of the R, G and B sites."""
    if len(pattern) != 4 or sorted(pattern) != ["B", "G", "G", "R"]:
        raise ValueError(f"unsupported Bayer pattern {pattern!r}")
    m = torch.zeros(3, height, width, device=device)
    for k, ch in enumerate(pattern):
        dy, dx = divmod(k, 2)
        m["RGB".index(ch), dy::2, dx::2] = 1.0
    return m


def mosaic(rgb: torch.Tensor, pattern: str = "RGGB") -> torch.Tensor:
    """(3, H, W) -> (H, W): each pixel keeps the channel of its colour filter."""
    return (rgb * bayer_masks(rgb.shape[1], rgb.shape[2], pattern, rgb.device)).sum(0)


def demosaic(raw: torch.Tensor, pattern: str = "RGGB") -> torch.Tensor:
    """Malvar-He-Cutler demosaicing of an (H, W) Bayer frame -> (3, H, W)."""
    if pattern != "RGGB":
        raise ValueError("only RGGB is implemented (all LUCID colour models here are BayerRG)")
    H, W = raw.shape
    k = torch.tensor([_G_AT_RB, _RB_AT_G_ROW, _RB_AT_G_COL, _RB_AT_BR], dtype=raw.dtype, device=raw.device) / 8.0
    x = F.pad(raw[None, None], (2, 2, 2, 2), mode="reflect")
    g_rb, rb_row, rb_col, rb_diag = F.conv2d(x, k[:, None])[0]
    r, g, b = raw.clone(), raw.clone(), raw.clone()
    # R site (0, 0): G and B interpolated
    g[0::2, 0::2] = g_rb[0::2, 0::2]
    b[0::2, 0::2] = rb_diag[0::2, 0::2]
    # G site in a red row (0, 1): red left/right, blue above/below
    r[0::2, 1::2] = rb_row[0::2, 1::2]
    b[0::2, 1::2] = rb_col[0::2, 1::2]
    # G site in a blue row (1, 0): red above/below, blue left/right
    r[1::2, 0::2] = rb_col[1::2, 0::2]
    b[1::2, 0::2] = rb_row[1::2, 0::2]
    # B site (1, 1)
    g[1::2, 1::2] = g_rb[1::2, 1::2]
    r[1::2, 1::2] = rb_diag[1::2, 1::2]
    return torch.stack([r, g, b])


def srgb_encode(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-12) ** (1 / 2.4) - 0.055)


def srgb_decode(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a (C, H, W) image (reflect padding)."""
    if sigma <= 0.05:
        return img
    r = max(1, int(math.ceil(3.0 * sigma)))
    t = torch.arange(-r, r + 1, dtype=img.dtype, device=img.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    c = img.shape[0]
    x = F.pad(img[None], (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x[0]


def lens_grids(rig: Rig, plan: dict, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """``grid_sample`` grids (3, H, W, 2) taking the pinhole render to the distorted image per colour, and the
    relative-illumination map (1, H, W).

    Output pixel (u, v) (OpenCV pixel centres) -> distorted normalised (x_d, y_d) -> undistorted (x_u, y_u) ->
    pinhole render pixel.  Lateral colour scales the undistorted radius of red up and blue down by half the
    corner separation each.
    """
    H, W = rig.height, rig.width
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    xd, yd = (u - rig.cx) / rig.fx, (v - rig.cy) / rig.fx
    xu, yu = undistort(xd, yd, rig.lens)
    r_corner_px = math.hypot(rig.cx + 0.5, rig.cy + 0.5)
    # lateral colour scales the undistorted radius by 1 +- half; at the corner that must separate red and blue by
    # lateral_color_px *output* pixels, so divide by the undistorted corner radius and the distortion's radial
    # stretch there (d r_distorted / d r_undistorted)
    rd = r_corner_px / rig.fx
    dx, dy = (rig.cx + 0.5) / r_corner_px, (rig.cy + 0.5) / r_corner_px
    xu0, yu0 = undistort(np.array([rd * dx]), np.array([rd * dy]), rig.lens)
    ru = float(math.hypot(xu0[0], yu0[0]))
    eps = 1e-4 * ru
    xa, ya = distort(np.array([(ru + eps) * dx]), np.array([(ru + eps) * dy]), rig.lens)
    xb, yb = distort(np.array([(ru - eps) * dx]), np.array([(ru - eps) * dy]), rig.lens)
    stretch = float(math.hypot(xa[0] - xb[0], ya[0] - yb[0]) / (2 * eps))
    half = 0.5 * rig.lens.lateral_color_px / (ru * rig.fx * stretch)
    wr, hr, fxr = plan["width"], plan["height"], plan["fx"]
    grids = []
    for scale in (1.0 + half, 1.0, 1.0 - half):                      # R, G, B
        ur = xu * scale * fxr + (wr - 1) / 2.0
        vr = yu * scale * fxr + (hr - 1) / 2.0
        # align_corners=False: -1 / +1 are the outer edges of the first / last pixel
        grids.append(np.stack([(ur + 0.5) / wr * 2.0 - 1.0, (vr + 0.5) / hr * 2.0 - 1.0], axis=-1))
    grid = torch.tensor(np.stack(grids), dtype=torch.float32, device=device)
    # relative illumination: a cos^4-like falloff (1 + a r^2)^-2 that reaches the lens's corner value
    ri = min(max(rig.lens.relative_illumination, 0.05), 1.0)
    a = ri ** -0.5 - 1.0
    rn2 = ((u - rig.cx) ** 2 + (v - rig.cy) ** 2) / (r_corner_px ** 2)
    vig = torch.tensor((1.0 + a * rn2) ** -2.0, dtype=torch.float32, device=device)[None]
    return grid, vig


class LucidSensor:
    """One camera's optics, sensor and image pipeline; keeps its auto-exposure state between frames."""

    def __init__(self, rig: Rig, plan: dict, *, device="cpu", seed: int = 0, settings: SensorSettings | None = None,
                 reference_illuminance: float | None = None, fps: float = 25.0):
        self.rig, self.plan, self.device = rig, plan, torch.device(device)
        self.s = settings or SensorSettings()
        self.fps = fps
        self.grid, self.vignette = lens_grids(rig, plan, self.device)
        self.masks = bayer_masks(rig.height, rig.width, rig.body.sensor.bayer, self.device)
        wb = torch.tensor(self.s.wb_gains, dtype=torch.float32, device=self.device)
        self.wb = wb.view(3, 1, 1)
        self.gen = torch.Generator(device=self.device).manual_seed(int(seed))
        self.prnu = 1.0 + self.s.prnu * torch.randn(rig.height, rig.width, generator=self.gen, device=self.device)
        self.full_scale = float(2 ** rig.body.sensor.adc_bits - 1)
        self.adc_electrons = ADC_HEADROOM * rig.body.sensor.full_well_e     # electrons at full scale, 0 dB
        # electrons per second at one unit of render radiance, through this lens, on this pixel (see the
        # module docstring: valid as far as scene_lux describes the scene)
        self.e_per_s_per_unit = None
        if reference_illuminance:
            lux_per_unit = self.s.scene_lux / float(reference_illuminance)
            n = rig.lens.f_number
            # the real pixel, also with resolution_scale < 1: a quick low-resolution run keeps the real camera's
            # exposure time and per-pixel noise
            pixel_um2 = rig.body.sensor.pixel_um ** 2
            # radiance unit -> cd/m^2 (= lux_per_unit, since a white diffuser under E has L = E / pi) ->
            # image-plane illuminance E = pi L T / (4 N^2)
            self.e_per_s_per_unit = (lux_per_unit * math.pi * LENS_TRANSMISSION / (4.0 * n * n)
                                     * ELECTRONS_PER_LUX_S_UM2 * pixel_um2)
        self.k = None                       # raw signal (fraction of ADC full scale, G channel) per radiance unit
        self.frames = 0

    def reset_exposure(self) -> None:
        """Let auto exposure converge from scratch on the next frame (e.g. when the input changes units)."""
        self.k = None

    # --- optics --------------------------------------------------------------
    def optics(self, hdr: torch.Tensor) -> torch.Tensor:
        """(Hr, Wr, >=3) linear render -> (3, H, W) image-plane radiance, in render units."""
        # an overflowed (inf) fp16 pixel is a very bright one (a sun glint), not a black one
        x = torch.nan_to_num(hdr[..., :3].to(self.device, torch.float32), nan=0.0, posinf=65504.0,
                             neginf=0.0).clamp_min(0.0)
        x = x.permute(2, 0, 1)
        ss = self.plan.get("supersample", 1.0)
        if ss > 1.05:                                   # pre-filter a supersampled render before resampling
            x = gaussian_blur(x, 0.5 * ss)
        if x.shape[1:] != (self.plan["height"], self.plan["width"]):
            raise ValueError(f"render is {tuple(x.shape[1:])}, the plan expects "
                             f"{(self.plan['height'], self.plan['width'])}")
        out = F.grid_sample(x[:, None], self.grid, mode="bilinear", padding_mode="border", align_corners=False)
        out = out[:, 0] * self.vignette
        return gaussian_blur(out, self.rig.lens.blur_px)

    # --- exposure ------------------------------------------------------------
    def _meter(self, raw_lin: torch.Tensor, k: float) -> float:
        """Mean of the clipped raw signal at exposure k (what the camera's auto exposure measures)."""
        return float((raw_lin * k).clamp(0.0, 1.0).mean())

    def expose(self, raw_lin: torch.Tensor) -> tuple[float, float, float]:
        """(k, exposure_us, gain_db) for this frame; raw_lin is the (H, W) mosaic in render units."""
        if self.s.exposure == "fixed":
            if self.e_per_s_per_unit is None:
                raise ValueError("a fixed exposure needs reference_illuminance (the scene's calibration)")
            g = 10 ** (self.s.gain_db / 20.0)
            k = self.e_per_s_per_unit * self.s.exposure_us * 1e-6 * g / self.adc_electrons
            return k, self.s.exposure_us, self.s.gain_db
        mean = float(raw_lin.mean())
        if mean <= 0:
            k = self.k or 1.0
        else:
            k_new = self.k or self.s.ae_target / mean
            # the clipped meter is not linear in k: a few Newton-free steps converge (first frame), or one damped
            # step per frame afterwards (the camera's continuous auto exposure)
            for _ in range(8 if self.k is None else 1):
                m = max(self._meter(raw_lin, k_new), 1e-6)
                k_new = k_new * (self.s.ae_target / m) ** (1.0 if self.k is None else self.s.ae_speed)
            k = k_new
        self.k = k
        exposure_us = float("nan")
        if self.e_per_s_per_unit:
            exposure_us = k * self.adc_electrons / self.e_per_s_per_unit * 1e6
        return k, exposure_us, 0.0

    # --- sensor --------------------------------------------------------------
    def sense(self, signal: torch.Tensor, gain_db: float = 0.0) -> torch.Tensor:
        """(H, W) signal as a fraction of the ADC full scale -> 12-bit raw DN (float tensor)."""
        sen = self.rig.body.sensor
        g = 10 ** (gain_db / 20.0)
        electrons = (signal * self.adc_electrons / g).clamp_min(0.0) * self.prnu
        if self.s.noise:
            electrons = torch.poisson(electrons, generator=self.gen)
        electrons = electrons.clamp(max=sen.full_well_e)
        if self.s.noise:
            electrons = electrons + sen.read_noise_e * torch.randn(electrons.shape, generator=self.gen,
                                                                  device=self.device)
        dn = electrons * g * self.full_scale / self.adc_electrons
        return torch.round(dn).clamp(0.0, self.full_scale)

    # --- image pipeline ------------------------------------------------------
    def isp(self, raw_dn: torch.Tensor) -> torch.Tensor:
        """12-bit Bayer DN -> (H, W, 3) uint8."""
        rgb = demosaic(raw_dn / self.full_scale, self.rig.body.sensor.bayer) * self.wb
        rgb = rgb.clamp(0.0, 1.0)
        if self.s.gamma == "srgb":
            rgb = srgb_encode(rgb)
        else:
            rgb = rgb.clamp_min(0.0) ** float(self.s.gamma)
        return (rgb * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous()

    def process(self, hdr: torch.Tensor) -> dict:
        """One frame: returns {'rgb' (H, W, 3) uint8, 'raw' (H, W) int16 12-bit DN, 'exposure_us', 'gain_db',
        'saturated' (fraction of clipped pixels), 'mean_radiance' (render units, G)}."""
        img = self.optics(hdr)
        raw_lin = (img / self.wb * self.masks).sum(0)               # the colour filter: one channel per pixel
        k, exposure_us, gain_db = self.expose(raw_lin)
        raw_dn = self.sense(raw_lin * k, gain_db)
        self.frames += 1
        return {"rgb": self.isp(raw_dn), "raw": raw_dn.to(torch.int16), "exposure_us": exposure_us,
                "gain_db": gain_db, "saturated": float((raw_dn >= self.full_scale).float().mean()),
                "mean_radiance": float(img[1].mean())}


def ldr_to_linear(rgb_u8: torch.Tensor) -> torch.Tensor:
    """Fallback input when HdrColor is not available: Isaac's tone-mapped 8-bit RGB, sRGB-decoded.  Highlights
    stay clipped at 1 and the tone curve is not undone, so this is only an approximation of the linear render."""
    return srgb_decode(rgb_u8[..., :3].float() / 255.0)
