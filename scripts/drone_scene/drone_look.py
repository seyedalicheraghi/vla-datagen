"""Make the hovering Crazyflie read as a real Crazyflie 2.1 in close-ups (the hero camera: 70 mm, ~1 m away).

``scene.apply_drone_materials`` binds flat UsdPreviewSurface constants by mesh name.  ``apply_drone_look``
goes on top of it (call it afterwards; it rebinds the parts it improves) and adds what flat constants
cannot give:

- ``cf_body_001`` (the PCB) and ``battery_holder`` (a black deck) get a procedural texture set: black
  solder mask over a copper pour with trace relief, white silkscreen, gold ENIG pads and vias, small black
  IC packages and passives (albedo, roughness, metallic and normal PNGs, 2048 px).
- four status LEDs (tiny emissive boxes on the PCB, front pair red/green, rear pair blue, as seen from above).
- ``prop_blur``: a hovering Crazyflie spins its props at 250-350 rev/s, so a photo shows a translucent
  disc, not blades.  The blades are made invisible and a thin disc with a radial alpha texture (the blades'
  real angular coverage per radius, measured on the mesh) replaces them.
- off-white ABS motor mounts, brushed-steel motors, a foil battery with a printed label, black plastic
  pin headers with gold pins.

UVs: the asset's ``primvars:st`` is authored but EMPTY on the PCB, the mounts, the battery, the holder and
the pin headers; the props' UVs are 55 % degenerate triangles.  Only the motor cylinders have a usable
unwrap (Blender cylinder: side strip in v >= 0.5, top cap circle at (0.25, 0.25), bottom cap at
(0.75, 0.25)).  So the PCB, the holder and the battery get a planar top-down projection written over
their empty ``st`` (face-varying).  It is the standard set on purpose: every material reads ``st``, and a
renderer that builds its tangent frame for normal maps from the first UV set finds data there.  Top and
bottom faces map by object-space x, y.  Side faces map into a small "edge" patch in the corner of the atlas
(cut fibreglass, foil edge) whose texels are all the same, so no texel is smeared over a wall; they are
tilted by (distance along the wall, height) only so the triangles keep a non-zero UV area.
``uv_wireframe`` renders the layouts for inspection.

UsdPreviewSurface only, so RTX and the Mitsuba preview see the same inputs.  Notes for the renderers:

- RTX draws the prop disc as a translucent surface (``opacity`` from the texture alpha, ``opacityThreshold``
  0 = blended, not cut out); path tracing does that natively, the real-time renderer needs translucency
  enabled (``RenderCfg(enable_translucency=True)``, set by ``quadcopter_pbr.py``).  The disc also carries
  ``primvars:doNotCastShadows``, a hint that is NOT verified: it is the name believed to back Omniverse's
  per-prim "Cast Shadows" switch, nothing in the preview reads it, and RTX is only expected to let light
  through ``opacity`` < 1 on its own.  If the disc shadows too hard in Isaac, toggle Cast Shadows on it in the
  Property panel and read the attribute name it authors.
- ``mitsuba_preview`` needs two things from its scene, or the drone preview looks wrong (the flight render
  has no drone and is unaffected):

  1. The ``volpath`` integrator whenever a material has fractional opacity.  ``path`` casts shadow rays
     through a ``mask`` BSDF as if it were opaque (measured: a 20 %-opaque disc shadows its ground to 1 %,
     ``volpath`` gives the physical 65 %), so every disc draws a hard shadow on the mounts and arms, which
     render dark blue and speckled.  Mitsuba has no per-shape shadow switch.  Use ``prop_blur=False`` for
     previews until the integrator is switched.
  2. A low ``sampling_weight`` on emissive materials' ``area`` emitters.  Mitsuba picks emitters uniformly,
     so four LED emitters drop the sun's share from 1/2 to 1/6 and the frame is about 1.8x noisier at equal
     spp (0.02 restores the noise level and the LEDs still render).  Without it use ``leds=False`` unless
     the LEDs are in view.
- LED ``emissiveColor`` is in the radiance unit the lights use (``mitsuba_preview``: DistantLight irradiance
  = intensity, so a white diffuse card in the sun has radiance intensity / pi).  The default is 3x that.
  The scale is not checked against RTX; tune it with ``led_radiance=``.

Everything that makes pixels is numpy + Pillow; pxr is imported inside the functions that author USD.
"""

from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = ["apply_drone_look", "pcb_textures", "battery_textures", "motor_textures", "blade_profile",
           "prop_blur_texture", "planar_st", "uv_wireframe", "LEDS", "led_nits"]

VERSION = "2"
UV_SET = "st"                         # the primvar every material reads; the planar UVs are written over the asset's empty one

PCB_SIZE = 2048
PCB_SPAN_MM = 64.0                    # the planar projection maps this square (board centre at the middle) to 0..1
BATTERY_SIZE = 1024
BATTERY_SPAN_MM = 32.0
MOTOR_SIZE = 1024
PROP_SIZE = 1024
EDGE_PX = 64                          # corner patch of an atlas that side faces sample

# The Crazyflie body frame: +x front, +y left (ROS).  LED sites are on the PCB's corner lobes beside the
# arm roots, where the board is visible between the battery holder and the motor mounts.
LED_SITE_MM = 12.9
LED_SIZE_M = (0.0016, 0.0008, 0.0006)
LEDS = {                              # name: (sign x, sign y, colour): front pair red / green, rear pair blue
    "led_fl": (1, 1, "red"),
    "led_fr": (1, -1, "green"),
    "led_rl": (-1, 1, "blue"),
    "led_rr": (-1, -1, "blue"),
}
LED_RGB = {"red": (1.0, 0.05, 0.03), "green": (0.08, 1.0, 0.18), "blue": (0.05, 0.18, 1.0)}
LED_NITS_DEFAULT = 3600.0             # 3x a white card in the calibrated sun (3859 / pi = 1228)
LED_TO_CARD = 3.0

MOTORS = {"M1": (1, -1), "M2": (-1, -1), "M3": (-1, 1), "M4": (1, 1)}      # the asset's motor positions
BLADE_NAMES = ("ccw_prop", "cw_prop")

# linear-light colours
SOLDER_MASK = 0.022
GOLD = (0.90, 0.68, 0.28)             # ENIG: a paler gold than bulk gold
TIN = (0.74, 0.74, 0.76)
SILK = (0.70, 0.70, 0.67)
EDGE_FIBRE = (0.085, 0.075, 0.05)     # cut FR4 edge: tan in life, kept dark because the model's board is 2.5 mm thick


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _srgb8(lin: np.ndarray) -> np.ndarray:
    x = np.clip(lin, 0.0, 1.0)
    s = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return (s * 255.0 + 0.5).astype(np.uint8)


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a 2-D float array (periodic edges), in the Fourier domain."""
    if sigma <= 0:
        return a.astype(np.float32)
    fy = np.fft.fftfreq(a.shape[0])[:, None]
    fx = np.fft.rfftfreq(a.shape[1])[None, :]
    g = np.exp(-2.0 * (math.pi * sigma) ** 2 * (fx * fx + fy * fy))
    return np.fft.irfft2(np.fft.rfft2(a) * g, s=a.shape).astype(np.float32)


def _noise(rng: np.random.Generator, shape, sigma: float) -> np.ndarray:
    """Smooth noise with unit standard deviation."""
    n = _blur(rng.standard_normal(shape).astype(np.float32), sigma)
    return n / (float(n.std()) + 1e-9)


def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _normal_map(height_um: np.ndarray, pitch_um: float, strength: float = 1.0, max_slope: float = 1.5) -> np.ndarray:
    """OpenGL tangent-space normal map (uint8 RGB) of a height field in um; row 0 is the top (+v) of the image."""
    gy, gx = np.gradient(height_um.astype(np.float32) / np.float32(pitch_um))
    sx, sy = -gx * strength, gy * strength            # n = (-dh/du, -dh/dv, 1) and v points up the image
    k = np.minimum(1.0, max_slope / (np.hypot(sx, sy) + 1e-9))
    n = np.stack([sx * k, sy * k, np.ones_like(sx)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return ((n * 0.5 + 0.5) * 255.0 + 0.5).astype(np.uint8)


def _write(path: str, arr: np.ndarray) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part.png"
    Image.fromarray(arr).save(tmp)
    os.replace(tmp, path)
    return path


def _font(px: int):
    try:
        return ImageFont.load_default(size=px)        # Pillow >= 10.1 with FreeType: a scalable built-in font
    except (TypeError, OSError):
        pass
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def _cache_ok(folder: str, tag: str, files: dict[str, str]) -> bool:
    meta = os.path.join(folder, f"{tag}.json")
    try:
        with open(meta, encoding="utf-8") as f:
            ok = json.load(f).get("version") == VERSION
    except (OSError, ValueError):
        return False
    return ok and all(os.path.exists(p) for p in files.values())


def _stamp(folder: str, tag: str) -> None:
    with open(os.path.join(folder, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"version": VERSION}, f)


class _Art:
    """Supersampled mask layers of one texture atlas; drawing takes millimetres in board coordinates.

    The atlas looks down on the board: +x to the right, +y up the image, the centre at the middle.
    """

    def __init__(self, size: int, span_mm: float, layers, ss: int = 2):
        self.size, self.ss = size, ss
        self.n = size * ss
        self.k = self.n / span_mm                       # pixels per mm
        self.img = {name: Image.new("L", (self.n, self.n), 0) for name in layers}
        self.draw = {name: ImageDraw.Draw(im) for name, im in self.img.items()}

    def px(self, x: float, y: float) -> tuple[float, float]:
        return self.n / 2 + x * self.k, self.n / 2 - y * self.k

    def poly(self, layer: str, pts, fill: int = 255) -> None:
        self.draw[layer].polygon([self.px(*p) for p in pts], fill=fill)

    def rect(self, layer: str, cx: float, cy: float, w: float, h: float, ang: float = 0.0, fill: int = 255) -> None:
        c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        pts = [(cx + dx * c - dy * s, cy + dx * s + dy * c)
               for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]
        self.poly(layer, pts, fill)

    def frame(self, layer: str, cx: float, cy: float, w: float, h: float, width: float, ang: float = 0.0) -> None:
        """Rectangular outline (silkscreen)."""
        for sx, sy, ww, hh in ((0, 1, w, width), (0, -1, w, width), (1, 0, width, h), (-1, 0, width, h)):
            c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            ox, oy = sx * (w - width) / 2, sy * (h - width) / 2
            self.rect(layer, cx + ox * c - oy * s, cy + ox * s + oy * c, ww, hh, ang)

    def line(self, layer: str, pts, width: float, fill: int = 255) -> None:
        """Polyline with round joints and ends."""
        p = [self.px(*q) for q in pts]
        w = max(1, int(round(width * self.k)))
        self.draw[layer].line(p, fill=fill, width=w, joint="curve")
        r = w / 2
        for x, y in (p[0], p[-1]) if len(p) > 1 else p:
            self.draw[layer].ellipse([x - r, y - r, x + r, y + r], fill=fill)

    def disc(self, layer: str, cx: float, cy: float, r: float, fill: int = 255) -> None:
        x, y = self.px(cx, cy)
        self.draw[layer].ellipse([x - r * self.k, y - r * self.k, x + r * self.k, y + r * self.k], fill=fill)

    def text(self, layer: str, cx: float, cy: float, s: str, height_mm: float, ang: float = 0.0) -> None:
        font = _font(max(6, int(round(height_mm * self.k * 1.35))))
        l, t, r, b = font.getbbox(s)
        tmp = Image.new("L", (r - l + 4, b - t + 4), 0)
        ImageDraw.Draw(tmp).text((2 - l, 2 - t), s, fill=255, font=font)
        if ang:
            tmp = tmp.rotate(ang, expand=True, resample=Image.BICUBIC)
        x, y = self.px(cx, cy)
        x0, y0 = int(round(x - tmp.width / 2)), int(round(y - tmp.height / 2))
        self.img[layer].paste(255, (x0, y0, x0 + tmp.width, y0 + tmp.height), tmp)

    def mask(self, layer: str) -> np.ndarray:
        im = self.img[layer]
        if self.ss > 1:
            im = im.reduce(self.ss)
        return np.asarray(im, np.float32) / 255.0


def _readable(ang: float) -> float:
    """Rotate a text angle by 180 degrees if it would read upside down."""
    ang = (ang + 180.0) % 360.0 - 180.0
    return ang if -90.0 < ang <= 90.0 else ang + 180.0


# ---------------------------------------------------------------------------
# PCB texture set
# ---------------------------------------------------------------------------

def _arm(a: _Art, sx: int, sy: int, label: str) -> None:
    """One PCB arm.  (u, v): millimetres along the arm from the board centre, and across it.

    The motor mount covers most of the arm; the PCB shows through three ~4 mm windows at u = 26, 32.5, 39.
    """
    d = np.array([sx, sy]) / math.sqrt(2)
    p = np.array([-sy, sx]) / math.sqrt(2)
    ang = math.degrees(math.atan2(d[1], d[0]))

    def at(u, v):
        q = u * d + v * p
        return float(q[0]), float(q[1])

    for v in (-0.85, 0.85):                                    # motor supply traces
        a.line("copper", [at(14.0, v * 1.4), at(18.0, v), at(38.2, v)], 0.45)
    for v in (-1.75, 1.75):                                    # signal traces
        a.line("copper", [at(16.0, v), at(30.5, v)], 0.2)
    # window 1: a 0402 capacitor across the arm
    for v in (-0.62, 0.62):
        a.rect("tin", *at(26.0, v), 0.7, 0.6, ang)
    a.rect("passive", *at(26.0, 0.0), 0.5, 0.95, ang)
    a.rect("tin", *at(26.0, -0.47), 0.5, 0.2, ang)
    a.rect("tin", *at(26.0, 0.47), 0.5, 0.2, ang)
    a.frame("silk", *at(26.0, 0.0), 1.7, 2.1, 0.11, ang)
    # window 2: vias and the motor label
    for v in (-1.0, 1.0):
        a.disc("gold", *at(32.5, v), 0.38)
        a.disc("hole", *at(32.5, v), 0.17)
    a.line("copper", [at(32.5, -1.0), at(32.5, 1.0)], 0.2)
    a.text("silk", *at(32.5, 0.0), label, 0.85, _readable(ang + 90.0))
    # window 3: the motor wire pads
    for v in (-1.2, 1.2):
        a.rect("gold", *at(38.9, v), 1.7, 1.3, ang)
    a.text("silk", *at(36.7, 1.55), "+", 0.7, _readable(ang))
    a.text("silk", *at(36.7, -1.55), "-", 0.7, _readable(ang))


def _lobe(a: _Art, sx: int, sy: int, label: str) -> None:
    """A corner lobe of the board beside the battery holder: a MOSFET, an LED footprint, vias."""
    def at(x, y):
        return sx * x, sy * y

    arm = math.degrees(math.atan2(sy, sx))
    cx, cy = at(12.3, 10.5)                                    # SOT-23-like MOSFET (black package)
    a.rect("ic", cx, cy, 2.9, 1.4)
    for i in (-0.95, 0.0, 0.95):
        a.rect("gold", cx + sx * i, cy - sy * 1.0, 0.55, 0.7)
    for i in (-0.95, 0.95):
        a.rect("gold", cx + sx * i, cy + sy * 1.0, 0.55, 0.7)
    a.disc("mark", cx - sx * 1.0, cy - sy * 0.35, 0.14)
    a.frame("silk", cx, cy, 3.5, 2.8, 0.11)
    lx, ly = at(LED_SITE_MM, LED_SITE_MM)                      # LED footprint, long axis along the arm
    for s in (-1.0, 1.0):
        off = np.array([math.cos(math.radians(arm)), math.sin(math.radians(arm))]) * s * 0.95
        a.rect("gold", lx + off[0], ly + off[1], 0.75, 1.0, arm)
    a.rect("silk", *(np.array([lx, ly]) + np.array([-math.sin(math.radians(arm)), math.cos(math.radians(arm))]) * 0.9),
           1.9, 0.1, arm)
    a.text("silk", *at(11.1, 13.0), label, 0.8, _readable(0.0))
    a.line("copper", [at(13.25, 11.5), at(13.25, 12.0), at(12.23, 12.23)], 0.25)
    a.line("copper", [at(13.57, 13.57), at(14.6, 12.5), at(14.5, 11.3)], 0.2)
    a.line("copper", [at(11.35, 11.5), at(10.4, 12.0), at(10.4, 13.4)], 0.2)
    for y in (10.0, 11.3):                                     # via fence
        a.disc("gold", *at(14.5, y), 0.36)
        a.disc("hole", *at(14.5, y), 0.16)
    a.line("copper", [at(14.5, 10.0), at(13.7, 10.0)], 0.2)


def _nose_and_tail(a: _Art) -> None:
    """The nose tab at +x (a test-pad row and a front arrow) and the strip at the rear."""
    for i in range(3):
        a.rect("gold", 16.9, -0.4 + 1.1 * i, 0.9, 0.8)
        a.line("copper", [(16.9, -0.4 + 1.1 * i), (14.8, -0.4 + 1.1 * i)], 0.2)
    a.poly("silk", [(15.2, 2.3), (15.2, -0.7), (16.2, 0.8)])                  # points to the front
    for i in range(2):
        a.rect("gold", -15.0, 0.3 + 1.6 * i, 0.9, 0.9)
    a.text("silk", -14.7, 3.6, "R", 0.8)


def _deck(a: _Art) -> None:
    """The battery-holder deck (central 20 x 28 mm of the atlas, hidden on the PCB by the battery)."""
    xs = [-9.05 + 2.0 * k for k in range(10)]
    for y in (-11.05, 11.05):
        for x in xs:
            a.disc("gold", x, y, 0.72)
            a.rect("hole", x, y, 0.62, 0.62)
        a.line("copper", [(xs[0], y * 1.1), (xs[-1], y * 1.1)], 0.3)
        a.frame("silk", 0.0, y, 20.6, 2.5, 0.14)
        a.disc("silk", xs[0] - 0.9, y - math.copysign(1.55, y), 0.2)
    a.text("silk", 0.0, 13.15, "BATTERY HOLDER", 0.8)
    a.text("silk", 0.0, -13.15, "V1.2", 0.8)
    for x in (-9.5, 9.5):
        a.disc("gold", x, 0.0, 0.7)
        a.disc("hole", x, 0.0, 0.32)


def pcb_textures(cache_dir: str) -> dict[str, str]:
    """Write (once) and return the PCB atlas: ``{'albedo', 'roughness', 'metallic', 'normal'}`` -> PNG path."""
    folder = os.path.join(cache_dir, "drone")
    files = {k: os.path.join(folder, f"pcb_{k}.png") for k in ("albedo", "roughness", "metallic", "normal")}
    if _cache_ok(folder, "pcb", files):
        return files
    rng = np.random.default_rng(21)
    a = _Art(PCB_SIZE, PCB_SPAN_MM, ("copper", "gold", "tin", "hole", "silk", "ic", "passive", "mark"))
    for (label, (sx, sy)), lobe_label in zip(MOTORS.items(), ("D1", "D2", "D3", "D4")):
        _arm(a, sx, sy, label)
        _lobe(a, sx, sy, lobe_label)
    _nose_and_tail(a)
    _deck(a)
    m = {k: a.mask(k) for k in a.img}
    del a

    n = PCB_SIZE
    pitch_um = PCB_SPAN_MM * 1000.0 / n
    signal = np.maximum.reduce([m["copper"], m["gold"], m["tin"], m["hole"]])
    clearance = _smoothstep(0.02, 0.22, _blur(signal, 3.5))
    pour = np.clip(1.0 - clearance, 0.0, 1.0)                  # ground pour with a moat around everything else
    copper = np.maximum(m["copper"], pour)
    hole, ic, passive = m["hole"], m["ic"], m["passive"]
    metal = np.maximum(m["gold"], m["tin"]) * (1.0 - hole)
    mask_noise = _noise(rng, (n, n), 6.0)

    base = SOLDER_MASK * (1.0 + 0.07 * mask_noise)             # mask thickness varies a little
    alb = base[..., None] * np.array([1.0, 1.0, 1.04], np.float32)
    alb = alb * (1.0 + 0.32 * copper[..., None])               # mask over copper is thinner
    layers = [(ic, (0.030, 0.030, 0.033)), (m["mark"], (0.10, 0.10, 0.10)), (passive, (0.25, 0.16, 0.08)),
              (m["tin"], TIN), (m["gold"], GOLD), (hole, (0.004, 0.004, 0.004)), (m["silk"], SILK)]
    for mask, color in layers:
        alb = alb * (1.0 - mask[..., None]) + np.asarray(color, np.float32) * mask[..., None]
    alb[-EDGE_PX:, :EDGE_PX] = EDGE_FIBRE

    rough = 0.50 + 0.03 * mask_noise
    rough = rough * (1.0 - 0.16 * copper)                      # smoother over copper
    for mask, value in ((ic, 0.62), (m["mark"], 0.35), (passive, 0.60), (m["tin"], 0.38), (m["gold"], 0.30),
                        (hole, 0.90), (m["silk"], 0.62)):
        rough = rough * (1.0 - mask) + value * mask
    rough[-EDGE_PX:, :EDGE_PX] = 0.75

    metallic = metal
    metallic[-EDGE_PX:, :EDGE_PX] = 0.0

    height = (30.0 * _blur(copper, 1.4) + 14.0 * _blur(m["silk"], 0.8) + 150.0 * _blur(ic, 1.5)
              + 110.0 * _blur(passive, 1.2) + 25.0 * _blur(metal, 1.0) - 40.0 * _blur(hole, 1.0)
              + 1.5 * _noise(rng, (n, n), 0.7))
    height[-EDGE_PX:, :EDGE_PX] = 0.0
    normal = _normal_map(height, pitch_um, 1.0)

    _write(files["albedo"], _srgb8(alb))
    _write(files["roughness"], (np.clip(rough, 0.0, 1.0) * 255 + 0.5).astype(np.uint8))
    _write(files["metallic"], (np.clip(metallic, 0.0, 1.0) * 255 + 0.5).astype(np.uint8))
    _write(files["normal"], normal)
    _stamp(folder, "pcb")
    return files


# ---------------------------------------------------------------------------
# Battery: silver foil with a printed label
# ---------------------------------------------------------------------------

def battery_textures(cache_dir: str) -> dict[str, str]:
    """The LiPo pouch atlas (top face, planar): foil with wrinkles, a printed label.  PNG paths by role."""
    folder = os.path.join(cache_dir, "drone")
    files = {k: os.path.join(folder, f"battery_{k}.png") for k in ("albedo", "roughness", "metallic", "normal")}
    if _cache_ok(folder, "battery", files):
        return files
    n = BATTERY_SIZE
    pitch_um = BATTERY_SPAN_MM * 1000.0 / n
    rng = np.random.default_rng(5)
    a = _Art(n, BATTERY_SPAN_MM, ("label", "ink", "band", "bar", "white"))
    a.rect("label", 0.0, 0.0, 19.6, 12.6)
    a.rect("band", 0.0, 4.9, 19.6, 2.3)                         # coloured band on the label
    a.text("white", -5.2, 4.9, "Li-Po", 1.6)
    a.text("ink", -3.6, 2.0, "3.7V  240mAh", 1.35)
    a.text("ink", -3.6, 0.2, "0.89Wh   1S  20C", 1.1)
    a.text("ink", -3.2, -1.7, "CHARGE 4.2V MAX 1C", 0.9)
    for i in range(38):                                         # a barcode
        a.rect("bar", -6.0 + i * 0.22, -4.2, 0.05 + 0.07 * ((i * 7) % 3), 1.8)
    a.poly("ink", [(7.6, 0.4), (9.0, 0.4), (8.3, 1.7)])
    label, ink, band, bar, white = (a.mask(k) for k in ("label", "ink", "band", "bar", "white"))
    del a
    ink = np.maximum(ink, bar)

    foil = 0.34 * _noise(rng, (n, n), 9.0) + 0.5 * _noise(rng, (n, n), 3.0) + 0.45 * _noise(rng, (n, n), 1.2)
    streak = _blur(rng.standard_normal((n, 1)).astype(np.float32) * np.ones((1, n), np.float32), 1.5)
    alb = np.broadcast_to((0.80 * (1.0 + 0.04 * foil))[..., None], (n, n, 3)) * np.array([1.0, 1.0, 1.02], np.float32)
    paper = (0.74, 0.74, 0.72)
    alb = alb * (1.0 - label[..., None]) + np.asarray(paper, np.float32) * label[..., None]
    alb = alb * (1.0 - band[..., None]) + np.array([0.02, 0.10, 0.45], np.float32) * band[..., None]
    alb = alb * (1.0 - ink[..., None] * 0.95) + np.array([0.02, 0.02, 0.025], np.float32) * ink[..., None] * 0.95
    alb = alb * (1.0 - white[..., None]) + np.array([0.80, 0.80, 0.80], np.float32) * white[..., None]
    rough = np.clip(0.34 + 0.07 * foil + 0.03 * streak, 0.15, 0.7)
    rough = rough * (1.0 - label) + 0.62 * label
    metallic = np.clip(0.9 * (1.0 - label), 0.0, 1.0)
    height = 20.0 * foil * (1.0 - label) + 3.0 * label * _noise(rng, (n, n), 1.0) + 8.0 * _blur(ink, 0.8)
    normal = _normal_map(height, pitch_um, 1.0)
    alb[-EDGE_PX:, :EDGE_PX] = (0.78, 0.78, 0.80)
    rough[-EDGE_PX:, :EDGE_PX] = 0.36
    metallic[-EDGE_PX:, :EDGE_PX] = 0.9
    normal[-EDGE_PX:, :EDGE_PX] = (128, 128, 255)

    _write(files["albedo"], _srgb8(alb))
    _write(files["roughness"], (rough * 255 + 0.5).astype(np.uint8))
    _write(files["metallic"], (metallic * 255 + 0.5).astype(np.uint8))
    _write(files["normal"], normal)
    _stamp(folder, "battery")
    return files


# ---------------------------------------------------------------------------
# Motors: brushed steel (the cylinder's own unwrap)
# ---------------------------------------------------------------------------

def motor_textures(cache_dir: str) -> dict[str, str]:
    """Brushed steel for the motor cylinders (side strip v >= 0.5 brushed around; cap circles turned)."""
    folder = os.path.join(cache_dir, "drone")
    files = {k: os.path.join(folder, f"motor_{k}.png") for k in ("albedo", "roughness", "normal")}
    if _cache_ok(folder, "motor", files):
        return files
    n = MOTOR_SIZE
    rng = np.random.default_rng(9)
    rows = np.arange(n, dtype=np.float32)[:, None]
    cols = np.arange(n, dtype=np.float32)[None, :]
    # brushing: grooves run around the circumference (along u), so the streaks vary with v (the row) only
    groove = _blur(rng.standard_normal((n, n)).astype(np.float32)[:, :1] * np.ones((1, n), np.float32), 0.9)
    streak = groove / (float(groove.std()) + 1e-9) + 0.35 * _noise(rng, (n, n), 0.6)
    # cap circles (top cap at (0.25, 0.25) of uv, bottom cap at (0.75, 0.25); image rows are 1 - v)
    caps = np.zeros((n, n), np.float32)
    r_cap = np.zeros((n, n), np.float32)
    ring = np.zeros((n, n), np.float32)
    for cx in (0.25, 0.75):
        rr = np.hypot((cols / n - cx) * n, (rows / n - 0.75) * n) / (0.25 * n)
        inside = rr < 1.0
        caps = np.where(inside, 1.0, caps)
        r_cap = np.where(inside, rr, r_cap)
        ring = np.where(inside, 0.5 * np.sin(rr * 60.0) + 0.25 * np.sin(rr * 23.0), ring)
    side = 1.0 - caps
    shaft = _smoothstep(0.17, 0.12, r_cap) * caps                # dark bearing / shaft hole at the cap centres
    crimp = np.exp(-((rows / n - 0.2) / 0.012) ** 2) * side      # the crimped ring of the motor can
    steel = 0.60 * (1.0 + 0.05 * streak) * (1.0 - 0.18 * crimp)
    alb = np.broadcast_to(steel[..., None], (n, n, 3)) * np.array([1.0, 1.0, 1.03], np.float32)
    alb = alb * (1.0 - 0.8 * shaft[..., None])
    rough = np.clip(0.34 + 0.10 * streak + 0.06 * ring * caps + 0.25 * shaft + 0.12 * crimp, 0.12, 0.8)
    height = 6.0 * streak * side + 8.0 * ring * caps - 25.0 * crimp
    normal = _normal_map(height, 20.0, 1.0)
    _write(files["albedo"], _srgb8(alb))
    _write(files["roughness"], (rough * 255 + 0.5).astype(np.uint8))
    _write(files["normal"], normal)
    _stamp(folder, "motor")
    return files


# ---------------------------------------------------------------------------
# Propeller blur
# ---------------------------------------------------------------------------

def blade_profile(points_xy: np.ndarray, triangles: np.ndarray, radius: float, bins: int = 96,
                  size: int = 1536) -> np.ndarray:
    """Fraction of each annulus (``bins`` radii from 0 to ``radius``) covered by the blades seen from above.

    A spinning prop sweeps this fraction of the time at every point of the annulus, so for opaque blades
    it is also the alpha of the long-exposure blur.
    """
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    s = size / (2.0 * radius * 1.02)
    for tri in triangles:
        d.polygon([(size / 2 + points_xy[i, 0] * s, size / 2 - points_xy[i, 1] * s) for i in tri], fill=255)
    cover = np.asarray(img, np.float32) / 255.0
    yy, xx = np.mgrid[0:size, 0:size]
    rr = np.hypot(xx - size / 2 + 0.5, yy - size / 2 + 0.5) / s / radius
    which = np.minimum((rr * bins).astype(int), bins)
    total = np.bincount(which.ravel(), minlength=bins + 1)[:bins]
    got = np.bincount(which.ravel(), weights=cover.ravel(), minlength=bins + 1)[:bins]
    return (got / np.maximum(total, 1)).astype(np.float32)


def _blur_alpha(profile: np.ndarray, hub: float, gain: float = 1.0) -> np.ndarray:
    """Radial alpha from the measured coverage: smoothed, with the opaque hub and a soft tip."""
    n = len(profile)
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    padded = np.concatenate([profile[:1].repeat(6), profile, np.zeros(6)])
    sm = np.convolve(padded, k / k.sum(), mode="valid") * gain
    r = (np.arange(n) + 0.5) / n
    sm = np.maximum(sm, 0.85 * (1.0 - _smoothstep(hub, hub * 2.0, r)))
    return np.clip(sm * (1.0 - _smoothstep(0.88, 1.0, r)), 0.0, 0.92)


def prop_blur_texture(cache_dir: str, profile: np.ndarray, hub_fraction: float = 0.11, gain: float = 1.0) -> str:
    """RGBA PNG of the blur disc (radius = the half width of the image): dark plastic colour, radial alpha.

    ``gain`` scales the alpha (1 = the blades' real angular coverage, i.e. a long exposure of opaque blades).
    """
    key = hashlib.sha1(np.round(np.asarray(profile, np.float32), 3).tobytes() + repr(
        (hub_fraction, gain)).encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, "drone", f"prop_blur_{key}_{VERSION}.png")
    if os.path.exists(path):
        return path
    alpha_r = _blur_alpha(np.asarray(profile, np.float32), hub_fraction, gain)
    n = PROP_SIZE
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - n / 2 + 0.5, yy - n / 2 + 0.5) / (n / 2)
    alpha = np.interp(rr, (np.arange(len(alpha_r)) + 0.5) / len(alpha_r), alpha_r, right=0.0)
    alpha = np.where(rr >= 1.0, 0.0, alpha)
    rgba = np.empty((n, n, 4), np.uint8)
    rgba[..., :3] = _srgb8(np.asarray((0.030, 0.030, 0.034), np.float32))
    rgba[..., 3] = (alpha * 255 + 0.5).astype(np.uint8)
    return _write(path, rgba)


# ---------------------------------------------------------------------------
# Mesh helpers (pxr)
# ---------------------------------------------------------------------------

def _mesh_arrays(prim):
    from pxr import UsdGeom

    m = UsdGeom.Mesh(prim)
    return (np.array(m.GetPointsAttr().Get(), float), np.array(m.GetFaceVertexCountsAttr().Get(), np.int64),
            np.array(m.GetFaceVertexIndicesAttr().Get(), np.int64))


def _face_normals(points: np.ndarray, counts: np.ndarray, idx: np.ndarray, unit: bool = True) -> np.ndarray:
    """Per-face normals by Newell's method (polygons of any size): unit length, or of length 2 x area."""
    starts = np.cumsum(counts) - counts
    face = np.repeat(np.arange(len(counts)), counts)
    local = np.arange(len(idx)) - starts[face]
    nxt = idx[starts[face] + (local + 1) % counts[face]]
    a, b = points[idx], points[nxt]
    contrib = np.stack([(a[:, 1] - b[:, 1]) * (a[:, 2] + b[:, 2]), (a[:, 2] - b[:, 2]) * (a[:, 0] + b[:, 0]),
                        (a[:, 0] - b[:, 0]) * (a[:, 1] + b[:, 1])], axis=1)
    n = np.stack([np.bincount(face, weights=contrib[:, i], minlength=len(counts)) for i in range(3)], axis=1)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30) if unit else n


def planar_st(points: np.ndarray, counts: np.ndarray, idx: np.ndarray, span_m: tuple[float, float],
              edge_uv: tuple[float, float], edge_half: float = 0.0) -> np.ndarray:
    """Face-varying (one per face corner) top-down planar UVs of a mesh whose points are in the drone frame.

    Top and bottom faces (|normal.z| >= 0.5) map by x, y over ``span_m`` centred on the origin; every other
    face lands in the texel region centred on ``edge_uv``.  With ``edge_half`` = 0 that is the single point
    ``edge_uv``; with the region's half width in uv units the wall corners spread over at most half of it,
    by (distance along the wall, height), so the wall triangles keep a UV area and a tangent frame.
    """
    n = _face_normals(points, counts, idx)
    face = np.repeat(np.arange(len(counts)), counts)
    flat = np.abs(n[face, 2]) >= 0.5
    p = points[idx]
    st = np.stack([p[:, 0] / span_m[0] + 0.5, p[:, 1] / span_m[1] + 0.5], axis=1)
    st[~flat] = edge_uv
    if edge_half > 0.0:
        nh = n[face, :2]
        t = np.stack([-nh[:, 1], nh[:, 0]], axis=1) / np.maximum(np.linalg.norm(nh, axis=1, keepdims=True), 1e-9)
        # (t, up, outward) is right-handed, like (x, y, up) on the top faces
        wall = np.stack([(p[:, :2] * t).sum(axis=1), p[:, 2]], axis=1) * (0.5 * edge_half / max(span_m))
        centre = np.asarray(edge_uv, float)
        st[~flat] = np.clip(centre + wall, centre - 0.9 * edge_half, centre + 0.9 * edge_half)[~flat]
    return st.astype(np.float32)


def _to_frame(points: np.ndarray, matrix) -> np.ndarray:
    m = np.array(matrix, float)
    return points @ m[:3, :3] + m[3, :3]


def _set_primvar(prim, name: str, values: np.ndarray, interpolation: str):
    from pxr import Gf, Sdf, UsdGeom, Vt

    api = UsdGeom.PrimvarsAPI(prim)
    old = api.GetPrimvar(name)                 # the asset's own (empty) ``st`` is float2[]: keep its type name
    pv = api.CreatePrimvar(name, old.GetTypeName() if old else Sdf.ValueTypeNames.TexCoord2fArray, interpolation)
    pv.Set(Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in values]))
    return pv


def uv_wireframe(prim, out_png: str, size: int = 1024, varname: str = "st", background: str | None = None) -> dict:
    """Draw a mesh's UV triangles (variable ``varname``) to a PNG, optionally over a texture; returns statistics.

    Degenerate (zero-area) triangles are drawn red.  Face-varying and vertex primvars both work; an empty
    primvar gives an empty plot and ``triangles == 0`` UV coverage.
    """
    from pxr import UsdGeom

    points, counts, idx = _mesh_arrays(prim)
    pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar(varname)
    uv = np.asarray(pv.Get(), float).reshape(-1, 2) if pv and pv.HasValue() and pv.Get() is not None else np.zeros((0, 2))
    interp = pv.GetInterpolation() if pv else "vertex"
    im = Image.new("RGB", (size, size), (16, 16, 16))
    if background:
        with Image.open(background) as bg:
            im = bg.convert("RGB").resize((size, size)).point(lambda v: v // 2)
    d = ImageDraw.Draw(im)
    for t in np.linspace(0, 1, 5):
        d.line([(t * size, 0), (t * size, size)], fill=(60, 60, 60))
        d.line([(0, t * size), (size, t * size)], fill=(60, 60, 60))
    stats = {"corners": int(len(uv)), "uv_min": None, "uv_max": None, "degenerate": 0, "faces": int(len(counts)),
             "valid": False}
    valid = len(uv) > 0 and len(uv) == (len(idx) if interp == "faceVarying" else len(points))
    if valid:
        corner_uv = uv if interp == "faceVarying" else uv[idx]
        starts = np.cumsum(counts) - counts
        stats.update(uv_min=[float(x) for x in uv.min(0)], uv_max=[float(x) for x in uv.max(0)], valid=True)
        for s0, c in zip(starts, counts):
            q = corner_uv[s0:s0 + c]
            area = 0.5 * abs(sum(q[i, 0] * q[(i + 1) % c, 1] - q[(i + 1) % c, 0] * q[i, 1] for i in range(c)))
            bad = area < 1e-9
            stats["degenerate"] += int(bad)
            d.polygon([(u * size, (1 - v) * size) for u, v in q], outline=(255, 70, 70) if bad else (110, 230, 110))
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    im.save(out_png)
    return stats


# ---------------------------------------------------------------------------
# USD authoring
# ---------------------------------------------------------------------------

def _surface(stage, path: str, *, varname: str = "st", color=(0.5, 0.5, 0.5), albedo: str | None = None,
             roughness: float = 0.5, roughness_tex: str | None = None, metallic: float = 0.0,
             metallic_tex: str | None = None, normal: str | None = None, clearcoat: float = 0.0,
             clearcoat_roughness: float = 0.08, alpha_from_albedo: bool = False, emissive=None):
    """UsdPreviewSurface with optional textures (read through primvar ``varname``, OpenGL-style normals)."""
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    pbr = UsdShade.Shader.Define(stage, f"{path}/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    pbr.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    pbr.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(float(clearcoat))
    pbr.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(float(clearcoat_roughness))
    st = None
    if any((albedo, roughness_tex, metallic_tex, normal)):
        reader = UsdShade.Shader.Define(stage, f"{path}/stReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set(varname)
        st = reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    def tex(name: str, file: str, raw: bool, scale=(1, 1, 1, 1), bias=(0, 0, 0, 0)):
        t = UsdShade.Shader.Define(stage, f"{path}/{name}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(file).replace("\\", "/"))
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw" if raw else "sRGB")
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
        if scale != (1, 1, 1, 1):
            t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
        if bias != (0, 0, 0, 0):
            t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
        return t

    F, C3, N3 = Sdf.ValueTypeNames.Float, Sdf.ValueTypeNames.Color3f, Sdf.ValueTypeNames.Normal3f
    if albedo:
        a = tex("albedoTex", albedo, raw=False)
        pbr.CreateInput("diffuseColor", C3).ConnectToSource(a.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
        if alpha_from_albedo:
            pbr.CreateInput("opacity", F).ConnectToSource(a.CreateOutput("a", F))
            pbr.CreateInput("opacityThreshold", F).Set(0.0)          # blended, not cut out
    else:
        pbr.CreateInput("diffuseColor", C3).Set(Gf.Vec3f(*color))
    for name, tex_file, const, node in (("roughness", roughness_tex, roughness, "roughTex"),
                                        ("metallic", metallic_tex, metallic, "metalTex")):
        if tex_file:
            pbr.CreateInput(name, F).ConnectToSource(tex(node, tex_file, raw=True).CreateOutput("r", F))
        else:
            pbr.CreateInput(name, F).Set(float(const))
    if normal:
        n = tex("normalTex", normal, raw=True, scale=(2.0, 2.0, 2.0, 1.0), bias=(-1.0, -1.0, -1.0, 0.0))
        pbr.CreateInput("normal", N3).ConnectToSource(n.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    if emissive is not None:
        pbr.CreateInput("emissiveColor", C3).Set(Gf.Vec3f(*[float(c) for c in emissive]))
    return mat


def led_nits(stage, default: float = LED_NITS_DEFAULT) -> float:
    """LED radiance: ``LED_TO_CARD`` times a white diffuse card facing the stage's sun (intensity / pi)."""
    from pxr import Usd, UsdLux

    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.IsA(UsdLux.DistantLight):
            light = UsdLux.DistantLight(prim)
            e = float(light.GetIntensityAttr().Get() or 0.0) * 2.0 ** float(light.GetExposureAttr().Get() or 0.0)
            c = light.GetColorAttr().Get() or (1.0, 1.0, 1.0)
            lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
            if e > 0:
                return LED_TO_CARD * e * lum / math.pi
    return default


def _led_boxes(stage, parent_path: str, parent_to_root: np.ndarray, z_top: float, mats: dict) -> int:
    from pxr import Gf, UsdGeom

    from . import scene

    UsdGeom.Xform.Define(stage, parent_path)
    lx, ly, lz = LED_SIZE_M
    half = (lx / 2, ly / 2)
    inv = np.linalg.inv(parent_to_root)
    for name, (sx, sy, _) in LEDS.items():
        data = scene.box(-half[0], half[0], -half[1], half[1], 0.0, lz, 1.0)
        path = f"{parent_path}/{name}"
        scene._mesh(stage, path, data, mats[name])
        pos = np.array([sx * LED_SITE_MM * 1e-3, sy * LED_SITE_MM * 1e-3, z_top, 1.0]) @ inv
        x = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        x.ClearXformOpOrder()
        x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos[:3]]))
        x.AddRotateZOp().Set(math.degrees(math.atan2(sy, sx)))        # long axis along the arm
    return len(LEDS)


def _pin_subset(stage, mesh_prim, gold) -> int:
    """GeomSubset ``pins`` of the pin-header mesh: the faces of the 0.5 mm pins (xy extent < 0.7 mm)."""
    from pxr import UsdGeom, UsdShade, Vt

    points, counts, idx = _mesh_arrays(mesh_prim)
    starts = np.cumsum(counts) - counts
    xy = points[idx][:, :2]
    ext = np.stack([np.maximum.reduceat(xy[:, i], starts) - np.minimum.reduceat(xy[:, i], starts) for i in (0, 1)])
    pins = np.nonzero(ext.max(axis=0) < 0.0007)[0]
    path = mesh_prim.GetPath().AppendChild("pins")
    if len(pins) == 0 or len(pins) == len(counts):
        stage.RemovePrim(path)
        return 0
    sub = UsdGeom.Subset.Define(stage, path)
    sub.CreateElementTypeAttr().Set(UsdGeom.Tokens.face)
    sub.CreateIndicesAttr().Set(Vt.IntArray([int(i) for i in pins]))
    sub.CreateFamilyNameAttr().Set("materialBind")
    UsdGeom.Subset.SetFamilyType(UsdGeom.Imageable(mesh_prim), "materialBind", UsdGeom.Tokens.nonOverlapping)
    UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(gold)
    return int(len(pins))


def _blade_geometry(blade_prim) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(points in the prop prim's frame, fan triangles, radius from the prop axis, mid-plane height)."""
    from pxr import UsdGeom

    points, counts, idx = _mesh_arrays(blade_prim)
    p = _to_frame(points, np.array(UsdGeom.Xformable(blade_prim).GetLocalTransformation(), float))
    starts = np.cumsum(counts) - counts
    corner = np.concatenate([[s, s + k + 1, s + k + 2] for s, c in zip(starts, counts) for k in range(c - 2)])
    tris = idx[corner].reshape(-1, 3)
    return p, tris, float(np.hypot(p[:, 0], p[:, 1]).max()), float(p[:, 2].mean())


def _blur_disc(stage, blade_prim, material, radius: float, zc: float) -> str:
    """A thin double-sided disc of ``radius`` under the blade's parent prim, planar-UV mapped for a radial texture."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    path = str(blade_prim.GetParent().GetPath().AppendChild("blur_disc"))
    seg = 64
    ang = np.linspace(0.0, 2 * math.pi, seg, endpoint=False)
    ring = np.stack([radius * np.cos(ang), radius * np.sin(ang), np.full(seg, zc)], axis=1)
    verts = np.vstack([[0.0, 0.0, zc], ring])
    disc = UsdGeom.Mesh.Define(stage, path)
    disc.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*[float(c) for c in v]) for v in verts]))
    disc.CreateFaceVertexCountsAttr(Vt.IntArray([3] * seg))
    tris = [(0, 1 + i, 1 + (i + 1) % seg) for i in range(seg)]
    disc.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for t in tris for i in t]))
    disc.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * len(verts)))
    disc.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    disc.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    disc.CreateDoubleSidedAttr(True)
    disc.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-radius, -radius, zc), Gf.Vec3f(radius, radius, zc)]))
    st = 0.5 + 0.5 * verts[:, :2] / radius
    _set_primvar(disc.GetPrim(), "st", st, UsdGeom.Tokens.vertex)
    # an unverified hint for RTX (see the module docstring); the preview ignores it
    UsdGeom.PrimvarsAPI(disc).CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool,
                                            UsdGeom.Tokens.constant).Set(True)
    UsdShade.MaterialBindingAPI.Apply(disc.GetPrim()).Bind(material, UsdShade.Tokens.strongerThanDescendants)
    return path


def apply_drone_look(stage, drone_root: str, cache_dir: str, *, prop_blur: bool = True, leds: bool = True,
                     looks_root: str = "/World/Looks/DroneLook", led_radiance: float | None = None,
                     blur_gain: float = 1.0) -> dict:
    """Give the Crazyflie under ``drone_root`` the Crazyflie 2.1 look; call after ``scene.apply_drone_materials``.

    ``cache_dir``: textures go to ``<cache_dir>/drone/`` (generated once, 2048 px for the PCB).
    ``prop_blur``: hide the blades and add a translucent blur disc under each ``m*_prop`` prim; without it
    the blades stay visible (and any disc from an earlier call is removed).  ``leds``: add the four status
    LEDs (without it they are removed).  ``led_radiance``: emissive radiance of an LED (default
    ``led_nits(stage)``).  ``blur_gain``: scales the blur disc's alpha (1 = the blades' measured coverage).
    Idempotent: the same arguments leave the stage's layer text unchanged, and other arguments replace the
    discs, blur materials and LEDs of an earlier call instead of piling up.  Returns a summary dict.
    """
    from pxr import Usd, UsdGeom, UsdShade

    root = stage.GetPrimAtPath(drone_root)
    if not root or not root.IsValid():
        raise ValueError(f"no prim at {drone_root!r}")
    meshes: dict[str, list] = {}
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            meshes.setdefault(prim.GetName(), []).append(prim)
    xc = UsdGeom.XformCache()

    def to_root(prim) -> np.ndarray:
        return np.array(xc.ComputeRelativeTransform(prim, root)[0], float)

    tex = {"pcb": pcb_textures(cache_dir), "battery": battery_textures(cache_dir), "motor": motor_textures(cache_dir)}
    L = looks_root
    mats = {
        "pcb": _surface(stage, f"{L}/pcb", varname=UV_SET, albedo=tex["pcb"]["albedo"],
                        roughness_tex=tex["pcb"]["roughness"], metallic_tex=tex["pcb"]["metallic"],
                        normal=tex["pcb"]["normal"]),
        "mount": _surface(stage, f"{L}/mount", color=(0.70, 0.69, 0.65), roughness=0.42, clearcoat=0.15,
                          clearcoat_roughness=0.35),
        "motor": _surface(stage, f"{L}/motor", albedo=tex["motor"]["albedo"], roughness_tex=tex["motor"]["roughness"],
                          metallic=1.0, normal=tex["motor"]["normal"]),
        "battery": _surface(stage, f"{L}/battery", varname=UV_SET, albedo=tex["battery"]["albedo"],
                            roughness_tex=tex["battery"]["roughness"], metallic_tex=tex["battery"]["metallic"],
                            normal=tex["battery"]["normal"]),
        "housing": _surface(stage, f"{L}/housing", color=(0.012, 0.012, 0.013), roughness=0.45),
        "gold": _surface(stage, f"{L}/gold", color=GOLD, roughness=0.26, metallic=1.0),
        "blade": _surface(stage, f"{L}/blade", color=(0.030, 0.030, 0.034), roughness=0.42),
    }
    summary: dict = {"textures": {k: dict(v) for k, v in tex.items()}, "bound": {}, "uv": {}}

    def bind(name: str, key: str, strength=UsdShade.Tokens.strongerThanDescendants) -> int:
        for prim in meshes.get(name, []):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mats[key], strength)
            summary["bound"][str(prim.GetPath())] = key
        return len(meshes.get(name, []))

    # planar UVs and materials of the PCB, the deck and the battery
    pcb_top = None
    for name, span, atlas_px, key in (("cf_body_001", PCB_SPAN_MM, PCB_SIZE, "pcb"),
                                      ("battery_holder", PCB_SPAN_MM, PCB_SIZE, "pcb"),
                                      ("battery", BATTERY_SPAN_MM, BATTERY_SIZE, "battery")):
        edge = EDGE_PX / 2 / atlas_px                  # centre and half width of the edge patch, in uv units
        for prim in meshes.get(name, []):
            points, counts, idx = _mesh_arrays(prim)
            p = _to_frame(points, to_root(prim))
            _set_primvar(prim, UV_SET, planar_st(p, counts, idx, (span * 1e-3, span * 1e-3), (edge, edge), edge),
                         UsdGeom.Tokens.faceVarying)
            summary["uv"][str(prim.GetPath())] = UV_SET
            if name == "cf_body_001":
                pcb_top = _top_plane(p, counts, idx)
        bind(name, key)
    bind("motor_mount", "mount")
    bind("Cylinder", "motor")

    pin_faces = 0
    for prim in meshes.get("pin_headers", []):
        # the housing is the mesh's own binding; weaker than the descendant subset that carries the gold pins
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mats["housing"], UsdShade.Tokens.weakerThanDescendants)
        summary["bound"][str(prim.GetPath())] = "housing"
        pin_faces += _pin_subset(stage, prim, mats["gold"])
    summary["pin_faces"] = pin_faces

    # propellers: blur discs, or the blades themselves
    discs, radii, blur_mats = [], [], set()
    for name in BLADE_NAMES:
        for blade in meshes.get(name, []):
            if not UsdShade.MaterialBindingAPI(blade).GetDirectBindingRel().GetTargets():
                UsdShade.MaterialBindingAPI.Apply(blade).Bind(mats["blade"], UsdShade.Tokens.strongerThanDescendants)
            disc_path = blade.GetParent().GetPath().AppendChild("blur_disc")
            if prop_blur:
                p, tris, radius, zc = _blade_geometry(blade)
                png = prop_blur_texture(cache_dir, blade_profile(p[:, :2], tris, radius), gain=blur_gain)
                mat_name = f"prop_blur_{os.path.basename(png)[10:20]}"
                blur_mat = _surface(stage, f"{L}/{mat_name}", albedo=png, roughness=0.5, alpha_from_albedo=True)
                blur_mats.add(mat_name)
                discs.append(_blur_disc(stage, blade, blur_mat, radius, zc))
                radii.append(radius)
                UsdGeom.Imageable(blade).MakeInvisible()
            else:
                UsdGeom.Imageable(blade).MakeVisible()
                stage.RemovePrim(disc_path)
    looks = stage.GetPrimAtPath(L)
    for child in (looks.GetChildren() if looks else []):          # blur materials of an earlier gain or call
        if child.GetName().startswith("prop_blur_") and child.GetName() not in blur_mats:
            stage.RemovePrim(child.GetPath())
    summary.update(prop_discs=discs, prop_radius=radii, blades_hidden=bool(prop_blur))

    # status LEDs
    nodes = meshes.get("cf_body_001", [])
    parent = nodes[0].GetParent() if nodes else None
    led_root = f"{parent.GetPath()}/leds" if parent else f"{drone_root}/leds"
    if leds and parent and pcb_top is not None:
        nits = led_nits(stage) if led_radiance is None else float(led_radiance)
        led_mats = {}                  # one material per LED: Mitsuba attaches an emitter to a single shape
        for name, (_, _, color) in LEDS.items():
            rgb = LED_RGB[color]
            led_mats[name] = _surface(stage, f"{L}/{name}", color=tuple(0.35 * c for c in rgb), roughness=0.15,
                                      clearcoat=0.5, clearcoat_roughness=0.05, emissive=[c * nits for c in rgb])
        summary["leds"] = _led_boxes(stage, led_root, to_root(parent), pcb_top, led_mats)
        summary["led_radiance"] = nits
    else:
        stage.RemovePrim(led_root)
        for name in LEDS:                                          # and their materials from an earlier call
            stage.RemovePrim(f"{L}/{name}")
        summary["leds"] = 0
    summary["meshes_bound"] = len(summary["bound"])
    return summary


def _top_plane(points: np.ndarray, counts: np.ndarray, idx: np.ndarray) -> float:
    """Height of the up-facing plane with the most area (the PCB's top face)."""
    raw = _face_normals(points, counts, idx, unit=False)
    area = 0.5 * np.linalg.norm(raw, axis=1)
    up = raw[:, 2] > 0.9 * 2.0 * area
    face = np.repeat(np.arange(len(counts)), counts)
    z = np.bincount(face, weights=points[idx][:, 2], minlength=len(counts)) / counts
    uniq, inv = np.unique(np.round(z[up] * 1e5).astype(np.int64), return_inverse=True)
    return float(uniq[np.argmax(np.bincount(inv, weights=area[up]))]) * 1e-5
