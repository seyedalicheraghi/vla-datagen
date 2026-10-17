"""Offline path-traced preview of a composed USD stage with Mitsuba 3 (no Isaac Sim needed).

    python scripts/drone_scene/mitsuba_preview.py stage.usda --camera /World/Cameras/demo --out demo.png \
        [--width 960 --height 540 --spp 64 --ev 0 --tonemap aces --exr demo.exr]

``render(stage, camera, out_png, ...)`` reads the stage with pxr (meshes, PointInstancer and native
instances, UsdPreviewSurface and MDL/OmniPBR materials, UsdLux lights, a UsdGeom.Camera), builds a Mitsuba
scene and writes a tone-mapped PNG.  ``render_many`` renders several cameras from one scene build.  Assets
must be local files (mirror Nucleus references first).  Everything that decides the picture is explained
here, so a difference to Isaac's RTX path tracer can be traced to one assumption.

Frames and units: Z up (anything else is rendered as authored, no conversion), stage units; world
transforms from ``UsdGeom.XformCache`` are the truth, so a centimetre asset referenced under a 0.01 scale
is metres.  DistantLight: irradiance at normal incidence = ``intensity * 2**exposure * color`` (the cone
``angle`` does not change the power; the light is a sharp delta light).  DomeLight: radiance =
``intensity * 2**exposure * color * texel``.  Light and emission values are used as authored, with no
hidden scale, so DomeLight intensity, DistantLight intensity and OmniPBR ``emissive_intensity`` share one
unit (radiance, with irradiance / pi for the sun); ``exposure_ev`` scales the final image.  Whether
Isaac's RTX renderer scales any of them differently is not checked here.

Time: the stage's start time code if it has a time range, else the earliest time sample (so animation
that only has time samples is read at its first sample), unless ``time_code`` is given.

Camera: a USD camera looks down -Z with +Y up.  Mitsuba's perspective sensor looks down +Z with +X to the
left, so the sensor matrix is the USD world matrix times ``diag(-1, 1, -1)`` (checked: a quad at +Y of a
camera looking along +X lands in the left half of the image).  ``fov`` is horizontal, from
``horizontalAperture / focalLength``; width/height may differ from the apertures' ratio (the horizontal
field of view is kept).  ``fStop > 0`` makes a thin lens, ``aperture radius = focalLength / (2 fStop)``
with the focal length taken in mm.

Dome orientation: the UsdLux DomeLight specification, which RTX implements (pxr usdLux/schema.usda,
Z-up stage, ``poleAxis`` "scene"): the texture zenith is +Z, longitude falls from +pi at ``u = 0`` to -pi
at ``u = 1``, ``u = 0.5`` (longitude 0) looks along -Y and ``u = 0.25`` along +X, i.e. the OpenEXR
lat-long layout of a photographed panorama.  ``hdr.py`` uses the mirror image of this (``u = 0.5`` along
+X, azimuth counter-clockwise); the two are related by ``usd_dir(u, v) = Rz(-90) hdr_dir(1 - u, v)``.
Mitsuba's envmap has +Y up and ``u = atan2(x, -z) / 2pi``; ``to_world`` is the light prim's rotation times
``Rz(-90)`` times the cyclic axis permutation ``[[0,0,1],[1,0,0],[0,1,0]]`` (measured by sampling the
envmap along known directions; the unit tests use single bright pixels).

Textures: images are decoded with PIL (``hdr.read_hdr`` for Radiance .hdr, which reads all four row and
column orientations and divides out EXPOSURE; Mitsuba for .exr/.pfm), sRGB-decoded when the shader says so,
scaled and biased, and handed to Mitsuba as float bitmaps.  USD's v axis points up, Mitsuba's down, so
images are flipped vertically once and UV transforms stay in USD space.  Wrap modes: repeat, mirror and
clamp (black and useMetadata become clamp and repeat); a single Mitsuba wrap mode serves both axes.

Normal maps (measured with a constant "tilt towards +U / +V" map on a quad lit from four sides):
Mitsuba's ``normalmap`` reads ``n = 2 * texel - 1`` with R along +U and G along +V, i.e. OpenGL style.  So
UsdPreviewSurface needs no flip (its ``normal`` is the texel after ``scale``/``bias``, used as is);
OmniPBR's ``flip_tangent_v`` defaults to true (DirectX maps, which NVIDIA's textures are) and flips G.
``flip_green_preview`` / ``flip_green_mdl`` set the flip per material type (None keeps these defaults).
Mitsuba's tangent frame is (dp/du, cross(n, dp/du)), so its V axis ignores the UV handedness, while
OmniPBR's tangent_v follows the UV layout.  Triangles whose UVs wind against their normals (mirrored UV
islands, e.g. two sides of ``scene.box``, or an instance under a mirroring transform) are therefore split
off and get the opposite green flip.

OmniPBR, from NVIDIA's OmniPBR.mdl / OmniPBR_ClearCoat.mdl source: UVs are ``R(texture_rotate) *
(texture_scale * uv) + texture_translate`` (so a larger ``texture_scale`` repeats the texture more
often); albedo = ``texture * albedo_brightness + albedo_add``, desaturated, times ``diffuse_tint``
(``diffuse_color_constant`` when no texture); roughness = ``lerp(constant, texture, influence)`` with the
texture being ORM.g when ``enable_ORM_texture``; the influences default to 0; ORM.r (AO) only darkens
the albedo by ``ao_to_diffuse`` (default 0); opacity needs ``enable_opacity``.  Defaults come, in order,
from the built-in OmniPBR table, the MDL file next to the asset (regex over its OmniPBR call), and the
shader prim's authored inputs.  Roughness is taken as perceptual (alpha = roughness squared), as for
UsdPreviewSurface, which is an assumption for MDL.  ``project_uvw`` (triplanar) is not supported.

Two-sidedness: we assume Isaac's RTX renderer shades both sides of every mesh (its back-face culling is
off by default), so ``doubleSided`` is ignored (``backface_culling=False``).  With ``backface_culling=True``
meshes whose ``doubleSided`` is false (USD's fallback) keep a one-sided BSDF (their backs render dark;
nothing is really culled) and ``doubleSided = true`` meshes stay two-sided.
``leftHanded`` meshes and negative-determinant transforms get their winding flipped.  A camera's
``clippingRange`` is used only when authored (USD's fallback near plane of 1 unit would cut off a drone).

Variants and instancing: identical meshes (native instances, PointInstancer prototypes, the same prim
referenced twice) become one Mitsuba shapegroup plus ``instance`` objects.  The LLVM variant traces every
shape and instance once per render (about 40 ms each), so scenes with more than ``MAX_LLVM_SHAPES``
shapes render with ``scalar_rgb`` instead (same image, similar speed, no start-up cost).

Output: linear HDR, times ``2**exposure_ev``, ACES (Narkowicz fit) or no tone curve, sRGB encode, 8-bit PNG;
``save_exr`` stores the linear radiance before exposure.

Implicit prims (Cube, Sphere, Cylinder, Cone, Capsule, Plane) are tessellated with smooth normals and
simple UVs.  Not supported (logged where it matters): curves and points, subdivision
(the base mesh is drawn), displacement, UsdPreviewSurface ``occlusion`` and ``ior``, triplanar
``project_uvw``, MDL glass and other non-OmniPBR MDL (a diffuse-like input and roughness 0.6 stand in;
an MDL file that is missing locally counts as OmniPBR only if the shader authors OmniPBR inputs),
orthographic cameras, spot/IES shaping, light linking, motion blur, the sun's angular size (sharp
shadows), more than one dome light.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import re
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

if __package__ in (None, ""):               # run as a script: make the sibling modules importable as drone_scene.*
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_scene.hdr import read_hdr  # noqa: E402

__all__ = ["render", "render_many", "tonemap_image", "parse_mdl", "MaterialSpec", "TexSrc"]

Log = Callable[[str], None]

LLVM, SCALAR = "llvm_ad_rgb", "scalar_rgb"
MAX_LLVM_SHAPES = 150
_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
# Mitsuba envmap to_world for the UsdLux DomeLight lat-long layout on a Z-up stage (see the docstring)
_DOME_AXES = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], float) @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
_RENDER_PURPOSES = (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)
_MIN_ROUGHNESS = 0.03


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def srgb_to_linear(x) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x) -> np.ndarray:
    x = np.clip(np.asarray(x, np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def tonemap_image(linear: np.ndarray, exposure_ev: float = 0.0, tonemap: str = "aces") -> np.ndarray:
    """(H, W, 3) linear radiance -> uint8 sRGB: exposure, then 'aces' (Narkowicz fit) or 'none', then sRGB."""
    x = np.nan_to_num(np.asarray(linear, np.float32), nan=0.0, posinf=1e6, neginf=0.0)
    x = np.maximum(x * np.float32(2.0 ** exposure_ev), 0.0)
    if tonemap == "aces":
        x = np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)
    elif tonemap != "none":
        raise ValueError(f"unknown tonemap {tonemap!r} (use 'aces' or 'none')")
    return np.round(linear_to_srgb(x) * 255.0).astype(np.uint8)


class _Report:
    """Log sink that also records each distinct fallback once."""

    def __init__(self, log: Log | None):
        self.log = log
        self.fallbacks: list[str] = []

    def say(self, msg: str) -> None:
        if self.log is not None:
            self.log(msg)

    def fallback(self, msg: str) -> None:
        if msg not in self.fallbacks:
            self.fallbacks.append(msg)
            self.say(f"[mitsuba_preview] fallback: {msg}")


def _get(attr, tc, default=None):
    """Attribute value at tc (earliest sample when only time samples exist), else default."""
    if not attr or not attr.IsValid():
        return default
    v = attr.Get(tc)
    if v is None and tc == Usd.TimeCode.Default():
        v = attr.Get(Usd.TimeCode.EarliestTime())
    return default if v is None else v


def _py(v):
    """Plain Python value of a USD value (vectors become tuples of floats)."""
    if isinstance(v, (bool, int, float, str)) or v is None:
        return v
    if isinstance(v, Sdf.AssetPath):
        return v
    try:
        return tuple(float(c) for c in v)
    except TypeError:
        return str(v)


def _proto_path(prim: Usd.Prim) -> str:
    """Path of the prim in its prototype for instance proxies (identical for all instances)."""
    return str(prim.GetPrimInPrototype().GetPath()) if prim.IsInstanceProxy() else str(prim.GetPath())


def _resolve_asset(value, attr=None) -> str | None:
    """Local file for an asset-valued attribute (None when it does not exist on disk)."""
    if value is None:
        return None
    path = value.path if isinstance(value, Sdf.AssetPath) else str(value)
    if not path:
        return None
    cands = [getattr(value, "resolvedPath", "") or ""]
    if attr is not None:
        try:
            for spec in attr.GetPropertyStack(Usd.TimeCode.Default()):
                cands.append(Sdf.ComputeAssetPathRelativeToLayer(spec.layer, path))
        except Exception:  # noqa: BLE001 - odd proxies; the other candidates still apply
            pass
    cands.append(path)
    for c in cands:
        if c and not c.startswith(("http://", "https://")) and os.path.isfile(c):
            return os.path.abspath(c)
    return None


def _mat3(scale=(1.0, 1.0), rotate_deg=0.0, translate=(0.0, 0.0)) -> np.ndarray:
    """2D homogeneous T * R * S, the order of UsdTransform2d and of OmniPBR's texture_scale/rotate/translate."""
    c, s = math.cos(math.radians(rotate_deg)), math.sin(math.radians(rotate_deg))
    return np.array([[c * scale[0], -s * scale[1], translate[0]],
                     [s * scale[0], c * scale[1], translate[1]],
                     [0.0, 0.0, 1.0]])


def _is_identity(m: np.ndarray | None) -> bool:
    return m is None or np.allclose(m, np.eye(3), atol=1e-7)


# ---------------------------------------------------------------------------
# Textures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TexSrc:
    """One image lookup: file, colour decode, UsdUVTexture scale/bias, output channel and UV transform."""

    path: str
    srgb: bool | None = None            # None: sRGB for 8-bit images, linear otherwise (USD 'auto')
    channel: str = "rgb"                # rgb | r | g | b | a | luma | avg | max
    scale: tuple = (1.0, 1.0, 1.0, 1.0)
    bias: tuple = (0.0, 0.0, 0.0, 0.0)
    wrap: tuple = ("repeat", "repeat")
    uv: tuple | None = None             # row-major 3x3, USD (u, v, 1) -> texture coordinates
    uvset: str = "st"
    nearest: bool = False               # UsdUVTexture magFilter 'nearest' (otherwise bilinear)


_SRGB_LUT = srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0)


class _Images:
    """Image files decoded once (thread-safe); 8-bit data stays 8-bit until a lookup needs floats."""

    def __init__(self, max_px: int, report: _Report):
        self.max_px = max_px
        self.report = report
        self._raw: dict[str, tuple[np.ndarray, np.ndarray | None, float] | None] = {}
        self._lock = threading.Lock()
        self._path_locks: dict[str, threading.Lock] = {}

    def raw(self, path: str) -> tuple[np.ndarray, np.ndarray | None, float] | None:
        """(rgb (H, W, 3), alpha (H, W) or None, full-scale value) as stored, or None if unreadable."""
        with self._lock:
            if path in self._raw:
                return self._raw[path]
            plock = self._path_locks.setdefault(path, threading.Lock())
        with plock:
            if path not in self._raw:
                try:
                    got = self._load(path)
                except Exception as e:  # noqa: BLE001 - any decoder error means "no texture"
                    self.report.fallback(f"cannot read texture {path}: {e}")
                    got = None
                with self._lock:
                    self._raw[path] = got
        return self._raw[path]

    def _factor(self, h: int, w: int) -> int:
        return max(1, int(math.ceil(max(h, w) / self.max_px)))

    def _shrink(self, a: np.ndarray) -> np.ndarray:
        f = self._factor(*a.shape[:2])
        if f > 1:
            h, w = a.shape[0] // f * f, a.shape[1] // f * f
            a = a[:h, :w].reshape(h // f, f, w // f, f, *a.shape[2:]).mean(axis=(1, 3)).astype(a.dtype)
        return a

    def _load(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".hdr":       # Mitsuba's reader only takes "-Y h +X w" files; hdr.py reads all orientations
            try:
                return self._shrink(read_hdr(path)), None, 1.0
            except ValueError:
                pass            # not plain Radiance RGBE as hdr.py knows it: let Mitsuba try
        if ext in (".exr", ".hdr", ".pfm"):
            import mitsuba as mi
            bm = mi.Bitmap(path).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, False)
            return self._shrink(np.array(bm, np.float32)), None, 1.0
        img = Image.open(path)
        img.load()
        if img.mode.startswith("I;16") or img.mode in ("I", "F"):
            a = np.asarray(img)
            full = 1.0 if img.mode == "F" else 65535.0
            a = self._shrink(a.astype(np.float32 if img.mode == "F" else np.uint16))
            return np.ascontiguousarray(np.repeat(a[..., None], 3, axis=2)), None, full
        alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
        img = img.convert("RGBA" if alpha else "RGB")
        f = self._factor(img.height, img.width)
        if f > 1:
            img = img.reduce(f)
        a = np.asarray(img)
        return (a[..., :3], a[..., 3], 255.0) if alpha else (a, None, 255.0)

    @staticmethod
    def _decode(x: np.ndarray, full: float, srgb: bool) -> np.ndarray:
        if x.dtype == np.uint8:
            return _SRGB_LUT[x] if srgb else x.astype(np.float32) * np.float32(1.0 / 255.0)
        y = x.astype(np.float32) / np.float32(full)
        return srgb_to_linear(y) if srgb else y

    @staticmethod
    def _affine(x: np.ndarray, scale, bias) -> np.ndarray:
        scale, bias = np.asarray(scale, np.float32), np.asarray(bias, np.float32)
        return x if bool(np.all(scale == 1.0) and np.all(bias == 0.0)) else x * scale + bias

    def values(self, src: TexSrc) -> np.ndarray | None:
        """(H, W, 3) or (H, W, 1) float32 lookup result (a fresh array), or None when the file cannot be read."""
        got = self.raw(src.path)
        if got is None:
            return None
        rgb, alpha, full = got
        ch = src.channel
        srgb = src.srgb if src.srgb is not None else rgb.dtype == np.uint8
        if ch == "a":
            a = np.ones(rgb.shape[:2], np.float32) if alpha is None else alpha.astype(np.float32) / np.float32(full)
            return self._affine(a[..., None], src.scale[3], src.bias[3])
        if ch in ("r", "g", "b"):
            i = "rgb".index(ch)
            return self._affine(self._decode(rgb[..., i], full, srgb)[..., None], src.scale[i], src.bias[i])
        c = self._affine(self._decode(rgb, full, srgb), src.scale[:3], src.bias[:3])
        if ch == "rgb":
            return c
        if ch == "luma":
            return (c @ _LUMA)[..., None]
        return c.mean(axis=2, keepdims=True) if ch == "avg" else c.max(axis=2, keepdims=True)


# ---------------------------------------------------------------------------
# MDL source parsing (only what OmniPBR-style materials need)
# ---------------------------------------------------------------------------

def _strip_mdl_comments(src: str) -> str:
    out, i, n, in_str = [], 0, len(src), False
    while i < n:
        c = src[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(src[i + 1])
                i += 2
                continue
            in_str = c != '"'
            i += 1
        elif c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _balanced_end(s: str, i: int) -> int:
    """Index just after the bracket group opened at s[i] ('(' or '['), string literals skipped."""
    depth, in_str, n = 0, False, len(s)
    while i < n:
        c = s[i]
        if in_str:
            if c == "\\":
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _split_top(s: str) -> list[str]:
    parts, depth, in_str, start, i = [], 0, False, 0, 0
    while i < len(s):
        c = s[i]
        if in_str:
            if c == "\\":
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(s[start:i])
            start = i + 1
        i += 1
    if s[start:].strip():
        parts.append(s[start:])
    return parts


_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?[fF]?$")


def _mdl_value(v: str):
    """('tex', path | None, gamma) for texture_2d(...), a float tuple for colorN/floatN, float, bool or str."""
    v = v.strip()
    m = re.match(r"texture_2d\s*\((.*)\)\s*$", v, re.S)
    if m:
        args = _split_top(m.group(1))
        pm = re.match(r'\s*"((?:[^"\\]|\\.)*)"', args[0]) if args else None
        if not pm:
            return ("tex", None, "default")
        gamma = "srgb" if len(args) > 1 and "srgb" in args[1] else "linear" if len(args) > 1 and "linear" in args[1] \
            else "default"
        return ("tex", pm.group(1), gamma)
    m = re.match(r"(?:color|float[234])\s*\((.*)\)\s*$", v, re.S)
    if m:
        vals = [float(_mdl_value(a)) for a in _split_top(m.group(1))]
        n = 3 if v.startswith("color") else int(v[5])
        return tuple(vals * n) if len(vals) == 1 else tuple(vals)
    if v in ("true", "false"):
        return v == "true"
    if _NUM.match(v):
        return float(v.rstrip("fF"))
    return v.split("::")[-1]


def parse_mdl(text: str, sub_identifier: str | None = None) -> tuple[str, dict[str, Any]] | None:
    """(callee, {argument: value}) of ``export material <sub> ... = callee(args)``; None if it is not that form.

    Handles NVIDIA's generated materials (``= ::OmniPBR::OmniPBR(diffuse_texture: texture_2d("x.png",
    ::tex::gamma_srgb), ...)``); anything with a ``let`` body or typed parameter defaults gives None.
    """
    src = _strip_mdl_comments(text)
    defs = list(re.finditer(r"\bmaterial\s+(\w+)\s*\(", src))
    if not defs:
        return None
    pick = next((d for d in defs if d.group(1) == sub_identifier), defs[0] if sub_identifier is None else None)
    if pick is None:
        return None
    i = _balanced_end(src, pick.end() - 1)
    while i < len(src) and src[i].isspace():
        i += 1
    if src.startswith("[[", i):                 # annotation block
        i = _balanced_end(src, i)
        while i < len(src) and src[i].isspace():
            i += 1
    if i >= len(src) or src[i] != "=":
        return None
    m = re.compile(r"\s*([\w:]+)\s*\(").match(src, i + 1)
    if not m or m.group(1) == "let":
        return None
    body = src[m.end():_balanced_end(src, m.end() - 1) - 1]
    args: dict[str, Any] = {}
    for a in _split_top(body):
        am = re.match(r"^\s*(\w+)\s*:(?!:)\s*(.*)$", a, re.S)
        if am:
            args[am.group(1)] = _mdl_value(am.group(2))
    return m.group(1), args


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

@dataclass
class MaterialSpec:
    """Renderer-independent PBR description of one USD material (what the Mitsuba BSDF is built from)."""

    name: str
    kind: str = "preview"               # preview | omnipbr | default | fallback
    uv_set: str = "st"
    base_color: tuple = (0.5, 0.5, 0.5)
    base_tex: TexSrc | None = None
    base_scale: float = 1.0             # OmniPBR albedo_brightness / add / desaturation apply to the texture only
    base_add: float = 0.0
    base_desat: float = 0.0
    base_tint: tuple = (1.0, 1.0, 1.0)
    ao_tex: TexSrc | None = None
    ao_weight: float = 0.0
    roughness: float = 0.5              # lerp(roughness, roughness_tex, roughness_mix)
    roughness_tex: TexSrc | None = None
    roughness_mix: float = 1.0
    metallic: float = 0.0
    metallic_tex: TexSrc | None = None
    metallic_mix: float = 1.0
    specular: float = 0.5
    normal_tex: TexSrc | None = None
    normal_encoded: bool = True         # True: texel in [0, 1] (n = 2t - 1); False: texel is already n
    normal_strength: float = 1.0
    flip_u: bool = False
    flip_v: bool = False
    opacity: float = 1.0
    opacity_tex: TexSrc | None = None
    opacity_threshold: float = 0.0
    emission: tuple = (0.0, 0.0, 0.0)
    emission_scale: float = 1.0
    emission_tex: TexSrc | None = None
    emission_mask: TexSrc | None = None
    clearcoat: float = 0.0
    clearcoat_roughness: float = 0.01
    vertex_color: bool = False          # base colour from the mesh's displayColor primvar

    @property
    def emissive(self) -> bool:
        return self.emission_scale > 0 and (self.emission_tex is not None or max(self.emission) > 0)


# Built-in OmniPBR inputs (Kit's OmniPBR.mdl 2.1 / OmniPBR_ClearCoat.mdl); textures default to unset.
_OMNIPBR_DEFAULTS: dict[str, Any] = {
    "diffuse_color_constant": (0.2, 0.2, 0.2), "albedo_desaturation": 0.0, "albedo_add": 0.0,
    "albedo_brightness": 1.0, "diffuse_tint": (1.0, 1.0, 1.0),
    "reflection_roughness_constant": 0.5, "reflection_roughness_texture_influence": 0.0,
    "metallic_constant": 0.0, "metallic_texture_influence": 0.0, "specular_level": 0.5,
    "enable_ORM_texture": False, "ao_to_diffuse": 0.0,
    "enable_emission": False, "emissive_color": (1.0, 0.1, 0.1), "emissive_intensity": 40.0,
    "enable_opacity": False, "opacity_constant": 1.0, "enable_opacity_texture": False,
    "opacity_mode": "mono_average", "opacity_threshold": 0.0,
    "bump_factor": 1.0, "flip_tangent_u": False, "flip_tangent_v": True,
    "project_uvw": False, "uv_space_index": 0,
    "texture_translate": (0.0, 0.0), "texture_rotate": 0.0, "texture_scale": (1.0, 1.0),
    "enable_clearcoat": False, "clearcoat_weight": 1.0, "clearcoat_reflection_roughness": 0.0,
}
_OMNIPBR_NAMES = ("omnipbr", "omnipbr_opacity", "omnipbr_clearcoat", "omnipbr_clearcoat_opacity",
                  "aperturepbr_opacity")
# Inputs that mark a shader as OmniPBR-style when its MDL file cannot be read (OmniSurface, OmniGlass and
# custom MDL have none of them)
_OMNIPBR_INPUTS = ("diffuse_texture", "diffuse_color_constant", "reflection_roughness_constant",
                   "reflectionroughness_texture", "ORM_texture", "normalmap_texture", "metallic_constant",
                   "metallic_texture", "opacity_constant", "opacity_texture", "emissive_color",
                   "emissive_color_texture", "albedo_brightness", "texture_scale", "enable_ORM_texture",
                   "enable_emission", "enable_opacity")
_SRGB_INPUTS = ("diffuse_texture", "emissive_color_texture")
_MONO = {"mono_alpha": "a", "mono_average": "avg", "mono_luminance": "luma", "mono_maximum": "max",
         0: "a", 1: "avg", 2: "luma", 3: "max"}
_COLOR_WORDS = ("diffuse", "albedo", "base_color", "basecolor")


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class _Materials:
    """USD material prims -> MaterialSpec, cached per (prototype) material path."""

    def __init__(self, tc, report: _Report, flip_green_preview: bool | None, flip_green_mdl: bool | None):
        self.tc = tc
        self.report = report
        self.flip_preview = flip_green_preview
        self.flip_mdl = flip_green_mdl
        self._cache: dict[Any, MaterialSpec] = {}

    def default(self, rgb: tuple | None = None, vertex_color: bool = False) -> MaterialSpec:
        """Unbound mesh: displayColor if authored, else grey 0.5, roughness 0.6."""
        key = ("vertex",) if vertex_color else ("rgb",) + tuple(round(float(c), 5) for c in (rgb or (0.5,) * 3))
        if key not in self._cache:
            self._cache[key] = MaterialSpec("displayColor" if rgb or vertex_color else "default", kind="default",
                                            base_color=tuple(rgb or (0.5, 0.5, 0.5)), roughness=0.6,
                                            vertex_color=vertex_color)
        return self._cache[key]

    def spec_for(self, material: UsdShade.Material) -> MaterialSpec:
        key = _proto_path(material.GetPrim())
        if key not in self._cache:
            self._cache[key] = self._build(material, material.GetPrim().GetName())
        return self._cache[key]

    def _build(self, material: UsdShade.Material, name: str) -> MaterialSpec:
        shader = None
        for ctx in ("mdl", ""):                 # Isaac's RTX renderer prefers the MDL surface
            src = material.ComputeSurfaceSource(ctx)
            if src and src[0] and src[0].GetPrim().IsValid():
                shader = src[0]
                break
        if shader is None:
            self.report.fallback(f"{name}: material has no surface shader; using grey")
            return MaterialSpec(name, kind="fallback", roughness=0.6)
        try:
            if shader.GetShaderId() == "UsdPreviewSurface":
                return self._preview(shader, name)
            if shader.GetImplementationSource() == UsdShade.Tokens.sourceAsset:
                return self._mdl(shader, name)
        except Exception as e:  # noqa: BLE001 - a broken material must not abort the whole preview
            self.report.fallback(f"{name}: material could not be read ({type(e).__name__}: {e}); using grey")
            return MaterialSpec(name, kind="fallback", roughness=0.6)
        self.report.fallback(f"{name}: unsupported shader {shader.GetShaderId() or shader.GetPath()}; using grey")
        return MaterialSpec(name, kind="fallback", roughness=0.6)

    # -- shader graph access -------------------------------------------------

    def _source(self, sh: UsdShade.Shader, name: str):
        """("value", v, attr) | ("out", UsdShade.Shader, output name) | None for a shader input."""
        inp = sh.GetInput(name)
        if not inp:
            return None
        for attr in UsdShade.Utils.GetValueProducingAttributes(inp):
            if attr.GetName().startswith("outputs:"):
                return ("out", UsdShade.Shader(attr.GetPrim()), attr.GetName()[len("outputs:"):])
            v = _get(attr, self.tc)
            if v is not None:
                return ("value", v, attr)
        return None

    def _val(self, sh: UsdShade.Shader, name: str, default):
        r = self._source(sh, name)
        return _py(r[1]) if r is not None and r[0] == "value" else default

    def _st_chain(self, node: UsdShade.Shader, input_name: str = "st") -> tuple[str, np.ndarray | None]:
        r = self._source(node, input_name)
        if r is None or r[0] != "out":
            return "st", None
        n = r[1]
        sid = n.GetShaderId() or ""
        if sid.startswith("UsdPrimvarReader"):
            return str(self._val(n, "varname", "st") or "st"), None
        if sid == "UsdTransform2d":
            varname, inner = self._st_chain(n, "in")
            m = _mat3(tuple(self._val(n, "scale", (1.0, 1.0))), float(self._val(n, "rotation", 0.0)),
                      tuple(self._val(n, "translation", (0.0, 0.0))))
            return varname, m if inner is None else m @ inner
        self.report.fallback(f"texture coordinates from unsupported node {sid or n.GetPath()}")
        return "st", None

    def _uvtexture(self, node: UsdShade.Shader, out: str) -> TexSrc | None:
        f = self._source(node, "file")
        if f is None or f[0] != "value":
            return None
        path = _resolve_asset(f[1], f[2])
        if path is None:
            self.report.fallback(f"texture not found: {getattr(f[1], 'path', f[1])}")
            return None
        srgb = {"sRGB": True, "raw": False}.get(str(self._val(node, "sourceColorSpace", "auto")))
        scale = tuple(self._val(node, "scale", (1.0,) * 4))
        bias = tuple(self._val(node, "bias", (0.0,) * 4))
        wrap = tuple(str(self._val(node, w, "useMetadata")) for w in ("wrapS", "wrapT"))
        uvset, m = self._st_chain(node)
        return TexSrc(path, srgb, out if out in ("r", "g", "b", "a") else "rgb", scale, bias, wrap,
                      None if _is_identity(m) else tuple(float(x) for x in m.ravel()), uvset,
                      str(self._val(node, "magFilter", "linear")) == "nearest")

    # -- UsdPreviewSurface ---------------------------------------------------

    def _preview(self, sh: UsdShade.Shader, name: str) -> MaterialSpec:
        s = MaterialSpec(name, kind="preview")
        uvsets: list[str] = []

        def param(inp: str, default):
            r = self._source(sh, inp)
            if r is None:
                return default, None
            if r[0] == "value":
                return _py(r[1]), None
            if r[1].GetShaderId() == "UsdUVTexture":
                t = self._uvtexture(r[1], r[2])
                if t is not None:
                    uvsets.append(t.uvset)
                    return default, t
            else:
                self.report.fallback(f"{name}: {inp} is connected to unsupported node "
                                     f"{r[1].GetShaderId() or r[1].GetPath()}")
            return default, None

        s.base_color, s.base_tex = param("diffuseColor", (0.18, 0.18, 0.18))
        s.roughness, s.roughness_tex = param("roughness", 0.5)
        s.metallic, s.metallic_tex = param("metallic", 0.0)
        s.opacity, s.opacity_tex = param("opacity", 1.0)
        s.opacity_threshold = float(param("opacityThreshold", 0.0)[0])
        s.emission, s.emission_tex = param("emissiveColor", (0.0, 0.0, 0.0))
        s.clearcoat = float(param("clearcoat", 0.0)[0])
        s.clearcoat_roughness = float(param("clearcoatRoughness", 0.01)[0])
        if int(param("useSpecularWorkflow", 0)[0]):
            spec_c = param("specularColor", (0.0, 0.0, 0.0))[0]
            s.metallic, s.specular = 0.0, float(np.clip(np.mean(spec_c) / 0.08, 0.0, 1.0))
        _, s.normal_tex = param("normal", None)
        s.normal_encoded = False
        s.flip_v = bool(self.flip_preview)
        if uvsets:
            s.uv_set = uvsets[0]
            if len(set(uvsets)) > 1:
                self.report.fallback(f"{name}: textures use several UV sets {sorted(set(uvsets))}; using {uvsets[0]}")
        return s

    # -- MDL -------------------------------------------------------------------

    def _mdl_inputs(self, sh: UsdShade.Shader) -> dict[str, Any]:
        """Authored shader inputs; textures as ("asset", AssetPath, attr, srgb | None)."""
        out: dict[str, Any] = {}
        for inp in sh.GetInputs():
            for attr in UsdShade.Utils.GetValueProducingAttributes(inp):
                if attr.GetName().startswith("outputs:"):
                    continue
                v = _get(attr, self.tc)
                if v is None:
                    continue
                if isinstance(v, Sdf.AssetPath):
                    cs = attr.GetColorSpace()
                    out[inp.GetBaseName()] = ("asset", v, attr, {"sRGB": True, "raw": False}.get(cs))
                else:
                    out[inp.GetBaseName()] = _py(v)
                break
        return out

    def _texture(self, name: str, params: dict, key: str, mdl_dir: str | None, channel: str = "rgb",
                 uv: tuple | None = None, uvset: str = "st") -> tuple[TexSrc | None, str]:
        """(TexSrc | None, state) for a texture input; state is 'unset', 'missing' or 'ok'."""
        v = params.get(key)
        if not isinstance(v, tuple) or not v or v[0] not in ("tex", "asset"):
            return None, "unset"
        if v[0] == "asset":
            if not v[1].path:
                return None, "unset"
            path, srgb, shown = _resolve_asset(v[1], v[2]), v[3], v[1].path
        else:
            if not v[1]:
                return None, "unset"
            shown = v[1]
            cand = os.path.normpath(os.path.join(mdl_dir, v[1])) if mdl_dir and not os.path.isabs(v[1]) else v[1]
            path, srgb = (cand if os.path.isfile(cand) else None), {"srgb": True, "linear": False}.get(v[2])
        if path is None:
            self.report.fallback(f"{name}: texture {shown} ({key}) not found; input ignored")
            return None, "missing"
        return TexSrc(path, key in _SRGB_INPUTS if srgb is None else srgb, channel, uv=uv, uvset=uvset), "ok"

    def _mdl(self, sh: UsdShade.Shader, name: str) -> MaterialSpec:
        asset = sh.GetSourceAsset("mdl")
        sub = (sh.GetSourceAssetSubIdentifier("mdl") or "").split("::")[-1]     # '.::materials::Foo::Foo' -> 'Foo'
        aname = os.path.basename(asset.path).lower() if asset else ""
        params: dict[str, Any] = {}
        mdl_dir: str | None = None
        omni, mdl_missing = False, False
        if aname.replace(".mdl", "") in _OMNIPBR_NAMES or sub.lower() in _OMNIPBR_NAMES:
            omni, params = True, dict(_OMNIPBR_DEFAULTS)
            if "opacity" in aname or sub.lower().endswith("opacity"):
                params["enable_opacity"] = True
        else:
            path = _resolve_asset(asset, sh.GetPrim().GetAttribute("info:mdl:sourceAsset")) if asset else None
            parsed = None
            if path:
                mdl_dir = os.path.dirname(path)
                with open(path, encoding="utf-8", errors="replace") as f:
                    parsed = parse_mdl(f.read(), sub or None)
            else:
                mdl_missing = True
            if parsed and parsed[0].split("::")[-1].lower().startswith("omnipbr"):
                omni, params = True, {**_OMNIPBR_DEFAULTS, **parsed[1]}
            elif mdl_missing:
                # only the shader's own inputs tell what it is: OmniPBR's, or something else (diffuse-like fallback)
                omni = any(sh.GetInput(k) for k in _OMNIPBR_INPUTS)
                shown = getattr(asset, "path", "?")
                if omni:
                    params = dict(_OMNIPBR_DEFAULTS)
                    self.report.fallback(f"{name}: MDL {shown} not found locally; assuming OmniPBR with its "
                                         "built-in defaults")
                else:
                    self.report.fallback(f"{name}: MDL {shown} not found locally and the shader authors no "
                                         "OmniPBR inputs; not treated as OmniPBR")
            else:
                params = dict(parsed[1]) if parsed else {}
        authored = self._mdl_inputs(sh)
        if omni and mdl_missing:    # a texture authored without its influence input only makes sense at influence 1
            if "ORM_texture" in authored and "enable_ORM_texture" not in authored:
                authored["enable_ORM_texture"] = True
            if "reflectionroughness_texture" in authored or "ORM_texture" in authored:
                authored.setdefault("reflection_roughness_texture_influence", 1.0)
            if "metallic_texture" in authored:
                authored.setdefault("metallic_texture_influence", 1.0)
        params.update(authored)
        if not omni:
            return self._generic_mdl(name, sub, params, mdl_dir)
        return self._omnipbr(name, params, mdl_dir)

    def _generic_mdl(self, name: str, sub: str, params: dict, mdl_dir: str | None) -> MaterialSpec:
        self.report.fallback(f"{name}: unsupported MDL '{sub}'; using its diffuse-like inputs, roughness 0.6")
        s = MaterialSpec(name, kind="fallback", roughness=0.6)
        for key, v in params.items():
            lk = key.lower()
            if not any(w in lk for w in _COLOR_WORDS) or "normal" in lk:
                continue
            if isinstance(v, tuple) and v and v[0] in ("tex", "asset"):
                t, _ = self._texture(name, params, key, mdl_dir)
                if t is not None and s.base_tex is None:
                    s.base_tex = TexSrc(t.path, t.srgb if t.srgb is not None else True)
            elif isinstance(v, tuple) and len(v) == 3:
                s.base_color = v
        return s

    def _omnipbr(self, name: str, p: dict, mdl_dir: str | None) -> MaterialSpec:
        scale = tuple(p["texture_scale"])
        m = _mat3(scale, float(p["texture_rotate"]), tuple(p["texture_translate"]))
        uv = None if _is_identity(m) else tuple(float(x) for x in m.ravel())
        index = int(p["uv_space_index"])
        uvset = "st" if index == 0 else f"st_{index}"
        if p["project_uvw"]:
            self.report.fallback(f"{name}: project_uvw (triplanar) is not supported; using the mesh UVs")

        def tex(key: str, channel: str = "rgb"):
            return self._texture(name, p, key, mdl_dir, channel, uv, uvset)

        s = MaterialSpec(name, kind="omnipbr", uv_set=uvset)
        s.base_color, (s.base_tex, _) = tuple(p["diffuse_color_constant"]), tex("diffuse_texture")
        s.base_scale, s.base_add = float(p["albedo_brightness"]), float(p["albedo_add"])
        s.base_desat, s.base_tint = float(p["albedo_desaturation"]), tuple(p["diffuse_tint"])
        orm = bool(p["enable_ORM_texture"])
        s.ao_weight = float(p["ao_to_diffuse"])
        if s.ao_weight > 0:
            s.ao_tex = tex("ORM_texture", "r")[0] if orm else tex("ao_texture", "avg")[0]

        def blended(const: str, influence: str, key: str, orm_channel: str):
            """(constant, texture, mix): lerp(constant, texture, influence); ORM.g / ORM.b when enabled."""
            c, mix = float(p[const]), float(p[influence])
            t, state = tex("ORM_texture", orm_channel) if orm else tex(key)
            if t is None:       # an unset texture reads as 0 in MDL; a missing file is ignored (constant kept)
                return (_lerp(c, 0.0, mix) if state == "unset" else c), None, mix
            return c, t, mix

        s.roughness, s.roughness_tex, s.roughness_mix = blended(
            "reflection_roughness_constant", "reflection_roughness_texture_influence",
            "reflectionroughness_texture", "g")
        s.metallic, s.metallic_tex, s.metallic_mix = blended(
            "metallic_constant", "metallic_texture_influence", "metallic_texture", "b")
        s.specular = float(p["specular_level"])
        s.normal_tex = tex("normalmap_texture")[0]
        s.normal_encoded, s.normal_strength = True, float(p["bump_factor"])
        s.flip_u = bool(p["flip_tangent_u"])
        s.flip_v = bool(p["flip_tangent_v"]) if self.flip_mdl is None else bool(self.flip_mdl)
        if bool(p["enable_opacity"]):
            s.opacity_threshold = float(p["opacity_threshold"])
            s.opacity = float(p["opacity_constant"])
            if bool(p["enable_opacity_texture"]):
                mode = p["opacity_mode"]
                mono = _MONO.get(mode if isinstance(mode, str) else int(mode), "avg")
                s.opacity_tex = tex("opacity_texture", mono)[0]
        if bool(p["enable_emission"]):
            s.emission = tuple(p["emissive_color"])
            s.emission_scale = float(p["emissive_intensity"])
            s.emission_tex = tex("emissive_color_texture")[0]
            s.emission_mask = tex("emissive_mask_texture", "avg")[0]
        if bool(p["enable_clearcoat"]):
            s.clearcoat = float(p["clearcoat_weight"])
            s.clearcoat_roughness = float(p["clearcoat_reflection_roughness"])
        return s


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass
class MeshArrays:
    """A triangle mesh in Mitsuba's layout (one entry per vertex)."""

    positions: np.ndarray               # (N, 3) float32
    faces: np.ndarray                   # (T, 3) uint32
    normals: np.ndarray | None = None   # (N, 3) float32, unit
    uv: np.ndarray | None = None        # (N, 2) float32
    colors: np.ndarray | None = None    # (N, 3) float32


@dataclass
class _Attr:
    """A primvar-like attribute: values plus how they attach to the mesh."""

    interp: str                         # constant | uniform | vertex | varying | faceVarying
    data: np.ndarray


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > 1e-12, v / np.maximum(n, 1e-12), np.array([0.0, 0.0, 1.0], v.dtype)).astype(np.float32)


class _RawMesh:
    """A UsdGeomMesh triangulated by fan, with its normals, UV sets and colours kept in USD interpolation."""

    def __init__(self, prim: Usd.Prim, tc, report: _Report):
        mesh = UsdGeom.Mesh(prim)
        self.prim, self.tc, self.report = prim, tc, report
        pts = _get(mesh.GetPointsAttr(), tc)
        counts = _get(mesh.GetFaceVertexCountsAttr(), tc)
        idx = _get(mesh.GetFaceVertexIndicesAttr(), tc)
        self.ok = pts is not None and counts is not None and idx is not None and len(pts) > 0
        if not self.ok:
            return
        self.points = np.asarray(pts, np.float32).reshape(-1, 3)
        counts = np.asarray(counts, np.int64)
        self.idx = np.asarray(idx, np.int64)
        bad_index = len(self.idx) > 0 and (self.idx.min() < 0 or self.idx.max() >= len(self.points))
        if counts.sum() != len(self.idx) or bad_index:
            report.fallback(f"{prim.GetPath()}: inconsistent mesh topology; skipped")
            self.ok = False
            return
        self.n_faces = len(counts)
        n_tri = np.maximum(counts - 2, 0)
        starts = np.cumsum(counts) - counts
        self.tri_face = np.repeat(np.arange(self.n_faces), n_tri)
        first = np.repeat(starts, n_tri)
        k = np.arange(len(self.tri_face)) - np.repeat(np.cumsum(n_tri) - n_tri, n_tri)
        self.tri_corner = np.stack([first, first + k + 1, first + k + 2], axis=1)
        if _get(mesh.GetOrientationAttr(), tc) == UsdGeom.Tokens.leftHanded:
            self.tri_corner = self.tri_corner[:, [0, 2, 1]]
        self.normals = self._normals(mesh)
        # subdivisionScheme 'none' without authored normals means faceted shading
        self.flat = self.normals is None and _get(mesh.GetSubdivisionSchemeAttr(), tc) == UsdGeom.Tokens.none
        self._smooth: np.ndarray | None = None

    def _expected(self, interp: str) -> int:
        return {"constant": 1, "uniform": self.n_faces, "vertex": len(self.points), "varying": len(self.points),
                "faceVarying": len(self.idx)}[interp]

    def _attr(self, values, interp: str, what: str) -> _Attr | None:
        if values is None or interp not in ("constant", "uniform", "vertex", "varying", "faceVarying"):
            return None
        data = np.asarray(values, np.float32)
        if data.size == 0:
            return None
        data = data.reshape(len(data), -1)
        if len(data) != self._expected(interp):
            self.report.fallback(f"{self.prim.GetPath()}: {what} has {len(data)} values for '{interp}' "
                                 f"interpolation (expected {self._expected(interp)}); ignored")
            return None
        return _Attr(interp, data)

    def _normals(self, mesh: UsdGeom.Mesh) -> _Attr | None:
        pv = UsdGeom.PrimvarsAPI(self.prim).GetPrimvar("normals")
        if pv and pv.HasValue():
            return self._attr(pv.ComputeFlattened(self.tc), pv.GetInterpolation(), "normals")
        attr = mesh.GetNormalsAttr()
        if attr and attr.HasValue():
            return self._attr(_get(attr, self.tc), mesh.GetNormalsInterpolation(), "normals")
        return None

    def uv_attr(self, wanted: str) -> _Attr | None:
        """The UV set for a material's varname: that name, then st/st0/UVMap/map1, then any texCoord2f primvar."""
        api = UsdGeom.PrimvarsAPI(self.prim)
        names = [wanted, wanted.replace("_", ""), "st", "st0", "UVMap", "map1"]
        names += [p.GetPrimvarName() for p in api.GetPrimvars() if str(p.GetTypeName()).startswith("texCoord2")]
        for n in names:
            pv = api.GetPrimvar(n)
            if pv and pv.HasValue() and str(pv.GetTypeName()).startswith(("texCoord2", "float2", "double2")):
                a = self._attr(pv.ComputeFlattened(self.tc), pv.GetInterpolation(), f"primvar {n}")
                if a is not None and a.data.shape[1] == 2:
                    return a
        return None

    def color_attr(self) -> _Attr | None:
        pv = UsdGeom.PrimvarsAPI(self.prim).GetPrimvar("displayColor")
        if pv and pv.HasValue():
            a = self._attr(pv.ComputeFlattened(self.tc), pv.GetInterpolation(), "displayColor")
            return a if a is not None and a.data.shape[1] == 3 else None
        return None

    def smooth_normals(self) -> np.ndarray:
        """Area-weighted vertex normals of the whole mesh."""
        if self._smooth is None:
            tri = self.idx[self.tri_corner]
            p = self.points.astype(np.float64)
            fn = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
            vn = np.zeros((len(p), 3))
            for corner in range(3):
                for axis in range(3):
                    vn[:, axis] += np.bincount(tri[:, corner], weights=fn[:, axis], minlength=len(p))
            self._smooth = _unit(vn)
        return self._smooth

    def _rows(self, a: _Attr, tc: np.ndarray, tf: np.ndarray) -> np.ndarray:
        """Per triangle corner (3T, k) values."""
        if a.interp == "constant":
            return np.broadcast_to(a.data[0], (tc.size, a.data.shape[1])).copy()
        if a.interp == "uniform":
            return np.repeat(a.data[tf], 3, axis=0)
        if a.interp == "faceVarying":
            return a.data[tc.ravel()]
        return a.data[self.idx[tc].ravel()]

    def build(self, uv: _Attr | None, colors: _Attr | None, sel: np.ndarray | None = None) -> MeshArrays:
        """Mitsuba-ready arrays for all triangles (or those chosen by the index array ``sel``)."""
        tc = self.tri_corner if sel is None else self.tri_corner[sel]
        tf = self.tri_face if sel is None else self.tri_face[sel]
        nrm = self.normals
        shared = all(a is None or a.interp in ("constant", "vertex", "varying") for a in (nrm, uv, colors)) \
            and not self.flat
        if shared:      # one vertex per point: smooth shading keeps its shared normals
            tri = self.idx[tc]
            if sel is None:
                used, faces = None, tri
            else:
                used, inv = np.unique(tri.ravel(), return_inverse=True)
                faces = inv.reshape(-1, 3)
            n_pts = len(self.points) if used is None else len(used)

            def per_vertex(a: _Attr) -> np.ndarray:
                if a.interp == "constant":
                    return np.broadcast_to(a.data[0], (n_pts, a.data.shape[1])).copy()
                return a.data if used is None else a.data[used]

            pos = self.points if used is None else self.points[used]
            normals = per_vertex(nrm) if nrm is not None else (
                self.smooth_normals() if used is None else self.smooth_normals()[used])
            return MeshArrays(pos.astype(np.float32), faces.astype(np.uint32), _unit(normals),
                              None if uv is None else per_vertex(uv).astype(np.float32),
                              None if colors is None else per_vertex(colors).astype(np.float32))
        pos = self.points[self.idx[tc].ravel()]
        faces = np.arange(tc.size, dtype=np.uint32).reshape(-1, 3)
        if nrm is not None:
            normals = self._rows(nrm, tc, tf)
        elif self.flat:
            p = pos.reshape(-1, 3, 3).astype(np.float64)
            normals = np.repeat(_unit(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])), 3, axis=0)
        else:
            normals = self.smooth_normals()[self.idx[tc].ravel()]
        return MeshArrays(pos.astype(np.float32), faces, _unit(normals),
                          None if uv is None else self._rows(uv, tc, tf).astype(np.float32),
                          None if colors is None else self._rows(colors, tc, tf).astype(np.float32))


def _bake(a: MeshArrays, m: np.ndarray) -> MeshArrays:
    """Mesh arrays transformed by a USD row-vector matrix; the winding flips when it mirrors."""
    r = m[:3, :3]
    pos = (a.positions.astype(np.float64) @ r + m[3, :3]).astype(np.float32)
    nrm = None
    if a.normals is not None:
        nrm = _unit(a.normals.astype(np.float64) @ np.linalg.inv(r).T)
    faces = a.faces[:, [0, 2, 1]] if np.linalg.det(r) < 0 else a.faces
    return MeshArrays(pos, np.ascontiguousarray(faces), nrm, a.uv, a.colors)


def _mirrored_triangles(a: MeshArrays) -> np.ndarray:
    """Per triangle: True when its UVs wind against its normal (cross(dp/du, dp/dv) points inward)."""
    tri, uv = a.positions[a.faces].astype(np.float64), a.uv[a.faces].astype(np.float64)
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    d1, d2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    uv_det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    facing = np.einsum("tc,tc->t", np.cross(e1, e2), a.normals[a.faces].astype(np.float64).sum(axis=1))
    return uv_det * facing < 0


def _take_triangles(a: MeshArrays, keep: np.ndarray) -> MeshArrays:
    """The triangles selected by the boolean mask ``keep``, with only the vertices they use."""
    used, inv = np.unique(a.faces[keep].ravel(), return_inverse=True)
    return MeshArrays(a.positions[used], inv.reshape(-1, 3).astype(np.uint32),
                      *(None if x is None else x[used] for x in (a.normals, a.uv, a.colors)))


def _split_by_uv_handedness(a: MeshArrays) -> list[tuple[MeshArrays, bool]]:
    """[(arrays, uv_mirrored)]: the triangles of ``a`` grouped by UV handedness (one entry if they all agree)."""
    if a.uv is None or a.normals is None or len(a.faces) == 0:
        return [(a, False)]
    mirrored = _mirrored_triangles(a)
    if not mirrored.any():
        return [(a, False)]
    if mirrored.all():
        return [(a, True)]
    return [(_take_triangles(a, ~mirrored), False), (_take_triangles(a, mirrored), True)]


_IMPLICIT = (UsdGeom.Cube, UsdGeom.Sphere, UsdGeom.Cylinder, UsdGeom.Cone, UsdGeom.Capsule, UsdGeom.Plane)


def _lathe(profile: np.ndarray, nu: int) -> MeshArrays:
    """Surface of revolution around Z from rows (radius, z, n_radial, n_z), bottom to top (CCW seen from outside)."""
    ang = np.linspace(0.0, 2.0 * np.pi, nu + 1)
    ca, sa = np.cos(ang)[None, :], np.sin(ang)[None, :]
    r, z, nr, nz = (profile[:, k][:, None] for k in range(4))
    pos = np.stack([r * ca, r * sa, z + 0 * ca], axis=-1).reshape(-1, 3)
    nrm = np.stack([nr * ca, nr * sa, nz + 0 * ca], axis=-1).reshape(-1, 3)
    shape = (len(profile), nu + 1)
    uv = np.stack([np.broadcast_to(ang / (2 * np.pi), shape),
                   np.broadcast_to(np.linspace(0, 1, len(profile))[:, None], shape)], -1).reshape(-1, 2)
    k, i = np.meshgrid(np.arange(len(profile) - 1), np.arange(nu), indexing="ij")
    a, b = (k * (nu + 1) + i).ravel(), (k * (nu + 1) + i + 1).ravel()
    c, d = b + nu + 1, a + nu + 1
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    tri = pos[faces]                                            # drop zero-area triangles (poles, repeated rings)
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return MeshArrays(pos.astype(np.float32), faces[area2 > 1e-12].astype(np.uint32), nrm.astype(np.float32),
                      uv.astype(np.float32))


def _implicit_arrays(prim: Usd.Prim, tc) -> MeshArrays | None:
    """Tessellation of a UsdGeom Cube / Sphere / Cylinder / Cone / Capsule / Plane in its own space."""
    def attr(name, default):
        return float(_get(prim.GetAttribute(name), tc, default))

    axis = str(_get(prim.GetAttribute("axis"), tc, "Z"))
    nu = 64
    if prim.IsA(UsdGeom.Cube):
        h = attr("size", 2.0) / 2.0
        pos, nrm, uv, faces = [], [], [], []
        for k in range(3):
            ea, eb = np.eye(3)[(k + 1) % 3], np.eye(3)[(k + 2) % 3]
            for sign in (1.0, -1.0):                    # ea x (sign * eb) = sign * e_k: CCW from outside
                n = sign * np.eye(3)[k]
                base = len(pos)
                pos += [h * n + h * (sa * ea + sb * sign * eb) for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                nrm += [n] * 4
                uv += [(0, 0), (1, 0), (1, 1), (0, 1)]
                faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        return MeshArrays(np.array(pos, np.float32), np.array(faces, np.uint32), np.array(nrm, np.float32),
                          np.array(uv, np.float32))
    if prim.IsA(UsdGeom.Plane):
        hw, hl = attr("width", 2.0) / 2, attr("length", 2.0) / 2
        pos = np.array([(-hw, -hl, 0), (hw, -hl, 0), (hw, hl, 0), (-hw, hl, 0)], np.float32)
        a = MeshArrays(pos, np.array([(0, 1, 2), (0, 2, 3)], np.uint32), np.tile(np.float32([0, 0, 1]), (4, 1)),
                       np.array([(0, 0), (1, 0), (1, 1), (0, 1)], np.float32))
    else:
        rad = attr("radius", 1.0)
        if prim.IsA(UsdGeom.Sphere):
            t = np.linspace(np.pi, 0.0, 33)                                  # bottom pole to top pole
            a = _lathe(np.stack([rad * np.sin(t), rad * np.cos(t), np.sin(t), np.cos(t)], 1), nu)
        else:
            h = attr("height", 2.0) / 2.0
            if prim.IsA(UsdGeom.Cylinder):
                prof = [(0, -h, 0, -1), (rad, -h, 0, -1), (rad, -h, 1, 0), (rad, h, 1, 0), (rad, h, 0, 1),
                        (0, h, 0, 1)]
            elif prim.IsA(UsdGeom.Cone):
                s = math.hypot(2 * h, rad)                                   # side normal tilts by rad / slant
                prof = [(0, -h, 0, -1), (rad, -h, 0, -1), (rad, -h, 2 * h / s, rad / s), (0, h, 2 * h / s, rad / s)]
            else:                                                            # capsule: hemispheres on a cylinder
                t = np.linspace(-np.pi / 2, 0.0, 9)
                low = [(rad * np.cos(x), -h + rad * np.sin(x), np.cos(x), np.sin(x)) for x in t]
                prof = low + [(rad * np.cos(x), h + rad * np.sin(-x), np.cos(x), np.sin(-x)) for x in t[::-1]]
            a = _lathe(np.array(prof, float), nu)
    if axis in ("X", "Y"):          # the shapes are built along Z: a cyclic permutation turns Z into X or Y
        perm = [2, 0, 1] if axis == "X" else [1, 2, 0]
        a = MeshArrays(a.positions[:, perm], a.faces, a.normals[:, perm], a.uv)
    return a


# ---------------------------------------------------------------------------
# Stage walk: meshes, lights, camera
# ---------------------------------------------------------------------------

@dataclass
class _Part:
    arrays: MeshArrays
    spec: MaterialSpec
    name: str
    double_sided: bool = True           # UsdGeomGprim doubleSided (only matters with backface_culling)
    uv_mirrored: bool = False           # UVs wind against the normals (tracked for normal-mapped parts only)


@dataclass
class _Group:
    """Parts that share geometry and materials, with one world matrix (USD row-vector 4x4) per occurrence."""

    parts: list[_Part]
    mats: list[np.ndarray]
    mirrored: bool = False              # negative-determinant occurrences (only tracked with normal maps)


@dataclass
class _Dome:
    radiance: np.ndarray                # intensity * 2**exposure * color
    texture: str | None
    rotation: np.ndarray                # 3x3, column-vector convention (light prim's rotation)


@dataclass
class _Distant:
    direction: np.ndarray               # unit vector the light travels along
    irradiance: np.ndarray


@dataclass
class _AreaLight:
    kind: str                           # sphere | rect | disk
    world: np.ndarray                   # 4x4, column-vector convention
    size: tuple
    radiance: np.ndarray


@dataclass
class _Camera:
    path: str
    to_world: np.ndarray                # 4x4 Mitsuba sensor matrix (column-vector convention)
    fov_x: float                        # degrees
    near: float
    far: float
    lens_radius: float                  # 0: pinhole
    focus: float


def _blackbody_rgb(kelvin: float) -> np.ndarray:
    """Linear RGB tint of a black body, normalised so 6500 K is white (UsdLux colorTemperature)."""
    def raw(k: float) -> np.ndarray:
        t = np.clip(k, 1000.0, 15000.0) / 100.0
        r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
        g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
        b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
        return srgb_to_linear(np.clip([r, g, b], 0.0, 255.0) / 255.0).astype(np.float64)
    return raw(kelvin) / raw(6500.0)


def _unit3(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-12)


class _Collector:
    """Walks the composed stage and groups mesh occurrences by shared geometry and materials."""

    def __init__(self, stage: Usd.Stage, tc, report: _Report, mats: _Materials):
        self.stage, self.tc, self.report, self.mats = stage, tc, report, mats
        self.xc = UsdGeom.XformCache(tc)
        self.groups: dict[tuple[int, bool], _Group] = {}
        self._raw: dict[str, _RawMesh | None] = {}
        self._parts: dict[tuple, list[_Part]] = {}
        self._bound: dict[str, UsdShade.Material | None] = {}
        self.skipped = 0

    def run(self) -> None:
        for prim, m in self._occurrences(self.stage.GetPseudoRoot(), None):
            self._add(prim, m)

    # -- traversal -----------------------------------------------------------

    def _matrix(self, prim: Usd.Prim, frame_root: Usd.Prim | None) -> np.ndarray:
        if frame_root is None:
            return np.array(self.xc.GetLocalToWorldTransform(prim), np.float64)
        return np.array(self.xc.ComputeRelativeTransform(prim, frame_root)[0], np.float64)

    def _occurrences(self, root: Usd.Prim, frame_root: Usd.Prim | None) -> list[tuple[Usd.Prim, np.ndarray]]:
        """(mesh prim, matrix) for every visible mesh below root, PointInstancers expanded.

        Matrices are world transforms, or relative to ``frame_root`` when walking a prototype.
        """
        out: list[tuple[Usd.Prim, np.ndarray]] = []
        it = iter(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
        for prim in it:
            if not prim.IsA(UsdGeom.Imageable):
                continue
            img = UsdGeom.Imageable(prim)
            if (img.ComputeVisibility(self.tc) == UsdGeom.Tokens.invisible
                    or img.ComputePurpose() not in _RENDER_PURPOSES or prim.IsA(UsdShade.Material)):
                it.PruneChildren()
            elif prim.IsA(UsdGeom.PointInstancer):
                out += self._instancer(prim, frame_root)
                it.PruneChildren()
            elif prim.IsA(UsdGeom.Mesh) or any(prim.IsA(t) for t in _IMPLICIT):
                out.append((prim, self._matrix(prim, frame_root)))
        return out

    def _instancer(self, inst: Usd.Prim, frame_root: Usd.Prim | None) -> list[tuple[Usd.Prim, np.ndarray]]:
        pi = UsdGeom.PointInstancer(inst)
        protos = pi.GetPrototypesRel().GetTargets()
        idx = _get(pi.GetProtoIndicesAttr(), self.tc)
        if not protos or idx is None or len(idx) == 0:
            return []
        idx = np.asarray(idx, np.int64)
        xf = pi.ComputeInstanceTransformsAtTime(self.tc, self.tc, UsdGeom.PointInstancer.IncludeProtoXform,
                                                UsdGeom.PointInstancer.IgnoreMask)
        mats = np.array([np.array(x) for x in xf], np.float64).reshape(-1, 4, 4)
        mask = np.asarray(pi.ComputeMaskAtTime(self.tc), bool)
        visible = mask if len(mask) == len(idx) else np.ones(len(idx), bool)
        if len(mats) != len(idx):
            self.report.fallback(f"{inst.GetPath()}: could not compute instance transforms; skipped")
            return []
        inst_m = self._matrix(inst, frame_root)
        out: list[tuple[Usd.Prim, np.ndarray]] = []
        for pid, target in enumerate(protos):
            sel = np.nonzero((idx == pid) & visible)[0]
            proto = self.stage.GetPrimAtPath(target)
            if len(sel) == 0 or not proto:
                continue
            for prim, m_local in self._occurrences(proto, proto):
                out.extend((prim, m) for m in (m_local @ mats[sel]) @ inst_m)
        return out

    # -- one mesh ----------------------------------------------------------------

    def _bound_material(self, prim: Usd.Prim) -> UsdShade.Material | None:
        key = str(prim.GetPath())
        if key not in self._bound:
            api = UsdShade.MaterialBindingAPI(prim)
            mat = None
            for purpose in (UsdShade.Tokens.full, UsdShade.Tokens.preview):    # 'full' falls back to all-purpose
                m = api.ComputeBoundMaterial(purpose)[0]
                if m and m.GetPrim().IsValid():
                    mat = m
                    break
            self._bound[key] = mat
        return self._bound[key]

    def _parts_for(self, prim: Usd.Prim) -> list[_Part]:
        key = _proto_path(prim)
        bound = self._bound_material(prim)
        ck = (key, _proto_path(bound.GetPrim()) if bound else None)
        if ck in self._parts:
            return self._parts[ck]
        parts: list[_Part] = []
        double = bool(_get(UsdGeom.Gprim(prim).GetDoubleSidedAttr(), self.tc, False))
        if not prim.IsA(UsdGeom.Mesh):
            arrays = _implicit_arrays(prim, self.tc)
            dc = _get(UsdGeom.Gprim(prim).GetDisplayColorAttr(), self.tc)
            spec = self.mats.spec_for(bound) if bound is not None else self.mats.default(
                tuple(float(c) for c in dc[0]) if dc is not None and len(dc) else None)
            parts.append(_Part(arrays, spec, str(prim.GetPath()), double))
            self._parts[ck] = parts
            return parts
        if key not in self._raw:
            raw = _RawMesh(prim, self.tc, self.report)
            self._raw[key] = raw if raw.ok else None
        raw = self._raw[key]
        if raw is not None and len(raw.tri_face):
            name = str(prim.GetPath())
            subsets = UsdGeom.Subset.GetGeomSubsets(UsdGeom.Imageable(prim), UsdGeom.Tokens.face, "materialBind")
            assigned = np.zeros(raw.n_faces, bool)
            for sub in subsets:
                ids = np.asarray(_get(sub.GetIndicesAttr(), self.tc, []), np.int64)
                ids = ids[(ids >= 0) & (ids < raw.n_faces)]
                assigned[ids] = True
                mat = self._bound_material(sub.GetPrim())
                self._add_part(parts, raw, name + "/" + sub.GetPrim().GetName(), mat,
                               np.nonzero(np.isin(raw.tri_face, ids))[0], double)
            rest = np.nonzero(~assigned[raw.tri_face])[0]
            if len(rest):
                self._add_part(parts, raw, name, bound, None if not assigned.any() else rest, double)
        self._parts[ck] = parts
        return parts

    def _add_part(self, parts: list[_Part], raw: _RawMesh, name: str, mat: UsdShade.Material | None,
                  sel: np.ndarray | None, double_sided: bool) -> None:
        if sel is not None and len(sel) == 0:
            return
        colors = None
        if mat is not None:
            spec = self.mats.spec_for(mat)
        else:
            dc = raw.color_attr()
            if dc is None:
                spec = self.mats.default()
            elif dc.interp == "constant":
                spec = self.mats.default(tuple(float(c) for c in dc.data[0]))
            else:
                spec, colors = self.mats.default(vertex_color=True), dc
        arrays = raw.build(raw.uv_attr(spec.uv_set), colors, sel)
        pieces = _split_by_uv_handedness(arrays) if spec.normal_tex is not None else [(arrays, False)]
        parts.extend(_Part(a, spec, name, double_sided, mirrored) for a, mirrored in pieces)

    def _add(self, prim: Usd.Prim, m: np.ndarray) -> None:
        parts = self._parts_for(prim)
        if not parts:
            self.skipped += 1
            return
        mirrored = any(p.spec.normal_tex is not None for p in parts) and bool(np.linalg.det(m[:3, :3]) < 0)
        self.groups.setdefault((id(parts), mirrored), _Group(parts, [], mirrored)).mats.append(m)


def _read_lights(stage: Usd.Stage, tc, xc: UsdGeom.XformCache, report: _Report):
    domes: list[_Dome] = []
    distants: list[_Distant] = []
    areas: list[_AreaLight] = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Imageable) or not (prim.IsA(UsdLux.DomeLight) or prim.IsA(UsdLux.DistantLight)
                                                    or prim.IsA(UsdLux.SphereLight) or prim.IsA(UsdLux.RectLight)
                                                    or prim.IsA(UsdLux.DiskLight)):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility(tc) == UsdGeom.Tokens.invisible:
            continue

        def g(name: str, default):
            return _get(prim.GetAttribute(f"inputs:{name}"), tc, default)

        color = np.array(g("color", (1.0, 1.0, 1.0)), float)
        if g("enableColorTemperature", False):
            color = color * _blackbody_rgb(float(g("colorTemperature", 6500.0)))
        radiance = color * float(g("intensity", 1.0)) * 2.0 ** float(g("exposure", 0.0))
        m = np.array(xc.GetLocalToWorldTransform(prim), np.float64)
        rot = m[:3, :3] / np.maximum(np.linalg.norm(m[:3, :3], axis=1, keepdims=True), 1e-12)
        if prim.IsA(UsdLux.DomeLight):
            attr = prim.GetAttribute("inputs:texture:file")
            tex = _resolve_asset(_get(attr, tc), attr)
            if _get(attr, tc) is not None and _get(attr, tc).path and tex is None:
                report.fallback(f"{prim.GetPath()}: dome texture {_get(attr, tc).path} not found; constant colour")
            if str(g("texture:format", "automatic")) not in ("automatic", "latlong"):
                report.fallback(f"{prim.GetPath()}: texture:format {g('texture:format', '')} read as latlong")
            domes.append(_Dome(radiance, tex, rot.T))
        elif prim.IsA(UsdLux.DistantLight):
            distants.append(_Distant(_unit3(np.array([0.0, 0.0, -1.0]) @ rot), radiance))
        else:
            if g("normalize", False):
                report.fallback(f"{prim.GetPath()}: inputs:normalize is ignored; radiance = intensity * color")
            if prim.IsA(UsdLux.SphereLight):
                areas.append(_AreaLight("sphere", m.T, (float(g("radius", 0.5)),), radiance))
            elif prim.IsA(UsdLux.RectLight):
                areas.append(_AreaLight("rect", m.T, (float(g("width", 1.0)), float(g("height", 1.0))), radiance))
            else:
                areas.append(_AreaLight("disk", m.T, (float(g("radius", 0.5)),), radiance))
    return domes, distants, areas


def _read_camera(stage: Usd.Stage, prim: Usd.Prim, tc, xc: UsdGeom.XformCache, report: _Report) -> _Camera:
    cam = UsdGeom.Camera(prim)
    focal = float(_get(cam.GetFocalLengthAttr(), tc, 50.0))
    h_ap = float(_get(cam.GetHorizontalApertureAttr(), tc, 20.955))
    # USD's fallback near plane is 1 unit, which would cut off a drone half a metre away: only authored values count
    clip = _get(cam.GetClippingRangeAttr(), tc) if cam.GetClippingRangeAttr().HasAuthoredValue() else None
    clip = clip if clip is not None else Gf.Vec2f(0.01, 1.0e6)
    fstop = float(_get(cam.GetFStopAttr(), tc, 0.0))
    focus = float(_get(cam.GetFocusDistanceAttr(), tc, 0.0))
    if _get(cam.GetProjectionAttr(), tc, "perspective") != "perspective":
        report.fallback(f"{prim.GetPath()}: orthographic projection is rendered as perspective")
    m = np.array(xc.GetLocalToWorldTransform(prim), np.float64).T      # column-vector convention
    r = m[:3, :3] / np.maximum(np.linalg.norm(m[:3, :3], axis=0, keepdims=True), 1e-12)
    cam_m = np.eye(4)
    cam_m[:3, :3], cam_m[:3, 3] = r, m[:3, 3]
    mpu = UsdGeom.GetStageMetersPerUnit(stage) if stage.HasAuthoredMetadata("metersPerUnit") else 1.0
    lens = 0.0
    if fstop > 0:
        if focus > 0:
            lens = 0.5 * (focal / fstop) * 1e-3 / mpu           # focal length is in mm, the lens lives in stage units
        else:
            report.fallback(f"{prim.GetPath()}: fStop {fstop} without focusDistance; pinhole")
    return _Camera(str(prim.GetPath()), cam_m @ np.diag([-1.0, 1.0, -1.0, 1.0]),
                   math.degrees(2.0 * math.atan(h_ap / (2.0 * focal))), float(clip[0]), float(clip[1]), lens, focus)


# ---------------------------------------------------------------------------
# Mitsuba scene
# ---------------------------------------------------------------------------

_WRAP = {"repeat": "repeat", "mirror": "mirror", "clamp": "clamp", "black": "clamp", "useMetadata": "repeat"}


def _constant_opacity(spec: MaterialSpec) -> float:
    o = float(np.clip(spec.opacity, 0.0, 1.0))
    return (1.0 if o >= spec.opacity_threshold else 0.0) if spec.opacity_threshold > 0 else o


def _fit(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resample of an (H, W, C) array to another size (for AO/opacity/mask maps)."""
    if a.shape[:2] == shape:
        return a
    ys = np.minimum((np.arange(shape[0]) * a.shape[0] / shape[0]).astype(int), a.shape[0] - 1)
    xs = np.minimum((np.arange(shape[1]) * a.shape[1] / shape[1]).astype(int), a.shape[1] - 1)
    return a[ys][:, xs]


def _flip_green(encoded: np.ndarray) -> np.ndarray:
    """Normal map encoded as 0.5 n + 0.5 with its green (V) component negated."""
    out = encoded.copy()
    out[..., 1] = 1.0 - out[..., 1]
    return out


@dataclass
class _Prepared:
    """Texture arrays of one material, ready to become Mitsuba bitmaps."""

    color: np.ndarray | None = None         # (H, W, 3) base colour; all arrays are flipped vertically
    rough: np.ndarray | None = None         # (H, W, 1)
    metal: np.ndarray | None = None
    normal: np.ndarray | None = None        # (H, W, 3) encoded as 0.5 n + 0.5
    opacity: np.ndarray | None = None       # (H, W, 1) mask
    opaque: bool = True
    emission: np.ndarray | None = None      # (H, W, 3) radiance


class _Builder:
    """Turns MaterialSpecs, mesh groups, lights and cameras into Mitsuba objects (variant already set)."""

    def __init__(self, mi, images: _Images, report: _Report, backface_culling: bool):
        self.mi, self.images, self.report = mi, images, report
        self.culling = backface_culling
        probe = mi.traverse(mi.Mesh("probe", 3, 1))
        self.FloatArr, self.UIntArr = type(probe["vertex_positions"]), type(probe["faces"])
        self._bsdfs: dict[tuple[int, bool, bool], Any] = {}     # (id(spec), two-sided, green flipped)
        self._emitters: dict[int, Any] = {}

    def _variant(self, part: _Part, group: _Group) -> tuple[int, bool, bool]:
        """Key of the BSDF a part needs: its material, two-sidedness and a flip of the normal map's green channel."""
        flip = part.spec.normal_tex is not None and part.uv_mirrored != group.mirrored
        return id(part.spec), (not self.culling) or part.double_sided, flip

    # -- textures: numpy preparation (thread-safe), then Mitsuba objects ------------------------

    def _prepare(self, spec: MaterialSpec) -> _Prepared:
        """All texture arrays of one material; pure numpy, so several materials can run in threads."""
        p = _Prepared()
        img = self.images
        if not spec.vertex_color and spec.base_tex is not None:
            a = img.values(spec.base_tex)
            if a is not None:
                a = np.repeat(a, 3, axis=2) if a.shape[2] == 1 else a
                if spec.base_scale != 1.0 or spec.base_add != 0.0:
                    a = a * np.float32(spec.base_scale) + np.float32(spec.base_add)
                if spec.base_desat > 0:
                    a = a + ((a @ _LUMA)[..., None] - a) * np.float32(spec.base_desat)
                if tuple(spec.base_tint) != (1.0, 1.0, 1.0):
                    a = a * np.asarray(spec.base_tint, np.float32)
                ao = img.values(spec.ao_tex) if spec.ao_tex is not None and spec.ao_weight > 0 else None
                if ao is not None:
                    a = a * (1.0 + (_fit(ao[..., :1], a.shape[:2]) - 1.0) * np.float32(spec.ao_weight))
                p.color = np.maximum(a, 0.0, out=a) if a.flags.writeable else np.maximum(a, 0.0)
        p.rough = self._scalar(spec.roughness, spec.roughness_tex, spec.roughness_mix, _MIN_ROUGHNESS)
        p.metal = self._scalar(spec.metallic, spec.metallic_tex, spec.metallic_mix)
        p.normal = self._normal(spec)
        p.opacity, p.opaque = self._opacity(spec)
        if spec.emissive and spec.emission_tex is not None:
            a = img.values(spec.emission_tex)
            if a is not None:
                a = np.repeat(a, 3, axis=2) if a.shape[2] == 1 else a
                mask = img.values(spec.emission_mask) if spec.emission_mask is not None else None
                if mask is not None:
                    a = a * _fit(mask[..., :1], a.shape[:2])
                p.emission = a * np.float32(spec.emission_scale)
        for name in ("color", "rough", "metal", "normal", "opacity", "emission"):
            a = getattr(p, name)
            if a is not None:       # USD's v axis points up, Mitsuba's down: flip once, in the worker thread
                setattr(p, name, np.ascontiguousarray(a[::-1], np.float32))
        return p

    def _scalar(self, const: float, tex: TexSrc | None, mix: float, lo: float = 0.0) -> np.ndarray | None:
        a = self.images.values(tex) if tex is not None else None
        if a is None:
            return None
        a = (a @ _LUMA)[..., None] if a.shape[2] == 3 else a
        return np.clip(const * (1.0 - mix) + a * np.float32(mix), lo, 1.0).astype(np.float32)

    def _normal(self, spec: MaterialSpec) -> np.ndarray | None:
        src = spec.normal_tex
        if src is None:
            return None
        if src.srgb is None:
            src = dataclasses.replace(src, srgb=False)
        n = self.images.values(src)
        if n is None:
            return None
        plain = src.scale[:3] == (1.0, 1.0, 1.0) and src.bias[:3] == (0.0, 0.0, 0.0)
        if spec.normal_encoded or (plain and float(n.min()) >= 0.0):    # a plain [0, 1] map is read as 2t - 1
            n *= 2.0
            n -= 1.0
        if spec.flip_u:
            n[..., 0] *= -1.0
        if spec.flip_v:
            n[..., 1] *= -1.0
        if spec.normal_strength != 1.0:
            n[..., :2] *= np.float32(spec.normal_strength)
        np.maximum(n[..., 2], 0.05, out=n[..., 2])
        n /= np.sqrt(np.einsum("hwc,hwc->hw", n, n))[..., None]
        n *= 0.5
        n += 0.5
        return n

    def _opacity(self, spec: MaterialSpec) -> tuple[np.ndarray | None, bool]:
        """(mask texture or None, fully opaque)"""
        thr = spec.opacity_threshold
        a = self.images.values(spec.opacity_tex) if spec.opacity_tex is not None else None
        if a is None:
            return None, _constant_opacity(spec) >= 1.0
        a = (a @ _LUMA)[..., None] if a.shape[2] == 3 else a
        a = np.clip((a >= thr).astype(np.float32) if thr > 0 else a, 0.0, 1.0)
        return (None, True) if float(a.min()) >= 1.0 else (a, False)

    def _bitmap(self, arr: np.ndarray, src: TexSrc) -> dict:
        wrap = _WRAP.get(src.wrap[0], "repeat")
        if src.wrap[0] != src.wrap[1]:
            self.report.fallback(f"{os.path.basename(src.path)}: wrapS {src.wrap[0]} != wrapT {src.wrap[1]}; "
                                 "Mitsuba has one wrap mode, using wrapS")
        d = {"type": "bitmap", "bitmap": self.mi.Bitmap(arr), "raw": True, "wrap_mode": wrap,
             "filter_type": "nearest" if src.nearest else "bilinear"}
        if src.uv is not None:
            d["to_uv"] = self.mi.ScalarTransform3f(np.array(src.uv, float).reshape(3, 3))
        return d

    def _make_bsdf(self, spec: MaterialSpec, p: _Prepared, two_sided: bool, flip_green: bool):
        if spec.vertex_color:
            color: dict = {"type": "mesh_attribute", "name": "vertex_color"}
        elif p.color is not None:
            color = self._bitmap(p.color, spec.base_tex)
        else:
            tint = np.asarray(spec.base_tint, float)
            color = {"type": "rgb", "value": [float(c) for c in np.asarray(spec.base_color, float) * tint]}
        principled: dict[str, Any] = {
            "type": "principled", "base_color": color, "specular": float(np.clip(spec.specular, 0, 1)),
            "roughness": self._bitmap(p.rough, spec.roughness_tex) if p.rough is not None
            else float(np.clip(spec.roughness, _MIN_ROUGHNESS, 1.0)),
            "metallic": self._bitmap(p.metal, spec.metallic_tex) if p.metal is not None
            else float(np.clip(spec.metallic, 0.0, 1.0))}
        if spec.clearcoat > 0:
            principled["clearcoat"] = float(min(spec.clearcoat, 1.0))
            principled["clearcoat_gloss"] = float(np.clip((0.1 - spec.clearcoat_roughness ** 2) / 0.099, 0.0, 1.0))
        d = principled
        if p.normal is not None:
            normal = _flip_green(p.normal) if flip_green else p.normal
            d = {"type": "normalmap", "normalmap": self._bitmap(normal, spec.normal_tex), "bsdf": d}
        if two_sided:
            d = {"type": "twosided", "material": d}
        if not p.opaque:                    # twosided cannot wrap a mask, so the mask goes outside
            if p.opacity is not None:
                opacity: dict = self._bitmap(p.opacity, spec.opacity_tex)
            else:
                o = _constant_opacity(spec)
                opacity = {"type": "rgb", "value": [o, o, o]}
            d = {"type": "mask", "opacity": opacity, "material": d}
        return self.mi.load_dict(d)

    def _make_emitter(self, spec: MaterialSpec, p: _Prepared):
        if p.emission is not None:
            radiance: dict = self._bitmap(p.emission, spec.emission_tex)
        else:
            radiance = {"type": "rgb", "value": [float(c * spec.emission_scale) for c in spec.emission]}
        return self.mi.load_dict({"type": "area", "radiance": radiance})

    def build_materials(self, wanted: dict[int, tuple[MaterialSpec, set[tuple[bool, bool]]]]) -> None:
        """BSDFs for each wanted spec and its {(two_sided, flip_green)} variants; textures are prepared in threads."""
        todo = list(wanted.values())
        workers = max(1, min(4, os.cpu_count() or 1))
        pending: deque = deque()
        queue = iter(todo)
        with ThreadPoolExecutor(workers) as pool:
            for item in queue:
                pending.append((item, pool.submit(self._prepare, item[0])))
                if len(pending) > workers:
                    break
            while pending:
                (spec, variants), fut = pending.popleft()
                try:
                    prepared = fut.result()
                except Exception as e:  # noqa: BLE001 - one broken texture must not abort the preview
                    self.report.fallback(f"{spec.name}: textures could not be prepared ({type(e).__name__}: {e}); "
                                         "constants only")
                    prepared = _Prepared()
                nxt = next(queue, None)
                if nxt is not None:
                    pending.append((nxt, pool.submit(self._prepare, nxt[0])))
                for two_sided, flip in sorted(variants):
                    try:
                        self._bsdfs[(id(spec), two_sided, flip)] = self._make_bsdf(spec, prepared, two_sided, flip)
                    except Exception as e:  # noqa: BLE001
                        self.report.fallback(f"{spec.name}: Mitsuba rejected the material ({type(e).__name__}: "
                                             f"{e}); grey")
                        self._bsdfs[(id(spec), two_sided, flip)] = self.mi.load_dict({
                            "type": "diffuse", "reflectance": {"type": "rgb", "value": [0.5, 0.5, 0.5]}})
                if spec.emissive:
                    try:
                        self._emitters[id(spec)] = self._make_emitter(spec, prepared)
                    except Exception as e:  # noqa: BLE001
                        self.report.fallback(f"{spec.name}: Mitsuba rejected the emission ({type(e).__name__}: "
                                             f"{e}); not emissive")
                del prepared

    # -- shapes ----------------------------------------------------------------------

    def mesh(self, name: str, a: MeshArrays, bsdf, emitter=None):
        mi = self.mi
        props = mi.Properties()
        props["bsdf"] = bsdf
        if emitter is not None:
            props["emitter"] = emitter
        m = mi.Mesh(name, len(a.positions), len(a.faces), props=props, has_vertex_normals=a.normals is not None,
                    has_vertex_texcoords=a.uv is not None)
        p = mi.traverse(m)
        p["vertex_positions"] = self.FloatArr(np.ascontiguousarray(a.positions, np.float32).ravel())
        p["faces"] = self.UIntArr(np.ascontiguousarray(a.faces, np.uint32).ravel())
        if a.normals is not None:
            p["vertex_normals"] = self.FloatArr(np.ascontiguousarray(a.normals, np.float32).ravel())
        if a.uv is not None:
            p["vertex_texcoords"] = self.FloatArr(np.ascontiguousarray(a.uv, np.float32).ravel())
        p.update()
        if a.colors is not None:
            m.add_attribute("vertex_color", 3, self.FloatArr(np.ascontiguousarray(a.colors, np.float32).ravel()))
        return m

    def shapes(self, groups: list[_Group]) -> tuple[dict, dict]:
        """Scene-dict entries for all mesh groups, and counts (meshes, instances, triangles)."""
        mi = self.mi
        out: dict[str, Any] = {}
        n = {"meshes": 0, "instances": 0, "triangles": 0, "unique_triangles": 0}
        wanted: dict[int, tuple[MaterialSpec, set[tuple[bool, bool]]]] = {}
        for g in groups:
            for p in g.parts:
                wanted.setdefault(id(p.spec), (p.spec, set()))[1].add(self._variant(p, g)[1:])
        self.build_materials(wanted)
        for gi, g in enumerate(groups):
            tris = sum(len(p.arrays.faces) for p in g.parts)
            n["triangles"] += tris * len(g.mats)
            n["unique_triangles"] += tris
            baked = len(g.mats) == 1 or any(p.spec.emissive for p in g.parts)   # emitters cannot be instanced
            if baked:
                for k, m in enumerate(g.mats):
                    for pi, part in enumerate(g.parts):
                        out[f"m{gi}_{k}_{pi}"] = self.mesh(part.name, _bake(part.arrays, m),
                                                           self._bsdfs[self._variant(part, g)],
                                                           self._emitters.get(id(part.spec)))
                        n["meshes"] += 1
                continue
            out[f"g{gi}"] = {"type": "shapegroup", **{
                f"p{pi}": self.mesh(part.name, part.arrays, self._bsdfs[self._variant(part, g)])
                for pi, part in enumerate(g.parts)}}
            n["meshes"] += len(g.parts)
            for k, m in enumerate(g.mats):
                out[f"i{gi}_{k}"] = {"type": "instance", "shapegroup": {"type": "ref", "id": f"g{gi}"},
                                     "to_world": mi.ScalarTransform4f(m.T)}
                n["instances"] += 1
        return out, n

    # -- lights and cameras --------------------------------------------------------------

    def lights(self, domes: list[_Dome], distants: list[_Distant], areas: list[_AreaLight]) -> dict:
        mi = self.mi
        out: dict[str, Any] = {}
        if len(domes) > 1:
            self.report.fallback(f"{len(domes)} dome lights; only the first is rendered")
        for dome in domes[:1]:
            img = self.images.values(TexSrc(dome.texture)) if dome.texture else None
            if img is None:
                out["env"] = {"type": "constant",
                              "radiance": {"type": "rgb", "value": [float(c) for c in dome.radiance]}}
            else:
                img = np.repeat(img, 3, axis=2) if img.shape[2] == 1 else img
                data = np.ascontiguousarray(img * dome.radiance.astype(np.float32), np.float32)
                to_world = np.eye(4)
                to_world[:3, :3] = dome.rotation @ _DOME_AXES
                out["env"] = {"type": "envmap", "bitmap": mi.Bitmap(data), "to_world": mi.ScalarTransform4f(to_world)}
        for i, sun in enumerate(distants):
            out[f"sun{i}"] = {"type": "directional", "direction": [float(c) for c in sun.direction],
                              "irradiance": {"type": "rgb", "value": [float(c) for c in sun.irradiance]}}
        for i, light in enumerate(areas):
            w = np.array(light.world)
            em = {"type": "area", "radiance": {"type": "rgb", "value": [float(c) for c in light.radiance]}}
            if light.kind == "sphere":
                out[f"area{i}"] = {"type": "sphere", "radius": light.size[0],
                                   "to_world": mi.ScalarTransform4f(w), "emitter": em}
                continue
            # UsdLux rect/disk lights shine along -Z, Mitsuba's rectangle/disk along +Z: turn them over
            local = np.diag([light.size[0] / 2, light.size[-1] / 2, 1.0, 1.0]) if light.kind == "rect" \
                else np.diag([light.size[0], light.size[0], 1.0, 1.0])
            flip = np.diag([1.0, 1.0, -1.0, 1.0])
            out[f"area{i}"] = {"type": "rectangle" if light.kind == "rect" else "disk",
                               "to_world": mi.ScalarTransform4f(w @ local @ flip), "emitter": em}
        return out

    def sensor(self, cam: _Camera, width: int, height: int, spp: int, rfilter: str) -> dict:
        d: dict[str, Any] = {
            "type": "thinlens" if cam.lens_radius > 0 else "perspective", "fov": cam.fov_x, "fov_axis": "x",
            "near_clip": cam.near, "far_clip": cam.far, "to_world": self.mi.ScalarTransform4f(cam.to_world),
            "film": {"type": "hdrfilm", "width": width, "height": height, "pixel_format": "rgb",
                     "rfilter": {"type": rfilter}},
            "sampler": {"type": "independent", "sample_count": spp}}
        if cam.lens_radius > 0:
            d["aperture_radius"], d["focus_distance"] = cam.lens_radius, cam.focus
        return d


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _set_variant(want: str, report: _Report) -> str:
    import mitsuba as mi
    try:
        mi.set_variant(want)
        return want
    except Exception as e:  # noqa: BLE001 - e.g. no LLVM library installed
        if want == SCALAR:
            raise
        report.fallback(f"Mitsuba variant {want} unavailable ({e}); using {SCALAR}")
        mi.set_variant(SCALAR)
        return SCALAR


def _render_passes(mi, scene, sensor: int, spp: int, variant: str, width: int, height: int,
                   report: _Report) -> np.ndarray:
    """Average of several passes (bounded wavefront on LLVM, progress lines on both)."""
    per_pass = 16 if variant == SCALAR else max(1, min(spp, (1 << 23) // (width * height)))
    acc = np.zeros((height, width, 3), np.float64)
    done, seed = 0, 0
    while done < spp:
        n = min(per_pass, spp - done)
        img = np.array(mi.render(scene, sensor=sensor, spp=n, seed=seed), np.float64)
        acc += img[..., :3] * n
        done, seed = done + n, seed + 1
        if spp > per_pass:
            report.say(f"[mitsuba_preview]   {done}/{spp} spp")
    return (acc / spp).astype(np.float32)


def _render_jobs(stage, jobs: Sequence[tuple[str, str, str | None]], *, width: int, height: int, spp: int,
                 max_depth: int, exposure_ev: float, tonemap: str, log: Log | None, variant: str,
                 backface_culling: bool, flip_green_preview: bool | None, flip_green_mdl: bool | None,
                 max_texture_px: int, rfilter: str, time_code: float | None) -> list[dict]:
    t0 = time.time()
    report = _Report(log)
    if tonemap not in ("aces", "none"):
        raise ValueError(f"unknown tonemap {tonemap!r} (use 'aces' or 'none')")
    if variant not in ("auto", LLVM, SCALAR):
        raise ValueError(f"unknown variant {variant!r} (use 'auto', {LLVM!r} or {SCALAR!r})")
    if width < 1 or height < 1 or spp < 1:
        raise ValueError("width, height and spp must be positive")
    if isinstance(stage, (str, os.PathLike)):
        stage = Usd.Stage.Open(os.fspath(stage))
    tc = Usd.TimeCode(time_code) if time_code is not None else (
        Usd.TimeCode(stage.GetStartTimeCode()) if stage.HasAuthoredTimeCodeRange() else Usd.TimeCode.EarliestTime())
    for cam_path, _, _ in jobs:
        if not stage.GetPrimAtPath(cam_path).IsA(UsdGeom.Camera):
            raise ValueError(f"{cam_path} is not a UsdGeom.Camera prim of the stage")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        report.fallback("the stage is not Z-up; the dome light's orientation convention assumes Z up")

    mats = _Materials(tc, report, flip_green_preview, flip_green_mdl)
    col = _Collector(stage, tc, report, mats)
    col.run()
    groups = list(col.groups.values())
    xc = UsdGeom.XformCache(tc)
    domes, distants, areas = _read_lights(stage, tc, xc, report)
    cams = [_read_camera(stage, stage.GetPrimAtPath(p), tc, xc, report) for p, _, _ in jobs]
    if not domes and not distants and not areas:
        report.fallback("the stage has no lights; the image will be black")
    n_shapes = sum(len(g.parts) + (len(g.mats) if len(g.mats) > 1 else 0) for g in groups)
    n_materials = len({id(p.spec) for g in groups for p in g.parts})
    report.say(f"[mitsuba_preview] stage read in {time.time() - t0:.1f} s: {len(groups)} mesh groups, "
               f"{sum(len(g.mats) for g in groups)} occurrences, {n_materials} materials")

    variant = _set_variant(variant if variant != "auto" else (SCALAR if n_shapes > MAX_LLVM_SHAPES else LLVM), report)
    import mitsuba as mi
    if variant == LLVM and n_shapes > MAX_LLVM_SHAPES:
        report.say(f"[mitsuba_preview] {n_shapes} shapes on {LLVM}: expect about {0.04 * n_shapes:.0f} s of "
                   "tracing per pass")
    images = _Images(max_texture_px, report)
    builder = _Builder(mi, images, report, backface_culling)
    shapes, counts = builder.shapes(groups)
    scene_dict: dict[str, Any] = {"type": "scene", "integrator": {"type": "path", "max_depth": max_depth}}
    scene_dict.update(shapes)
    scene_dict.update(builder.lights(domes, distants, areas))
    for i, cam in enumerate(cams):
        scene_dict[f"sensor{i}"] = builder.sensor(cam, width, height, spp, rfilter)
    scene = mi.load_dict(scene_dict)
    build_s = time.time() - t0
    report.say(f"[mitsuba_preview] scene built in {build_s:.1f} s ({variant}, {counts['meshes']} meshes, "
               f"{counts['instances']} instances, {counts['triangles']:,} triangles)")

    results = []
    for i, (cam_path, out_png, out_exr) in enumerate(jobs):
        t1 = time.time()
        img = _render_passes(mi, scene, i, spp, variant, width, height, report)
        render_s = time.time() - t1
        if out_png:
            os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
            Image.fromarray(tonemap_image(img, exposure_ev, tonemap)).save(out_png)
        if out_exr:
            os.makedirs(os.path.dirname(os.path.abspath(out_exr)), exist_ok=True)
            mi.Bitmap(np.ascontiguousarray(img)).write(out_exr)
        report.say(f"[mitsuba_preview] {cam_path} -> {out_png} in {render_s:.1f} s")
        results.append({
            "camera": cam_path, "png": out_png, "exr": out_exr, "variant": variant, "width": width, "height": height,
            "spp": spp, "triangles": counts["triangles"], "unique_triangles": counts["unique_triangles"],
            "shapes": counts["meshes"] + counts["instances"], "meshes": counts["meshes"],
            "instances": counts["instances"], "materials": n_materials,
            "lights": len(domes) + len(distants) + len(areas),
            "build_seconds": build_s, "render_seconds": render_s, "mean_linear": float(img.mean()),
            "fallbacks": list(report.fallbacks)})
    return results


def camera_from_matrix(world: np.ndarray, *, focal_length: float = 18.147, horizontal_aperture: float = 20.955,
                       near: float = 0.01, far: float = 1.0e6, f_stop: float = 0.0, focus_distance: float = 0.0,
                       name: str = "camera") -> _Camera:
    """A camera from a USD-style world matrix (row-vector convention: rows 0-2 = camera x, y, z axes, row 3 =
    position; the camera looks down its -Z), with the same lens conventions as a UsdGeom.Camera prim."""
    m = np.asarray(world, np.float64).T
    r = m[:3, :3] / np.maximum(np.linalg.norm(m[:3, :3], axis=0, keepdims=True), 1e-12)
    cam_m = np.eye(4)
    cam_m[:3, :3], cam_m[:3, 3] = r, m[:3, 3]
    lens = 0.5 * (focal_length / f_stop) * 1e-3 if f_stop > 0 and focus_distance > 0 else 0.0
    return _Camera(name, cam_m @ np.diag([-1.0, 1.0, -1.0, 1.0]),
                   math.degrees(2.0 * math.atan(horizontal_aperture / (2.0 * focal_length))), near, far, lens,
                   focus_distance)


class Session:
    """Build a stage's Mitsuba scene once, then render any number of camera poses against it (animations).

    ``aovs=True`` also returns first-hit albedo and world shading normals (for a denoiser).  The stage's own
    cameras are ignored; pass ``camera_from_matrix(...)`` per frame.
    """

    def __init__(self, stage, *, max_depth: int = 8, variant: str = "auto", log: Log | None = None,
                 backface_culling: bool = False, flip_green_preview: bool | None = None,
                 flip_green_mdl: bool | None = None, max_texture_px: int = 2048, time_code: float | None = None,
                 aovs: bool = False):
        t0 = time.time()
        self.report = _Report(log)
        if isinstance(stage, (str, os.PathLike)):
            stage = Usd.Stage.Open(os.fspath(stage))
        tc = Usd.TimeCode(time_code) if time_code is not None else (
            Usd.TimeCode(stage.GetStartTimeCode()) if stage.HasAuthoredTimeCodeRange() else Usd.TimeCode.EarliestTime())
        mats = _Materials(tc, self.report, flip_green_preview, flip_green_mdl)
        col = _Collector(stage, tc, self.report, mats)
        col.run()
        groups = list(col.groups.values())
        domes, distants, areas = _read_lights(stage, tc, UsdGeom.XformCache(tc), self.report)
        n_shapes = sum(len(g.parts) + (len(g.mats) if len(g.mats) > 1 else 0) for g in groups)
        self.variant = _set_variant(variant if variant != "auto" else (SCALAR if n_shapes > MAX_LLVM_SHAPES else LLVM),
                                    self.report)
        import mitsuba as mi
        self.mi = mi
        self.builder = _Builder(mi, _Images(max_texture_px, self.report), self.report, backface_culling)
        shapes, self.counts = self.builder.shapes(groups)
        path = {"type": "path", "max_depth": max_depth}
        self.aovs = aovs
        integrator = {"type": "aov", "aovs": "albedo:albedo,normal:sh_normal", "inner": path} if aovs else path
        scene_dict: dict[str, Any] = {"type": "scene", "integrator": integrator}
        scene_dict.update(shapes)
        scene_dict.update(self.builder.lights(domes, distants, areas))
        self.scene = mi.load_dict(scene_dict)
        self.report.say(f"[mitsuba_preview] session scene built in {time.time() - t0:.1f} s ({self.variant}, "
                        f"{self.counts['triangles']:,} triangles)")

    def render(self, camera: _Camera, width: int, height: int, spp: int, *, seed: int = 0,
               rfilter: str = "gaussian") -> dict[str, np.ndarray]:
        """{'color': linear (H, W, 3)} plus 'albedo' and 'normal' with ``aovs``."""
        sensor = self.mi.load_dict(self.builder.sensor(camera, width, height, spp, rfilter))
        per_pass = 16 if self.variant == SCALAR else max(1, min(spp, (1 << 23) // (width * height)))
        acc, done, k = None, 0, 0
        while done < spp:
            n = min(per_pass, spp - done)
            img = np.array(self.mi.render(self.scene, sensor=sensor, spp=n, seed=seed * 1009 + k), np.float64)
            acc = img * n if acc is None else acc + img * n
            done, k = done + n, k + 1
        img = (acc / spp).astype(np.float32)
        out = {"color": img[..., :3]}
        if self.aovs:
            out["albedo"], out["normal"] = img[..., 3:6], img[..., 6:9]
        return out


def render(stage, camera: str, out_png: str, *, width: int = 960, height: int = 540, spp: int = 64,
           max_depth: int = 8, exposure_ev: float = 0.0, tonemap: str = "aces", save_exr: str | None = None,
           log: Log | None = None, variant: str = "auto", backface_culling: bool = False,
           flip_green_preview: bool | None = None, flip_green_mdl: bool | None = None,
           max_texture_px: int = 2048, rfilter: str = "gaussian", time_code: float | None = None) -> dict:
    """Path-trace ``stage`` (a ``Usd.Stage`` or a file path) through the UsdGeom.Camera at ``camera``.

    Writes ``out_png`` (exposure, tone map, sRGB, 8-bit) and optionally ``save_exr`` (linear radiance
    before exposure).  ``variant`` is 'auto' (LLVM unless the scene has more than ``MAX_LLVM_SHAPES``
    shapes), 'llvm_ad_rgb' or 'scalar_rgb'.  ``flip_green_*`` override the normal-map green convention
    per material type (None keeps each type's default).  Returns statistics: triangles (instances
    counted), shapes, materials, build/render seconds, variant and the fallbacks that were used.
    ``time_code`` picks the USD time (default: the stage's start time if it has a time range, else the earliest
    time sample).
    """
    return _render_jobs(stage, [(camera, out_png, save_exr)], width=width, height=height, spp=spp,
                        max_depth=max_depth, exposure_ev=exposure_ev, tonemap=tonemap, log=log, variant=variant,
                        backface_culling=backface_culling, flip_green_preview=flip_green_preview,
                        flip_green_mdl=flip_green_mdl, max_texture_px=max_texture_px, rfilter=rfilter,
                        time_code=time_code)[0]


def render_many(stage, cameras: dict[str, str], *, exr_dir: str | None = None, width: int = 960,
                height: int = 540, spp: int = 64, max_depth: int = 8, exposure_ev: float = 0.0,
                tonemap: str = "aces", log: Log | None = None, variant: str = "auto",
                backface_culling: bool = False, flip_green_preview: bool | None = None,
                flip_green_mdl: bool | None = None, max_texture_px: int = 2048, rfilter: str = "gaussian",
                time_code: float | None = None) -> list[dict]:
    """``render`` for several cameras ({camera path: png path}) from one scene build (reading the stage dominates)."""
    def exr(png: str) -> str | None:
        return os.path.join(exr_dir, os.path.splitext(os.path.basename(png))[0] + ".exr") if exr_dir else None

    jobs = [(cam, png, exr(png)) for cam, png in cameras.items()]
    return _render_jobs(stage, jobs, width=width, height=height, spp=spp, max_depth=max_depth,
                        exposure_ev=exposure_ev, tonemap=tonemap, log=log, variant=variant,
                        backface_culling=backface_culling, flip_green_preview=flip_green_preview,
                        flip_green_mdl=flip_green_mdl, max_texture_px=max_texture_px, rfilter=rfilter,
                        time_code=time_code)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage")
    p.add_argument("--camera", required=True, help="path of a UsdGeom.Camera prim")
    p.add_argument("--out", required=True, help="output PNG")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--spp", type=int, default=64)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--ev", type=float, default=0.0, help="exposure in stops")
    p.add_argument("--tonemap", choices=("aces", "none"), default="aces")
    p.add_argument("--exr", default=None, help="also save the linear HDR image")
    p.add_argument("--variant", choices=("auto", LLVM, SCALAR), default="auto")
    p.add_argument("--backface-culling", action="store_true", help="shade single-sided meshes one-sided")
    a = p.parse_args()
    stats = render(a.stage, a.camera, a.out, width=a.width, height=a.height, spp=a.spp, max_depth=a.max_depth,
                   exposure_ev=a.ev, tonemap=a.tonemap, save_exr=a.exr, variant=a.variant,
                   backface_culling=a.backface_culling, log=lambda m: print(m, flush=True))
    print({k: v for k, v in stats.items() if k != "fallbacks"})


if __name__ == "__main__":
    main()
