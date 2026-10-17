"""LUCID Vision Labs machine-vision cameras for the drone: datasheet optics, calibration and body geometry.

A ``Rig`` is one camera body plus one lens.  It gives everything the simulation needs to reproduce what that
camera would record:

- the image the sensor sees: resolution, pixel pitch, focal length -> intrinsics ``K``; the lens's radial and
  tangential distortion (OpenCV "plumb_bob" ``D = [k1, k2, p1, p2, k3]``), relative illumination (vignetting),
  lateral colour and blur; global shutter (all three Sony Pregius sensors here are global shutter);
- how to render it in Isaac Sim: a pinhole render that is a little wider than the lens (``render_plan``),
  because barrel distortion pulls in more of the scene at the corners.  ``sensor_sim.py`` warps that render
  into the distorted image on the GPU, then simulates the sensor and the camera's image pipeline;
- the sensor's photometry for ``sensor_sim``: full well, read noise, ADC depth, Bayer layout;
- a camera_info dictionary (ROS ``sensor_msgs/CameraInfo`` fields) written next to every video;
- ``author_body``: the camera and lens as PBR USD geometry, so the drone visibly carries them.

Catalogue numbers (datasheets via Edmund Optics / LUCID / AutonomouStuff listings, Oct 2026):

    TRI032S-CC  Triton 3.2 MP  Sony IMX265  2048 x 1536  3.45 um  1/1.8"  global  38 fps  C-mount  29x29x45 mm  67 g
    TRI050S-CC  Triton 5.0 MP  Sony IMX264  2448 x 2048  3.45 um  2/3"    global  ~22 fps C-mount  29x29x45 mm  67 g
    PHX016S-CS  Phoenix 1.6 MP Sony IMX273  1440 x 1080  3.45 um  1/2.9"  global  77 fps S-mount  24x24x27 mm  30 g

The 3.45 um Pregius pixel: ~10.5 ke- full well and ~2.4-2.8 e- read noise (EMVA 1288 reports of IMX265
cameras give 10.9 ke- / 2.35 e-; LUCID lists 71.3 dB dynamic range, i.e. 10.5 ke- / 2.8 e-), 12-bit ADC.

The lenses are generic fixed-focal machine-vision lenses, not a specific product: focal length and f-number
are what you would buy, the distortion / vignetting / lateral-colour numbers are typical of such lenses.
Replace them with your own calibration (``Lens(k1=..., ...)``) to match a real unit.

Pure numpy (pxr only inside ``author_body``), so it imports before Isaac Sim starts.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Sensor:
    name: str
    width: int                  # active pixels
    height: int
    pixel_um: float
    full_well_e: float = 10500.0
    read_noise_e: float = 2.8
    adc_bits: int = 12
    bayer: str = "RGGB"         # colour filter layout from the top-left pixel (LUCID "BayerRG")
    qe: float = 0.64            # peak quantum efficiency (mono, ~525 nm)
    shutter: str = "global"


@dataclass(frozen=True)
class CameraBody:
    model: str
    family: str
    sensor: Sensor
    max_fps: float
    mount: str                  # "C" (17.526 mm flange) or "S" (M12 x 0.5)
    size_mm: tuple[float, float, float]     # width, height, depth (without lens and connectors)
    mass_g: float
    housing_rgb: tuple[float, float, float] = (0.045, 0.047, 0.050)   # dark anodised aluminium


@dataclass(frozen=True)
class Lens:
    name: str
    focal_mm: float
    f_number: float
    mount: str
    # OpenCV plumb_bob distortion of normalised image coordinates
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    relative_illumination: float = 0.7      # corner / centre irradiance at this f-number
    lateral_color_px: float = 0.5           # red-blue separation at the image corner, pixels
    blur_px: float = 0.5                    # Gaussian sigma of the lens + pixel blur, pixels
    image_circle_mm: float = 11.0           # largest sensor diagonal the lens covers
    length_mm: float = 30.0                 # in front of the mount
    diameter_mm: float = 30.0
    mass_g: float = 45.0


IMX265 = Sensor("Sony IMX265 (1/1.8\")", 2048, 1536, 3.45)
IMX264 = Sensor("Sony IMX264 (2/3\")", 2448, 2048, 3.45)
IMX273 = Sensor("Sony IMX273 (1/2.9\")", 1440, 1080, 3.45)

CAMERAS: dict[str, CameraBody] = {
    "TRI032S-CC": CameraBody("TRI032S-CC", "Triton 3.2 MP", IMX265, 38.0, "C", (29.0, 29.0, 45.0), 67.0),
    "TRI050S-CC": CameraBody("TRI050S-CC", "Triton 5.0 MP", IMX264, 22.0, "C", (29.0, 29.0, 45.0), 67.0),
    "PHX016S-CS": CameraBody("PHX016S-CS", "Phoenix 1.6 MP", IMX273, 77.0, "S", (24.0, 24.0, 27.35), 30.0),
}

LENSES: dict[str, Lens] = {
    # wide C-mount for 1/1.8": ~83 deg horizontal on the IMX265, about -10 % barrel at the corner
    "C4": Lens("4 mm f/2.8 C-mount", 4.0, 2.8, "C", k1=-0.10, k2=0.02, relative_illumination=0.55,
               lateral_color_px=0.8, image_circle_mm=9.0, length_mm=28.0, diameter_mm=30.0, mass_g=45.0),
    "C6": Lens("6 mm f/2.4 C-mount", 6.0, 2.4, "C", k1=-0.05, k2=0.01, relative_illumination=0.68,
               lateral_color_px=0.6, image_circle_mm=11.0, length_mm=30.0, diameter_mm=29.0, mass_g=50.0),
    "C8": Lens("8 mm f/2.0 C-mount", 8.0, 2.0, "C", k1=-0.02, relative_illumination=0.75,
               lateral_color_px=0.5, image_circle_mm=11.0, length_mm=28.0, diameter_mm=29.0, mass_g=45.0),
    # M12 board lens for the Phoenix: wide and strongly barrel-distorted, as M12 lenses are
    "S3.6": Lens("3.6 mm f/2.0 M12", 3.6, 2.0, "S", k1=-0.18, k2=0.04, relative_illumination=0.5,
                 lateral_color_px=1.2, blur_px=0.6, image_circle_mm=6.6, length_mm=16.0, diameter_mm=14.0,
                 mass_g=6.0),
}

DEFAULT_CAMERA = "TRI032S-CC"
# the widest lens in the catalogue that covers each sensor
DEFAULT_LENS = {"TRI032S-CC": "C4", "TRI050S-CC": "C6", "PHX016S-CS": "S3.6"}


# ---------------------------------------------------------------------------
# Distortion (OpenCV plumb_bob on normalised coordinates x = X/Z, y = Y/Z)
# ---------------------------------------------------------------------------

def distort(x, y, lens: Lens):
    """Undistorted normalised coordinates -> distorted (OpenCV ``projectPoints`` model)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r2 = x * x + y * y
    radial = 1.0 + lens.k1 * r2 + lens.k2 * r2 * r2 + lens.k3 * r2 * r2 * r2
    xd = x * radial + 2.0 * lens.p1 * x * y + lens.p2 * (r2 + 2.0 * x * x)
    yd = y * radial + lens.p1 * (r2 + 2.0 * y * y) + 2.0 * lens.p2 * x * y
    return xd, yd


def undistort(xd, yd, lens: Lens, iterations: int = 30):
    """Distorted normalised coordinates -> undistorted (fixed-point iteration, as ``cv2.undistortPoints``)."""
    xd, yd = np.asarray(xd, float), np.asarray(yd, float)
    x, y = xd.copy(), yd.copy()
    for _ in range(iterations):
        r2 = x * x + y * y
        radial = 1.0 + lens.k1 * r2 + lens.k2 * r2 * r2 + lens.k3 * r2 * r2 * r2
        dx = 2.0 * lens.p1 * x * y + lens.p2 * (r2 + 2.0 * x * x)
        dy = lens.p1 * (r2 + 2.0 * y * y) + 2.0 * lens.p2 * x * y
        x, y = (xd - dx) / radial, (yd - dy) / radial
    return x, y


# ---------------------------------------------------------------------------
# Camera + lens
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rig:
    body: CameraBody
    lens: Lens
    resolution_scale: float = 1.0       # < 1 renders a smaller image for quick tests (not a camera mode)

    # --- image geometry ----------------------------------------------------
    @property
    def width(self) -> int:
        return max(16, int(round(self.body.sensor.width * self.resolution_scale / 2)) * 2)

    @property
    def height(self) -> int:
        return max(16, int(round(self.body.sensor.height * self.resolution_scale / 2)) * 2)

    @property
    def pixel_mm(self) -> float:
        """Pixel pitch of the output image (the sensor's, or larger with resolution_scale < 1)."""
        return self.body.sensor.pixel_um * 1e-3 * self.body.sensor.width / self.width

    @property
    def fx(self) -> float:
        return self.lens.focal_mm / self.pixel_mm

    @property
    def cx(self) -> float:
        return (self.width - 1) / 2.0           # OpenCV convention: pixel centres at integers

    @property
    def cy(self) -> float:
        return (self.height - 1) / 2.0

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fx, self.cy], [0.0, 0.0, 1.0]])

    def D(self) -> list[float]:
        L = self.lens
        return [L.k1, L.k2, L.p1, L.p2, L.k3]

    def _edges(self, n: int = 257):
        """Distorted normalised coordinates along the image border (pixel-area edges)."""
        u = np.linspace(-0.5, self.width - 0.5, n)
        v = np.linspace(-0.5, self.height - 0.5, n)
        top = (u, np.full(n, -0.5))
        bottom = (u, np.full(n, self.height - 0.5))
        left = (np.full(n, -0.5), v)
        right = (np.full(n, self.width - 0.5), v)
        uu = np.concatenate([e[0] for e in (top, bottom, left, right)])
        vv = np.concatenate([e[1] for e in (top, bottom, left, right)])
        return (uu - self.cx) / self.fx, (vv - self.cy) / self.fx

    def field_of_view_deg(self) -> tuple[float, float, float]:
        """(horizontal, vertical, diagonal) field of view of the distorted image, degrees."""
        def ang(px, py):
            x, y = undistort(np.array([px]), np.array([py]), self.lens)
            return math.degrees(math.atan(math.hypot(x[0], y[0])))
        hx, hy = (self.width / 2) / self.fx, (self.height / 2) / self.fx
        return 2 * ang(hx, 0.0), 2 * ang(0.0, hy), 2 * ang(hx, hy)

    def render_plan(self, supersample: float = 1.0) -> dict:
        """The pinhole render that covers the whole distorted image.

        Returns width / height (pixels, even), ``fx`` of the render, and the USD camera lens:
        ``focal_length`` (mm, the real lens's) and ``horizontal_aperture`` / ``vertical_aperture`` (mm), with
        square pixels, so Isaac Lab's intrinsics (``fx = width * focal / aperture``) agree with ``fx``.
        """
        xd, yd = self._edges()
        xu, yu = undistort(xd, yd, self.lens)
        half_x = float(np.abs(xu).max()) * 1.002 + 1.0 / self.fx      # a pixel of margin for the filter
        half_y = float(np.abs(yu).max()) * 1.002 + 1.0 / self.fx
        fx_r = self.fx * supersample
        w = int(math.ceil(half_x * fx_r)) * 2
        h = int(math.ceil(half_y * fx_r)) * 2
        pixel = self.lens.focal_mm / fx_r
        return {"width": w, "height": h, "fx": fx_r, "supersample": supersample,
                "focal_length": self.lens.focal_mm, "horizontal_aperture": w * pixel, "vertical_aperture": h * pixel,
                "overscan": (w / fx_r) / (self.width / self.fx)}

    # --- focus -------------------------------------------------------------
    def hyperfocal_m(self, coc_px: float = 2.0) -> float:
        """Focus distance at which everything from half of it to infinity is sharp (circle of confusion in px)."""
        f = self.lens.focal_mm
        return (f * f / (self.lens.f_number * coc_px * self.pixel_mm) + f) * 1e-3

    # --- checks and metadata ----------------------------------------------
    def sensor_diagonal_mm(self) -> float:
        s = self.body.sensor
        return math.hypot(s.width, s.height) * s.pixel_um * 1e-3

    def warnings(self, fps: float) -> list[str]:
        """What a real camera of this rig could not do (or would show) at this frame rate."""
        out = []
        if fps > self.body.max_fps * 1.0001:
            out.append(f"{self.body.model} runs at most {self.body.max_fps:g} fps at full resolution; recording at "
                       f"{fps:g} fps is faster than the real camera can deliver")
        if self.sensor_diagonal_mm() > self.lens.image_circle_mm * 1.01:     # format sizes are nominal
            out.append(f"the {self.lens.name} lens covers a {self.lens.image_circle_mm:g} mm image circle, the "
                       f"{self.body.sensor.name} sensor is {self.sensor_diagonal_mm():.1f} mm across: a real one "
                       f"would show dark corners")
        return out

    def mass_g(self) -> float:
        return self.body.mass_g + self.lens.mass_g

    def length_mm(self) -> float:
        """Lens front to the back of the housing (connectors excluded)."""
        return self.lens.length_mm + self.body.size_mm[2]

    def camera_info(self, name: str) -> dict:
        """ROS sensor_msgs/CameraInfo fields plus the hardware description."""
        K = self.K()
        P = np.zeros((3, 4))
        P[:, :3] = K
        h, v, d = self.field_of_view_deg()
        return {
            "camera": name, "width": self.width, "height": self.height,
            "distortion_model": "plumb_bob", "D": self.D(),
            "K": K.reshape(-1).tolist(), "R": np.eye(3).reshape(-1).tolist(), "P": P.reshape(-1).tolist(),
            "fov_deg": {"horizontal": h, "vertical": v, "diagonal": d},
            "model": self.body.model, "family": self.body.family, "sensor": asdict(self.body.sensor),
            "lens": asdict(self.lens), "resolution_scale": self.resolution_scale,
            "mass_g": self.mass_g(), "manufacturer": "LUCID Vision Labs",
        }


def make_rig(camera: str = DEFAULT_CAMERA, lens: str | None = None, resolution_scale: float = 1.0) -> Rig:
    if camera not in CAMERAS:
        raise KeyError(f"unknown camera {camera!r}; choose from {sorted(CAMERAS)}")
    body = CAMERAS[camera]
    lens = lens or DEFAULT_LENS[camera]
    if lens not in LENSES:
        raise KeyError(f"unknown lens {lens!r}; choose from {sorted(LENSES)}")
    if LENSES[lens].mount != body.mount:
        raise ValueError(f"lens {lens} is {LENSES[lens].mount}-mount, {camera} is {body.mount}-mount")
    if not 0.05 <= resolution_scale <= 1.0:
        raise ValueError(f"resolution_scale must be in [0.05, 1], got {resolution_scale}")
    return Rig(body, LENSES[lens], resolution_scale)


# ---------------------------------------------------------------------------
# USD geometry of the camera on the drone
# ---------------------------------------------------------------------------

# UsdPreviewSurface constants: (base colour linear RGB, roughness, metallic, clearcoat)
BODY_MATERIALS = {
    "housing": ((0.045, 0.047, 0.050), 0.42, 0.85, 0.0),     # bead-blasted anodised aluminium
    "lens_barrel": ((0.025, 0.025, 0.026), 0.55, 0.0, 0.0),  # matte black barrel
    "focus_ring": ((0.035, 0.035, 0.036), 0.75, 0.0, 0.0),   # knurled rubbery ring
    "glass": ((0.010, 0.012, 0.016), 0.04, 0.0, 1.0),        # coated front element: dark, mirror-like
    "connector": ((0.62, 0.62, 0.63), 0.30, 1.0, 0.0),       # nickel-plated M12 / M8 connectors
    "bracket": ((0.60, 0.61, 0.62), 0.35, 1.0, 0.0),         # machined aluminium bracket
}


def author_body(stage, path: str, rig: Rig, *, looks_root: str = "/World/Looks/LucidCamera",
                bracket_mm: float = 0.0) -> str:
    """The camera + lens as USD prims under ``path`` (an Xform in the camera's own frame).

    Frame: like a USD camera, the lens looks down -Z with +Y up and the origin at the front of the lens, so
    give this Xform the same pose as the camera prim.  The housing extends behind it along +Z.  With
    ``bracket_mm`` > 0 a flat aluminium bracket of that length continues behind the housing (to the drone).
    No collision, no mass: the drone's physics is unchanged.
    """
    from pxr import Gf, UsdGeom, UsdShade

    from .scene import preview_surface

    mats = {}
    for k, (c, r, m, cc) in BODY_MATERIALS.items():
        p = f"{looks_root}/{k}"
        mats[k] = UsdShade.Material(stage.GetPrimAtPath(p)) if stage.GetPrimAtPath(p) else \
            preview_surface(stage, p, color=c, roughness=r, metallic=m, clearcoat=cc)
    mm = 1e-3
    root = UsdGeom.Xform.Define(stage, path)
    w, h, d = (v * mm for v in rig.body.size_mm)
    L, R = rig.lens.length_mm * mm, rig.lens.diameter_mm * mm / 2

    def bind(prim, key):
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mats[key])

    def cube(name, size, center, key):
        c = UsdGeom.Cube.Define(stage, f"{path}/{name}")
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*center))
        c.AddScaleOp().Set(Gf.Vec3f(*size))
        bind(c, key)
        return c

    def cylinder(name, radius, z0, z1, key, xy=(0.0, 0.0)):
        c = UsdGeom.Cylinder.Define(stage, f"{path}/{name}")
        c.CreateAxisAttr("Z")
        c.CreateRadiusAttr(radius)
        c.CreateHeightAttr(abs(z1 - z0))
        c.AddTranslateOp().Set(Gf.Vec3d(xy[0], xy[1], (z0 + z1) / 2))
        bind(c, key)
        return c

    # lens: front element, barrel, focus ring, mount ring
    cylinder("glass", R * 0.72, 0.0, 0.0008, "glass")
    cylinder("barrel_front", R, 0.0008, L * 0.30, "lens_barrel")
    cylinder("focus_ring", R * 1.02, L * 0.30, L * 0.62, "focus_ring")
    cylinder("barrel_rear", R * 0.92, L * 0.62, L, "lens_barrel")
    # housing behind the lens: the square body, a slightly smaller rear cap and the connectors
    cube("housing", (w, h, d * 0.94), (0.0, 0.0, L + d * 0.47), "housing")
    cube("rear_cap", (w * 0.92, h * 0.92, d * 0.06), (0.0, 0.0, L + d * 0.97), "housing")
    cylinder("gige_m12", 0.0060, L + d, L + d + 0.012, "connector", xy=(0.0, -h * 0.18))
    cylinder("gpio_m8", 0.0040, L + d, L + d + 0.010, "connector", xy=(w * 0.24, h * 0.22))
    if bracket_mm > 0:
        # a 3 mm plate under the housing, from its front third to bracket_mm behind it
        z0, z1 = L + d * 0.3, L + d + bracket_mm * mm
        cube("bracket", (w * 0.9, 0.003, z1 - z0), (0.0, -h / 2 - 0.0015, (z0 + z1) / 2), "bracket")
    return root.GetPath().pathString


def describe(rig: Rig) -> str:
    h, v, d = rig.field_of_view_deg()
    return (f"{rig.body.model} ({rig.body.family}, {rig.body.sensor.name}, {rig.width}x{rig.height}, "
            f"{rig.body.sensor.shutter} shutter) + {rig.lens.name}: fx {rig.fx:.1f} px, FOV {h:.1f} x {v:.1f} deg "
            f"(diag {d:.1f}), {rig.mass_g():.0f} g with lens")
