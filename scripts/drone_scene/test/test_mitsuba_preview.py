"""Offline checks of the Mitsuba preview renderer (usd-core + Mitsuba 3; no Isaac Sim).

    python scripts/drone_scene/test/test_mitsuba_preview.py

Synthetic stages are written with pxr into a temp dir and rendered at tiny sizes.  Two tests download
a few assets from NVIDIA's public asset server (about 37 MB, cached in ``$TMPDIR/mitsuba_preview_cache``)
and are skipped when the network is unreachable.  Set ``MITSUBA_PREVIEW_TEST_OUT`` to keep the PNGs.
"""

from __future__ import annotations

import atexit
import math
import os
import shutil
import sys
import tempfile
import time
import unittest
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

from drone_scene import mitsuba_preview as mp  # noqa: E402

NUCLEUS_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
CACHE = os.path.join(tempfile.gettempdir(), "mitsuba_preview_cache")
OUT = os.environ.get("MITSUBA_PREVIEW_TEST_OUT") or tempfile.mkdtemp(prefix="mitsuba_preview_test_")
os.makedirs(OUT, exist_ok=True)
if not os.environ.get("MITSUBA_PREVIEW_TEST_OUT"):      # a throw-away directory is not left behind
    atexit.register(shutil.rmtree, OUT, ignore_errors=True)
FAST = dict(width=160, height=90, spp=16, variant="scalar_rgb")


# ---------------------------------------------------------------------------
# Stage-building helpers
# ---------------------------------------------------------------------------

def new_stage() -> Usd.Stage:
    st = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    UsdGeom.Xform.Define(st, "/World")
    return st


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """USD row-vector camera matrix (looks down -Z, +Y up)."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = (target - eye) / np.linalg.norm(target - eye)
    if abs(float(f @ up)) > 0.999:                      # looking straight up or down
        up = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3], m[3, :3] = r, np.cross(r, f), -f, eye
    return m


def add_camera(st, path="/World/Cam", eye=(0, -5, 2), target=(0, 0, 0), up=(0, 0, 1), focal=24.0, aperture=36.0,
               clip=(0.01, 1000.0), **extra):
    cam = UsdGeom.Camera.Define(st, path)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(aperture)
    cam.CreateVerticalApertureAttr(aperture * 9 / 16)
    cam.CreateClippingRangeAttr(Gf.Vec2f(*clip))
    for k, v in extra.items():
        getattr(cam, f"Create{k}Attr")(v)
    cam.AddTransformOp().Set(Gf.Matrix4d(look_at(eye, target, up).tolist()))
    return path


def add_mesh(st, path, points, counts, indices, normals=None, uv=None, uv_interp="vertex", uv_indices=None,
             scheme="none", orientation=None) -> UsdGeom.Mesh:
    m = UsdGeom.Mesh.Define(st, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in points]))
    m.CreateFaceVertexCountsAttr(list(counts))
    m.CreateFaceVertexIndicesAttr(list(indices))
    m.CreateSubdivisionSchemeAttr(scheme)
    if normals is not None:
        m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, n)) for n in normals]))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if uv is not None:
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, uv_interp)
        pv.Set(Vt.Vec2fArray([Gf.Vec2f(*map(float, t)) for t in uv]))
        if uv_indices is not None:
            pv.SetIndices(Vt.IntArray(list(uv_indices)))
    if orientation:
        m.CreateOrientationAttr(orientation)
    return m


def add_quad(st, path, center=(0, 0, 0), half=(1, 1), axis="z", uv_scale=1.0, flip=False, **kw):
    """Quad facing +axis (or -axis with flip); u runs along the first in-plane axis, v along the second."""
    a, b = {"z": ((1, 0, 0), (0, 1, 0)), "x": ((0, 1, 0), (0, 0, 1)), "y": ((0, 0, 1), (1, 0, 0))}[axis]
    a, b, c = np.array(a, float), np.array(b, float), np.array(center, float)
    pts = [c - a * half[0] - b * half[1], c + a * half[0] - b * half[1], c + a * half[0] + b * half[1],
           c - a * half[0] + b * half[1]]
    n = np.cross(a, b)
    uv = [(0, 0), (uv_scale, 0), (uv_scale, uv_scale), (0, uv_scale)]
    idx = [0, 1, 2, 3]
    if flip:
        idx, n = idx[::-1], -n
    return add_mesh(st, path, pts, [4], idx, normals=kw.pop("normals", [n] * 4), uv=uv, **kw)


def uv_sphere(st, path, radius=1.0, center=(0, 0, 0), nu=32, nv=16):
    pts, counts, idx = [], [], []
    for j in range(nv + 1):
        for i in range(nu):
            a, b = 2 * math.pi * i / nu, math.pi * j / nv
            pts.append(np.array([math.sin(b) * math.cos(a), math.sin(b) * math.sin(a), math.cos(b)]))
    for j in range(nv):
        for i in range(nu):
            i1 = (i + 1) % nu
            counts.append(4)
            idx += [j * nu + i, j * nu + i1, (j + 1) * nu + i1, (j + 1) * nu + i]
    return add_mesh(st, path, [np.array(center) + radius * p for p in pts], counts, idx, normals=pts)


def preview_material(st, path, color=(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0, albedo=None, normal=None,
                     normal_scale_bias=True, opacity_tex=None, opacity_threshold=0.0, emissive=None, wrap="repeat",
                     mag_filter=None, roughness_tex=None, st_transform=None):
    mat = UsdShade.Material.Define(st, path)
    sh = UsdShade.Shader.Define(st, path + "/PBR")
    sh.CreateIdAttr("UsdPreviewSurface")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    reader = UsdShade.Shader.Define(st, path + "/stReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    st_source = reader
    if st_transform:                                    # {scale, rotation, translation} of a UsdTransform2d
        st_source = UsdShade.Shader.Define(st, path + "/stXform")
        st_source.CreateIdAttr("UsdTransform2d")
        st_source.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        for key, value in st_transform.items():
            if key == "rotation":
                st_source.CreateInput(key, Sdf.ValueTypeNames.Float).Set(float(value))
            else:
                st_source.CreateInput(key, Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*value))

    def tex(name, file, raw, out, scale=None, bias=None):
        t = UsdShade.Shader.Define(st, f"{path}/{name}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(file))
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw" if raw else "sRGB")
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(wrap)
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set(wrap)
        if mag_filter:
            t.CreateInput("magFilter", Sdf.ValueTypeNames.Token).Set(mag_filter)
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_source.ConnectableAPI(), "result")
        if scale is not None:
            t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
            t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
        return t.CreateOutput(out, Sdf.ValueTypeNames.Float3 if out == "rgb" else Sdf.ValueTypeNames.Float)

    def inp(name, typ):
        return sh.CreateInput(name, typ)

    if albedo:
        inp("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex("albedoTex", albedo, False, "rgb"))
    else:
        inp("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    if roughness_tex:
        inp("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(tex("roughnessTex", roughness_tex, True, "r"))
    else:
        inp("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    inp("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    if normal:
        sb = normal_scale_bias if isinstance(normal_scale_bias, tuple) else \
            (((2, 2, 2, 1), (-1, -1, -1, 0)) if normal_scale_bias else (None, None))
        inp("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(tex("normalTex", normal, True, "rgb", *sb))
    if opacity_tex:
        inp("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(tex("opacityTex", opacity_tex, True, "a"))
        inp("opacityThreshold", Sdf.ValueTypeNames.Float).Set(float(opacity_threshold))
    if emissive:
        inp("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    return mat


def omnipbr_material(st, path, sub="OmniPBR", asset="OmniPBR.mdl", **inputs):
    mat = UsdShade.Material.Define(st, path)
    sh = UsdShade.Shader.Define(st, path + "/Shader")
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset(Sdf.AssetPath(asset), "mdl")
    sh.SetSourceAssetSubIdentifier(sub, "mdl")
    for k, v in inputs.items():
        if isinstance(v, Sdf.AssetPath):
            sh.CreateInput(k, Sdf.ValueTypeNames.Asset).Set(v)
        elif isinstance(v, bool):
            sh.CreateInput(k, Sdf.ValueTypeNames.Bool).Set(v)
        elif isinstance(v, float):
            sh.CreateInput(k, Sdf.ValueTypeNames.Float).Set(v)
        elif len(v) == 2:
            sh.CreateInput(k, Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*v))
        else:
            sh.CreateInput(k, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*v))
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mat


def bind(prim_or_mesh, material):
    prim = prim_or_mesh.GetPrim() if hasattr(prim_or_mesh, "GetPrim") else prim_or_mesh
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def add_dome(st, path="/World/Dome", color=(1, 1, 1), intensity=1.0, texture=None, rotate_z=0.0):
    d = UsdLux.DomeLight.Define(st, path)
    d.CreateIntensityAttr(intensity)
    d.CreateColorAttr(Gf.Vec3f(*color))
    if texture:
        d.CreateTextureFileAttr(Sdf.AssetPath(texture))
        d.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    if rotate_z:
        UsdGeom.Xformable(d).AddRotateZOp().Set(float(rotate_z))
    return d


def add_distant(st, path="/World/Sun", travel=(0, 0, -1), intensity=3.0, color=(1, 1, 1)):
    """DistantLight shining along ``travel`` (its local -Z is rotated onto it)."""
    s = UsdLux.DistantLight.Define(st, path)
    s.CreateIntensityAttr(intensity)
    s.CreateColorAttr(Gf.Vec3f(*color))
    rot = Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(*map(float, travel)))
    UsdGeom.Xformable(s).AddOrientOp().Set(Gf.Quatf(rot.GetQuat()))
    return s


def write_rgbe(path, rgb, orient="-Y +X"):
    """Flat (uncompressed) Radiance .hdr from an (H, W, 3) float array shown top row first, left column first.

    ``orient`` is the file's axis order, e.g. "+Y -X": the scanlines then run bottom-up and right to left.
    """
    h, w = rgb.shape[:2]
    ry, rx = orient.split()
    if ry[0] == "+":
        rgb = rgb[::-1]
    if rx[0] == "-":
        rgb = rgb[:, ::-1]
    m, e = np.frexp(rgb.max(axis=2))
    scale = np.where(m > 0, m * 256.0 / np.maximum(rgb.max(axis=2), 1e-30), 0.0)
    pix = np.zeros((h, w, 4), np.uint8)
    pix[..., :3] = np.clip(rgb * scale[..., None], 0, 255).astype(np.uint8)
    pix[..., 3] = np.where(m > 0, e + 128, 0).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n{ry[0]}Y {h} {rx[0]}X {w}\n".encode())
        f.write(pix.tobytes())


def dome_probe(hdr, direction, rotate_z=0.0) -> float:
    """Mean linear radiance seen through a 10 degree window looking along ``direction`` from inside a dome.

    (A tiny black quad far below keeps the scene from being empty.)
    """
    st = new_stage()
    bind(add_quad(st, "/World/Far", center=(0, 0, -50), half=(1, 1)),
         preview_material(st, "/World/Looks/M", color=(0, 0, 0)))
    add_dome(st, texture=hdr, rotate_z=rotate_z)
    add_camera(st, eye=(0, 0, 0), target=direction, focal=200.0)
    return render(st, tonemap="none", width=16, height=9, spp=4, name="dome_probe")[1]["mean_linear"]


def png(path, arr):
    Image.fromarray(np.asarray(arr, np.uint8)).save(path)
    return path


def render(st, cam="/World/Cam", name="out", **kw):
    """Render with the fast defaults; returns (float image 0..1, stats)."""
    opts = {**FAST, **kw}
    out = os.path.join(OUT, name + ".png")
    stats = mp.render(st, cam, out, **opts)
    return np.asarray(Image.open(out).convert("RGB"), np.float32) / 255.0, stats


def lum(img):
    return img @ np.array([0.2126, 0.7152, 0.0722])


def fetch(rel: str, dest_root: str) -> str:
    """Download NUCLEUS_ROOT/rel into dest_root/rel (cached); skips the test when offline."""
    dest = os.path.join(dest_root, *rel.split("/"))
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        with urllib.request.urlopen(f"{NUCLEUS_ROOT}/{rel}", timeout=120) as r, open(dest + ".part", "wb") as f:
            shutil.copyfileobj(r, f)
    except OSError as e:
        raise unittest.SkipTest(f"cannot download {rel}: {e}")
    os.replace(dest + ".part", dest)
    return dest


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTonemap(unittest.TestCase):
    def test_curves(self):
        x = np.linspace(0, 8, 33, dtype=np.float32)[:, None, None] * np.ones((1, 1, 3), np.float32)
        aces = mp.tonemap_image(x, 0.0, "aces")[:, 0, 0].astype(int)
        self.assertEqual(aces[0], 0)
        self.assertTrue(np.all(np.diff(aces) >= 0))
        self.assertGreater(aces[-1], 240)
        none = mp.tonemap_image(np.array([[[0.0, 0.18, 2.0]]], np.float32), 0.0, "none")[0, 0]
        self.assertEqual(list(none), [0, 118, 255])                       # sRGB(0.18) = 0.46
        one_stop_up = mp.tonemap_image(np.full((1, 1, 3), 0.1, np.float32), 1.0, "none")[0, 0, 0]
        double = mp.tonemap_image(np.full((1, 1, 3), 0.2, np.float32), 0.0, "none")[0, 0, 0]
        self.assertAlmostEqual(int(one_stop_up), int(double), delta=1)
        with self.assertRaises(ValueError):
            mp.tonemap_image(x, 0.0, "reinhard")


class TestMdlParser(unittest.TestCase):
    TEXT = '''mdl 1.4;
import ::OmniPBR::OmniPBR;
export material Foo_Mat(*)
[[ ::anno::display_name("Omni PBR "), ::anno::key_words(string[]("omni", "PBR")) ]]
 = ::OmniPBR::OmniPBR(
    diffuse_color_constant: color(0.5f, .25, 1.f),   // a comment, with a comma
    diffuse_texture: texture_2d("./tex/a b.png" /* tag 1, version 2 */, ::tex::gamma_srgb),
    reflectionroughness_texture: texture_2d(),
    reflection_roughness_texture_influence: .411,
    enable_ORM_texture: true,
    texture_scale: float2(4.f),
    opacity_mode: base::mono_alpha,
    normalmap_texture: texture_2d("./tex/n.png", ::tex::gamma_linear));
'''

    def test_values(self):
        callee, a = mp.parse_mdl(self.TEXT, "Foo_Mat")
        self.assertEqual(callee, "::OmniPBR::OmniPBR")
        self.assertEqual(a["diffuse_color_constant"], (0.5, 0.25, 1.0))
        self.assertEqual(a["diffuse_texture"], ("tex", "./tex/a b.png", "srgb"))
        self.assertEqual(a["reflectionroughness_texture"], ("tex", None, "default"))
        self.assertEqual(a["normalmap_texture"], ("tex", "./tex/n.png", "linear"))
        self.assertAlmostEqual(a["reflection_roughness_texture_influence"], 0.411)
        self.assertIs(a["enable_ORM_texture"], True)
        self.assertEqual(a["texture_scale"], (4.0, 4.0))
        self.assertEqual(a["opacity_mode"], "mono_alpha")

    def test_other_forms(self):
        self.assertIsNone(mp.parse_mdl(self.TEXT, "Other"))
        self.assertIsNone(mp.parse_mdl("export material X(float a = 1.0) = let { } in material();", "X"))


class TestCamera(unittest.TestCase):
    def test_quad_in_front_is_centred_and_left_is_left(self):
        st = new_stage()
        red = preview_material(st, "/World/Looks/Red", color=(0.9, 0.05, 0.05), roughness=1.0)
        blue = preview_material(st, "/World/Looks/Blue", color=(0.05, 0.05, 0.9), roughness=1.0)
        q = add_quad(st, "/World/Red", center=(5, 0, 0), half=(0.5, 0.5), axis="x", flip=True)
        bind(q, red)
        q2 = add_quad(st, "/World/Blue", center=(5, 2, 0), half=(0.5, 0.5), axis="x", flip=True)
        bind(q2, blue)
        add_dome(st)
        add_camera(st, eye=(0, 0, 0), target=(1, 0, 0))              # at the origin, looking along +X
        img, _ = render(st, tonemap="none", name="camera_convention")
        h, w = img.shape[:2]
        is_red = (img[..., 0] > 0.5) & (img[..., 1] < 0.4) & (img[..., 2] < 0.4)
        is_blue = (img[..., 2] > 0.5) & (img[..., 0] < 0.4)
        self.assertGreater(is_red.sum(), 50)
        ys, xs = np.nonzero(is_red)
        self.assertAlmostEqual(xs.mean(), (w - 1) / 2, delta=1.5)
        self.assertAlmostEqual(ys.mean(), (h - 1) / 2, delta=1.5)
        # +Y is to the left of a camera that looks along +X with Z up
        bx = np.nonzero(is_blue)[1].mean()
        self.assertLess(bx, w * 0.35)

    def test_horizontal_fov_from_aperture(self):
        # focal 36, aperture 36 -> 53.13 degrees: a quad 2*tan(26.57) wide at distance 4 spans the image width
        st = new_stage()
        q = add_quad(st, "/World/Q", center=(4, 0, 0), half=(4 * 0.5, 4 * 0.5), axis="x", flip=True)
        bind(q, preview_material(st, "/World/Looks/M", color=(0.8, 0.8, 0.8), roughness=1.0))
        add_quad(st, "/World/Bar", center=(3.999, 0, 0), half=(0.2, 0.2), axis="x", flip=True)   # unlit marker
        add_dome(st)
        add_camera(st, eye=(0, 0, 0), target=(1, 0, 0), focal=36.0, aperture=36.0)
        img, _ = render(st, tonemap="none", name="fov")
        bright = lum(img) > 0.2
        cols = np.nonzero(bright.any(axis=0))[0]
        self.assertLessEqual(cols.min(), 1)
        self.assertGreaterEqual(cols.max(), img.shape[1] - 2)


class TestLights(unittest.TestCase):
    def _floor(self, travel, albedo=0.5):
        st = new_stage()
        bind(add_quad(st, "/World/Floor", half=(40, 40)),
             preview_material(st, "/World/Looks/M", color=(albedo,) * 3, roughness=1.0))
        add_distant(st, travel=travel, intensity=3.0)
        add_camera(st, eye=(0, 0, 4), target=(0, 0, 0), up=(0, 1, 0))
        return st

    def test_distant_light_direction_and_cosine(self):
        def mean(travel, name):
            return render(self._floor(travel), tonemap="none", name=name)[1]["mean_linear"]

        overhead = mean((0, 0, -1), "distant_down")
        self.assertAlmostEqual(overhead, 0.5 / math.pi * 3.0, delta=0.06)         # albedo / pi * E
        c60 = mean((math.sin(math.radians(60)), 0, -math.cos(math.radians(60))), "distant_60")
        self.assertAlmostEqual(c60 / overhead, 0.5, delta=0.08)                   # tilting reduces the power by cos
        grazing = mean((math.sin(math.radians(85)), 0, -math.cos(math.radians(85))), "distant_85")
        self.assertLess(grazing, c60 * 0.3)
        self.assertLess(mean((0, 0, 1), "distant_up"), 0.01)                      # shining away from the floor

    def test_light_colour_and_exposure(self):
        st = self._floor((0, 0, -1))
        base = render(st, tonemap="none", name="exp0")[1]["mean_linear"]
        UsdLux.DistantLight(st.GetPrimAtPath("/World/Sun")).CreateExposureAttr(1.0)
        self.assertAlmostEqual(render(st, tonemap="none", name="exp1")[1]["mean_linear"] / base, 2.0, delta=0.05)

    def test_dome_orientation_follows_usd_spec(self):
        w, h = 129, 66                                   # pixel (row 16, col 64) is exactly u = 0.5, v = 0.25
        sky = np.zeros((h, w, 3), np.float32)
        sky[15:18, 63:66] = 200.0
        hdr = os.path.join(OUT, "single_pixel.hdr")
        write_rgbe(hdr, sky)
        el = math.radians(45)

        def look(direction, rot=0.0):
            return dome_probe(hdr, direction, rot)

        # UsdLux, Z-up: u = 0.5 (longitude 0) looks along -Y; v = 0.25 is 45 degrees up
        c, s = math.cos(el), math.sin(el)
        bright = look((0, -c, s))
        self.assertGreater(bright, 10.0)
        for wrong in ((c, 0, s), (-c, 0, s), (0, c, s), (0, -c, -s), (0, -c, 0.0), (0, 0, 1)):
            self.assertLess(look(wrong), bright * 0.02, wrong)
        # the light prim's own rotation is applied on top: +90 degrees about Z turns -Y into +X
        self.assertGreater(look((c, 0, s), rot=90.0), 10.0)
        self.assertLess(look((0, -c, s), rot=90.0), bright * 0.02)

    def test_dome_azimuth_is_not_mirrored(self):
        # quarter turns (UsdLux: longitude falls as u grows): u = 0.75 is longitude -pi/2 = -X,
        # u = 0.25 is longitude +pi/2 = +X
        w, h = 129, 66
        c, s = math.cos(math.radians(45)), math.sin(math.radians(45))
        for col, toward, away in ((96, (-c, 0, s), (c, 0, s)), (32, (c, 0, s), (-c, 0, s))):
            sky = np.zeros((h, w, 3), np.float32)
            sky[15:18, col - 1:col + 2] = 200.0                 # row 16: 45 degrees up
            hdr = os.path.join(OUT, f"quarter_{col}.hdr")
            write_rgbe(hdr, sky)
            bright = dome_probe(hdr, toward)
            self.assertGreater(bright, 10.0, col)
            for wrong in (away, (0, c, s), (0, -c, s)):
                self.assertLess(dome_probe(hdr, wrong), bright * 0.02, (col, wrong))

    def test_hdr_files_of_every_orientation_are_read(self):
        sky = np.zeros((66, 129, 3), np.float32)
        sky[15:18, 95:98] = 200.0                               # -X, 45 degrees up, as seen top row first
        c, s = math.cos(math.radians(45)), math.sin(math.radians(45))
        for orient in ("-Y +X", "+Y +X", "-Y -X", "+Y -X"):     # Mitsuba's own reader takes only the first
            hdr = os.path.join(OUT, "orient.hdr")
            write_rgbe(hdr, sky, orient)
            got = mp._Images(2048, mp._Report(None)).values(mp.TexSrc(hdr))
            np.testing.assert_allclose(got, sky, atol=1e-3, err_msg=orient)
            self.assertGreater(dome_probe(hdr, (-c, 0, s)), 10.0, orient)

    def test_dome_lights_a_sphere_from_the_right_side(self):
        sky = np.zeros((66, 129, 3), np.float32)
        sky[28:31, 63:66] = 300.0                         # u = 0.5: towards -Y, 10 degrees up
        hdr = os.path.join(OUT, "blob.hdr")
        write_rgbe(hdr, sky)

        def halves(rot):
            st = new_stage()
            bind(uv_sphere(st, "/World/Ball", 1.0), preview_material(st, "/World/Looks/M", color=(0.8,) * 3,
                                                                     roughness=1.0))
            add_dome(st, texture=hdr, rotate_z=rot)
            add_camera(st, eye=(0, 0, 6), target=(0, 0, 0), up=(0, 1, 0), focal=40.0)
            img, _ = render(st, tonemap="none", name=f"dome_ball_{int(rot)}", width=64, height=36, spp=8)
            y = lum(img)
            h, w = y.shape
            return (y[:, w // 2:].sum(), y[:, :w // 2].sum()), (y[:h // 2].sum(), y[h // 2:].sum())

        # camera looking down with up = +Y: image right is +X, image top is +Y
        _, (top, bottom) = halves(0.0)                   # the blob is at -Y: image bottom
        self.assertGreater(bottom, 3 * top)
        (right, left), _ = halves(90.0)                  # +90 degrees about Z: -Y becomes +X
        self.assertGreater(right, 3 * left)
        _, (top, bottom) = halves(180.0)                 # +Y
        self.assertGreater(top, 3 * bottom)
        (right, left), _ = halves(-90.0)                 # -X
        self.assertGreater(left, 3 * right)


class TestMaterials(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="mp_tex_")
        a = np.zeros((2, 2, 3), np.uint8)
        a[0, 0], a[0, 1], a[1, 0], a[1, 1] = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)
        cls.quadrants = png(os.path.join(cls.dir, "quadrants.png"), a)
        n = np.zeros((4, 4, 3), np.uint8)
        n[...] = (191, 128, 238)                           # normal (0.5, 0, 0.866): tilted towards +U
        cls.tilt_u = png(os.path.join(cls.dir, "tilt_u.png"), n)
        n2 = np.zeros((4, 4, 3), np.uint8)
        n2[...] = (128, 191, 238)                          # tilted towards +V (green up)
        cls.tilt_v = png(os.path.join(cls.dir, "tilt_v.png"), n2)
        cls.half_alpha = os.path.join(cls.dir, "half_alpha.png")
        al = np.zeros((4, 4, 4), np.uint8)
        al[..., :3] = 255
        al[:, 2:, 3] = 255                                 # left half transparent
        Image.fromarray(al, "RGBA").save(cls.half_alpha)
        c = (np.indices((256, 256)) // 32).sum(0) % 2
        cls.checker = png(os.path.join(cls.dir, "checker.png"),
                          np.where(c[..., None] == 1, 235, 30).astype(np.uint8).repeat(3, 2))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_uv_orientation_v_up(self):
        st = new_stage()
        q = add_quad(st, "/World/Q", half=(1, 1))
        bind(q, preview_material(st, "/World/Looks/M", albedo=self.quadrants, roughness=1.0))
        add_dome(st)
        add_camera(st, eye=(0, 0, 3), target=(0, 0, 0), up=(0, 1, 0))
        img, _ = render(st, tonemap="none", name="uv_orientation")
        h, w = img.shape[:2]
        tl, tr, bl, br = img[int(h * .30), int(w * .39)], img[int(h * .30), int(w * .61)], \
            img[int(h * .70), int(w * .39)], img[int(h * .70), int(w * .61)]
        self.assertGreater(tl[0], 0.5), self.assertLess(tl[1], 0.3)      # image row 0 (top) is v = 1: red on top left
        self.assertGreater(tr[1], 0.5), self.assertLess(tr[0], 0.3)
        self.assertGreater(bl[2], 0.5), self.assertLess(bl[0], 0.3)
        self.assertGreater(br.min(), 0.5)

    def test_mag_filter_and_wrap_modes(self):
        def crop(mag, wrap="repeat", uv_scale=1.0, name="filter"):
            st = new_stage()
            bind(add_quad(st, "/World/Q", half=(1, 1), uv_scale=uv_scale),
                 preview_material(st, "/World/Looks/M", albedo=self.quadrants, roughness=1.0, mag_filter=mag, wrap=wrap))
            add_dome(st)
            add_camera(st, eye=(0, 0, 3), target=(0, 0, 0), up=(0, 1, 0))
            return render(st, tonemap="none", name=name, width=128, height=72, spp=4)[0]

        row = int(72 * 0.30)                                       # image row above the horizontal texel boundary
        lin, near = crop("linear")[row, 40:88], crop("nearest")[row, 40:88]      # columns inside the quad
        # columns where red and green are both lit: bilinear blends across the texel, nearest only at the antialiased edge
        blend = lambda img: int((np.minimum(img[:, 0], img[:, 1]) > 0.3).sum())      # noqa: E731
        self.assertGreater(blend(lin), 12)
        self.assertLessEqual(blend(near), 4)
        # clamp: with uv in [0, 2] the texture is not repeated, the last texel row/column is stretched
        clamped = crop("nearest", wrap="clamp", uv_scale=2.0, name="clamp")
        repeated = crop("nearest", wrap="repeat", uv_scale=2.0, name="repeat")
        self.assertGreater(np.abs(clamped - repeated).mean(), 0.05)

    def test_faceVarying_indexed_uv_and_subsets(self):
        st = new_stage()
        pts = [(-1, -1, 0), (0, -1, 0), (1, -1, 0), (-1, 1, 0), (0, 1, 0), (1, 1, 0)]
        m = add_mesh(st, "/World/Two", pts, [4, 4], [0, 1, 4, 3, 1, 2, 5, 4], normals=[(0, 0, 1)] * 6,
                     uv=[(0, 0), (1, 0), (1, 1), (0, 1)], uv_interp="faceVarying",
                     uv_indices=[0, 1, 2, 3, 0, 1, 2, 3])
        red = preview_material(st, "/World/Looks/R", color=(0.9, 0.05, 0.05), roughness=1.0)
        green = preview_material(st, "/World/Looks/G", color=(0.05, 0.9, 0.05), roughness=1.0)
        for name, faces, mat in (("L", [0], red), ("R", [1], green)):
            sub = UsdGeom.Subset.Define(st, f"/World/Two/{name}")
            sub.CreateElementTypeAttr("face")
            sub.CreateFamilyNameAttr("materialBind")
            sub.CreateIndicesAttr(faces)
            bind(sub, mat)
        add_dome(st)
        add_camera(st, eye=(0, 0, 3), target=(0, 0, 0), up=(0, 1, 0))
        img, stats = render(st, tonemap="none", name="subsets")
        h, w = img.shape[:2]
        left, right = img[h // 2, int(w * 0.40)], img[h // 2, int(w * 0.60)]
        self.assertGreater(left[0], 0.5), self.assertLess(left[1], 0.3)
        self.assertGreater(right[1], 0.5), self.assertLess(right[0], 0.3)
        self.assertEqual(stats["materials"], 2)
        raw = mp._RawMesh(m.GetPrim(), Usd.TimeCode.Default(), mp._Report(None))
        arrays = raw.build(raw.uv_attr("st"), None)
        self.assertEqual(arrays.faces.shape, (4, 3))
        self.assertEqual(arrays.uv.shape, (12, 2))                    # faceVarying data is unrolled per corner

    def test_normal_map_direction(self):
        def floor_mean(material_fn, travel, name, **kw):
            st = new_stage()
            q = add_quad(st, "/World/Floor", half=(40, 40))
            bind(q, material_fn(st))
            add_distant(st, travel=travel, intensity=3.0)
            add_camera(st, eye=(0, 0, 4), target=(0, 0, 0), up=(0, 1, 0))
            return render(st, tonemap="none", name=name, **kw)[1]["mean_linear"]

        c = math.sqrt(0.5)
        from_px, from_mx = (-c, 0, -c), (c, 0, -c)        # light travelling towards -X comes from +X
        from_py, from_my = (0, -c, -c), (0, c, -c)

        def ratio(material_fn, a, b, tag, **kw):
            """Floor brightness lit from direction a over lit from direction b."""
            return floor_mean(material_fn, a, f"nm_{tag}_a", **kw) / floor_mean(material_fn, b, f"nm_{tag}_b", **kw)

        def prev(tex):
            return lambda st: preview_material(st, "/World/Looks/M", color=(0.5,) * 3, roughness=1.0, normal=tex)

        def omni(tex):
            return lambda st: omnipbr_material(st, "/World/Looks/M", diffuse_color_constant=(0.5, 0.5, 0.5),
                                               reflection_roughness_constant=1.0, normalmap_texture=Sdf.AssetPath(tex))

        # UsdPreviewSurface: the map is used as is (G = +V); a map tilted to +U is brighter lit from +U, one tilted
        # to +V brighter lit from +V (n = (0.5, 0, 0.866): cos 15 / cos 75 degrees = 3.7)
        self.assertGreater(ratio(prev(self.tilt_u), from_px, from_mx, "pu"), 2.0)
        self.assertGreater(ratio(prev(self.tilt_v), from_py, from_my, "pv"), 2.0)
        flat = floor_mean(lambda st: preview_material(st, "/World/Looks/M", color=(0.5,) * 3, roughness=1.0),
                          from_px, "nm_flat")
        self.assertAlmostEqual(floor_mean(prev(self.tilt_u), from_px, "nm_tilted") / flat, 0.966 / 0.707, delta=0.1)
        # OmniPBR defaults to flip_tangent_v = true (DirectX maps): V is reversed, U is not
        self.assertGreater(ratio(omni(self.tilt_u), from_px, from_mx, "ou"), 2.0)
        self.assertGreater(ratio(omni(self.tilt_v), from_my, from_py, "ov"), 2.0)
        # ... and the per-type override flips it back
        self.assertGreater(ratio(omni(self.tilt_v), from_py, from_my, "ovf", flip_green_mdl=False), 2.0)

    def test_opacity_cutout_and_emission(self):
        st = new_stage()
        q = add_quad(st, "/World/Q", half=(1, 1))
        bind(q, preview_material(st, "/World/Looks/M", color=(0.9, 0.05, 0.05), roughness=1.0,
                                 opacity_tex=self.half_alpha, opacity_threshold=0.5))
        add_dome(st)
        add_camera(st, eye=(0, 0, 3), target=(0, 0, 0), up=(0, 1, 0))
        img, _ = render(st, tonemap="none", name="cutout")
        h, w = img.shape[:2]
        left, right = img[h // 2, int(w * 0.40)], img[h // 2, int(w * 0.60)]
        self.assertGreater(left.min(), 0.8)                           # transparent: the white sky shows through
        self.assertGreater(right[0], 0.5), self.assertLess(right[1], 0.35)     # opaque: the red quad
        st = new_stage()
        bind(add_quad(st, "/World/Q", half=(40, 40)),
             preview_material(st, "/World/Looks/M", color=(0, 0, 0), emissive=(2.0, 1.0, 0.0)))
        add_camera(st, eye=(0, 0, 3), target=(0, 0, 0), up=(0, 1, 0))
        self.assertAlmostEqual(render(st, tonemap="none", name="emissive")[1]["mean_linear"], 1.0, delta=0.05)

    def test_normal_map_follows_uv_handedness(self):
        """Mitsuba's tangent frame ignores UV mirroring; OmniPBR's tangent_v follows the UV layout (as does USD's)."""
        pts = [(-40, -40, 0), (40, -40, 0), (40, 40, 0), (-40, 40, 0)]
        uvs = {"plain": [(0, 0), (1, 0), (1, 1), (0, 1)],
               "v_mirrored": [(0, 1), (1, 1), (1, 0), (0, 0)],          # v runs along -y
               "u_mirrored": [(1, 0), (0, 0), (0, 1), (1, 1)]}          # u runs along -x
        c = math.sqrt(0.5)
        lit_from = {"+x": (-c, 0, -c), "-x": (c, 0, -c), "+y": (0, -c, -c), "-y": (0, c, -c)}

        def prev(tex):
            return lambda st: preview_material(st, "/World/Looks/M", color=(0.5,) * 3, roughness=1.0, normal=tex)

        def omni(tex):
            return lambda st: omnipbr_material(st, "/World/Looks/M", diffuse_color_constant=(0.5, 0.5, 0.5),
                                               reflection_roughness_constant=1.0, normalmap_texture=Sdf.AssetPath(tex))

        def brighter(layout, material_fn, tex_dir, a, b, scale=None):
            """Mean radiance lit from side a over lit from side b; the map tilts towards +U or +V of the texture."""
            def mean(side):
                st = new_stage()
                q = add_mesh(st, "/World/Floor", pts, [4], [0, 1, 2, 3], normals=[(0, 0, 1)] * 4, uv=uvs[layout])
                if scale:
                    UsdGeom.Xformable(q).AddScaleOp().Set(Gf.Vec3d(*scale))
                bind(q, material_fn(st))
                add_distant(st, travel=lit_from[side], intensity=3.0)
                add_camera(st, eye=(0, 0, 4), target=(0, 0, 0), up=(0, 1, 0))
                return render(st, tonemap="none", name=f"handed_{layout}_{tex_dir}_{side}")[1]["mean_linear"]
            return mean(a) / mean(b)

        # +V is +y on the plain layout, -y when v is mirrored, and +y again when only u is mirrored
        self.assertGreater(brighter("plain", prev(self.tilt_v), "v", "+y", "-y"), 2.0)
        self.assertGreater(brighter("v_mirrored", prev(self.tilt_v), "v", "-y", "+y"), 2.0)
        self.assertGreater(brighter("u_mirrored", prev(self.tilt_v), "v", "+y", "-y"), 2.0)
        self.assertGreater(brighter("u_mirrored", prev(self.tilt_u), "u", "-x", "+x"), 2.0)   # +U is -x there
        # a mirroring transform reverses the handedness as well: x -> -x turns +U into -x, +V stays +y
        mirror = (-1, 1, 1)
        self.assertGreater(brighter("plain", prev(self.tilt_v), "v", "+y", "-y", scale=mirror), 2.0)
        self.assertGreater(brighter("plain", prev(self.tilt_u), "u", "-x", "+x", scale=mirror), 2.0)
        # OmniPBR flips green (DirectX maps) on top of that
        self.assertGreater(brighter("plain", omni(self.tilt_v), "v", "-y", "+y"), 2.0)
        self.assertGreater(brighter("v_mirrored", omni(self.tilt_v), "v", "+y", "-y"), 2.0)

    def test_omnipbr_and_transform2d_uv_transforms_at_render_level(self):
        """Quadrants: red / green on top, blue / white below (v up). Probe the colour at known uv positions."""
        def classify(rgb):
            return "W" if rgb.min() > 0.5 else "RGB"[int(np.argmax(rgb))]

        def row(material_fn, v, name):
            st = new_stage()
            bind(add_quad(st, "/World/Q", half=(1, 1)), material_fn(st))
            add_dome(st)
            add_camera(st, eye=(0, 0, 3), target=(0, 0, 0), up=(0, 1, 0))      # image right is +x = +u, up is +v
            img = render(st, tonemap="none", name=name, width=160, height=90, spp=8)[0]
            f = (img.shape[1] / 2) / (3 * 0.75)                                  # pixels per metre at the quad
            y = int(round(img.shape[0] / 2 - (2 * v - 1) * f))
            return "".join(classify(img[y, int(round(img.shape[1] / 2 + (2 * u - 1) * f))])
                           for u in (0.125, 0.375, 0.625, 0.875))

        def omni(**kw):
            return lambda st: omnipbr_material(st, "/World/Looks/M", diffuse_texture=Sdf.AssetPath(self.quadrants),
                                               reflection_roughness_constant=1.0, **kw)

        def xform(**kw):
            return lambda st: preview_material(st, "/World/Looks/M", albedo=self.quadrants, roughness=1.0,
                                               st_transform=kw)

        plain = lambda st: preview_material(st, "/World/Looks/M", albedo=self.quadrants, roughness=1.0)  # noqa: E731
        self.assertEqual(row(plain, 0.75, "uvt_plain"), "RRGG")
        # uv' = uv + (0.5, 0): the texture slides left by half its width and repeats
        self.assertEqual(row(omni(texture_translate=(0.5, 0.0)), 0.75, "uvt_omni_translate"), "GGRR")
        self.assertEqual(row(xform(translation=(0.5, 0.0)), 0.75, "uvt_xf_translate"), "GGRR")
        # uv' = (2u, v): two repeats across the quad (and not 0.5 of one)
        self.assertEqual(row(omni(texture_scale=(2.0, 1.0)), 0.75, "uvt_omni_scale"), "RGRG")
        self.assertEqual(row(xform(scale=(2.0, 1.0)), 0.75, "uvt_xf_scale"), "RGRG")
        # uv' = R(90 degrees) uv = (-v, u): the top row (v = 0.75) reads u' = 0.25 and v' = u
        self.assertEqual(row(omni(texture_rotate=90.0), 0.75, "uvt_omni_rotate"), "BBRR")
        self.assertEqual(row(xform(rotation=90.0), 0.75, "uvt_xf_rotate"), "BBRR")
        # the order is translate(rotate(scale(uv))): u' = 2u + 0.5 (scaling after the offset would give 2u + 1)
        self.assertEqual(row(omni(texture_scale=(2.0, 1.0), texture_translate=(0.5, 0.0)), 0.75, "uvt_omni_both"),
                         "GRGR")
        self.assertEqual(row(xform(scale=(2.0, 1.0), translation=(0.5, 0.0)), 0.75, "uvt_xf_both"), "GRGR")

    def test_metallic_and_roughness_blends(self):
        def floor_mean(name, **material):
            st = new_stage()
            bind(add_quad(st, "/World/Floor", half=(40, 40)), preview_material(st, "/World/Looks/M", **material))
            add_distant(st, travel=(-math.sqrt(0.5), 0, -math.sqrt(0.5)), intensity=3.0)
            add_camera(st, eye=(0, 0, 4), target=(0, 0, 0), up=(0, 1, 0))      # 45 degrees off the sun's mirror image
            return render(st, tonemap="none", name=name)[1]["mean_linear"]

        rough_diffuse = floor_mean("metal0", color=(0.5,) * 3, roughness=1.0, metallic=0.0)
        smooth_metal = floor_mean("metal1", color=(0.5,) * 3, roughness=0.05, metallic=1.0)
        self.assertGreater(rough_diffuse, 0.2)
        self.assertLess(smooth_metal, rough_diffuse * 0.1)                 # a mirror shows the sun in one direction
        # the prepared roughness / metallic maps: lerp(constant, texture, mix)
        rep = mp._Report(None)
        builder = object.__new__(mp._Builder)
        builder.images = mp._Images(2048, rep)
        d = tempfile.mkdtemp(prefix="mp_blend_")
        try:
            tex = mp.TexSrc(png(os.path.join(d, "g.png"), np.full((4, 4, 3), 51, np.uint8)), srgb=False)   # 0.2
            spec = mp.MaterialSpec("m", roughness=0.8, roughness_tex=tex, roughness_mix=0.25, metallic=1.0,
                                   metallic_tex=tex, metallic_mix=0.5)
            prepared = builder._prepare(spec)
            np.testing.assert_allclose(prepared.rough, 0.8 * 0.75 + 0.2 * 0.25, atol=1e-5)
            np.testing.assert_allclose(prepared.metal, 1.0 * 0.5 + 0.2 * 0.5, atol=1e-5)
            full = builder._prepare(mp.MaterialSpec("m", roughness=0.8, roughness_tex=tex, roughness_mix=1.0))
            np.testing.assert_allclose(full.rough, 0.2, atol=1e-5)         # UsdPreviewSurface: the texture alone
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_missing_mdl_is_omnipbr_only_with_omnipbr_inputs(self):
        st = new_stage()
        rep = mp._Report(None)
        mats = mp._Materials(Usd.TimeCode.Default(), rep, None, None)
        # OmniSurface-like: its colour input is read, nothing is assumed about the rest
        surf = omnipbr_material(st, "/World/Looks/Surf", sub="OmniSurface", asset="OmniSurface.mdl",
                                base_color=(0.8, 0.1, 0.1))
        spec = mats.spec_for(UsdShade.Material(surf))
        self.assertEqual((spec.kind, spec.roughness), ("fallback", 0.6))
        np.testing.assert_allclose(spec.base_color, (0.8, 0.1, 0.1), atol=1e-6)
        self.assertTrue(any("not found locally" in f and "not treated as OmniPBR" in f for f in rep.fallbacks),
                        rep.fallbacks)
        # the same name with a texture input that is not OmniPBR's
        custom = omnipbr_material(st, "/World/Looks/Custom", sub="Custom", asset="Custom.mdl",
                                  albedo_map=Sdf.AssetPath(self.checker))
        spec = mats.spec_for(UsdShade.Material(custom))
        self.assertEqual(spec.kind, "fallback")
        self.assertEqual(spec.base_tex.path, os.path.abspath(self.checker))
        # OmniPBR inputs on a missing .mdl: OmniPBR with a texture influence of 1 (the grass clump's case)
        rep2 = mp._Report(None)
        omni = omnipbr_material(st, "/World/Looks/Grass", sub="lawn", asset="lawn.mdl",
                                diffuse_texture=Sdf.AssetPath(self.checker),
                                reflectionroughness_texture=Sdf.AssetPath(self.checker))
        spec = mp._Materials(Usd.TimeCode.Default(), rep2, None, None).spec_for(UsdShade.Material(omni))
        self.assertEqual(spec.kind, "omnipbr")
        self.assertEqual(spec.roughness_mix, 1.0)
        self.assertEqual(spec.base_tex.path, os.path.abspath(self.checker))
        self.assertTrue(any("assuming OmniPBR" in f for f in rep2.fallbacks), rep2.fallbacks)

    def test_omnipbr_inputs_and_uv_transform(self):
        st = new_stage()
        mat = omnipbr_material(st, "/World/Looks/M", diffuse_texture=Sdf.AssetPath(self.checker),
                               texture_scale=(4.0, 4.0), texture_rotate=90.0, albedo_brightness=0.5,
                               reflection_roughness_constant=0.8, metallic_constant=0.25)
        bind(add_quad(st, "/World/Q", half=(1, 1)), mat)
        rep = mp._Report(None)
        spec = mp._Materials(Usd.TimeCode.Default(), rep, None, None).spec_for(UsdShade.Material(mat))
        self.assertEqual(spec.kind, "omnipbr")
        self.assertEqual(spec.base_tex.path, os.path.abspath(self.checker))
        self.assertTrue(spec.base_tex.srgb)
        self.assertEqual(spec.base_scale, 0.5)
        np.testing.assert_allclose(np.array(spec.base_tex.uv).reshape(3, 3), mp._mat3((4, 4), 90.0), atol=1e-9)
        # uv' = R(90) * (4 uv): (1, 0) -> (0, 4)
        np.testing.assert_allclose(np.array(spec.base_tex.uv).reshape(3, 3) @ [1, 0, 1], [0, 4, 1], atol=1e-9)
        self.assertAlmostEqual(spec.roughness, 0.8, places=6)
        self.assertEqual((spec.roughness_tex, spec.metallic), (None, 0.25))
        self.assertTrue(spec.flip_v and not spec.flip_u)              # DirectX normal maps by default
        self.assertEqual(rep.fallbacks, [])

    def test_omnipbr_mdl_defaults_from_the_mdl_file(self):
        d = tempfile.mkdtemp(prefix="mp_mdl_")
        try:
            os.makedirs(os.path.join(d, "textures"))
            png(os.path.join(d, "textures", "c.png"), np.full((8, 8, 3), 128, np.uint8))
            png(os.path.join(d, "textures", "r.png"), np.full((8, 8, 3), 255, np.uint8))
            with open(os.path.join(d, "Mat.mdl"), "w") as f:
                f.write('mdl 1.4;\nusing ::OmniPBR import OmniPBR;\nexport material Mat(*)\n = OmniPBR(\n'
                        '  diffuse_texture: texture_2d("./textures/c.png", ::tex::gamma_srgb),\n'
                        '  reflection_roughness_constant: 0.9f,\n  reflection_roughness_texture_influence: 1.f,\n'
                        '  reflectionroughness_texture: texture_2d("./textures/r.png", ::tex::gamma_linear),\n'
                        '  enable_ORM_texture: false, enable_opacity: true, opacity_threshold: 0.3f);\n')
            st = new_stage()
            mat = omnipbr_material(st, "/World/Looks/M", sub="Mat", asset=os.path.join(d, "Mat.mdl"))
            spec = mp._Materials(Usd.TimeCode.Default(), mp._Report(None), None, None).spec_for(UsdShade.Material(mat))
            self.assertTrue(spec.base_tex.path.endswith("c.png"))
            self.assertTrue(spec.roughness_tex.path.endswith("r.png"))
            self.assertAlmostEqual(spec.roughness, 0.9, places=6)
            self.assertEqual(spec.roughness_mix, 1.0)
            self.assertAlmostEqual(spec.opacity_threshold, 0.3, places=6)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestGeometry(unittest.TestCase):
    def test_triangulation_orientation_and_normals(self):
        st = new_stage()
        pent = [(math.cos(2 * math.pi * i / 5), math.sin(2 * math.pi * i / 5), 0.0) for i in range(5)]
        m = add_mesh(st, "/World/P", pent, [5], [0, 1, 2, 3, 4])
        tc = Usd.TimeCode.Default()
        raw = mp._RawMesh(m.GetPrim(), tc, mp._Report(None))
        self.assertEqual(raw.tri_corner.tolist(), [[0, 1, 2], [0, 2, 3], [0, 3, 4]])      # fan
        self.assertTrue(raw.flat)                                                          # scheme none: faceted
        a = raw.build(None, None)
        np.testing.assert_allclose(a.normals, [[0, 0, 1]] * 9, atol=1e-6)
        left = add_mesh(st, "/World/L", pent, [5], [4, 3, 2, 1, 0], orientation="leftHanded")
        a = mp._RawMesh(left.GetPrim(), tc, mp._Report(None)).build(None, None)
        np.testing.assert_allclose(a.normals, [[0, 0, 1]] * 9, atol=1e-6)                  # winding flipped to face +Z
        smooth = add_mesh(st, "/World/S", pent, [5], [0, 1, 2, 3, 4], scheme="catmullClark")
        raw = mp._RawMesh(smooth.GetPrim(), tc, mp._Report(None))
        self.assertFalse(raw.flat)
        self.assertEqual(raw.build(None, None).positions.shape, (5, 3))                    # shared vertices

    def test_primvar_normals_uv_names_and_display_colour(self):
        st = new_stage()
        tc = Usd.TimeCode.Default()
        pts = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        m = add_mesh(st, "/World/M", pts, [4], [0, 1, 2, 3], scheme="catmullClark")
        pvs = UsdGeom.PrimvarsAPI(m)
        nrm = pvs.CreatePrimvar("normals", Sdf.ValueTypeNames.Normal3fArray, UsdGeom.Tokens.faceVarying)
        nrm.Set(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1), Gf.Vec3f(0, 1, 0)]))
        nrm.SetIndices(Vt.IntArray([0, 0, 1, 1]))                           # indexed faceVarying normals
        uv = pvs.CreatePrimvar("UVMap", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        uv.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
        dc = pvs.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        dc.Set(Vt.Vec3fArray([Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, 0, 1), Gf.Vec3f(1, 1, 1)]))
        raw = mp._RawMesh(m.GetPrim(), tc, mp._Report(None))
        a = raw.build(raw.uv_attr("st"), raw.color_attr())                   # 'st' is missing: UVMap is found
        # triangles (0,1,2) and (0,2,3); corner normals come from the indices [0, 0, 1, 1]
        np.testing.assert_allclose(a.normals, [[0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]],
                                   atol=1e-6)
        np.testing.assert_allclose(a.uv[:2], [[0, 0], [1, 0]])
        self.assertEqual(a.colors.shape, (6, 3))                             # unrolled: normals are faceVarying
        np.testing.assert_allclose(a.colors[:3], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # rendered: unbound meshes use displayColor, grey otherwise
        add_dome(st)
        add_camera(st, eye=(0.5, 0.5, 3), target=(0.5, 0.5, 0), up=(0, 1, 0), focal=40.0)
        img, stats = render(st, tonemap="none", name="display_color")
        self.assertEqual(stats["materials"], 1)
        h, w = img.shape[:2]
        quad = img[h // 4:3 * h // 4, w // 3:2 * w // 3].reshape(-1, 3)
        self.assertGreater(np.ptp(quad, axis=0).max(), 0.2)                  # the vertex colours, not flat grey

    def test_implicit_primitives(self):
        st = new_stage()
        tc = Usd.TimeCode.Default()
        cube = UsdGeom.Cube.Define(st, "/World/Cube")
        cube.CreateSizeAttr(2.0)
        sph = UsdGeom.Sphere.Define(st, "/World/Sphere")
        sph.CreateRadiusAttr(0.5)
        cyl = UsdGeom.Cylinder.Define(st, "/World/Cyl")
        cyl.CreateRadiusAttr(0.5)
        cyl.CreateHeightAttr(3.0)
        cyl.CreateAxisAttr("X")
        cone = UsdGeom.Cone.Define(st, "/World/Cone")
        cone.CreateRadiusAttr(1.0)
        cone.CreateHeightAttr(2.0)
        cap = UsdGeom.Capsule.Define(st, "/World/Cap")
        cap.CreateRadiusAttr(0.5)
        cap.CreateHeightAttr(2.0)
        cap.CreateAxisAttr("Y")
        plane = UsdGeom.Plane.Define(st, "/World/Plane")
        plane.CreateWidthAttr(4.0)
        plane.CreateLengthAttr(2.0)
        plane.CreateAxisAttr("Z")
        expected = {"Cube": ([-1] * 3, [1] * 3), "Sphere": ([-.5] * 3, [.5] * 3),
                    "Cyl": ([-1.5, -.5, -.5], [1.5, .5, .5]), "Cone": ([-1, -1, -1], [1, 1, 1]),
                    "Cap": ([-.5, -1.5, -.5], [.5, 1.5, .5]), "Plane": ([-2, -1, 0], [2, 1, 0])}
        for name, (lo, hi) in expected.items():
            a = mp._implicit_arrays(st.GetPrimAtPath(f"/World/{name}"), tc)
            np.testing.assert_allclose(a.positions.min(0), lo, atol=1e-5, err_msg=name)
            np.testing.assert_allclose(a.positions.max(0), hi, atol=1e-5, err_msg=name)
            np.testing.assert_allclose(np.linalg.norm(a.normals, axis=1), 1.0, atol=1e-4, err_msg=name)
            tri = a.positions[a.faces]
            geometric = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            shading = a.normals[a.faces].sum(axis=1)
            self.assertGreater((geometric * shading).sum(axis=1).min(), 0.0, name)   # winding agrees with the normals
            self.assertGreater(np.linalg.norm(geometric, axis=1).min(), 0.0, name)    # no degenerate triangles
        # sphere and cube normals point away from the centre
        a = mp._implicit_arrays(st.GetPrimAtPath("/World/Sphere"), tc)
        self.assertGreater((a.positions * a.normals).sum(axis=1).min(), 0.49)
        a = mp._implicit_arrays(st.GetPrimAtPath("/World/Cube"), tc)
        self.assertGreater((a.positions * a.normals).sum(axis=1).min(), 0.99)
        # rendered, lit from the front: the cube covers its 2 m square and is shaded by the dome and the sun
        for n in ("Sphere", "Cyl", "Cone", "Cap", "Plane"):
            st.RemovePrim(f"/World/{n}")
        UsdGeom.Xformable(cube).AddScaleOp().Set(Gf.Vec3d(1, 1, 0.5))
        bind(cube, preview_material(st, "/World/Looks/M", color=(0.8, 0.8, 0.8), roughness=1.0))
        add_dome(st, intensity=0.3)
        add_distant(st, travel=(-0.4, -0.2, -1.0))
        add_camera(st, eye=(0, 0, 8), target=(0, 0, 0), up=(0, 1, 0), focal=36.0)
        img, stats = render(st, tonemap="none", name="implicit_cube")
        self.assertEqual((stats["triangles"], stats["meshes"]), (12, 1))
        h, w = img.shape[:2]
        inside = lum(img)[h // 2, w // 2]
        corner = lum(img)[2, 2]
        self.assertAlmostEqual(corner, float(mp.linear_to_srgb(0.3)), delta=0.03)   # the white dome alone, intensity 0.3
        self.assertGreater(inside, corner + 0.1)                                   # the cube is lit by the sun as well

    def test_visibility_and_purpose(self):
        st = new_stage()
        m = preview_material(st, "/World/Looks/M", color=(0.9, 0.9, 0.9), roughness=1.0)
        for name in ("Vis", "Hidden", "Guide", "Proxy", "Render"):
            bind(add_quad(st, f"/World/{name}", half=(0.3, 0.3)), m)
        UsdGeom.Imageable(st.GetPrimAtPath("/World/Hidden")).MakeInvisible()
        UsdGeom.Imageable(st.GetPrimAtPath("/World/Guide")).CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdGeom.Imageable(st.GetPrimAtPath("/World/Proxy")).CreatePurposeAttr(UsdGeom.Tokens.proxy)
        UsdGeom.Imageable(st.GetPrimAtPath("/World/Render")).CreatePurposeAttr(UsdGeom.Tokens.render)
        add_dome(st)
        add_camera(st, eye=(0, 0, 3), target=(0, 0, 0), up=(0, 1, 0))
        _, stats = render(st, name="visibility")
        # only the default-purpose and the render-purpose quads (two triangles each) are drawn
        self.assertEqual(stats["triangles"], 2 * 2)

    def test_negative_scale_keeps_front_faces(self):
        def mean(scale):
            st = new_stage()
            q = add_quad(st, "/World/Floor", half=(40, 40))
            UsdGeom.Xformable(q).AddScaleOp().Set(Gf.Vec3d(*scale))
            bind(q, preview_material(st, "/World/Looks/M", color=(0.5,) * 3, roughness=1.0))
            add_distant(st, travel=(0, 0, -1))
            add_camera(st, eye=(0, 0, 4), target=(0, 0, 0), up=(0, 1, 0))
            return render(st, tonemap="none", name="mirrored", backface_culling=True)[1]["mean_linear"]

        self.assertAlmostEqual(mean((-1, 1, 1)), mean((1, 1, 1)), delta=0.02)    # a mirrored quad still faces up

    @staticmethod
    def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
        """(first, last) column of each run of columns that contain a True pixel."""
        cols = mask.any(axis=0)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], cols.astype(int), [0]])))
        return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2])]

    def test_point_instancer(self):
        st = new_stage()
        inst = UsdGeom.PointInstancer.Define(st, "/World/Inst")
        UsdGeom.Xformable(inst).AddTranslateOp().Set(Gf.Vec3d(10, 0, 0))
        UsdGeom.Scope.Define(st, "/World/Inst/Protos")
        proto = UsdGeom.Xform.Define(st, "/World/Inst/Protos/P")
        proto.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.5))                      # the prototype root's own transform counts
        bind(add_quad(st, "/World/Inst/Protos/P/Q", half=(1, 1)),
             preview_material(st, "/World/Looks/M", color=(0.8, 0.02, 0.02), roughness=1.0))
        inst.CreatePrototypesRel().SetTargets([proto.GetPath()])
        inst.CreateProtoIndicesAttr([0, 0, 0, 0])
        inst.CreatePositionsAttr(Vt.Vec3fArray([Gf.Vec3f(x, 0, 0) for x in (-3, 0, 3, 6)]))
        inst.CreateIdsAttr([0, 1, 2, 3])
        inst.CreateInvisibleIdsAttr([3])
        add_dome(st)
        add_camera(st, eye=(10, 0, 9), target=(10, 0, 0), up=(0, 1, 0), focal=18.0)     # 90 degrees: x in [1, 19]
        img, stats = render(st, tonemap="none", name="point_instancer")
        self.assertEqual((stats["instances"], stats["meshes"], stats["triangles"]), (3, 1, 6))
        llvm = render(st, tonemap="none", name="point_instancer_llvm", variant="llvm_ad_rgb")[1]
        self.assertEqual((llvm["variant"], llvm["instances"]), ("llvm_ad_rgb", 3))
        self.assertAlmostEqual(llvm["mean_linear"], stats["mean_linear"], delta=0.03 * stats["mean_linear"])
        red = (img[..., 0] > 0.5) & (img[..., 1] < 0.35)
        runs = self._runs(red[img.shape[0] // 2 - 4:img.shape[0] // 2 + 4])
        self.assertEqual(len(runs), 3, runs)                                  # x = 7, 10, 13; the fourth is invisible
        w = img.shape[1]
        for (a, b), x in zip(runs, (7, 10, 13)):
            self.assertAlmostEqual((a + b) / 2, (x - 1) / 18 * w, delta=3)
            self.assertAlmostEqual(b - a + 1, w / 18, delta=3)               # 1 m wide (2 m quad, scale 0.5)

    def test_native_instances_share_one_mesh(self):
        d = tempfile.mkdtemp(prefix="mp_inst_")
        try:
            sub = new_stage()
            bind(add_quad(sub, "/World/Q", half=(0.5, 0.5)),
                 preview_material(sub, "/World/Looks/M", color=(0.8, 0.02, 0.02), roughness=1.0))
            sub.SetDefaultPrim(sub.GetPrimAtPath("/World"))
            sub.GetRootLayer().Export(os.path.join(d, "unit.usda"))
            st = new_stage()
            for i, x in enumerate((-2.0, 2.0)):
                p = UsdGeom.Xform.Define(st, f"/World/I{i}")
                p.GetPrim().GetReferences().AddReference(os.path.join(d, "unit.usda"))
                p.GetPrim().SetInstanceable(True)
                p.AddTranslateOp().Set(Gf.Vec3d(x, 0, 0))
            add_dome(st)
            add_camera(st, eye=(0, 0, 6), target=(0, 0, 0), up=(0, 1, 0), focal=18.0)
            img, stats = render(st, tonemap="none", name="native_instances")
            self.assertEqual((stats["instances"], stats["meshes"], stats["materials"]), (2, 1, 1))
            runs = self._runs((img[..., 0] > 0.5) & (img[..., 1] < 0.35))
            self.assertEqual(len(runs), 2, runs)
            self.assertLess(runs[0][1], img.shape[1] / 2)
            self.assertGreater(runs[1][0], img.shape[1] / 2)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_double_sided_is_honoured_when_culling(self):
        def back_mean(double_sided, culling):
            st = new_stage()
            q = add_quad(st, "/World/Q", half=(40, 40))
            q.CreateDoubleSidedAttr(double_sided)
            bind(q, preview_material(st, "/World/Looks/M", color=(0.5,) * 3, roughness=1.0))
            add_dome(st)
            add_camera(st, eye=(0, 0, -4), target=(0, 0, 0), up=(0, 1, 0))       # looks up at the quad's back
            return render(st, tonemap="none", name="double_sided", backface_culling=culling)[1]["mean_linear"]

        lit = back_mean(True, False)
        self.assertAlmostEqual(lit, 0.5, delta=0.05)                             # albedo * white sky
        self.assertAlmostEqual(back_mean(False, False), lit, delta=0.02)         # ignored by default (RTX shades both)
        self.assertAlmostEqual(back_mean(True, True), lit, delta=0.02)           # double-sided stays lit with culling
        self.assertLess(back_mean(False, True), 0.02)                            # one-sided: the back is dark

    def test_time_sampled_only_stage_is_read_at_its_earliest_sample(self):
        st = new_stage()
        bind(add_quad(st, "/World/Floor", half=(40, 40)),
             preview_material(st, "/World/Looks/M", color=(0.5,) * 3, roughness=1.0))
        add_dome(st)
        cam = UsdGeom.Camera.Define(st, "/World/Cam")
        cam.CreateFocalLengthAttr(24.0)
        cam.CreateHorizontalApertureAttr(36.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
        op = cam.AddTranslateOp()
        op.Set(Gf.Vec3d(0, 0, 4), 5.0)                  # above the floor, looking down
        op.Set(Gf.Vec3d(0, 0, -4), 10.0)                # below it: only the sky is seen
        self.assertFalse(st.HasAuthoredTimeCodeRange())
        self.assertAlmostEqual(render(st, tonemap="none", name="samples")[1]["mean_linear"], 0.53, delta=0.04)
        self.assertAlmostEqual(render(st, tonemap="none", name="samples_t10", time_code=10.0)[1]["mean_linear"],
                               1.0, delta=0.03)
        # a PointInstancer with time-sampled positions: read at the first sample too
        inst = UsdGeom.PointInstancer.Define(st, "/World/Inst")
        UsdGeom.Scope.Define(st, "/World/Inst/Protos")
        proto = UsdGeom.Xform.Define(st, "/World/Inst/Protos/P")
        bind(add_quad(st, "/World/Inst/Protos/P/Q", center=(0, 0, 1), half=(0.4, 0.4)),
             preview_material(st, "/World/Looks/Red", color=(0.8, 0.02, 0.02), roughness=1.0))
        inst.CreatePrototypesRel().SetTargets([proto.GetPath()])
        inst.CreateProtoIndicesAttr([0, 0])
        pos = inst.CreatePositionsAttr()
        pos.Set(Vt.Vec3fArray([Gf.Vec3f(-1.2, 0, 0), Gf.Vec3f(1.2, 0, 0)]), 5.0)
        pos.Set(Vt.Vec3fArray([Gf.Vec3f(-1.2, 80, 0), Gf.Vec3f(1.2, 80, 0)]), 10.0)
        img, stats = render(st, tonemap="none", name="samples_instancer")
        red = (img[..., 0] > 0.5) & (img[..., 1] < 0.35)
        self.assertEqual(stats["instances"], 2)
        self.assertEqual(len(self._runs(red)), 2)

    def test_mirrored_uv_triangles_are_split_off(self):
        from drone_scene import scene
        pts, counts, idx, nrm, uv = scene.box(0, 2, 0, 1, 0, 1, 1.0)
        st = new_stage()
        m = add_mesh(st, "/World/Box", pts, counts.tolist(), idx.tolist(), normals=nrm, uv=uv)
        raw = mp._RawMesh(m.GetPrim(), Usd.TimeCode.Default(), mp._Report(None))
        pieces = mp._split_by_uv_handedness(raw.build(raw.uv_attr("st"), None))
        # scene.box maps the +y and -x sides with a mirrored (u, v): two of its five faces
        self.assertEqual([(len(a.faces), mirrored) for a, mirrored in pieces], [(6, False), (4, True)])
        mirrored_arrays = pieces[1][0]
        self.assertEqual((np.abs(mirrored_arrays.normals).sum(0) > 1e-6).tolist(), [True, True, False])   # sides
        self.assertTrue(np.all(mp._mirrored_triangles(mirrored_arrays)))
        self.assertFalse(np.any(mp._mirrored_triangles(pieces[0][0])))
        # a mesh without UVs or normals is left alone
        a = raw.build(None, None)
        self.assertEqual(mp._split_by_uv_handedness(a), [(a, False)])

    def test_variant_selection_and_agreement(self):
        st = new_stage()
        bind(uv_sphere(st, "/World/Ball", 1.0), preview_material(st, "/World/Looks/M", color=(0.5,) * 3, roughness=0.5))
        add_dome(st)
        add_distant(st, travel=(0, 0.3, -1))
        add_camera(st, eye=(0, -4, 1.5), target=(0, 0, 0))
        a = render(st, name="variant_scalar", spp=32)[1]
        b = render(st, name="variant_llvm", spp=32, variant="llvm_ad_rgb")[1]
        self.assertEqual((a["variant"], b["variant"]), ("scalar_rgb", "llvm_ad_rgb"))
        self.assertAlmostEqual(a["mean_linear"], b["mean_linear"], delta=0.03 * a["mean_linear"])
        old = mp.MAX_LLVM_SHAPES
        try:
            mp.MAX_LLVM_SHAPES = 0
            auto = render(st, name="variant_auto", variant="auto")[1]
        finally:
            mp.MAX_LLVM_SHAPES = old
        self.assertEqual(auto["variant"], "scalar_rgb")


class TestMoreFeatures(unittest.TestCase):
    def test_rect_light_shines_along_minus_z(self):
        def mean(rotate_x):
            st = new_stage()
            bind(add_quad(st, "/World/Floor", half=(40, 40)),
                 preview_material(st, "/World/Looks/M", color=(0.5,) * 3, roughness=1.0))
            r = UsdLux.RectLight.Define(st, "/World/Rect")
            r.CreateIntensityAttr(20.0)
            r.CreateWidthAttr(2.0)
            r.CreateHeightAttr(2.0)
            xf = UsdGeom.Xformable(r)
            xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1.5))
            xf.AddRotateXOp().Set(float(rotate_x))
            add_camera(st, eye=(0, -8, 1), target=(0, -4, 0), focal=50.0)          # the light itself stays out of view
            return render(st, tonemap="none", name=f"rect_{int(rotate_x)}")[1]["mean_linear"]

        lit, away = mean(0.0), mean(180.0)
        self.assertGreater(lit, 0.2)
        self.assertLess(away, lit * 0.05)

    def test_sphere_and_disk_lights(self):
        def mean(kind):
            st = new_stage()
            bind(add_quad(st, "/World/Floor", half=(40, 40)),
                 preview_material(st, "/World/Looks/M", color=(0.5,) * 3, roughness=1.0))
            light = {"sphere": UsdLux.SphereLight, "disk": UsdLux.DiskLight}[kind].Define(st, "/World/L")
            light.CreateIntensityAttr(30.0)
            light.CreateRadiusAttr(0.5)
            UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(0, 0, 1.5))
            add_camera(st, eye=(0, -8, 1), target=(0, -4, 0), focal=50.0)
            return render(st, tonemap="none", name=f"light_{kind}")[1]["mean_linear"]

        self.assertGreater(mean("sphere"), 0.1)
        self.assertGreater(mean("disk"), 0.05)

    def test_thin_lens_blurs_what_is_out_of_focus(self):
        def sharpness(**cam):
            st = new_stage()
            c = (np.indices((256, 256)) // 8).sum(0) % 2
            with tempfile.TemporaryDirectory() as d:
                tex = png(os.path.join(d, "c.png"), np.where(c[..., None] == 1, 235, 30).astype(np.uint8).repeat(3, 2))
                bind(add_quad(st, "/World/Wall", center=(10, 0, 0), half=(10, 10), axis="x", flip=True),
                     preview_material(st, "/World/Looks/M", albedo=tex, roughness=1.0))
                add_dome(st)
                add_camera(st, eye=(0, 0, 0), target=(1, 0, 0), focal=50.0, **cam)
                y = lum(render(st, tonemap="none", name="dof_" + ("blur" if cam else "sharp"), width=96, height=54)[0])
            return float((np.diff(y, axis=1) ** 2).mean())             # gradient energy; the total variation survives blur

        sharp = sharpness()
        blurred = sharpness(FStop=1.4, FocusDistance=1.0)            # focused 9 m in front of the wall
        in_focus = sharpness(FStop=1.4, FocusDistance=10.0)
        self.assertLess(blurred, sharp * 0.6)
        self.assertGreater(in_focus, sharp * 0.8)

    def test_transform2d_and_unknown_mdl_fallback(self):
        st = new_stage()
        mat = UsdShade.Material.Define(st, "/World/Looks/M")
        sh = UsdShade.Shader.Define(st, "/World/Looks/M/PBR")
        sh.CreateIdAttr("UsdPreviewSurface")
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        reader = UsdShade.Shader.Define(st, "/World/Looks/M/reader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("uvs")
        xf = UsdShade.Shader.Define(st, "/World/Looks/M/xf")
        xf.CreateIdAttr("UsdTransform2d")
        xf.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        xf.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(3, 3))
        xf.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.5, 0))
        with tempfile.TemporaryDirectory() as d:
            tex = png(os.path.join(d, "t.png"), np.full((4, 4, 3), 90, np.uint8))
            t = UsdShade.Shader.Define(st, "/World/Looks/M/tex")
            t.CreateIdAttr("UsdUVTexture")
            t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex))
            t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(xf.ConnectableAPI(), "result")
            t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("mirror")
            t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(t.ConnectableAPI(), "rgb")
            rep = mp._Report(None)
            spec = mp._Materials(Usd.TimeCode.Default(), rep, None, None).spec_for(mat)
            self.assertEqual(spec.uv_set, "uvs")
            np.testing.assert_allclose(np.array(spec.base_tex.uv).reshape(3, 3) @ [1, 2, 1], [3.5, 6, 1])
            self.assertEqual(spec.base_tex.wrap, ("mirror", "clamp"))
            self.assertTrue(spec.base_tex.srgb is None)                            # 'auto'
            # an MDL we do not know: its diffuse-like texture input is used, the rest is grey/0.6
            tex_in = Sdf.AssetPath(tex)
            m2 = omnipbr_material(st, "/World/Looks/Custom", sub="Weird", asset="Weird.mdl")
            sh2 = UsdShade.Shader(st.GetPrimAtPath("/World/Looks/Custom/Shader"))
            sh2.CreateInput("albedo_map", Sdf.ValueTypeNames.Asset).Set(tex_in)
            sh2.CreateInput("roughness_x", Sdf.ValueTypeNames.Float).Set(0.1)
            spec2 = mp._Materials(Usd.TimeCode.Default(), rep, None, None).spec_for(m2)
        self.assertEqual(spec2.kind, "fallback")                    # missing .mdl, no OmniPBR inputs: not OmniPBR
        self.assertEqual((spec2.base_tex.path, spec2.roughness), (os.path.abspath(tex), 0.6))
        self.assertTrue(any("not found locally" in f for f in rep.fallbacks), rep.fallbacks)
        d = tempfile.mkdtemp(prefix="mp_weird_")
        try:
            with open(os.path.join(d, "Weird.mdl"), "w") as f:
                f.write('mdl 1.6;\nexport material Weird(*)\n = ::nvidia::Weird(base_color: color(0.1, 0.4, 0.2),'
                        ' roughness: 0.2f);\n')
            m3 = omnipbr_material(st, "/World/Looks/Custom2", sub="Weird", asset=os.path.join(d, "Weird.mdl"))
            spec3 = mp._Materials(Usd.TimeCode.Default(), rep, None, None).spec_for(m3)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        self.assertEqual((spec3.kind, spec3.roughness), ("fallback", 0.6))
        self.assertEqual(spec3.base_color, (0.1, 0.4, 0.2))
        self.assertTrue(any("unsupported MDL" in f for f in rep.fallbacks), rep.fallbacks)

    def test_llvm_variant_handles_the_full_material_stack(self):
        with tempfile.TemporaryDirectory() as d:
            a = np.zeros((4, 4, 4), np.uint8)
            a[..., :3], a[..., 3] = (200, 120, 60), 255
            a[:, :2, 3] = 0
            tex = os.path.join(d, "t.png")
            Image.fromarray(a, "RGBA").save(tex)
            n = png(os.path.join(d, "n.png"), np.full((4, 4, 3), (160, 128, 230), np.uint8))
            st = new_stage()
            bind(add_quad(st, "/World/Q", half=(1, 1)),
                 preview_material(st, "/World/Looks/M", albedo=tex, normal=n, opacity_tex=tex, opacity_threshold=0.5))
            add_dome(st)
            add_distant(st, travel=(-0.3, 0.2, -1))
            add_camera(st, eye=(0, 0, 3), target=(0, 0, 0), up=(0, 1, 0))
            img, stats = render(st, name="llvm_stack", variant="llvm_ad_rgb")
        self.assertEqual(stats["variant"], "llvm_ad_rgb")
        h, w = img.shape[:2]
        self.assertGreater(lum(img)[h // 2, int(w * 0.6)], 0.15)

    def test_command_line_with_relative_assets(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            png(os.path.join(d, "green.png"), np.full((8, 8, 3), (30, 200, 30), np.uint8))
            write_rgbe(os.path.join(d, "sky.hdr"), np.full((16, 32, 3), 1.0, np.float32))
            st = new_stage()
            bind(uv_sphere(st, "/World/Ball", 1.0), preview_material(st, "/World/Looks/M", albedo="./green.png"))
            add_dome(st, texture="./sky.hdr")
            add_camera(st, eye=(0, -4, 1), target=(0, 0, 0.5))
            usd, out = os.path.join(d, "s.usda"), os.path.join(d, "o.png")
            st.GetRootLayer().Export(usd)                       # textures stay relative to the layer
            r = subprocess.run([sys.executable, mp.__file__, usd, "--camera", "/World/Cam", "--out", out,
                                "--width", "64", "--height", "36", "--spp", "8", "--variant", "scalar_rgb"],
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertNotIn("fallback", r.stdout)
            with Image.open(out) as im:
                self.assertEqual(im.size, (64, 36))
                ball = np.asarray(im.convert("RGB"), np.float32)[14:22, 28:36].reshape(-1, 3).mean(axis=0)
            self.assertGreater(ball[1], 1.5 * ball[0])           # the green texture, lit by the white sky


class TestSyntheticScene(unittest.TestCase):
    def test_checker_ground_metal_sphere_dome_sun(self):
        d = tempfile.mkdtemp(prefix="mp_synth_")
        try:
            c = (np.indices((256, 256)) // 32).sum(0) % 2
            checker = png(os.path.join(d, "checker.png"),
                          np.where(c[..., None] == 1, 235, 30).astype(np.uint8).repeat(3, 2))
            st = new_stage()
            ground = add_quad(st, "/World/Ground", half=(6, 6), uv_scale=3.0)
            bind(ground, preview_material(st, "/World/Looks/Ground", albedo=checker, roughness=0.8))
            bind(uv_sphere(st, "/World/Ball", 0.8, center=(0, 0, 0.8)),
                 preview_material(st, "/World/Looks/Metal", color=(0.9, 0.6, 0.2), roughness=0.3, metallic=1.0))
            add_dome(st, color=(0.5, 0.6, 0.8), intensity=1.0)
            add_distant(st, travel=(-0.5, 0.3, -0.8), intensity=3.0)
            add_camera(st, eye=(4, -5, 2.5), target=(0, 0, 0.6))
            t0 = time.time()
            img, stats = render(st, name="synthetic", width=160, height=90, spp=16, variant="auto")
            self.assertTrue(os.path.isfile(os.path.join(OUT, "synthetic.png")))
            self.assertEqual(img.shape, (90, 160, 3))
            y = lum(img)
            self.assertGreater(y.mean(), 0.05)
            self.assertLess(y.mean(), 0.95)
            self.assertGreater(y.std(), 0.1)                                   # not a flat image
            ground_rows = y[60:]                                               # the checker, seen in perspective
            self.assertGreater(np.percentile(ground_rows, 90) - np.percentile(ground_rows, 10), 0.25)
            edges = np.abs(np.diff(ground_rows, axis=1)).mean()
            self.assertGreater(edges, 0.01)
            self.assertEqual((stats["meshes"], stats["materials"], stats["lights"]), (2, 2, 2))
            self.assertEqual(stats["fallbacks"], [])
            self.assertGreater(stats["render_seconds"], 0)
            self.assertLess(time.time() - t0, 60)
            # saving the linear image as EXR as well
            exr = os.path.join(OUT, "synthetic.exr")
            mp.render(st, "/World/Cam", os.path.join(OUT, "synthetic_exr.png"), save_exr=exr, **FAST)
            self.assertGreater(os.path.getsize(exr), 1000)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_missing_texture_falls_back_and_reports(self):
        st = new_stage()
        bind(add_quad(st, "/World/Q", half=(2, 2)),
             preview_material(st, "/World/Looks/M", albedo="/nonexistent/none.png", roughness=1.0))
        add_dome(st)
        add_camera(st, eye=(0, 0, 3), target=(0, 0, 0), up=(0, 1, 0))
        img, stats = render(st, name="missing_texture", tonemap="none")
        self.assertTrue(any("not found" in f for f in stats["fallbacks"]), stats["fallbacks"])
        self.assertGreater(lum(img).mean(), 0.1)                           # still rendered, with the default colour

    def test_not_a_camera(self):
        st = new_stage()
        with self.assertRaises(ValueError):
            mp.render(st, "/World", os.path.join(OUT, "x.png"), **FAST)


class TestAssets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(CACHE, "nucleus")

    def test_crazyflie_at_one_to_one(self):
        base = "Isaac/Robots/Bitcraze/Crazyflie"
        usd = fetch(f"{base}/cf2x.usd", self.root)
        fetch(f"{base}/configuration/cf2x_robot_schema.usd", self.root)
        st = new_stage()
        drone = UsdGeom.Xform.Define(st, "/World/Crazyflie")
        drone.GetPrim().GetReferences().AddReference(usd)
        drone.ClearXformOpOrder()
        drone.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.5))
        bind(add_quad(st, "/World/Floor", half=(5, 5)),
             preview_material(st, "/World/Looks/Floor", color=(0.06, 0.06, 0.06), roughness=1.0))
        add_dome(st, intensity=1.0)
        add_distant(st, travel=(-0.3, 0.2, -1.0), intensity=2.0)
        add_camera(st, eye=(0.35, -0.3, 0.75), target=(0, 0, 0.5), focal=35.0, aperture=20.955)
        img, stats = render(st, name="crazyflie", width=240, height=135, spp=16, tonemap="aces", exposure_ev=-1.0)
        self.assertGreater(stats["unique_triangles"], 3500)                  # ten visual meshes, collision cube is a guide
        self.assertEqual(stats["meshes"], 11)
        self.assertGreater(lum(img).std(), 0.05)
        # the drone, not just floor and sky, covers the middle of the picture: strong local contrast there
        mid = lum(img)[45:90, 80:160]
        self.assertGreater(np.abs(np.diff(mid, axis=1)).mean(), 0.01)
        self.assertGreater(mid.max() - mid.min(), 0.2)

    def test_nvidia_normal_maps_are_directx_and_match_the_height_field(self):
        """Lit at grazing angles, Grass_Cut's normal map must shade like its own height map (DirectX green)."""
        base = "NVIDIA/Materials/Base/Natural/Grass_Cut"
        height = np.asarray(Image.open(fetch(f"{base}/Grass_Cut_H.png", self.root)), np.float64) / 255.0
        normal = fetch(f"{base}/Grass_Cut_Normal.png", self.root)
        size = height.shape[0]
        c0, c1 = int(0.40 * size), int(0.55 * size)                         # a window of the tile, one texel per pixel
        win = height[c0:c1, c0:c1]
        n = win.shape[0]
        gu, gv = np.gradient(win, axis=1), -np.gradient(win, axis=0)         # u to the right, v up
        corner_uv = [(0.40, 1 - c1 / size), (0.55, 1 - c1 / size), (0.55, 1 - c0 / size), (0.40, 1 - c0 / size)]

        def corr(material, azimuth, **kw):
            st = new_stage()
            q = add_mesh(st, "/World/Q", [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [4], [0, 1, 2, 3],
                         normals=[(0, 0, 1)] * 4, uv=corner_uv)
            bind(q, material(st))
            el, az = math.radians(35), math.radians(azimuth)
            add_distant(st, travel=(-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el)))
            add_camera(st, eye=(0, 0, 2), target=(0, 0, 0), up=(0, 1, 0), focal=36.0, aperture=36.0)
            img = render(st, tonemap="none", width=n, height=n, spp=16, max_texture_px=4096, name="nvidia_normal", **kw)[0]
            seen = mp.srgb_to_linear(img[..., 1])
            nx, ny = -8.0 * gu, -8.0 * gv                                   # shading expected from the height field
            lit = np.maximum(0, (nx * math.cos(el) * math.cos(az) + ny * math.cos(el) * math.sin(az) + math.sin(el))
                             / np.sqrt(nx ** 2 + ny ** 2 + 1))
            return float(np.corrcoef(seen.ravel(), lit.ravel())[0, 1])

        flip_g = ((2, -2, 2, 1), (-1, 1, -1, 0))                            # what drone_scene's DirectX materials author
        preview = lambda flip: lambda st: preview_material(                    # noqa: E731
            st, "/World/Looks/M", color=(0.5,) * 3, roughness=1.0, normal=normal,
            normal_scale_bias=flip_g if flip else True)
        omni = lambda st: omnipbr_material(st, "/World/Looks/M", diffuse_color_constant=(0.5,) * 3,   # noqa: E731
                                           reflection_roughness_constant=1.0, normalmap_texture=Sdf.AssetPath(normal))
        for az in (30, 120, 220):
            self.assertGreater(corr(preview(True), az), 0.4, az)             # green flipped: matches the geometry
            self.assertGreater(corr(omni, az), 0.4, az)                      # OmniPBR's flip_tangent_v does the same
        self.assertLess(corr(preview(False), 120), 0.1)                      # not flipped: wrong in V
        self.assertLess(corr(omni, 120, flip_green_mdl=False), 0.1)

    def test_grass_clump_textures_load(self):
        shrub = "NVIDIA/Assets/Vegetation/Shrub"
        usd = fetch(f"{shrub}/Grass_Short_A.usd", self.root)
        fetch(f"{shrub}/materials/lawngrass_a_mat.mdl", self.root)
        src = Usd.Stage.Open(usd)
        shader = next(p for p in src.Traverse() if p.GetTypeName() == "Shader")
        for attr in shader.GetAttributes():                                  # the textures listed in the USD
            v = attr.Get()
            if isinstance(v, Sdf.AssetPath) and v.path.endswith(".png"):
                fetch(f"{shrub}/{v.path[2:]}", self.root)
        st = new_stage()
        grass = UsdGeom.Xform.Define(st, "/World/Grass")
        grass.GetPrim().GetReferences().AddReference(usd)
        grass.ClearXformOpOrder()
        grass.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
        grass.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.01, 0.01, 0.01))   # cm asset, metre stage
        add_dome(st, color=(0.7, 0.8, 1.0))
        add_distant(st, travel=(-0.4, 0.3, -1.0), intensity=3.0)
        add_camera(st, eye=(0.0, -1.6, 0.9), target=(0, 0, 0.1), focal=24.0, aperture=20.955)
        img, stats = render(st, name="grass_clump", width=240, height=135, spp=16, tonemap="aces", max_texture_px=1024)
        self.assertEqual([f for f in stats["fallbacks"] if "texture" in f or "MDL" in f], [])
        self.assertGreater(stats["unique_triangles"], 30000)
        rgb = img.reshape(-1, 3)
        green = (rgb[:, 1] > rgb[:, 0] * 1.15) & (rgb[:, 1] > rgb[:, 2] * 1.25) & (rgb[:, 1] > 0.12)
        self.assertGreater(green.mean(), 0.02, "no green pixels: the grass texture did not load")
        specs = mp._Materials(Usd.TimeCode.Default(), mp._Report(None), None, None)
        mat = UsdShade.Material(next(p for p in Usd.PrimRange(st.GetPseudoRoot(), Usd.TraverseInstanceProxies())
                                     if p.IsA(UsdShade.Material)))
        spec = specs.spec_for(mat)
        self.assertEqual(spec.kind, "omnipbr")
        self.assertTrue(spec.base_tex.path.endswith("lawngrass_a_basecolor.png"))
        self.assertTrue(spec.roughness_tex.path.endswith("lawngrass_a_roughness.png"))
        self.assertEqual(spec.roughness_mix, 1.0)                              # from the .mdl, not the default 0
        self.assertTrue(spec.normal_tex.path.endswith("lawngrass_a_normal.png"))
        self.assertEqual(spec.normal_strength, 0.5)                            # bump_factor in the .mdl


if __name__ == "__main__":
    unittest.main(verbosity=2)
