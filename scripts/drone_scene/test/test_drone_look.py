"""Offline checks of the Crazyflie 2.1 look (usd-core + numpy + Pillow; no Isaac Sim, no network, no Mitsuba).

Runs on a synthetic drone with the real prim names (boxes for meshes) and, when the mirrored asset exists
(``prepare.py --mirror``), on the real ``cf2x.usd`` too.

    python scripts/drone_scene/test/test_drone_look.py
"""

from __future__ import annotations

import atexit
import inspect
import math
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

try:
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade
    HAVE_PXR = True
except ImportError:  # pragma: no cover
    HAVE_PXR = False

from drone_scene import drone_look as DL  # noqa: E402
from drone_scene import scene  # noqa: E402

CACHE = os.environ.get("DRONE_SCENE_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "drone_scene"))
CF2X = os.path.join(CACHE, "Isaac", "Robots", "Bitcraze", "Crazyflie", "cf2x.usd")
ROOT = "/World/Crazyflie"
BODY = f"{ROOT}/body/body_visual"
TEX_CACHE = tempfile.mkdtemp()                         # textures are generated once for the whole module
atexit.register(shutil.rmtree, TEX_CACHE, ignore_errors=True)


def box_mesh(stage, path: str, lo, hi) -> None:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    scene._mesh(stage, path, scene.box(x0, x1, y0, y1, z0, z1, 1.0))


def fake_drone(stage) -> None:
    """The real prim names and rough sizes; every mesh is a box (quads), with ``st`` that is not used."""
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, ROOT)
    UsdGeom.Xform.Define(stage, f"{ROOT}/body")
    UsdGeom.Xform.Define(stage, BODY)
    mm = 1e-3
    box_mesh(stage, f"{BODY}/cf_body_001", (-29 * mm, -29 * mm, 13.75 * mm), (29 * mm, 29 * mm, 16.25 * mm))
    box_mesh(stage, f"{BODY}/battery_holder", (-10 * mm, -14 * mm, 24.5 * mm), (10 * mm, 14 * mm, 27.5 * mm))
    box_mesh(stage, f"{BODY}/battery", (-14 * mm, -9 * mm, 17.5 * mm), (14 * mm, 9 * mm, 22.5 * mm))
    box_mesh(stage, f"{BODY}/motor_mount", (-38 * mm, -3 * mm, 1.3 * mm), (38 * mm, 3 * mm, 18.3 * mm))
    box_mesh(stage, f"{BODY}/Cylinder", (-33 * mm, -3 * mm, 8.7 * mm), (33 * mm, 3 * mm, 19.3 * mm))
    pins = [scene.box(-9.3 * mm + 2 * mm * i, -8.8 * mm + 2 * mm * i, 10.8 * mm, 11.3 * mm, 10.5 * mm, 28.5 * mm, 1.0)
            for i in range(10)]
    pins.append(scene.box(-10 * mm, 10 * mm, 10 * mm, 12 * mm, 14.5 * mm, 15.5 * mm, 1.0))
    scene._mesh(stage, f"{BODY}/pin_headers", scene.merge(*pins))
    for i, (x, y, blade) in enumerate(((31, -31, "ccw_prop"), (-31, -31, "cw_prop"), (-31, 31, "ccw_prop"),
                                       (31, 31, "cw_prop")), start=1):
        prop = UsdGeom.Xform.Define(stage, f"{ROOT}/m{i}_prop")
        prop.AddTranslateOp().Set(Gf.Vec3d(x * mm, y * mm, 21 * mm))
        box_mesh(stage, f"{ROOT}/m{i}_prop/{blade}", (-23.5 * mm, -4.4 * mm, -2 * mm), (23.5 * mm, 4.4 * mm, 2 * mm))


def real_drone(stage) -> None:
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, ROOT).GetPrim().GetReferences().AddReference(CF2X)


def add_sun(stage, intensity: float = 3859.0) -> None:
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr().Set(intensity)
    sun.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))


def load(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im)


def uv_area(st: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Area in uv space of each face of a face-varying coordinate array (shoelace)."""
    starts = np.cumsum(counts) - counts
    face = np.repeat(np.arange(len(counts)), counts)
    nxt = st[starts[face] + (np.arange(len(st)) - starts[face] + 1) % counts[face]]
    cross = st[:, 0] * nxt[:, 1] - nxt[:, 0] * st[:, 1]
    return 0.5 * np.abs(np.bincount(face, weights=cross, minlength=len(counts)))


def png_info(path: str) -> tuple[tuple[int, int], str]:
    with Image.open(path) as im:
        return im.size, im.mode


def make_stage(builder, **kw):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    builder(stage)
    scene.apply_drone_materials(stage, ROOT)
    summary = DL.apply_drone_look(stage, ROOT, TEX_CACHE, **kw)
    return stage, summary


def meshes(stage, root: str = ROOT) -> dict[str, Usd.Prim]:
    return {p.GetName(): p for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if p.IsA(UsdGeom.Mesh)}


def bound_name(prim) -> str | None:
    mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    return str(mat.GetPath()) if mat else None


def shader_of(stage, material_path: str):
    mat = UsdShade.Material(stage.GetPrimAtPath(material_path))
    return mat.ComputeSurfaceSource()[0]


@unittest.skipUnless(HAVE_PXR, "needs usd-core")
class Common:
    """Checks that must hold on any drone with the Crazyflie's prim names."""

    builder = None

    @classmethod
    def setUpClass(cls):
        cls.stage, cls.summary = make_stage(cls.builder)

    def blades(self):
        return [p for p in Usd.PrimRange(self.stage.GetPrimAtPath(ROOT)) if p.GetName() in DL.BLADE_NAMES]

    def discs(self):
        return [self.stage.GetPrimAtPath(f"{ROOT}/m{i}_prop/blur_disc") for i in range(1, 5)]

    def test_every_part_is_bound_to_a_look_material(self):
        by_name = meshes(self.stage)
        for name in ("cf_body_001", "battery_holder", "battery", "motor_mount", "Cylinder", "pin_headers"):
            self.assertIn("/World/Looks/DroneLook/", bound_name(by_name[name]) + "/", name)
        for p in Usd.PrimRange(self.stage.GetPrimAtPath(ROOT)):
            if p.IsA(UsdGeom.Mesh):
                self.assertIsNotNone(bound_name(p), f"{p.GetPath()} has no material")
        self.assertEqual(self.summary["meshes_bound"], 6)

    def test_textures_have_the_right_sizes(self):
        tex = self.summary["textures"]
        for role, path in tex["pcb"].items():
            self.assertEqual(png_info(path)[0], (2048, 2048), role)
        self.assertEqual(png_info(tex["pcb"]["albedo"])[1], "RGB")
        self.assertEqual(png_info(tex["pcb"]["roughness"])[1], "L")
        self.assertEqual(png_info(tex["pcb"]["normal"])[1], "RGB")
        for role, path in tex["battery"].items():
            self.assertEqual(png_info(path)[0], (1024, 1024), role)
        for role, path in tex["motor"].items():
            self.assertEqual(png_info(path)[0], (1024, 1024), role)
        for path in tex["pcb"].values():
            self.assertTrue(path.startswith(os.path.join(TEX_CACHE, "drone")))

    def test_pcb_texture_content(self):
        t = {k: load(p) for k, p in self.summary["textures"]["pcb"].items()}
        alb, rough, metal, nrm = t["albedo"].astype(float) / 255, t["roughness"] / 255.0, t["metallic"] / 255.0, t["normal"]
        lum = alb.mean(axis=2)
        self.assertLess(np.median(lum), 0.2, "mostly matte-black solder mask")
        self.assertGreater((lum > 0.7).mean(), 0.0002, "white silkscreen")
        gold = (metal > 0.9)
        self.assertGreater(gold.mean(), 0.0005, "metallic pads")
        self.assertGreater(alb[gold][:, 0].mean(), alb[gold][:, 2].mean() * 1.2, "gold: red above blue")
        self.assertGreater(rough.std(), 0.02, "roughness varies")
        self.assertLess(rough[~gold].min(), 0.45)
        self.assertGreater(rough[~gold].max(), 0.55)
        n = nrm.astype(float) / 255 * 2 - 1
        self.assertGreater(float(np.abs(n[..., :2]).max()), 0.3, "copper relief tilts the normal")
        self.assertGreater(n[..., 2].min(), 0.0)
        self.assertGreater(float(np.median(n[..., 2])), 0.97, "flat where there is no feature")

    def test_edge_patch_is_uniform(self):
        for key, path in (("pcb", self.summary["textures"]["pcb"]["albedo"]),
                          ("battery", self.summary["textures"]["battery"]["albedo"])):
            a = load(path)[-DL.EDGE_PX:, :DL.EDGE_PX]
            self.assertEqual(a.reshape(-1, 3).std(axis=0).max(), 0.0, key)

    def test_planar_uvs_authored_face_varying(self):
        by_name = meshes(self.stage)
        for name, size in (("cf_body_001", 2048), ("battery_holder", 2048), ("battery", 1024)):
            prim = by_name[name]
            pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")           # the standard set, over the asset's empty one
            self.assertEqual(pv.GetInterpolation(), "faceVarying")
            self.assertEqual(self.summary["uv"][str(prim.GetPath())], "st")
            st = np.array(pv.Get())
            self.assertEqual(len(st), len(prim.GetAttribute("faceVertexIndices").Get()))
            self.assertGreaterEqual(float(st.min()), 0.0)
            self.assertLessEqual(float(st.max()), 1.0)
            self.assertFalse(pv.GetIndicesAttr().HasAuthoredValue(), "no stale indices from the asset's primvar")
        # the PCB's top face maps by object-space x, y over the 64 mm atlas, the walls into the edge patch
        pts, counts, idx = DL._mesh_arrays(by_name["cf_body_001"])
        st = np.array(UsdGeom.PrimvarsAPI(by_name["cf_body_001"]).GetPrimvar("st").Get())
        normal_z = DL._face_normals(pts, counts, idx)[np.repeat(np.arange(len(counts)), counts), 2]
        flat, walls = np.abs(normal_z) >= 0.5, np.abs(normal_z) < 0.5
        self.assertTrue(flat.any() and walls.any())
        np.testing.assert_allclose(st[flat][:, 0], pts[idx][flat][:, 0] / 0.064 + 0.5, atol=1e-5)
        np.testing.assert_allclose(st[flat][:, 1], pts[idx][flat][:, 1] / 0.064 + 0.5, atol=1e-5)
        half = DL.EDGE_PX / 2 / 2048                                   # the patch is [0, 2 half]^2
        self.assertTrue(np.all(np.abs(st[walls] - half) <= 0.5 * half + 1e-6), "walls stay inside the edge patch")
        # ... with a real uv area on every wall that has a real area (a tangent frame needs one)
        area3d = 0.5 * np.linalg.norm(DL._face_normals(pts, counts, idx, unit=False), axis=1)
        wall_faces = np.array([bool(walls[s]) for s in np.cumsum(counts) - counts])
        self.assertTrue(np.all(uv_area(st, counts)[wall_faces & (area3d > 1e-10)] > 0.0))

    def test_pcb_material_reads_the_new_primvar(self):
        sh = shader_of(self.stage, "/World/Looks/DroneLook/pcb")
        self.assertEqual(sh.GetIdAttr().Get(), "UsdPreviewSurface")
        reader = UsdShade.Shader(self.stage.GetPrimAtPath("/World/Looks/DroneLook/pcb/stReader"))
        self.assertEqual(reader.GetInput("varname").Get(), "st")
        for name in ("diffuseColor", "roughness", "metallic", "normal"):
            self.assertTrue(sh.GetInput(name).HasConnectedSource(), name)
        n = UsdShade.Shader(self.stage.GetPrimAtPath("/World/Looks/DroneLook/pcb/normalTex"))
        self.assertEqual(n.GetInput("sourceColorSpace").Get(), "raw")
        self.assertEqual(tuple(n.GetInput("scale").Get()), (2.0, 2.0, 2.0, 1.0), "OpenGL normals: no green flip")

    def test_leds(self):
        leds = [self.stage.GetPrimAtPath(f"{BODY}/leds/{n}") for n in DL.LEDS]
        self.assertTrue(all(p.IsValid() for p in leds))
        self.assertEqual(self.summary["leds"], 4)
        cache = UsdGeom.XformCache()
        top = 0.01625
        for name, p in zip(DL.LEDS, leds):
            sx, sy, color = DL.LEDS[name]
            m = np.array(cache.GetLocalToWorldTransform(p))
            pts = np.array(UsdGeom.Mesh(p).GetPointsAttr().Get()) @ m[:3, :3] + m[3, :3]
            size = pts.max(axis=0) - pts.min(axis=0)
            self.assertAlmostEqual(float(pts[:, 2].min()), top, places=6, msg="sits on the PCB top")
            self.assertAlmostEqual(float(pts[:, 2].max() - pts[:, 2].min()), 0.6e-3, places=6)
            # about 1.6 x 0.8 mm, turned along the arm
            self.assertAlmostEqual(float(np.sort(np.hypot(*(pts[:, :2] - pts[:, :2].mean(axis=0)).T))[-1]),
                                   0.5 * math.hypot(1.6e-3, 0.8e-3), delta=2e-5)
            centre = pts.mean(axis=0)
            self.assertEqual((np.sign(centre[0]), np.sign(centre[1])), (sx, sy))
            self.assertAlmostEqual(float(np.hypot(centre[0], centre[1])), math.sqrt(2) * DL.LED_SITE_MM * 1e-3,
                                   delta=1e-4)
            self.assertLess(size.max(), 2.2e-3)
            # emissive, in the LED's colour
            sh = UsdShade.Shader(self.stage.GetPrimAtPath(f"/World/Looks/DroneLook/{name}/PBR"))
            e = np.array(sh.GetInput("emissiveColor").Get())
            self.assertEqual(int(np.argmax(e)), {"red": 0, "green": 1, "blue": 2}[color], name)
            self.assertGreater(float(e.max()), 1000.0)
            self.assertEqual(bound_name(p), f"/World/Looks/DroneLook/{name}")
        front = [DL.LEDS[n][2] for n in DL.LEDS if DL.LEDS[n][0] > 0]
        rear = [DL.LEDS[n][2] for n in DL.LEDS if DL.LEDS[n][0] < 0]
        self.assertEqual(sorted(front), ["green", "red"])
        self.assertEqual(rear, ["blue", "blue"])

    def test_led_radiance_follows_the_sun_and_stays_small(self):
        self.assertEqual(DL.led_nits(self.stage), DL.LED_NITS_DEFAULT)         # no sun in this stage
        stage = Usd.Stage.CreateInMemory()
        add_sun(stage, 3859.0)
        self.assertAlmostEqual(DL.led_nits(stage), 3.0 * 3859.0 / math.pi, places=3)
        # an LED throws little light: irradiance at 1 cm from a 1.6 x 0.8 mm patch is ~1 % of the sun's
        e_led = DL.led_nits(stage) * (1.6e-3 * 0.8e-3) / 0.01 ** 2
        self.assertLess(e_led / 3859.0, 0.02)

    def test_prop_discs(self):
        discs = self.discs()
        self.assertTrue(all(d.IsValid() for d in discs))
        self.assertEqual(len(self.summary["prop_discs"]), 4)
        for blade, disc in zip(sorted(self.blades(), key=lambda b: str(b.GetPath())),
                               sorted(discs, key=lambda d: str(d.GetPath()))):
            bp = np.array(UsdGeom.Mesh(blade).GetPointsAttr().Get())
            extent = float(np.hypot(bp[:, 0], bp[:, 1]).max())
            dp = np.array(UsdGeom.Mesh(disc).GetPointsAttr().Get())
            self.assertAlmostEqual(float(np.hypot(dp[:, 0], dp[:, 1]).max()), extent, delta=extent * 0.01)
            self.assertEqual(blade.GetParent().GetPath(), disc.GetParent().GetPath())
            self.assertLess(float(dp[:, 2].max() - dp[:, 2].min()), 1e-6, "flat")
            self.assertTrue(UsdGeom.Mesh(disc).GetDoubleSidedAttr().Get())
            self.assertTrue(UsdGeom.PrimvarsAPI(disc).GetPrimvar("doNotCastShadows").Get())
            self.assertEqual(len(UsdGeom.PrimvarsAPI(disc).GetPrimvar("st").Get()), len(dp))

    def test_prop_disc_material_is_translucent_not_cut_out(self):
        disc = self.discs()[0]
        mat = UsdShade.MaterialBindingAPI(disc).ComputeBoundMaterial()[0]
        sh = mat.ComputeSurfaceSource()[0]
        self.assertEqual(sh.GetInput("opacityThreshold").Get(), 0.0)
        src = sh.GetInput("opacity").GetConnectedSources()[0][0]
        self.assertEqual(src.sourceName, "a")
        tex = UsdShade.Shader(src.source.GetPrim())
        path = str(tex.GetInput("file").Get().path)
        a = load(path)
        self.assertEqual(a.shape, (1024, 1024, 4))
        alpha = a[..., 3] / 255.0
        row = alpha[512, 512:]
        self.assertGreater(row[0], 0.8, "opaque hub")
        mid = row[int(0.4 * 512)]
        self.assertTrue(0.1 < mid < 0.5, mid)
        self.assertLess(row[int(0.9 * 512)], mid, "thinner towards the tip")
        self.assertLess(row[int(0.9 * 512)], 0.15)
        self.assertEqual(alpha[0, 0], 0.0, "nothing outside the disc")
        self.assertTrue(np.all(np.diff(row[40:480].astype(float)) <= 2 / 255.0), "radially smooth and decreasing")

    def test_blur_gain_scales_the_alpha(self):
        def mid_alpha(stage) -> float:
            disc = stage.GetPrimAtPath(f"{ROOT}/m1_prop/blur_disc")
            mat = UsdShade.MaterialBindingAPI(disc).ComputeBoundMaterial()[0]
            src = mat.ComputeSurfaceSource()[0].GetInput("opacity").GetConnectedSources()[0][0]
            png = str(UsdShade.Shader(src.source.GetPrim()).GetInput("file").Get().path)
            return float(load(png)[512, 512 + 200, 3]) / 255

        stage, _ = make_stage(self.builder, blur_gain=1.6)
        self.assertAlmostEqual(mid_alpha(stage) / mid_alpha(self.stage), 1.6, delta=0.1)

    def test_blades_hidden_with_blur_and_visible_without(self):
        for b in self.blades():
            self.assertEqual(UsdGeom.Imageable(b).ComputeVisibility(), UsdGeom.Tokens.invisible)
        stage, summary = make_stage(self.builder, prop_blur=False)
        for p in Usd.PrimRange(stage.GetPrimAtPath(ROOT)):
            if p.GetName() in DL.BLADE_NAMES:
                self.assertEqual(UsdGeom.Imageable(p).ComputeVisibility(), UsdGeom.Tokens.inherited)
            self.assertNotEqual(p.GetName(), "blur_disc")
        self.assertEqual(summary["prop_discs"], [])
        self.assertFalse(summary["blades_hidden"])
        # and back: the same stage again with blur
        DL.apply_drone_look(stage, ROOT, TEX_CACHE, prop_blur=True)
        self.assertTrue(stage.GetPrimAtPath(f"{ROOT}/m1_prop/blur_disc").IsValid())
        blade = next(p for p in Usd.PrimRange(stage.GetPrimAtPath(f"{ROOT}/m1_prop")) if p.GetName() in DL.BLADE_NAMES)
        self.assertEqual(UsdGeom.Imageable(blade).ComputeVisibility(), UsdGeom.Tokens.invisible)

    def test_leds_off(self):
        stage, summary = make_stage(self.builder, leds=False)
        self.assertFalse(stage.GetPrimAtPath(f"{BODY}/leds").IsValid())
        self.assertEqual(summary["leds"], 0)
        DL.apply_drone_look(stage, ROOT, TEX_CACHE, leds=True)
        self.assertTrue(stage.GetPrimAtPath(f"{BODY}/leds/led_fl").IsValid())
        DL.apply_drone_look(stage, ROOT, TEX_CACHE, leds=False)
        self.assertFalse(stage.GetPrimAtPath(f"{BODY}/leds").IsValid())

    def test_other_arguments_replace_instead_of_piling_up(self):
        def look_children(stage, prefix):
            return sorted(c.GetName() for c in stage.GetPrimAtPath("/World/Looks/DroneLook").GetChildren()
                          if c.GetName().startswith(prefix))

        stage, _ = make_stage(self.builder)
        first = look_children(stage, "prop_blur_")
        self.assertIn(len(first), (1, 2), "one blur material per measured blade profile")
        DL.apply_drone_look(stage, ROOT, TEX_CACHE, blur_gain=1.6)
        again = look_children(stage, "prop_blur_")
        self.assertEqual(len(again), len(first))
        self.assertFalse(set(again) & set(first), "the gain-1 materials are gone")
        for d in range(1, 5):
            mat = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(f"{ROOT}/m{d}_prop/blur_disc")).ComputeBoundMaterial()[0]
            self.assertIn(mat.GetPrim().GetName(), again)
        DL.apply_drone_look(stage, ROOT, TEX_CACHE, prop_blur=False, leds=False)
        self.assertEqual(look_children(stage, "prop_blur_"), [])
        self.assertEqual(look_children(stage, "led_"), [])
        DL.apply_drone_look(stage, ROOT, TEX_CACHE)
        self.assertEqual(look_children(stage, "prop_blur_"), first)
        self.assertEqual(len(look_children(stage, "led_")), 4)

    def test_idempotent(self):
        stage, first = make_stage(self.builder)
        text1 = stage.GetRootLayer().ExportToString()
        mtimes = {p: os.path.getmtime(p) for tex in first["textures"].values() for p in tex.values()}
        again = DL.apply_drone_look(stage, ROOT, TEX_CACHE)
        self.assertEqual(stage.GetRootLayer().ExportToString(), text1, "a second call changes nothing")
        self.assertEqual(again, first)
        for p, t in mtimes.items():
            self.assertEqual(os.path.getmtime(p), t, "textures are generated once")
        n_prims = len(list(Usd.PrimRange(stage.GetPseudoRoot())))
        DL.apply_drone_look(stage, ROOT, TEX_CACHE)
        self.assertEqual(len(list(Usd.PrimRange(stage.GetPseudoRoot()))), n_prims)

    def test_pin_headers_split_into_housing_and_pins(self):
        pins = self.stage.GetPrimAtPath(f"{BODY}/pin_headers/pins")
        self.assertTrue(pins.IsValid())
        self.assertGreater(self.summary["pin_faces"], 0)
        sub = UsdGeom.Subset(pins)
        self.assertEqual(sub.GetFamilyNameAttr().Get(), "materialBind")
        self.assertEqual(bound_name(pins), "/World/Looks/DroneLook/gold")
        self.assertEqual(bound_name(self.stage.GetPrimAtPath(f"{BODY}/pin_headers")), "/World/Looks/DroneLook/housing")
        mesh = UsdGeom.Mesh(self.stage.GetPrimAtPath(f"{BODY}/pin_headers"))
        self.assertLess(len(sub.GetIndicesAttr().Get()), len(mesh.GetFaceVertexCountsAttr().Get()))
        gold = shader_of(self.stage, "/World/Looks/DroneLook/gold")
        self.assertEqual(gold.GetInput("metallic").Get(), 1.0)
        # the housing is weaker than the subset, so the subset's material wins on the pins
        rel = UsdShade.MaterialBindingAPI(self.stage.GetPrimAtPath(f"{BODY}/pin_headers")).GetDirectBindingRel()
        self.assertEqual(UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(rel), UsdShade.Tokens.weakerThanDescendants)

    def test_materials_are_plain_preview_surfaces(self):
        for mat in Usd.PrimRange(self.stage.GetPrimAtPath("/World/Looks/DroneLook")):
            if mat.IsA(UsdShade.Material):
                shader = shader_of(self.stage, str(mat.GetPath()))
                self.assertEqual(shader.GetIdAttr().Get(), "UsdPreviewSurface", mat.GetPath())

    def test_mount_motor_and_battery_look(self):
        mount = shader_of(self.stage, "/World/Looks/DroneLook/mount")
        self.assertGreater(min(mount.GetInput("diffuseColor").Get()), 0.55, "off-white")
        self.assertGreater(mount.GetInput("clearcoat").Get(), 0.0, "a slight sheen")
        self.assertEqual(mount.GetInput("metallic").Get(), 0.0)
        motor = shader_of(self.stage, "/World/Looks/DroneLook/motor")
        self.assertEqual(motor.GetInput("metallic").Get(), 1.0)
        self.assertTrue(motor.GetInput("roughness").HasConnectedSource(), "brushing is in the roughness texture")
        battery = shader_of(self.stage, "/World/Looks/DroneLook/battery")
        self.assertTrue(battery.GetInput("metallic").HasConnectedSource())


class TestSyntheticDrone(Common, unittest.TestCase):
    builder = staticmethod(fake_drone)


@unittest.skipUnless(HAVE_PXR and os.path.exists(CF2X), "needs the mirrored cf2x.usd (prepare.py --mirror)")
class TestRealCrazyflie(Common, unittest.TestCase):
    builder = staticmethod(real_drone)

    def test_all_ten_meshes_bound(self):
        names = [p.GetName() for p in Usd.PrimRange(self.stage.GetPrimAtPath(ROOT)) if p.IsA(UsdGeom.Mesh)
                 and p.GetName() not in ("blur_disc",) and not p.GetName().startswith("led_")]
        self.assertEqual(len(names), 10)
        for p in Usd.PrimRange(self.stage.GetPrimAtPath(ROOT)):
            if p.IsA(UsdGeom.Mesh):
                self.assertIsNotNone(bound_name(p))

    def test_real_blade_extent_and_hub(self):
        r = {round(r, 4) for r in self.summary["prop_radius"]}
        self.assertEqual(r, {0.0235, 0.0237})
        self.assertEqual(len(self.summary["prop_discs"]), 4)

    def test_pins_are_the_0p5_mm_pins(self):
        self.assertEqual(self.summary["pin_faces"], 240)

    def test_led_radiance_default_without_sun(self):
        self.assertEqual(self.summary["led_radiance"], DL.LED_NITS_DEFAULT)

    def test_uv_wireframes(self):
        out = tempfile.mkdtemp()
        by_name = meshes(self.stage)
        asset = Usd.Stage.Open(CF2X)                                   # the asset as shipped, before the look
        raw = meshes(asset, "/crazyflie")
        # the asset's own st: empty on the PCB (so planar), real on the motors, half degenerate on the props
        pcb_raw = DL.uv_wireframe(raw["cf_body_001"], os.path.join(out, "pcb_asset_st.png"), 256, "st")
        self.assertFalse(pcb_raw["valid"])
        self.assertEqual(pcb_raw["corners"], 0)
        motor = DL.uv_wireframe(by_name["Cylinder"], os.path.join(out, "motor_st.png"), 256, "st")
        self.assertTrue(motor["valid"])
        self.assertEqual(motor["degenerate"], 0)
        prop = DL.uv_wireframe(self.blades()[0], os.path.join(out, "prop_st.png"), 256, "st")
        self.assertGreater(prop["degenerate"], prop["faces"] * 0.4)
        # the look's planar st: complete, with no degenerate triangle on the PCB or the battery (the holder
        # mesh has four zero-area triangles of its own)
        pcb = DL.uv_wireframe(by_name["cf_body_001"], os.path.join(out, "pcb_st.png"), 256, "st",
                              background=self.summary["textures"]["pcb"]["albedo"])
        self.assertTrue(pcb["valid"])
        self.assertEqual(pcb["degenerate"], 0)
        self.assertEqual(png_info(os.path.join(out, "pcb_st.png"))[0], (256, 256))
        self.assertEqual(DL.uv_wireframe(by_name["battery"], os.path.join(out, "battery_st.png"), 256)["degenerate"], 0)
        self.assertLessEqual(DL.uv_wireframe(by_name["battery_holder"], os.path.join(out, "holder_st.png"), 256)
                             ["degenerate"], 4)

    def test_leds_sit_on_the_board_and_are_not_covered(self):
        """Every LED footprint lies inside the PCB's top face and under no mount, holder or battery face."""
        def tris(name, zmin):
            p = meshes(self.stage)[name]
            pts, cnt, idx = DL._mesh_arrays(p)
            n = DL._face_normals(pts, cnt, idx)
            return np.array([pts[t][:, :2] for f, t in enumerate(idx.reshape(-1, 3))
                             if n[f, 2] > 0.5 and pts[t, 2].mean() > zmin])

        def inside(pt, tr):
            a, b, c = tr[:, 0], tr[:, 1], tr[:, 2]

            def s(p, q, r):
                return (p[0] - r[:, 0]) * (q[:, 1] - r[:, 1]) - (q[:, 0] - r[:, 0]) * (p[1] - r[:, 1])
            d1, d2, d3 = s(pt, a, b), s(pt, b, c), s(pt, c, a)
            return bool((~(((d1 < 0) | (d2 < 0) | (d3 < 0)) & ((d1 > 0) | (d2 > 0) | (d3 > 0)))).any())
        board = tris("cf_body_001", 0.0162)
        covers = np.concatenate([tris("motor_mount", 0.0165), tris("battery_holder", 0.0), tris("battery", 0.0205)])
        cache = UsdGeom.XformCache()
        for name in DL.LEDS:
            p = self.stage.GetPrimAtPath(f"{BODY}/leds/{name}")
            m = np.array(cache.GetLocalToWorldTransform(p))
            pts = np.array(UsdGeom.Mesh(p).GetPointsAttr().Get()) @ m[:3, :3] + m[3, :3]
            for x, y, _ in pts:
                self.assertTrue(inside((x, y), board), f"{name} corner off the board")
                self.assertFalse(inside((x, y), covers), f"{name} corner hidden from above")


@unittest.skipUnless(HAVE_PXR, "needs usd-core")
class TestPure(unittest.TestCase):
    def test_planar_st(self):
        cube = scene.box(-0.01, 0.01, -0.02, 0.02, 0.0, 0.01, 1.0)
        pts, cnt, idx = cube[0], cube[1], cube[2]
        st = DL.planar_st(pts, cnt, idx, (0.064, 0.048), (0.25, 0.75))
        self.assertEqual(st.shape, (len(idx), 2))
        # scene.box lists the top face first (4 corners), then four walls
        np.testing.assert_allclose(st[:4, 0], pts[idx][:4, 0] / 0.064 + 0.5, atol=1e-6)
        np.testing.assert_allclose(st[:4, 1], pts[idx][:4, 1] / 0.048 + 0.5, atol=1e-6)
        np.testing.assert_allclose(st[4:], np.tile([0.25, 0.75], (16, 1)), atol=1e-6)

    def test_planar_st_walls_keep_a_uv_area(self):
        cube = scene.box(-0.01, 0.01, -0.02, 0.02, 0.0, 0.01, 1.0)
        pts, cnt, idx = cube[0], cube[1], cube[2]
        st = DL.planar_st(pts, cnt, idx, (0.064, 0.064), (0.25, 0.75), edge_half=0.02)
        np.testing.assert_allclose(st[:4, 0], pts[idx][:4, 0] / 0.064 + 0.5, atol=1e-6)          # the top is unchanged
        walls = st[4:]
        self.assertTrue(np.all(np.abs(walls - [0.25, 0.75]) <= 0.5 * 0.02 + 1e-6), "inside the patch")
        area = uv_area(st, cnt)
        self.assertTrue(np.all(area[1:] > 0.0), "every wall has a uv area")
        # walls are mapped by (distance along the wall, height): the two long walls differ in orientation only
        self.assertGreater(float(np.ptp(walls[:, 1])), 0.0)
        # right-handed like the top: a counter-clockwise wall (seen from outside) stays counter-clockwise in uv
        n = DL._face_normals(pts, cnt, idx)
        starts = np.cumsum(cnt) - cnt
        for f in range(1, len(cnt)):
            q = st[starts[f]:starts[f] + cnt[f]]
            signed = sum(q[i, 0] * q[(i + 1) % len(q), 1] - q[(i + 1) % len(q), 0] * q[i, 1] for i in range(len(q)))
            self.assertGreater(signed, 0.0, f"wall {f} normal {n[f]}")

    def test_blade_profile_matches_geometry(self):
        """Two straight blades of width w (one bar through the hub): the arcs inside the bar give the coverage."""
        w, R = 0.006, 0.024
        pts = np.array([[-R, -w / 2], [R, -w / 2], [R, w / 2], [-R, w / 2]])
        tris = np.array([[0, 1, 2], [0, 2, 3]])
        prof = DL.blade_profile(pts, tris, R, bins=48)
        for k in (12, 24, 36):
            r = (k + 0.5) / 48 * R
            self.assertAlmostEqual(float(prof[k]), 4 * math.asin(w / (2 * r)) / (2 * math.pi), delta=0.02)
        self.assertGreater(prof[0], 0.95, "the hub of two crossed blades is solid")

    def test_prop_blur_texture_is_cached_by_profile(self):
        d = tempfile.mkdtemp()
        prof = np.linspace(1.0, 0.1, 96).astype(np.float32)
        p1 = DL.prop_blur_texture(d, prof)
        t = os.path.getmtime(p1)
        time.sleep(0.01)
        self.assertEqual(DL.prop_blur_texture(d, prof), p1)
        self.assertEqual(os.path.getmtime(p1), t)
        self.assertNotEqual(DL.prop_blur_texture(d, prof * 0.5), p1)

    def test_pcb_texture_cache_is_reused(self):
        files = DL.pcb_textures(TEX_CACHE)
        t = {k: os.path.getmtime(v) for k, v in files.items()}
        self.assertEqual({k: os.path.getmtime(v) for k, v in DL.pcb_textures(TEX_CACHE).items()}, t)


@unittest.skipUnless(HAVE_PXR, "needs usd-core")
class TestPreviewReadsTheLook(unittest.TestCase):
    """The offline preview (mitsuba_preview) must see what RTX sees; only the material graph is read here."""

    @classmethod
    def setUpClass(cls):
        try:
            from drone_scene import mitsuba_preview as MP
        except ImportError as e:  # pragma: no cover
            raise unittest.SkipTest(f"mitsuba_preview not importable: {e}")
        cls.MP = MP
        cls.stage, _ = make_stage(fake_drone)

    def spec(self, path: str):
        prim = self.stage.GetPrimAtPath(path)
        mats = self.MP._Materials(Usd.TimeCode.EarliestTime(), self.MP._Report(None), None, None)
        return mats.spec_for(UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0])

    def test_pcb_spec(self):
        s = self.spec(f"{BODY}/cf_body_001")
        self.assertEqual(s.uv_set, "st")
        self.assertIsNotNone(s.base_tex)
        self.assertIsNotNone(s.roughness_tex)
        self.assertIsNotNone(s.metallic_tex)
        self.assertIsNotNone(s.normal_tex)

    def test_disc_spec_is_blended_opacity(self):
        s = self.spec(f"{ROOT}/m1_prop/blur_disc")
        self.assertEqual(s.opacity_tex.channel, "a")
        self.assertEqual(s.opacity_threshold, 0.0)

    def test_led_spec_emits(self):
        s = self.spec(f"{BODY}/leds/led_fl")
        self.assertTrue(s.emissive)
        self.assertGreater(max(s.emission), 1000.0)

    def test_hidden_blades_are_skipped_by_the_collector(self):
        MP = self.MP
        col = MP._Collector(self.stage, Usd.TimeCode.EarliestTime(), MP._Report(None),
                            MP._Materials(Usd.TimeCode.EarliestTime(), MP._Report(None), None, None))
        paths = {str(p.GetPath()) for p, _ in col._occurrences(self.stage.GetPseudoRoot(), None)}
        self.assertFalse(any(p.endswith(("ccw_prop", "cw_prop")) for p in paths))
        self.assertEqual(sum(p.endswith("blur_disc") for p in paths), 4)


def ground_ratio(under: np.ndarray, free: np.ndarray) -> float:
    return float(under.mean() / free.mean())


@unittest.skipUnless(HAVE_PXR, "needs usd-core")
class TestMitsubaShadowsThroughMasks(unittest.TestCase):
    """The premise of the prop blur in the offline preview, on Mitsuba alone: a translucent ``mask`` BSDF.

    ``path`` shadows the ground below a 20 %-opaque disc as if it were solid; ``volpath`` lets 80 % through.
    If this fails after a Mitsuba upgrade, ``path`` has been fixed and the preview no longer needs ``volpath``.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import mitsuba as mi
        except ImportError as e:  # pragma: no cover
            raise unittest.SkipTest(f"mitsuba not importable: {e}")
        mi.set_variant("scalar_rgb")
        cls.mi = mi

    def ground_under_disc(self, integrator: str, opacity: float = 0.2) -> float:
        """Radiance of the ground below a mask disc, over the radiance of unshadowed ground (sun straight down)."""
        mi = self.mi
        scene_dict = {
            "type": "scene", "integrator": {"type": integrator, "max_depth": 4},
            "ground": {"type": "rectangle", "to_world": mi.ScalarTransform4f().scale([5, 5, 1]),
                       "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.5, 0.5, 0.5]}}},
            "disc": {"type": "disk", "to_world": mi.ScalarTransform4f().translate([0, 0, 0.5]),
                     "bsdf": {"type": "mask", "opacity": {"type": "rgb", "value": [opacity] * 3},
                              "material": {"type": "twosided", "material": {
                                  "type": "diffuse", "reflectance": {"type": "rgb", "value": [0.03] * 3}}}}},
            "sun": {"type": "directional", "direction": [0, 0, -1], "irradiance": {"type": "rgb", "value": [1, 1, 1]}},
            "sensor": {"type": "perspective", "fov": 20,
                       "to_world": mi.ScalarTransform4f().look_at(origin=[0, 0, 8], target=[0, 0, 0], up=[0, 1, 0]),
                       "film": {"type": "hdrfilm", "width": 16, "height": 16, "rfilter": {"type": "box"}},
                       "sampler": {"type": "independent", "sample_count": 512}}}
        img = np.array(mi.render(mi.load_dict(scene_dict)))[..., :3].mean(axis=-1)
        return float(img[6:10, 6:10].mean() / (0.5 / math.pi))

    def test_path_shadows_a_translucent_disc_fully(self):
        self.assertLess(self.ground_under_disc("path"), 0.1)

    def test_volpath_lets_the_light_through(self):
        # 80 % of the sun through the disc, and 80 % of the ground's light back through it to the camera
        self.assertAlmostEqual(self.ground_under_disc("volpath"), 0.8 * 0.8, delta=0.06)


@unittest.skipUnless(HAVE_PXR, "needs usd-core")
class TestPreviewDiscShadow(unittest.TestCase):
    """End to end in ``mitsuba_preview``: the sun reaches the ground below a blur disc, seen from above.

    Needs the preview's ``integrator="volpath"`` option, so it is skipped until ``mitsuba_preview.render`` has
    it (without it the discs shadow the mounts and arms as solid; use ``prop_blur=False`` then).
    """

    @classmethod
    def setUpClass(cls):
        try:
            from drone_scene import mitsuba_preview as MP
        except ImportError as e:  # pragma: no cover
            raise unittest.SkipTest(f"mitsuba_preview not importable: {e}")
        if "integrator" not in inspect.signature(MP.render).parameters:
            raise unittest.SkipTest("mitsuba_preview.render has no integrator option yet")
        cls.MP = MP

    def stage(self) -> Usd.Stage:
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        fake_drone(stage)
        scene.apply_drone_materials(stage, ROOT)
        DL.apply_drone_look(stage, ROOT, TEX_CACHE, leds=False)
        ground = scene.preview_surface(stage, "/World/Looks/Ground", color=(0.5, 0.5, 0.5), roughness=1.0)
        scene._mesh(stage, "/World/Ground", scene.box(-0.4, 0.4, -0.4, 0.4, -0.01, 0.0, 1.0), ground)
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")          # shines along -Z
        sun.CreateIntensityAttr().Set(1.0)
        cam = UsdGeom.Camera.Define(stage, "/World/Cam")               # above the m1 prop, looking down
        cam.CreateFocalLengthAttr(70.0)
        cam.CreateHorizontalApertureAttr(36.0)
        cam.CreateVerticalApertureAttr(20.25)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))
        cam.AddTranslateOp().Set(Gf.Vec3d(0.031, -0.031, 0.2))
        return stage

    def ratio(self, integrator: str) -> float:
        import mitsuba as mi
        exr = os.path.join(tempfile.mkdtemp(dir=TEX_CACHE), "r.exr")
        self.MP.render(self.stage(), "/World/Cam", os.path.join(os.path.dirname(exr), "r.png"), width=128, height=72,
                       spp=48, variant="scalar_rgb", save_exr=exr, integrator=integrator)
        img = np.array(mi.Bitmap(exr))[..., :3].mean(axis=-1)
        # the image centre is the prop axis; 1 px = 0.8 mm, the disc is 29 px in radius.  Ground below the disc
        # at 0.3-0.6 of its radius, away from the body (down and to the right), against ground far outside it.
        return ground_ratio(img[42:48, 70:76], img[56:68, 110:124])

    def test_sun_reaches_the_ground_below_the_disc(self):
        r = self.ratio("volpath")
        self.assertGreater(r, 0.4, "the disc must not shadow like a solid")
        self.assertLess(r, 0.9)

    def test_path_is_the_default_and_shadows_solid(self):
        self.assertLess(self.ratio("path"), 0.25)       # measured 0.12 (the disc's own dark reflection)


if __name__ == "__main__":
    unittest.main()
