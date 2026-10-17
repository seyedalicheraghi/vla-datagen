"""Offline checks of the drone scene layout and its USD (usd-core + numpy; no Isaac Sim, no network).

    python scripts/drone_scene/test/test_scene.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

try:
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade
    HAVE_PXR = True
except ImportError:  # pragma: no cover
    HAVE_PXR = False

from drone_scene import assets, scene  # noqa: E402

CACHE = os.environ.get("DRONE_SCENE_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "drone_scene"))
CF2X = os.path.join(CACHE, "Isaac", "Robots", "Bitcraze", "Crazyflie", "cf2x.usd")      # mirrored by prepare.py
PAVERS_MDL = os.path.join(CACHE, "NVIDIA", "Materials", "Base", "Masonry", "Brick_Pavers.mdl")


def fake_manifest(tmp: str) -> dict:
    """Texture/sky paths are just strings; props are tiny local cm-unit USDs (a 100 cm cube)."""
    props = {}
    for name in assets.PROPS:
        path = os.path.join(tmp, f"{name}.usda")
        st = Usd.Stage.CreateNew(path)
        UsdGeom.SetStageMetersPerUnit(st, 0.01)
        UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(st, "/Root")
        root.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))          # like NVIDIA's assets: own ops on the root
        root.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))
        UsdGeom.Cube.Define(st, "/Root/Box").CreateSizeAttr(100.0)
        st.SetDefaultPrim(root.GetPrim())
        st.GetRootLayer().Save()
        props[name] = path
    textures = {n: {"albedo": f"{tmp}/{n}_c.png", "orm": f"{tmp}/{n}_orm.png", "normal": f"{tmp}/{n}_n.png"}
                for n in ("pavers", "lawn", "mulch", "curb")}
    sky = {"file": f"{tmp}/sky.hdr", "dome_intensity": 900.0, "sun_intensity": 50000.0, "sun_color": [1.0, 0.95, 0.9]}
    return {"textures": textures, "sky": sky, "props": props, "crazyflie": CF2X}


@unittest.skipUnless(HAVE_PXR, "needs usd-core")
class TestDroneScene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.cfg = scene.SceneCfg()
        cls.man = fake_manifest(cls.tmp)
        cls.stage = Usd.Stage.CreateInMemory()
        cls.summary = scene.build(cls.stage, cls.cfg, cls.man, physics=True)
        cls.xf = UsdGeom.XformCache()

    def test_summary(self):
        s = self.summary
        self.assertEqual(s["materials"], 4)
        self.assertGreater(s["grass_clumps"], 300)
        self.assertEqual(s["leaves"], self.cfg.leaves)
        self.assertEqual(s["props"], len(scene.PLANTING))
        self.assertEqual(s["colliders"], 4)

    def test_ground_material_network(self):
        mat = UsdShade.Material(self.stage.GetPrimAtPath("/World/Scenery/Looks/pavers"))
        pbr = mat.ComputeSurfaceSource()[0]
        self.assertEqual(pbr.GetIdAttr().Get(), "UsdPreviewSurface")
        rough_src = pbr.GetInput("roughness").GetConnectedSources()[0][0]
        self.assertEqual(rough_src.sourceName, "g", "roughness from the ORM green channel")
        n = UsdShade.Shader(self.stage.GetPrimAtPath("/World/Scenery/Looks/pavers/normalTex"))
        self.assertEqual(n.GetInput("sourceColorSpace").Get(), "raw")
        scale, bias = n.GetInput("scale").Get(), n.GetInput("bias").Get()
        # DirectX normal map: green is flipped -> n = (2r-1, 1-2g, 2b-1)
        self.assertEqual((scale[0], scale[1], scale[2]), (2.0, -2.0, 2.0))
        self.assertEqual((bias[0], bias[1], bias[2]), (-1.0, 1.0, -1.0))
        a = UsdShade.Shader(self.stage.GetPrimAtPath("/World/Scenery/Looks/pavers/albedoTex"))
        self.assertEqual(a.GetInput("sourceColorSpace").Get(), "sRGB")

    def test_texture_scale_in_metres(self):
        patio = UsdGeom.Mesh(self.stage.GetPrimAtPath("/World/Scenery/Ground/Patio"))
        pts = np.array(patio.GetPointsAttr().Get())
        st = np.array(UsdGeom.PrimvarsAPI(patio).GetPrimvar("st").Get())
        top = np.isclose(np.array(patio.GetNormalsAttr().Get())[:, 2], 1.0)       # top-face vertices
        np.testing.assert_allclose(st[top], pts[top, :2] / assets.MATERIALS["pavers"].tile_m, atol=1e-5)

    def test_drone_hovers_over_pavers(self):
        x0, x1, y0, y1 = self.cfg.patio
        dx, dy, dz = self.cfg.drone_pos
        self.assertTrue(x0 + 1.0 < dx < x1 - 1.0 and y0 + 1.0 < dy < y1 - 1.0, "at least 1 m of pavers around")
        self.assertAlmostEqual(dz - self.cfg.patio_top, 0.5)

    def test_grass_clear_of_hard_surfaces(self):
        clumps = scene.grass_instances(self.cfg)
        half = {t[0]: t[1] for t in scene.GRASS_TIERS}
        rects = scene.hard_rects(self.cfg)
        for proto, x, y in clumps:
            # corners of the clump's square footprint at the largest scale and any yaw
            r = half[proto] * scene.GRASS_MAX_SCALE * math.sqrt(2)
            for a in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                cx, cy = x + r * math.cos(a), y + r * math.sin(a)
                for x0, x1, y0, y1 in rects:
                    self.assertFalse(x0 < cx < x1 and y0 < cy < y1, f"{proto} at ({x:.2f}, {y:.2f}) pokes into {x0, x1, y0, y1}")
        by_proto = {p: np.array([(x, y) for q, x, y in clumps if q == p]) for p in half}
        for proto, _, spacing, _ in scene.GRASS_TIERS:
            pts = by_proto[proto]
            self.assertGreater(len(pts), 20, proto)
            d = np.linalg.norm(pts[:, None] - pts[None], axis=-1) + np.eye(len(pts)) * 1e9
            self.assertGreaterEqual(d.min(), spacing - 1e-9, f"{proto}: Poisson spacing")
        # small clumps reach the curb: some clump within 0.25 m of the patio curb
        d_patio = [scene.rect_distance(x, y, rects[0]) for _, x, y in clumps]
        self.assertLess(min(d_patio), 0.25, "grass grows right up to the curb")

    def test_leaves_on_patio_not_under_drone(self):
        pts = scene.leaf_positions(self.cfg)
        x0, x1, y0, y1 = self.cfg.patio
        self.assertEqual(len(pts), self.cfg.leaves)
        self.assertTrue(np.all((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)))
        self.assertGreater(np.hypot(pts[:, 0], pts[:, 1]).min(), 0.35)
        inst = UsdGeom.PointInstancer(self.stage.GetPrimAtPath("/World/Scenery/Patio/Leaves"))
        self.assertEqual(len(inst.GetPositionsAttr().Get()), self.cfg.leaves)

    def test_props_scaled_and_grounded(self):
        prim = self.stage.GetPrimAtPath("/World/Scenery/Props/maple_1")
        m = self.xf.GetLocalToWorldTransform(prim)
        sx = Gf.Vec3d(m[0][0], m[0][1], m[0][2]).GetLength()
        self.assertAlmostEqual(sx, assets.PROPS["maple"].units * 1.0, places=6, msg="cm asset scaled to metres")
        self.assertAlmostEqual(m.ExtractTranslation()[2], self.cfg.lawn_z, places=6)
        leaf = self.xf.GetLocalToWorldTransform(self.stage.GetPrimAtPath("/World/Scenery/Props/leaves_maple_1"))
        self.assertAlmostEqual(leaf.ExtractTranslation()[2], self.cfg.patio_top, places=6, msg="leaves lie on the pavers")

    def test_sun_points_from_the_sun(self):
        sun = self.stage.GetPrimAtPath("/World/Scenery/Sun")
        self.assertTrue(sun.IsA(UsdLux.DistantLight))
        m = self.xf.GetLocalToWorldTransform(sun)
        travel = np.array(m.TransformDir(Gf.Vec3d(0, 0, -1)))       # a DistantLight shines along local -Z
        np.testing.assert_allclose(travel / np.linalg.norm(travel), -scene.sun_direction(self.cfg.sun), atol=1e-5)
        self.assertGreater(scene.sun_direction(self.cfg.sun)[2], math.sin(math.radians(40)))

    def test_cameras_look_at_targets(self):
        for c in self.cfg.cameras:
            prim = self.stage.GetPrimAtPath(f"/World/Cameras/{c.name}")
            m = self.xf.GetLocalToWorldTransform(prim)
            eye = np.array(m.ExtractTranslation())
            fwd = np.array(m.TransformDir(Gf.Vec3d(0, 0, -1)))
            to_target = np.subtract(c.target, c.eye)
            np.testing.assert_allclose(eye, c.eye, atol=1e-6)
            self.assertGreater(np.dot(fwd / np.linalg.norm(fwd), to_target / np.linalg.norm(to_target)), 0.9999, c.name)
        demo = self.cfg.camera("demo")
        self.assertEqual((demo.eye, demo.target), ((0.5, 0.5, 1.0), (0.0, 0.0, 0.5)), "quadcopter.py's view")

    def test_drone_in_hero_and_demo_frames(self):
        for name in ("demo", "hero"):
            c = self.cfg.camera(name)
            m = scene.look_at_matrix(c.eye, c.target)
            p = np.append(self.cfg.drone_pos, 1.0) @ np.linalg.inv(m)          # world -> camera
            f = c.focal_length / c.horizontal_aperture * c.width
            u, v = c.width / 2 + f * p[0] / -p[2], c.height / 2 - f * p[1] / -p[2]
            self.assertLess(p[2], 0, f"drone in front of the {name} camera")
            self.assertTrue(0.2 * c.width < u < 0.8 * c.width and 0.2 * c.height < v < 0.8 * c.height, (name, u, v))

    def test_quat_z_to(self):
        rng = np.random.default_rng(0)
        for d in rng.normal(size=(20, 3)):
            d /= np.linalg.norm(d)
            w, x, y, z = scene.quat_z_to(d)
            r = Gf.Rotation(Gf.Quatd(w, x, y, z))
            np.testing.assert_allclose(np.array(r.TransformDir(Gf.Vec3d(0, 0, 1))), d, atol=1e-9)

    @unittest.skipUnless(os.path.exists(CF2X), "Crazyflie USD not mirrored yet (run prepare.py --mirror)")
    def test_drone_materials_bind_every_mesh(self):
        st = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(st, "/World")
        d = UsdGeom.Xform.Define(st, "/World/Crazyflie")
        d.GetPrim().GetReferences().AddReference(CF2X)
        n = scene.apply_drone_materials(st, "/World/Crazyflie")
        meshes = [p for p in Usd.PrimRange(st.GetPrimAtPath("/World/Crazyflie")) if p.IsA(UsdGeom.Mesh)]
        self.assertEqual(n, len(meshes))
        self.assertEqual(n, 10)
        for p in meshes:
            self.assertTrue(UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0], p.GetPath())

    def test_mdl_texture_parsing(self):
        text = open(PAVERS_MDL).read() if os.path.exists(PAVERS_MDL) else \
            'diffuse_texture: texture_2d("./Brick_Pavers/Brick_Pavers_BaseColor.png", ::tex::gamma_srgb),\n' \
            'ORM_texture: texture_2d("./Brick_Pavers/Brick_Pavers_ORM.png", ::tex::gamma_linear),\n' \
            'normalmap_texture: texture_2d("./Brick_Pavers/Brick_Pavers_N.png", ::tex::gamma_linear),\n' \
            'reflectionroughness_texture: texture_2d(),'
        t = assets.mdl_textures(text)
        self.assertEqual(t, {"albedo": "./Brick_Pavers/Brick_Pavers_BaseColor.png",
                             "orm": "./Brick_Pavers/Brick_Pavers_ORM.png",
                             "normal": "./Brick_Pavers/Brick_Pavers_N.png"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
