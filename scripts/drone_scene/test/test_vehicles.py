"""Checks of the drone choices, the LMF2 materials and the camera mounts (numpy + pxr; no Isaac Sim).

    python scripts/drone_scene/test/test_vehicles.py

The tests on the real NVIDIA assets run when they are in the local cache (``prepare.py --mirror`` or
``nucleus.mirror_usd``); otherwise they are skipped.
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from drone_scene import flight, lucid, scene, vehicles  # noqa: E402

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
except ImportError:              # pragma: no cover
    Usd = None

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "drone_scene")


def box_points(x0, x1, y0, y1, z0, z1, n=12):
    g = np.stack(np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n), np.linspace(z0, z1, n)), -1)
    return g.reshape(-1, 3)


class TestMounts(unittest.TestCase):
    def setUp(self):
        self.rig = lucid.make_rig("TRI032S-CC")
        # a 20 x 16 x 10 cm body, plus an arm sticking out sideways at the mount height (outside the camera)
        self.pts = np.concatenate([box_points(-0.10, 0.10, -0.08, 0.08, 0.0, 0.10),
                                   box_points(0.0, 0.30, 0.10, 0.12, 0.04, 0.06)])

    def test_cameras_clear_the_body(self):
        m = vehicles.mount_cameras(self.pts, self.rig, 0.05, pitch_down_deg=0.0, clearance_m=0.016)
        length = self.rig.length_mm() * 1e-3
        self.assertAlmostEqual(m["front"].surface_x, 0.10, places=6)          # the arm is outside the band
        self.assertAlmostEqual(m["front"].position[0], 0.10 + 0.016 + length, places=6)
        self.assertAlmostEqual(m["back"].position[0], -(0.10 + 0.016 + length), places=6)
        self.assertEqual(m["front"].position[1:], (0.0, 0.05))

    def test_tilt_lowers_the_lens(self):
        m = vehicles.mount_cameras(self.pts, self.rig, 0.05, pitch_down_deg=10.0)
        length = self.rig.length_mm() * 1e-3
        self.assertAlmostEqual(m["front"].position[2], 0.05 - length * math.sin(math.radians(10)), places=6)

    def test_no_points(self):
        m = vehicles.mount_cameras(np.zeros((0, 3)), self.rig, 0.0, pitch_down_deg=0.0)
        self.assertGreater(m["front"].position[0], 0.0)

    def test_onboard_camera_pose_matches_mount(self):
        m = vehicles.mount_cameras(self.pts, self.rig, 0.05)
        cams = {n: flight.OnboardCamera(n, mt.position, mt.look) for n, mt in m.items()}
        f = flight.Flight(cameras=cams)
        pos, _ = f.camera_local("back")
        np.testing.assert_allclose(pos, m["back"].position)
        M = f.camera_matrix(0.0, "front")
        p, R = f.pose(0.0)
        np.testing.assert_allclose(M[3, :3], p + R @ np.asarray(m["front"].position), atol=1e-12)

    def test_payload(self):
        self.assertIn("OK", vehicles.payload_report(vehicles.VEHICLES["lmf2"], self.rig))
        self.assertIn("TOO HEAVY", vehicles.payload_report(vehicles.VEHICLES["crazyflie"], self.rig))


class TestFraming(unittest.TestCase):
    def test_frame_scene(self):
        cfg = scene.SceneCfg()
        big = vehicles.frame_scene(cfg, vehicles.VEHICLES["lmf2"])
        for name in ("hero", "garden", "demo"):
            a, b = cfg.camera(name), big.camera(name)
            d0 = np.linalg.norm(np.subtract(a.eye, a.target))
            d1 = np.linalg.norm(np.subtract(b.eye, b.target))
            self.assertGreater(d1, 2.0 * d0)
            self.assertEqual(a.target, b.target)
            self.assertGreaterEqual(b.eye[2], 0.2)
        self.assertEqual(cfg.camera("top"), big.camera("top"))
        self.assertIs(vehicles.frame_scene(cfg, vehicles.VEHICLES["crazyflie"]), cfg)

    def test_subset_stem(self):
        self.assertEqual(vehicles.subset_stem("blue_anodized_aluminum3"), "blue_anodized_aluminum")
        self.assertEqual(vehicles.subset_stem("clear_glass_1"), "clear_glass")
        self.assertEqual(vehicles.subset_stem("matte_aluminum"), "matte_aluminum")


@unittest.skipIf(Usd is None, "pxr (usd-core) not installed")
class TestMaterials(unittest.TestCase):
    def make_stage(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World/Drone/base_link")
        mesh = UsdGeom.Mesh.Define(stage, "/World/Drone/base_link/visuals/mesh")
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)] * 2)
        mesh.CreateFaceVertexCountsAttr([4, 4, 4, 4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3] * 4)

        def omnipbr(name, colour):
            mat = UsdShade.Material.Define(stage, f"/World/Drone/Looks/{name}")
            sh = UsdShade.Shader.Define(stage, f"/World/Drone/Looks/{name}/{name}")
            sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
            return mat

        black, green, red = omnipbr("Material_0", (0, 0, 0)), omnipbr("green", (0.1, 0.2, 0.1)), omnipbr("c16", (1, .3, .3))
        for i, (name, mat) in enumerate((("matte_aluminum3", black), ("green_low_gloss_plastic_2", green),
                                         ("color_16", red), ("Material_01", black))):
            s = UsdGeom.Subset.CreateGeomSubset(mesh, name, UsdGeom.Tokens.face, [i], "materialBind")
            UsdShade.MaterialBindingAPI.Apply(s.GetPrim()).Bind(mat)
        return stage

    def bound(self, stage, name):
        p = stage.GetPrimAtPath(f"/World/Drone/base_link/visuals/mesh/{name}")
        return UsdShade.MaterialBindingAPI(p).GetDirectBinding().GetMaterialPath().name

    def test_rebinding(self):
        stage = self.make_stage()
        faces = vehicles.apply_lmf2_materials(stage, "/World/Drone")
        self.assertEqual(self.bound(stage, "matte_aluminum3"), "matte_aluminum")
        self.assertTrue(self.bound(stage, "green_low_gloss_plastic_2").startswith("green_low_gloss_plastic_"))
        self.assertEqual(self.bound(stage, "color_16"), "c16")              # a real CAD colour is kept
        self.assertEqual(self.bound(stage, "Material_01"), "black_unnamed")
        self.assertEqual(sum(faces.values()), 3)
        pbr = UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/LMF2/matte_aluminum/PBR"))
        self.assertEqual(pbr.GetInput("metallic").Get(), 1.0)

    def test_glass_is_translucent(self):
        stage = Usd.Stage.CreateInMemory()
        mesh = UsdGeom.Mesh.Define(stage, "/D/visuals/m")
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        UsdGeom.Subset.CreateGeomSubset(mesh, "clear_glass_1", UsdGeom.Tokens.face, [0], "materialBind")
        vehicles.apply_lmf2_materials(stage, "/D")
        pbr = UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/LMF2/clear_glass/PBR"))
        self.assertLess(pbr.GetInput("opacity").Get(), 0.5)

    def test_body_points_in_body_frame(self):
        stage = Usd.Stage.CreateInMemory()
        body = UsdGeom.Xform.Define(stage, "/D/body")
        body.AddTranslateOp().Set(Gf.Vec3d(5, 0, 1))
        body.AddRotateZOp().Set(90.0)
        m = UsdGeom.Mesh.Define(stage, "/D/body/part")
        m.AddTranslateOp().Set(Gf.Vec3d(0.1, 0, 0))
        m.CreatePointsAttr([(0, 0, 0), (0.01, 0, 0)])
        pts = vehicles.body_points(stage, "/D/body")
        np.testing.assert_allclose(pts, [[0.1, 0, 0], [0.11, 0, 0]], atol=1e-9)


@unittest.skipIf(Usd is None or not os.path.exists(os.path.join(CACHE, vehicles.VEHICLES["lmf2"].usd)),
                 "LMF2 asset not cached")
class TestRealLMF2(unittest.TestCase):
    def test_asset(self):
        v = vehicles.VEHICLES["lmf2"]
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        x = UsdGeom.Xform.Define(stage, v.prim)
        x.GetPrim().GetReferences().AddReference(os.path.join(CACHE, v.usd))
        self.assertTrue(stage.GetPrimAtPath(f"{v.prim}/{v.body}").IsValid())
        faces = vehicles.apply_lmf2_materials(stage, v.prim)
        # the black-bound aluminium, anodised and glass groups are the bulk of the drone
        self.assertGreater(faces.get("matte_aluminum", 0) + faces.get("blue_anodized_aluminum", 0), 150000)
        pts = vehicles.body_points(stage, f"{v.prim}/{v.body}")
        self.assertGreater(len(pts), 100000)
        self.assertAlmostEqual(float(pts[:, 0].max() - pts[:, 0].min()), 0.515, delta=0.01)
        m = vehicles.mount_cameras(pts, lucid.make_rig(), v.camera_z)
        self.assertGreater(m["front"].position[0], 0.15)
        self.assertLess(m["back"].position[0], -0.15)


if __name__ == "__main__":
    unittest.main()
