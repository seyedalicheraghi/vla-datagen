"""Offline validation of the USD scene the Isaac runner renders (needs usd-core + numpy; no Isaac Sim).

    pip install usd-core
    python scripts/heavy_machines/test/test_usd_scene.py
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import numpy as np
    from pxr import Usd, UsdGeom, UsdShade
    HAVE_DEPS = True
except ImportError:  # pragma: no cover
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "needs usd-core and numpy")
class TestUsdScene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from heavy_machines import kinematics as K
        from heavy_machines.scenarios import adt_approach_hex
        from heavy_machines.scenarios.usd_scene import SceneOptions, build_scene, pose_machines

        cls.K, cls.pose = K, staticmethod(pose_machines)
        sc = adt_approach_hex.build()
        # same scene, smaller areas and textures so the test runs in about a minute
        cls.sc = dataclasses.replace(sc, local_rect=(-25.0, 25.0, -12.0, 16.0), near_rect=(-12.0, 12.0, -6.0, 9.0),
                                     far_rect=(-240.0, 240.0, -240.0, 240.0))
        cls.tmp = tempfile.TemporaryDirectory()
        cls.stage = Usd.Stage.CreateInMemory()
        o = SceneOptions(terrain_step=0.5, texture_ppm=4.0, texture_ppm_near=6.0, far_ppm=0.2, boulders=25,
                         gravel_per_m2=0.05, rock_tex_size=64, machine_tex_size=128)
        cls.ops = build_scene(cls.stage, cls.sc, os.path.join(cls.tmp.name, "tex"), o, log=lambda m: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def meshes(self):
        return [UsdGeom.Mesh(p) for p in self.stage.Traverse() if p.IsA(UsdGeom.Mesh)]

    def test_expected_prims(self):
        for path in ("/World/Terrain/Near", "/World/Terrain/Local", "/World/Terrain/Far", "/World/Looks"):
            self.assertTrue(self.stage.GetPrimAtPath(path).IsValid(), path)
        for actor in self.sc.actors:
            for lk in actor.spec.links:
                self.assertTrue(self.stage.GetPrimAtPath(f"/World/{actor.name}/{lk.name}").IsValid())
        inst = [p for p in self.stage.Traverse() if p.IsA(UsdGeom.PointInstancer)]
        self.assertGreaterEqual(len(inst), 1)

    def test_meshes_well_formed_and_bound(self):
        n = 0
        for m in self.meshes():
            pts = m.GetPointsAttr().Get()
            idx = np.array(m.GetFaceVertexIndicesAttr().Get())
            cnt = np.array(m.GetFaceVertexCountsAttr().Get())
            self.assertTrue((cnt == 3).all())
            self.assertEqual(len(idx), 3 * len(cnt))
            self.assertTrue(idx.min() >= 0 and idx.max() < len(pts), m.GetPath())
            self.assertEqual(len(m.GetNormalsAttr().Get()), len(pts))
            st = UsdGeom.PrimvarsAPI(m).GetPrimvar("st")
            self.assertTrue(st and len(st.Get()) == len(pts), m.GetPath())
            mat, _ = UsdShade.MaterialBindingAPI(m.GetPrim()).ComputeBoundMaterial()
            self.assertTrue(mat, f"{m.GetPath()} has no material")
            n += 1
        self.assertGreater(n, 30)

    def test_materials_and_textures(self):
        checked = 0
        for prim in self.stage.Traverse():
            if not prim.IsA(UsdShade.Material):
                continue
            mat = UsdShade.Material(prim)
            src = mat.GetSurfaceOutput().GetConnectedSources()[0]
            self.assertTrue(src, prim.GetPath())
            surf = UsdShade.Shader(src[0].source.GetPrim())
            self.assertEqual(surf.GetIdAttr().Get(), "UsdPreviewSurface")
            for name in ("diffuseColor", "roughness", "normal"):
                inp = surf.GetInput(name)
                if not inp or not inp.HasConnectedSource():
                    continue
                tex = UsdShade.Shader(inp.GetConnectedSources()[0][0].source.GetPrim())
                self.assertEqual(tex.GetIdAttr().Get(), "UsdUVTexture")
                path = tex.GetInput("file").Get().path
                self.assertTrue(os.path.isfile(path), path)
                cs = tex.GetInput("sourceColorSpace").Get()
                self.assertEqual(cs, "sRGB" if name == "diffuseColor" else "raw", f"{prim.GetPath()} {name}")
                if name == "normal":
                    self.assertEqual(tuple(tex.GetInput("scale").Get()), (2.0, 2.0, 2.0, 1.0))
                    self.assertEqual(tuple(tex.GetInput("bias").Get()), (-1.0, -1.0, -1.0, 0.0))
                st_src = tex.GetInput("st").GetConnectedSources()[0][0].source
                self.assertEqual(UsdShade.Shader(st_src.GetPrim()).GetIdAttr().Get(), "UsdPrimvarReader_float2")
                checked += 1
        self.assertGreater(checked, 20)

    def test_instancers_valid(self):
        for prim in self.stage.Traverse():
            if not prim.IsA(UsdGeom.PointInstancer):
                continue
            pi = UsdGeom.PointInstancer(prim)
            protos = pi.GetPrototypesRel().GetTargets()
            idx = np.array(pi.GetProtoIndicesAttr().Get())
            self.assertTrue(len(idx) > 0 and idx.max() < len(protos))
            self.assertEqual(len(pi.GetPositionsAttr().Get()), len(idx))
            self.assertEqual(len(pi.GetScalesAttr().Get()), len(idx))
            self.assertEqual(len(pi.GetOrientationsAttr().Get()), len(idx))
            for t in protos:
                self.assertTrue(self.stage.GetPrimAtPath(t).IsValid())

    def test_link_transforms_match_kinematics(self):
        K = self.K
        cache = UsdGeom.XformCache()
        for t in (0.0, 4.0, self.sc.duration):
            self.pose(self.ops, self.sc, t)
            cache.Clear()
            for actor in self.sc.actors:
                base, q = actor.root_tf(t)
                tfs = K.link_transforms(actor.spec, q)
                for lk in actor.spec.links:
                    want = base @ tfs[lk.name]
                    M = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(f"/World/{actor.name}/{lk.name}"))
                    p = (0.3, -0.2, 0.5)
                    got = M.Transform(tuple(float(v) for v in p))
                    exp = want.apply(p)
                    for a, b in zip(got, exp):
                        self.assertAlmostEqual(a, b, places=4, msg=f"{actor.name}/{lk.name} at t={t}")

    def test_round_trip_usda(self):
        text = self.stage.GetRootLayer().ExportToString()
        self.assertIn("UsdPreviewSurface", text)
        s2 = Usd.Stage.CreateInMemory()
        self.assertTrue(s2.GetRootLayer().ImportFromString(text))
        self.assertTrue(s2.GetPrimAtPath("/World/Terrain/Near").IsValid())


if __name__ == "__main__":
    unittest.main(verbosity=2)
