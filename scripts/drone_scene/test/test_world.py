"""Checks of the large-map builder's geometry and logic on synthetic data (numpy, shapely, pxr; no network,
no Isaac Sim).

    python scripts/drone_scene/test/test_world.py
"""

from __future__ import annotations

import datetime as dt
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

try:
    import shapely  # noqa: F401
    from shapely import affinity
    from shapely.geometry import LineString, Polygon, box
    from pxr import Usd, UsdGeom
    DEPS = True
except ImportError:              # pragma: no cover
    DEPS = False

from drone_scene.world import geo  # noqa: E402
from drone_scene.world import landcover as lc  # noqa: E402
from drone_scene.world import terrain as T  # noqa: E402


def normals(mesh, faces=None):
    P = mesh.points[mesh.faces if faces is None else faces]
    return np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])


class TestGeo(unittest.TestCase):
    def test_frame_roundtrip(self):
        f = geo.Frame(42.025, -93.69)
        x, y = f.to_local([-93.69, -93.66], [42.025, 42.04])
        self.assertAlmostEqual(float(x[0]), 0.0, places=6)
        self.assertAlmostEqual(float(y[0]), 0.0, places=6)
        lon, lat = f.to_lonlat(x, y)
        np.testing.assert_allclose(lon, [-93.69, -93.66], atol=1e-9)
        np.testing.assert_allclose(lat, [42.025, 42.04], atol=1e-9)
        self.assertAlmostEqual(float(x[1]), 0.03 * 111320 * math.cos(math.radians(42.03)), delta=30)

    def test_grid_sampling(self):
        g = geo.Grid.empty((0, 100, 0, 50), 10.0)
        X, Y = g.centres()
        g.data = (3 * X + 2 * Y).astype(np.float32)
        xs, ys = np.array([5.0, 33.0, 77.7]), np.array([45.0, 12.5, 30.0])
        np.testing.assert_allclose(g.sample(xs, ys), 3 * xs + 2 * ys, atol=1e-4)
        self.assertEqual(g.rect, (0.0, 100.0, 0.0, 50.0))

    def test_sun(self):
        # NOAA solar calculator: Boulder, CO, 2024-06-21 12:00 MDT -> elevation 68.7, azimuth 136.5
        el, az = geo.sun_position(40.0150, -105.2705, dt.datetime(2024, 6, 21, 18, 0))
        self.assertAlmostEqual(el, 68.75, delta=0.3)
        self.assertAlmostEqual(az, 136.5, delta=0.6)
        el, _ = geo.sun_position(40.0150, -105.2705, dt.datetime(2024, 6, 21, 8, 0))       # 2 am MDT
        self.assertLess(el, -10)
        self.assertAlmostEqual(geo.compass_to_ccw_from_east(90.0), 0.0)
        self.assertAlmostEqual(geo.compass_to_ccw_from_east(0.0), 90.0)


class TestTerrain(unittest.TestCase):
    def test_tile_orientation_and_skirt(self):
        m = T.tile_mesh(lambda x, y: 2 * np.sin(np.asarray(x) / 40), (0, 120, 0, 80), 8.0)
        n = normals(m)
        ng = m.meta["n_ground_faces"]
        self.assertTrue((n[:ng, 2] > 0).all())
        c = m.points[m.faces[ng:]].mean(1)[:, :2] - (60, 40)
        self.assertTrue(((n[ng:, :2] * c).sum(1) > 0).all(), "skirts face outwards")
        ground_min = m.points[:m.faces[:ng].max() + 1, 2].min()
        self.assertAlmostEqual(m.points[:, 2].min(), ground_min - 2.0, delta=1e-9)

    def test_pads(self):
        x = np.array([0.0, 10.0, 20.0, 40.0])
        z = T.apply_pads(x, np.zeros(4), np.full(4, 5.0), [T.Pad(0, 0, 1.0, 12, 30)])
        self.assertEqual(z[0], 1.0)
        self.assertEqual(z[1], 1.0)
        self.assertTrue(1.0 < z[2] < 5.0)
        self.assertEqual(z[3], 5.0)

    def test_lod_and_tiles(self):
        lod = T.Lod()
        self.assertEqual(lod.cell(0), 1.0)
        self.assertEqual(lod.cell(5000), lod.far)
        ts = T.tiles((0, 600, 0, 300), 250)
        self.assertEqual(len(ts), 6)
        self.assertEqual(ts[-1], (500.0, 600.0, 250.0, 300.0))
        self.assertEqual(T.rect_segment_distance((0, 10, 0, 10), (-5, 5), (30, 5)), 0.0)

    def test_uvs(self):
        m = T.tile_mesh(lambda x, y: np.zeros_like(np.asarray(x, float)), (0, 10, 0, 10), 5.0, skirt_m=0)
        n = m.n_faces
        uv = T.face_uvs(m, np.full(n, 2.0), np.zeros(n), np.zeros((n, 2)))
        P = m.points[m.faces]
        np.testing.assert_allclose(uv[..., 0], P[..., 0] / 2.0)
        uv90 = T.face_uvs(m, np.full(n, 1.0), np.full(n, math.pi / 2), np.zeros((n, 2)))
        np.testing.assert_allclose(uv90[..., 0], P[..., 1], atol=1e-9)

    def test_patches(self):
        rng = np.random.default_rng(0)
        x, y = rng.uniform(0, 500, 20000), rng.uniform(0, 500, 20000)
        ids = T.patch_ids(x, y, 25.0)
        self.assertTrue(250 < len(np.unique(ids)) < 600)
        np.testing.assert_array_equal(ids, T.patch_ids(x, y, 25.0))


class TestHorizon(unittest.TestCase):
    def test_ring_surrounds_the_far_zone(self):
        from drone_scene.world import build
        far, ring = (-6000.0, 7000.0, -6000.0, 7000.0), (-24000.0, 26000.0, -24000.0, 26000.0)
        h = geo.Grid(np.zeros((200, 200), np.float32), ring[0], ring[3], 250.0)
        m, mats = build.horizon_ring(h, far, ring, (0.05, 0.07, 0.04))
        cen = T.face_centroids(m)
        top = np.abs(normals(m)[:, 2]) > 1e-9                      # ground faces, not skirts
        inside = (cen[:, 0] > far[0]) & (cen[:, 0] < far[1]) & (cen[:, 1] > far[2]) & (cen[:, 1] < far[3])
        self.assertFalse((inside & top).any(), "no ring ground inside the far zone")
        self.assertAlmostEqual(m.points[:, 0].min(), ring[0])
        self.assertAlmostEqual(m.points[:, 1].max(), ring[3])
        self.assertEqual(len(mats), 1)


class TestLandcover(unittest.TestCase):
    def test_worldcover_mapping(self):
        g = geo.Grid(np.array([[10, 30], [40, 80]], np.uint8), 0.0, 20.0, 10.0)
        c = lc.from_worldcover(g)
        np.testing.assert_array_equal(c.data, [[lc.FOREST, lc.MEADOW], [lc.CROP, lc.WATER]])

    def test_upsample_is_smooth(self):
        d = np.full((10, 10), lc.CROP, np.uint8)
        d[:, 5:] = lc.FOREST
        g = geo.Grid(d, 0.0, 100.0, 10.0)
        up = lc.upsample_classes(g, (0, 100, 0, 100), 2.0)
        self.assertEqual(up.shape, (50, 50))
        self.assertEqual(int(up.data[25, 2]), lc.CROP)
        self.assertEqual(int(up.data[25, 47]), lc.FOREST)
        col = np.argmax(up.data == lc.FOREST, axis=1)
        self.assertTrue((np.abs(col - 25) <= 4).all(), "boundary near x = 50 m")
        self.assertGreater(len(np.unique(col)), 1, "boundary is not a straight pixel edge")

    @unittest.skipUnless(DEPS, "shapely/pxr missing")
    def test_burn(self):
        g = geo.Grid(np.full((20, 20), lc.LAWN, np.uint8), 0.0, 40.0, 2.0)
        n = lc.burn(g, [box(10, 10, 20, 20)], lc.WATER)
        self.assertEqual(n, 25)
        self.assertEqual(int(g.data[12, 7]), lc.WATER)       # x 15, y 15


@unittest.skipUnless(DEPS, "shapely/pxr missing")
class TestBuildings(unittest.TestCase):
    def test_house_and_shop(self):
        from drone_scene.world import buildings as B
        g = lambda x, y: 0.01 * np.asarray(x, float)               # noqa: E731
        house = affinity.rotate(box(0, 0, 12, 8), 25, origin=(0, 0))
        m, info = B.building_mesh(B.Footprint(house, cls="house"), g)
        n = normals(m)
        walls = np.abs(n[:, 2]) < 1e-9
        c = m.points[m.faces].mean(1)[:, :2] - np.array(house.centroid.coords[0])
        self.assertTrue(((n[walls][:, :2] * c[walls]).sum(1) > 0).all(), "walls face out")
        self.assertTrue((n[~walls][:, 2] > 0).all(), "roofs face up")
        self.assertFalse(info["flat"])
        self.assertGreater(info["top"], info["eave"])
        shop = B.Footprint(box(50, 0, 90, 30), cls="commercial", height=8.0)
        m2, info2 = B.building_mesh(shop, g)
        self.assertTrue(info2["flat"])
        self.assertAlmostEqual(info2["eave"], 8.0, delta=0.01)
        self.assertEqual(m2.material.max() >= len(B.WALL_STYLES), True)

    def test_L_shaped_house_gets_two_roofs(self):
        from drone_scene.world import buildings as B
        L = Polygon([(0, 0), (14, 0), (14, 6), (6, 6), (6, 14), (0, 14)])
        _, rects = B.cover_rectangles(L)
        self.assertEqual(len(rects), 2)
        area = sum((r[1] - r[0]) * (r[3] - r[2]) for r in rects)
        self.assertGreater(area, 0.9 * L.area)

    def test_heights_from_floors(self):
        from drone_scene.world import buildings as B
        _, info = B.building_mesh(B.Footprint(box(0, 0, 20, 20), floors=4, roof_shape="flat"), lambda x, y: 0 * np.asarray(x))
        self.assertAlmostEqual(info["eave"], 12.0, delta=0.01)


@unittest.skipUnless(DEPS, "shapely/pxr missing")
class TestRoads(unittest.TestCase):
    def test_ribbons(self):
        from drone_scene.world import roads as R
        g = lambda x, y: np.zeros_like(np.asarray(x, float))         # noqa: E731
        m, st = R.roads_mesh([R.Road(LineString([(0, 0), (100, 0)]), "residential"),
                              R.Road(LineString([(50, -50), (50, 50)]), "primary"),
                              R.Road(LineString([(0, 20), (30, 40)]), "track", "unpaved")], g)
        self.assertEqual(st["segments"], 3)
        self.assertTrue((normals(m)[:, 2] > 0).all())
        res = m.points[m.faces][m.material == R.SURFACES.index("asphalt")]
        self.assertAlmostEqual(res[..., 1].max(), 50.0, delta=0.01)        # primary road runs to y = 50
        self.assertTrue((m.material == R.SURFACES.index("gravel")).any())
        self.assertTrue((m.material == R.SURFACES.index("marking_yellow")).any(), "centre line on the primary")
        primary = m.points[m.faces][m.material == R.SURFACES.index("asphalt")]
        self.assertAlmostEqual(primary[..., 0].max() - primary[..., 0].min(), 100.0, delta=0.5)


@unittest.skipUnless(DEPS, "shapely/pxr missing")
class TestVegetation(unittest.TestCase):
    def test_detect(self):
        from drone_scene.world import vegetation as V
        g = geo.Grid.empty((0, 100, 0, 100), 1.0)
        X, Y = g.centres()
        truth = [(20, 20, 18, 5), (50, 50, 8, 3), (80, 30, 25, 7), (30, 80, 12, 4), (70, 75, 5, 2), (26, 20, 15, 4)]
        h = np.zeros_like(X)
        for x, y, ht, r in truth:
            h = np.maximum(h, ht * np.clip(1 - ((X - x) ** 2 + (Y - y) ** 2) / (r * r * 1.5), 0, 1))
        g.data = h.astype(np.float32)
        t = V.detect_trees(g)
        self.assertEqual(len(t), len(truth))
        for x, y, ht, _ in truth:
            d = np.hypot(t[:, 0] - x, t[:, 1] - y)
            self.assertLess(d.min(), 1.0)
            self.assertAlmostEqual(t[np.argmin(d), 2], ht, delta=0.6)
        kept = V.drop_trees(t, buildings=[box(45, 45, 55, 55)], roads=[(LineString([(0, 30), (100, 30)]), 7.0)],
                            pads=[(20, 20, 3)])
        self.assertEqual(len(kept), 3)
        sp = V.assign_species(t)
        inst = V.tree_instances(t, sp, {s.key: 20.0 for s in V.SPECIES}, lambda x, y: np.zeros_like(np.asarray(x, float)))
        self.assertEqual(sum(len(v["positions"]) for v in inst.values()), len(t))
        for k, v in inst.items():
            sp_ = next(s for s in V.SPECIES if s.key == k)
            np.testing.assert_allclose(v["scales"][:, 2] / sp_.units * 20.0, v["heights"], rtol=1e-6)


@unittest.skipUnless(DEPS, "shapely/pxr missing")
class TestSite(unittest.TestCase):
    def test_backyard_and_meadow(self):
        from drone_scene.world import site as S
        cls = geo.Grid(np.full((300, 300), lc.LAWN, np.uint8), -1500.0, 1500.0, 10.0)
        X, Y = cls.centres()
        cls.data[X > 600] = lc.MEADOW                    # open grass east of x = 600 m
        houses = [box(-30, 20, -18, 30), box(10, 20, 22, 30)]
        roads = [(LineString([(-1500, 0), (1500, 0)]), 8.0)]                  # street along y = 0, houses north of it
        canopy = geo.Grid(np.zeros((300, 300), np.float32), -1500.0, 1500.0, 10.0)
        st = S.choose(houses, houses, roads, [], cls, canopy, distance=1000.0)
        self.assertGreater(st.takeoff[1], 30.0, "backyard: on the far side of the house from the street")
        self.assertAlmostEqual(st.distance, 1000.0, delta=60)
        self.assertEqual(int(cls.sample(np.array([st.landing[0]]), np.array([st.landing[1]]), "nearest")[0]), lc.MEADOW)
        self.assertGreater(abs(st.landing[1]), 35.0, "away from the road")


@unittest.skipUnless(DEPS, "shapely/pxr missing")
class TestUsd(unittest.TestCase):
    def test_write_mesh_with_materials(self):
        from drone_scene.world import usdio
        m = T.tile_mesh(lambda x, y: np.zeros_like(np.asarray(x, float)), (0, 50, 0, 50), 5.0)
        cen = T.face_centroids(m)
        m.material = (cen[:, 0] > 25).astype(np.int32)
        m.uv = T.face_uvs(m, np.ones(m.n_faces), np.zeros(m.n_faces), np.zeros((m.n_faces, 2)))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.usdc")
            st = Usd.Stage.CreateNew(path)
            mesh = usdio.write_mesh(st, "/Map/T", m, [usdio.MaterialSpec("a", color=(0.1, 0.2, 0.1)),
                                                      usdio.MaterialSpec("b", color=(0.3, 0.2, 0.1), opacity=0.5)])
            st.GetRootLayer().Save()
            st2 = Usd.Stage.Open(path)
            m2 = UsdGeom.Mesh(st2.GetPrimAtPath("/Map/T"))
            self.assertEqual(len(m2.GetFaceVertexCountsAttr().Get()), m.n_faces)
            subs = UsdGeom.Subset.GetAllGeomSubsets(m2)
            self.assertEqual(sorted(s.GetPrim().GetName() for s in subs), ["a", "b"])
            self.assertTrue(UsdGeom.Subset.ValidateFamily(m2, UsdGeom.Tokens.face, "materialBind")[0])
            self.assertEqual(len(UsdGeom.PrimvarsAPI(m2).GetPrimvar("st").Get()), 3 * m.n_faces)
            del mesh


if __name__ == "__main__":
    unittest.main()
