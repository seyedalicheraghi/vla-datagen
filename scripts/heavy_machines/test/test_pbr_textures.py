"""Offline checks of the procedural PBR texture sets (numpy only; no Isaac Sim).

    python scripts/heavy_machines/test/test_pbr_textures.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import numpy as np
    HAVE_NP = True
except ImportError:  # pragma: no cover
    HAVE_NP = False


def _seam_ratios(a: np.ndarray) -> tuple[float, float]:
    """(wrap-edge difference) / (mean neighbour difference) along columns and rows."""
    a = a.astype(np.float64)
    if a.ndim == 2:
        a = a[..., None]
    interior_x = np.abs(a[:, 1:] - a[:, :-1]).mean() + 1e-9
    interior_y = np.abs(a[1:] - a[:-1]).mean() + 1e-9
    seam_x = np.abs(a[:, 0] - a[:, -1]).mean()
    seam_y = np.abs(a[0] - a[-1]).mean()
    return seam_x / interior_x, seam_y / interior_y


def _decode(n8: np.ndarray) -> np.ndarray:
    return n8.astype(np.float64) / 255.0 * 2.0 - 1.0


@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestMachineTextures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from heavy_machines import spec
        from heavy_machines.scenarios import pbr_textures as P

        cls.P = P
        cls.spec = spec
        cls.size = 256
        cls.sets = {name: P.machine_textures(name, rgba[:3], seed=3, size=cls.size)
                    for name, rgba in spec.DEFAULT_MATERIALS.items()}

    def test_dtype_shape_fields(self):
        n = self.size
        for name, ts in self.sets.items():
            with self.subTest(name):
                self.assertEqual(ts.albedo.shape, (n, n, 3))
                self.assertEqual(ts.albedo.dtype, np.uint8)
                self.assertEqual(ts.roughness.shape, (n, n))
                self.assertEqual(ts.roughness.dtype, np.uint8)
                self.assertEqual(ts.normal.shape, (n, n, 3))
                self.assertEqual(ts.normal.dtype, np.uint8)
                self.assertIsInstance(ts.metallic, float)
                self.assertTrue(1.0 <= ts.metres_per_tile <= 4.0, ts.metres_per_tile)
                self.assertTrue(0.0 <= ts.opacity <= 1.0)
                self.assertTrue(0.0 <= ts.clearcoat <= 1.0)

    def test_default_size_is_1024(self):
        ts = self.P.machine_textures("steel", (0.5, 0.5, 0.52))
        self.assertEqual(ts.albedo.shape, (1024, 1024, 3))

    def test_material_properties(self):
        S = self.sets
        for name, ts in S.items():
            self.assertEqual(ts.metallic, 1.0 if name == "steel" else 0.0, name)
        self.assertTrue(0.35 <= S["glass"].opacity <= 0.5)
        for name in ("body", "accent", "dark", "steel", "rubber", "hub", "red", "orange", "white", "rust"):
            self.assertEqual(S[name].opacity, 1.0, name)
        rough = {k: v.roughness.mean() / 255.0 for k, v in S.items()}
        self.assertTrue(0.35 <= rough["body"] <= 0.60, rough["body"])
        self.assertTrue(0.30 <= rough["steel"] <= 0.52, rough["steel"])
        self.assertTrue(0.78 <= rough["rubber"] <= 0.95, rough["rubber"])
        self.assertLess(rough["glass"], 0.2)
        self.assertGreater(rough["rust"], 0.75)
        # rubber is near black, glass dark, body is the saturated yellow, white is light
        lum = {k: v.albedo.astype(float).mean() for k, v in S.items()}
        self.assertLess(lum["rubber"], 90)
        self.assertGreater(lum["white"], 150)
        body = S["body"].albedo.reshape(-1, 3).astype(float).mean(0)
        self.assertTrue(body[0] > body[1] > body[2] + 60, body)         # yellow: R > G >> B

    def test_dust_lightens_and_roughens_paint(self):
        """Dusty pixels of the body paint are lighter / less saturated and rougher than clean ones."""
        ts = self.sets["body"]
        rough = ts.roughness.astype(float) / 255.0
        a = ts.albedo.astype(float)
        sat = (a.max(-1) - a.min(-1)) / (a.max(-1) + 1.0)
        lo, hi = rough < np.percentile(rough, 20), rough > np.percentile(rough, 80)
        self.assertLess(sat[hi].mean(), sat[lo].mean())

    def test_flat_panel_paint_tiles_are_large(self):
        """Big tiles (>= 3 m) keep a 6 m body panel from showing the same marks twice in a row."""
        for name in ("body", "red", "orange", "white", "hub"):
            self.assertGreaterEqual(self.sets[name].metres_per_tile, 3.0, name)

    def test_paint_has_no_near_black_marks_or_landmarks(self):
        """Dirt on paint is brown-grey and made of many small faint marks: no near-black pen-like spots (darkest
        0.05 % of pixels stays above 25 % of the median luminance; the old black primer chips / drips gave 2-13 %)
        and no tile region that holds a cluster of marks the others lack."""
        P = self.P
        for name in ("body", "white", "orange", "red", "hub"):
            ts = P.machine_textures(name, self.spec.DEFAULT_MATERIALS[name][:3], seed=6, size=512)
            lum = P.srgb_to_linear(ts.albedo.astype(np.float32) / 255.0) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
            with self.subTest(name):
                self.assertGreater(np.percentile(lum, 0.05) / np.median(lum), 0.25)
                hp = lum - P.blur(lum, 8.0)
                dark = P.blur(np.clip(-hp / lum.mean(), 0, None) ** 2, 12.0)
                cell = dark.reshape(4, 128, 4, 128).mean((1, 3))
                self.assertLess(cell.max() / cell.mean(), 1.5, cell.max() / cell.mean())

    def test_dark_materials_are_dark_and_neutral(self):
        """Black trim and tyre rubber read as black plastic / rubber, not as brown stone: dark, nearly neutral,
        without large soft blotches."""
        P = self.P
        spec_ranges = {"rubber": (25, 56), "accent": (32, 60), "dark": (40, 66)}
        for name, (lo, hi) in spec_ranges.items():
            ts = P.machine_textures(name, self.spec.DEFAULT_MATERIALS[name][:3], seed=2, size=512)
            a = ts.albedo.astype(np.float32)
            mean = a.reshape(-1, 3).mean(0)
            with self.subTest(name):
                self.assertTrue(lo <= mean.mean() <= hi, mean)
                self.assertLess(mean.max() - mean.min(), 8.0, mean)           # neutral grey, not brown
                blot = P.blur(a.mean(-1), 12.0).std()
                self.assertLess(blot, 3.0, blot)

    def test_tileable(self):
        for name, ts in self.sets.items():
            for label, arr in (("albedo", ts.albedo), ("roughness", ts.roughness), ("normal", ts.normal)):
                rx, ry = _seam_ratios(arr)
                with self.subTest(name=name, map=label):
                    self.assertLess(rx, 1.35, (rx, ry))
                    self.assertLess(ry, 1.35, (rx, ry))

    def test_tileable_large_features(self):
        """Seams must also vanish at the default 1024 px size (large blobs, scratches, streaks wrap)."""
        for name in ("body", "rubber", "steel"):
            ts = self.P.machine_textures(name, self.spec.DEFAULT_MATERIALS[name][:3], seed=5, size=1024)
            for arr in (ts.albedo, ts.roughness, ts.normal):
                rx, ry = _seam_ratios(arr)
                self.assertLess(max(rx, ry), 1.35, name)

    def test_normals_unit_and_upward(self):
        for name, ts in self.sets.items():
            v = _decode(ts.normal)
            norm = np.linalg.norm(v, axis=-1)
            with self.subTest(name):
                self.assertTrue(np.all(ts.normal[..., 2] > 128))
                self.assertLess(abs(norm - 1.0).max(), 0.03)
                # a micro-relief map is mostly near-flat but not perfectly flat
                self.assertGreater(np.median(v[..., 2]), 0.85)
                self.assertGreater(v[..., :2].std(), 0.002)

    def test_flat_height_gives_flat_normal(self):
        n = self.P.height_to_normal(np.zeros((64, 64), np.float32), 1e-3)
        self.assertTrue(np.all(n[..., 0] == 128))
        self.assertTrue(np.all(n[..., 1] == 128))
        self.assertTrue(np.all(n[..., 2] == 255))
        n = self.P.height_to_normal(np.full((64, 64), 0.37, np.float32), 1e-3)
        self.assertTrue(np.all(n == np.array([128, 128, 255], np.uint8)))

    def test_normal_orientation_opengl(self):
        """OpenGL tangent space: +X = image columns, +Y = towards image row 0.  A height that rises
        towards +X leans the normal to -X (R < 128); one that rises towards row 0 leans it to -Y (G < 128)."""
        n = 64
        col = np.arange(n, dtype=np.float32)[None, :].repeat(n, 0)
        nm = self.P.height_to_normal(1e-3 * np.sin(col * 2 * np.pi / n), 1e-2)    # rising at col 0, falling at col n/2
        self.assertLess(nm[10, 0, 0], 128)
        self.assertGreater(nm[10, n // 2, 0], 128)
        self.assertTrue(np.all(nm[..., 1] == 128))
        row = np.arange(n, dtype=np.float32)[:, None].repeat(n, 1)
        nm2 = self.P.height_to_normal(1e-3 * np.sin(-row * 2 * np.pi / n), 1e-2)  # rising towards row 0 at row 0
        self.assertLess(nm2[0, 10, 1], 128)
        self.assertGreater(nm2[n // 2, 10, 1], 128)
        self.assertTrue(np.all(nm2[..., 0] == 128))

    def test_deterministic(self):
        P, rgb = self.P, self.spec.DEFAULT_MATERIALS["body"][:3]
        a = P.machine_textures("body", rgb, seed=11, size=128)
        b = P.machine_textures("body", rgb, seed=11, size=128)
        c = P.machine_textures("body", rgb, seed=12, size=128)
        for k in ("albedo", "roughness", "normal"):
            self.assertTrue(np.array_equal(getattr(a, k), getattr(b, k)), k)
            self.assertFalse(np.array_equal(getattr(a, k), getattr(c, k)), k)

    def test_names_differ_with_same_seed_and_color(self):
        P = self.P
        a = P.machine_textures("red", (0.7, 0.1, 0.1), seed=1, size=128)
        b = P.machine_textures("orange", (0.7, 0.1, 0.1), seed=1, size=128)
        self.assertFalse(np.array_equal(a.albedo, b.albedo))

    def test_unknown_name_derives_from_base_colour(self):
        P = self.P
        ts = P.machine_textures("mystery_green", (0.1, 0.55, 0.15), seed=2, size=128)
        m = ts.albedo.reshape(-1, 3).astype(float).mean(0)
        self.assertTrue(m[1] > m[0] and m[1] > m[2], m)
        self.assertEqual(ts.metallic, 0.0)
        self.assertEqual(ts.opacity, 1.0)
        # also accepts 0..255 colours
        ts2 = P.machine_textures("mystery_green", (26, 140, 38), seed=2, size=128)
        m2 = ts2.albedo.reshape(-1, 3).astype(float).mean(0)
        self.assertTrue(m2[1] > m2[0] and m2[1] > m2[2], m2)

    def test_save_texture_set_roundtrip(self):
        from PIL import Image

        ts = self.sets["steel"]
        with tempfile.TemporaryDirectory() as d:
            paths = self.P.save_texture_set(ts, d, "steel")
            self.assertEqual(set(paths), {"albedo", "roughness", "normal"})
            for key, p in paths.items():
                self.assertTrue(os.path.isabs(p))
                self.assertTrue(os.path.exists(p))
            for key in ("albedo", "roughness", "normal"):
                with Image.open(paths[key]) as im:
                    self.assertTrue(np.array_equal(np.array(im), getattr(ts, key)), key)
            with Image.open(paths["roughness"]) as im:
                self.assertEqual(im.mode, "L")


@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestTerrainDetail(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from heavy_machines.scenarios import adt_approach_hex, pbr_textures as P
        from heavy_machines.scenarios.terrain import bake_albedo

        cls.P = P
        cls.bake = staticmethod(bake_albedo)
        cls.terrain = adt_approach_hex.build().terrain

    def test_shapes_match_bake_albedo(self):
        for rect, ppm in (((0.0, 6.0, -2.0, 3.0), 10.0),
                          ((0.3, 5.7, -1.1, 2.9), 7.5),
                          ((-4.0, 3.5, 3.0, 9.0), 12.0),
                          ((-20.0, 20.0, -12.0, 12.0), 1.0)):
            with self.subTest(rect=rect, ppm=ppm):
                albedo = self.bake(self.terrain, rect, ppm, seed=1)
                normal, rough = self.P.terrain_detail_maps(self.terrain, rect, ppm, seed=1)
                self.assertEqual(normal.shape, albedo.shape)
                self.assertEqual(normal.dtype, np.uint8)
                self.assertEqual(rough.shape, albedo.shape[:2])
                self.assertEqual(rough.dtype, np.uint8)

    def test_normals_unit_up_and_detailed(self):
        rect = (-4.0, 8.0, -6.0, 9.0)
        normal, rough = self.P.terrain_detail_maps(self.terrain, rect, 16.0, seed=2)
        v = _decode(normal)
        self.assertTrue(np.all(normal[..., 2] > 128))
        self.assertLess(abs(np.linalg.norm(v, axis=-1) - 1.0).max(), 0.03)
        self.assertGreater(v[..., :2].std(), 0.03)               # real micro relief
        self.assertGreater(np.median(v[..., 2]), 0.8)
        self.assertGreater(rough.min(), 60)                       # never mirror-like

    def test_coarse_texture_is_nearly_flat(self):
        """At 0.5 px/m the relief is below the pixel size and must not alias into noise."""
        normal, rough = self.P.terrain_detail_maps(self.terrain, (-30.0, 30.0, -20.0, 20.0), 0.5, seed=0)
        self.assertLess(abs(normal[..., 0].astype(int) - 128).mean(), 2.5)
        self.assertLess(abs(normal[..., 1].astype(int) - 128).mean(), 2.5)

    def test_deterministic_and_seed_dependent(self):
        rect, ppm = (0.0, 4.0, -2.0, 2.0), 12.0
        a = self.P.terrain_detail_maps(self.terrain, rect, ppm, seed=4)
        b = self.P.terrain_detail_maps(self.terrain, rect, ppm, seed=4)
        c = self.P.terrain_detail_maps(self.terrain, rect, ppm, seed=5)
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
        self.assertFalse(np.array_equal(a[0], c[0]))

    def test_world_anchored_crop_consistency(self):
        """The same patch of ground looks the same whichever rect it is baked in."""
        ppm = 12.0
        big = self.P.terrain_detail_maps(self.terrain, (-6.0, 10.0, -6.0, 8.0), ppm, seed=1)
        small = self.P.terrain_detail_maps(self.terrain, (0.0, 4.0, -2.0, 2.0), ppm, seed=1)
        # (0, 4, -2, 2) sits at columns 72..120 and rows (8 - 2) * 12 ... (8 + 2) * 12 of the big map
        sub_n = big[0][72:120, 72:120].astype(int)
        diff = abs(sub_n - small[0].astype(int)).mean()
        self.assertLess(diff, 3.0)
        diff_r = abs(big[1][72:120, 72:120].astype(int) - small[1].astype(int)).mean()
        self.assertLess(diff_r, 4.0)

    def test_lane_is_smoother_than_loose_ground(self):
        rect = (-30.0, 30.0, -12.0, 22.0)
        ppm = 4.0
        normal, rough = self.P.terrain_detail_maps(self.terrain, rect, ppm, seed=1)
        W, H = rough.shape[1], rough.shape[0]
        xs = rect[0] + (np.arange(W) + 0.5) / ppm
        ys = rect[3] - (np.arange(H) + 0.5) / ppm
        X, Y = np.meshgrid(xs, ys)
        lane = self.terrain.lane_mask(X, Y)
        lane_r = rough[lane > 0.95].astype(float).mean()
        loose_r = rough[(lane < 0.05) & (Y > 8.0) & (Y < 15.0)].astype(float).mean()      # bench top
        self.assertLess(lane_r, loose_r - 5.0, (lane_r, loose_r))
        self.assertGreater(lane_r / 255.0, 0.5)
        self.assertLess(loose_r / 255.0, 1.0)

    def test_tyre_tread_only_in_tracks(self):
        P = self.P
        ppm = 24.0
        rect = (0.0, 4.0, -5.0, 5.0)
        h, info, W, H = P._terrain_maps(self.terrain, rect, ppm, 1)
        xs = rect[0] + (np.arange(W) + 0.5) / ppm
        ys = rect[3] - (np.arange(H) + 0.5) / ppm
        X, Y = np.meshgrid(xs, ys)
        z = np.zeros_like(X)
        th, band = P._tread_height(self.terrain, X, Y, ppm, z, z, info["lane"])
        track_rows = np.abs(Y - 1.16) < 0.2
        free_rows = (np.abs(Y - 1.16) > 0.9) & (np.abs(Y + 1.16) > 0.9) & (np.abs(Y - 0.2) > 0.9) & (np.abs(Y - 2.6) > 0.9) \
            & (np.abs(Y + 2.9) > 0.9) & (info["lane"] > 0.99)
        self.assertGreater(abs(th[track_rows]).mean(), 5 * abs(th[free_rows]).mean() + 1e-6)
        self.assertGreater(th[track_rows].min(), -0.05)
        # the chevrons repeat with the albedo's 0.32 m pitch: autocorrelate along x on a track row
        row = th[np.argmin(abs(ys - 1.16))]
        row = row - row.mean()
        lag = int(round(0.32 * ppm))
        ac = (row[:-lag] * row[lag:]).sum() / ((row * row).sum() + 1e-12)
        self.assertGreater(ac, 0.5)

    def test_degenerate_rects_match_bake_albedo(self):
        """bake_albedo copes with 0 / 1 / 2 pixel rects; the detail maps and cavity must give the same shapes
        (they used to raise from np.gradient)."""
        P = self.P
        for rect, ppm in (((0.0, 0.04, 0.0, 0.04), 10.0),       # 0 x 0 px
                          ((0.0, 1.0, 0.0, 1.0), 0.3),          # 0 x 0 px
                          ((0.0, 0.1, 0.0, 0.1), 10.0),         # 1 x 1 px
                          ((0.0, 0.5, 0.0, 0.1), 10.0),         # 5 wide, 1 high
                          ((1.0, 1.1, 0.0, 0.5), 10.0),         # 1 wide, 5 high
                          ((0.0, 0.04, 0.0, 0.04), 50.0)):      # 2 x 2 px
            with self.subTest(rect=rect, ppm=ppm):
                albedo = self.bake(self.terrain, rect, ppm, seed=1)
                normal, rough = P.terrain_detail_maps(self.terrain, rect, ppm, seed=1)
                cav = P.terrain_detail_cavity(self.terrain, rect, ppm, seed=1)
                self.assertEqual(normal.shape, albedo.shape)
                self.assertEqual(rough.shape, albedo.shape[:2])
                self.assertEqual(cav.shape, albedo.shape[:2])
                self.assertEqual((normal.dtype, rough.dtype, cav.dtype), (np.uint8, np.uint8, np.uint8))
                if normal.size:
                    self.assertTrue(np.all(normal[..., 2] > 128))
                    self.assertTrue(np.all(rough > 60))

    def test_banded_processing_equals_full_image_gradient(self):
        """The normal map is built band by band (with a one-row halo); it must equal the plain np.gradient of the
        full relief, also across the band seam (480 rows = 384 + 96)."""
        P = self.P
        rect, ppm = (0.0, 6.0, -12.0, 28.0), 12.0
        normal, rough = P.terrain_detail_maps(self.terrain, rect, ppm, seed=1)
        self.assertEqual(normal.shape[0], 480)
        h, info, W, H = P._terrain_maps(self.terrain, rect, ppm, 1)
        h = h * P._RELIEF_GAIN
        ref = P._encode_normal(-np.gradient(h, axis=1) * ppm, np.gradient(h, axis=0) * ppm)
        self.assertLessEqual(np.abs(normal.astype(int) - ref.astype(int)).max(), 1)
        self.assertGreater(np.abs(normal[380:388].astype(int) - 128).mean(), 0.0)

    def test_banded_cavity_matches_full_image_blur(self):
        """terrain_detail_cavity works in 512-row bands with a blur halo; it equals the single-pass result."""
        P = self.P
        rect, ppm = (0.0, 10.0, -12.0, 28.0), 20.0               # 800 rows: two bands
        cav = P.terrain_detail_cavity(self.terrain, rect, ppm, seed=1)
        h, info, W, H = P._terrain_maps(self.terrain, rect, ppm, 1)
        occ = P._blur_nonperiodic(h, max(1.0, 0.06 * ppm)) - h
        ref = np.clip((1.0 - np.clip(occ * 11.0, 0.0, 0.36)) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        self.assertEqual(cav.shape, (H, W))
        self.assertLessEqual(np.abs(cav.astype(int) - ref.astype(int)).max(), 1)

    def test_detail_px_per_m_helper(self):
        P = self.P
        self.assertAlmostEqual(P.detail_px_per_m(24.0), 60.0)
        rect = (-100.0, 100.0, -50.0, 50.0)                  # 20000 m2: 60 px/m would be 72 Mpx
        capped = P.detail_px_per_m(24.0, rect=rect)
        self.assertLess(capped, 60.0)
        self.assertGreaterEqual(capped, 24.0)
        self.assertLessEqual((rect[1] - rect[0]) * capped * (rect[3] - rect[2]) * capped, 40e6 * 1.001)
        self.assertEqual(P.detail_px_per_m(24.0, rect=(0.0, 10.0, 0.0, 10.0)), 60.0)

    def test_rock_facets_only_on_the_bench_face(self):
        P = self.P
        rect, ppm = (-10.0, 30.0, -8.0, 14.0), 4.0
        W, H = int(round(40 * ppm)), int(round(22 * ppm))
        F = P._coarse_fields(self.terrain, rect, ppm, W, H)
        X, Y = np.meshgrid(F["xs"], F["ys"])
        rock = F["rock"]
        self.assertGreater(rock.max(), 0.5)
        self.assertLess(rock[F["lane"] > 0.95].max(), 0.05)
        self.assertTrue(np.all(Y[rock > 0.3] > self.terrain.lane_left_y - 0.5))      # on the bench side only
        self.assertLess(rock[(Y < 0.0)].max(), 0.01)                                  # none on the right berm

    def test_terrain_tiles_are_seamless_and_deterministic(self):
        P = self.P
        for kind in ("gravel", "cobble", "clod", "grain", "cracks", "rills", "field", "facets"):
            P._terrain_tile.cache_clear()
            a = P._terrain_tile(kind, 5).levels[0]
            P._terrain_tile.cache_clear()
            b = P._terrain_tile(kind, 5).levels[0]
            with self.subTest(kind):
                self.assertTrue(np.array_equal(a, b))
                rx, ry = _seam_ratios(a)
                self.assertLess(max(rx, ry), 1.5, (rx, ry))

    def test_save_terrain_detail_maps(self):
        from PIL import Image

        normal, rough = self.P.terrain_detail_maps(self.terrain, (0.0, 3.0, -1.0, 1.0), 8.0, seed=1)
        with tempfile.TemporaryDirectory() as d:
            paths = self.P.save_terrain_detail_maps(normal, rough, d, "local")
            self.assertEqual(set(paths), {"normal", "roughness"})
            for key, arr in (("normal", normal), ("roughness", rough)):
                self.assertTrue(os.path.isabs(paths[key]))
                with Image.open(paths[key]) as im:
                    self.assertTrue(np.array_equal(np.array(im), arr), key)

    def test_cavity_map(self):
        rect = (0.0, 4.0, -2.0, 2.0)
        cav = self.P.terrain_detail_cavity(self.terrain, rect, 12.0, seed=1)
        self.assertEqual(cav.shape, (48, 48))
        self.assertEqual(cav.dtype, np.uint8)
        self.assertGreater(cav.min(), 100)
        self.assertLess(cav.min(), 255)


if __name__ == "__main__":
    unittest.main()
