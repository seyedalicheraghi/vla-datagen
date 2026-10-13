"""Offline checks of the camera-look post-processing (numpy only; no Isaac Sim).

    python scripts/heavy_machines/test/test_camera_look.py
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import numpy as np
    HAVE_NP = True
except ImportError:  # pragma: no cover
    HAVE_NP = False

if HAVE_NP:
    from heavy_machines.scenarios import camera_look as CL


def _rand_rgb(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _flat(h, w, rgb):
    out = np.empty((h, w, 3), np.uint8)
    out[:] = np.asarray(rgb, np.uint8)
    return out


def _luma(img):
    f = img.astype(np.float64)
    return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]


def _pit_wall_scene(h=180, w=320, ripple=0.04, wide=0.0, flat_sky=False):
    """Blue sky above a rippled far silhouette (700 m), near ground below row 0.75 h; returns rgb, depth, sky mask."""
    x = np.arange(w)
    sil = 0.52 * h + ripple * h * np.sin(2 * np.pi * x / w * 1.1 + 0.7) + 0.02 * h * np.sin(2 * np.pi * x / w * 3.7) + wide * (x - w / 2)
    rows = np.arange(h)[:, None]
    sky = rows < sil[None, :]
    g = np.linspace(0, 1, h)[:, None]
    rgb = np.empty((h, w, 3), np.uint8)
    rgb[..., 0] = (90 + 60 * g).astype(np.uint8)
    rgb[..., 1] = (140 + 50 * g).astype(np.uint8)
    rgb[..., 2] = (225 - 10 * g).astype(np.uint8)
    if flat_sky:
        rgb[:] = (110, 150, 225)
    rgb[~sky] = (105, 90, 70)
    depth = np.where(sky, np.inf, 700.0).astype(np.float32)
    depth[int(0.75 * h):] = 25.0
    return rgb, depth, sky


@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestIdentity(unittest.TestCase):
    def test_identity_passthrough_rgb(self):
        rgb = _rand_rgb(90, 160, 1)
        depth = np.linspace(2, 1500, 90 * 160, dtype=np.float32).reshape(90, 160)
        out = CL.apply(rgb, depth, CL.identity_params(), seed=3)
        self.assertEqual(out.dtype, np.uint8)
        self.assertLessEqual(np.abs(out.astype(int) - rgb.astype(int)).max(), 1)

    def test_identity_passthrough_all_levels(self):
        ramp = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (4, 1, 3))
        out = CL.apply(ramp, None, CL.identity_params())
        self.assertLessEqual(np.abs(out.astype(int) - ramp.astype(int)).max(), 1)

    def test_identity_rgba_and_sky_depth(self):
        rgb = _rand_rgb(48, 64, 2)
        rgba = np.concatenate([rgb, np.full((48, 64, 1), 77, np.uint8)], axis=2)
        depth = np.full((48, 64), np.inf, np.float32)
        depth[24:] = 30.0
        out = CL.apply(rgba, depth, CL.identity_params())
        self.assertEqual(out.shape, (48, 64, 3))
        self.assertLessEqual(np.abs(out.astype(int) - rgb.astype(int)).max(), 1)

    def test_default_is_not_identity(self):
        rgb = _rand_rgb(48, 64, 3)
        out = CL.apply(rgb, None, CL.LookParams())
        self.assertGreater(np.abs(out.astype(int) - rgb.astype(int)).mean(), 2.0)

    def test_input_not_modified(self):
        rgb = _rand_rgb(40, 60, 4)
        depth = np.full((40, 60), 80.0, np.float32)
        rgb0, depth0 = rgb.copy(), depth.copy()
        CL.apply(rgb, depth, CL.LookParams(dust_in_air=0.5), seed=1)
        self.assertTrue((rgb == rgb0).all())
        self.assertTrue((depth == depth0).all())


@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestShapesAndInputs(unittest.TestCase):
    def test_shapes_dtypes(self):
        for shape in ((72, 128), (37, 53), (9, 11), (2, 3), (1, 7), (7, 1), (1, 1)):
            h, w = shape
            rgb = _rand_rgb(h, w)
            for depth in (None, np.full((h, w), 25.0, np.float32), np.full((h, w), 25.0)):
                out = CL.apply(rgb, depth, CL.LookParams(), seed=1)
                self.assertEqual(out.shape, (h, w, 3))
                self.assertEqual(out.dtype, np.uint8)

    def test_rgba_input(self):
        rgba = np.concatenate([_rand_rgb(60, 80), np.full((60, 80, 1), 255, np.uint8)], axis=2)
        out = CL.apply(rgba, None, CL.LookParams(), seed=0)
        self.assertEqual(out.shape, (60, 80, 3))
        ref = CL.apply(rgba[..., :3], None, CL.LookParams(), seed=0)
        self.assertTrue((out == ref).all())

    def test_none_depth_equals_no_haze(self):
        rgb = _rand_rgb(60, 80, 5)
        a = CL.apply(rgb, None, CL.LookParams(), seed=2)
        b = CL.apply(rgb, None, CL.LookParams(haze_strength=0.0), seed=2)
        self.assertTrue((a == b).all())

    def test_sky_values_inf_nan_zero_all_work(self):
        rgb = _rand_rgb(60, 80, 6)
        d = np.full((60, 80), 120.0, np.float32)
        d[:10] = np.inf
        d[10:20] = np.nan
        d[20:30] = 0.0
        out = CL.apply(rgb, d, CL.LookParams(dust_in_air=0.3), seed=0)
        self.assertEqual(out.shape, (60, 80, 3))
        self.assertTrue(np.isfinite(out.astype(float)).all())

    def test_depth_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            CL.apply(_rand_rgb(20, 30), np.ones((30, 20), np.float32), CL.LookParams())
        with self.assertRaises(ValueError):
            CL.apply(np.zeros((20, 30), np.uint8), None, CL.LookParams())

    def test_float_input_read_as_unit_range(self):
        rgb = _rand_rgb(30, 40, 7)
        a = CL.apply(rgb, None, CL.identity_params())
        b = CL.apply(rgb.astype(np.float32) / 255.0, None, CL.identity_params())
        self.assertLessEqual(np.abs(a.astype(int) - b.astype(int)).max(), 1)

    def test_deterministic(self):
        rgb = _rand_rgb(50, 70, 8)
        d = np.full((50, 70), 200.0, np.float32)
        a = CL.apply(rgb, d, CL.LookParams(), seed=5)
        b = CL.apply(rgb, d, CL.LookParams(), seed=5)
        self.assertTrue((a == b).all())

    def test_presets_run(self):
        rgb = _rand_rgb(60, 90, 9)
        d = np.full((60, 90), 150.0, np.float32)
        for fn in (CL.quarry_midday, CL.dusty_afternoon, CL.overcast):
            out = CL.apply(rgb, d, fn(), seed=0)
            self.assertEqual(out.shape, (60, 90, 3))


@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestHaze(unittest.TestCase):
    def _haze_only(self, **kw):
        return CL.identity_params().replace(haze_strength=1.0, **kw)

    def test_haze_increases_with_depth_toward_haze_color(self):
        h, w = 8, 400
        base = (70, 50, 30)                                  # dark brown, darker than the haze in every channel
        rgb = _flat(h, w, base)
        depth = np.tile(np.linspace(0.0, 4000.0, w, dtype=np.float32)[None] + 0.5, (h, 1))
        p = self._haze_only(haze_visibility_m=900.0, haze_color=(0.82, 0.84, 0.86))
        out = CL.apply(rgb, depth, p)[0].astype(int)         # (W, 3)
        hz = np.round(np.array(p.haze_color) * 255).astype(int)
        dist = np.abs(out - hz[None]).sum(axis=1)
        self.assertTrue((np.diff(dist) <= 1).all(), "distance to the haze colour must not grow with depth")
        self.assertTrue((np.diff(out, axis=0) >= -1).all(), "every channel brightens toward the haze colour")
        self.assertGreater(dist[0], 150)                     # near: still the original dark colour
        self.assertLessEqual(np.abs(out[-1] - hz).max(), 2)  # far: the haze colour
        self.assertLessEqual(np.abs(out[0] - np.array(base)).max(), 4)

    def test_haze_matches_beer_lambert(self):
        rgb = _flat(4, 4, (60, 60, 60))
        d = np.full((4, 4), 900.0, np.float32)               # depth == visibility -> T = exp(-1)
        p = self._haze_only(haze_visibility_m=900.0, haze_color=(1.0, 1.0, 1.0))
        out = CL.apply(rgb, d, p)[0, 0, 0] / 255.0
        lin = lambda c: ((c + 0.055) / 1.055) ** 2.4          # noqa: E731
        t = np.exp(-1.0)
        expect = 1.055 * (t * lin(60 / 255.0) + (1 - t)) ** (1 / 2.4) - 0.055
        self.assertAlmostEqual(float(out), float(expect), delta=0.006)

    def test_lower_visibility_means_more_haze(self):
        rgb = _flat(6, 6, (60, 60, 60))
        d = np.full((6, 6), 300.0, np.float32)
        clear = CL.apply(rgb, d, self._haze_only(haze_visibility_m=5000.0))[0, 0, 0]
        hazy = CL.apply(rgb, d, self._haze_only(haze_visibility_m=400.0))[0, 0, 0]
        self.assertGreater(int(hazy), int(clear))

    def test_haze_strength_zero_is_off(self):
        rgb = _rand_rgb(20, 20, 1)
        d = np.full((20, 20), 800.0, np.float32)
        out = CL.apply(rgb, d, CL.identity_params())
        self.assertLessEqual(np.abs(out.astype(int) - rgb.astype(int)).max(), 1)

    def test_sky_glows_toward_horizon(self):
        h, w = 120, 160
        sky_rgb = np.zeros((h, w, 3), np.uint8)
        sky_rgb[:] = (110, 150, 225)                         # saturated blue everywhere
        depth = np.full((h, w), np.inf, np.float32)
        depth[80:] = 200.0                                   # ground below row 80, sky above (horizon at row 79)
        p = CL.identity_params().replace(haze_strength=1.0)
        out = CL.apply(sky_rgb, depth, p).astype(int)
        hz = np.round(np.array(p.haze_color) * 255)
        near_horizon = out[75, 80]
        top = out[2, 80]
        self.assertLess(np.abs(near_horizon - hz).sum(), np.abs(top - hz).sum())
        self.assertLess(np.abs(near_horizon - hz).sum(), 90)        # almost the haze colour at the horizon
        self.assertGreater(top[2] - top[0], 40)                      # the zenith keeps its blue

    def test_dust_adds_warm_extinction(self):
        rgb = _flat(6, 6, (40, 70, 40))
        d = np.full((6, 6), 160.0, np.float32)
        clear = CL.apply(rgb, d, CL.identity_params()).astype(int)[0, 0]
        dusty = CL.apply(rgb, d, CL.identity_params().replace(dust_in_air=1.0)).astype(int)[0, 0]
        self.assertGreater(dusty[0], clear[0])                       # red rises toward the warm dust colour
        self.assertGreater(dusty[2], clear[2])
        self.assertGreater(dusty[0] - dusty[2], clear[0] - clear[2] + 10)


@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestOptics(unittest.TestCase):
    def test_vignette_darkens_corners_not_centre(self):
        h, w = 101, 161
        rgb = _flat(h, w, (150, 150, 150))
        out = CL.apply(rgb, None, CL.identity_params().replace(vignette=0.3))
        c = int(out[h // 2, w // 2, 0])
        self.assertLessEqual(abs(c - 150), 1)
        for y, x in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
            self.assertLess(int(out[y, x, 0]), 150 - 15)
        # monotone toward the corner along the diagonal and mirror-symmetric
        diag = np.array([out[int(i * (h - 1) / 50), int(i * (w - 1) / 50), 0] for i in range(25, 51)], int)
        self.assertTrue((np.diff(diag) <= 0).all())
        self.assertLessEqual(np.abs(out.astype(int) - out[::-1, ::-1].astype(int)).max(), 1)
        # the gain is photometric: the corner ends at about (1 - v) in linear light
        lin = lambda c: ((c / 255.0 + 0.055) / 1.055) ** 2.4   # noqa: E731
        self.assertAlmostEqual(lin(float(out[0, 0, 0])) / lin(150.0), 0.7, delta=0.04)

    def test_default_vignette_is_mild(self):
        rgb = _flat(72, 128, (150, 150, 150))
        p = CL.identity_params().replace(vignette=CL.LookParams().vignette)
        out = CL.apply(rgb, None, p)
        self.assertGreater(int(out[0, 0, 0]), 150 * 0.8)

    def test_distortion_keeps_centre_fixed(self):
        h, w = 101, 161                                              # odd sizes: a true centre pixel
        yy, xx = np.mgrid[0:h, 0:w]
        rgb = np.stack([xx * 255 // (w - 1), yy * 255 // (h - 1), (xx + yy) * 255 // (h + w - 2)], -1).astype(np.uint8)
        for k1 in (-0.08, -0.025, 0.05):
            out = CL.apply(rgb, None, CL.identity_params().replace(distortion_k1=k1))
            self.assertLessEqual(np.abs(out[h // 2, w // 2].astype(int) - rgb[h // 2, w // 2].astype(int)).max(), 1)

    def test_distortion_moves_the_edges_and_fills_the_frame(self):
        h, w = 101, 161
        rgb = _flat(h, w, (255, 255, 255))
        rgb[:, :, :] = 255
        rgb[::10, :, :] = 0                                          # horizontal black lines: bend under distortion
        out = CL.apply(rgb, None, CL.identity_params().replace(distortion_k1=-0.15))
        self.assertGreater(np.abs(out.astype(int) - rgb.astype(int)).mean(), 5)
        white = CL.apply(_flat(h, w, (255, 255, 255)), None, CL.identity_params().replace(distortion_k1=-0.15))
        self.assertGreaterEqual(int(white.min()), 254)              # barrel zoom keeps the frame filled (no black corners)

    def test_distortion_even_size_centre_close(self):
        h, w = 72, 128
        yy, xx = np.mgrid[0:h, 0:w]
        rgb = np.stack([xx * 2, yy * 3, xx + yy], -1).astype(np.uint8)
        out = CL.apply(rgb, None, CL.identity_params().replace(distortion_k1=-0.025))
        self.assertLessEqual(np.abs(out[h // 2, w // 2].astype(int) - rgb[h // 2, w // 2].astype(int)).max(), 3)

    def test_chromatic_aberration_radial_and_centred(self):
        h, w = 101, 161
        rgb = _flat(h, w, (0, 0, 0))
        rgb[:, 120:122] = 255                                        # a bright vertical bar right of centre
        out = CL.apply(rgb, None, CL.identity_params().replace(chromatic_aberration_px=3.0)).astype(float)
        row = out[h // 2]
        centroid = lambda ch: (row[:, ch] * np.arange(w)).sum() / row[:, ch].sum()   # noqa: E731
        r, g, b = centroid(0), centroid(1), centroid(2)
        self.assertGreater(r, g)                                     # red pushed outward (to the right here)
        self.assertLess(b, g)                                        # blue pulled inward
        # centre column untouched: grey content has identical channels there
        grey = _flat(h, w, (128, 128, 128))
        grey[:, 78:83] = 220
        o2 = CL.apply(grey, None, CL.identity_params().replace(chromatic_aberration_px=3.0))
        self.assertLessEqual(np.abs(o2[h // 2, w // 2].astype(int) - o2[h // 2, w // 2, 1]).max(), 1)

    def test_no_ca_keeps_grey_grey(self):
        rgb = _flat(60, 80, (120, 120, 120))
        rgb[20:40, 30:50] = 220
        out = CL.apply(rgb, None, CL.LookParams(white_balance=(1, 1, 1), saturation=1.0, grain=0.0,
                                                chromatic_aberration_px=0.0)).astype(int)
        self.assertLessEqual(np.abs(out[..., 0] - out[..., 2]).max(), 1)

    def test_bloom_spreads_light_from_bright_areas(self):
        h, w = 120, 200
        rgb = _flat(h, w, (20, 20, 20))
        rgb[:, :60] = 255                                            # bright block on the left
        off = CL.apply(rgb, None, CL.identity_params())
        on = CL.apply(rgb, None, CL.identity_params().replace(bloom_strength=0.5, bloom_radius_px=18.0))
        near = (slice(40, 80), slice(62, 72))
        far = (slice(40, 80), slice(185, 199))
        self.assertGreater(on[near].mean(), off[near].mean() + 3)
        self.assertGreater(on[near].mean() - off[near].mean(), on[far].mean() - off[far].mean())
        # dark-only image: nothing to bloom, only the (tiny) veil
        dark = _flat(h, w, (20, 20, 20))
        d_on = CL.apply(dark, None, CL.identity_params().replace(bloom_strength=0.5))
        self.assertLessEqual(np.abs(d_on.astype(int) - dark.astype(int)).max(), 2)

    def test_sharpen_creates_edge_overshoot_not_flat_changes(self):
        rgb = _flat(60, 100, (80, 80, 80))
        rgb[:, 50:] = 180
        out = CL.apply(rgb, None, CL.identity_params().replace(sharpen=1.0)).astype(int)
        self.assertGreater(out[30, 51, 0], 180)
        self.assertLess(out[30, 48, 0], 80)
        self.assertEqual(int(out[30, 5, 0]), 80)
        self.assertEqual(int(out[30, 95, 0]), 180)

    def test_lens_blur_softens_detail(self):
        rgb = _flat(40, 40, (0, 0, 0))
        rgb[:, ::2] = 255
        out = CL.apply(rgb, None, CL.identity_params().replace(lens_blur_px=0.7))
        self.assertLess(out.astype(float)[..., 1].std(), 0.8 * rgb.astype(float)[..., 1].std())


@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestTone(unittest.TestCase):
    def test_tone_curve_is_monotone_and_bounded(self):
        ramp = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (3, 1, 3))
        p = CL.identity_params().replace(contrast=1.5, highlight_rolloff=1.0, saturation=0.9)
        out = CL.apply(ramp, None, p).astype(int)[1]
        self.assertTrue((np.diff(out[:, 1]) >= 0).all())
        self.assertLessEqual(out.max(), 255)
        self.assertLess(out[255, 1], 255)                             # soft shoulder: white is rolled off a little
        self.assertGreater(out[255, 1], 215)
        self.assertGreater(out[200, 1] - out[190, 1], 0)              # ... but the highlights are not flattened

    def test_contrast_is_an_s_curve(self):
        ramp = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (2, 1, 3))
        out = CL.apply(ramp, None, CL.identity_params().replace(contrast=1.0)).astype(int)[0, :, 1]
        self.assertLess(out[64], 64)
        self.assertGreater(out[192], 192)
        self.assertLessEqual(abs(int(out[128]) - 128), 2)

    def test_exposure_and_white_balance(self):
        rgb = _flat(8, 8, (90, 90, 90))
        bright = CL.apply(rgb, None, CL.identity_params().replace(exposure=1.6))[0, 0]
        self.assertTrue((bright > 90).all())
        warm = CL.apply(rgb, None, CL.identity_params().replace(white_balance=(1.2, 1.0, 0.8)))[0, 0].astype(int)
        self.assertGreater(warm[0], warm[1])
        self.assertGreater(warm[1], warm[2])

    def test_saturation(self):
        rgb = _flat(8, 8, (200, 120, 40))
        gray = CL.apply(rgb, None, CL.identity_params().replace(saturation=0.0))[0, 0].astype(int)
        self.assertLessEqual(gray.max() - gray.min(), 2)
        more = CL.apply(rgb, None, CL.identity_params().replace(saturation=1.4))[0, 0].astype(int)
        self.assertGreater(more[0] - more[2], 200 - 40)

    def test_saturated_yellow_stays_yellow_under_default_look(self):
        rgb = _flat(40, 40, (235, 180, 15))
        out = CL.apply(rgb, np.full((40, 40), 20.0, np.float32), CL.LookParams(), seed=0).astype(float)
        m = out[10:30, 10:30].reshape(-1, 3).mean(axis=0)
        self.assertGreater(m[0], 190)
        self.assertGreater(m[0] - m[2], 110)


@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestGrain(unittest.TestCase):
    def setUp(self):
        self.rgb = _flat(144, 256, (128, 128, 128))
        self.p = CL.identity_params().replace(grain=0.02)

    def test_grain_changes_with_seed_but_mean_is_stable(self):
        a = CL.apply(self.rgb, None, self.p, seed=1).astype(float)
        b = CL.apply(self.rgb, None, self.p, seed=2).astype(float)
        self.assertGreater(np.abs(a - b).mean(), 0.5)
        self.assertLess(abs(a.mean() - b.mean()), 0.3)
        self.assertLess(abs(a.mean() - 128.0), 0.4)
        same = CL.apply(self.rgb, None, self.p, seed=1).astype(float)
        self.assertTrue((a == same).all())

    def test_grain_amplitude_is_sensor_like(self):
        a = CL.apply(self.rgb, None, self.p, seed=1).astype(float)
        sd = _luma(a.astype(np.uint8)).std()
        self.assertGreater(sd, 0.8)
        self.assertLess(sd, 6.0)
        off = CL.apply(self.rgb, None, CL.identity_params(), seed=1)
        self.assertEqual(int(off.std()), 0)

    def test_grain_scales_with_parameter(self):
        lo = CL.apply(self.rgb, None, self.p.replace(grain=0.01), seed=3).astype(float).std()
        hi = CL.apply(self.rgb, None, self.p.replace(grain=0.04), seed=3).astype(float).std()
        self.assertGreater(hi, 2.5 * lo)

    def test_grain_weaker_in_deep_shadows_and_highlights_than_midtones(self):
        def sd(v):
            return CL.apply(_flat(120, 200, (v, v, v)), None, self.p, seed=1).astype(float).std()
        self.assertGreater(sd(128), sd(250))

    def test_grain_is_temporally_uncorrelated(self):
        a = CL.apply(self.rgb, None, self.p, seed=10).astype(float)[..., 1] - 128
        b = CL.apply(self.rgb, None, self.p, seed=11).astype(float)[..., 1] - 128
        corr = (a * b).mean() / (a.std() * b.std())
        self.assertLess(abs(corr), 0.1)

    def test_grain_has_chroma_component(self):
        a = CL.apply(self.rgb, None, self.p, seed=4).astype(float)
        self.assertGreater(np.abs(a[..., 0] - a[..., 2]).mean(), 0.2)


SPEC_FIELDS = ["haze_visibility_m", "haze_color", "haze_strength", "exposure", "contrast", "saturation",
               "white_balance", "highlight_rolloff", "bloom_strength", "bloom_radius_px", "vignette",
               "chromatic_aberration_px", "distortion_k1", "sharpen", "grain", "dust_in_air"]


@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestParamsContract(unittest.TestCase):
    def test_field_order_is_the_specified_one(self):
        names = [f.name for f in dataclasses.fields(CL.LookParams)]
        self.assertEqual(names[:len(SPEC_FIELDS)], SPEC_FIELDS)
        self.assertEqual(names[len(SPEC_FIELDS):len(SPEC_FIELDS) + 2], ["lens_blur_px", "dust_color"])   # extras come after

    def test_positional_construction_follows_the_spec(self):
        p = CL.LookParams(850.0, (0.7, 0.7, 0.7), 0.5, 1.2, 1.5, 0.8, (1.0, 1.0, 1.0), 0.6, 0.1, 25.0, 0.3, 1.0, -0.05,
                          0.4, 0.02, 0.15)
        for name, value in zip(SPEC_FIELDS, (850.0, (0.7, 0.7, 0.7), 0.5, 1.2, 1.5, 0.8, (1.0, 1.0, 1.0), 0.6, 0.1,
                                             25.0, 0.3, 1.0, -0.05, 0.4, 0.02, 0.15)):
            self.assertEqual(getattr(p, name), value, name)
        q = CL.LookParams(900.0, (0.82, 0.84, 0.86), 1.0, 1.3)               # the 4th positional is exposure ...
        self.assertEqual(q.exposure, 1.3)
        self.assertEqual(q.dust_in_air, 0.0)                                 # ... and dust stays off

    def test_defaults_match_the_spec(self):
        p = CL.LookParams()
        spec = dict(haze_visibility_m=900.0, haze_color=(0.82, 0.84, 0.86), haze_strength=1.0, exposure=1.0,
                    contrast=1.0, saturation=0.9, white_balance=(1.02, 1.0, 0.97), highlight_rolloff=0.8,
                    bloom_strength=0.06, bloom_radius_px=18.0, vignette=0.22, chromatic_aberration_px=0.7,
                    distortion_k1=-0.025, sharpen=0.25, grain=0.018, dust_in_air=0.0)
        for k, v in spec.items():
            self.assertEqual(getattr(p, k), v, k)
        self.assertIsNone(p.horizon_row)


@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestHorizonStability(unittest.TestCase):
    """The sky glow must not pump when a boom / truck sweeps through the frame or the camera pans."""

    HAZE = None

    @classmethod
    def setUpClass(cls):
        cls.HAZE = CL.identity_params().replace(haze_strength=1.0)

    def _sweep(self, params, ripple, step=10, wid=60, occluder=(70, 70, 72)):
        h, w = 180, 320
        rgb, depth, sky = _pit_wall_scene(h, w, ripple)
        rmax = int(0.55 * h)
        prev = prev_mask = None
        worst = 0
        for x0 in range(-wid + step, w, step):
            xa, xb = max(x0, 0), min(x0 + wid, w)
            r, d = rgb.copy(), depth.copy()
            r[:rmax, xa:xb] = occluder
            d[:rmax, xa:xb] = 8.0
            out = CL.apply(r, d, params).astype(int)
            mask = sky.copy()
            mask[:rmax + 6, max(xa - 12, 0):min(xb + 12, w)] = False           # ignore the occluder's own edge
            if prev is not None:
                worst = max(worst, int(np.abs(out - prev).max(axis=2)[mask & prev_mask].max()))
            prev, prev_mask = out, mask
        return worst

    def test_occluder_sweep_does_not_pop_the_sky(self):
        for ripple in (0.0, 0.03, 0.06):
            worst = self._sweep(self.HAZE, ripple)
            self.assertLessEqual(worst, 2, f"ripple {ripple}: sky changed by {worst} levels for a 10 px occluder step")

    def test_occluder_sweep_full_look(self):
        p = dataclasses.replace(CL.LookParams(), grain=0.0)
        for ripple in (0.0, 0.05):
            worst = self._sweep(p, ripple)
            self.assertLessEqual(worst, 2, f"ripple {ripple}: sky changed by {worst} levels")

    def test_explicit_horizon_row_ignores_everything_in_the_frame(self):
        p = self.HAZE.replace(horizon_row=0.52 * 180)
        self.assertEqual(self._sweep(p, 0.06), 0)
        self.assertEqual(self._sweep(p, 0.0, step=25, wid=90), 0)

    def test_pan_and_tilt_are_stable(self):
        h, w = 180, 320
        rgb, depth, sky = _pit_wall_scene(h + 40, w + 40, 0.03, flat_sky=True)   # flat sky colour: only the glow can change
        prev = prev_sky = None
        worst = 0
        for k in range(24):
            r, d, sk = rgb[k:k + h, k:k + w], depth[k:k + h, k:k + w], sky[k:k + h, k:k + w]
            out = CL.apply(r, d, self.HAZE).astype(int)
            if prev is not None:
                m = sk & prev_sky
                m[int(0.3 * h):] = False                                         # interior sky, away from the silhouette
                worst = max(worst, int(np.abs(out - prev).max(axis=2)[m].max()))
            prev, prev_sky = out, sk
        self.assertLessEqual(worst, 1)

    def test_estimate_ignores_near_boundaries(self):
        h, w = 200, 400
        depth = np.full((h, w), np.inf, np.float32)
        depth[100:] = 600.0                                                      # far horizon at row 99
        depth[:150, 120:260] = 20.0                                              # a machine (35 % of the columns) sticks into the sky
        sky = ~np.isfinite(depth)
        est = CL._horizon_estimate(sky, depth)
        self.assertAlmostEqual(est, 99.0, delta=1.0)
        # without the depth gating the machine's top edge (row 149) would drag the estimate to about 110+
        near_everywhere = np.full((h, w), np.inf, np.float32)
        near_everywhere[120:] = 30.0                                             # only near geometry: still defined
        est2 = CL._horizon_estimate(~np.isfinite(near_everywhere), near_everywhere)
        self.assertAlmostEqual(est2, 119.0, delta=1.0)
        self.assertIsNone(CL._horizon_estimate(np.zeros((10, 10), bool), np.full((10, 10), 5.0, np.float32)))

    def test_all_sky_frame(self):
        rgb = _flat(40, 60, (100, 150, 220))
        out = CL.apply(rgb, np.full((40, 60), np.inf, np.float32), self.HAZE)
        self.assertEqual(out.shape, (40, 60, 3))
        self.assertLessEqual(out[0, 0].astype(int).sum(), out[-1, 0].astype(int).sum() + 3 * 255)   # runs; glow toward the bottom

    def test_explicit_horizon_row_places_the_glow(self):
        h, w = 120, 160
        rgb = _flat(h, w, (110, 150, 225))
        depth = np.full((h, w), np.inf, np.float32)
        hz = np.round(np.array(self.HAZE.haze_color) * 255)
        for row in (30, 90):
            out = CL.apply(rgb, depth, self.HAZE.replace(horizon_row=float(row))).astype(int)
            near = np.abs(out[row - 2, 80] - hz).sum()                          # just above the horizon: almost haze colour
            far = np.abs(out[2, 80] - hz).sum()
            self.assertLess(near, 0.75 * far)
        a = CL.apply(rgb, depth, self.HAZE.replace(horizon_row=30.0)).astype(int)
        b = CL.apply(rgb, depth, self.HAZE.replace(horizon_row=90.0)).astype(int)
        self.assertGreater(np.abs(a[60, 80] - b[60, 80]).sum(), 10)             # the glow moved with the horizon

    def test_horizon_tilt_follows_the_roll(self):
        h, w = 120, 160
        rgb = _flat(h, w, (110, 150, 225))
        depth = np.full((h, w), np.inf, np.float32)
        hz = np.round(np.array(self.HAZE.haze_color) * 255)
        out = CL.apply(rgb, depth, self.HAZE.replace(horizon_row=60.0, horizon_tilt=0.4)).astype(int)
        # the horizon sinks 0.4 rows per column: far left it is at row 28, far right at row 92; at row 50 the left is
        # already below the horizon (full haze), the right is still 40 rows above it
        left = np.abs(out[50, 2] - hz).sum()
        right = np.abs(out[50, w - 3] - hz).sum()
        self.assertLess(left, right - 20)

    def test_horizon_from_camera(self):
        row, tilt = CL.horizon_from_camera(721, 600.0)
        self.assertAlmostEqual(row, 360.0)
        self.assertEqual(tilt, 0.0)
        row, _ = CL.horizon_from_camera(721, 600.0, pitch_down_deg=10.0)
        self.assertAlmostEqual(row, 360.0 - 600.0 * math.tan(math.radians(10.0)), places=6)
        self.assertLess(row, 360.0)                                              # looking down: the horizon moves up
        _, tilt = CL.horizon_from_camera(721, 600.0, roll_deg=5.0)
        self.assertAlmostEqual(tilt, math.tan(math.radians(5.0)), places=9)


@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestBanding(unittest.TestCase):
    """The pipeline runs in cache-sized bands; the result must not depend on the band height."""

    def tearDown(self):
        CL._FORCE_BAND_ROWS = None

    def test_band_height_does_not_change_the_image(self):
        rng = np.random.default_rng(3)
        for h, w in ((61, 97), (150, 41), (33, 200)):
            rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            rgb[:h // 4, :w // 2] = (210, 230, 250)                              # bright patch: exercises bloom
            depth = np.full((h, w), 120.0, np.float32)
            depth[:h // 3] = np.inf
            depth[h // 2:] = np.linspace(5, 400, w, dtype=np.float32)[None]
            for p in (CL.LookParams(), CL.LookParams(dust_in_air=0.3, lens_blur_px=1.3, sharpen=0.6, bloom_radius_px=40.0),
                      CL.LookParams(horizon_row=h * 0.3, horizon_tilt=0.05, saturation=1.3)):
                CL._FORCE_BAND_ROWS = 10_000
                ref = CL.apply(rgb, depth, p, seed=4).astype(int)
                for bands in (4, 12, 20, 36):
                    CL._FORCE_BAND_ROWS = bands
                    out = CL.apply(rgb, depth, p, seed=4).astype(int)
                    self.assertLessEqual(np.abs(out - ref).max(), 1, f"{h}x{w} bands={bands}")
                    self.assertLess(np.abs(out - ref).mean(), 0.02)

    def test_tiny_frames_and_odd_sizes_run(self):
        for h, w in ((1, 1), (1, 9), (9, 1), (3, 3), (5, 130), (131, 5)):
            rgb = _rand_rgb(h, w, h * 31 + w)
            out = CL.apply(rgb, np.full((h, w), 50.0, np.float32), CL.LookParams(dust_in_air=0.2), seed=1)
            self.assertEqual(out.shape, (h, w, 3))


@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestCaches(unittest.TestCase):
    def test_warm_up_builds_the_lens_maps(self):
        CL._warp_maps.cache_clear()
        CL.warm_up(72, 128, run_frame=False)
        self.assertEqual(len(CL._warp_maps._items), 1)
        CL.apply(_rand_rgb(72, 128), None, CL.LookParams())
        self.assertEqual(len(CL._warp_maps._items), 1)                           # reused, not rebuilt
        CL._warp_maps.cache_clear()
        CL.warm_up(72, 128)                                                      # default: also pushes one blank frame through
        self.assertEqual(len(CL._warp_maps._items), 1)

    def test_warm_up_removes_the_first_frame_penalty(self):
        h, w = 720, 1280
        rgb = np.random.default_rng(5).integers(0, 256, (h, w, 3), dtype=np.uint8)
        depth = np.full((h, w), 200.0, np.float32)
        p = CL.LookParams()
        CL._warp_maps.cache_clear()
        CL._vignette_gain.cache_clear()
        CL.warm_up(h, w, p)
        t0 = time.perf_counter()
        CL.apply(rgb, depth, p, seed=1)
        self.assertLess(time.perf_counter() - t0, 0.6)

    def test_map_cache_respects_its_memory_budget(self):
        cache = CL._warp_maps
        old = cache._budget
        try:
            cache.cache_clear()
            cache._budget = 1000                                                 # far too small for any frame
            out = CL.apply(_rand_rgb(40, 60), None, CL.LookParams(), seed=0)
            self.assertEqual(out.shape, (40, 60, 3))
            self.assertEqual(len(cache._items), 0)                               # computed, not stored
            cache._budget = old
            CL.apply(_rand_rgb(40, 60), None, CL.LookParams(), seed=0)
            self.assertEqual(len(cache._items), 1)
        finally:
            cache._budget = old
            cache.cache_clear()


@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestSpeed(unittest.TestCase):
    def test_720p_under_budget(self):
        h, w = 720, 1280
        rng = np.random.default_rng(0)
        rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        depth = np.broadcast_to(5.0 + 1500.0 * y ** 2, (h, w)).astype(np.float32).copy()
        depth[:200] = np.inf
        p = CL.LookParams(dust_in_air=0.2)
        CL.apply(rgb, depth, p, seed=0)                            # first frame builds the per-size lens maps
        best = min(self._time(rgb, depth, p, s) for s in range(1, 5))
        self.assertLess(best, 0.6, f"1280x720 frame took {best:.3f} s")

    def test_1080p_scales_roughly_linearly(self):
        h, w = 1080, 1920
        rgb = np.random.default_rng(2).integers(0, 256, (h, w, 3), dtype=np.uint8)
        depth = np.full((h, w), 300.0, np.float32)
        depth[:300] = np.inf
        p = CL.LookParams()
        CL.apply(rgb, depth, p, seed=0)
        best = min(self._time(rgb, depth, p, s) for s in range(1, 4))
        self.assertLess(best, 1.6, f"1920x1080 frame took {best:.3f} s")             # 2.25x the pixels of 720p

    def test_cold_first_frame_still_reasonable(self):
        h, w = 720, 1280
        rgb = np.random.default_rng(1).integers(0, 256, (h, w, 3), dtype=np.uint8)
        CL._warp_maps.cache_clear()
        CL._vignette_gain.cache_clear()
        t0 = time.perf_counter()
        CL.apply(rgb, None, CL.LookParams(), seed=0)
        self.assertLess(time.perf_counter() - t0, 1.2)

    @staticmethod
    def _time(rgb, depth, p, seed):
        t0 = time.perf_counter()
        CL.apply(rgb, depth, p, seed=seed)
        return time.perf_counter() - t0


if __name__ == "__main__":
    unittest.main(verbosity=2)
