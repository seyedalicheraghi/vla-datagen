"""Tests for the realism-loop tools (realism/metrics.py, montage.py, run_probe.sh); numpy + Pillow only.

    python scripts/drone_scene/test/test_realism.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from drone_scene.realism import metrics as M  # noqa: E402
from drone_scene.realism import montage  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pink(n: int, seed: int = 0, alpha: float = 1.0) -> np.ndarray:
    """Random-phase noise with amplitude ~ 1/f^alpha (power ~ 1/f^(2 alpha)), scaled to [0, 1]."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n)
    freq = np.hypot(*np.meshgrid(f, f, indexing="ij"))
    amp = np.where(freq > 0, 1.0 / np.maximum(freq, 1e-9) ** alpha, 0.0)
    img = np.real(np.fft.ifft2(amp * np.exp(2j * np.pi * rng.random((n, n)))))
    return (img - img.min()) / (img.max() - img.min())


class TestStatistics(unittest.TestCase):
    def test_spectral_slope(self):
        self.assertAlmostEqual(M.spectral_slope(pink(512)), -2.0, delta=0.25)
        white = np.random.default_rng(1).random((512, 512))
        self.assertAlmostEqual(M.spectral_slope(white), 0.0, delta=0.25)
        self.assertLess(M.spectral_slope(pink(512, alpha=1.5)), M.spectral_slope(pink(512)) - 0.5, "smoother = steeper")

    def test_noise_sigma(self):
        rng = np.random.default_rng(2)
        img = 0.5 + rng.normal(0.0, 5.0 / 255.0, (400, 600))
        self.assertAlmostEqual(M.noise_sigma(img), 5.0, delta=0.4)
        self.assertLess(M.noise_sigma(np.full((100, 100), 0.3)), 1e-9)

    def test_repetition_finds_tiling(self):
        rng = np.random.default_rng(3)
        tiled = np.tile(rng.random((32, 32)), (16, 8))           # 512 x 256: the lower half is 256 x 256
        noise = rng.random((512, 256))
        self.assertGreater(M.repetition(tiled), 0.8)
        self.assertLess(M.repetition(noise), 0.3)

    def test_colour(self):
        grey = np.full((50, 50, 3), 0.4, np.float32)
        self.assertAlmostEqual(M.colorfulness(grey), 0.0, places=6)
        red = np.zeros((50, 50, 3), np.float32)
        red[..., 0] = 1.0
        self.assertAlmostEqual(M.colorfulness(red), 0.3 * np.hypot(255.0, 127.5), places=3)
        self.assertAlmostEqual(M.saturation(red), 1.0, places=6)
        np.testing.assert_allclose(M.srgb_to_lab(np.ones((1, 1, 3)))[0, 0], [100.0, 0.0, 0.0], atol=0.02)
        np.testing.assert_allclose(M.srgb_to_lab(np.zeros((1, 1, 3)))[0, 0], [0.0, 0.0, 0.0], atol=1e-9)

    def test_box_blur(self):
        a = np.random.default_rng(4).random((40, 60))
        b = M.box_blur(a, 3)
        self.assertEqual(b.shape, a.shape)
        self.assertAlmostEqual(float(b[20, 30]), float(a[17:24, 27:34].mean()), places=9)
        np.testing.assert_allclose(M.box_blur(np.full((20, 20), 2.5), 4), 2.5)

    def test_haze(self):
        img = np.zeros((90, 60, 3), np.float32)
        img[:30] = [0.5, 0.55, 0.65]                             # flat and bluish on top: far away
        img[60:] = np.random.default_rng(5).random((30, 60, 3))  # contrasty below: near
        c, blue = M.haze(img)
        self.assertLess(c, 0.05)
        self.assertGreater(blue, 0.1)

    def test_image_stats_keys(self):
        s = M.image_stats(np.random.default_rng(6).random((256, 320, 3)).astype(np.float32))
        for k in M.KEYS + ["size", "ground_lab"]:
            self.assertIn(k, s)
        self.assertEqual(s["size"], [320, 256])

    def test_compare_warns(self):
        a = {"x.png": {**{k: 0.0 for k in M.KEYS}, "mean_luma": 0.2, "spectral_slope": -2.0, "ground_lab": [50, 0, 0]}}
        b = {"x.png": {**a["x.png"], "clip_hi": 0.05, "spectral_slope": -2.6, "ground_lab": [50, 3, 4]}}
        c = M.compare(a, b)["x.png"]
        self.assertEqual(len(c["warnings"]), 2)
        self.assertAlmostEqual(c["ground_lab_shift"], 5.0)
        self.assertEqual(M.compare(a, a)["x.png"]["warnings"], [])


class TestFolders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.a, self.b = os.path.join(self.tmp, "a"), os.path.join(self.tmp, "b")
        for d, colour in ((self.a, (200, 30, 30)), (self.b, (30, 30, 200))):
            os.makedirs(d)
            for name in ("front_cruise_mid.png", "front_final.png", "chase_cruise_mid.png"):
                Image.new("RGB", (200, 160), colour).save(os.path.join(d, name))
        with open(os.path.join(self.a, "probe.json"), "w") as f:
            json.dump({"stills": [{"file": "front_final.png", "moment": "final", "exposure_us": 300.0}]}, f)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_measure_and_main(self):
        run = M.measure(self.a)
        self.assertEqual(sorted(run), ["chase_cruise_mid.png", "front_cruise_mid.png", "front_final.png"])
        self.assertEqual(run["front_final.png"]["probe"]["exposure_us"], 300.0)
        self.assertEqual(M.main([self.a, self.b, "--reference", self.a]), 0)
        with open(os.path.join(self.b, "metrics.json")) as f:
            res = json.load(f)
        self.assertEqual(set(res["B_minus_A"]), set(run))
        self.assertIn("reference_range", res["reference"])

    def test_ab_is_blind_and_keyed(self):
        out = os.path.join(self.tmp, "ab")
        key = montage.ab(self.a, self.b, out, seed=7)
        stems = {"front_cruise_mid", "front_final", "chase_cruise_mid"}
        self.assertEqual(set(key), {s + x for s in stems for x in ("", "_swap", "_detail", "_detail_swap")})
        with open(os.path.join(out, "key.json")) as f:
            self.assertEqual(json.load(f)["pairs"], key)
        for name, k in key.items():
            im = np.asarray(Image.open(os.path.join(out, name + ".jpg")).convert("RGB")).astype(int)
            h, w = im.shape[:2]
            left, right = im[h // 2, w // 4], im[h // 2, 3 * w // 4]
            self.assertEqual(left[0] > left[2], k["1"] == "A", name)      # A is red
            self.assertEqual(right[0] > right[2], k["2"] == "A", name)
            self.assertEqual(set(k.values()), {"A", "B"})
        for s in stems:
            self.assertEqual(key[s]["1"], key[s + "_swap"]["2"], "the swapped picture has the sides reversed")

    def test_paired_with_real_pictures(self):
        ref = os.path.join(self.tmp, "ref")
        os.makedirs(ref)
        shutil.copy(os.path.join(self.a, "front_final.png"), ref)
        same = M.paired(self.a, ref)
        self.assertEqual(list(same), ["front_final.png"])
        self.assertAlmostEqual(same["front_final.png"]["colour_dE"], 0.0, places=6)
        self.assertGreater(same["front_final.png"]["ssim_coarse"], 0.99)
        self.assertGreater(M.paired(self.b, ref)["front_final.png"]["colour_dE"], 50.0, "blue against red")
        noise = np.random.default_rng(8).random((300, 300, 3))
        self.assertAlmostEqual(M.coarse_ssim(noise, noise), 1.0, places=6)
        self.assertLess(M.coarse_ssim(noise, np.random.default_rng(9).random((300, 300, 3))), 0.5)

    def test_ab_order_varies(self):
        sides = {montage.a_is_left(f"front_m{i}.png", 3) for i in range(20)}
        self.assertEqual(sides, {True, False})
        self.assertEqual(montage.a_is_left("x.png", 1), montage.a_is_left("x.png", 1))

    def test_contact_sheet(self):
        path = montage.contact(self.a)
        with Image.open(path) as im:
            self.assertEqual(im.size, (2 * (640 + 8) + 8, 2 * (160 + 8) + 8))   # 2 cameras x up to 2 moments

    def test_flight_order(self):
        names = ["front_final.png", "front_ground.png", "front_cruise_mid.png", "back_climb.png"]
        self.assertEqual(sorted(names, key=montage._sort_key),
                         ["back_climb.png", "front_ground.png", "front_cruise_mid.png", "front_final.png"])
        times = ["front_t10.png", "front_t2.png", "front_t6.png", "front_final.png"]
        self.assertEqual(sorted(times, key=montage._sort_key),
                         ["front_final.png", "front_t2.png", "front_t6.png", "front_t10.png"])

    def test_empty_reference_folder(self):
        empty = os.path.join(self.tmp, "footage")
        os.makedirs(empty)
        self.assertEqual(M.main([self.a, "--reference", empty]), 0)
        with open(os.path.join(self.a, "metrics.json")) as f:
            self.assertNotIn("reference", json.load(f))

    def test_paired_skips_squares_without_imagery(self):
        ref = os.path.join(self.tmp, "naip")
        os.makedirs(ref)
        Image.new("RGB", (200, 160), (0, 0, 0)).save(os.path.join(ref, "front_final.png"))
        self.assertIn("skipped", M.paired(self.a, ref)["front_final.png"])


class TestRunProbeScript(unittest.TestCase):
    def test_bash_syntax(self):
        if shutil.which("bash") is None:
            self.skipTest("no bash")
        res = subprocess.run(["bash", "-n", os.path.join(HERE, "realism", "run_probe.sh")], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
