"""Checks of the LUCID camera models and the simulated sensor (numpy + torch on the CPU; no Isaac Sim).

    python scripts/drone_scene/test/test_lucid.py
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from drone_scene import lucid  # noqa: E402

try:
    import torch

    from drone_scene import sensor_sim as ss
except ImportError:              # pragma: no cover
    torch = None


class TestCatalogue(unittest.TestCase):
    def test_datasheet_geometry(self):
        r = lucid.make_rig("TRI032S-CC")
        self.assertEqual((r.width, r.height), (2048, 1536))
        self.assertAlmostEqual(r.sensor_diagonal_mm(), 8.83, places=2)        # 1/1.8"
        self.assertAlmostEqual(r.fx, 4.0 / 0.00345, places=6)
        self.assertEqual(lucid.make_rig("TRI050S-CC").lens.name, lucid.LENSES["C6"].name)
        self.assertEqual(lucid.make_rig("PHX016S-CS").body.mount, "S")

    def test_mount_mismatch_and_unknowns(self):
        with self.assertRaises(ValueError):
            lucid.make_rig("PHX016S-CS", "C4")
        with self.assertRaises(KeyError):
            lucid.make_rig("TRI999")
        with self.assertRaises(KeyError):
            lucid.make_rig("TRI032S-CC", "C99")

    def test_warnings(self):
        self.assertEqual(lucid.make_rig("TRI032S-CC").warnings(25), [])
        self.assertTrue(any("fps" in w for w in lucid.make_rig("TRI050S-CC").warnings(25)))
        self.assertTrue(any("image circle" in w for w in lucid.make_rig("TRI050S-CC", "C4").warnings(20)))

    def test_hyperfocal(self):
        r = lucid.make_rig("TRI032S-CC")
        c = 2 * 0.00345
        self.assertAlmostEqual(r.hyperfocal_m(), (16 / (2.8 * c) + 4) / 1000, places=6)

    def test_resolution_scale(self):
        r = lucid.make_rig("TRI032S-CC", resolution_scale=0.5)
        self.assertEqual((r.width, r.height), (1024, 768))
        full = lucid.make_rig("TRI032S-CC")
        np.testing.assert_allclose(r.field_of_view_deg(), full.field_of_view_deg(), atol=0.05)


class TestDistortion(unittest.TestCase):
    def test_roundtrip(self):
        lens = lucid.Lens("t", 4.0, 2.8, "C", k1=-0.12, k2=0.03, p1=0.001, p2=-0.0005, k3=0.002)
        rng = np.random.default_rng(1)
        x, y = rng.uniform(-1.0, 1.0, 500), rng.uniform(-0.75, 0.75, 500)
        xd, yd = lucid.distort(x, y, lens)
        xu, yu = lucid.undistort(xd, yd, lens)
        np.testing.assert_allclose(xu, x, atol=1e-9)
        np.testing.assert_allclose(yu, y, atol=1e-9)

    def test_barrel_widens_the_view(self):
        r = lucid.make_rig("TRI032S-CC")
        h, v, d = r.field_of_view_deg()
        pinhole_h = 2 * math.degrees(math.atan(r.width / 2 / r.fx))
        self.assertGreater(h, pinhole_h)
        self.assertLess(h, pinhole_h + 8)

    def test_render_plan_covers_the_image(self):
        for cam in lucid.CAMERAS:
            r = lucid.make_rig(cam)
            plan = r.render_plan()
            self.assertEqual(plan["width"] % 2, 0)
            self.assertAlmostEqual(plan["width"] * plan["focal_length"] / plan["horizontal_aperture"], plan["fx"])
            self.assertAlmostEqual(plan["horizontal_aperture"] / plan["vertical_aperture"],
                                   plan["width"] / plan["height"])
            xd, yd = r._edges()
            xu, yu = lucid.undistort(xd, yd, r.lens)
            ur = xu * plan["fx"] + (plan["width"] - 1) / 2
            vr = yu * plan["fx"] + (plan["height"] - 1) / 2
            self.assertGreaterEqual(ur.min(), -0.5)
            self.assertLessEqual(ur.max(), plan["width"] - 0.5)
            self.assertGreaterEqual(vr.min(), -0.5)
            self.assertLessEqual(vr.max(), plan["height"] - 0.5)
            self.assertLess(plan["overscan"], 1.25)

    def test_camera_info(self):
        r = lucid.make_rig("TRI032S-CC")
        info = r.camera_info("front")
        self.assertEqual(info["distortion_model"], "plumb_bob")
        self.assertEqual(len(info["D"]), 5)
        self.assertEqual(len(info["K"]), 9)
        self.assertEqual(info["K"][2], (2048 - 1) / 2)
        self.assertEqual(info["sensor"]["shutter"], "global")


try:
    from pxr import Usd, UsdGeom
except ImportError:              # pragma: no cover
    Usd = None


@unittest.skipIf(Usd is None, "pxr (usd-core) not installed")
class TestBody(unittest.TestCase):
    def test_body_geometry(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        rig = lucid.make_rig("TRI032S-CC")
        path = lucid.author_body(stage, "/World/Cam/body", rig, bracket_mm=20.0)
        prim = stage.GetPrimAtPath(path)
        self.assertTrue(prim.IsValid())
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeLocalBound(prim).ComputeAlignedRange()
        size = box.GetSize()
        self.assertAlmostEqual(size[0], 0.030, delta=0.0015)                   # lens barrel 30 mm across
        self.assertAlmostEqual(box.GetMin()[2], 0.0, delta=1e-6)               # lens front at the origin
        self.assertGreater(box.GetMax()[2], (28 + 45) * 1e-3)                  # housing + connectors behind it
        for name in ("glass", "housing", "gige_m12", "bracket"):
            self.assertTrue(stage.GetPrimAtPath(f"{path}/{name}").IsValid(), name)
        # a second camera reuses the materials
        lucid.author_body(stage, "/World/Cam2/body", rig)
        self.assertEqual(len([p for p in stage.Traverse() if p.GetTypeName() == "Material"]),
                         len(lucid.BODY_MATERIALS))


@unittest.skipIf(torch is None, "torch not installed")
class TestSensor(unittest.TestCase):
    def setUp(self):
        self.rig = lucid.make_rig("TRI032S-CC", resolution_scale=0.125)        # 256 x 192
        self.plan = self.rig.render_plan()

    def flat(self, value=1.0):
        return torch.full((self.plan["height"], self.plan["width"], 4), float(value))

    def test_demosaic_constant_colour_is_exact(self):
        rgb = torch.tensor([0.2, 0.5, 0.8]).view(3, 1, 1).expand(3, 16, 20)
        out = ss.demosaic(ss.mosaic(rgb))
        np.testing.assert_allclose(out.numpy(), rgb.numpy(), atol=1e-6)

    def test_demosaic_smooth_gradient(self):
        y, x = torch.meshgrid(torch.linspace(0, 1, 32), torch.linspace(0, 1, 40), indexing="ij")
        rgb = torch.stack([0.2 + 0.5 * x, 0.3 + 0.4 * y, 0.6 - 0.3 * x * y])
        out = ss.demosaic(ss.mosaic(rgb))
        self.assertLess(float((out - rgb)[:, 3:-3, 3:-3].abs().max()), 2e-3)

    def test_bayer_layout(self):
        m = ss.bayer_masks(4, 4, "RGGB")
        self.assertEqual(m[0, 0, 0], 1)          # R at (0, 0)
        self.assertEqual(m[1, 0, 1], 1)          # G
        self.assertEqual(m[1, 1, 0], 1)          # G
        self.assertEqual(m[2, 1, 1], 1)          # B at (1, 1)
        np.testing.assert_array_equal(m.sum(0).numpy(), np.ones((4, 4)))

    def test_lens_maps_a_point_like_opencv(self):
        """A bright dot rendered on the pinhole image lands where the OpenCV model projects that ray."""
        sensor = ss.LucidSensor(self.rig, self.plan, settings=ss.SensorSettings(noise=False))
        for xu, yu in ((0.0, 0.0), (0.7, 0.45), (-0.8, -0.5)):
            hdr = torch.zeros(self.plan["height"], self.plan["width"], 3)
            ur = xu * self.plan["fx"] + (self.plan["width"] - 1) / 2
            vr = yu * self.plan["fx"] + (self.plan["height"] - 1) / 2
            iu, iv = int(round(ur)), int(round(vr))
            hdr[iv - 1:iv + 2, iu - 1:iu + 2] = 1.0
            # the true dot centre after rounding to whole render pixels
            xu_t = (iu - (self.plan["width"] - 1) / 2) / self.plan["fx"]
            yu_t = (iv - (self.plan["height"] - 1) / 2) / self.plan["fx"]
            img = sensor.optics(hdr)[1]
            vv, uu = torch.meshgrid(torch.arange(img.shape[0]), torch.arange(img.shape[1]), indexing="ij")
            w = img.sum()
            u_c, v_c = float((img * uu).sum() / w), float((img * vv).sum() / w)
            xd, yd = lucid.distort(np.array(xu_t), np.array(yu_t), self.rig.lens)
            self.assertAlmostEqual(u_c, float(xd) * self.rig.fx + self.rig.cx, delta=0.25)
            self.assertAlmostEqual(v_c, float(yd) * self.rig.fx + self.rig.cy, delta=0.25)

    def test_vignetting(self):
        vig = ss.LucidSensor(self.rig, self.plan).vignette[0]
        H, W = vig.shape
        self.assertAlmostEqual(float(vig[H // 2, W // 2]), 1.0, places=3)
        self.assertAlmostEqual(float(vig[0, 0]), self.rig.lens.relative_illumination, delta=0.03)

    def test_auto_exposure_hits_target(self):
        sensor = ss.LucidSensor(self.rig, self.plan, settings=ss.SensorSettings(noise=False, prnu=0.0))
        hdr = self.flat(37.0)
        out = sensor.process(hdr)
        raw_lin = (sensor.optics(hdr) / sensor.wb * sensor.masks).sum(0)
        self.assertAlmostEqual(sensor._meter(raw_lin, sensor.k), 0.22, delta=0.002)
        # nothing clips, so the raw frame's mean is the target too (up to rounding to whole DN)
        self.assertAlmostEqual(float(out["raw"].float().mean()) / sensor.full_scale, 0.22, delta=0.003)
        self.assertEqual(out["saturated"], 0.0)

    def test_auto_exposure_with_a_bright_sky(self):
        """The top eighth of the frame 100x brighter (sky): it clips, and AE meters the clipped signal."""
        sensor = ss.LucidSensor(self.rig, self.plan, settings=ss.SensorSettings(noise=False, prnu=0.0))
        hdr = self.flat(1.0)
        hdr[: self.plan["height"] // 8] = 100.0
        out = sensor.process(hdr)
        self.assertGreater(out["saturated"], 0.05)          # the sky is blown, as with a real camera
        raw_lin = (sensor.optics(hdr) / sensor.wb * sensor.masks).sum(0)
        self.assertAlmostEqual(sensor._meter(raw_lin, sensor.k), 0.22, delta=0.002)

    def test_auto_exposure_is_damped(self):
        sensor = ss.LucidSensor(self.rig, self.plan, settings=ss.SensorSettings(noise=False))
        sensor.process(self.flat(10.0))
        k0 = sensor.k
        sensor.process(self.flat(20.0))                    # the scene doubles in brightness
        k1 = sensor.k
        self.assertLess(k1, k0)
        self.assertGreater(k1, k0 / 2)                      # moves only part of the way in one frame
        for _ in range(40):
            sensor.process(self.flat(20.0))
        self.assertAlmostEqual(sensor.k, k0 / 2, delta=k0 * 0.01)

    def test_reset_exposure_converges_at_once(self):
        sensor = ss.LucidSensor(self.rig, self.plan, settings=ss.SensorSettings(noise=False))
        sensor.process(self.flat(0.2))                        # e.g. the 8-bit fallback image
        sensor.reset_exposure()
        sensor.process(self.flat(200.0))                      # then the HDR render, 1000x the scale
        fresh = ss.LucidSensor(self.rig, self.plan, settings=ss.SensorSettings(noise=False))
        fresh.process(self.flat(200.0))
        self.assertAlmostEqual(sensor.k, fresh.k, delta=fresh.k * 1e-6)

    def test_shot_and_read_noise(self):
        """A flat grey field: the raw green values scatter like Poisson electrons + read noise."""
        sensor = ss.LucidSensor(self.rig, self.plan, settings=ss.SensorSettings(prnu=0.0, wb_gains=(1, 1, 1)))
        sensor.vignette = torch.ones_like(sensor.vignette)
        sen = self.rig.body.sensor
        frac = 0.25
        dn = sensor.sense(torch.full((self.rig.height, self.rig.width), frac))
        e = dn / sensor.full_scale * sensor.adc_electrons
        expected_std = math.sqrt(frac * sensor.adc_electrons + sen.read_noise_e ** 2)
        quant = sensor.adc_electrons / sensor.full_scale / math.sqrt(12)
        self.assertAlmostEqual(float(e.mean()), frac * sensor.adc_electrons, delta=5)
        self.assertAlmostEqual(float(e.std()), math.hypot(expected_std, quant), delta=expected_std * 0.05)

    def test_saturation_clips(self):
        sensor = ss.LucidSensor(self.rig, self.plan)
        dn = sensor.sense(torch.full((self.rig.height, self.rig.width), 3.0))
        self.assertTrue(bool((dn == sensor.full_scale).all()))

    def test_fixed_exposure_needs_calibration(self):
        sensor = ss.LucidSensor(self.rig, self.plan, settings=ss.SensorSettings(exposure="fixed"))
        with self.assertRaises(ValueError):
            sensor.process(self.flat())

    def test_exposure_time_is_plausible_for_daylight(self):
        """An 18 % grey card under the reference illuminance (100 klx): auto exposure lands at a fraction of a ms."""
        e_ref = 1000.0
        sensor = ss.LucidSensor(self.rig, self.plan, reference_illuminance=e_ref,
                                settings=ss.SensorSettings(noise=False))
        out = sensor.process(self.flat(0.18 * e_ref / math.pi))
        self.assertGreater(out["exposure_us"], 50)
        self.assertLess(out["exposure_us"], 2000)
        fixed = ss.LucidSensor(self.rig, self.plan, reference_illuminance=e_ref, settings=ss.SensorSettings(
            exposure="fixed", exposure_us=out["exposure_us"], noise=False))
        np.testing.assert_allclose(fixed.process(self.flat(0.18 * e_ref / math.pi))["raw"].numpy(),
                                   out["raw"].numpy(), atol=1)

    def test_process_outputs(self):
        sensor = ss.LucidSensor(self.rig, self.plan, seed=3)
        hdr = torch.rand(self.plan["height"], self.plan["width"], 4) * 5
        out = sensor.process(hdr)
        self.assertEqual(tuple(out["rgb"].shape), (self.rig.height, self.rig.width, 3))
        self.assertEqual(out["rgb"].dtype, torch.uint8)
        self.assertEqual(tuple(out["raw"].shape), (self.rig.height, self.rig.width))
        self.assertLessEqual(int(out["raw"].max()), 4095)
        again = ss.LucidSensor(self.rig, self.plan, seed=3).process(hdr)
        self.assertTrue(torch.equal(out["rgb"], again["rgb"]))
        with self.assertRaises(ValueError):
            sensor.process(torch.zeros(10, 10, 3))

    def test_colour_survives_the_pipeline(self):
        """White balance undoes the colour filter: a coloured flat field comes out with the same hue."""
        sensor = ss.LucidSensor(self.rig, self.plan, settings=ss.SensorSettings(noise=False, prnu=0.0))
        hdr = torch.tensor([0.30, 0.50, 0.20]).expand(self.plan["height"], self.plan["width"], 3).clone()
        rgb = ss.srgb_decode(sensor.process(hdr)["rgb"].float() / 255.0)
        c = rgb[self.rig.height // 2 - 4:self.rig.height // 2 + 4, self.rig.width // 2 - 4:self.rig.width // 2 + 4]
        c = c.reshape(-1, 3).mean(0)
        np.testing.assert_allclose((c / c[1]).numpy(), [0.6, 1.0, 0.4], atol=0.03)

    def test_overflowed_pixels_stay_bright(self):
        sensor = ss.LucidSensor(self.rig, self.plan, settings=ss.SensorSettings(noise=False, prnu=0.0))
        hdr = torch.full((self.plan["height"], self.plan["width"], 3), 100.0, dtype=torch.float16)
        cy, cx = self.plan["height"] // 2, self.plan["width"] // 2
        hdr[cy - 10:cy + 10, cx - 10:cx + 10] = float("inf")
        out = sensor.process(hdr)["rgb"]
        self.assertEqual(int(out[self.rig.height // 2, self.rig.width // 2].min()), 255)

    def test_lateral_colour_matches_the_lens_at_the_corner(self):
        """Red-blue separation of the image corner, measured in output pixels through the distortion."""
        for cam in ("TRI032S-CC", "PHX016S-CS"):
            rig = lucid.make_rig(cam, resolution_scale=0.25)
            plan = rig.render_plan()
            g, _ = ss.lens_grids(rig, plan)
            # render-pixel positions sampled for the corner output pixel, red and blue
            to_px = lambda q: ((q[..., 0] + 1) / 2 * plan["width"] - 0.5, (q[..., 1] + 1) / 2 * plan["height"] - 0.5)  # noqa: E731
            ur, vr = to_px(g[0, 0, 0].numpy())
            ub, vb = to_px(g[2, 0, 0].numpy())
            sep_render = math.hypot(float(ur - ub), float(vr - vb))
            # local render-px per output-px along the diagonal, from the green grid's neighbour
            u0, v0 = to_px(g[1, 0, 0].numpy())
            u1, v1 = to_px(g[1, 1, 1].numpy())
            jac = math.hypot(float(u1 - u0), float(v1 - v0)) / math.sqrt(2)
            self.assertAlmostEqual(sep_render / jac, rig.lens.lateral_color_px, delta=0.06 * rig.lens.lateral_color_px)

    def test_lateral_colour_separates_red_and_blue_at_the_corner(self):
        sensor = ss.LucidSensor(self.rig, self.plan)
        g = sensor.grid
        sep = (g[0, 0, 0] - g[2, 0, 0]) / 2 * torch.tensor([self.plan["width"], self.plan["height"]])
        # in render pixels, close to the lens's corner value scaled by the local distortion
        self.assertGreater(float(sep.norm()), 0.3 * self.rig.lens.lateral_color_px)
        self.assertLess(float(sep.norm()), 3.0 * self.rig.lens.lateral_color_px)


if __name__ == "__main__":
    unittest.main()
