"""Offline checks of scenario geometry, motion and framing (numpy; no Isaac Sim).

    python scripts/heavy_machines/test/test_scenarios.py
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import numpy as np
    HAVE_NP = True
except ImportError:  # pragma: no cover
    HAVE_NP = False


@unittest.skipUnless(HAVE_NP, "needs numpy")
class TestAdtApproachHex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from heavy_machines import kinematics as K
        from heavy_machines.scenarios import adt_approach_hex

        cls.K = K
        cls.sc = adt_approach_hex.build()
        cls.adt = cls.sc.actor("ADT")
        cls.hex = cls.sc.actor("HEX")
        cls.times = [i / cls.sc.fps for i in range(cls.sc.n_frames + 1)]

    def _world_points(self, actor, t, link_filter=None, collision=True):
        K = self.K
        base, q = actor.root_tf(t)
        tfs = K.link_transforms(actor.spec, q)
        pts = []
        for lk in actor.spec.links:
            if link_filter and lk.name not in link_filter:
                continue
            for s in lk.shapes:
                if collision and not s.collision:
                    continue
                st = base @ K.shape_tf(tfs[lk.name], s)
                pts.extend(st.apply(p) for p in K.shape_local_points(s, 12))
        return np.array(pts)

    def test_motion_profile(self):
        xs = [self.adt.state(t).x for t in self.times]
        self.assertAlmostEqual(xs[-1], 0.0, places=6, msg="truck must stop at the origin")
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(xs, xs[1:])), "truck never reverses")
        speeds = [(b - a) * self.sc.fps for a, b in zip(xs, xs[1:])]
        jumps = max(abs(b - a) for a, b in zip(speeds, speeds[1:]))
        self.assertLess(jumps, 0.2, "speed changes smoothly frame to frame")
        self.assertEqual(speeds[-1], 0.0)
        self.assertGreater(max(speeds), 2.0)
        self.assertLess(max(speeds), 4.5, "slow approach, not road speed")

    def test_wheels_on_lane_and_flat(self):
        K, terr = self.K, self.sc.terrain
        wheels = [j.child for j in self.adt.spec.joints if j.role == "wheel"]
        for t in self.times[::15]:
            base, q = self.adt.root_tf(t)
            tfs = K.link_transforms(self.adt.spec, q)
            for w in wheels:
                tyre = max((s for s in self.adt.spec.link(w).shapes if s.collision), key=lambda s: s.geom.radius)
                z = K.shape_min_z(base @ tfs[w], tyre)            # exact contact height
                c = (base @ K.shape_tf(tfs[w], tyre)).t
                self.assertAlmostEqual(z, 0.0, delta=0.005, msg=f"{w} touches z = 0 at t={t:.2f}")
                ground = float(terr.height(np.array([c[0]]), np.array([c[1]]))[0])
                self.assertTrue(-0.06 < ground < 0.05, f"{w}: lane surface {ground:.3f} m at t={t:.2f}")

    def test_truck_clears_berms_and_bench(self):
        terr = self.sc.terrain
        for t in self.times[::10]:
            pts = self._world_points(self.adt, t)
            ground = terr.height(pts[:, 0], pts[:, 1])
            pen = ground - pts[:, 2]
            self.assertLess(pen.max(), 0.06, f"truck body inside terrain at t={t:.2f} (by {pen.max():.2f} m)")
        half_w = max(abs(p[1]) for p in self._world_points(self.adt, 0.0))
        self.assertLess(half_w, terr.gate_gap / 2 - 1.0, "at least 1 m each side through the berm gap")

    def test_excavator_parked_level_on_bench(self):
        terr = self.sc.terrain
        pts = self._world_points(self.hex, 0.0, {"undercarriage"})
        bottom = pts[pts[:, 2] < pts[:, 2].min() + 0.02]
        ground = terr.height(bottom[:, 0], bottom[:, 1])
        self.assertTrue(np.all(np.abs(ground - terr.bench_top()) < 0.03), "tracks rest on the level pad")
        self.assertTrue(np.all(np.abs(bottom[:, 2] - terr.bench_top()) < 0.03))
        # the excavator stays on the bench top, back from its crest
        self.assertGreater(pts[:, 1].min(), terr.lane_left_y + terr.bench_face_run + 0.3)

    def test_safe_distance_at_stop(self):
        a = self._world_points(self.adt, self.sc.duration, collision=False)
        h = self._world_points(self.hex, self.sc.duration, collision=False)
        d = min(np.min(np.linalg.norm(h[:, :2] - p[:2], axis=1)) for p in a[::7])
        self.assertGreater(d, 2.0, f"excavator (incl. raised arm) {d:.2f} m from the truck at the stop")

    def test_excavator_in_front_camera(self):
        K = self.K
        cam = next(c for c in self.sc.cameras if c.name == "front")
        f = cam.width * cam.focal_length / cam.horizontal_aperture
        base, q = self.hex.root_tf(0.0)
        target = (base @ K.link_transforms(self.hex.spec, q)["upper_structure"]).t
        sizes = []
        for t in (0.0, self.sc.duration / 2, self.sc.duration):
            ctf = self.sc.camera_tf(cam, t)
            c = K.mat_vec(K.transpose(ctf.R), K.vsub(target, ctf.t))
            self.assertGreater(c[0], 1.0)
            u = cam.width / 2 - f * c[1] / c[0]
            v = cam.height / 2 - f * c[2] / c[0]
            self.assertTrue(0 < u < cam.width / 2, f"excavator should be ahead-LEFT at t={t} (u={u:.0f})")
            self.assertTrue(0 < v < cam.height, f"excavator in frame vertically at t={t} (v={v:.0f})")
            sizes.append(f / c[0])
        self.assertTrue(sizes[0] < sizes[1] < sizes[2], "excavator grows as the truck approaches")

    def test_whole_excavator_in_frame(self):
        """The raised arm must not be cropped at any point of the approach."""
        cam = next(c for c in self.sc.cameras if c.name == "front")
        f = cam.width * cam.focal_length / cam.horizontal_aperture
        pts = self._world_points(self.hex, 0.0, collision=False)
        for t in (0.0, 3.0, 5.0, 6.0, 8.5, self.sc.duration):
            ctf = self.sc.camera_tf(cam, t)
            pc = (pts - np.array(ctf.t)) @ np.array(self.K.transpose(ctf.R)).T
            self.assertTrue((pc[:, 0] > 0.5).all(), f"excavator behind the camera at t={t}")
            u = cam.width / 2 - f * pc[:, 1] / pc[:, 0]
            v = cam.height / 2 - f * pc[:, 2] / pc[:, 0]
            self.assertGreater(v.min(), 10, f"arm cropped at the top at t={t} (v_min={v.min():.0f})")
            self.assertGreater(u.min(), 0, f"excavator cropped at the left at t={t}")

    def test_event_order_and_narrowing(self):
        t = self.sc.terrain
        gate_back = t.gate_x + 1.5                       # half base of the gate berms
        bench_nose = t.bench_start_x - 3.5
        self.assertGreater(bench_nose - gate_back, 2.0, "gate and bench read as two features")
        self.assertGreaterEqual(t.right_berm[0], t.gate_x, "the lane berm starts after the gate")
        width = lambda x: t.lane_left_y - (t.right_berm_center(x) + 1.5)
        self.assertLess(width(t.right_berm[2]) / width(t.gate_x + 1.0), 0.75, "lane visibly narrows")
        # the truck's right tyres keep at least 1 m from the right berm toe at the stop
        self.assertGreater(-1.48 - (t.right_berm_center(0.0) + 1.5), 1.0)

    def test_split_terrain_seamless(self):
        from heavy_machines.scenarios.terrain import split_terrain_meshes
        rect, far = (-20.0, 20.0, -10.0, 15.0), (-200.0, 200.0, -200.0, 200.0)
        near_m, far_m = split_terrain_meshes(self.sc.terrain, rect, far, fine=0.5, cap=10.0)
        near_m.validate()
        far_m.validate()
        on_border = lambda p: (np.isclose(p[:, 0], rect[0]) | np.isclose(p[:, 0], rect[1])
                               | np.isclose(p[:, 1], rect[2]) | np.isclose(p[:, 1], rect[3]))
        a = {tuple(np.round(p, 5)) for p in near_m.points[on_border(near_m.points)]}
        b = {tuple(np.round(p, 5)) for p in far_m.points[on_border(far_m.points)]
             if rect[0] - 1e-6 <= p[0] <= rect[1] + 1e-6 and rect[2] - 1e-6 <= p[1] <= rect[3] + 1e-6}
        self.assertEqual(a, b, "both meshes share exactly the same border vertices")
        cen = far_m.points[far_m.triangles].mean(axis=1)
        inside = (cen[:, 0] > rect[0]) & (cen[:, 0] < rect[1]) & (cen[:, 1] > rect[2]) & (cen[:, 1] < rect[3])
        self.assertFalse(inside.any(), "no far triangles under the detailed area")

    def test_camera_height(self):
        cam = next(c for c in self.sc.cameras if c.name == "front")
        for t in (0.0, self.sc.duration):
            self.assertAlmostEqual(self.sc.camera_tf(cam, t).t[2], 4.0, places=3)

    def test_meshes_and_textures(self):
        from heavy_machines.scenarios.terrain import bake_albedo, grid_mesh, scatter_rocks
        m = grid_mesh(self.sc.terrain, (-20.0, 20.0, -10.0, 15.0), 0.5)
        m.validate()
        img = bake_albedo(self.sc.terrain, (-20.0, 20.0, -10.0, 15.0), 4.0)
        self.assertEqual(img.shape, (100, 160, 3))
        self.assertEqual(img.dtype, np.uint8)
        far = grid_mesh(self.sc.terrain, (-300.0, 300.0, -300.0, 300.0), 30.0, sink_inside=self.sc.local_rect, sink=0.3)
        far.validate()
        rocks = scatter_rocks(self.sc.terrain, self.sc.local_rect, 60, keep_clear=self.sc.keep_clear)
        self.assertEqual(len(rocks), 60)
        lane = self.sc.terrain.lane_mask(np.array([r.x for r in rocks]), np.array([r.y for r in rocks]))
        self.assertTrue(np.all(lane < 0.05), "no rocks on the haul lane")


if __name__ == "__main__":
    unittest.main(verbosity=2)
