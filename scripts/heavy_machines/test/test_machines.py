"""Offline tests for heavy_machines (pure Python — no Isaac Sim, numpy or torch).

    python scripts/heavy_machines/test/test_machines.py        # unittest
    python -m pytest scripts/heavy_machines/test               # or pytest
"""

from __future__ import annotations

import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from heavy_machines import kinematics as K  # noqa: E402
from heavy_machines import registry  # noqa: E402
from heavy_machines.checks import obb_penetration, validate, _obb  # noqa: E402
from heavy_machines.spec import Link, box  # noqa: E402
from heavy_machines.urdf_builder import ASSET_ROOT, build_urdf, link_inertia, metadata  # noqa: E402


class TestRegistry(unittest.TestCase):
    def test_machines_registered(self):
        self.assertGreaterEqual(len(registry.names()), 1)

    def test_every_machine_file_loads(self):
        self.assertEqual(registry.load_errors(), {}, "\n".join(registry.load_errors().values()))

    def test_unique_abbreviations(self):
        abbrevs = [s.abbrev for s in registry.all_specs().values()]
        self.assertEqual(len(abbrevs), len(set(abbrevs)), abbrevs)

    def test_lookup_by_abbreviation(self):
        for s in registry.all_specs().values():
            self.assertIs(registry.get(s.abbrev.lower()), s)


class TestEveryMachine(unittest.TestCase):
    def test_validates(self):
        for name, spec in registry.all_specs().items():
            with self.subTest(machine=name):
                rep = validate(spec)
                self.assertTrue(rep.ok, rep.format())

    def test_committed_assets_in_sync(self):
        """assets/heavy_machines must match the specs (run check_machine.py to regenerate)."""
        for name, spec in registry.all_specs().items():
            with self.subTest(machine=name):
                urdf = os.path.join(ASSET_ROOT, name, f"{name}.urdf")
                meta = os.path.join(ASSET_ROOT, name, f"{name}.json")
                self.assertTrue(os.path.exists(urdf), f"missing {urdf}")
                with open(urdf) as f:
                    self.assertEqual(f.read(), build_urdf(spec), f"{urdf} is stale")
                with open(meta) as f:
                    self.assertEqual(json.load(f), json.loads(json.dumps(metadata(spec))),
                                     f"{meta} is stale")

    def test_followers_and_camera_fk_finite(self):
        for name, spec in registry.all_specs().items():
            with self.subTest(machine=name):
                for q in (None, {t.joint: spec.joint(t.joint).upper for t in spec.tools
                                 if spec.joint(t.joint).type != "continuous"}):
                    tfs = K.link_transforms(spec, q)
                    for cam in spec.cameras:
                        tf = tfs[cam.link] @ K.Tf.from_xyz_rpy(cam.xyz, cam.rpy)
                        self.assertTrue(all(math.isfinite(v) for v in tf.t))
                        qw = K.mat_to_quat_wxyz(tf.R)
                        self.assertAlmostEqual(sum(v * v for v in qw), 1.0, places=6)

    def test_cameras_inside_machine_footprint(self):
        """Cameras should sit on the machine, not metres away from it."""
        for name, spec in registry.all_specs().items():
            with self.subTest(machine=name):
                lo, hi = K.bounding_box(spec)
                tfs = K.link_transforms(spec)
                for cam in spec.cameras:
                    p = (tfs[cam.link] @ K.Tf.from_xyz_rpy(cam.xyz, cam.rpy)).t
                    for i in range(3):
                        self.assertGreaterEqual(p[i], lo[i] - 0.5, f"camera {cam.name} axis {i}")
                        self.assertLessEqual(p[i], hi[i] + 0.5, f"camera {cam.name} axis {i}")


class TestKinematics(unittest.TestCase):
    def test_rpy_orthonormal(self):
        R = K.rpy_to_mat((0.3, -1.1, 2.0))
        RtR = K.mat_mul(K.transpose(R), R)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(RtR[i][j], 1.0 if i == j else 0.0, places=9)

    def test_negative_y_axis_raises_forward_point(self):
        R = K.axis_angle_to_mat((0, -1, 0), 0.5)
        p = K.mat_vec(R, (1.0, 0.0, 0.0))
        self.assertGreater(p[2], 0.0)

    def test_quaternion_matches_matrix(self):
        R = K.rpy_to_mat((0.2, 0.4, -0.7))
        w, x, y, z = K.mat_to_quat_wxyz(R)
        # rotate a vector by the quaternion and compare
        v = (0.3, -0.5, 0.8)
        qv = (0.0, *v)

        def qmul(a, b):
            return (a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0])
        r = qmul(qmul((w, x, y, z), qv), (w, -x, -y, -z))[1:]
        for a, b in zip(r, K.mat_vec(R, v)):
            self.assertAlmostEqual(a, b, places=9)

    def test_ackermann_no_slip(self):
        """Every steered wheel's heading is perpendicular to the line to the ICR."""
        L, ys = 2.8, [0.9, -0.9]
        for axle in ("front", "rear"):
            for delta in (0.4, -0.3):
                angles = K.ackermann_wheel_angles(delta, L, ys, axle)
                s = 1.0 if axle == "front" else -1.0
                r_icr = s * L / math.tan(delta)
                for y, a in zip(ys, angles):
                    # wheel at (s*L, y) relative to the fixed axle centre; ICR at (0, r_icr)
                    to_icr = (0.0 - s * L, r_icr - y)
                    heading = (math.cos(a), math.sin(a))
                    self.assertAlmostEqual(to_icr[0] * heading[0] + to_icr[1] * heading[1], 0.0, places=9)
        self.assertEqual(K.ackermann_wheel_angles(0.0, L, ys), [0.0, 0.0])

    def test_ackermann_inner_wheel_turns_more(self):
        left, right = K.ackermann_wheel_angles(0.4, 2.8, [0.9, -0.9])
        self.assertGreater(left, right)        # turning left: the left wheel is inner

    def test_articulated_no_slip(self):
        """Integrate the articulated model and check both axles roll without side slip."""
        Lf, Lr, dt = 1.6, 1.7, 1e-4
        x = y = th = 0.0
        g, gdot, v = -0.2, -0.3, 2.0
        for _ in range(2000):
            w = K.articulated_yaw_rate(v, g, gdot, Lf, Lr)
            # front axle point moves along the front heading
            x += v * math.cos(th) * dt
            y += v * math.sin(th) * dt
            th += w * dt
            g += gdot * dt
        # Rear axle lateral velocity at the final state (analytic):
        th_r = th + g
        n_r = (-math.sin(th_r), math.cos(th_r))
        w = K.articulated_yaw_rate(v, g, gdot, Lf, Lr)
        hinge_v = (v * math.cos(th) + Lf * w * math.sin(th), v * math.sin(th) - Lf * w * math.cos(th))
        wr = w + gdot
        rear_v = (hinge_v[0] + Lr * wr * math.sin(th_r), hinge_v[1] - Lr * wr * math.cos(th_r))
        self.assertAlmostEqual(rear_v[0] * n_r[0] + rear_v[1] * n_r[1], 0.0, places=9)

    def test_articulated_turns_left_with_negative_gamma(self):
        self.assertGreater(K.articulated_yaw_rate(2.0, -0.3, 0.0, 1.5, 1.5), 0.0)


class TestInertiaAndOverlap(unittest.TestCase):
    def test_single_box_inertia(self):
        m, com, I = link_inertia(Link("b", 12.0, [box(1.0, 2.0, 3.0)]))
        self.assertAlmostEqual(I[0][0], 12 * (4 + 9) / 12 + 1e-4 * 12, places=6)
        self.assertAlmostEqual(I[2][2], 12 * (1 + 4) / 12 + 1e-4 * 12, places=6)
        self.assertEqual(com, (0.0, 0.0, 0.0))

    def test_parallel_axis(self):
        # two unit cubes 2 m apart along x: Iyy = 2*(1/6) + 2*1*1^2
        m, com, I = link_inertia(Link("b", 2.0, [box(1, 1, 1, at=(-1, 0, 0)), box(1, 1, 1, at=(1, 0, 0))]))
        self.assertAlmostEqual(I[1][1], 2 * (1 / 6) + 2.0 + 2e-4, places=6)
        self.assertAlmostEqual(I[0][0], 2 * (1 / 6) + 2e-4, places=6)

    def test_obb_overlap(self):
        a = _obb(K.Tf(), box(1, 1, 1))
        self.assertGreater(obb_penetration(a, _obb(K.Tf(), box(1, 1, 1, at=(0.8, 0, 0)))), 0.1)
        self.assertLess(obb_penetration(a, _obb(K.Tf(), box(1, 1, 1, at=(1.2, 0, 0)))), 0.0)
        rotated = _obb(K.Tf(), box(1, 1, 1, at=(1.15, 0, 0), rpy=(0, 0, math.pi / 4)))
        self.assertGreater(obb_penetration(a, rotated), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
