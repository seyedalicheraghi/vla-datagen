"""Checks of the drone's garden flight and its onboard cameras (numpy only).

    python scripts/drone_scene/test/test_flight.py
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from drone_scene import flight, scene  # noqa: E402


def quat_to_rot(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class TestFlight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = flight.Flight()
        cls.ts = np.linspace(0, cls.f.duration, 2001)
        cls.P = np.array([cls.f.position(t) for t in cls.ts])
        cls.V = np.array([cls.f.velocity(t) for t in cls.ts])
        cls.A = np.array([cls.f.acceleration(t) for t in cls.ts])

    def test_derivatives(self):
        dt = self.ts[1] - self.ts[0]
        np.testing.assert_allclose(np.gradient(self.P, dt, axis=0)[3:-3], self.V[3:-3], atol=1e-4)
        np.testing.assert_allclose(np.gradient(self.V, dt, axis=0)[3:-3], self.A[3:-3], atol=1e-2)

    def test_takes_off_and_lands_on_the_hover_spot(self):
        cfg = scene.SceneCfg()
        np.testing.assert_allclose(self.P[0], cfg.drone_pos, atol=1e-9)
        np.testing.assert_allclose(self.V[0], 0.0, atol=1e-9)
        np.testing.assert_allclose(self.V[-1], 0.0, atol=1e-9)
        self.assertLess(np.linalg.norm(self.P[-1][:2]), 0.05)

    def test_gentle_flight(self):
        self.assertLess(np.linalg.norm(self.V, axis=1).max(), 2.5, "a slow garden tour")
        tilt = np.degrees(np.arctan(np.linalg.norm(self.A[:, :2], axis=1) / flight.G))
        self.assertLess(tilt.max(), 15.0)
        self.assertGreater(self.P[:, 2].min(), 0.45)
        self.assertLess(self.P[:, 2].max(), 2.0)

    def test_clear_of_shrubs_and_tree(self):
        cfg = scene.SceneCfg()
        bx0, bx1, by0, by1 = cfg.bed
        over_bed = (self.P[:, 0] > bx0 - 0.3) & (self.P[:, 0] < bx1 + 0.3) & (self.P[:, 1] > by0) & (self.P[:, 1] < by1)
        self.assertFalse(over_bed.any(), "never over the shrub bed")
        self.assertGreater(self.P[:, 0].min(), -2.6, "outside the maple's canopy")

    def test_attitude_is_quadrotor_like(self):
        for t in np.linspace(0, self.f.duration, 41):
            R = self.f.rotation(t)
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=9)
            thrust = self.f.acceleration(t) + np.array([0, 0, flight.G])
            np.testing.assert_allclose(R[:, 2], thrust / np.linalg.norm(thrust), atol=1e-9)
            v = self.f.velocity(t)
            if np.linalg.norm(v[:2]) > 0.5:
                h = R[:, 0][:2] / np.linalg.norm(R[:, 0][:2])
                self.assertGreater(np.dot(h, v[:2] / np.linalg.norm(v[:2])), 0.99, "nose along the travel")

    def test_quaternion_roundtrip(self):
        for t in np.linspace(0, self.f.duration, 23):
            R = self.f.rotation(t)
            np.testing.assert_allclose(quat_to_rot(flight.rot_to_quat(R)), R, atol=1e-9)

    def test_isaac_mount_matches_render_cameras(self):
        """Body pose x (offset, quaternion) of the Isaac camera = the world matrix the renders use."""
        for name in flight.CAMERAS:
            pos, q = self.f.camera_local(name)
            Rl = quat_to_rot(q)
            for t in (0.0, 3.3, 7.0, 12.4, 19.0):
                p, R = self.f.pose(t)
                world_R, world_p = R @ Rl, p + R @ pos
                m = self.f.camera_matrix(t, name)
                np.testing.assert_allclose(m[3, :3], world_p, atol=1e-9)
                np.testing.assert_allclose(m[:3, :3], world_R.T, atol=1e-9)        # rows = camera axes

    def test_cameras_look_forward_and_back(self):
        for t in (4.0, 8.0, 13.0):
            v = self.f.velocity(t)
            fwd = -self.f.camera_matrix(t, "front")[2, :3]
            back = -self.f.camera_matrix(t, "back")[2, :3]
            self.assertGreater(np.dot(fwd[:2], v[:2]), 0)
            self.assertLess(np.dot(back[:2], v[:2]), 0)
            self.assertLess(fwd[2], 0, "tilted down")
            fov = 2 * math.degrees(math.atan(flight.CAMERAS["front"].horizontal_aperture / 2
                                             / flight.CAMERAS["front"].focal_length))
            self.assertAlmostEqual(fov, 90.0, delta=0.5)


class TestSmoothMove(unittest.TestCase):
    def check(self, m):
        ts = np.linspace(0, m.duration, 4001)
        S = np.array([m.at(t) for t in ts])
        dt = ts[1] - ts[0]
        self.assertAlmostEqual(S[0, 0], 0.0, places=9)
        self.assertAlmostEqual(S[-1, 0], m.distance, places=6)
        self.assertAlmostEqual(S[-1, 1], 0.0, places=6)
        np.testing.assert_allclose(np.gradient(S[:, 0], dt)[2:-2], S[2:-2, 1], atol=2e-3 * max(1, m.speed))
        np.testing.assert_allclose(np.gradient(S[:, 1], dt)[2:-2], S[2:-2, 2], atol=2e-2 * max(1, m.speed))
        self.assertLessEqual(S[:, 1].max(), m.speed + 1e-9)
        self.assertTrue((np.diff(S[:, 0]) >= -1e-12).all(), "never moves backwards")
        return S

    def test_long_move(self):
        m = flight.SmoothMove(2000.0, 10.0, 8.0)
        S = self.check(m)
        self.assertAlmostEqual(m.duration, 2000 / 10 + 8, places=9)
        self.assertAlmostEqual(S[:, 2].max(), 2 * 10 / 8, places=3)          # peak acceleration 2.5 m/s^2

    def test_short_move_never_reaches_speed(self):
        m = flight.SmoothMove(3.0, 10.0, 2.0)
        S = self.check(m)
        self.assertAlmostEqual(S[:, 1].max(), 1.5, places=3)

    def test_zero(self):
        self.assertEqual(flight.SmoothMove(0.0, 3.0, 2.0).duration, 0.0)


class TestStraightFlight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = flight.StraightFlight(start=(10.0, -5.0, 101.1), end_ground_z=93.0, distance=2000.0, heading_deg=30.0,
                                      cruise_z=145.0, ground_clearance=0.1)
        cls.ts = np.linspace(0, cls.f.duration, 20001)
        cls.P = np.array([cls.f.position(t) for t in cls.ts])
        cls.V = np.array([cls.f.velocity(t) for t in cls.ts])
        cls.A = np.array([cls.f.acceleration(t) for t in cls.ts])

    def test_from_start_to_destination(self):
        np.testing.assert_allclose(self.P[0], (10.0, -5.0, 101.1), atol=1e-9)
        h = math.radians(30.0)
        np.testing.assert_allclose(self.P[-1], (10.0 + 2000 * math.cos(h), -5.0 + 2000 * math.sin(h), 93.1), atol=1e-6)
        np.testing.assert_allclose(self.f.end, self.P[-1], atol=1e-9)
        np.testing.assert_allclose(self.V[[0, -1]], 0.0, atol=1e-12)

    def test_continuous(self):
        dt = self.ts[1] - self.ts[0]
        np.testing.assert_allclose(np.gradient(self.P, dt, axis=0)[2:-2], self.V[2:-2], atol=2e-3)
        np.testing.assert_allclose(np.gradient(self.V, dt, axis=0)[2:-2], self.A[2:-2], atol=2e-2)
        jerk = np.abs(np.diff(self.A, axis=0)).max() / dt
        self.assertLess(jerk, 6.0, "no acceleration jumps (peak jerk of the 2 s vertical ramps is ~4.7 m/s^3)")

    def test_speeds_and_heights(self):
        sp = np.linalg.norm(self.V, axis=1)
        self.assertAlmostEqual(sp.max(), 10.0, places=6)
        self.assertLessEqual(self.P[:, 2].max(), 145.0 + 1e-9)
        horiz = np.linalg.norm(self.V[:, :2], axis=1) > 0.01
        self.assertTrue((self.P[horiz, 2] > 145.0 - 1e-9).all(), "horizontal flight only at cruise height")
        # touchdown slower than 0.5 m/s, and the last 4 m flown slowly
        low = self.P[:, 2] < 93.1 + 4.0 - 1e-6
        self.assertLessEqual(sp[low & (self.ts > 100)].max(), 0.5 + 1e-9)
        self.assertGreater(self.f.duration, 2000 / 10.0)

    def test_attitude(self):
        for t in np.linspace(0, self.f.duration, 61):
            R = self.f.rotation(t)
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
            h = math.radians(30.0)
            self.assertGreater(np.dot(R[:2, 0] / np.linalg.norm(R[:2, 0]), [math.cos(h), math.sin(h)]), 0.9999)
        t_cruise = self.f.phases[3][0] + self.f.phases[3][1] / 2
        self.assertEqual(self.f.phase_name(t_cruise), "cruise")
        tilt = math.degrees(math.acos(self.f.rotation(t_cruise)[2, 2]))
        self.assertAlmostEqual(tilt, math.degrees(math.atan(0.021 * 100 / flight.G)), places=6)   # ~12 deg
        self.assertGreater(tilt, 10.0)
        R = self.f.rotation(t_cruise)
        self.assertGreater(np.dot(R[:, 2], self.f.direction), 0, "leans into the flight direction")

    def test_cameras(self):
        t = self.f.phases[3][0] + 50.0
        fwd = -self.f.camera_matrix(t, "front")[2, :3]
        back = -self.f.camera_matrix(t, "back")[2, :3]
        self.assertGreater(np.dot(fwd, self.f.direction), 0.8)
        self.assertLess(np.dot(back, self.f.direction), -0.8)
        self.assertLess(fwd[2], -math.sin(math.radians(20)), "pitch + forward lean: looks well down")

    def test_rejects_low_cruise(self):
        with self.assertRaises(ValueError):
            flight.StraightFlight(start=(0, 0, 50.0), cruise_z=40.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
