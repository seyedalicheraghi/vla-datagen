"""A 20 s garden flight for the Crazyflie and its two onboard cameras (pure numpy).

The drone takes off from its hover spot over the patio, tours the lawn around the patio at 0.9-1.6 m,
swings past the shrub bed and comes back to hover where it started.  The path is a C2 cubic spline
through timed waypoints, with zero velocity at both ends.  Attitude follows quadrotor physics: the
thrust (body +Z) points along ``a + g``, so the drone tilts into every acceleration, and the heading
(body +X, the Crazyflie's front) follows the direction of travel.

Cameras: ``front`` at the nose looking forward and ``back`` at the tail looking backward, both tilted
10 deg down.  ``CAMERAS`` is the Crazyflie default (a 90 deg FPV lens); ``quadcopter_pbr.py`` replaces it with
the two LUCID cameras (``lucid.py``) at the mounts ``vehicles.mount_cameras`` finds on the drone:
``Flight(cameras=...)``.

    flight = Flight()
    p, R = flight.pose(t)                  # position (3,), body-to-world rotation (3, 3)
    M = flight.camera_matrix(t, "front")   # USD row-vector world matrix of the camera (looks down its -Z)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

G = 9.81

# (t s, x, y, z): over the patio, out over the south-east lawn, up the east side, across the north lawn,
# past the north end of the shrub bed (clear of the maple's canopy, x > -2.6), back over the patio
WAYPOINTS = [
    (0.0, 0.0, 0.0, 0.50),
    (2.5, 1.1, -0.8, 0.90),
    (5.5, 3.4, -2.5, 1.30),
    (9.0, 3.7, 1.8, 1.50),
    (12.0, 1.0, 4.4, 1.60),
    (14.5, -1.8, 3.9, 1.40),
    (17.5, -1.2, 0.8, 1.00),
    (20.0, 0.0, 0.0, 0.55),
]


@dataclass
class OnboardCamera:
    name: str
    offset: tuple[float, float, float]        # in the body frame (m)
    look: float                               # +1 forward (+X), -1 backward
    pitch_down_deg: float = 10.0
    focal_length: float = 10.48               # mm on a 20.955 mm aperture: 90 deg horizontal FOV
    horizontal_aperture: float = 20.955
    vertical_aperture: float | None = None    # None: 16:9


CAMERAS = {
    "front": OnboardCamera("front", (0.045, 0.0, 0.020), +1.0),
    "back": OnboardCamera("back", (-0.045, 0.0, 0.020), -1.0),
}


def _natural_cubic(t: np.ndarray, y: np.ndarray):
    """Clamped (zero end velocity) cubic spline coefficients for each column of y over knots t."""
    n = len(t)
    h = np.diff(t)
    # second derivatives M solve A M = r with clamped ends (y'(t0) = y'(tn) = 0)
    A = np.zeros((n, n))
    r = np.zeros((n, y.shape[1]))
    A[0, 0], A[0, 1] = 2 * h[0], h[0]
    r[0] = 6 * ((y[1] - y[0]) / h[0])
    A[-1, -2], A[-1, -1] = h[-1], 2 * h[-1]
    r[-1] = 6 * (-(y[-1] - y[-2]) / h[-1])
    for i in range(1, n - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        r[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    return np.linalg.solve(A, r)


@dataclass
class Flight:
    waypoints: list = field(default_factory=lambda: list(WAYPOINTS))
    cameras: dict = field(default_factory=lambda: dict(CAMERAS))

    def __post_init__(self):
        w = np.asarray(self.waypoints, float)
        self.t, self.p = w[:, 0], w[:, 1:]
        self.m = _natural_cubic(self.t, self.p)
        self.duration = float(self.t[-1])

    # --- spline evaluation -------------------------------------------------
    def _seg(self, t: float):
        t = min(max(t, self.t[0]), self.t[-1])
        i = int(min(np.searchsorted(self.t, t, side="right") - 1, len(self.t) - 2))
        h = self.t[i + 1] - self.t[i]
        a, b = (self.t[i + 1] - t) / h, (t - self.t[i]) / h
        return i, h, a, b

    def position(self, t: float) -> np.ndarray:
        i, h, a, b = self._seg(t)
        p, m = self.p, self.m
        return a * p[i] + b * p[i + 1] + ((a ** 3 - a) * m[i] + (b ** 3 - b) * m[i + 1]) * h * h / 6

    def velocity(self, t: float) -> np.ndarray:
        i, h, a, b = self._seg(t)
        p, m = self.p, self.m
        return (p[i + 1] - p[i]) / h + ((1 - 3 * a * a) * m[i] + (3 * b * b - 1) * m[i + 1]) * h / 6

    def acceleration(self, t: float) -> np.ndarray:
        i, _, a, b = self._seg(t)
        return a * self.m[i] + b * self.m[i + 1]

    # --- attitude ----------------------------------------------------------
    def heading(self, t: float) -> float:
        """Yaw (rad) along the horizontal velocity; near-standstill ends borrow it from 1 s inside the flight."""
        v = self.velocity(t)[:2]
        if np.linalg.norm(v) < 0.25:
            v = self.velocity(min(max(t, 1.0), self.duration - 1.0))[:2]
        return math.atan2(v[1], v[0])

    def rotation(self, t: float) -> np.ndarray:
        """Body-to-world rotation: body +Z along the thrust (a + g), body +X towards the heading."""
        z = self.acceleration(t) + np.array([0.0, 0.0, G])
        z /= np.linalg.norm(z)
        yaw = self.heading(t)
        x_h = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        y = np.cross(z, x_h)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        return np.column_stack([x, y, z])

    def pose(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        return self.position(t), self.rotation(t)

    def quat_wxyz(self, t: float) -> tuple[float, float, float, float]:
        return rot_to_quat(self.rotation(t))

    # --- cameras ------------------------------------------------------------
    def camera_matrix(self, t: float, name: str) -> np.ndarray:
        """USD row-vector world matrix of an onboard camera (it looks down its -Z, +Y up)."""
        cam = self.cameras[name]
        p, R = self.pose(t)
        pos = p + R @ np.asarray(cam.offset)
        pitch = math.radians(cam.pitch_down_deg)
        fwd_b = np.array([cam.look * math.cos(pitch), 0.0, -math.sin(pitch)])
        right_b = np.array([0.0, -cam.look, 0.0])                       # looking +X, right is -Y
        up_b = np.cross(right_b, fwd_b)
        m = np.eye(4)
        m[0, :3], m[1, :3], m[2, :3], m[3, :3] = R @ right_b, R @ up_b, -(R @ fwd_b), pos
        return m

    def camera_local(self, name: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """The camera's pose in the body frame: (offset, (w, x, y, z)) of a camera looking down its -Z."""
        cam = self.cameras[name]
        pitch = math.radians(cam.pitch_down_deg)
        fwd = np.array([cam.look * math.cos(pitch), 0.0, -math.sin(pitch)])
        right = np.array([0.0, -cam.look, 0.0])
        up = np.cross(right, fwd)
        return np.asarray(cam.offset), rot_to_quat(np.column_stack([right, up, -fwd]))


def rot_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """(w, x, y, z) of a rotation matrix (column vectors)."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    q = q / np.linalg.norm(q)
    return tuple(float(v) for v in (q if q[0] >= 0 else -q))
