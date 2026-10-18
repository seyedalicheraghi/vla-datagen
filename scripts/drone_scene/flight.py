"""Drone flights and their two onboard cameras (pure numpy): ``Flight``, a 20 s garden tour, and
``StraightFlight``, a take-off / straight cruise / landing mission between two places on a large map.

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


class _Trajectory:
    """What every flight shares: the pose, and the poses of the cameras riding on the body.  Subclasses
    provide ``position(t)``, ``rotation(t)`` and a ``cameras`` dict."""

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


@dataclass
class Flight(_Trajectory):
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
        """Yaw (rad) along the horizontal velocity.  Near standstill (the ends) the direction of travel 1 s inside
        the flight is blended in, fading out smoothly by 0.5 m/s, so the nose never snaps round."""
        v = self.velocity(t)[:2]
        ref = self.velocity(min(max(t, 1.0), self.duration - 1.0))[:2]
        ref = ref / max(np.linalg.norm(ref), 1e-9)
        v = v + max(0.0, 0.5 - float(np.linalg.norm(v))) * ref
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

# ---------------------------------------------------------------------------
# Point-to-point flight (a delivery-style mission over a large map)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SmoothMove:
    """Rest-to-rest move over ``distance``: the acceleration rises and falls as sin^2 over ``ramp`` seconds
    (no jerk spikes), then a constant ``speed`` cruise, then the mirror-image stop.  Short moves never reach
    ``speed``."""

    distance: float
    speed: float
    ramp: float

    @property
    def v_max(self) -> float:
        return min(self.speed, self.distance / self.ramp) if self.ramp > 0 else self.speed

    @property
    def duration(self) -> float:
        if self.distance <= 0:
            return 0.0
        v = self.v_max
        return 2 * self.ramp + (self.distance - v * self.ramp) / v

    def _ramp(self, t: float) -> tuple[float, float, float]:
        """(s, v, a) after t seconds of the speed-up ramp."""
        T, A = self.ramp, 2 * self.v_max / self.ramp
        w = 2 * math.pi / T
        a = A * math.sin(math.pi * t / T) ** 2
        v = A * (t / 2 - math.sin(w * t) / (2 * w))
        s = A * (t * t / 4 + (math.cos(w * t) - 1) / (2 * w * w))
        return s, v, a

    def at(self, t: float) -> tuple[float, float, float]:
        """(distance travelled, speed, acceleration) at time t (clamped to the move)."""
        if self.distance <= 0:
            return 0.0, 0.0, 0.0
        T, D, v = self.ramp, self.distance, self.v_max
        t = min(max(t, 0.0), self.duration)
        if t <= T:
            return self._ramp(t)
        if t >= self.duration - T:
            s, sv, sa = self._ramp(self.duration - t)
            return D - s, sv, -sa
        return v * T / 2 + v * (t - T), v, 0.0


@dataclass
class StraightFlight(_Trajectory):
    """Take off from ``start``, climb vertically to ``cruise_z``, fly ``distance`` metres straight along
    ``heading_deg`` (counter-clockwise from +X), stop, descend vertically (slowly for the last
    ``final_height`` metres) and land on the ground at the destination (``end_ground_z``).

    ``start`` is the body origin on the ground at take-off (ground height + ``ground_clearance``, the body
    origin's height above its landing gear).  Heights are world z.  Attitude follows quadrotor physics with
    air drag: body +Z is along ``a + g + drag |v_air| v_air`` (thrust also cancels drag, so the drone leans
    into the air stream even at constant speed: ~12 deg at 10 m/s with the default ``drag``), body +X along
    ``heading_deg``.
    """

    start: tuple = (0.0, 0.0, 0.10)
    end_ground_z: float = 0.0
    distance: float = 2000.0
    heading_deg: float = 0.0
    cruise_z: float = 40.0
    speed: float = 10.0
    climb_rate: float = 3.0
    descent_rate: float = 2.0
    final_height: float = 4.0
    final_descent_rate: float = 0.5
    ground_clearance: float = 0.10
    ramp: float = 8.0                     # s from hover to cruise speed
    vertical_ramp: float = 2.0
    idle: float = 2.0                     # s on the ground before take-off and after landing
    hover: float = 1.5                    # s of hover between the phases
    drag: float = 0.021                   # quadratic drag per unit mass (1/m)
    wind: tuple = (0.0, 0.0, 0.0)         # m/s, world frame
    cameras: dict = field(default_factory=lambda: dict(CAMERAS))

    def __post_init__(self):
        h = math.radians(self.heading_deg)
        self.direction = np.array([math.cos(h), math.sin(h), 0.0])
        p0 = np.asarray(self.start, float)
        land_z = self.end_ground_z + self.ground_clearance
        if self.cruise_z < max(p0[2], land_z + self.final_height):
            raise ValueError(f"cruise_z {self.cruise_z:.1f} is below the take-off or landing approach height")
        up, down = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])
        # (direction, move) phases; None = stand still for that many seconds
        plan = [
            (None, self.idle),
            (up, SmoothMove(self.cruise_z - p0[2], self.climb_rate, self.vertical_ramp)),
            (None, self.hover),
            (self.direction, SmoothMove(self.distance, self.speed, self.ramp)),
            (None, self.hover),
            (down, SmoothMove(self.cruise_z - (land_z + self.final_height), self.descent_rate, self.vertical_ramp)),
            (None, self.hover),
            (down, SmoothMove(self.final_height, self.final_descent_rate, self.vertical_ramp)),
            (None, self.idle),
        ]
        self.phases, t, p = [], 0.0, p0.copy()
        for d, m in plan:
            T = m if d is None else m.duration
            self.phases.append((t, T, p.copy(), d, m))
            if d is not None:
                p = p + d * m.distance
            t += T
        self.duration = t
        self.end = p
        self.yaw = h

    def _state(self, t: float):
        t = min(max(t, 0.0), self.duration)
        for t0, T, p0, d, m in self.phases:
            if t <= t0 + T or (t0, T) == self.phases[-1][:2]:
                if d is None:
                    return p0, np.zeros(3), np.zeros(3)
                s, v, a = m.at(t - t0)
                return p0 + d * s, d * v, d * a
        raise AssertionError("unreachable")

    def position(self, t: float) -> np.ndarray:
        return self._state(t)[0]

    def velocity(self, t: float) -> np.ndarray:
        return self._state(t)[1]

    def acceleration(self, t: float) -> np.ndarray:
        return self._state(t)[2]

    def phase_name(self, t: float) -> str:
        names = ["ground", "climb", "hover", "cruise", "hover", "descent", "hover", "landing", "landed"]
        for (t0, T, *_), name in zip(self.phases, names):
            if t <= t0 + T:
                return name
        return names[-1]

    def heading(self, t: float) -> float:
        return self.yaw

    def rotation(self, t: float) -> np.ndarray:
        """Body-to-world rotation: body +Z along the thrust (a + g + drag), body +X towards the heading."""
        v_air = self.velocity(t) - np.asarray(self.wind, float)
        z = self.acceleration(t) + np.array([0.0, 0.0, G]) + self.drag * np.linalg.norm(v_air) * v_air
        z /= np.linalg.norm(z)
        x_h = np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])
        y = np.cross(z, x_h)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        return np.column_stack([x, y, z])


# ---------------------------------------------------------------------------
# Probe moments: a few still frames that show the whole flight (quadcopter_pbr.py --probe)
# ---------------------------------------------------------------------------

def _time_at(move: SmoothMove, s: float) -> float:
    """Time into ``move`` at which it has covered ``s`` metres (its distance only grows, so bisect)."""
    lo, hi = 0.0, move.duration
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if move.at(mid)[0] < s else (lo, mid)
    return (lo + hi) / 2


def probe_moments(fl) -> dict[str, float]:
    """Named flight times for still-frame probes, in flight order, so each phase can be looked at without
    rendering the whole flight.  ``StraightFlight``: standing on the patio, 1.5 m up, mid-climb, at 15 %, 50 % and
    85 % of the route (at cruise speed, leaning into the air stream), mid-descent, and 1 m above the landing
    grass.  ``Flight`` (the garden tour): every 4 s."""
    if not isinstance(fl, StraightFlight):
        return {f"t{t:g}": float(t) for t in np.arange(2.0, fl.duration, 4.0)}
    ground, climb, cruise, descent, landing = (fl.phases[i] for i in (0, 1, 3, 5, 7))
    out = {"ground": ground[0] + ground[1] / 2,
           "liftoff": climb[0] + _time_at(climb[4], min(1.5, climb[4].distance / 2)),
           "climb": climb[0] + _time_at(climb[4], climb[4].distance / 2)}
    for name, frac in (("cruise_early", 0.15), ("cruise_mid", 0.5), ("cruise_late", 0.85)):
        out[name] = cruise[0] + _time_at(cruise[4], frac * cruise[4].distance)
    out["descent"] = descent[0] + _time_at(descent[4], descent[4].distance / 2)
    out["final"] = landing[0] + _time_at(landing[4], max(0.0, landing[4].distance - 1.0))
    return {k: float(v) for k, v in out.items()}


def parse_probe(labels: list[str], moments: dict[str, float], duration: float) -> dict[str, float]:
    """``--probe`` labels -> {label: time}.  A label is a moment name or a time in seconds (``95`` or ``t95``)
    within the flight; no labels means every moment."""
    if not labels:
        return dict(moments)
    out = {}
    for lab in labels:
        if lab in moments:
            out[lab] = moments[lab]
            continue
        try:
            t = float(lab[1:] if lab.startswith("t") else lab)
        except ValueError:
            raise ValueError(f"unknown probe moment {lab!r}; use a time in seconds or one of {', '.join(moments)}") from None
        if not 0.0 <= t <= duration:
            raise ValueError(f"probe time {t:g} s is outside the flight (0 to {duration:.1f} s)")
        out[f"t{t:g}"] = t
    return out


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
