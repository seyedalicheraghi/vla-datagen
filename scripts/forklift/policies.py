"""Forklift policies for the closed loop (pure Python, no Isaac Sim imports).

A policy turns one observation into one action, every control step (30 Hz):

    obs = {
        "images": {"front_cabin": HxWx3 uint8, "top_left": ..., "top_right": ...},
        "state":  8 floats [x, y, yaw, vx, vy, omega_z, fork_height, grabbed_idx],
        "prompt": str,
    }
    action = (v_x [m/s], omega_z [rad/s], fork_cmd [-1..1])

These are the same fields the LeRobot dataset stores, so a model trained on
that dataset plugs into eval_closed_loop.py by implementing `reset` and `act`.

`ScriptedExpert` is the demonstrator behind scripted_collect.py. It is
*privileged*: it reads the target cargo pose from the episode spec, while a
learned policy has to find the cargo in the camera images.
"""

from __future__ import annotations

import math
from typing import Optional, Protocol

# Truck limits — keep in sync with forklift_env.py and the teleop scripts.
WHEEL_BASE = 1.65                                   # m (ForkliftEnvCfg.wheel_base)
MAX_STEER = 0.6                                     # rad (teleop STEER_ANGLE)
MAX_CURVATURE = math.tan(MAX_STEER) / WHEEL_BASE    # 1/m → ~2.4 m turning radius
PARK_DIST = 1.5         # m, truck root → cargo centre with the tines in the pocket
FORK_POCKET = -0.15     # lift-joint value that puts the tines in the pallet pocket
FORK_TOP = 1.49         # just under the env's 1.5 m lift limit
FORK_DOWN = -0.29       # below the env's release threshold (-0.25)
FORK_M_PER_CMD = 0.16   # lift-joint change per control step at fork_cmd = 1

_ZERO = (0.0, 0.0, 0.0)


class Policy(Protocol):
    def reset(self, episode: dict) -> None: ...
    def act(self, obs: dict) -> tuple[float, float, float]: ...


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class ScriptedExpert:
    """APPROACH → INSERT → LIFT_UP → HOLD → LIFT_DOWN → DONE (or FAILED).

    APPROACH is pure pursuit along the pallet's fork axis: steer at a point
    LOOKAHEAD metres ahead of the truck's projection onto that axis, so the
    truck converges onto the axis and arrives straight. Curvature is capped at
    the truck's real turning radius (no turning on the spot) and speed ramps
    with a fixed acceleration, so the recorded actions look like driving.
    """

    LOOKAHEAD = 1.2       # m — short enough to line up within a 4.5 m run-in
                          # (offline sweep: ≤3.5 cm / ≤1.3° at the pallet)
    V_MAX = 1.5           # m/s
    V_MIN = 0.25          # m/s, creep speed on the final approach
    ACCEL = 0.8           # m/s²
    ARRIVE_TOL = 0.02     # m left along the axis
    STALL_STEPS = 45      # 1.5 s without progress → as far in as it will go
    INSERT_TIMEOUT = 30   # steps waiting for the env to register the grab
    LIFT_CMD = 0.15       # ≈0.7 m/s fork speed (real trucks lift 0.3–0.6 m/s)
    HOLD_STEPS = 30       # ~1 s at 30 Hz

    def __init__(self, dt: float = 1.0 / 30.0):
        self.dt = dt
        self.reset({"target": {"cargo_xyz": [0.0, 0.0, 0.0], "cargo_yaw_rad": 0.0}})

    def reset(self, episode: dict) -> None:
        tgt = episode["target"]
        cx, cy = float(tgt["cargo_xyz"][0]), float(tgt["cargo_xyz"][1])
        self.axis_yaw = float(tgt["cargo_yaw_rad"])
        self.ux, self.uy = math.cos(self.axis_yaw), math.sin(self.axis_yaw)
        # Pre-grasp point: PARK_DIST behind the cargo on its fork axis.
        self.px = cx - PARK_DIST * self.ux
        self.py = cy - PARK_DIST * self.uy

        self.state = "APPROACH"
        self.v = 0.0
        self._best_s = math.inf
        self._stall = 0
        self._wait = 0
        self._hold = 0
        self.approach_steps = 0
        self.insert_error: Optional[dict] = None   # alignment when the forks go in

        self.n_grabs = 0
        self.n_drops = 0
        self._prev_grabbed = -1
        self.last_event: Optional[str] = None      # "grab" / "drop" on that step
        self.fork_max_seen = -math.inf
        self.fork_min_seen = math.inf

    @property
    def done(self) -> bool:
        return self.state in ("DONE", "FAILED")

    def act(self, obs: dict) -> tuple[float, float, float]:
        x, y, yaw, _vx, _vy, _wz, fork_h, gidx = (float(v) for v in obs["state"])
        grabbed = gidx >= 0

        self.last_event = None
        if grabbed and self._prev_grabbed < 0:
            self.n_grabs += 1
            self.last_event = "grab"
        elif not grabbed and self._prev_grabbed >= 0:
            self.n_drops += 1
            self.last_event = "drop"
        self._prev_grabbed = int(gidx)
        self.fork_max_seen = max(self.fork_max_seen, fork_h)
        self.fork_min_seen = min(self.fork_min_seen, fork_h)

        if self.state == "APPROACH":
            return self._approach(x, y, yaw, fork_h)

        if self.state == "INSERT":
            if grabbed:
                self.state = "LIFT_UP"
                return _ZERO
            self._wait += 1
            if self._wait > self.INSERT_TIMEOUT:
                self.state = "FAILED"
                return _ZERO
            return (0.0, 0.0, 0.05)    # fork_cmd > 0.01 triggers the env's grab check

        if self.state == "LIFT_UP":
            if fork_h >= FORK_TOP:
                self.state = "HOLD"
                return _ZERO
            return (0.0, 0.0, self.LIFT_CMD)

        if self.state == "HOLD":
            self._hold += 1
            if self._hold >= self.HOLD_STEPS:
                self.state = "LIFT_DOWN"
            return _ZERO

        if self.state == "LIFT_DOWN":
            if fork_h <= FORK_DOWN:
                self.state = "DONE"
                return _ZERO
            return (0.0, 0.0, -self.LIFT_CMD)

        return _ZERO                   # DONE / FAILED

    def _approach(self, x: float, y: float, yaw: float,
                  fork_h: float) -> tuple[float, float, float]:
        self.approach_steps += 1
        # Distance still to go along the axis, and sideways offset (+ = left).
        s = (self.px - x) * self.ux + (self.py - y) * self.uy
        lateral = self.ux * (y - self.py) - self.uy * (x - self.px)

        if s < self._best_s - 0.01:
            self._best_s, self._stall = s, 0
        else:
            self._stall += 1
        if s <= self.ARRIVE_TOL or self._stall >= self.STALL_STEPS:
            self.insert_error = {
                "lateral_m": lateral,
                "heading_deg": math.degrees(wrap_angle(yaw - self.axis_yaw)),
                "remaining_m": s,
            }
            self.state = "INSERT"
            self.v = 0.0
            return _ZERO

        # Carrot: LOOKAHEAD ahead of the truck's projection onto the axis.
        tx = self.px - (s - self.LOOKAHEAD) * self.ux
        ty = self.py - (s - self.LOOKAHEAD) * self.uy
        alpha = wrap_angle(math.atan2(ty - y, tx - x) - yaw)
        dist = max(math.hypot(tx - x, ty - y), 1e-6)
        curvature = _clamp(2.0 * math.sin(alpha) / dist, -MAX_CURVATURE, MAX_CURVATURE)

        # Fastest speed that can still stop at the pre-grasp point, ramped up
        # at ACCEL from the previous step.
        v_goal = _clamp(math.sqrt(2.0 * self.ACCEL * max(s, 0.0)), self.V_MIN, self.V_MAX)
        self.v = min(v_goal, self.v + self.ACCEL * self.dt)

        # Keep the tines at pocket height so they slide into the pallet. Only
        # ever lower while driving: the env grabs any pallet in reach as soon
        # as fork_cmd > 0.01, which would pick the cargo up mid-approach.
        fork_cmd = _clamp((FORK_POCKET - fork_h) / FORK_M_PER_CMD, -1.0, 0.0)
        return (self.v, self.v * curvature, fork_cmd)
