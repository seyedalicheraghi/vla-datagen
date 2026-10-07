"""Spawn, cargo and distractor randomizers for forklift recording.

All produce env-local (x, y, yaw_rad) tuples. For a single-env recorder,
env-local equals world coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SpawnRandomizer:
    region_xyxy: tuple[float, float, float, float]   # xmin, ymin, xmax, ymax
    seed: int = 0
    yaw_range_rad: tuple[float, float] = (-math.pi, math.pi)
    _rng: Optional[np.random.Generator] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def sample(self) -> tuple[float, float, float]:
        xmin, ymin, xmax, ymax = self.region_xyxy
        x = float(self._rng.uniform(xmin, xmax))
        y = float(self._rng.uniform(ymin, ymax))
        yaw = float(self._rng.uniform(*self.yaw_range_rad))
        return x, y, yaw


@dataclass
class CargoRandomizer:
    region_xyxy: tuple[float, float, float, float]
    seed: int = 0
    keepout_from_forklift: float = 1.0
    keepout_between_cargo: float = 0.8
    yaw_range_rad: tuple[float, float] = (-math.pi, math.pi)
    _rng: Optional[np.random.Generator] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def sample(
        self,
        n_pallets: int,
        forklift_xy: tuple[float, float],
    ) -> list[tuple[float, float, float]]:
        xmin, ymin, xmax, ymax = self.region_xyxy
        fx, fy = forklift_xy
        positions: list[tuple[float, float, float]] = []
        attempts = 0
        max_attempts = 5000
        while len(positions) < n_pallets and attempts < max_attempts:
            attempts += 1
            x = float(self._rng.uniform(xmin, xmax))
            y = float(self._rng.uniform(ymin, ymax))
            if math.hypot(x - fx, y - fy) < self.keepout_from_forklift:
                continue
            if any(math.hypot(x - px, y - py) < self.keepout_between_cargo
                   for px, py, _ in positions):
                continue
            yaw = float(self._rng.uniform(*self.yaw_range_rad))
            positions.append((x, y, yaw))
        if len(positions) < n_pallets:
            raise RuntimeError(
                f"CargoRandomizer: only placed {len(positions)}/{n_pallets} "
                f"pallets after {max_attempts} attempts. Region "
                f"{self.region_xyxy} too small or keepouts too aggressive."
            )
        return positions


@dataclass
class DistractorRandomizer:
    """Place visual distractor cargos around the warehouse.

    Distractors share the same asset as the target cargo so the policy
    must use the language prompt to disambiguate. They are NOT in the
    interactable pallet list, so the grab logic never sees them.
    """
    region_xyxy: tuple[float, float, float, float]
    seed: int = 0
    keepout: float = 1.5
    yaw_range_rad: tuple[float, float] = (-math.pi, math.pi)
    z_ground: float = 0.0
    _rng: Optional[np.random.Generator] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def sample(
        self,
        n: int,
        blocked_xys: list[tuple[float, float]],
    ) -> list[tuple[float, float, float, float]]:
        """Return n (x, y, z, yaw) poses, each ≥ keepout from every blocked
        XY and from previously-placed distractors."""
        if n <= 0:
            return []
        xmin, ymin, xmax, ymax = self.region_xyxy
        placed: list[tuple[float, float, float, float]] = []
        attempts = 0
        max_attempts = max(2000, n * 200)
        while len(placed) < n and attempts < max_attempts:
            attempts += 1
            x = float(self._rng.uniform(xmin, xmax))
            y = float(self._rng.uniform(ymin, ymax))
            if any(math.hypot(x - bx, y - by) < self.keepout
                   for bx, by in blocked_xys):
                continue
            if any(math.hypot(x - px, y - py) < self.keepout
                   for px, py, _, _ in placed):
                continue
            yaw = float(self._rng.uniform(*self.yaw_range_rad))
            placed.append((x, y, self.z_ground, yaw))
        return placed


@dataclass
class ApproachSpawnRandomizer:
    """Forklift start pose for drive-to-cargo episodes.

    The pre-grasp point sits `park_dist` behind the cargo on its fork axis
    (where the tines are in the pocket). The forklift starts `along_range`
    metres further back on that axis, shifted sideways by up to `lateral_max`,
    with its heading perturbed by up to `yaw_noise_rad` from the approach
    direction — so every episode needs a short drive and a correction.
    """
    along_range: tuple[float, float] = (4.5, 6.5)     # → 6–8 m from the cargo
    lateral_max: float = 1.2
    yaw_noise_rad: float = math.radians(15.0)
    park_dist: float = 1.5
    seed: int = 0
    _rng: Optional[np.random.Generator] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def sample(self, cargo_xy: tuple[float, float],
               cargo_yaw: float) -> tuple[float, float, float]:
        cx, cy = cargo_xy
        ux, uy = math.cos(cargo_yaw), math.sin(cargo_yaw)   # approach direction
        back = self.park_dist + float(self._rng.uniform(*self.along_range))
        side = float(self._rng.uniform(-self.lateral_max, self.lateral_max))
        x = cx - back * ux - side * uy
        y = cy - back * uy + side * ux
        yaw = cargo_yaw + float(self._rng.uniform(-self.yaw_noise_rad,
                                                  self.yaw_noise_rad))
        return x, y, yaw


def corridor_points(start_xy: tuple[float, float], start_yaw: float,
                    goal_xy: tuple[float, float], spacing: float = 0.5,
                    tail: float = 2.0) -> list[tuple[float, float]]:
    """Points along the straight start→goal segment, plus `tail` metres
    behind the start (the truck's body), for distractor keep-out."""
    sx, sy = start_xy
    gx, gy = goal_xy
    n = max(1, int(math.hypot(gx - sx, gy - sy) / spacing))
    pts = [(sx + (gx - sx) * i / n, sy + (gy - sy) * i / n) for i in range(n + 1)]
    pts.append((sx - tail * math.cos(start_yaw), sy - tail * math.sin(start_yaw)))
    return pts


def sample_scene(seed: int, cargo_region: tuple[float, float, float, float],
                 drive: bool = True,
                 along_range: tuple[float, float] = (4.5, 6.5),
                 lateral_max: float = 1.2,
                 yaw_noise_deg: float = 15.0,
                 park_dist: float = 1.5) -> dict:
    """One episode's layout: target cargo, forklift start, distractor keep-out.

    Cargo yaw snaps to {0, π}: the two-way pallet only takes forks along its
    long axis, and the env's carry logic assumes that alignment. With
    `drive=False` the truck starts parked with its tines already in the pocket.
    Shared by scripted_collect.py and eval_closed_loop.py so a seed always
    means the same scene.
    """
    cargo = CargoRandomizer(region_xyxy=cargo_region, seed=seed,
                            keepout_from_forklift=2.0, keepout_between_cargo=1.0)
    cx, cy, cyaw = cargo.sample(1, (-1e6, -1e6))[0]
    cyaw = 0.0 if math.cos(cyaw) >= 0.0 else math.pi
    if drive:
        spawn = ApproachSpawnRandomizer(
            along_range=along_range, lateral_max=lateral_max,
            yaw_noise_rad=math.radians(yaw_noise_deg), park_dist=park_dist,
            seed=seed + 20_000,        # disjoint from the cargo / distractor streams
        ).sample((cx, cy), cyaw)
    else:
        spawn = (cx - park_dist * math.cos(cyaw), cy - park_dist * math.sin(cyaw), cyaw)
    return {
        "cargo": (cx, cy, cyaw),
        "spawn": spawn,
        "keepout": corridor_points(spawn[:2], spawn[2], (cx, cy)),
    }


def parse_region(s: str) -> tuple[float, float, float, float]:
    """Parse 'xmin,ymin,xmax,ymax' into a 4-tuple."""
    parts = [float(p) for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"region must be 'xmin,ymin,xmax,ymax', got {s!r}")
    return tuple(parts)  # type: ignore[return-value]
