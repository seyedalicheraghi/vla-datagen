"""Spawn and cargo position randomizers for forklift recording.

Both produce env-local (x, y, yaw_rad) tuples. For a single-env recorder,
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


def parse_region(s: str) -> tuple[float, float, float, float]:
    """Parse 'xmin,ymin,xmax,ymax' into a 4-tuple."""
    parts = [float(p) for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"region must be 'xmin,ymin,xmax,ymax', got {s!r}")
    return tuple(parts)  # type: ignore[return-value]
