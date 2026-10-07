"""Quantize spawn/cargo coordinates and render natural-language prompts.

Quantization snaps positions to a 5 cm grid and yaw to 5° before template
rendering, so near-identical setups share the same prompt string and the
same LeRobot task_index.
"""

from __future__ import annotations

import math
from typing import Optional


def quantize_xy(v: float, grid_m: float = 0.05) -> float:
    return round(v / grid_m) * grid_m


def quantize_yaw_deg(yaw_rad: float, step_deg: float = 5.0) -> float:
    deg = math.degrees(yaw_rad)
    return round(deg / step_deg) * step_deg


def bearing_and_distance(
    forklift_xy: tuple[float, float],
    forklift_yaw_rad: float,
    cargo_xy: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Return (dx, dy, distance_m, bearing_deg) — bearing in forklift body frame."""
    fx, fy = forklift_xy
    cx, cy = cargo_xy
    dx = cx - fx
    dy = cy - fy
    distance = math.hypot(dx, dy)
    world_bearing = math.atan2(dy, dx)
    body_bearing = world_bearing - forklift_yaw_rad
    body_bearing = (body_bearing + math.pi) % (2 * math.pi) - math.pi
    return dx, dy, distance, math.degrees(body_bearing)


def build_placeholders(
    cargo_xyz: tuple[float, float, float],
    cargo_yaw_rad: float,
    forklift_xyz: Optional[tuple[float, float, float]],
    forklift_yaw_rad: Optional[float],
    pallet_idx: int,
    include_forklift: bool,
) -> dict:
    """Quantize all coordinates and produce the placeholder dict."""
    cx, cy, cz = (quantize_xy(v) for v in cargo_xyz)
    cyaw_deg = quantize_yaw_deg(cargo_yaw_rad)

    p: dict = {
        "cx": cx, "cy": cy, "cz": cz,
        "cyaw_deg": cyaw_deg,
        "pallet_idx": pallet_idx,
    }

    if include_forklift and forklift_xyz is not None and forklift_yaw_rad is not None:
        fx, fy, fz = (quantize_xy(v) for v in forklift_xyz)
        fyaw_deg = quantize_yaw_deg(forklift_yaw_rad)
        # Compute spawn-time bearing/distance using the QUANTIZED forklift yaw
        # so the prompt is self-consistent with what's rendered.
        dx, dy, dist, bearing_deg = bearing_and_distance(
            (fx, fy), math.radians(fyaw_deg), (cx, cy)
        )
        p.update({
            "fx": fx, "fy": fy, "fz": fz,
            "fyaw_deg": fyaw_deg,
            "dx": quantize_xy(dx),
            "dy": quantize_xy(dy),
            "distance_m": quantize_xy(dist),
            "bearing_deg": quantize_yaw_deg(math.radians(bearing_deg)),
        })

    return p


def render(
    template: str,
    cargo_xyz: tuple[float, float, float],
    cargo_yaw_rad: float,
    forklift_xyz: Optional[tuple[float, float, float]] = None,
    forklift_yaw_rad: Optional[float] = None,
    pallet_idx: int = 0,
    include_forklift: bool = True,
) -> tuple[str, dict]:
    """Quantize and render. Returns (rendered_string, placeholders_dict).

    Raises KeyError if template references a placeholder we did not provide
    (e.g. {fx} when include_forklift=False).
    """
    placeholders = build_placeholders(
        cargo_xyz=cargo_xyz,
        cargo_yaw_rad=cargo_yaw_rad,
        forklift_xyz=forklift_xyz,
        forklift_yaw_rad=forklift_yaw_rad,
        pallet_idx=pallet_idx,
        include_forklift=include_forklift,
    )
    rendered = template.format(**placeholders)
    return rendered, placeholders


DEFAULT_TEMPLATE = (
    "lift the cargo at coordinate ({cx:.2f}, {cy:.2f}, {cz:.2f}) "
    "from forklift at ({fx:.2f}, {fy:.2f}, yaw {fyaw_deg:.0f} deg)"
)

# Scripted collection (scripted_collect.py) and closed-loop testing
# (eval_closed_loop.py) must use the same wording, or a language-conditioned
# policy is tested on prompts it never saw.
DRIVE_LIFT_TEMPLATE = (
    "drive to the cargo at coordinate ({cx:.2f}, {cy:.2f}, {cz:.2f}) "
    "from forklift at ({fx:.2f}, {fy:.2f}, yaw {fyaw_deg:.0f} deg), "
    "lift it to maximum fork height, then lower it"
)
LIFT_LOWER_TEMPLATE = (
    "lift the cargo at coordinate ({cx:.2f}, {cy:.2f}, {cz:.2f}) "
    "from forklift at ({fx:.2f}, {fy:.2f}, yaw {fyaw_deg:.0f} deg) "
    "to maximum fork height, then lower it"
)
