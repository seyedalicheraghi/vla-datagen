"""Offline tests for the scripted expert and the drive-to-cargo scene sampler.

No Isaac Sim: the expert is rolled out in a 30 Hz kinematic model that
integrates heading and position the same way forklift_env.py does (heading
first, then position along the new heading), with the env's fork limits,
grab trigger (fork_cmd > 0.01) and release threshold (lift joint < -0.25).
"""

from __future__ import annotations

import math
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "forklift"))

from policies import FORK_POCKET, MAX_CURVATURE, PARK_DIST, ScriptedExpert, wrap_angle
from randomizers import ApproachSpawnRandomizer, corridor_points, sample_scene

CARGO_REGION = (-3.0, -4.0, 3.0, 4.0)     # scripted_collect.py default
DT = 1.0 / 30.0
WALL = 16.0                               # warehouse half-size (forklift_env._WAREHOUSE_HALF)


def _rollout(scene: dict, max_steps: int = 1500):
    cx, cy, cyaw = scene["cargo"]
    x, y, yaw = scene["spawn"]
    policy = ScriptedExpert(DT)
    policy.reset({"target": {"cargo_xyz": [cx, cy, 0.0], "cargo_yaw_rad": cyaw}})
    fork, grabbed = FORK_POCKET, -1
    max_kappa = 0.0
    approach_grab = False
    for _ in range(max_steps):
        v, w, f = policy.act({"state": [x, y, yaw, 0.0, 0.0, 0.0, fork, grabbed]})
        if v > 1e-6:
            max_kappa = max(max_kappa, abs(w / v))
        yaw += w * DT
        x += v * math.cos(yaw) * DT
        y += v * math.sin(yaw) * DT
        fork = min(1.5, max(-0.3, fork + f * 0.16))
        if grabbed < 0 and f > 0.01:
            grabbed = 0
            approach_grab = approach_grab or policy.state == "APPROACH"
        elif grabbed >= 0 and fork < -0.25:
            grabbed = -1
        if policy.done:
            break
    return policy, max_kappa, approach_grab


@pytest.mark.parametrize("seed", range(200))
def test_expert_drives_in_straight_and_completes(seed):
    policy, max_kappa, approach_grab = _rollout(sample_scene(seed, CARGO_REGION))
    assert policy.state == "DONE", f"ended in {policy.state}"
    assert not approach_grab, "forks raised while still driving"
    err = policy.insert_error
    assert abs(err["lateral_m"]) < 0.05, err
    assert abs(err["heading_deg"]) < 2.0, err
    assert max_kappa <= MAX_CURVATURE + 1e-6, "tighter than the truck can turn"
    assert policy.n_grabs == 1 and policy.n_drops == 1
    assert policy.fork_max_seen >= 1.49


def test_parked_start_skips_the_drive():
    policy, _, _ = _rollout(sample_scene(7, CARGO_REGION, drive=False))
    assert policy.state == "DONE"
    assert policy.approach_steps <= 2


def test_spawns_stay_inside_the_warehouse():
    for seed in range(500):
        sx, sy, syaw = sample_scene(seed, CARGO_REGION)["spawn"]
        # Root plus ~3 m of truck body behind it must stay inside the walls.
        for d in (0.0, -3.0):
            px, py = sx + d * math.cos(syaw), sy + d * math.sin(syaw)
            assert abs(px) < WALL - 0.5 and abs(py) < WALL - 0.5, (seed, px, py)


def test_spawn_is_behind_the_pregrasp_point_and_roughly_facing_it():
    rng = ApproachSpawnRandomizer(seed=3)
    for _ in range(200):
        sx, sy, syaw = rng.sample((0.0, 0.0), math.pi)
        # Cargo at origin facing -x → pre-grasp at (+1.5, 0), start further +x.
        assert sx >= PARK_DIST + 4.5 - 1e-9
        assert abs(sy) <= 1.2 + 1e-9
        assert abs(wrap_angle(syaw - math.pi)) <= math.radians(15) + 1e-9


def test_scene_is_deterministic_per_seed():
    assert sample_scene(11, CARGO_REGION) == sample_scene(11, CARGO_REGION)
    assert sample_scene(11, CARGO_REGION) != sample_scene(12, CARGO_REGION)


def test_cargo_yaw_snaps_to_fork_axis():
    for seed in range(100):
        assert sample_scene(seed, CARGO_REGION)["cargo"][2] in (0.0, math.pi)


def test_corridor_covers_path_and_truck_body():
    pts = corridor_points((0.0, 0.0), 0.0, (4.0, 0.0), spacing=0.5, tail=2.0)
    assert pts[0] == (0.0, 0.0) and pts[-2] == (4.0, 0.0)
    assert pts[-1] == (-2.0, 0.0)                     # behind the start
    gaps = [math.dist(a, b) for a, b in zip(pts[:-2], pts[1:-1])]
    assert max(gaps) <= 0.5 + 1e-9
