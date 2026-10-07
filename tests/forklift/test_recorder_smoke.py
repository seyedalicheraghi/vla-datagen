"""Smoke tests for randomization + prompt templating used by record_lerobot.py.

These do NOT spin up Isaac Sim — they verify the components in isolation:
  * SpawnRandomizer / CargoRandomizer determinism + keepout
  * prompt_builder quantization round-trip
  * 3 episodes with 3 different seeds yield 3 unique rendered prompts
  * Cargo positions across episodes differ by > 0.1 m on average

A separate integration test (skipped by default) would launch the sim and
verify tasks.jsonl + extras_episodes.jsonl on disk.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

# Make scripts/forklift importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "forklift"))

from randomizers import SpawnRandomizer, CargoRandomizer, DistractorRandomizer, parse_region
import prompt_builder


# ── Randomizer determinism ─────────────────────────────────────────────────

def test_spawn_randomizer_determinism():
    a = SpawnRandomizer(region_xyxy=(-2, -3, 2, 3), seed=42)
    b = SpawnRandomizer(region_xyxy=(-2, -3, 2, 3), seed=42)
    for _ in range(5):
        assert a.sample() == b.sample()


def test_spawn_randomizer_different_seeds_diverge():
    a = SpawnRandomizer(region_xyxy=(-2, -3, 2, 3), seed=1).sample()
    b = SpawnRandomizer(region_xyxy=(-2, -3, 2, 3), seed=2).sample()
    assert a != b


def test_cargo_randomizer_keepout():
    cr = CargoRandomizer(
        region_xyxy=(5, -3, 11, 3), seed=0,
        keepout_from_forklift=1.0, keepout_between_cargo=0.8,
    )
    fl = (8.5, 0.0)
    positions = cr.sample(n_pallets=3, forklift_xy=fl)
    assert len(positions) == 3
    for x, y, _ in positions:
        assert math.hypot(x - fl[0], y - fl[1]) >= 1.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            assert math.hypot(dx, dy) >= 0.8


def test_distractor_randomizer_keepout_and_count():
    dr = DistractorRandomizer(region_xyxy=(0, -8, 15, 8), seed=7, keepout=1.5)
    blocked = [(8.5, 0.0), (10.0, 0.0)]
    poses = dr.sample(n=8, blocked_xys=blocked)
    assert len(poses) == 8
    for x, y, z, _ in poses:
        assert z == 0.0
        for bx, by in blocked:
            assert math.hypot(x - bx, y - by) >= 1.5
    # mutual keepout
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            assert math.hypot(poses[i][0] - poses[j][0],
                              poses[i][1] - poses[j][1]) >= 1.5


def test_distractor_randomizer_zero_n_returns_empty():
    dr = DistractorRandomizer(region_xyxy=(0, -8, 15, 8), seed=0)
    assert dr.sample(n=0, blocked_xys=[(0.0, 0.0)]) == []


def test_distractor_randomizer_determinism():
    a = DistractorRandomizer(region_xyxy=(0, -8, 15, 8), seed=42).sample(5, [(8.5, 0.0)])
    b = DistractorRandomizer(region_xyxy=(0, -8, 15, 8), seed=42).sample(5, [(8.5, 0.0)])
    assert a == b


def test_parse_region():
    assert parse_region("5,-3,11,3") == (5.0, -3.0, 11.0, 3.0)
    with pytest.raises(ValueError):
        parse_region("5,-3,11")


# ── Prompt builder ─────────────────────────────────────────────────────────

def test_quantize_xy_5cm_grid():
    assert prompt_builder.quantize_xy(0.034) == 0.05
    assert prompt_builder.quantize_xy(0.024) == 0.0
    assert prompt_builder.quantize_xy(-0.07) == -0.05


def test_quantize_yaw_5deg_grid():
    assert prompt_builder.quantize_yaw_deg(math.radians(2.4)) == 0.0
    assert prompt_builder.quantize_yaw_deg(math.radians(2.6)) == 5.0
    assert prompt_builder.quantize_yaw_deg(math.radians(-87.5)) in (-85.0, -90.0)


def test_render_default_template():
    s, p = prompt_builder.render(
        prompt_builder.DEFAULT_TEMPLATE,
        cargo_xyz=(7.13, -2.04, 0.00),
        cargo_yaw_rad=0.0,
        forklift_xyz=(8.50, 0.00, 0.00),
        forklift_yaw_rad=0.0,
    )
    assert "lift the cargo at coordinate" in s
    assert s.startswith("lift the cargo at coordinate (7.15, -2.05, 0.00)")
    # Forklift placeholders populated when include_forklift=True
    assert "fx" in p and "fyaw_deg" in p


def test_render_no_forklift_placeholders():
    template = "lift cargo at ({cx:.2f}, {cy:.2f})"
    s, _ = prompt_builder.render(
        template,
        cargo_xyz=(1.0, 2.0, 0.0),
        cargo_yaw_rad=0.0,
        include_forklift=False,
    )
    assert s == "lift cargo at (1.00, 2.00)"


# ── End-to-end episode-level: 3 seeds → 3 unique prompts ──────────────────

def _simulate_episode(seed: int) -> tuple[str, tuple[float, float]]:
    spawn = SpawnRandomizer(region_xyxy=(-2, -3, 2, 3), seed=seed)
    cargo = CargoRandomizer(
        region_xyxy=(5, -3, 11, 3), seed=seed + 1,
        keepout_from_forklift=1.0, keepout_between_cargo=0.8,
    )
    sx, sy, syaw = spawn.sample()
    cargo_xy_yaw = cargo.sample(n_pallets=1, forklift_xy=(sx, sy))
    cx, cy, cyaw = cargo_xy_yaw[0]
    rendered, _ = prompt_builder.render(
        template=prompt_builder.DEFAULT_TEMPLATE,
        cargo_xyz=(cx, cy, 0.0),
        cargo_yaw_rad=cyaw,
        forklift_xyz=(sx, sy, 0.0),
        forklift_yaw_rad=syaw,
    )
    return rendered, (cx, cy)


def test_three_seeds_three_unique_prompts():
    results = [_simulate_episode(s) for s in (0, 1, 2)]
    prompts = {r[0] for r in results}
    assert len(prompts) == 3, f"expected 3 unique prompts, got {prompts}"


def test_cargo_xy_diverges_across_seeds_by_0_1m():
    results = [_simulate_episode(s) for s in (0, 1, 2)]
    xys = [r[1] for r in results]
    pairs = [(0, 1), (0, 2), (1, 2)]
    avg_dist = sum(math.hypot(xys[a][0] - xys[b][0],
                              xys[a][1] - xys[b][1])
                   for a, b in pairs) / len(pairs)
    assert avg_dist > 0.1, f"avg pairwise cargo distance too small: {avg_dist:.3f} m"


# ── LeRobot round trip (the same calls the recorders make) ────────────────

def test_lerobot_roundtrip_discard_and_finalize(tmp_path):
    """create → add+save → add+clear (a discarded attempt) → add+save →
    finalize → reload. The discarded frames must not leak into the next
    episode, and the dataset must load (LeRobot v3 needs finalize())."""
    lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
    import numpy as np
    from PIL import Image

    cams = ("front_cabin", "top_left", "top_right")
    features = {
        f"observation.images.{c}": {"dtype": "image", "shape": (224, 224, 3),
                                    "names": ["height", "width", "channel"]}
        for c in cams
    }
    features["observation.state"] = {"dtype": "float32", "shape": (8,), "names": ["state"]}
    features["action"] = {"dtype": "float32", "shape": (5,), "names": ["action"]}
    root = tmp_path / "forklift_ds"
    ds = lerobot_dataset.LeRobotDataset.create(
        repo_id="test/forklift", fps=30, root=str(root), robot_type="forklift",
        features=features, use_videos=True, image_writer_threads=1)

    def add(n: int, task: str, marker: int) -> None:
        for _ in range(n):
            frame = {f"observation.images.{c}":
                     Image.fromarray(np.full((224, 224, 3), marker, np.uint8)) for c in cams}
            frame["observation.state"] = np.full(8, marker, np.float32)
            frame["action"] = np.zeros(5, np.float32)
            frame["task"] = task
            ds.add_frame(frame)

    add(12, "task A", 10)
    ds.save_episode()
    add(7, "discarded attempt", 99)
    ds.clear_episode_buffer()
    add(15, "task B", 20)
    ds.save_episode()
    ds.finalize()

    loaded = lerobot_dataset.LeRobotDataset("test/forklift", root=str(root))
    assert loaded.meta.total_episodes == 2
    assert len(loaded) == 27
    assert loaded.fps == 30
    assert not any("lidar" in k for k in loaded.meta.features)
    assert loaded.meta.get_task_index("task B") is not None
    assert loaded.meta.get_task_index("discarded attempt") is None
    markers = {int(loaded[i]["observation.state"][0]) for i in range(len(loaded))}
    assert markers == {10, 20}, f"discarded frames leaked: {markers}"
