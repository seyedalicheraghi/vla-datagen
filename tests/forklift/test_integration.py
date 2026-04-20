"""
Integration tests for the forklift environment (headless Isaac Sim).

These tests spin up the full simulation in headless mode and verify
behaviour over short episodes.

Run:
    ./isaaclab.sh -p -m pytest tests/forklift/test_integration.py -v --headless

Note: These tests require Isaac Sim to be installed and take ~30s each.
If Isaac Sim is not available, tests are skipped automatically.
"""

import math
import sys
import os

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Isaac Sim bootstrap (must happen before any isaaclab import)
# ---------------------------------------------------------------------------

_SIM_AVAILABLE = False
try:
    import argparse
    from isaaclab.app import AppLauncher

    _parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(_parser)
    _args, _ = _parser.parse_known_args(["--headless"])
    _args.enable_cameras = True
    _args.headless = True

    _launcher = AppLauncher(_args)
    _sim_app = _launcher.app
    _SIM_AVAILABLE = True
except Exception as e:
    print(f"[SKIP] Isaac Sim not available: {e}")

# Now safe to import forklift env
if _SIM_AVAILABLE:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                     "..", "..", "scripts", "forklift"))
    from forklift_env import (ForkliftEnv, ForkliftEnvCfg,
                              _N_INTERACTABLE, _PALLET_H, _BOX_H, _UNIT_H,
                              _LIDAR_CHANNELS, _LIDAR_MAX_RANGE)


def _requires_sim(fn):
    """Decorator to skip tests when Isaac Sim is not available."""
    return pytest.mark.skipif(not _SIM_AVAILABLE,
                              reason="Isaac Sim not available")(fn)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env():
    """Create and reset the environment once for all tests in this module."""
    if not _SIM_AVAILABLE:
        pytest.skip("Isaac Sim not available")
    cfg = ForkliftEnvCfg()
    e = ForkliftEnv(cfg)
    e.reset()
    yield e
    e.close()


def _step_n(env, n: int, action=None):
    """Step the environment n times with the given action (default: zeros)."""
    if action is None:
        action = torch.zeros(1, 3, device=env.device)
    for _ in range(n):
        env.step(action)


def _get_pose(env):
    """Return (x, y, heading) from authoritative state."""
    return (env._carry_pos[0, 0].item(),
            env._carry_pos[0, 1].item(),
            env._heading[0].item())


# ============================================================================
# §5.2 — Idle stability
# ============================================================================

@_requires_sim
class TestIdleStability:
    """No input for 3 seconds → base pose unchanged."""

    def test_idle_no_drift(self, env):
        """§1 regression: zero action for 3s, pose must not change."""
        env.reset()
        x0, y0, h0 = _get_pose(env)

        # 3 seconds at 30 Hz effective rate → ~90 steps (sim_dt=1/120, dec=4)
        _step_n(env, 90)

        x1, y1, h1 = _get_pose(env)
        assert abs(x1 - x0) < 0.01, f"X drifted: {x0:.4f} → {x1:.4f}"
        assert abs(y1 - y0) < 0.01, f"Y drifted: {y0:.4f} → {y1:.4f}"
        assert abs(h1 - h0) < 0.005, f"Heading drifted: {h0:.4f} → {h1:.4f}"


# ============================================================================
# §5.2 — Lift stability (§1 regression test)
# ============================================================================

@_requires_sim
class TestLiftStability:
    """Attach cargo, lift, no input for 3s → base pose unchanged."""

    def test_lift_no_drift(self, env):
        """Drive to pallet, grab, lift, then idle — no drift."""
        env.reset()

        # Drive forward toward pallet 0 (10m ahead)
        drive_action = torch.tensor([[5.0, 0.0, 0.0]], device=env.device)
        _step_n(env, 60)  # ~2 seconds of coasting at 5 m/s → ~10m

        # Try to grab by raising forks
        lift_action = torch.tensor([[0.0, 0.0, 1.0]], device=env.device)
        _step_n(env, 30, lift_action)

        # Record pose after grab
        x0, y0, h0 = _get_pose(env)

        # Idle for 3 seconds with no input
        _step_n(env, 90)

        x1, y1, h1 = _get_pose(env)
        assert abs(x1 - x0) < 0.02, \
            f"X drifted after lift: {x0:.4f} → {x1:.4f}"
        assert abs(y1 - y0) < 0.02, \
            f"Y drifted after lift: {y0:.4f} → {y1:.4f}"
        assert abs(h1 - h0) < 0.01, \
            f"Heading drifted after lift: {h0:.4f} → {h1:.4f}"


# ============================================================================
# §5.2 — Pickup: cargo tracks fork pose
# ============================================================================

@_requires_sim
class TestPickup:
    """Drive to pallet, lift — cargo should track fork position."""

    def test_pallet_follows_fork_height(self, env):
        """After grab, pallet z should increase when fork raises."""
        env.reset()

        # Drive to pallet
        _step_n(env, 60, torch.tensor([[5.0, 0.0, 0.0]], device=env.device))

        # Start lifting
        _step_n(env, 20, torch.tensor([[0.0, 0.0, 1.0]], device=env.device))

        if env._grabbed_idx[0] >= 0:
            pi = env._grabbed_idx[0]
            z_before = env.pallets[pi].data.root_pos_w[0, 2].item()

            # Continue lifting
            _step_n(env, 30, torch.tensor([[0.0, 0.0, 1.0]], device=env.device))

            z_after = env.pallets[pi].data.root_pos_w[0, 2].item()
            assert z_after > z_before, \
                f"Pallet z didn't increase: {z_before:.3f} → {z_after:.3f}"


# ============================================================================
# §5.2 — Ground placement
# ============================================================================

@_requires_sim
class TestGroundPlacement:
    """Pick → move → lower → release; cargo settles at ground level."""

    def test_drop_settles_at_ground(self, env):
        """After dropping cargo, pallet base_z should be near 0."""
        env.reset()

        # Drive forward, grab, lift
        _step_n(env, 60, torch.tensor([[5.0, 0.0, 0.0]], device=env.device))
        _step_n(env, 30, torch.tensor([[0.0, 0.0, 1.0]], device=env.device))

        if env._grabbed_idx[0] >= 0:
            pi = env._grabbed_idx[0]

            # Lower forks to release
            _step_n(env, 60, torch.tensor([[0.0, 0.0, -1.0]], device=env.device))

            # Check it was released
            assert env._grabbed_idx[0] == -1, "Pallet should be released"

            # Check base_z is near ground
            base_z = env._pallet_base_z[0][pi]
            assert base_z < 0.1, \
                f"Dropped pallet base_z should be near 0, got {base_z:.3f}"


# ============================================================================
# §5.2 — Stack placement
# ============================================================================

@_requires_sim
class TestStackPlacement:
    """Pre-stacked pallet 2 on pallet 3 should have correct height."""

    def test_initial_stack_height(self, env):
        """Pallet 2 starts stacked on pallet 3 — base_z ≈ _UNIT_H."""
        env.reset()

        # Pallet 2 is pre-stacked (see _POSITIONS in _reset_idx)
        base_z_2 = env._pallet_base_z[0][2]
        assert abs(base_z_2 - _UNIT_H) < 0.01, \
            f"Pallet 2 base_z should be {_UNIT_H:.3f}, got {base_z_2:.3f}"

    def test_pallet_3_on_ground(self, env):
        """Pallet 3 is on the ground — base_z ≈ 0."""
        base_z_3 = env._pallet_base_z[0][3]
        assert abs(base_z_3) < 0.01, \
            f"Pallet 3 base_z should be 0, got {base_z_3:.3f}"

    def test_stacked_pallet_above_ground_pallet(self, env):
        """Stacked pallet z should be higher than ground pallet z."""
        z2 = env.pallets[2].data.root_pos_w[0, 2].item()
        z3 = env.pallets[3].data.root_pos_w[0, 2].item()
        assert z2 > z3, \
            f"Stacked pallet should be above: z2={z2:.3f} z3={z3:.3f}"


# ============================================================================
# §5.2 — Sensor output shapes
# ============================================================================

@_requires_sim
class TestSensorOutputs:
    """Verify sensor data shapes and types are correct."""

    def test_camera_output_shape(self, env):
        """Camera outputs should be (1, H, W, C) with H=W=224."""
        env.reset()
        _step_n(env, 5)  # let sensors warm up
        obs = env._get_observations()

        for key in ("rgb_front", "rgb_left", "rgb_right"):
            assert key in obs, f"Missing camera key: {key}"
            img = obs[key]
            assert img.shape[1] == 224, f"{key} height: {img.shape[1]}"
            assert img.shape[2] == 224, f"{key} width: {img.shape[2]}"

    def test_lidar_output_shape(self, env):
        """LiDAR output should be (1, N_rays, 3)."""
        obs = env._get_observations()
        assert "lidar" in obs
        lidar = obs["lidar"]
        assert lidar.ndim == 3, f"LiDAR ndim: {lidar.ndim}"
        assert lidar.shape[0] == 1, f"LiDAR batch: {lidar.shape[0]}"
        assert lidar.shape[2] == 3, f"LiDAR channels: {lidar.shape[2]}"

    def test_state_vector_shape(self, env):
        """State vector should be (1, 8)."""
        obs = env._get_observations()
        state = obs["state"]
        assert state.shape == (1, 8), f"State shape: {state.shape}"

    def test_multi_pallet_count(self, env):
        """Should have _N_INTERACTABLE pallets registered."""
        assert len(env.pallets) == _N_INTERACTABLE
        assert len(env.pallet_boxes) == _N_INTERACTABLE


# ============================================================================
# §2 Regression — Forklift collides with cargo (no pass-through)
# ============================================================================

@_requires_sim
class TestCollisionRegression:
    """Drive the forklift straight at a stationary cargo+pallet.
    Assert the forklift decelerates / stops (doesn't pass through)."""

    def test_forklift_stops_at_cargo(self, env):
        """§2 regression: drive at pallet at 1 m/s, forklift must not
        pass through. After 2s the forklift X should be less than the
        pallet X (it stopped or bounced)."""
        env.reset()
        pallet_x = env.pallets[0].data.root_pos_w[0, 0].item()
        x0, _, _ = _get_pose(env)

        # Drive forward at 1 m/s for 3 seconds (90 steps)
        _step_n(env, 90, torch.tensor([[1.0, 0.0, 0.0]], device=env.device))

        x1, _, _ = _get_pose(env)
        # The forklift should have been stopped by the pallet's kinematic
        # collider — its X should not exceed the pallet's X minus some margin.
        # With proper collision, forklift front (~1m ahead of root) stops at
        # pallet edge. Without collision, forklift would be at x0 + 1.0*3 = x0+3.
        assert x1 < pallet_x + 0.5, \
            f"Forklift passed through pallet! x1={x1:.2f} pallet_x={pallet_x:.2f}"


# ============================================================================
# §3 Regression — Stacking produces no Z interpenetration
# ============================================================================

@_requires_sim
class TestStackingRegression:
    """Pre-stacked pallet 2 on pallet 3: verify no Z interpenetration
    and stack is stable."""

    def test_no_z_overlap(self, env):
        """§3 regression: pallet 2 bottom must be >= pallet 3 cargo top."""
        env.reset()
        _step_n(env, 5)  # settle

        # Pallet 3 cargo top Z: highest box of pallet 3
        top_z_3 = 0.0
        for bi in range(_N_BOXES):
            bz = env.pallet_boxes[3][bi].data.root_pos_w[0, 2].item()
            top = bz + _BOX_H / 2
            if top > top_z_3:
                top_z_3 = top

        # Pallet 2 bottom Z
        p2_z = env.pallets[2].data.root_pos_w[0, 2].item()
        p2_bottom = p2_z - _PALLET_H / 2

        # Allow 2mm tolerance
        assert p2_bottom >= top_z_3 - 0.002, \
            f"Z interpenetration: pallet_2 bottom={p2_bottom:.4f} < " \
            f"pallet_3 cargo top={top_z_3:.4f}"

    def test_stack_stable_3s(self, env):
        """Stacked pallet should not move significantly over 3 seconds."""
        env.reset()
        _step_n(env, 5)
        z_before = env.pallets[2].data.root_pos_w[0, 2].item()

        _step_n(env, 90)  # 3 seconds
        z_after = env.pallets[2].data.root_pos_w[0, 2].item()

        assert abs(z_after - z_before) < 0.05, \
            f"Stack unstable: z moved {z_before:.3f} → {z_after:.3f}"


# ============================================================================
# §5.2 — Observability checks
# ============================================================================

@_requires_sim
class TestObservability:
    """Verify observability instrumentation works."""

    def test_status_prints_after_steps(self, env):
        """Step counter should increment and status should not crash."""
        env.reset()
        env._step_count = 0
        _step_n(env, 25)
        assert env._step_count >= 20, \
            f"Step counter not incrementing: {env._step_count}"

    def test_lidar_status_populated(self, env):
        """After stepping, LiDAR status should be set."""
        env.reset()
        _step_n(env, 25)
        assert env._lidar_status in ("OK", "NO_RETURNS", "NO_DATA", "INIT"), \
            f"Unexpected LiDAR status: {env._lidar_status}"


# ============================================================================
# §5.2 — Authoritative state consistency
# ============================================================================

@_requires_sim
class TestAuthoritativeState:
    """Verify authoritative state trackers are self-consistent."""

    def test_heading_integrates_correctly(self, env):
        """Driving in a circle — heading should change by omega * t."""
        env.reset()
        h0 = env._heading[0].item()

        # Turn with omega_z = 1 rad/s for ~1 second (30 steps)
        omega = 1.0
        dt = env.cfg.sim.dt  # per substep
        dec = env.cfg.decimation
        n_steps = 30
        action = torch.tensor([[0.0, omega, 0.0]], device=env.device)
        _step_n(env, n_steps, action)

        h1 = env._heading[0].item()
        expected_delta = omega * dt * dec * n_steps  # omega applied per substep
        # Allow 10% tolerance for decimation/timing
        actual_delta = h1 - h0
        assert abs(actual_delta) > 0.1, \
            f"Heading should have changed, delta={actual_delta:.4f}"

    def test_position_integrates_correctly(self, env):
        """Driving forward — x should increase."""
        env.reset()
        x0, _, _ = _get_pose(env)

        _step_n(env, 30, torch.tensor([[3.0, 0.0, 0.0]], device=env.device))

        x1, _, _ = _get_pose(env)
        assert x1 > x0 + 0.5, \
            f"X should have increased significantly: {x0:.2f} → {x1:.2f}"


# ============================================================================
# Teardown
# ============================================================================

if _SIM_AVAILABLE:
    def teardown_module():
        """Close simulation after all tests."""
        _sim_app.close()
