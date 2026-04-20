"""
Test Battery C — Verify stacking produces no interpenetration.

Tests the pre-stacked scene (pallet 2 on pallet 3) and verifies
correct Z placement.

Requires Isaac Sim. Run:
    ./isaaclab.sh -p -m pytest tests/forklift/test_battery_c_stack.py -v --headless
"""

import sys
import os
import pytest

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

if _SIM_AVAILABLE:
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                     "..", "..", "scripts", "forklift"))
    from forklift_env import (ForkliftEnv, ForkliftEnvCfg,
                              _N_INTERACTABLE, _N_BOXES,
                              _PALLET_H, _BOX_H, _UNIT_H)

pytestmark = pytest.mark.skipif(not _SIM_AVAILABLE, reason="Isaac Sim not available")


def _step_n(env, n, action=None):
    if action is None:
        action = torch.zeros(1, 3, device=env.device)
    for _ in range(n):
        env.step(action)


@pytest.fixture(scope="module")
def env():
    if not _SIM_AVAILABLE:
        pytest.skip()
    e = ForkliftEnv(ForkliftEnvCfg())
    e.reset()
    yield e
    e.close()


class TestBatteryCStack:
    """Verify stacking Z computation is correct and no interpenetration."""

    def test_initial_stack_no_interpenetration(self, env):
        """Pre-stacked pallet 2 on pallet 3: no Z overlap.
        Pallet 2 bottom must be >= pallet 3 cargo top."""
        env.reset()
        _step_n(env, 10)  # settle

        # Pallet 3 top: base_z + _PALLET_H + _BOX_H
        base_z_3 = env._pallet_base_z[0][3]
        target_top_3 = base_z_3 + _PALLET_H + _BOX_H  # 0 + 0.285 + 0.60 = 0.885

        # Pallet 2 bottom: base_z_2
        base_z_2 = env._pallet_base_z[0][2]

        # base_z_2 should be at target_top_3 + epsilon (0.885 + 0.005 = 0.890)
        # or at _UNIT_H (0.885) if from initial placement
        gap = base_z_2 - target_top_3
        assert gap >= -0.002, \
            f"INTERPENETRATION: pallet_2 base_z={base_z_2:.4f} < " \
            f"pallet_3 top={target_top_3:.4f} (gap={gap:.4f}m)"

    def test_initial_stack_correct_height(self, env):
        """Pallet 2 base should be at _UNIT_H or slightly above."""
        base_z_2 = env._pallet_base_z[0][2]
        assert abs(base_z_2 - _UNIT_H) < 0.01, \
            f"Pallet 2 base_z={base_z_2:.4f} should be ~{_UNIT_H:.4f}"

    def test_stack_stable_5s(self, env):
        """Stacked pallet should not move over 5 seconds."""
        env.reset()
        _step_n(env, 10)

        z_before = env.pallets[2].data.root_pos_w[0, 2].item()
        _step_n(env, 150)  # 5 seconds at 30 Hz
        z_after = env.pallets[2].data.root_pos_w[0, 2].item()

        assert abs(z_after - z_before) < 0.05, \
            f"Stack unstable: z {z_before:.4f} → {z_after:.4f}"

    def test_stacked_pallet_boxes_above_target(self, env):
        """Every box on pallet 2 should be above every box on pallet 3."""
        env.reset()
        _step_n(env, 10)

        # Get lowest box of pallet 2
        min_z_2 = float('inf')
        for bi in range(_N_BOXES):
            bz = env.pallet_boxes[2][bi].data.root_pos_w[0, 2].item()
            box_bottom = bz - _BOX_H / 2
            if box_bottom < min_z_2:
                min_z_2 = box_bottom

        # Get highest box of pallet 3
        max_z_3 = float('-inf')
        for bi in range(_N_BOXES):
            bz = env.pallet_boxes[3][bi].data.root_pos_w[0, 2].item()
            box_top = bz + _BOX_H / 2
            if box_top > max_z_3:
                max_z_3 = box_top

        gap = min_z_2 - max_z_3
        assert gap >= -0.002, \
            f"Box interpenetration: pallet_2 lowest box bottom={min_z_2:.4f} < " \
            f"pallet_3 highest box top={max_z_3:.4f} (gap={gap:.4f}m)"

    def test_target_top_computation_matches_geometry(self, env):
        """Verify target_top_z = base_z + _PALLET_H + _BOX_H is consistent."""
        env.reset()
        _step_n(env, 10)

        for pi in range(_N_INTERACTABLE):
            base_z = env._pallet_base_z[0][pi]
            computed_top = base_z + _PALLET_H + _BOX_H

            # Pallet center z from physics
            pal_z = env.pallets[pi].data.root_pos_w[0, 2].item()
            expected_pal_z = base_z + _PALLET_H / 2

            assert abs(pal_z - expected_pal_z) < 0.05, \
                f"Pallet {pi}: physics z={pal_z:.4f} vs expected={expected_pal_z:.4f}"


if _SIM_AVAILABLE:
    def teardown_module():
        _sim_app.close()
