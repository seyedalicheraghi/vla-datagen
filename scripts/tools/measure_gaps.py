"""
Measure actual gaps between objects and ground in the live sim.
Reports exactly what's wrong with the placement.

Usage:
    ./isaaclab.sh -p scripts/tools/measure_gaps.py --headless
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "forklift"))
from forklift_env import (ForkliftEnv, ForkliftEnvCfg,
                          _PALLET_H, _BOX_H, _UNIT_H, _PALLET_BOT_H,
                          _N_INTERACTABLE, _N_BOXES, _BOX_LOCAL_OFFSETS)

cfg = ForkliftEnvCfg()
cfg.enable_sensors = False
env = ForkliftEnv(cfg)
env.reset()

# Step to settle
for _ in range(30):
    env.step(torch.zeros(1, 3, device=env.device))

print("\n" + "=" * 70, flush=True)
print("  GAP MEASUREMENT — WHAT THE SIM ACTUALLY LOOKS LIKE", flush=True)
print("=" * 70, flush=True)

print(f"\n  Constants:", flush=True)
print(f"    _PALLET_H    = {_PALLET_H:.4f}m  (deck+stringer+bottom)", flush=True)
print(f"    _PALLET_BOT_H= {_PALLET_BOT_H:.4f}m  (bottom blocks)", flush=True)
print(f"    _BOX_H       = {_BOX_H:.4f}m", flush=True)
print(f"    _UNIT_H      = {_UNIT_H:.4f}m  (pallet+boxes)", flush=True)

# The pallet is built with z=0 as pallet bottom, but the rigid body
# root is at the Xform origin. Let's check what _build_compound_pallet
# actually creates:
# - top_deck at z = bot_h + stg_h + top_h/2 = 0.060 + 0.200 + 0.0125 = 0.2725
# - bottom blocks at z = bot_h/2 = 0.030
# So the pallet geometry spans from z=0 (bottom of bottom blocks) to
# z = 0.285 (top of top deck).
# The root Xform is at z=0 in the pallet's local frame.
# When we place the pallet, we set root z = base_z + _PALLET_H / 2
# But _PALLET_H/2 = 0.1425, meaning the root is at the CENTER of the
# pallet height range. However, the pallet geometry starts at LOCAL z=0
# (bottom), not at LOCAL z=-_PALLET_H/2.
#
# THIS IS THE BUG: the pallet root is NOT at the center of the pallet.
# The pallet bottom is at LOCAL z=0, top at LOCAL z=_PALLET_H.
# So the root IS at the bottom corner, not the center.
# Setting root z = base_z + _PALLET_H/2 lifts the entire pallet up
# by _PALLET_H/2, creating a gap between the pallet bottom and the ground.

print(f"\n  Pallet local geometry (from _build_compound_pallet):", flush=True)
print(f"    bottom blocks: z = {_PALLET_BOT_H/2:.4f}m (center)", flush=True)
print(f"    top deck:      z = {0.060 + 0.200 + 0.025/2:.4f}m (center)", flush=True)
print(f"    ROOT Xform is at LOCAL z=0 (the very bottom of the pallet)", flush=True)
print(f"    So setting root z = base_z + _PALLET_H/2 = {_PALLET_H/2:.4f}m", flush=True)
print(f"    means pallet bottom is FLOATING at z = {_PALLET_H/2:.4f}m above base_z!", flush=True)

for pi in range(_N_INTERACTABLE):
    base_z = env._pallet_base_z[0][pi]
    pal_pos = env.pallets[pi].data.root_pos_w[0].cpu()
    pal_root_z = pal_pos[2].item()

    # The pallet bottom in world frame = root_z + 0 (local bottom is at 0)
    # But we set root_z = base_z + _PALLET_H/2, so bottom = base_z + _PALLET_H/2
    # The ground is at base_z. Gap = _PALLET_H/2 = 0.1425m!
    pal_bottom_world = pal_root_z  # root IS at the bottom
    gap_to_ground = pal_bottom_world - base_z

    print(f"\n  Pallet {pi}:", flush=True)
    print(f"    base_z = {base_z:.4f}m", flush=True)
    print(f"    root_z (physics) = {pal_root_z:.4f}m", flush=True)
    print(f"    gap to ground/target = {gap_to_ground:.4f}m", flush=True)

    if gap_to_ground > 0.01:
        print(f"    >>> GAP DETECTED: {gap_to_ground*100:.1f}cm floating above surface!", flush=True)

    # Check boxes
    for bi in range(_N_BOXES):
        bz = env.pallet_boxes[pi][bi].data.root_pos_w[0, 2].item()
        local_lz = _BOX_LOCAL_OFFSETS[bi][2]  # local z offset from pallet bottom
        expected_bz = base_z + local_lz  # if root is at bottom
        expected_bz_centered = base_z + _PALLET_H/2 + local_lz  # if root is at center
        # Which one matches?
        if bi == 0:  # just check first box
            print(f"    box_0 z = {bz:.4f}m", flush=True)
            print(f"      if root=bottom: expected = base_z + lz = {base_z + local_lz:.4f}m", flush=True)
            print(f"      if root=center: expected = base_z + H/2 + lz = {base_z + _PALLET_H/2 + local_lz:.4f}m", flush=True)
            print(f"      lz (local offset) = {local_lz:.4f}m = _PALLET_H + _BOX_H/2 = {_PALLET_H:.4f} + {_BOX_H/2:.4f}", flush=True)

# Now check: where does _place_pallet_and_cargo put things?
print(f"\n  _place_pallet_and_cargo sets:", flush=True)
print(f"    pal_pose z = base_z + _PALLET_H/2 = base_z + {_PALLET_H/2:.4f}", flush=True)
print(f"    box z = base_z + lz = base_z + {_PALLET_H + _BOX_H/2:.4f}", flush=True)
print(f"", flush=True)
print(f"  If pallet root is at LOCAL z=0 (bottom of geometry):", flush=True)
print(f"    Root at world z = base_z + {_PALLET_H/2:.4f}", flush=True)
print(f"    Pallet bottom at world z = base_z + {_PALLET_H/2:.4f} + 0 = base_z + {_PALLET_H/2:.4f}", flush=True)
print(f"    >>> FLOATING {_PALLET_H/2*100:.1f}cm above ground!", flush=True)
print(f"", flush=True)
print(f"  FIX: Set pal_pose z = base_z (not base_z + _PALLET_H/2)", flush=True)
print(f"  because the pallet geometry starts at local z=0.", flush=True)

print("\n" + "=" * 70, flush=True)

env.close()
simulation_app.close()
