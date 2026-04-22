"""
Verify stacking heights are correct by running the sim and measuring Z positions.

Usage:
    ./isaaclab.sh -p scripts/tools/verify_stacking.py
    ./isaaclab.sh -p scripts/tools/verify_stacking.py --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify stacking heights.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "forklift"))
from forklift_env import (ForkliftEnv, ForkliftEnvCfg,
                          _PALLET_H, _BOX_H, _UNIT_H, _N_INTERACTABLE, _N_BOXES)

cfg = ForkliftEnvCfg()
cfg.enable_sensors = False  # no cameras/lidar needed for height check
env = ForkliftEnv(cfg)
env.reset()

# Step 20 times to let everything settle
for _ in range(20):
    env.step(torch.zeros(1, 3, device=env.device))

print("\n" + "=" * 70, flush=True)
print("  STACKING HEIGHT VERIFICATION", flush=True)
print("=" * 70, flush=True)
print(f"  Constants: _PALLET_H={_PALLET_H:.3f}m  _BOX_H={_BOX_H:.3f}m  "
      f"_UNIT_H={_UNIT_H:.3f}m", flush=True)
print(flush=True)

all_ok = True

for pi in range(_N_INTERACTABLE):
    base_z = env._pallet_base_z[0][pi]
    pal_pos = env.pallets[pi].data.root_pos_w[0].cpu()
    pal_z = pal_pos[2].item()
    expected_pal_z = base_z + _PALLET_H / 2

    # Find min/max box Z for this pallet
    box_zs = []
    for bi in range(_N_BOXES):
        bz = env.pallet_boxes[pi][bi].data.root_pos_w[0, 2].item()
        box_zs.append(bz)
    min_box_z = min(box_zs)
    max_box_z = max(box_zs)
    expected_box_z = base_z + _PALLET_H + _BOX_H / 2

    pal_ok = abs(pal_z - expected_pal_z) < 0.05
    box_ok = abs(min_box_z - expected_box_z) < 0.05

    status_pal = "OK" if pal_ok else "MISMATCH"
    status_box = "OK" if box_ok else "MISMATCH"
    if not pal_ok or not box_ok:
        all_ok = False

    print(f"  Pallet {pi}:", flush=True)
    print(f"    base_z (authoritative) = {base_z:.4f} m", flush=True)
    print(f"    pallet center Z (physics) = {pal_z:.4f} m  "
          f"expected = {expected_pal_z:.4f} m  [{status_pal}]", flush=True)
    print(f"    pallet bottom Z = {pal_z - _PALLET_H/2:.4f} m  "
          f"pallet top Z = {pal_z + _PALLET_H/2:.4f} m", flush=True)
    print(f"    box center Z range = [{min_box_z:.4f}, {max_box_z:.4f}] m  "
          f"expected = {expected_box_z:.4f} m  [{status_box}]", flush=True)
    print(f"    box bottom Z = {min_box_z - _BOX_H/2:.4f} m  "
          f"box top Z = {max_box_z + _BOX_H/2:.4f} m", flush=True)
    print(flush=True)

# Check stacking gap between pallet 2 (stacked) and pallet 3 (ground)
p3_box_top = max(env.pallet_boxes[3][bi].data.root_pos_w[0, 2].item()
                 for bi in range(_N_BOXES)) + _BOX_H / 2
p2_pal_bottom = env.pallets[2].data.root_pos_w[0, 2].item() - _PALLET_H / 2
gap = p2_pal_bottom - p3_box_top

gap_ok = gap >= -0.002
if not gap_ok:
    all_ok = False

print(f"  Stack gap (pallet_2 bottom - pallet_3 cargo top):", flush=True)
print(f"    pallet_3 cargo top Z = {p3_box_top:.4f} m", flush=True)
print(f"    pallet_2 bottom Z    = {p2_pal_bottom:.4f} m", flush=True)
print(f"    gap = {gap*1000:.1f} mm  [{'OK' if gap_ok else 'INTERPENETRATION'}]", flush=True)
print(flush=True)

print("=" * 70, flush=True)
if all_ok:
    print("  ALL HEIGHTS CORRECT", flush=True)
else:
    print("  SOME HEIGHTS INCORRECT — see details above", flush=True)
print("=" * 70, flush=True)

env.close()
simulation_app.close()
