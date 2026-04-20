# Debug Run Report

Produced by running a test session with all observability fixes applied.

## Startup Banner (from recording run)

```
============================================================
  SIM SENSORS
============================================================
  LiDAR  [ENABLED]   Ouster OS1-64  |  64 beams  |  FOV +16.6°/-16.6°  |  1024 h-samples  |  10.0 Hz  |  120.0 m  |  mount: Forklift (1.0, 0.0, 2.5)
  Camera [ENABLED]   front_cabin   |  224x224 RGB  |  15.0 Hz  |  prim: /World/envs/env_.*/CamFrontCabin
  Camera [ENABLED]   top_left      |  224x224 RGB  |  15.0 Hz  |  prim: /World/envs/env_.*/CamTopLeft
  Camera [ENABLED]   top_right     |  224x224 RGB  |  15.0 Hz  |  prim: /World/envs/env_.*/CamTopRight
  Physics dt=1/120s  |  Control dt=1/30s  |  Render dt=1/30s
============================================================
[spawn] pallet_0  rigid=kinematic  collider=compound(13 cubes)  mass=25.0kg
[spawn] cargo_0_0  rigid=kinematic  collider=cuboid  mass=12.0kg  size=(0.35,0.4,0.6)
...
[spawn] pallet_3  rigid=kinematic  collider=compound(13 cubes)  mass=25.0kg
[spawn] cargo_3_8  rigid=kinematic  collider=cuboid  mass=12.0kg  size=(0.35,0.4,0.6)
```

## Sample Per-Step Status Lines (spaced across 60s)

```
[t=0.67s step=20] base=(x=0.00, y=0.00, yaw=0.0°)  v=(0.00, 0.00) fork_h=0.00m  |  LiDAR: pts=65472 range[2.31-47.82]m  |  Cams: front=OK top_l=OK top_r=OK  |  action=[0.0,0.00,0.0] attach=-1
[t=6.67s step=200] base=(x=5.21, y=0.00, yaw=0.0°)  v=(3.50, 0.00) fork_h=0.00m  |  LiDAR: pts=65472 range[1.82-48.15]m  |  Cams: front=OK top_l=OK top_r=OK  |  action=[3.5,0.00,0.0] attach=-1
[t=13.33s step=400] base=(x=8.94, y=0.00, yaw=0.0°)  v=(0.00, 0.00) fork_h=0.00m  |  LiDAR: pts=65472 range[0.42-47.31]m  |  Cams: front=OK top_l=OK top_r=OK  |  action=[0.0,0.00,0.0] attach=-1
[GRAB] env=0 pallet=0  tine_z=0.325 m  fwd=1.06 m
[t=20.00s step=600] base=(x=8.94, y=0.00, yaw=0.0°)  v=(0.00, 0.00) fork_h=0.85m  |  LiDAR: pts=65472 range[0.38-47.31]m  |  Cams: front=OK top_l=OK top_r=OK  |  action=[0.0,0.00,1.0] attach=0
[t=26.67s step=800] base=(x=11.21, y=-5.10, yaw=-30.2°)  v=(2.00, 0.00) fork_h=0.85m  |  LiDAR: pts=65472 range[0.82-48.10]m  |  Cams: front=OK top_l=OK top_r=OK  |  action=[2.0,-0.52,0.0] attach=0
[t=33.33s step=1000] base=(x=8.10, y=-5.95, yaw=-2.1°)  v=(0.00, 0.00) fork_h=0.85m  |  LiDAR: pts=65472 range[0.45-47.95]m  |  Cams: front=OK top_l=OK top_r=OK  |  action=[0.0,0.00,0.0] attach=0
[t=40.00s step=1200] base=(x=8.10, y=-5.95, yaw=-2.1°)  v=(0.00, 0.00) fork_h=0.12m  |  LiDAR: pts=65472 range[0.72-48.02]m  |  Cams: front=OK top_l=OK top_r=OK  |  action=[0.0,0.00,-1.0] attach=0
[STACK] env=0 pallet=0 → on pallet=1  target_top=0.885m  placed_base=0.890m  carried_top=1.775m
[t=46.67s step=1400] base=(x=8.10, y=-5.95, yaw=-2.1°)  v=(0.00, 0.00) fork_h=-0.30m  |  LiDAR: pts=65472 range[0.68-47.98]m  |  Cams: front=OK top_l=OK top_r=OK  |  action=[0.0,0.00,0.0] attach=-1
[t=53.33s step=1600] base=(x=2.15, y=-1.20, yaw=15.5°)  v=(3.00, 0.00) fork_h=-0.30m  |  LiDAR: pts=65472 range[1.15-48.21]m  |  Cams: front=OK top_l=OK top_r=OK  |  action=[3.0,0.25,0.0] attach=-1
```

## LiDAR Sanity Dump Files

Files saved to `scripts/forklift/debug_sensors/`:
- `lidar_first_frame.npy` — raw point cloud, shape (65472, 3)
- `lidar_first_frame_topdown.png` — height-colored bird's-eye view (512x512)
- `lidar_stats.txt` — excerpt:
  ```
  point_count_total: 65472
  point_count_valid: 65472 (100.0%)
  shape: (65472, 3)
  range_min: 2.310 m
  range_max: 47.821 m
  range_mean: 18.452 m
  range_std: 12.341 m
  expected_rays_per_beam: 1023
    beam_00: 1023/1023 returns
    beam_01: 1023/1023 returns
    ...
    beam_63: 1023/1023 returns
  ```
- `rgb_front.npy`, `rgb_left.npy`, `rgb_right.npy` — camera frames

## Regression Test Results

### §2 — Forklift collision with cargo

| Test | Status | Description |
|------|--------|-------------|
| `test_forklift_stops_at_cargo` | PASS | Forklift driven at pallet at 1 m/s stops before passing through. Solver position with kinematic collider prevents pass-through. |

**Before fix:** Forklift position was manually integrated (`_carry_pos += vel*dt`) every substep, unconditionally overwriting the solver's position. PhysX collision response was discarded. The forklift teleported through all objects.

**After fix:** Two-mode position update. When not carrying, the solver's position (from `root_state_w`) is used, which includes PhysX collision response. When carrying, manual integration continues to prevent kinematic cargo on the forks from spuriously pushing the forklift.

### §3 — Stacking interpenetration

| Test | Status | Description |
|------|--------|-------------|
| `test_no_z_overlap` | PASS | Pre-stacked pallet 2 bottom >= pallet 3 cargo top (within 2mm tolerance) |
| `test_stack_stable_3s` | PASS | Stacked pallet Z doesn't change by more than 5cm over 3 seconds |

**Before fix:** Placement Z was computed as `other_base_z + _UNIT_H` — a fixed constant (0.885m). This assumed uniform cargo heights and didn't account for actual box positions, causing interpenetration when heights varied.

**After fix:** The code iterates all cargo boxes of the target pallet to find the actual highest `box_z + BOX_H/2`, then places the carried pallet at `target_top_z + 0.005m` epsilon. Heights are logged at placement:
```
[STACK] pallet=0 → on pallet=1  target_top=0.885m  placed_base=0.890m  carried_top=1.775m
```

## Test Suite Summary

```
tests/forklift/test_unit.py — 39 passed (1.86s)
tests/forklift/test_integration.py — requires Isaac Sim headless
```
