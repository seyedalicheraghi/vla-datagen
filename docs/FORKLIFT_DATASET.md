# Forklift Dataset Specification

Dataset for fine-tuning pi0 (OpenPI) on a simulated forklift pick-and-place stacking task.

## Coordinate System

- **X-forward**, **Y-left**, **Z-up** (Isaac Sim / USD convention)
- Heading (yaw) = 0 means facing +X; positive yaw = counter-clockwise
- All positions in metres, angles in radians, velocities in m/s or rad/s

---

## Observation Fields

### `observation.images.front_cabin`

| Property | Value |
|----------|-------|
| Shape | `(224, 224, 3)` |
| Dtype | `uint8` |
| Units | RGB pixel values [0, 255] |
| Frame | Camera frame (pinhole, world-pose-updated) |
| Update rate | 15 Hz |
| Mount | Inside cabin, 0.3m forward, 1.8m up, pitched 34 deg down. Fork tips visible in lower frame. |
| Focal length | 10.0 mm |
| Horizontal aperture | 20.955 mm |

### `observation.images.top_left`

| Property | Value |
|----------|-------|
| Shape | `(224, 224, 3)` |
| Dtype | `uint8` |
| Mount | Roof left side, 0.6m lateral, 2.3m up, 23 deg outward yaw, 20 deg down pitch |
| Update rate | 15 Hz |

### `observation.images.top_right`

| Property | Value |
|----------|-------|
| Shape | `(224, 224, 3)` |
| Dtype | `uint8` |
| Mount | Mirror of top_left on right side |
| Update rate | 15 Hz |

### `observation.lidar`

| Property | Value |
|----------|-------|
| Shape | `(64, 1024)` |
| Dtype | `float32` |
| Units | Range in metres per pixel |
| Sensor | Ouster OS1-64 parameters |
| Beams | 64 vertical channels |
| Vertical FOV | -16.6 deg to +16.6 deg (33.2 deg total) |
| Horizontal FOV | 360 deg (1024 samples) |
| Max range | 120 m |
| Update rate | 10 Hz |
| Mount | Front of forklift, 1.0m forward, 2.5m up |
| Storage | Range image (fixed shape), rays that miss return `max_range` |

### `observation.state`

| Index | Name | Units | Range |
|-------|------|-------|-------|
| 0 | x | metres | warehouse bounds |
| 1 | y | metres | warehouse bounds |
| 2 | yaw (heading) | radians | [-pi, pi] |
| 3 | vx (world frame) | m/s | [-5, 5] |
| 4 | vy (world frame) | m/s | [-5, 5] |
| 5 | omega_z | rad/s | continuous |
| 6 | fork_height | metres | [-0.3, 1.5] |
| 7 | grabbed_idx | float | -1 (none) or 0..3 (pallet index) |

Shape: `(8,)`, dtype: `float32`

---

## Action Field

### `action`

| Index | Name | Units | Sign convention | Range |
|-------|------|-------|-----------------|-------|
| 0 | v_forward | m/s | positive = forward (+X) | [-5, 5] |
| 1 | yaw_rate | rad/s | positive = counter-clockwise | continuous |
| 2 | fork_lift_vel | normalized | positive = raise forks | [-1, 1] |
| 3 | fork_tilt_vel | normalized | reserved (always 0) | 0 |
| 4 | attach_toggle | binary | reserved (always 0) | 0 |

Shape: `(5,)`, dtype: `float32`

---

## Sensor Specifications

### Camera Intrinsics (all three cameras)

```
focal_length:        10.0 mm
horizontal_aperture: 20.955 mm
resolution:          224 x 224
clipping_range:      0.1 - 80.0 m
```

### Camera Extrinsics (relative to forklift base frame at z=0, heading=0)

| Camera | Position (x, y, z) m | Pitch (deg) | Yaw offset (deg) |
|--------|---------------------|-------------|-------------------|
| front_cabin | (0.3, 0.0, 1.8) | -34 | 0 |
| top_left | (0.2, 0.6, 2.3) | -20 | +23 |
| top_right | (0.2, -0.6, 2.3) | -20 | -23 |

### LiDAR (Ouster OS1-64 simulation via RayCaster)

```
channels:         64
vertical_fov:     -16.6 to +16.6 deg
horizontal_fov:   -180.0 to +180.0 deg (360 deg)
horizontal_res:   0.3516 deg (~1024 samples)
max_distance:     120.0 m
ray_alignment:    "base" (full 6-DOF tracking)
mount_position:   (1.0, 0.0, 2.5) relative to forklift root
```

---

## Scene Configuration

- Warehouse: 32m x 32m floor with 4m walls
- 4 interactable pallets with 9 cargo boxes each:
  - Pallet 0 (blue): 10m ahead, primary pick target
  - Pallet 1 (brown): 8m ahead, 6m right
  - Pallet 2 (dark brown): pre-stacked on pallet 3
  - Pallet 3 (medium brown): 8m ahead, 6m left
- 50 static scatter pallets with boxes (decoration/obstacles)
- GMA 48x40 pallet: 1.219m x 1.500m x 0.285m, 25 kg
- Cargo boxes: 0.35m x 0.40m x 0.60m, 12 kg each

---

## How to Run a Teleop Recording Session

```bash
cd ~/Projects/IsaacLab

# Record 10 episodes with keyboard teleop
./isaaclab.sh -p scripts/forklift/record_lerobot.py \
    --num_episodes 10 \
    --task_instruction "pick up the blue pallet and place it on top of the brown pallet"

# Output saved to: datasets/forklift_teleop/
```

Controls during recording:
- **W/S** — forward/backward
- **A/D** — turn left/right
- **E/Q** — raise/lower forks
- **SPACE** — save episode and reset
- **R** — discard episode and reset

---

## How to Load the Dataset in OpenPI

### Version compatibility

This recording script uses **lerobot 0.4.4** which writes **v3.0** format.
OpenPI uses **lerobot 0.1.0** which reads **v2.1** format.

To convert v3.0 to v2.1 for OpenPI compatibility, use lerobot's built-in
converter or write the features to match OpenPI's expected schema.

### OpenPI training config

Add to `~/Projects/openpi/src/openpi/training/config.py`:

```python
@dataclass
class ForkliftConfig(TrainConfig):
    dataset_repo_id: str = "forklift/teleop"
    # Adjust paths as needed
    action_dim: int = 5
    action_horizon: int = 16
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="pi0",
    ))
    data: DataConfig = field(default_factory=lambda: DataConfig(
        repo_id="forklift/teleop",
        root=str(Path.home() / "Projects/IsaacLab/datasets/forklift_teleop"),
    ))
```

A custom `RepackTransform` is needed to map:
- `observation.images.front_cabin` → `image/base_0_rgb`
- `observation.images.top_left` → `image/left_wrist_0_rgb`
- `observation.images.top_right` → `image/right_wrist_0_rgb`
- `observation.state` → `state`
- `action` → `actions`
- `task` → `prompt`

### Training command

```bash
cd ~/Projects/openpi
uv run scripts/compute_norm_stats.py --config-name pi0_forklift
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_forklift \
    --exp-name=forklift_v1
```

---

## Running the Test Suite

```bash
# Unit tests (no Isaac Sim required)
pytest tests/forklift/test_unit.py -v

# Integration tests (requires Isaac Sim, runs headless)
./isaaclab.sh -p -m pytest tests/forklift/test_integration.py -v --headless
```

---

## Known Limitations

1. **LiDAR only sees ground plane** — The RayCaster is configured with
   `mesh_prim_paths=["/World/Ground"]`. Walls, pallets, and boxes are not
   in the ray-cast mesh. Use `MultiMeshRayCaster` for full obstacle detection.

2. **No fork tilt** — The ForkliftC USD has a prismatic lift joint but no
   tilt joint. `fork_tilt_vel` (action index 3) is always 0.

3. **Kinematic carry** — Cargo is carried kinematically (pose follows forklift).
   There is no physics-based friction handoff during carry. On release, cargo
   is placed at the computed position.

4. **LeRobot v3.0 vs v2.1** — Recording uses v3.0 format. OpenPI expects v2.1.
   Manual conversion or a version-pinned lerobot install is needed.

5. **Single-layer stacking** — Stacking detection only checks the immediate
   pallet below. Deep stacks (3+ layers) are not tested.
