"""
Record forklift teleoperation episodes in LeRobot v2/v3 format.

Produces a dataset at datasets/forklift_teleop/ compatible with
LeRobot's LeRobotDataset loader and, after conversion, with OpenPI.

Usage:
    ./isaaclab.sh -p scripts/forklift/record_lerobot.py [--num_episodes 10]
    ./isaaclab.sh -p scripts/forklift/record_lerobot.py --headless  # no GUI

Schema:
    observation.images.front_cabin  — 224×224 RGB (primary VLA camera)
    observation.images.top_left     — 224×224 RGB
    observation.images.top_right    — 224×224 RGB
    observation.lidar               — (N_rays, 3) float32 point cloud
    observation.state               — (8,) float32 proprioception
    action                          — (5,) float32 teleop commands
    task                            — str (natural language instruction)

LiDAR note:
    LeRobot has no first-class point-cloud type. We store LiDAR as a
    range image of shape (64, 1024) with dtype float32, where each pixel
    is the range in metres. This is fixed-shape and video-pipeline-friendly,
    matching π₀-style patch tokenization. Rays that missed return max_range.
"""

# ---------------------------------------------------------------------------
# Step 1: AppLauncher must come before ALL other Isaac Lab imports
# ---------------------------------------------------------------------------

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record forklift teleop to LeRobot dataset.")
parser.add_argument("--num_episodes", type=int, default=10,
                    help="Number of episodes to record.")
parser.add_argument("--max_steps", type=int, default=3000,
                    help="Max steps per episode before auto-reset.")
parser.add_argument("--task_instruction", type=str,
                    default="pick up the blue pallet and place it on top of the brown pallet",
                    help="Natural language task instruction.")
parser.add_argument("--dataset_dir", type=str, default="datasets/forklift_teleop",
                    help="Output directory for the dataset.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Step 2: Import everything else after Isaac Sim is running
# ---------------------------------------------------------------------------

import math
import os
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from forklift_env import ForkliftEnv, ForkliftEnvCfg
from forklift_env import _LIDAR_CHANNELS, _LIDAR_MAX_RANGE

# LeRobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Keyboard input (same as teleop)
import carb.input
import omni.appwindow

# ---------------------------------------------------------------------------
# Sensor constants
# ---------------------------------------------------------------------------

# Compute expected LiDAR horizontal samples from the pattern config
_LIDAR_H_SAMPLES = 1024   # from 360° / 0.3516° resolution

# State vector: [x, y, yaw, vx, vy, omega_z, fork_height, grabbed_idx]
_STATE_DIM = 8

# Action vector: [v_forward, yaw_rate, fork_lift_vel, fork_tilt_vel, attach_toggle]
# fork_tilt_vel and attach_toggle are reserved (always 0 for now)
_ACTION_DIM = 5

# ---------------------------------------------------------------------------
# Keyboard handler
# ---------------------------------------------------------------------------

_keys = {
    "forward": False, "backward": False,
    "left": False, "right": False,
    "fork_up": False, "fork_down": False,
    "reset": False, "save": False,
}

def _make_keyboard_handler():
    _input = carb.input.acquire_input_interface()
    _keyboard = omni.appwindow.get_default_app_window().get_keyboard()

    def _on_key(event, *args):
        released = (event.type == carb.input.KeyboardEventType.KEY_RELEASE)
        pressed = (event.type == carb.input.KeyboardEventType.KEY_PRESS)
        k = event.input
        KI = carb.input.KeyboardInput

        if k in (KI.W, KI.UP):
            _keys["forward"] = not released
        elif k in (KI.S, KI.DOWN):
            _keys["backward"] = not released
        elif k in (KI.A, KI.LEFT):
            _keys["left"] = not released
        elif k in (KI.D, KI.RIGHT):
            _keys["right"] = not released
        elif k == KI.E:
            _keys["fork_up"] = not released
        elif k == KI.Q:
            _keys["fork_down"] = not released
        elif k == KI.R and pressed:
            _keys["reset"] = True
        elif k == KI.SPACE and pressed:
            _keys["save"] = True
        return True

    sub = _input.subscribe_to_keyboard_events(_keyboard, _on_key)
    return _input, _keyboard, sub


def _lidar_to_range_image(hits: np.ndarray, sensor_pos: np.ndarray,
                          channels: int, h_samples: int,
                          max_range: float) -> np.ndarray:
    """Convert raw LiDAR hit points to a range image (channels, h_samples).

    The RayCaster returns ray_hits_w of shape (N_rays, 3). Rays are ordered
    channels × h_samples (vertical × horizontal). Convert to distances.
    """
    # hits shape: (channels * h_samples, 3)
    # Compute distances from sensor origin
    diffs = hits - sensor_pos[np.newaxis, :]
    distances = np.linalg.norm(diffs, axis=-1)  # (N_rays,)

    # Clamp to max range (missed rays report very large values)
    distances = np.clip(distances, 0.0, max_range)

    # Reshape to (channels, h_samples)
    expected_rays = channels * h_samples
    if distances.shape[0] != expected_rays:
        # Pad or truncate if ray count doesn't match exactly
        if distances.shape[0] < expected_rays:
            distances = np.pad(distances, (0, expected_rays - distances.shape[0]),
                               constant_values=max_range)
        else:
            distances = distances[:expected_rays]

    return distances.reshape(channels, h_samples).astype(np.float32)


# ---------------------------------------------------------------------------
# Main recording loop
# ---------------------------------------------------------------------------

def main():
    env = ForkliftEnv(ForkliftEnvCfg())
    _input, _keyboard, _kb_sub = _make_keyboard_handler()

    # ── Create LeRobot dataset ────────────────────────────────────────
    ds_root = os.path.abspath(args_cli.dataset_dir)
    print(f"[INFO] Recording to: {ds_root}")

    dataset = LeRobotDataset.create(
        repo_id="forklift/teleop",
        fps=15,  # matches camera update rate
        root=ds_root,
        robot_type="forklift",
        features={
            "observation.images.front_cabin": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.top_left": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.top_right": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.lidar": {
                "dtype": "float32",
                "shape": (_LIDAR_CHANNELS, _LIDAR_H_SAMPLES),
                "names": ["channels", "horizontal_samples"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (_STATE_DIM,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (_ACTION_DIM,),
                "names": ["action"],
            },
        },
        use_videos=True,
        image_writer_threads=4,
    )

    obs, _ = env.reset()

    V_MAX = 5.0
    STEER_ANGLE = 0.6
    WHEEL_BASE = env.cfg.wheel_base
    STEP_DT = env.cfg.sim.dt * env.cfg.decimation
    ACCEL, DECEL = 4.0, 6.0
    current_v_x = 0.0

    episode_count = 0
    step = 0
    frame_count = 0  # frames in current episode

    print("\n" + "=" * 60)
    print("Forklift Recording — LeRobot Format")
    print("=" * 60)
    print("  W/S    → forward/backward")
    print("  A/D    → turn left/right")
    print("  E/Q    → raise/lower forks")
    print("  SPACE  → save episode and reset")
    print("  R      → discard episode and reset")
    print(f"  Recording {args_cli.num_episodes} episodes")
    print(f"  Task: {args_cli.task_instruction}")
    print("=" * 60 + "\n")

    while simulation_app.is_running() and episode_count < args_cli.num_episodes:
        # ── Handle reset / save ──────────────────────────────────────
        if _keys["save"]:
            _keys["save"] = False
            if frame_count > 10:  # need minimum frames
                dataset.save_episode()
                episode_count += 1
                print(f"\n[SAVED] Episode {episode_count}/{args_cli.num_episodes} "
                      f"({frame_count} frames)\n")
            else:
                print(f"[SKIP] Episode too short ({frame_count} frames)")
            obs, _ = env.reset()
            _keys.update({k: False for k in _keys})
            current_v_x = 0.0
            step = 0
            frame_count = 0
            continue

        if _keys["reset"]:
            _keys["reset"] = False
            print("[DISCARD] Episode discarded.")
            # Clear any buffered frames by not calling save_episode
            obs, _ = env.reset()
            _keys.update({k: False for k in _keys})
            current_v_x = 0.0
            step = 0
            frame_count = 0
            continue

        # ── Compute action ───────────────────────────────────────────
        target_v_x = (V_MAX if _keys["forward"] else 0.0) \
                   - (V_MAX if _keys["backward"] else 0.0)

        if abs(target_v_x) > abs(current_v_x) or (target_v_x * current_v_x < 0):
            rate = ACCEL * STEP_DT
        else:
            rate = DECEL * STEP_DT
        if current_v_x < target_v_x:
            current_v_x = min(current_v_x + rate, target_v_x)
        else:
            current_v_x = max(current_v_x - rate, target_v_x)

        v_x = current_v_x
        steer = (STEER_ANGLE if _keys["left"] else 0.0) \
              - (STEER_ANGLE if _keys["right"] else 0.0)
        omega_z = v_x * math.tan(steer) / WHEEL_BASE
        fork = (1.0 if _keys["fork_up"] else 0.0) \
             - (1.0 if _keys["fork_down"] else 0.0)

        action_env = torch.tensor([[v_x, omega_z, fork]], device=env.device)
        obs, reward, terminated, truncated, info = env.step(action_env)
        step += 1

        # ── Record frame ─────────────────────────────────────────────
        # Build action vector: [v_forward, yaw_rate, fork_lift_vel, fork_tilt_vel, attach_toggle]
        action_vec = np.array([v_x, omega_z, fork, 0.0, 0.0], dtype=np.float32)

        # Images: convert from torch RGBA to PIL RGB
        def _to_pil(tensor_rgba):
            arr = tensor_rgba[0].cpu().numpy()
            if arr.shape[-1] == 4:
                arr = arr[:, :, :3]
            return Image.fromarray(arr.astype(np.uint8))

        # LiDAR range image
        lidar_hits = obs["lidar"][0].cpu().numpy()
        lidar_sensor_pos = env.lidar.data.pos_w[0].cpu().numpy()
        range_image = _lidar_to_range_image(
            lidar_hits, lidar_sensor_pos,
            _LIDAR_CHANNELS, _LIDAR_H_SAMPLES, _LIDAR_MAX_RANGE)

        # State
        state_vec = obs["state"][0].cpu().numpy().astype(np.float32)

        frame_data = {
            "observation.images.front_cabin": _to_pil(obs["rgb_front"]),
            "observation.images.top_left": _to_pil(obs["rgb_left"]),
            "observation.images.top_right": _to_pil(obs["rgb_right"]),
            "observation.lidar": range_image,
            "observation.state": state_vec,
            "action": action_vec,
            "task": args_cli.task_instruction,
        }
        dataset.add_frame(frame_data)
        frame_count += 1

        # ── Auto-reset on timeout ────────────────────────────────────
        if step >= args_cli.max_steps or terminated.any() or truncated.any():
            if frame_count > 10:
                dataset.save_episode()
                episode_count += 1
                print(f"\n[AUTO-SAVED] Episode {episode_count}/{args_cli.num_episodes} "
                      f"({frame_count} frames)\n")
            obs, _ = env.reset()
            _keys.update({k: False for k in _keys})
            current_v_x = 0.0
            step = 0
            frame_count = 0

        # Status print
        if step % 60 == 0 and step > 0:
            print(f"  ep={episode_count} step={step} frames={frame_count} | "
                  f"v={v_x:+.1f} ω={omega_z:+.2f} fork={fork:+.1f}")

    # ── Cleanup ──────────────────────────────────────────────────────
    print(f"\n[DONE] Recorded {episode_count} episodes to {ds_root}")

    # Verify: load back and check
    try:
        loaded = LeRobotDataset(ds_root)
        print(f"[VERIFY] Loaded {len(loaded)} frames, "
              f"{loaded.meta.total_episodes} episodes")
        if len(loaded) > 0:
            sample = loaded[0]
            print(f"[VERIFY] Sample keys: {list(sample.keys())}")
            for k, v in sample.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: shape={v.shape} dtype={v.dtype}")
                elif hasattr(v, 'size'):
                    print(f"  {k}: size={v.size}")
                else:
                    print(f"  {k}: {type(v).__name__}")
    except Exception as e:
        print(f"[WARN] Verification failed: {e}")

    _input.unsubscribe_to_keyboard_events(_keyboard, _kb_sub)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
