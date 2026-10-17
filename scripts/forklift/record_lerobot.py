"""
Record forklift teleoperation episodes in LeRobot v2/v3 format.

Produces a dataset at datasets/forklift_teleop/ compatible with
LeRobot's LeRobotDataset loader and, after conversion, with OpenPI.

Usage:
    ./isaaclab.sh -p scripts/forklift/record_lerobot.py [--num_episodes 10]
    ./isaaclab.sh -p scripts/forklift/record_lerobot.py --headless  # no GUI

Schema (image-only — cameras are the only sensors):
    observation.images.front_cabin  — 224×224 RGB (primary VLA camera)
    observation.images.top_left     — 224×224 RGB
    observation.images.top_right    — 224×224 RGB
    observation.state               — (8,) float32 proprioception
    action                          — (5,) float32 teleop commands
    task                            — str (natural language instruction)

One frame is recorded per env.step, so the dataset fps equals the 30 Hz
control rate.
"""

# ---------------------------------------------------------------------------
# Step 1: AppLauncher must come before ALL other Isaac Lab imports
# ---------------------------------------------------------------------------

import argparse
# Load h5py's HDF5 before Isaac Sim starts: the windowed app loads its own
# hdf5.dll (isaacsim.sensors.rtx), and on Windows Isaac Lab's later
# `import h5py` then fails with "DLL load failed". Harmless elsewhere.
import h5py  # noqa: F401
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record forklift teleop to LeRobot dataset.")
parser.add_argument("--num_episodes", type=int, default=10,
                    help="Number of episodes to record.")
parser.add_argument("--max_steps", type=int, default=3000,
                    help="Max steps per episode before auto-reset.")
parser.add_argument("--task_instruction", type=str,
                    default="pick up the blue pallet and place it on top of the brown pallet",
                    help="Fallback task instruction (used only if --prompt-template is empty).")
parser.add_argument("--dataset_dir", type=str, default="datasets/forklift_teleop",
                    help="Output directory for the dataset.")

# Language-instruction templating + randomization
parser.add_argument("--prompt-template", type=str, default=None,
                    help="format-string with {cx}{cy}{cz}{cyaw_deg}{fx}{fy}{fz}"
                         "{fyaw_deg}{pallet_idx}{dx}{dy}{distance_m}{bearing_deg}. "
                         "Default: prompt_builder.DEFAULT_TEMPLATE.")
parser.add_argument("--prompt-include-forklift", type=lambda s: s.lower() != "false",
                    default=True,
                    help="If false, template must not reference any {f*} placeholders.")
parser.add_argument("--target-pallet", type=int, default=0,
                    help="Index of the interactable pallet that is the goal.")
parser.add_argument("--randomize-spawn", action="store_true",
                    help="Randomize forklift spawn pose each episode.")
parser.add_argument("--randomize-cargo", type=lambda s: s.lower() != "false",
                    default=True,
                    help="Randomize cargo positions each episode (default ON).")
parser.add_argument("--spawn-seed", type=int, default=0,
                    help="Master seed; spawn uses this, cargo uses spawn_seed+1.")
parser.add_argument("--spawn-region", type=str, default="-2,-3,2,3",
                    help="Forklift spawn region 'xmin,ymin,xmax,ymax' (env-local).")
parser.add_argument("--cargo-region", type=str, default="5,-3,11,3",
                    help="Cargo spawn region 'xmin,ymin,xmax,ymax' (env-local).")

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
from forklift_env import _N_INTERACTABLE
from randomizers import SpawnRandomizer, CargoRandomizer, parse_region
import prompt_builder
import json

# LeRobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Keyboard input (same as teleop)
import carb.input
import omni.appwindow

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

# State vector: [x, y, yaw, vx, vy, omega_z, fork_height, grabbed_idx]
_STATE_DIM = 8

# Action vector: [v_forward, yaw_rate, fork_lift_vel, fork_tilt_vel, attach_toggle]
# fork_tilt_vel and attach_toggle are reserved (always 0 for now)
_ACTION_DIM = 5

# LeRobot v3 datasets are only loadable after finalize() (it writes the
# parquet footers). Datasets registered here are finalized on every exit
# path, including crashes and Ctrl+C (see _finalize_lerobot / __main__).
_LEROBOT_DATASETS: list = []

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


def _finalize_lerobot() -> None:
    """Finalize every registered dataset once (LeRobot v3 needs this to
    write valid files; v2.1 has no finalize() and needs nothing)."""
    while _LEROBOT_DATASETS:
        ds = _LEROBOT_DATASETS.pop()
        if hasattr(ds, "finalize"):
            ds.finalize()


# ---------------------------------------------------------------------------
# Main recording loop
# ---------------------------------------------------------------------------

def _make_extras_writer(ds_root: str):
    """Return a callable(record_dict) that appends one JSON line."""
    path = os.path.join(ds_root, "extras_episodes.jsonl")
    os.makedirs(ds_root, exist_ok=True)

    def _write(record: dict) -> None:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    return _write, path


def _resolve_task_index(dataset, prompt: str) -> int:
    """Best-effort lookup of the task_index LeRobot assigned to a string."""
    meta = getattr(dataset, "meta", None)
    if meta is None:
        return -1
    # get_task_index() exists in LeRobot v2.1 and v3 (v3 keeps tasks in a
    # DataFrame, which the fallbacks below don't understand).
    get_task_index = getattr(meta, "get_task_index", None)
    if callable(get_task_index):
        idx = get_task_index(prompt)
        if idx is not None:
            return int(idx)
    task_to_idx = getattr(meta, "task_to_task_index", None)
    if isinstance(task_to_idx, dict) and prompt in task_to_idx:
        return int(task_to_idx[prompt])
    tasks = getattr(meta, "tasks", None)
    if isinstance(tasks, dict):
        for k, v in tasks.items():
            if v == prompt:
                return int(k)
    if isinstance(tasks, list):
        try:
            return tasks.index(prompt)
        except ValueError:
            return -1
    return -1


def main():
    env_cfg = ForkliftEnvCfg()
    env = ForkliftEnv(env_cfg)
    _input, _keyboard, _kb_sub = _make_keyboard_handler()

    # ── Randomizers (only used when the corresponding flag is set) ────
    spawn_rng = SpawnRandomizer(
        region_xyxy=parse_region(args_cli.spawn_region),
        seed=args_cli.spawn_seed,
    )
    cargo_rng = CargoRandomizer(
        region_xyxy=parse_region(args_cli.cargo_region),
        seed=args_cli.spawn_seed + 1,
    )
    template = args_cli.prompt_template or prompt_builder.DEFAULT_TEMPLATE

    # Distance from forklift root to pallet root when tines are inserted in
    # the pocket. Matches the fixed-scene default (forklift at 8.5, pallet at 10).
    _PARK_DIST = 1.5

    def _begin_episode():
        """Sample spawn + cargo, set env overrides, render the prompt.

        Caller must call env.reset() AFTER this returns.
        Returns (prompt_str, placeholders, episode_meta).

        Default behaviour: cargo is sampled (or fixed), then the forklift
        is parked 1.5 m behind the cargo, facing it, with tines in the pocket.
        --randomize-spawn decouples the forklift pose from the cargo.
        """
        # Cargo first (need its pose to position the forklift under it).
        if args_cli.randomize_cargo:
            # Use a far-away dummy "forklift" point so the cargo keepout
            # does not interfere with the to-be-derived spawn pose.
            cargo_xy_yaw = cargo_rng.sample(_N_INTERACTABLE, (-1e6, -1e6))
        else:
            cargo_xy_yaw = [(10.0, 0.0, 0.0)]

        ti = max(0, min(args_cli.target_pallet, _N_INTERACTABLE - 1))
        cx, cy, cyaw = cargo_xy_yaw[ti]

        # Forklift spawn — parked under the target cargo by default,
        # or independently randomized when --randomize-spawn is set.
        if args_cli.randomize_spawn:
            sx, sy, syaw = spawn_rng.sample()
        else:
            sx = cx - _PARK_DIST * math.cos(cyaw)
            sy = cy - _PARK_DIST * math.sin(cyaw)
            syaw = cyaw

        env.cfg.spawn_override = (sx, sy, syaw)
        env.cfg.cargo_override = tuple(cargo_xy_yaw)

        # Target pallet → render prompt from its (quantized) coords
        rendered, placeholders = prompt_builder.render(
            template=template,
            cargo_xyz=(cx, cy, 0.0),
            cargo_yaw_rad=cyaw,
            forklift_xyz=(sx, sy, 0.0),
            forklift_yaw_rad=syaw,
            pallet_idx=ti,
            include_forklift=args_cli.prompt_include_forklift,
        )

        meta = {
            "spawn": {"x": sx, "y": sy, "yaw_rad": syaw,
                      "seed": args_cli.spawn_seed},
            "target": {
                "pallet_idx": ti,
                "cargo_xyz": [cx, cy, 0.0],
                "cargo_yaw_rad": cyaw,
                "cargo_xyz_quantized": [
                    placeholders["cx"], placeholders["cy"], placeholders["cz"],
                ],
            },
            "spawn_to_cargo": {
                "dx": placeholders.get("dx", cx - sx),
                "dy": placeholders.get("dy", cy - sy),
                "distance_m": placeholders.get("distance_m",
                                               math.hypot(cx - sx, cy - sy)),
                "bearing_deg": placeholders.get("bearing_deg", 0.0),
            },
            "language_template": template,
        }
        return rendered, meta

    # ── Create LeRobot dataset ────────────────────────────────────────
    ds_root = os.path.abspath(args_cli.dataset_dir)
    print(f"[INFO] Recording to: {ds_root}")

    dataset = LeRobotDataset.create(
        repo_id="forklift/teleop",
        # One frame per env.step → fps = control rate (30 Hz). Timestamps
        # must be right: OmniVLA-style waypoint labels are sampled by time.
        fps=round(1.0 / (env.cfg.sim.dt * env.cfg.decimation)),
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
    _LEROBOT_DATASETS.append(dataset)

    extras_write, extras_path = _make_extras_writer(ds_root)

    current_prompt, current_meta = _begin_episode()
    obs, _ = env.reset()
    print(f"[EPISODE] task: {current_prompt}")

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
                ep_idx = episode_count       # 0-based, matches LeRobot convention
                episode_count += 1
                extras_write({
                    "episode_index": ep_idx,
                    "task_index": _resolve_task_index(dataset, current_prompt),
                    "tasks": [current_prompt],
                    **current_meta,
                    "distractors": list(getattr(env, "last_distractors", [])),
                    "length": frame_count,
                })
                print(f"\n[SAVED] Episode {episode_count}/{args_cli.num_episodes} "
                      f"({frame_count} frames)\n")
            else:
                print(f"[SKIP] Episode too short ({frame_count} frames)")
                # LeRobot keeps buffered frames until save/clear — drop them
                # so they don't leak into the next saved episode.
                dataset.clear_episode_buffer()
            current_prompt, current_meta = _begin_episode()
            obs, _ = env.reset()
            print(f"[EPISODE] task: {current_prompt}")
            _keys.update({k: False for k in _keys})
            current_v_x = 0.0
            step = 0
            frame_count = 0
            continue

        if _keys["reset"]:
            _keys["reset"] = False
            print("[DISCARD] Episode discarded.")
            # Not calling save_episode() is not enough: LeRobot keeps the
            # buffered frames and would prepend them to the next episode.
            dataset.clear_episode_buffer()
            current_prompt, current_meta = _begin_episode()
            obs, _ = env.reset()
            print(f"[EPISODE] task: {current_prompt}")
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

        # State
        state_vec = obs["state"][0].cpu().numpy().astype(np.float32)

        frame_data = {
            "observation.images.front_cabin": _to_pil(obs["rgb_front"]),
            "observation.images.top_left": _to_pil(obs["rgb_left"]),
            "observation.images.top_right": _to_pil(obs["rgb_right"]),
            "observation.state": state_vec,
            "action": action_vec,
            "task": current_prompt,
        }
        dataset.add_frame(frame_data)
        frame_count += 1

        # ── Auto-reset on timeout ────────────────────────────────────
        if step >= args_cli.max_steps or terminated.any() or truncated.any():
            if frame_count > 10:
                dataset.save_episode()
                ep_idx = episode_count
                episode_count += 1
                extras_write({
                    "episode_index": ep_idx,
                    "task_index": _resolve_task_index(dataset, current_prompt),
                    "tasks": [current_prompt],
                    **current_meta,
                    "distractors": list(getattr(env, "last_distractors", [])),
                    "length": frame_count,
                })
                print(f"\n[AUTO-SAVED] Episode {episode_count}/{args_cli.num_episodes} "
                      f"({frame_count} frames)\n")
            else:
                dataset.clear_episode_buffer()   # too short — don't leak frames
            current_prompt, current_meta = _begin_episode()
            obs, _ = env.reset()
            print(f"[EPISODE] task: {current_prompt}")
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

    # Drop an unfinished episode (window closed mid-episode), then finalize —
    # LeRobot v3 can't load the dataset until finalize() has run.
    if frame_count > 0:
        dataset.clear_episode_buffer()
    _finalize_lerobot()

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
    try:
        main()
    finally:
        # Keep the episodes saved so far loadable even after a crash or Ctrl+C.
        _finalize_lerobot()
    simulation_app.close()
