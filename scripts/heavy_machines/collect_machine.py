"""Language-conditioned episodes for any heavy machine → ROS 2 mcap bags + LeRobot.

Every episode: random spawn, a random instruction built from goal primitives
("swing left 90 degrees, then raise the boom to 70%, then drive forward 4 m"),
and a closed-loop scripted controller that carries it out.

    ./isaaclab.sh -p scripts/heavy_machines/collect_machine.py --machine HEX --num_episodes 5 --headless
    ./isaaclab.sh -p scripts/heavy_machines/collect_machine.py --machine wheel_loader --lerobot false

Bag contents (per episode, mcap, written with the pure-Python `rosbags` library — no rclpy):
    /robot_description  std_msgs/String       URDF (latched) → RViz2 RobotModel "Description Topic"
    /tf_static          tf2_msgs/TFMessage    fixed joints + camera mounts (latched)
    /tf                 tf2_msgs/TFMessage    world→<root link> and every movable joint, per step
    /joint_states       sensor_msgs/JointState
    /<machine>/odom     nav_msgs/Odometry     /<machine>/state  std_msgs/Float32MultiArray
    /action             std_msgs/Float32MultiArray (labels in the layout) and /action/cmd Twist
    /camera/{front,left,right}/image_raw   sensor_msgs/Image (throttled)
    /task, /machine, /spawn                std_msgs/String (latched)

Replay in RViz2:
    ros2 bag play <out>/episode_000 --qos-profile-overrides-path <out>/qos_overrides.yaml
    rviz2   # Fixed Frame: world; Add → RobotModel (Description Topic: /robot_description), TF, Image
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher  # noqa: E402

_BOOL = lambda s: s.lower() not in ("false", "0", "no", "off")  # noqa: E731
parser = argparse.ArgumentParser(description="Heavy machine episodes → ROS 2 bags + LeRobot.")
parser.add_argument("--machine", default="hydraulic_excavator")
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--base_seed", type=int, default=0)
parser.add_argument("--out_dir", default="~/datasets/heavy_machines/{machine}")
parser.add_argument("--ros", type=_BOOL, default=True, help="write mcap bags (default true)")
parser.add_argument("--lerobot", type=_BOOL, default=True, help="write a LeRobot dataset (default true)")
parser.add_argument("--lerobot_dir", default="~/datasets/heavy_machines/{machine}_lerobot")
parser.add_argument("--lerobot_image_size", type=int, default=224)
parser.add_argument("--camera_w", type=int, default=448)
parser.add_argument("--camera_h", type=int, default=448)
parser.add_argument("--image_rate_hz", type=float, default=10.0, help="bag image rate (env runs at 30 Hz)")
parser.add_argument("--goals_min", type=int, default=2)
parser.add_argument("--goals_max", type=int, default=4)
parser.add_argument("--max_steps_per_ep", type=int, default=3600)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from heavy_machines import registry  # noqa: E402
from heavy_machines.machine_env import HeavyMachineEnv, make_env_cfg  # noqa: E402
from heavy_machines.rosbag_io import EpisodeBagWriter, qos_overrides_yaml, typestore  # noqa: E402
from heavy_machines.tasks import TaskController, instruction, sample_goals  # noqa: E402
from heavy_machines.urdf_builder import build_urdf  # noqa: E402

_LEROBOT_DATASETS: list = []


# ---------------------------------------------------------------------------
# LeRobot
# ---------------------------------------------------------------------------

def make_lerobot(root: str, spec, image_size: int, fps: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    img = {"dtype": "image", "shape": (image_size, image_size, 3), "names": ["height", "width", "channel"]}
    ds = LeRobotDataset.create(
        repo_id=f"heavy_machines/{spec.name}", fps=fps, root=root, robot_type=spec.name,
        features={
            "observation.images.front": img,
            "observation.images.left": img,
            "observation.images.right": img,
            "observation.state": {"dtype": "float32", "shape": (len(spec.state_labels),),
                                  "names": list(spec.state_labels)},
            "action": {"dtype": "float32", "shape": (spec.action_dim,), "names": list(spec.action_labels)},
        },
        use_videos=True, image_writer_threads=4)
    _LEROBOT_DATASETS.append(ds)
    return ds


def finalize_lerobot():
    while _LEROBOT_DATASETS:
        ds = _LEROBOT_DATASETS.pop()
        if hasattr(ds, "finalize"):
            ds.finalize()


def to_pil_square(t, size):
    from PIL import Image
    arr = t.detach().cpu().numpy()[..., :3].astype(np.uint8)
    h, w = arr.shape[:2]
    s = min(h, w)
    arr = arr[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    img = Image.fromarray(arr)
    return img if img.size == (size, size) else img.resize((size, size), Image.BILINEAR)


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

def run_episode(env, spec, ep, seed, bag: EpisodeBagWriter | None, lerobot, image_period):
    rng = random.Random(seed)
    goals = sample_goals(spec, rng, args_cli.goals_min, args_cli.goals_max)
    prompt = instruction(spec, goals)
    dt = env.step_dt
    ctrl = TaskController(spec, goals, dt)

    obs = env._get_observations()
    if bag:
        bag.header(prompt, {"episode": ep, "seed": seed, **env.last_spawn[0]})

    step = 0
    for step in range(args_cli.max_steps_per_ep):
        state = obs["state"][0].tolist()
        action = ctrl.act(state)
        if bag:
            images = None
            if step % image_period == 0:
                images = {c.name: obs[f"rgb_{c.name}"][0].detach().cpu().numpy()[..., :3].astype(np.uint8)
                          for c in spec.cameras}
            bag.step(int(round(step * dt * 1e9)), state, action,
                     obs["joint_pos"][0].tolist(), obs["joint_vel"][0].tolist(),
                     obs["base_pose_w"][0].tolist(), obs["base_twist_body"][0].tolist(), images)

        if lerobot is not None:
            lerobot.add_frame({
                **{f"observation.images.{c.name}": to_pil_square(obs[f"rgb_{c.name}"][0],
                                                                 args_cli.lerobot_image_size)
                   for c in spec.cameras},
                "observation.state": np.asarray(state, dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "task": prompt,
            })

        if ctrl.done:
            break
        obs, _, _, _, _ = env.step(torch.tensor([action], device=env.device))

    return {"episode": ep, "seed": seed, "prompt": prompt, "steps": step + 1,
            "success": ctrl.done and ctrl.timeouts == 0, "goals_completed": ctrl.idx - ctrl.timeouts,
            "goals": len(goals), "spawn": env.last_spawn[0]}


def main():
    spec = registry.get(args_cli.machine)
    out_dir = Path(os.path.expanduser(args_cli.out_dir.format(machine=spec.name)))
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_env_cfg(spec.name, randomize_spawn=True, spawn_seed=args_cli.base_seed,
                       camera_resolution=(args_cli.camera_w, args_cli.camera_h), status_interval=0)
    cfg.sim.device = args_cli.device
    env = HeavyMachineEnv(cfg)
    fps = round(1.0 / env.step_dt)
    image_period = max(1, round(fps / args_cli.image_rate_hz))

    lerobot = None
    if args_cli.lerobot:
        root = Path(os.path.expanduser(args_cli.lerobot_dir.format(machine=spec.name)))
        if root.exists():
            print(f"[LeRobot] {root} exists — refusing to overwrite; LeRobot output disabled", flush=True)
        else:
            lerobot = make_lerobot(str(root), spec, args_cli.lerobot_image_size, fps)
            print(f"[LeRobot] writing {root}", flush=True)

    ts = typestore() if args_cli.ros else None
    if args_cli.ros:
        (out_dir / "qos_overrides.yaml").write_text(qos_overrides_yaml())
        (out_dir / f"{spec.name}.urdf").write_text(build_urdf(spec))

    summary = out_dir / "summary.jsonl"
    summary.unlink(missing_ok=True)
    for ep in range(args_cli.num_episodes):
        seed = args_cli.base_seed + ep
        env.reset()
        bag_path = out_dir / f"episode_{ep:03d}"
        if args_cli.ros:
            from rosbags.rosbag2 import StoragePlugin, Writer
            if bag_path.exists():
                shutil.rmtree(bag_path)
            with Writer(bag_path, version=9, storage_plugin=StoragePlugin.MCAP) as w:
                bag = EpisodeBagWriter(w, ts, spec, env.robot.joint_names)
                res = run_episode(env, spec, ep, seed, bag, lerobot, image_period)
        else:
            res = run_episode(env, spec, ep, seed, None, lerobot, image_period)
        if lerobot is not None:
            if res["steps"] > 10:
                lerobot.save_episode()
            else:
                lerobot.clear_episode_buffer()
        with open(summary, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"[EP {ep}] {res['steps']} steps success={res['success']}  {res['prompt']}", flush=True)

    finalize_lerobot()
    print(f"[done] {args_cli.num_episodes} episodes → {out_dir}", flush=True)
    # Isaac Sim's close() can hang on background threads; data is already flushed.
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        try:
            finalize_lerobot()
        finally:
            os._exit(1)
