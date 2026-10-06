"""Autonomous forklift episodes → ROS 2 mcap bags.

For N episodes:
  1. Reset env with a fresh per-episode seed → fresh forklift + cargo poses.
  2. A scripted FSM policy drives forklift → cargo, lifts to max, lowers.
  3. Per-step state, actions, cameras and TF written to a standalone
     mcap bag at <out_dir>/episode_NNN/.
  4. After all episodes, write summary.jsonl + README.md in <out_dir>.

Replay one bag with:
    source /opt/ros/jazzy/setup.bash
    ros2 bag play <out_dir>/episode_000/

This script does NOT use rclpy. It writes bags via the pure-Python
`rosbags` library so it runs inside the Isaac Sim venv.
"""

# ---------------------------------------------------------------------------
# Step 1 — AppLauncher must come before any Isaac Lab imports
# ---------------------------------------------------------------------------

import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted lift+lower → mcap bags.")
parser.add_argument("--num_episodes", type=int, default=5)
parser.add_argument("--out_dir", type=str,
                    default="~/datasets/forklift_lift_lower")
parser.add_argument("--base_seed", type=int, default=42)
parser.add_argument("--cargo_region", type=str, default="5,-3,11,3",
                    help="xmin,ymin,xmax,ymax (env-local). Forklift is "
                         "auto-parked 1.5 m behind the sampled cargo.")
parser.add_argument("--max_steps_per_ep", type=int, default=1500)
parser.add_argument("--image_rate_hz", type=float, default=10.0,
                    help="Throttle bag image rate (env runs at 30 Hz). "
                         "Lower keeps bag size manageable.")
# Keep the render square: LeRobot frames are then a clean downscale with the
# same field of view as the env's native 224x224 cameras. Non-square renders
# are center-cropped before resizing (never squashed).
parser.add_argument("--camera_w", type=int, default=448)
parser.add_argument("--camera_h", type=int, default=448)
parser.add_argument("--n-distractors-min", type=int, default=3)
parser.add_argument("--n-distractors-max", type=int, default=12)
parser.add_argument("--distractor-region", type=str, default="0,-8,15,8",
                    help="xmin,ymin,xmax,ymax for distractor cargo placement.")

# Output toggles (both default true — pass --ros false / --lerobot false
# to skip either writer).
_BOOL = lambda s: s.lower() not in ("false", "0", "no", "off")
parser.add_argument("--ros", type=_BOOL, default=True,
                    help="Write a ROS 2 (Robot Operating System v2) mcap bag "
                         "to --out_dir. Default true.")
parser.add_argument("--lerobot", type=_BOOL, default=True,
                    help="Write a LeRobot (Le Robot) dataset for OpenPI 0.5 "
                         "(Pi-Zero-Five) finetuning to --lerobot_dir. "
                         "Default true.")
parser.add_argument("--lerobot_dir", type=str,
                    default="~/datasets/forklift_lift_lower_lerobot",
                    help="Output directory for the LeRobot dataset.")
parser.add_argument("--lerobot_image_size", type=int, default=224,
                    help="Image resolution stored in LeRobot (square). "
                         "OpenPI 0.5 expects 224.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Step 2 — Imports after Isaac Sim is up
# ---------------------------------------------------------------------------

import json
import math
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from forklift_env import ForkliftEnv, ForkliftEnvCfg, _N_INTERACTABLE
from randomizers import CargoRandomizer, DistractorRandomizer, parse_region
import prompt_builder

from PIL import Image as PILImage
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rosbags.rosbag2 import Writer, StoragePlugin
from rosbags.typesys import Stores, get_typestore

# LeRobot tensor sizes (must match record_lerobot.py).
_STATE_DIM = 8
_ACTION_DIM = 5

# LeRobot v3 datasets are only loadable after finalize() (it writes the
# parquet footers). Datasets registered here are finalized on every exit
# path, including crashes and Ctrl+C (see _finalize_lerobot / __main__).
_LEROBOT_DATASETS: list = []

TYPESTORE = get_typestore(Stores.ROS2_JAZZY)

# ---------------------------------------------------------------------------
# Camera mount offsets in base_link frame.
# Mirrors forklift_env.py::_update_driver_cam (lines ~1305-1372). If those
# constants change, update these to match.
# ---------------------------------------------------------------------------

_CAM_MOUNTS = {
    "cam_front":     {"pos": (0.30,  0.00, 1.80), "yaw":  0.0,  "pitch": -0.60},
    "cam_top_left":  {"pos": (0.20,  0.60, 2.30), "yaw":  0.4,  "pitch": -0.35},
    "cam_top_right": {"pos": (0.20, -0.60, 2.30), "yaw": -0.4,  "pitch": -0.35},
}

WORLD_FRAME = "world"
BASE_FRAME = "base_link"


# ---------------------------------------------------------------------------
# Quaternion helpers (w, x, y, z)
# ---------------------------------------------------------------------------

def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def yaw_pitch_to_quat(yaw: float, pitch: float) -> tuple[float, float, float, float]:
    cY, sY = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    cP, sP = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    # ZYX convention matching env._yaw_pitch_quat
    return (cY * cP, sY * sP, -cY * sP, sY * cP)


# ---------------------------------------------------------------------------
# rosbags message factories
# ---------------------------------------------------------------------------

def _stamp(t_ns: int):
    Time = TYPESTORE.types["builtin_interfaces/msg/Time"]
    return Time(sec=int(t_ns // 1_000_000_000), nanosec=int(t_ns % 1_000_000_000))


def _header(t_ns: int, frame: str):
    Header = TYPESTORE.types["std_msgs/msg/Header"]
    return Header(stamp=_stamp(t_ns), frame_id=frame)


def _string_msg(s: str):
    return TYPESTORE.types["std_msgs/msg/String"](data=s)


def _float32_msg(v: float):
    return TYPESTORE.types["std_msgs/msg/Float32"](data=float(v))


def _int32_msg(v: int):
    return TYPESTORE.types["std_msgs/msg/Int32"](data=int(v))


def _float32_array_msg(arr, dim_names: list[str]):
    Float32MultiArray = TYPESTORE.types["std_msgs/msg/Float32MultiArray"]
    MultiArrayLayout = TYPESTORE.types["std_msgs/msg/MultiArrayLayout"]
    MultiArrayDimension = TYPESTORE.types["std_msgs/msg/MultiArrayDimension"]
    arr = np.asarray(arr, dtype=np.float32).flatten()
    dim = MultiArrayDimension(label=dim_names[0], size=int(arr.size), stride=int(arr.size))
    layout = MultiArrayLayout(dim=[dim], data_offset=0)
    return Float32MultiArray(layout=layout, data=arr)


def _vec3(x, y, z):
    return TYPESTORE.types["geometry_msgs/msg/Vector3"](x=float(x), y=float(y), z=float(z))


def _point(x, y, z):
    return TYPESTORE.types["geometry_msgs/msg/Point"](x=float(x), y=float(y), z=float(z))


def _quat(w, x, y, z):
    return TYPESTORE.types["geometry_msgs/msg/Quaternion"](
        x=float(x), y=float(y), z=float(z), w=float(w))


def _pose_msg(pos, quat):
    Pose = TYPESTORE.types["geometry_msgs/msg/Pose"]
    return Pose(position=_point(*pos), orientation=_quat(*quat))


def _twist_msg(lin, ang):
    Twist = TYPESTORE.types["geometry_msgs/msg/Twist"]
    return Twist(linear=_vec3(*lin), angular=_vec3(*ang))


def _pose_stamped_msg(t_ns: int, frame: str, pos, quat):
    PoseStamped = TYPESTORE.types["geometry_msgs/msg/PoseStamped"]
    return PoseStamped(header=_header(t_ns, frame), pose=_pose_msg(pos, quat))


def _odometry_msg(t_ns: int, pos, quat, lin, ang):
    Odometry = TYPESTORE.types["nav_msgs/msg/Odometry"]
    PoseWithCovariance = TYPESTORE.types["geometry_msgs/msg/PoseWithCovariance"]
    TwistWithCovariance = TYPESTORE.types["geometry_msgs/msg/TwistWithCovariance"]
    cov36 = np.zeros(36, dtype=np.float64)
    return Odometry(
        header=_header(t_ns, WORLD_FRAME),
        child_frame_id=BASE_FRAME,
        pose=PoseWithCovariance(pose=_pose_msg(pos, quat), covariance=cov36),
        twist=TwistWithCovariance(twist=_twist_msg(lin, ang), covariance=cov36),
    )


def _joint_state_msg(t_ns: int, joints: dict):
    JointState = TYPESTORE.types["sensor_msgs/msg/JointState"]
    names = list(joints.keys())
    pos = np.array([joints[n][0] for n in names], dtype=np.float64)
    vel = np.array([joints[n][1] for n in names], dtype=np.float64)
    eff = np.zeros(len(names), dtype=np.float64)
    return JointState(header=_header(t_ns, BASE_FRAME),
                      name=names, position=pos, velocity=vel, effort=eff)


def _tf_msg(t_ns: int, parent: str, child: str, pos, quat):
    TFMessage = TYPESTORE.types["tf2_msgs/msg/TFMessage"]
    TransformStamped = TYPESTORE.types["geometry_msgs/msg/TransformStamped"]
    Transform = TYPESTORE.types["geometry_msgs/msg/Transform"]
    tf = TransformStamped(
        header=_header(t_ns, parent),
        child_frame_id=child,
        transform=Transform(translation=_vec3(*pos), rotation=_quat(*quat)),
    )
    return TFMessage(transforms=[tf])


def _tf_static_bundle(t_ns: int, transforms: list):
    """transforms: list of (parent, child, pos, quat). Returns one TFMessage."""
    TFMessage = TYPESTORE.types["tf2_msgs/msg/TFMessage"]
    TransformStamped = TYPESTORE.types["geometry_msgs/msg/TransformStamped"]
    Transform = TYPESTORE.types["geometry_msgs/msg/Transform"]
    out = []
    for parent, child, pos, quat in transforms:
        out.append(TransformStamped(
            header=_header(t_ns, parent),
            child_frame_id=child,
            transform=Transform(translation=_vec3(*pos), rotation=_quat(*quat)),
        ))
    return TFMessage(transforms=out)


def _image_msg(t_ns: int, frame: str, rgb_uint8: np.ndarray):
    Image = TYPESTORE.types["sensor_msgs/msg/Image"]
    h, w, c = rgb_uint8.shape
    assert c == 3 and rgb_uint8.dtype == np.uint8
    return Image(
        header=_header(t_ns, frame),
        height=h, width=w, encoding="rgb8",
        is_bigendian=0, step=w * 3,
        data=np.ascontiguousarray(rgb_uint8).reshape(-1),
    )


# ---------------------------------------------------------------------------
# Topic schema and connection setup
# ---------------------------------------------------------------------------

TOPIC_SCHEMA = {
    "/tf_static":                  "tf2_msgs/msg/TFMessage",
    "/tf":                         "tf2_msgs/msg/TFMessage",
    "/forklift/odom":              "nav_msgs/msg/Odometry",
    "/forklift/joint_states":      "sensor_msgs/msg/JointState",
    "/forklift/pose":              "geometry_msgs/msg/PoseStamped",
    "/forklift/proprio":           "std_msgs/msg/Float32MultiArray",
    "/action/cmd":                 "geometry_msgs/msg/Twist",
    "/action/fork":                "std_msgs/msg/Float32",
    "/action/state":               "std_msgs/msg/String",
    "/grab/state":                 "std_msgs/msg/Int32",
    "/grab/event":                 "std_msgs/msg/String",
    "/task":                       "std_msgs/msg/String",
    "/target/cargo_pose":          "geometry_msgs/msg/PoseStamped",
    "/spawn":                      "std_msgs/msg/String",
    "/camera/front/image_raw":     "sensor_msgs/msg/Image",
    "/camera/top_left/image_raw":  "sensor_msgs/msg/Image",
    "/camera/top_right/image_raw": "sensor_msgs/msg/Image",
    "/scene/distractors":          "std_msgs/msg/String",
}


def make_connections(writer):
    return {
        topic: writer.add_connection(topic, msgtype, typestore=TYPESTORE)
        for topic, msgtype in TOPIC_SCHEMA.items()
    }


def write(writer, conns, topic: str, t_ns: int, msg) -> None:
    # No-op when the bag is disabled (--ros false). Lets run_episode
    # call write() unconditionally without a forest of `if conns:` checks.
    if conns is None or writer is None:
        return
    msgtype = TOPIC_SCHEMA[topic]
    writer.write(conns[topic], t_ns, TYPESTORE.serialize_cdr(msg, msgtype))


# ---------------------------------------------------------------------------
# LeRobot helpers (mirrored from record_lerobot.py — same conventions)
# ---------------------------------------------------------------------------

def _to_pil_resized(tensor_rgba, size: int):
    """Tensor RGBA → PIL RGB, center-cropped to a square, resized to (size × size).

    Cropping first keeps the aspect ratio — resizing a 4:3 frame straight to
    a square would squash the scene.
    """
    arr = tensor_rgba[0].cpu().numpy()
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    h, w = arr.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    arr = arr[top:top + side, left:left + side]
    img = PILImage.fromarray(arr.astype(np.uint8))
    if img.size != (size, size):
        img = img.resize((size, size), PILImage.BILINEAR)
    return img


def _make_lerobot_extras_writer(ds_root: str):
    path = os.path.join(ds_root, "extras_episodes.jsonl")
    os.makedirs(ds_root, exist_ok=True)

    def _write(record: dict) -> None:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    return _write


def _resolve_task_index(dataset, prompt: str) -> int:
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


def _finalize_lerobot() -> None:
    """Finalize every registered dataset once (LeRobot v3 needs this to
    write valid files; v2.1 has no finalize() and needs nothing)."""
    while _LEROBOT_DATASETS:
        ds = _LEROBOT_DATASETS.pop()
        if hasattr(ds, "finalize"):
            ds.finalize()


def _make_lerobot_dataset(root: str, image_size: int, fps: int) -> LeRobotDataset:
    """Create a fresh image-only LeRobot dataset (cameras + state + action)."""
    return LeRobotDataset.create(
        repo_id="forklift/scripted",
        fps=fps,
        root=root,
        robot_type="forklift",
        features={
            "observation.images.front_cabin": {
                "dtype": "image",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.top_left": {
                "dtype": "image",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.top_right": {
                "dtype": "image",
                "shape": (image_size, image_size, 3),
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


# ---------------------------------------------------------------------------
# Scripted FSM policy
# ---------------------------------------------------------------------------

class LiftAndLowerPolicy:
    """ALIGN → APPROACH → INSERT → LIFT_UP → HOLD → LIFT_DOWN → DONE."""

    HOLD_STEPS = 30   # ~1 s at 30 Hz

    def __init__(self):
        self.state = "ALIGN"
        self.hold_count = 0
        self.fork_max_seen = -1.0
        self.fork_min_seen = 2.0
        self.n_grabs = 0
        self.n_drops = 0
        self._prev_grabbed = -1

    def _bookkeep(self, gidx: int, fork_h: float):
        if gidx >= 0 and self._prev_grabbed < 0:
            self.n_grabs += 1
        elif gidx < 0 and self._prev_grabbed >= 0:
            self.n_drops += 1
        self._prev_grabbed = int(gidx)
        self.fork_max_seen = max(self.fork_max_seen, fork_h)
        self.fork_min_seen = min(self.fork_min_seen, fork_h)

    def transition_event(self, gidx: int) -> str | None:
        """Return 'grab'/'drop' on the step the transition occurred."""
        if gidx >= 0 and self._prev_grabbed < 0:
            return "grab"
        if gidx < 0 and self._prev_grabbed >= 0:
            return "drop"
        return None

    def act(self, env, target: dict) -> tuple[float, float, float]:
        proprio = env._get_proprioception()[0].cpu().numpy()
        x, y, yaw, _vx, _vy, _omg, fork_h, gidx = (float(v) for v in proprio)
        # Track grab/drop events BEFORE updating state machine
        event = self.transition_event(int(gidx))
        self._bookkeep(int(gidx), fork_h)

        cx, cy, _ = target["cargo_xyz"]
        dx, dy = cx - x, cy - y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        heading_err = (bearing - yaw + math.pi) % (2 * math.pi) - math.pi

        action = (0.0, 0.0, 0.0)

        if self.state == "ALIGN":
            if abs(heading_err) < math.radians(5):
                self.state = "APPROACH"
            else:
                action = (0.0, max(-1.0, min(1.0, 1.5 * heading_err)), 0.0)

        elif self.state == "APPROACH":
            if dist < 1.6 and abs(heading_err) < math.radians(8):
                self.state = "INSERT"
            else:
                v = max(0.0, min(1.5, 0.6 * dist))
                w = max(-0.6, min(0.6, 1.0 * heading_err))
                action = (v, w, -0.5)

        elif self.state == "INSERT":
            if gidx >= 0:
                self.state = "LIFT_UP"
            else:
                # fork_cmd > 0.01 is required by env to trigger the grab
                # (forklift_env.py:1216). Park-under-cargo spawn already
                # placed the tines in the pocket, so no forward motion needed.
                action = (0.0, 0.0, 0.05)

        elif self.state == "LIFT_UP":
            if fork_h >= 1.49:
                self.state = "HOLD"
                self.hold_count = 0
            else:
                action = (0.0, 0.0, 1.0)

        elif self.state == "HOLD":
            self.hold_count += 1
            if self.hold_count >= self.HOLD_STEPS:
                self.state = "LIFT_DOWN"

        elif self.state == "LIFT_DOWN":
            if fork_h <= -0.29:
                self.state = "DONE"
            else:
                action = (0.0, 0.0, -1.0)

        # Cache event for caller
        self._last_event = event
        return action


# ---------------------------------------------------------------------------
# Episode driver
# ---------------------------------------------------------------------------

def render_prompt(spawn: dict, target: dict) -> str:
    cx, cy, cz = target["cargo_xyz"]
    cyaw = target["cargo_yaw_rad"]
    fx, fy = spawn["xy"]
    fyaw = spawn["yaw_rad"]
    template = ("lift the cargo at coordinate ({cx:.2f}, {cy:.2f}, {cz:.2f}) "
                "from forklift at ({fx:.2f}, {fy:.2f}, yaw {fyaw_deg:.0f} deg) "
                "to maximum fork height, then lower it")
    rendered, _ = prompt_builder.render(
        template=template,
        cargo_xyz=(cx, cy, cz),
        cargo_yaw_rad=cyaw,
        forklift_xyz=(fx, fy, 0.0),
        forklift_yaw_rad=fyaw,
        include_forklift=True,
    )
    return rendered


def write_static_tf(writer, conns, t_ns: int) -> None:
    transforms = []
    for child, mount in _CAM_MOUNTS.items():
        q = yaw_pitch_to_quat(mount["yaw"], mount["pitch"])
        transforms.append((BASE_FRAME, child, mount["pos"], q))
    write(writer, conns, "/tf_static", t_ns, _tf_static_bundle(t_ns, transforms))


def _rgba_to_rgb_uint8(t):
    arr = t[0].cpu().numpy()
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return arr.astype(np.uint8)


def run_episode(env, writer, conns, max_steps: int, image_period_steps: int,
                ep_idx: int, lerobot_dataset=None,
                lerobot_image_size: int = 224) -> dict:
    spawn = dict(env.last_spawn)
    target = dict(env.last_target)
    prompt = render_prompt(spawn, target)

    # Latched messages at episode start (t = 0)
    t_ns = 0
    write_static_tf(writer, conns, t_ns)
    write(writer, conns, "/task", t_ns, _string_msg(prompt))
    write(writer, conns, "/spawn", t_ns,
          _string_msg(json.dumps({"spawn": spawn, "target": target,
                                  "episode": ep_idx})))
    cx, cy, cz = target["cargo_xyz"]
    write(writer, conns, "/target/cargo_pose", t_ns,
          _pose_stamped_msg(t_ns, WORLD_FRAME, (cx, cy, cz),
                            yaw_to_quat(target["cargo_yaw_rad"])))
    write(writer, conns, "/scene/distractors", t_ns,
          _string_msg(json.dumps({"distractors": list(env.last_distractors)})))

    step_dt = float(env.cfg.sim.dt * env.cfg.decimation)   # 1/30 s
    policy = LiftAndLowerPolicy()
    obs = env._get_observations()  # initial frame after reset

    n_steps = 0
    sim_t = 0.0
    success = False

    for step in range(max_steps):
        t_ns = int(sim_t * 1e9)

        # 1. Compute action
        v_x, omega_z, fork_cmd = policy.act(env, target)
        action_t = torch.tensor([[v_x, omega_z, fork_cmd]], device=env.device)

        # 2. Per-step bag writes (BEFORE step → snapshots state at t)
        pos, quat = env.get_world_pose()
        lin, ang = env.get_world_twist()
        joints = env.get_joint_state_dict()
        proprio = env._get_proprioception()[0].cpu().numpy().astype(np.float32)
        gidx = int(proprio[7])

        write(writer, conns, "/tf",                t_ns, _tf_msg(t_ns, WORLD_FRAME, BASE_FRAME, pos, quat))
        write(writer, conns, "/forklift/odom",     t_ns, _odometry_msg(t_ns, pos, quat, lin, ang))
        write(writer, conns, "/forklift/pose",     t_ns, _pose_stamped_msg(t_ns, WORLD_FRAME, pos, quat))
        write(writer, conns, "/forklift/joint_states", t_ns, _joint_state_msg(t_ns, joints))
        write(writer, conns, "/forklift/proprio",  t_ns,
              _float32_array_msg(proprio, ["proprio8"]))
        write(writer, conns, "/action/cmd",        t_ns, _twist_msg((v_x, 0, 0), (0, 0, omega_z)))
        write(writer, conns, "/action/fork",       t_ns, _float32_msg(fork_cmd))
        write(writer, conns, "/action/state",      t_ns, _string_msg(policy.state))
        write(writer, conns, "/grab/state",        t_ns, _int32_msg(gidx))
        if policy._last_event is not None:
            write(writer, conns, "/grab/event", t_ns, _string_msg(policy._last_event))

        # Images at throttled rate (sensor traffic is the bag bulk)
        if step % image_period_steps == 0 and "rgb_front" in obs:
            for topic, key, frame in (
                ("/camera/front/image_raw",     "rgb_front", "cam_front"),
                ("/camera/top_left/image_raw",  "rgb_left",  "cam_top_left"),
                ("/camera/top_right/image_raw", "rgb_right", "cam_top_right"),
            ):
                rgb = _rgba_to_rgb_uint8(obs[key])
                write(writer, conns, topic, t_ns, _image_msg(t_ns, frame, rgb))

        # 2b. LeRobot frame (every step, image throttling N/A — Pi-Zero-Five
        # finetuning expects per-control-step samples). Skip if disabled or
        # if the env has no cameras yet (first warmup tick).
        if lerobot_dataset is not None and "rgb_front" in obs:
            action_5 = np.array(
                [v_x, omega_z, fork_cmd, 0.0, 0.0], dtype=np.float32)
            lerobot_dataset.add_frame({
                "observation.images.front_cabin":
                    _to_pil_resized(obs["rgb_front"], lerobot_image_size),
                "observation.images.top_left":
                    _to_pil_resized(obs["rgb_left"],  lerobot_image_size),
                "observation.images.top_right":
                    _to_pil_resized(obs["rgb_right"], lerobot_image_size),
                "observation.state":  proprio,
                "action":             action_5,
                "task":               prompt,
            })

        # 3. Step env
        obs, _, _, _, _ = env.step(action_t)
        n_steps += 1
        sim_t += step_dt

        if policy.state == "DONE":
            success = True
            break

    return {
        "n_steps": n_steps,
        "success": success,
        "fork_max_reached_m": float(policy.fork_max_seen),
        "fork_min_reached_m": float(policy.fork_min_seen),
        "n_grabs": int(policy.n_grabs),
        "n_drops": int(policy.n_drops),
        "duration_sec": float(n_steps * step_dt),
        "rendered_prompt": prompt,
        "spawn": spawn,
        "target": target,
        "final_state": policy.state,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(os.path.expanduser(args_cli.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ForkliftEnvCfg()
    # We bypass the env's own randomize_on_reset path because we want the
    # forklift parked under the cargo (tines in pocket) rather than
    # independently random. Compute spawn from cargo each episode and
    # write through the override fields directly.
    cfg.randomize_on_reset = False
    cfg.camera_resolution = (args_cli.camera_w, args_cli.camera_h)
    cfg.n_distractors_min = int(args_cli.n_distractors_min)
    cfg.n_distractors_max = int(args_cli.n_distractors_max)
    cfg.distractor_region = parse_region(args_cli.distractor_region)
    env = ForkliftEnv(cfg)

    # Switch the viewport from default Perspective to the in-cabin camera.
    # No-op in headless mode.
    if not args_cli.headless:
        try:
            from omni.kit.viewport.utility import get_active_viewport
            vp = get_active_viewport()
            if vp is not None:
                vp.set_active_camera("/World/envs/env_0/CamFrontCabin")
                print("[viewport] active camera → /World/envs/env_0/CamFrontCabin")
        except Exception as e:
            print(f"[viewport] could not switch active camera: {e}")

    cargo_region = parse_region(args_cli.cargo_region)

    # Distance from forklift root to pallet root when tines are in the pocket.
    # Matches the env's hardcoded default (forklift at 8.5, pallet at 10).
    PARK_DIST = 1.5

    image_period_steps = max(1, int(round(30.0 / args_cli.image_rate_hz)))

    summary_path = out_dir / "summary.jsonl"
    if summary_path.exists():
        summary_path.unlink()

    if not args_cli.ros and not args_cli.lerobot:
        print("[error] both --ros and --lerobot are false — nothing to do.",
              flush=True)
        os._exit(2)

    # ── Optional LeRobot dataset (OpenPI 0.5 finetuning) ────────────
    lerobot_dataset = None
    lerobot_extras_write = None
    lerobot_root = None
    if args_cli.lerobot:
        lerobot_root = Path(os.path.expanduser(args_cli.lerobot_dir))
        if lerobot_root.exists():
            print(f"[LeRobot] {lerobot_root} already exists — refusing to "
                  f"overwrite. Delete it or pass --lerobot_dir <new>. "
                  f"Continuing without LeRobot output.", flush=True)
        else:
            print(f"[LeRobot] writing dataset to {lerobot_root}", flush=True)
            # One LeRobot frame per env.step → fps = control rate (30 Hz).
            # Timestamps must be right: OmniVLA-style waypoint labels are
            # sampled by time from the pose history.
            lerobot_fps = round(1.0 / (env.cfg.sim.dt * env.cfg.decimation))
            lerobot_dataset = _make_lerobot_dataset(
                str(lerobot_root), args_cli.lerobot_image_size, lerobot_fps)
            _LEROBOT_DATASETS.append(lerobot_dataset)
            lerobot_extras_write = _make_lerobot_extras_writer(str(lerobot_root))

    # QoS override file for `ros2 bag play` (latched topics need
    # transient_local durability so RViz late-subscribers receive them).
    if args_cli.ros:
        qos_yaml = (
            "/tf_static: {history: keep_last, depth: 1, reliability: reliable, durability: transient_local}\n"
            "/task: {history: keep_last, depth: 1, reliability: reliable, durability: transient_local}\n"
            "/spawn: {history: keep_last, depth: 1, reliability: reliable, durability: transient_local}\n"
            "/target/cargo_pose: {history: keep_last, depth: 1, reliability: reliable, durability: transient_local}\n"
            "/scene/distractors: {history: keep_last, depth: 1, reliability: reliable, durability: transient_local}\n"
        )
        (out_dir / "qos_overrides.yaml").write_text(qos_yaml)

    rendered_prompts: list[str] = []
    seeds_used: list[int] = []
    total_bytes = 0
    n_lerobot_saved = 0   # LeRobot's own episode_index (skipped episodes don't count)

    for ep in range(args_cli.num_episodes):
        seed = args_cli.base_seed + ep

        # Sample cargo first, then derive a parked forklift pose 1.5 m
        # behind it along its yaw axis with tines aligned to the pocket.
        cargo_rng = CargoRandomizer(
            region_xyxy=cargo_region, seed=seed,
            keepout_from_forklift=2.0, keepout_between_cargo=1.0,
        )
        cargo_xy_yaw = cargo_rng.sample(_N_INTERACTABLE, (-1e6, -1e6))
        cx, cy, cyaw = cargo_xy_yaw[0]
        # 2-way GMA pallet: only the long-side lanes (between stringers)
        # accept fork tines. Snap the sampled yaw to the nearest of {0, π}
        # so the forklift's approach axis always matches a valid lane.
        cyaw = 0.0 if math.cos(cyaw) >= 0.0 else math.pi
        cargo_xy_yaw = [(cx, cy, cyaw)]
        sx = cx - PARK_DIST * math.cos(cyaw)
        sy = cy - PARK_DIST * math.sin(cyaw)
        syaw = cyaw

        env.cfg.spawn_override = (sx, sy, syaw)
        env.cfg.cargo_override = tuple(cargo_xy_yaw)
        env.last_spawn = {"xy": (sx, sy), "yaw_rad": syaw, "seed": seed}
        env.last_target = {"pallet_idx": 0,
                           "cargo_xyz": [cx, cy, 0.0],
                           "cargo_yaw_rad": cyaw}
        # Per-episode distractor randomizer (different layout each seed).
        env.set_distractor_randomizer(DistractorRandomizer(
            region_xyxy=cfg.distractor_region,
            seed=seed + 10_000,    # disjoint from cargo seed
            keepout=1.5,
        ))
        obs, _ = env.reset()

        bag_path = out_dir / f"episode_{ep:03d}"
        if args_cli.ros and bag_path.exists():
            import shutil
            shutil.rmtree(bag_path)

        print(f"\n[EP {ep}] seed={seed}  spawn={env.last_spawn}  "
              f"cargo={env.last_target}", flush=True)

        run_kwargs = dict(
            max_steps=args_cli.max_steps_per_ep,
            image_period_steps=image_period_steps,
            ep_idx=ep,
            lerobot_dataset=lerobot_dataset,
            lerobot_image_size=args_cli.lerobot_image_size,
        )
        if args_cli.ros:
            with Writer(bag_path, version=9,
                        storage_plugin=StoragePlugin.MCAP) as writer:
                conns = make_connections(writer)
                ep_summary = run_episode(env, writer, conns, **run_kwargs)
        else:
            ep_summary = run_episode(env, None, None, **run_kwargs)

        # Persist the LeRobot episode (LeRobot deduplicates the task string
        # into tasks.jsonl automatically; we add structured per-episode
        # metadata to extras_episodes.jsonl alongside it).
        if lerobot_dataset is not None and ep_summary["n_steps"] > 10:
            lerobot_dataset.save_episode()
            lerobot_extras_write({
                "episode_index": n_lerobot_saved,
                "task_index": _resolve_task_index(
                    lerobot_dataset, ep_summary["rendered_prompt"]),
                "tasks": [ep_summary["rendered_prompt"]],
                "spawn": ep_summary["spawn"],
                "target": ep_summary["target"],
                "distractors": list(getattr(env, "last_distractors", [])),
                "length": ep_summary["n_steps"],
                "success": ep_summary["success"],
                "seed": seed,
            })
            n_lerobot_saved += 1
        elif lerobot_dataset is not None:
            # Too short to keep. LeRobot holds buffered frames until
            # save/clear, so drop them or they'd leak into the next episode.
            lerobot_dataset.clear_episode_buffer()

        ep_summary.update({
            "episode": ep, "seed": seed,
            "bag_path": bag_path.name,
        })
        with open(summary_path, "a") as f:
            f.write(json.dumps(ep_summary) + "\n")

        rendered_prompts.append(ep_summary["rendered_prompt"])
        seeds_used.append(seed)

        # Approximate bag size (0 if --ros false)
        if args_cli.ros and bag_path.exists():
            ep_bytes = sum(p.stat().st_size for p in bag_path.rglob("*")
                           if p.is_file())
            total_bytes += ep_bytes
            size_str = f"  size={ep_bytes/1e6:.1f} MB"
        else:
            size_str = ""
        print(f"[EP {ep}] DONE state={ep_summary['final_state']}  "
              f"steps={ep_summary['n_steps']}  success={ep_summary['success']}"
              f"{size_str}", flush=True)

    # README
    readme = out_dir / "README.md"
    readme.write_text(
        "# forklift_lift_lower bags\n\n"
        f"{args_cli.num_episodes} mcap bags, one per episode. Each bag contains:\n"
        "- /forklift/odom, /forklift/pose, /forklift/joint_states, /forklift/proprio\n"
        "- /action/cmd, /action/fork, /action/state\n"
        "- /grab/state, /grab/event\n"
        "- /tf (per step), /tf_static (camera mounts at episode start)\n"
        "- /camera/{front,top_left,top_right}/image_raw\n"
        "- Latched: /task, /spawn, /target/cargo_pose\n\n"
        "## Replay\n\n"
        "```bash\n"
        "source /opt/ros/jazzy/setup.bash\n"
        "ros2 bag play episode_000/\n"
        "rviz2  # Fixed Frame: world; add /forklift/odom + /camera/front/image_raw\n"
        "```\n\n"
        "## Per-episode metadata\n\n"
        "See `summary.jsonl` for spawn/target poses, prompts, success flags,\n"
        "fork stats, and grab counts.\n\n"
        "## LeRobot dataset\n\n"
        + (f"Image-only training data (3 cameras + state + action) is written\n"
           f"separately to `{lerobot_root}`. Each frame is center-cropped and\n"
           f"resized to {args_cli.lerobot_image_size}x{args_cli.lerobot_image_size}.\n"
           if args_cli.lerobot else
           "LeRobot output was disabled for this run (--lerobot false).\n")
    )

    print("\n" + "=" * 60)
    print(f"COLLECTED {args_cli.num_episodes} EPISODES")
    print("=" * 60)
    print(f"Out dir   : {out_dir}")
    print(f"Total size: {total_bytes/1e6:.1f} MB")
    print(f"Seeds     : {seeds_used}")
    print("Prompts:")
    for i, p in enumerate(rendered_prompts):
        print(f"  ep {i}: {p}")
    print("=" * 60)

    # Isaac Sim's simulation_app.close() reliably hangs in some configs
    # (background USD/material threads). The bag Writer context manager
    # already flushed the bags; finalize the LeRobot dataset, then force-exit.
    _finalize_lerobot()
    print("[exit] forcing process exit", flush=True)
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as e:
        import traceback
        traceback.print_exc()
        # Keep the episodes saved so far loadable (crash or Ctrl+C).
        try:
            _finalize_lerobot()
        except BaseException:
            traceback.print_exc()
        os._exit(1)
