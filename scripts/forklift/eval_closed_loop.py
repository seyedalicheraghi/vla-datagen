"""Closed-loop test: put a policy in the driver's seat and score it.

Every control step (30 Hz):

    simulator ──► observation (3 camera images + state + prompt)
        ▲                           │
        │                           ▼
    env.step(action) ◄──────── policy.act(obs)

The observation is built with the same transform the dataset writer uses
(media.dataset_image) and the same prompt wording (prompt_builder), so a model
trained on the LeRobot dataset sees exactly what it saw in training. Scenes
come from randomizers.sample_scene with held-out seeds (collection uses 42+,
evaluation defaults to 1000+).

Success is judged by the simulator, not by the policy: the target cargo must
be lifted at least LIFT_SUCCESS_M off the floor and then set back down
(released at floor level) within --max_steps.

    ./isaaclab.sh -p scripts/forklift/eval_closed_loop.py --headless --num_episodes 20
    ./isaaclab.sh -p scripts/forklift/eval_closed_loop.py --video outputs/blog/closed_loop.mp4

To test a trained model, add a class with `reset(episode)` and
`act(obs) -> (v_x, omega_z, fork_cmd)` (see policies.Policy) to make_policy().
"""

# ---------------------------------------------------------------------------
# Step 1 — AppLauncher must come before any Isaac Lab imports
# ---------------------------------------------------------------------------

import argparse
# Load h5py's HDF5 before Isaac Sim starts: the windowed app loads its own
# hdf5.dll (isaacsim.sensors.rtx), and on Windows Isaac Lab's later
# `import h5py` then fails with "DLL load failed". Harmless elsewhere.
import h5py  # noqa: F401
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Closed-loop evaluation of a forklift policy.")
parser.add_argument("--policy", default="scripted", choices=["scripted"],
                    help="Who drives. 'scripted' is the privileged expert — a "
                         "stand-in until a trained model is plugged in.")
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=1000,
                    help="First scene seed. Held out: data collection uses 42+.")
parser.add_argument("--max_steps", type=int, default=900,
                    help="Per-episode budget in control steps (900 = 30 s).")
parser.add_argument("--cargo_region", type=str, default="-3,-4,3,4")
parser.add_argument("--n-distractors-min", type=int, default=3)
parser.add_argument("--n-distractors-max", type=int, default=12)
parser.add_argument("--distractor-region", type=str, default="-13,-13,13,13")
parser.add_argument("--camera_size", type=int, default=448,
                    help="Render size (square). Must match data collection.")
parser.add_argument("--image_size", type=int, default=224,
                    help="Image size given to the policy (= the dataset's).")
parser.add_argument("--video", type=str, default="",
                    help="Write demo video (chase view + policy inputs). A path "
                         "ending in .mp4 = one video; a folder = one clip per episode.")
parser.add_argument("--results", type=str, default="",
                    help="Write per-episode results as JSON lines here.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Step 2 — Imports after Isaac Sim is up
# ---------------------------------------------------------------------------

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from forklift_env import ForkliftEnv, ForkliftEnvCfg
from media import DemoVideo, dataset_image, to_rgb_uint8, video_path
from policies import ScriptedExpert
from randomizers import DistractorRandomizer, parse_region, sample_scene
import prompt_builder

LIFT_SUCCESS_M = 1.0    # pallet bottom must rise at least this far off the floor
FLOOR_TOL_M = 0.02      # "set back down" = released with its bottom below this

# Dataset camera name → env observation key (same as scripted_collect.py).
_DATASET_CAMS = (("front_cabin", "rgb_front"),
                 ("top_left",    "rgb_left"),
                 ("top_right",   "rgb_right"))

# Videos are closed on every exit path (an unclosed MP4 is unplayable).
_OPEN_OUTPUTS: list = []


def make_policy(name: str, dt: float):
    if name == "scripted":
        return ScriptedExpert(dt)
    raise ValueError(f"unknown policy {name!r}")


def policy_observation(obs_env: dict, image_size: int, prompt: str):
    """Env observation → what the policy gets (identical to a dataset frame)."""
    images = {name: dataset_image(to_rgb_uint8(obs_env[key]), image_size)
              for name, key in _DATASET_CAMS}
    state = obs_env["state"][0].cpu().numpy().astype(np.float32)
    obs = {"images": {k: np.asarray(v) for k, v in images.items()},
           "state": state, "prompt": prompt}
    return obs, images


def run_episode(env, policy, seed: int, ep: int, tally: dict,
                video: DemoVideo | None) -> dict:
    scene = sample_scene(seed, parse_region(args_cli.cargo_region))
    cx, cy, cyaw = scene["cargo"]
    sx, sy, syaw = scene["spawn"]
    env.cfg.spawn_override = (sx, sy, syaw)
    env.cfg.cargo_override = ((cx, cy, cyaw),)
    env.cfg.distractor_keepout_xys = tuple(scene["keepout"])
    env.set_distractor_randomizer(DistractorRandomizer(
        region_xyxy=env.cfg.distractor_region, seed=seed + 10_000, keepout=2.2))
    obs_env, _ = env.reset()

    prompt, _ = prompt_builder.render(
        prompt_builder.DRIVE_LIFT_TEMPLATE,
        cargo_xyz=(cx, cy, 0.0), cargo_yaw_rad=cyaw,
        forklift_xyz=(sx, sy, 0.0), forklift_yaw_rad=syaw)
    # The scripted expert reads the target from here (privileged); a learned
    # policy only uses the observation: images + state + prompt.
    policy.reset({"target": {"cargo_xyz": [cx, cy, 0.0], "cargo_yaw_rad": cyaw},
                  "prompt": prompt})

    dt = float(env.cfg.sim.dt * env.cfg.decimation)
    lifted, success, max_lift_m, steps = False, False, 0.0, 0
    for step in range(args_cli.max_steps):
        obs, images = policy_observation(obs_env, args_cli.image_size, prompt)
        action = policy.act(obs)

        if video is not None:
            badge, color = (("SUCCESS", (40, 160, 70)) if success
                            else ("POLICY IN THE LOOP", (30, 110, 200)))
            video.add(to_rgb_uint8(obs_env["rgb_chase"]), images,
                      lines=[f"Closed-loop test · policy: {args_cli.policy}",
                             f"episode {ep + 1}/{args_cli.num_episodes} · "
                             f"t = {step * dt:4.1f} s · "
                             f"successes {tally['success']}/{tally['done']}"],
                      prompt=prompt, badge=badge, badge_color=color)

        obs_env, *_ = env.step(torch.tensor([action], dtype=torch.float32,
                                            device=env.device))
        steps = step + 1

        # Judge from the simulator's state, not the policy's.
        pallet_bottom = float(env.pallets[0].data.root_pos_w[0, 2])   # root = bottom
        max_lift_m = max(max_lift_m, pallet_bottom)
        lifted = lifted or pallet_bottom >= LIFT_SUCCESS_M
        released = env._grabbed_idx[0] < 0
        if lifted and released and env._pallet_base_z[0][0] < FLOOR_TOL_M:
            success = True
            break
        if getattr(policy, "done", False):
            break

    tally["done"] += 1
    tally["success"] += int(success)
    if video is not None:
        badge = ("SUCCESS", (40, 160, 70)) if success else ("FAILED", (200, 50, 50))
        obs, images = policy_observation(obs_env, args_cli.image_size, prompt)
        video.add(to_rgb_uint8(obs_env["rgb_chase"]), images,
                  lines=[f"Closed-loop test · policy: {args_cli.policy}",
                         f"episode {ep + 1}/{args_cli.num_episodes} · "
                         f"t = {steps * dt:4.1f} s · "
                         f"successes {tally['success']}/{tally['done']}"],
                  prompt=prompt, badge=badge[0], badge_color=badge[1])
        video.hold(25)

    return {"episode": ep, "seed": seed, "success": success,
            "time_s": round(steps * dt, 2), "max_lift_m": round(max_lift_m, 3),
            "prompt": prompt, "cargo": [cx, cy, cyaw], "spawn": [sx, sy, syaw],
            "distractors": len(env.last_distractors)}


def main():
    cfg = ForkliftEnvCfg()
    cfg.camera_resolution = (args_cli.camera_size, args_cli.camera_size)
    cfg.n_distractors_min = int(args_cli.n_distractors_min)
    cfg.n_distractors_max = int(args_cli.n_distractors_max)
    cfg.distractor_region = parse_region(args_cli.distractor_region)
    cfg.enable_chase_cam = bool(args_cli.video)
    env = ForkliftEnv(cfg)
    dt = float(env.cfg.sim.dt * env.cfg.decimation)

    if not args_cli.headless and cfg.enable_chase_cam:
        try:
            from omni.kit.viewport.utility import get_active_viewport
            vp = get_active_viewport()
            if vp is not None:
                vp.set_active_camera("/World/envs/env_0/CamChase")
        except Exception as e:
            print(f"[viewport] could not switch active camera: {e}")

    video = None                       # one video for the whole run (.mp4 path)
    per_episode = False
    if args_cli.video:
        path, per_episode = video_path(args_cli.video, 0, "closed_loop")
        if not per_episode:
            video = DemoVideo(path, fps=round(1.0 / dt))
            _OPEN_OUTPUTS.append(video)

    policy = make_policy(args_cli.policy, dt)
    tally = {"done": 0, "success": 0}
    results = []
    for ep in range(args_cli.num_episodes):
        ep_video = video
        if per_episode:                               # one short clip per episode
            ep_video = DemoVideo(video_path(args_cli.video, ep, "closed_loop")[0],
                                 fps=round(1.0 / dt))
            _OPEN_OUTPUTS.append(ep_video)
        r = run_episode(env, policy, args_cli.seed + ep, ep, tally, ep_video)
        if per_episode:
            _OPEN_OUTPUTS.remove(ep_video)
            ep_video.close()
        results.append(r)
        print(f"[EVAL {ep}] seed={r['seed']}  success={r['success']}  "
              f"time={r['time_s']:.1f}s  max_lift={r['max_lift_m']:.2f}m  "
              f"distractors={r['distractors']}", flush=True)

    if args_cli.results:
        rp = Path(os.path.expanduser(args_cli.results))
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text("".join(json.dumps(r) + "\n" for r in results))

    wins = [r for r in results if r["success"]]
    print("\n" + "=" * 60)
    print(f"CLOSED-LOOP RESULT  policy={args_cli.policy}  "
          f"seeds {args_cli.seed}..{args_cli.seed + args_cli.num_episodes - 1}")
    print(f"success rate : {len(wins)}/{len(results)} "
          f"({100.0 * len(wins) / max(len(results), 1):.0f}%)")
    if wins:
        print(f"mean time    : {np.mean([r['time_s'] for r in wins]):.1f} s (successful episodes)")
    print("=" * 60, flush=True)

    # Isaac Sim's simulation_app.close() can hang; close outputs, then exit.
    while _OPEN_OUTPUTS:
        _OPEN_OUTPUTS.pop().close()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
        while _OPEN_OUTPUTS:
            try:
                _OPEN_OUTPUTS.pop().close()
            except BaseException:
                traceback.print_exc()
        os._exit(1)
