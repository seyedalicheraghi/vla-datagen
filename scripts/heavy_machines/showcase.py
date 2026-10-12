"""Line up every registered machine in one scene and animate all their joints.

    ./isaaclab.sh -p scripts/heavy_machines/showcase.py
    ./isaaclab.sh -p scripts/heavy_machines/showcase.py --machines HEX CTL ADT

Bases are fixed in place; tools sweep sinusoidally through their full range,
wheels spin and steering / articulation joints swing left and right.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Heavy machine line-up.")
parser.add_argument("--machines", nargs="*", help="names/abbreviations (default: all)")
parser.add_argument("--gap", type=float, default=4.0, help="metres between machines")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane  # noqa: E402

from heavy_machines import kinematics as K  # noqa: E402
from heavy_machines import registry  # noqa: E402
from heavy_machines.machine_env import articulation_cfg  # noqa: E402
from heavy_machines.spec import ROLLING_ROLES  # noqa: E402


def pose(spec, t: float) -> dict[str, float]:
    """Joint positions of the animation at time t (pure function of the spec)."""
    q: dict[str, float] = {}
    for k, j in enumerate(spec.movable_joints):
        phase = 0.7 * k
        if j.role == "tool":
            if j.type == "continuous":
                q[j.name] = 0.6 * math.sin(0.25 * t + phase)
            else:
                mid, amp = (j.upper + j.lower) / 2, (j.upper - j.lower) / 2
                q[j.name] = mid + amp * math.sin(0.4 * t + phase)
        elif j.role in ROLLING_ROLES:
            q[j.name] = math.atan2(math.sin(1.5 * t), math.cos(1.5 * t))
        elif j.role == "articulation":
            q[j.name] = min(abs(j.lower), abs(j.upper)) * math.sin(0.3 * t)
    d = spec.drive
    steers = spec.joints_with_role("steer")
    if steers:                                    # true Ackermann angles
        jf = K.joint_frames(spec)
        delta = 0.8 * d.max_steer * math.sin(0.3 * t)
        for j, a in zip(steers, K.ackermann_wheel_angles(delta, d.wheelbase,
                                                         [jf[s.name].t[1] for s in steers], d.steer_axle)):
            q[j.name] = a
    return K.resolve_followers(spec, {**K.init_q(spec), **q})


def main():
    specs = ([registry.get(m) for m in args_cli.machines] if args_cli.machines
             else list(registry.all_specs().values()))
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=args_cli.device))
    spawn_ground_plane("/World/Ground", GroundPlaneCfg(color=(0.43, 0.37, 0.29), size=(800.0, 800.0)))
    light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.98, 0.95, 0.90))
    light.func("/World/Light", light)
    sim_utils.create_prim("/World/Lineup", "Xform")

    robots, cursor = [], 0.0          # machines are placed side by side toward -y
    sample_t = [i * 0.5 for i in range(160)]
    for spec in specs:
        # Space by the extent swept during the animation, not just the init pose.
        lo_y = min(K.bounding_box(spec, pose(spec, t))[0][1] for t in sample_t)
        hi_y = max(K.bounding_box(spec, pose(spec, t))[1][1] for t in sample_t)
        y = cursor - hi_y             # left edge of this machine touches the cursor
        cfg = articulation_cfg(spec, f"/World/Lineup/{spec.name}", fix_base=True)
        cfg.init_state.pos = (0.0, y, K.root_height(spec))
        robots.append((spec, Articulation(cfg)))
        print(f"[showcase] y={y:7.1f} m  {spec.display_name} ({spec.abbrev})", flush=True)
        cursor = y + lo_y - args_cli.gap
    y = cursor

    sim.set_camera_view(eye=(30.0, y / 2 + 25.0, 18.0), target=(0.0, y / 2, 1.5))
    sim.reset()

    t, dt = 0.0, sim.get_physics_dt()
    while simulation_app.is_running():
        t += dt
        for spec, robot in robots:
            target = pose(spec, t)
            q = robot.data.default_joint_pos.clone()
            for k, name in enumerate(robot.joint_names):
                q[:, k] = target[name]
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            robot.set_joint_position_target(q)
            robot.write_data_to_sim()
        sim.step()
        for _, robot in robots:
            robot.update(dt)

    simulation_app.close()


if __name__ == "__main__":
    main()
