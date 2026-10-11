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


def main():
    specs = ([registry.get(m) for m in args_cli.machines] if args_cli.machines
             else list(registry.all_specs().values()))
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60))
    spawn_ground_plane("/World/Ground", GroundPlaneCfg(color=(0.43, 0.37, 0.29), size=(800.0, 800.0)))
    light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.98, 0.95, 0.90))
    light.func("/World/Light", light)

    robots, cursor = [], 0.0          # machines are placed side by side toward -y
    for spec in specs:
        lo, hi = K.bounding_box(spec)
        y = cursor - hi[1]            # left edge of this machine touches the cursor
        cfg = articulation_cfg(spec, f"/World/Lineup/{spec.name}", fix_base=True)
        cfg.init_state.pos = (0.0, y, K.root_height(spec))
        robots.append((spec, Articulation(cfg)))
        print(f"[showcase] y={y:7.1f} m  {spec.display_name} ({spec.abbrev})", flush=True)
        cursor = y + lo[1] - args_cli.gap
    y = cursor

    sim.set_camera_view(eye=(30.0, y / 2 + 25.0, 18.0), target=(0.0, y / 2, 1.5))
    sim.reset()

    t, dt = 0.0, sim.get_physics_dt()
    while simulation_app.is_running():
        t += dt
        for spec, robot in robots:
            names = robot.joint_names
            q = robot.data.default_joint_pos.clone()
            for k, name in enumerate(names):
                j = spec.joint(name)
                phase = 0.7 * k
                if j.role == "tool":
                    if j.type == "continuous":
                        q[:, k] = 0.6 * math.sin(0.25 * t + phase)
                    else:
                        mid, amp = (j.upper + j.lower) / 2, (j.upper - j.lower) / 2
                        q[:, k] = mid + amp * math.sin(0.4 * t + phase)
                elif j.role in ROLLING_ROLES:
                    q[:, k] = math.atan2(math.sin(1.5 * t), math.cos(1.5 * t))
                elif j.role in ("steer", "articulation"):
                    lim = min(abs(j.lower), abs(j.upper))
                    q[:, k] = lim * math.sin(0.3 * t)
            for k, name in enumerate(names):
                j = spec.joint(name)
                if j.role == "follower" and j.follows:
                    src, kk, c = j.follows
                    q[:, k] = kk * q[:, names.index(src)] + c
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            robot.set_joint_position_target(q)
            robot.write_data_to_sim()
        sim.step()
        for _, robot in robots:
            robot.update(dt)

    simulation_app.close()


if __name__ == "__main__":
    main()
