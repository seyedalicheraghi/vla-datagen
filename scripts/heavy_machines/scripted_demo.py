"""Scripted motion for one machine: drive, turn, then sweep every tool.

Saves camera frames and a per-step state log, so each machine can be checked
in Isaac Sim without a keyboard (works with --headless).

    ./isaaclab.sh -p scripts/heavy_machines/scripted_demo.py --machine wheel_loader --headless
    ./isaaclab.sh -p scripts/heavy_machines/scripted_demo.py --machine HEX --out demo_out

    # every machine (one Isaac process each):
    for m in $(python scripts/heavy_machines/check_machine.py --list | awk '{print $1}'); do
        ./isaaclab.sh -p scripts/heavy_machines/scripted_demo.py --machine $m --headless
    done

Output: <out>/<machine>/state.csv, <out>/<machine>/frames/<step>_<camera>.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Scripted heavy-machine demo.")
parser.add_argument("--machine", default="hydraulic_excavator")
parser.add_argument("--out", default="heavy_machine_demo")
parser.add_argument("--frame-every", type=int, default=15, help="save camera frames every N control steps")
parser.add_argument("--no-cameras", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = not args_cli.no_cameras
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from heavy_machines import registry  # noqa: E402
from heavy_machines.machine_env import HeavyMachineEnv, make_env_cfg  # noqa: E402


def build_script(spec) -> list[tuple[str, list[float], float]]:
    # Drive at a realistic pace for road-speed machines too (and stay inside the
    # prop-free work area).
    drive = min(1.0, 2.5 / spec.drive.max_speed)
    """(phase name, action, seconds)."""
    n = spec.action_dim

    def act(**kw):
        a = [0.0] * n
        a[0], a[1] = kw.get("drive", 0.0), kw.get("turn", 0.0)
        if "tool" in kw:
            a[2 + kw["tool"]] = kw["cmd"]
        return a

    script = [("drive forward", act(drive=drive), 3.0),
              ("forward + turn left", act(drive=drive, turn=1.0), 3.0),
              ("reverse + turn right", act(drive=-drive, turn=-1.0), 2.0),
              ("stop", act(), 1.5)]
    for i, tool in enumerate(spec.tools):
        j = spec.joint(tool.joint)
        span = math.pi / 2 if j.type == "continuous" else (j.upper - j.lower)
        secs = min(max(span / j.velocity, 1.5), 5.0)
        script.append((f"{tool.label} {tool.positive or '+'}", act(tool=i, cmd=1.0), secs))
        script.append((f"{tool.label} {tool.negative or '-'}", act(tool=i, cmd=-1.0), secs))
    return script


def save_rgb(tensor, path):
    import numpy as np
    arr = tensor.detach().cpu().numpy()[..., :3].astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except ImportError:
        np.save(path.replace(".png", ".npy"), arr)


def main():
    spec = registry.get(args_cli.machine)
    out_dir = os.path.join(args_cli.out, spec.name)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cfg = make_env_cfg(spec.name, enable_cameras=args_cli.enable_cameras)
    cfg.sim.device = args_cli.device
    env = HeavyMachineEnv(cfg)
    obs, _ = env.reset()
    control_hz = round(1.0 / env.step_dt)
    script = build_script(spec)
    total = sum(int(s * control_hz) for _, _, s in script)
    print(f"[demo] {spec.display_name}: {len(script)} phases, {total} steps → {out_dir}", flush=True)

    with open(os.path.join(out_dir, "state.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "phase"] + [f"a_{x}" for x in spec.action_labels] + spec.state_labels)
        step = 0
        for phase, a, secs in script:
            print(f"[demo] {phase}", flush=True)
            action = torch.tensor([a] * env.num_envs, device=env.device)
            for _ in range(int(secs * control_hz)):
                if not simulation_app.is_running():
                    break
                obs, _, _, _, _ = env.step(action)
                w.writerow([step, phase] + a + [round(v, 5) for v in obs["state"][0].tolist()])
                if env.cams and step % args_cli.frame_every == 0:
                    for cam in env.cams:
                        save_rgb(obs[f"rgb_{cam}"][0], os.path.join(frames_dir, f"{step:05d}_{cam}.png"))
                step += 1
    print(f"[demo] done — {step} steps, final state {obs['state'][0].tolist()}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
