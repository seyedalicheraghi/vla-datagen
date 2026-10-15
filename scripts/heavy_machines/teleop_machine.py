"""Keyboard teleoperation for any heavy machine.

    ./isaaclab.sh -p scripts/heavy_machines/teleop_machine.py --machine hydraulic_excavator
    ./isaaclab.sh -p scripts/heavy_machines/teleop_machine.py --machine ADT      # abbreviations work
    ./isaaclab.sh -p scripts/heavy_machines/teleop_machine.py --list

Keys:
    W / S        drive forward / reverse
    A / D        turn left / right
    I K, J L, U O, T G, Y H, Z X, C V, N M
                 tool 1..8 positive / negative (the mapping is printed at start)
    R            reset (new random spawn with --random-spawn)
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heavy_machines import safe_console  # noqa: E402

safe_console()

from heavy_machines import registry  # noqa: E402  (pure Python, safe before AppLauncher)

parser = argparse.ArgumentParser(description="Heavy machine keyboard teleoperation.")
parser.add_argument("--machine", default="hydraulic_excavator", help="registry name or abbreviation")
parser.add_argument("--list", action="store_true", help="list machines and exit")
parser.add_argument("--random-spawn", action="store_true", help="random spawn pose on every reset")
parser.add_argument("--no-cameras", action="store_true", help="disable the three machine cameras")

if "--list" in sys.argv:
    for s in registry.all_specs().values():
        print(f"{s.name:28s} {s.abbrev:5s} {s.category:11s} {s.action_labels}")
    sys.exit(0)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = not args_cli.no_cameras
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time  # noqa: E402

import carb.input  # noqa: E402
import omni.appwindow  # noqa: E402
import torch  # noqa: E402

from heavy_machines import kinematics as K  # noqa: E402

from heavy_machines.checks import TOOL_KEY_PAIRS  # noqa: E402
from heavy_machines.machine_env import HeavyMachineEnv, make_env_cfg  # noqa: E402


def main():
    spec = registry.get(args_cli.machine)
    cfg = make_env_cfg(spec.name, enable_cameras=args_cli.enable_cameras,
                       randomize_spawn=args_cli.random_spawn)
    cfg.sim.device = args_cli.device
    env = HeavyMachineEnv(cfg)

    held: set[str] = set()
    reset_requested = [False]
    KI = carb.input.KeyboardInput
    wanted = {"W", "S", "A", "D", "R", "UP", "DOWN", "LEFT", "RIGHT"}
    wanted |= {k for pair in TOOL_KEY_PAIRS for k in pair}
    key_names = {getattr(KI, n): n for n in wanted if hasattr(KI, n)}

    def on_key(event, *args):
        name = key_names.get(event.input)
        if name is None:
            return True
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            held.add(name)
            if name == "R":
                reset_requested[0] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            held.discard(name)
        return True

    inp = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    sub = inp.subscribe_to_keyboard_events(keyboard, on_key)

    print("\n" + "=" * 64)
    print(f"{spec.display_name} ({spec.abbrev}) — {spec.reference}")
    print("  W / S  drive forward / reverse      A / D  turn left / right")
    for (kp, kn), tool in zip(TOOL_KEY_PAIRS, spec.tools):
        print(f"  {kp} / {kn}  {tool.label:14s} {tool.positive or '+'} / {tool.negative or '-'}")
    print("  R      reset")
    print("=" * 64 + "\n")

    def follow_camera():
        """Chase view that follows the machine (GUI only)."""
        vcc = env.viewport_camera_controller
        if vcc is None:
            return
        lo, hi = K.bounding_box(spec)
        size = max(hi[0] - lo[0], 6.0)
        vcc.update_view_to_asset_root("machine")
        vcc.update_view_location(eye=(-1.3 * size, -1.0 * size, 0.6 * size), lookat=(0.2 * size, 0.0, 1.5))

    env.reset()
    follow_camera()
    action = torch.zeros(env.num_envs, spec.action_dim, device=env.device)
    while simulation_app.is_running():
        t0 = time.perf_counter()
        if reset_requested[0]:
            reset_requested[0] = False
            env.reset()
        action.zero_()
        action[:, 0] = float("W" in held or "UP" in held) - float("S" in held or "DOWN" in held)
        action[:, 1] = float("A" in held or "LEFT" in held) - float("D" in held or "RIGHT" in held)
        for i, (kp, kn) in enumerate(TOOL_KEY_PAIRS[:len(spec.tools)]):
            action[:, 2 + i] = float(kp in held) - float(kn in held)
        env.step(action)
        time.sleep(max(0.0, env.step_dt - (time.perf_counter() - t0)))   # real-time pacing

    inp.unsubscribe_to_keyboard_events(keyboard, sub)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
