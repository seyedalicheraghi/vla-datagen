"""
Teleoperate the forklift using a PS4/PS5 controller.

Controls:
    Left stick up/down     → drive forward / backward
    Right stick left/right → turn left / right
    R2 (right trigger)     → raise forks
    L2 (left trigger)      → lower forks
    X button (A on XInput) → reset episode

    Keyboard fallback (if triggers don't respond):
    E key                  → raise forks
    Q key                  → lower forks
    R key                  → reset episode

Usage:
    ./isaaclab.sh -p scripts/forklift/teleop_forklift.py
"""

# ---------------------------------------------------------------------------
# Step 1: AppLauncher must come before ALL other Isaac Lab imports
# ---------------------------------------------------------------------------

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Forklift teleoperation with PS4 controller.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Step 2: Import everything else after Isaac Sim is running
# ---------------------------------------------------------------------------

import sys
import os
import torch
import carb.input

from isaaclab.devices import Se2Gamepad, Se2GamepadCfg

sys.path.insert(0, os.path.dirname(__file__))
from forklift_env import ForkliftEnv, ForkliftEnvCfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    env = ForkliftEnv(ForkliftEnvCfg())

    gamepad = Se2Gamepad(
        Se2GamepadCfg(
            v_x_sensitivity=2.0,
            v_y_sensitivity=0.0,
            omega_z_sensitivity=1.5,
            sim_device=str(env.device),
        )
    )

    # -----------------------------------------------------------------------
    # Fork state — two sources: gamepad triggers + keyboard arrow keys
    # -----------------------------------------------------------------------
    fork_trigger = {"up": 0.0, "down": 0.0}   # analog trigger values
    fork_key     = {"up": False, "down": False} # keyboard held state
    reset_requested = {"flag": False}

    # Direct gamepad subscription to read analog trigger values continuously.
    # Prints unknown inputs for the first 5 presses so we can spot wrong mappings.
    _seen_inputs = set()
    _known = {carb.input.GamepadInput.RIGHT_TRIGGER, carb.input.GamepadInput.LEFT_TRIGGER}

    def _on_gamepad_event(event: carb.input.GamepadEvent, *args):
        val = float(event.value)
        inp = event.input
        if inp == carb.input.GamepadInput.RIGHT_TRIGGER:
            fork_trigger["up"] = val
        elif event.input == carb.input.GamepadInput.LEFT_TRIGGER:
            fork_trigger["down"] = val
        # Debug: print any unfamiliar input that has a non-zero value
        elif val > 0.1 and inp not in _seen_inputs:
            print(f"  [GAMEPAD DEBUG] input={inp}  value={val:.3f}  "
                  f"(not mapped — check trigger binding)")
            _seen_inputs.add(inp)
        return True

    _input_iface = gamepad._input
    _gamepad_dev = gamepad._gamepad
    _fork_sub = _input_iface.subscribe_to_gamepad_events(_gamepad_dev, _on_gamepad_event)

    # Keyboard subscription for fork (arrow keys) and reset (R)
    import omni.appwindow
    _keyboard = omni.appwindow.get_default_app_window().get_keyboard()

    def _on_keyboard_event(event: carb.input.KeyboardEvent, *args):
        pressed  = (event.type == carb.input.KeyboardEventType.KEY_PRESS)
        released = (event.type == carb.input.KeyboardEventType.KEY_RELEASE)
        if event.input == carb.input.KeyboardInput.E:
            fork_key["up"]   = pressed if pressed else (not released and fork_key["up"])
        elif event.input == carb.input.KeyboardInput.Q:
            fork_key["down"] = pressed if pressed else (not released and fork_key["down"])
        elif event.input == carb.input.KeyboardInput.R and pressed:
            reset_requested["flag"] = True
        return True

    _keyboard_sub = _input_iface.subscribe_to_keyboard_events(_keyboard, _on_keyboard_event)

    gamepad.add_callback(carb.input.GamepadInput.A, lambda: reset_requested.update({"flag": True}))

    obs, _ = env.reset()
    gamepad.reset()

    print("\n" + "=" * 50)
    print("Forklift Teleoperation Ready")
    print("=" * 50)
    print("  Left stick       → drive forward / backward")
    print("  Right stick      → turn left / right")
    print("  R2               → raise forks  (or E key)")
    print("  L2               → lower forks  (or Q key)")
    print("  X / R key        → reset episode")
    print("  Ctrl+C           → quit")
    print("=" * 50)
    print("  [If R2/L2 do nothing, use E/Q keys instead]")
    print("  [Any unmapped gamepad input will be printed for debugging]")
    print("=" * 50 + "\n")

    step = 0

    while simulation_app.is_running():
        if reset_requested["flag"]:
            obs, _ = env.reset()
            gamepad.reset()
            fork_trigger["up"] = fork_trigger["down"] = 0.0
            fork_key["up"] = fork_key["down"] = False
            reset_requested["flag"] = False
            print("[INFO] Episode reset.")
            step = 0
            continue

        base_cmd = gamepad.advance()

        # Combine trigger + keyboard: keyboard gives full ±1, trigger gives analog
        fork_up   = max(fork_trigger["up"],   1.0 if fork_key["up"]   else 0.0)
        fork_down = max(fork_trigger["down"],  1.0 if fork_key["down"] else 0.0)
        fork_value = fork_up - fork_down

        action = torch.tensor([[
            base_cmd[0].item(),
            base_cmd[2].item(),
            fork_value,
        ]], device=env.device)

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        if step % 30 == 0:
            box_pos    = obs["box_pos"][0].cpu()
            pallet_pos = obs["pallet_pos"][0].cpu()
            dist = torch.norm(box_pos[:2] - pallet_pos[:2]).item()
            fork_pos_m = env.forklift.data.joint_pos[0, env._fork_idx].item()
            print(
                f"  step={step:4d} | "
                f"v_x={action[0,0]:.2f}  ω={action[0,1]:.2f}  fork_cmd={action[0,2]:+.2f} | "
                f"fork_joint={fork_pos_m:.3f}m | "
                f"box↔pallet={dist:.2f}m | reward={reward[0]:.3f}"
            )

        if terminated.any() or truncated.any():
            r = reward[0].item()
            reason = "SUCCESS" if terminated.any() else "TIMEOUT"
            print(f"\n[INFO] Episode ended ({reason}) — reward={r:.3f}. Resetting...\n")
            obs, _ = env.reset()
            gamepad.reset()
            fork_trigger["up"] = fork_trigger["down"] = 0.0
            fork_key["up"] = fork_key["down"] = False
            step = 0

    _input_iface.unsubscribe_to_gamepad_events(_gamepad_dev, _fork_sub)
    _input_iface.unsubscribe_to_keyboard_events(_keyboard, _keyboard_sub)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
