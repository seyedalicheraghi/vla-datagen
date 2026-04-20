"""
Teleoperate the forklift using the keyboard.

Controls:
    W / Arrow Up       → drive forward
    S / Arrow Down     → drive backward
    A / Arrow Left     → turn left
    D / Arrow Right    → turn right
    E                  → raise forks
    Q                  → lower forks
    R                  → reset episode
    Escape / Ctrl+C    → quit

Usage:
    ./isaaclab.sh -p scripts/forklift/teleop_forklift.py
"""

# ---------------------------------------------------------------------------
# Step 1: AppLauncher must come before ALL other Isaac Lab imports
# ---------------------------------------------------------------------------

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Forklift keyboard teleoperation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Step 2: Import everything else after Isaac Sim is running
# ---------------------------------------------------------------------------

import math
import sys
import os
import torch
import carb.input
import omni.appwindow

sys.path.insert(0, os.path.dirname(__file__))
from forklift_env import ForkliftEnv, ForkliftEnvCfg

# Optional: OpenCV for camera feed preview (requires GUI build, not headless)
_HAS_CV2 = False
try:
    import cv2
    import numpy as np
    # Probe for GUI support before committing
    _test = np.zeros((1, 1, 3), dtype=np.uint8)
    cv2.imshow("_probe", _test)
    cv2.destroyWindow("_probe")
    _HAS_CV2 = True
except Exception:
    pass  # headless build or no display — viewport window is used instead


# ---------------------------------------------------------------------------
# Keyboard state
# ---------------------------------------------------------------------------

_keys = {
    "forward":  False,
    "backward": False,
    "left":     False,
    "right":    False,
    "fork_up":  False,
    "fork_down": False,
    "reset":    False,
}

def _make_keyboard_handler():
    _input = carb.input.acquire_input_interface()
    _keyboard = omni.appwindow.get_default_app_window().get_keyboard()

    def _on_key(event: carb.input.KeyboardEvent, *args):
        pressed  = (event.type == carb.input.KeyboardEventType.KEY_PRESS)
        released = (event.type == carb.input.KeyboardEventType.KEY_RELEASE)
        held     = pressed  # KEY_PRESS fires on first press; KEY_REPEAT fires while held

        k = event.input
        KI = carb.input.KeyboardInput

        if k in (KI.W, KI.UP):
            _keys["forward"]  = not released
        elif k in (KI.S, KI.DOWN):
            _keys["backward"] = not released
        elif k in (KI.A, KI.LEFT):
            _keys["left"]     = not released
        elif k in (KI.D, KI.RIGHT):
            _keys["right"]    = not released
        elif k == KI.E:
            _keys["fork_up"]  = not released
        elif k == KI.Q:
            _keys["fork_down"] = not released
        elif k == KI.R and pressed:
            _keys["reset"] = True
        return True

    sub = _input.subscribe_to_keyboard_events(_keyboard, _on_key)
    return _input, _keyboard, sub


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _switch_viewport_to_sensor_cam():
    """Switch the Isaac Sim viewport to display the forklift driver camera sensor."""
    try:
        import omni.kit.viewport.utility as vp_utils
        viewport = vp_utils.get_active_viewport()
        if viewport is not None:
            cam_path = "/World/envs/env_0/DriverCam"
            viewport.camera_path = cam_path
            print(f"[INFO] Viewport switched to sensor camera: {cam_path}")
        else:
            print("[WARN] No active viewport found — GUI may not be running.")
    except Exception as e:
        print(f"[WARN] Could not switch viewport camera: {e}")


def main():
    env = ForkliftEnv(ForkliftEnvCfg())

    _input, _keyboard, _kb_sub = _make_keyboard_handler()

    obs, _ = env.reset()

    # Switch the Isaac Sim viewport to show what the front camera sees
    _switch_viewport_to_sensor_cam()

    print("\n" + "=" * 50)
    print("Forklift Keyboard Teleoperation")
    print("=" * 50)
    print("  W / Arrow Up    → forward")
    print("  S / Arrow Down  → backward")
    print("  A / Arrow Left  → turn left")
    print("  D / Arrow Right → turn right")
    print("  E               → raise forks")
    print("  Q               → lower forks")
    print("  R               → reset episode")
    print("  Ctrl+C          → quit")
    if _HAS_CV2:
        print("  [Camera preview: LEFT | CENTER | RIGHT]")
    print("=" * 50 + "\n")

    step = 0
    V_MAX       = 5.0   # m/s   — max drive speed
    STEER_ANGLE = 0.6   # rad   — max steering angle (~34°)
    WHEEL_BASE  = env.cfg.wheel_base
    # step dt = sim_dt * decimation
    STEP_DT     = env.cfg.sim.dt * env.cfg.decimation

    # Smooth velocity state
    current_v_x = 0.0
    ACCEL = 4.0   # m/s² — how fast speed builds up
    DECEL = 6.0   # m/s² — how fast it slows down / brakes

    while simulation_app.is_running():
        if _keys["reset"]:
            obs, _ = env.reset()
            _keys.update({k: False for k in _keys})
            current_v_x = 0.0
            print("[INFO] Episode reset.")
            step = 0
            continue

        target_v_x = (V_MAX if _keys["forward"]  else 0.0) \
                   - (V_MAX if _keys["backward"] else 0.0)

        # Accelerate toward target, decelerate faster when releasing
        if abs(target_v_x) > abs(current_v_x) or (target_v_x * current_v_x < 0):
            rate = ACCEL * STEP_DT
        else:
            rate = DECEL * STEP_DT
        if current_v_x < target_v_x:
            current_v_x = min(current_v_x + rate, target_v_x)
        else:
            current_v_x = max(current_v_x - rate, target_v_x)

        v_x = current_v_x

        # Ackermann: omega = v * tan(steer) / wheelbase → zero when stationary
        steer = (STEER_ANGLE if _keys["left"]  else 0.0) \
              - (STEER_ANGLE if _keys["right"] else 0.0)
        omega_z = v_x * math.tan(steer) / WHEEL_BASE

        fork  = (1.0 if _keys["fork_up"]   else 0.0) \
              - (1.0 if _keys["fork_down"] else 0.0)

        action = torch.tensor([[v_x, omega_z, fork]], device=env.device)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        # Live camera preview — 3 views side by side
        if _HAS_CV2 and "rgb_front" in obs:
            THUMB_W, THUMB_H = 320, 240
            views = []
            for key, label in [("rgb_left", "LEFT"), ("rgb_front", "FRONT"), ("rgb_right", "RIGHT")]:
                if key in obs:
                    frame = obs[key][0].cpu().numpy()[:, :, 2::-1]  # RGBA → BGR
                    frame = cv2.resize(frame, (THUMB_W, THUMB_H))
                    cv2.putText(frame, label, (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    views.append(frame)
            if views:
                combined = np.hstack(views)
                cv2.imshow("Forklift Cameras", combined)
                cv2.waitKey(1)

        if step % 30 == 0:
            pallet_pos   = obs["pallet_pos"][0].cpu()
            forklift_xy  = env._carry_pos[0].cpu()
            dist = torch.norm(forklift_xy - pallet_pos[:2]).item()
            fork_pos_m = env._fork_pos[0].item()
            grabbed = env._pallet_grabbed[0]
            print(
                f"  step={step:4d} | "
                f"v_x={v_x:+.1f}  steer={math.degrees(steer):+.0f}°  fork={fork:+.1f} | "
                f"fork_height={fork_pos_m:.3f}m | "
                f"dist={dist:.2f}m | {'[CARRYING]' if grabbed else '          '} | "
                f"reward={reward[0]:.3f}"
            )

        if terminated.any() or truncated.any():
            reason = "SUCCESS" if terminated.any() else "TIMEOUT"
            print(f"\n[INFO] Episode ended ({reason}). Resetting...\n")
            obs, _ = env.reset()
            _keys.update({k: False for k in _keys})
            step = 0

    if _HAS_CV2:
        cv2.destroyAllWindows()
    _input.unsubscribe_to_keyboard_events(_keyboard, _kb_sub)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
