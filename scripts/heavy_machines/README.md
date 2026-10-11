# Heavy machines for Isaac Lab

A fleet of construction machines (excavators, loaders, haulers, cranes, graders, rollers, …)
that load into Isaac Sim 5.1 / Isaac Lab 2.3, plus one environment, teleop, scripted demos and a
data collector (ROS 2 bags + LeRobot) that work for **every** machine.

Each machine is one Python file in `machines/`. It describes the machine as links built from
boxes, cylinders and spheres, connected by joints. Everything else is generated from that file:

```
machines/<name>.py ──► assets/heavy_machines/<name>/<name>.urdf   (imported by Isaac Sim)
                   └─► assets/heavy_machines/<name>/<name>.json   (action/state layout, limits, cameras)
```

<!-- MACHINE_TABLE_START -->
<!-- MACHINE_TABLE_END -->

## Quick start

```bash
# List machines (no Isaac Sim needed)
python scripts/heavy_machines/check_machine.py --list

# Drive one with the keyboard (name or abbreviation)
./isaaclab.sh -p scripts/heavy_machines/teleop_machine.py --machine HEX
./isaaclab.sh -p scripts/heavy_machines/teleop_machine.py --machine wheel_loader --random-spawn

# Every machine side by side, all joints animated
./isaaclab.sh -p scripts/heavy_machines/showcase.py

# Scripted drive + tool sweep, saves camera frames and a state log (works headless)
./isaaclab.sh -p scripts/heavy_machines/scripted_demo.py --machine ADT --headless

# Language-conditioned episodes → ROS 2 mcap bags + LeRobot dataset
./isaaclab.sh -p scripts/heavy_machines/collect_machine.py --machine HEX --num_episodes 5 --headless
```

Teleop keys: `W/S` drive, `A/D` turn, `R` reset. Tool keys are assigned in order:
`I/K`, `J/L`, `U/O`, `T/G`, `Y/H`, `Z/X`, `C/V`, `N/M` (the mapping is printed at start).

## Using the environment in your own code

```python
from heavy_machines.machine_env import HeavyMachineEnv, make_env_cfg

env = HeavyMachineEnv(make_env_cfg("articulated_dump_truck", randomize_spawn=True))
obs, _ = env.reset()
action = torch.zeros(env.num_envs, env.machine_spec.action_dim, device=env.device)
action[:, 0] = 1.0                      # drive forward
obs, _, _, _, _ = env.step(action)
print(env.action_labels, env.state_labels)
```

| | |
|---|---|
| **Action** (all in `[-1, 1]`) | `[drive, turn, tool_1 … tool_k]`. `drive` +1 = full forward speed; `turn` +1 = turn left; `tool_i` +1 = that joint at full rate in its "positive" direction (labels in the JSON file). |
| **`obs["state"]`** | `[x, y, yaw, speed, yaw_rate, steer_or_articulation, tool positions…]`, x/y relative to the env origin |
| **`obs["joint_pos"]`** | every joint, in articulation order (`env.robot.joint_names`) |
| **`obs["base_pose_w"]`** | world position + quaternion `(w, x, y, z)` of the root link |
| **`obs["rgb_front/left/right"]`** | three cameras: operator view and two outward-looking views |

Config fields worth knowing (`make_env_cfg(name, **overrides)`): `num_envs` via `scene`,
`enable_cameras`, `camera_resolution`, `spawn_pose`, `randomize_spawn`, `spawn_region`,
`spawn_seed`, `n_props` / `props_seed` (construction-site clutter: dirt piles, boulders,
barriers, cones, pipes), `episode_length_s`.

### How motion works (and its limits)

Same approach as the forklift environment: the base moves **kinematically**. Every physics
substep the environment integrates a drive model, writes the root pose and velocity, and writes
every joint position directly:

* **skid** (tracks, skid-steer wheels): yaw rate from `turn`, left/right track speeds for the
  cosmetic sprocket spin;
* **ackermann**: rate-limited virtual steer angle, true per-wheel Ackermann angles, front- or
  rear-axle steering;
* **articulated** (wheel loaders, ADTs, compactors, scrapers): rate-limited hinge angle and the
  exact two-axle no-slip yaw-rate model.

Gravity is off for the machine, so nothing sags; the ground and props still collide with it.
This gives exact, repeatable motion for data generation, but there is **no soil, payload,
tyre or hydraulic dynamics** — a bucket does not pick up material and a truck does not carry
load. Props are static.

## ROS 2 / RViz 2

`collect_machine.py` writes one mcap bag per episode with the URDF (`/robot_description`),
TF for every link (`/tf`, `/tf_static`), `/joint_states`, odometry, state, actions, the three
camera images and the language instruction (`/task`). Because the bag carries the URDF and the
full TF tree, RViz can show the real machine model moving:

```bash
source /opt/ros/jazzy/setup.bash
ros2 bag play ~/datasets/heavy_machines/hydraulic_excavator/episode_000 \
    --qos-profile-overrides-path ~/datasets/heavy_machines/hydraulic_excavator/qos_overrides.yaml
rviz2
#   Fixed Frame: world
#   Add → RobotModel  (Description Source: Topic, Description Topic: /robot_description)
#   Add → TF, Add → Image (/camera/front/image_raw)
```

## Checking and previewing machines without Isaac Sim

```bash
python scripts/heavy_machines/check_machine.py                     # validate all, regenerate assets
python scripts/heavy_machines/check_machine.py HEX --render /tmp/p # + side/top/front/3D PNGs (matplotlib)
python scripts/heavy_machines/test/test_machines.py                # offline tests
python scripts/heavy_machines/test/test_env_logic.py               # env control code (needs torch)
```

`check_machine.py` validates the kinematic tree, joint limits, drive geometry (wheelbase,
steering vs inner-wheel angle, articulation lengths), that wheels touch the ground and nothing
else does, size and mass against the real machine's datasheet, self-overlap, and that every
tool moves the way its label says (`raise`, `extend`, `left`, …).

## Adding a machine

1. Copy `machines/hydraulic_excavator.py` to `machines/<new_name>.py` and edit it. Put the
   real datasheet numbers you used in the docstring.
2. `python scripts/heavy_machines/check_machine.py <new_name> --render /tmp/p` until it prints
   OK, and look at the PNGs.
3. Run the two test files. Commit the `.py` file **and** the generated `assets/heavy_machines/<new_name>/`.

Nothing else needs registering — the teleop, demo, showcase, collector and environment pick it
up automatically.
