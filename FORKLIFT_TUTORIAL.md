# Forklift Simulation with Isaac Lab
### From Zero to Autonomous Data Collection with openpi

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Install Isaac Sim](#2-install-isaac-sim)
3. [Install Isaac Lab](#3-install-isaac-lab)
4. [Verify Installation](#4-verify-installation)
5. [Connect PS4/PS5 Controller](#5-connect-ps4ps5-controller)
6. [Import the Forklift Asset](#6-import-the-forklift-asset)
7. [Build the Forklift Environment](#7-build-the-forklift-environment)
8. [Control the Forklift with PS4 Controller](#8-control-the-forklift-with-ps4-controller)
9. [Add Boxes at Random Locations](#9-add-boxes-at-random-locations)
10. [Add RGB Camera and LiDAR](#10-add-rgb-camera-and-lidar)
11. [Record Teleoperation Demonstrations](#11-record-teleoperation-demonstrations)
12. [Convert Data to LeRobot Format for openpi](#12-convert-data-to-lerobot-format-for-openpi)
13. [Train with openpi](#13-train-with-openpi)
14. [Run Autonomous Inference](#14-run-autonomous-inference)
15. [Project Structure](#15-project-structure)
16. [Appendix — Keywords Glossary](#appendix--keywords-glossary)

---

## 1. Prerequisites

### Hardware
- Ubuntu 22.04
- NVIDIA GPU with ≥8GB VRAM (RTX 3060, 3090, A5000, 4090)
- PS4 or PS5 controller (Bluetooth or USB)
- At least 50GB free disk space

### Software dependencies
```bash
# Check your OS
lsb_release -a

# Check GPU
nvidia-smi

# Check display
echo $DISPLAY
```

### Install system dependencies
```bash
sudo apt-get update && sudo apt-get install -y \
    git \
    curl \
    libvulkan1 \
    vulkan-tools \
    libegl1-mesa-dev
```

Install `uv` (Python package and environment manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env   # or restart your terminal
```

---

## 2. Install Isaac Sim

Isaac Lab runs on top of Isaac Sim. Install Isaac Sim first.

### Option A — Via pip (recommended, Isaac Sim 5.x)

Create and activate a virtual environment with `uv`:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
```

> **Tip:** Add `source /home/ali/Projects/IsaacLab/.venv/bin/activate` to your `~/.bashrc` so the environment activates automatically in new terminals.

Then install Isaac Sim:

```bash
uv pip install isaacsim==5.1.0.0 \
    isaacsim-rl \
    isaacsim-replicator \
    isaacsim-extscache-physics \
    isaacsim-extscache-kit \
    isaacsim-extscache-kit-sdk \
    --extra-index-url https://pypi.nvidia.com
```

> **Note:** This downloads ~15GB. Make sure you have space and a stable internet connection.

### Option B — Via NVIDIA Omniverse Launcher (GUI)

1. Download the Omniverse Launcher from https://developer.nvidia.com/omniverse
2. Install it, open it, go to **Apps → Isaac Sim**
3. Click **Install** (select version 5.1.0)
4. Isaac Sim installs to `~/.local/share/ov/pkg/isaac-sim-5.1.0/`

### Verify Isaac Sim installed
```bash
python3 -c "import isaacsim; print('Isaac Sim OK')"
```

---

## 3. Install Isaac Lab

You already have the repo cloned at `~/Projects/IsaacLab`. Now install it:

```bash
cd ~/Projects/IsaacLab

# Install Isaac Lab and all sub-packages
./isaaclab.sh --install

# This installs:
#   isaaclab         — core framework
#   isaaclab_assets  — robot/sensor configs
#   isaaclab_tasks   — pre-built environments
#   isaaclab_rl      — RL training wrappers
#   isaaclab_mimic   — imitation learning data tools
```

> **What `isaaclab.sh` does:** It creates a Python environment linked to your Isaac Sim installation
> and installs all Isaac Lab packages in editable mode (`pip install -e`).

### Alternative — manual install
```bash
uv pip install -e source/isaaclab
uv pip install -e source/isaaclab_assets
uv pip install -e source/isaaclab_tasks
uv pip install -e source/isaaclab_rl
uv pip install -e source/isaaclab_mimic
```

---

## 4. Verify Installation

Run a built-in demo to confirm everything works:

```bash
cd ~/Projects/IsaacLab

# List all available environments
./isaaclab.sh -p scripts/environments/list_envs.py

# Run a simple arm demo (no robot hardware needed)
./isaaclab.sh -p scripts/demos/arms.py
```

You should see a 3D viewport open with a robot arm. If it opens, Isaac Lab is working.

```bash
# Run the pick-and-place demo — closest to your forklift task
./isaaclab.sh -p scripts/demos/pick_and_place.py
```

---

## 5. Connect PS4/PS5 Controller

### Via USB (simplest)
Just plug it in. Ubuntu detects it automatically.

```bash
# Verify it's recognized
ls /dev/input/js*
# Should show: /dev/input/js0

# Check what buttons do what
sudo apt install joystick
jstest /dev/input/js0
```

### Via Bluetooth
```bash
# Put controller in pairing mode:
# PS4: Hold Share + PS button until light bar flashes
# PS5: Hold Create + PS button until light bar flashes

bluetoothctl
> scan on
> pair XX:XX:XX:XX:XX:XX   # your controller's MAC address
> connect XX:XX:XX:XX:XX:XX
> trust XX:XX:XX:XX:XX:XX
> quit
```

### Install DS4DRV for better PS4 support (optional but recommended)
```bash
uv pip install ds4drv
sudo ds4drv
```

### How Isaac Lab sees the controller

Isaac Lab uses NVIDIA's Carb input system, which treats any XInput-compatible gamepad
(PS4/PS5 both work) as a standard gamepad. The mapping is:

```
Left stick  → Move forward/backward/strafe  (v_x, v_y)
Right stick → Rotate                         (omega_z)
L2/R2       → Custom actions (fork up/down)
X button    → Reset episode
```

The controller is accessed via Isaac Lab's `Se2Gamepad` class (designed for ground vehicles like forklifts). You configure sensitivity for forward speed, sideways speed, and rotation separately. The fork up/down action is added as a custom callback on the trigger buttons.

---

## 6. Import the Forklift Asset

### Get the forklift USD model

Isaac Lab's official asset library includes a forklift. It's stored on NVIDIA's Nucleus
server and downloaded automatically when needed:

```python
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# The forklift USD path on NVIDIA's Nucleus server
FORKLIFT_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Forklift/forklift.usd"
```

> **What is Nucleus?** NVIDIA's cloud asset server. Isaac Lab downloads assets from it
> automatically the first time you reference them and caches them locally.

### Alternative — use a custom FBX model

**What is FBX?** FBX is a 3D model file format commonly used in game engines and CAD tools. You can download free forklift models from sites like [Sketchfab](https://sketchfab.com) or [TurboSquid](https://www.turbosquid.com) in FBX format.

**Where to place it:** Copy your `.fbx` file into `~/Projects/IsaacLab/assets/forklift/`.

**Step 1 — Convert FBX to USD** (Isaac Sim's native format):
```bash
./isaaclab.sh -p scripts/tools/convert_mesh.py \
    --input assets/forklift/forklift.fbx \
    --output assets/forklift/forklift.usd \
    --make-instanceable
```

**Step 2 — Add joints (if needed):** A raw FBX is just a visual mesh with no physics joints. To make it a drivable robot, you need a URDF file describing the wheel and fork joints. Once you have one:
```bash
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    --input assets/forklift/forklift.urdf \
    --output assets/forklift/forklift_articulated.usd
```

> **Tip:** If you're just starting out, skip the custom model and use NVIDIA's built-in forklift USD from Nucleus — it already has joints defined and downloads automatically.

### Define the forklift configuration

The forklift configuration tells Isaac Lab how to treat the USD model as a robot — which joints are wheels, which is the fork, and what their speed/force limits are.

Create the file `source/isaaclab_assets/isaaclab_assets/robots/forklift.py` with an `ArticulationCfg` that points to your USD path and defines actuators for the wheel joints and fork lift joint.

> This file is the single place you'll edit if you want to change forklift physics properties (e.g. max speed, fork force limits).

---

## 7. Build the Forklift Environment

### What this environment does

The forklift environment is the simulation scene that runs the task. Each episode:

- A **red box** spawns at a random position on the warehouse floor
- A **green pallet** spawns at a different random position
- The forklift starts at the center
- The task: drive to the box, lift it with the forks, and place it on the pallet
- The episode ends when the box is within 30cm of the pallet (success) or 60 seconds pass (timeout)

The scene includes the forklift mesh, a warehouse floor, the red box, the green pallet, and an overhead RGB+depth camera.

### What to create

Create the file `scripts/forklift/forklift_env.py`. This file defines:

- **`ForkliftEnvCfg`** — settings like episode length, simulation step rate, and sensor resolution
- **`ForkliftEnv`** — the environment class with scene setup, randomization, observations, rewards, and action handling

### What to expect when it runs

When you launch this environment, a 3D window opens showing the warehouse. The forklift, box, and pallet will be visible. Without a controller or policy connected, nothing moves — the next steps (teleoperation and autonomous inference) attach control to this environment.

---

## 8. Control the Forklift with PS4 Controller

### What this script does

`scripts/forklift/teleop_forklift.py` is already created. It:

1. Initializes `AppLauncher` (with camera rendering enabled), then imports `ForkliftEnv`
2. Connects to the PS4 controller via Isaac Lab's `Se2Gamepad`
3. Maps left stick → forward/backward, right stick → turn, R2 → raise forks, L2 → lower forks, X button → manual reset
4. Sends `[v_x, omega_z, fork_height]` actions to the environment every step
5. Prints status to the terminal every ~1 second (speed, turn, fork, box-to-pallet distance, reward)
6. Auto-resets on success (box within 30cm of pallet) or timeout (60s)

> **Current limitation:** The Nucleus forklift USD is a visual mesh with no physics joints. Drive and fork commands are computed correctly but have no physical effect on the forklift mesh until an articulated USD is swapped in. The red box and green pallet are full physics objects and respond normally.

### Run it

```bash
cd ~/Projects/IsaacLab
./isaaclab.sh -p scripts/forklift/teleop_forklift.py
```

### What to expect

The warehouse window opens with the forklift mesh, red box, and green pallet. The terminal prints a control summary and live status updates. Controller input is read every step — you can see `v_x`, `ω`, and `fork` values changing as you move the sticks. Episode resets (manual or automatic) are printed with the final reward.

> **To exit:** Press **Ctrl+C** in the terminal. The window close button does not work in script mode.

---

## 9. Add Boxes at Random Locations

The environment in Step 7 already randomizes one box and one pallet every episode. To add multiple boxes, you modify `forklift_env.py` in two places:

- **In `_setup_scene()`:** Add each box as a separate `RigidObjectCfg` entry in a list (e.g. 5 boxes with identical physics but different prim paths like `/World/Box_0`, `/World/Box_1`, etc.)
- **In `_reset_idx()`:** Loop over the list and assign each box a new random position within the warehouse floor bounds at the start of every episode

The pallet stays as a single target. The task becomes: pick up any box and place it on the pallet.

---

## 10. Add RGB Camera and LiDAR

Both sensors are already included in the environment built in Step 7. This section explains what each one provides and how to adjust them.

### RGB Camera

The front camera is mounted 1.5m ahead of the forklift at 1.2m height, facing forward. It captures 224×224 RGB and depth images at 30 FPS. Each step, you can read the current frame from `camera.data.output["rgb"]` (shape: H×W×3) and depth from `camera.data.output["depth"]` (shape: H×W×1).

To add a **rear camera**, duplicate the `CameraCfg` entry in `_setup_scene()` with a different prim path (e.g. `/World/Forklift/RearCamera`) and set the position behind the forklift with the rotation flipped 180°.

### LiDAR (Ray Caster)

The LiDAR is a simulated 16-beam sensor running at 10 Hz with a 360° horizontal sweep. It works by casting rays from the forklift and returning the 3D world coordinates of each hit point. Each step, `lidar.data.ray_hits_w` gives you the hit positions and `lidar.data.distances` gives the distances in meters.

To increase resolution or beam count, adjust `channels` (number of vertical beams) and `horizontal_res` (degrees between horizontal rays) in the `LidarPatternCfg` inside `forklift_env.py`.

---

## 11. Record Teleoperation Demonstrations

Isaac Lab has a built-in demo recorder at `scripts/tools/record_demos.py`.

### Record demonstrations
```bash
cd ~/Projects/IsaacLab

./isaaclab.sh -p scripts/tools/record_demos.py \
    --task Isaac-Forklift-v0 \
    --teleop_device gamepad \
    --dataset_file ./datasets/forklift_demos.hdf5 \
    --num_demos 50 \
    --step_hz 30
```

- Drive the forklift with your PS4 controller to complete each task
- Press **X button** to mark an episode as successful and move to next
- Data is saved in HDF5 format automatically

### What gets recorded per timestep
```
datasets/forklift_demos.hdf5
├── data/
│   ├── demo_0/
│   │   ├── obs/
│   │   │   ├── rgb          (T, 224, 224, 3)   ← camera frames
│   │   │   ├── depth        (T, 224, 224, 1)
│   │   │   ├── lidar        (T, N_rays, 3)
│   │   │   └── state        (T, N_joints)      ← forklift joint angles
│   │   └── actions          (T, 3)             ← [v_x, omega_z, fork]
│   ├── demo_1/
│   │   └── ...
│   └── demo_N/
└── mask/
    └── successful_demos     ← which episodes were marked success
```

### Replay a recorded demo (sanity check)
```bash
./isaaclab.sh -p scripts/tools/replay_demos.py \
    --task Isaac-Forklift-v0 \
    --dataset_file ./datasets/forklift_demos.hdf5 \
    --demo_index 0
```

### Convert HDF5 to MP4 for review
```bash
./isaaclab.sh -p scripts/tools/hdf5_to_mp4.py \
    --dataset_file ./datasets/forklift_demos.hdf5 \
    --output_dir ./datasets/videos/
```

---

## 12. Convert Data to LeRobot Format for openpi

### Why convert?

The HDF5 file from Step 11 is Isaac Lab's internal format. openpi expects data in **LeRobot format** — a HuggingFace standard using parquet files for tabular data and MP4 video for camera streams. You need to convert before training.

### What the conversion script does

Create `scripts/forklift/convert_to_lerobot.py`. It:

1. Opens the HDF5 file and loops over every recorded demonstration
2. For each timestep, extracts the RGB frame, joint state, and action
3. Converts camera images from HWC to CHW layout (required by LeRobot)
4. Bundles each episode into a LeRobot dataset and saves it to disk
5. Computes normalization statistics across all episodes at the end

The output dataset is saved under your HuggingFace repo ID (e.g. `seyedalicheraghi/forklift_demos`) and can be pushed to HuggingFace Hub or kept locally for training.

### Run it

```bash
cd ~/Projects/IsaacLab
./isaaclab.sh -p scripts/forklift/convert_to_lerobot.py
```

The terminal will print each demo as it's added and confirm when normalization stats are done.

---

## 13. Train with openpi

Switch to the openpi repo and fine-tune π₀:

### Define a training config

Add a new training config entry to `~/Projects/openpi/src/openpi/training/config.py`. The config tells openpi:

- **Which dataset to use** — your LeRobot dataset repo ID from Step 12
- **Which keys are images, state, and actions** — matching the field names from the conversion script
- **Which base checkpoint to fine-tune from** — the pre-trained π₀ weights on NVIDIA's cloud storage
- **Input/output shapes** — front camera image (3×224×224), joint state (6 values), action (3 values: forward speed, turn, fork height)

Name the config `pi0_forklift` so the training commands below can reference it.

### Compute normalization stats
```bash
cd ~/Projects/openpi
uv run scripts/compute_norm_stats.py --config-name pi0_forklift
```

### Fine-tune (run on cloud A100/H100 for speed)
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_forklift \
    --exp-name=forklift_v1 \
    --overwrite
```

Training tips:
- Start with 5,000 steps to see if loss decreases
- 20,000–50,000 steps for a usable policy
- Monitor loss on Weights & Biases dashboard
- Save checkpoints every 5,000 steps

---

## 14. Run Autonomous Inference

Once training is done, you run two processes in parallel: the **policy server** (runs the π₀ model) and the **forklift simulation** (sends observations and receives actions).

### Terminal 1 — Start the policy server

```bash
cd ~/Projects/openpi
uv run scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config=pi0_forklift \
    --policy.dir=checkpoints/pi0_forklift/forklift_v1/20000
```

This loads your trained checkpoint and starts a WebSocket server on port 8000. It stays running and waits for the simulation to connect.

### Terminal 2 — Run the autonomous forklift

Create `scripts/forklift/autonomous_forklift.py`. This script:

1. Launches the forklift environment (same as Steps 7–8)
2. Connects to the policy server via WebSocket
3. Each step: sends the current camera image, joint state, and a text prompt ("move red box to green pallet") to the server
4. Receives an action back from the model and applies it to the forklift joints
5. Resets the episode when the task succeeds or times out

```bash
cd ~/Projects/IsaacLab
./isaaclab.sh -p scripts/forklift/autonomous_forklift.py \
    --host localhost \
    --port 8000
```

### What to expect

The forklift should navigate to the box, slide the forks under it, lift it, and drive it to the pallet — all without any human input. Performance depends on how many demos you recorded and how many training steps you ran. Start evaluating at 20,000 steps.

---

## 15. Project Structure

After following this tutorial, your file structure will look like:

```
~/Projects/
├── openpi/                              ← VLA model and training
│   ├── src/openpi/training/config.py   ← add pi0_forklift config here
│   ├── scripts/
│   │   ├── serve_policy.py             ← runs the policy server
│   │   ├── train.py                    ← fine-tuning
│   │   └── compute_norm_stats.py
│   └── checkpoints/
│       └── pi0_forklift/forklift_v1/   ← your trained model
│
└── IsaacLab/                           ← simulation framework
    ├── scripts/
    │   └── forklift/
    │       ├── forklift_env.py          ← environment (Step 7)
    │       ├── teleop_forklift.py       ← PS4 control (Step 8)
    │       ├── convert_to_lerobot.py    ← data conversion (Step 12)
    │       └── autonomous_forklift.py   ← inference (Step 14)
    ├── datasets/
    │   └── forklift_demos.hdf5          ← recorded demonstrations
    └── source/
        └── isaaclab_assets/
            └── robots/forklift.py       ← forklift config (Step 6)
```

---

## Appendix — Keywords Glossary

### VLA (Vision-Language-Action Models)

| Term | Definition |
|---|---|
| **VLA** | Vision-Language-Action model. Takes camera images and language instructions as input, outputs robot actions. π₀ is a VLA. |
| **π₀ (pi-zero)** | Physical Intelligence's VLA model. Pre-trained on 10k+ hours of robot data. |
| **Action chunk** | A sequence of future actions predicted at once (e.g. 10 steps). More stable than predicting one step at a time. |
| **Flow matching** | The generative technique π₀ uses to produce actions. Learns to transform noise into action distributions. |
| **Imitation learning** | Training a policy by copying expert demonstrations rather than reward signals. |
| **Behavior cloning (BC)** | The simplest form of imitation learning — supervised learning on (observation, action) pairs. |
| **LoRA** | Low-Rank Adaptation. Fine-tuning technique that only trains a small number of parameters. Requires less VRAM than full fine-tuning. |
| **Sim-to-real** | Training a policy in simulation, then deploying it on a real robot. |
| **Zero-shot** | A policy that works on tasks or environments it has never seen during training. |
| **Prompt** | The language instruction given to the VLA. E.g. "move box to left pallet". |
| **Action space** | The set of possible actions the robot can take (e.g. joint velocities, end-effector pose). |
| **Observation space** | What the robot perceives (camera images, joint angles, LiDAR). |
| **Episode** | One complete trial from reset to success or failure. |
| **Teleoperation** | A human controlling the robot to collect demonstration data. |
| **Norm stats** | Normalization statistics (mean, std) computed from training data. Helps training stability. |
| **LeRobot** | HuggingFace's robot learning dataset format. openpi uses this for training data. |
| **Policy server** | openpi's WebSocket server that runs the model and streams actions to the robot. |
| **Checkpoint** | Saved model weights at a specific training step. |
| **FAST tokenizer** | π₀-FAST's action tokenizer. Converts continuous actions to discrete tokens for autoregressive prediction. |
| **Autoregressive** | Generating outputs one token at a time, each conditioned on previous outputs. Used in π₀-FAST. |

---

### Isaac Lab / Isaac Sim

| Term | Definition |
|---|---|
| **Isaac Sim** | NVIDIA's robot simulator built on Omniverse. Provides physics and rendering. |
| **Isaac Lab** | Open-source framework on top of Isaac Sim for robot learning (RL, IL, data collection). |
| **PhysX** | NVIDIA's physics engine used by Isaac Sim for rigid body simulation. |
| **USD (Universal Scene Description)** | Pixar's file format for 3D scenes. Isaac Sim uses `.usd` files for robot and environment assets. |
| **Nucleus** | NVIDIA's cloud asset server. Isaac Lab downloads robot models from it automatically. |
| **Omniverse** | NVIDIA's platform for 3D simulation and collaboration. Isaac Sim runs on it. |
| **Articulation** | A robot model with joints and links. In Isaac Lab, `ArticulationCfg` defines a robot. |
| **RigidObject** | A non-articulated physics object (e.g. a box). Defined with `RigidObjectCfg`. |
| **CameraCfg** | Isaac Lab configuration for adding an RGB/depth camera to the scene. |
| **RayCasterCfg** | Isaac Lab's LiDAR simulation — casts rays and returns hit distances. |
| **Se2Gamepad** | Isaac Lab's PS4/Xbox gamepad interface for 2D velocity commands. |
| **Se3Gamepad** | Isaac Lab's PS4/Xbox gamepad interface for 3D end-effector control. |
| **AppLauncher** | Isaac Lab's launcher class that must be initialized before any Isaac imports. |
| **DirectRLEnv** | Base class for writing custom Isaac Lab environments. |
| **InteractiveSceneCfg** | Defines the scene layout (how many parallel environments, spacing). |
| **IsaacLab Mimic** | Isaac Lab's imitation learning data augmentation tool — generates more demos from a few. |
| **HDF5** | The file format Isaac Lab uses to store recorded demonstrations. |
| **Parallel environments** | Isaac Lab can run thousands of environments simultaneously on one GPU. |
| **Domain randomization** | Randomly varying simulation parameters (friction, lighting, object positions) to improve sim-to-real. |
| **RTX rendering** | Isaac Sim's photorealistic rendering using NVIDIA RTX ray tracing. |
| **Warp** | NVIDIA's GPU-accelerated Python framework. Used by Isaac Lab for fast tensor operations. |
| **Newton** | NVIDIA's new physics engine (experimental) that supports granular material simulation. |
| **FMU** | Functional Mock-up Unit. A standard for packaging physics simulations. |

---

### Robotics / ML General

| Term | Definition |
|---|---|
| **MDP** | Markov Decision Process. Mathematical framework for sequential decision making. All RL problems are MDPs. |
| **Reinforcement Learning (RL)** | Training by trial and error with reward signals. Agent learns by interacting with environment. |
| **State** | The complete description of the robot and environment at a given moment. |
| **Observation** | What the robot actually perceives (may be partial — e.g. only camera, no ground truth position). |
| **Reward function** | A function that scores each timestep. The agent maximizes cumulative reward. |
| **Policy** | A function mapping observations to actions. The "brain" of the robot. |
| **Trajectory** | A sequence of (state, action, reward) tuples from one episode. |
| **URDF** | Unified Robot Description Format. XML file describing a robot's links and joints. Used to import robots into simulators. |
| **MJCF** | MuJoCo's XML format for robot descriptions. Similar to URDF but MuJoCo-specific. |
| **End-effector** | The part of a robot arm that interacts with objects (e.g. gripper, hand). |
| **IK (Inverse Kinematics)** | Computing joint angles to achieve a desired end-effector position. |
| **FK (Forward Kinematics)** | Computing end-effector position from joint angles. |
| **DOF** | Degrees of Freedom. Number of independent axes a robot can move. |
| **Transformer** | The neural network architecture underlying most modern VLAs. Uses attention mechanisms. |
| **Diffusion model** | A generative model that learns to denoise. π₀ uses a variant of this for action generation. |
| **Foundation model** | A large model pre-trained on broad data that can be fine-tuned for specific tasks. π₀ is a robotics foundation model. |
| **Embodiment** | The physical form of the robot. Different embodiments (arms, forklifts) have different action spaces. |
| **Workspace** | The volume of space a robot arm can reach. |
| **Manipulation** | Robot tasks involving physically interacting with objects (grasping, pushing, placing). |
| **Navigation** | Robot tasks involving moving through an environment without necessarily manipulating objects. |
| **Locomanipulation** | Tasks combining locomotion (moving the base) and manipulation (using arms). |
| **Sim-to-real gap** | The difference between simulated and real-world physics that causes policies trained in sim to fail on real robots. |
| **Data augmentation** | Artificially increasing training data by transforming existing demos (flipping images, adding noise). |
| **HuggingFace Hub** | Platform for sharing ML models and datasets. LeRobot datasets are stored here. |
