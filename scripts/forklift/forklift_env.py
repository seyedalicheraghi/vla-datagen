"""
Forklift warehouse environment.

Task: Drive the forklift to a red box, lift it with the forks, and place it
on a green pallet. Box and pallet positions are randomized each episode.

This file defines ForkliftEnv and ForkliftEnvCfg as importable classes.
AppLauncher must be initialized by the caller BEFORE importing this module.

Standalone usage (for scene inspection):
    ./isaaclab.sh -p scripts/forklift/forklift_env.py
"""

# ---------------------------------------------------------------------------
# Isaac Lab imports — safe because caller has already run AppLauncher
# ---------------------------------------------------------------------------

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

# FORKLIFT_CFG is imported lazily inside _setup_scene() because
# isaaclab_assets.robots.forklift pulls in pxr which requires Isaac Sim
# to be running. Importing it here would break the standalone entry point.


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@configclass
class ForkliftEnvCfg(DirectRLEnvCfg):
    """Settings for the forklift environment."""

    # Simulation: 120 Hz physics
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=4)

    # Policy runs every 4 physics steps → effective policy rate = 30 Hz
    decimation: int = 4

    # Scene: single environment with 10m spacing
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=10.0)

    # Episode ends after 60 seconds if the task isn't completed
    episode_length_s: float = 60.0

    # Spaces (used by RL wrappers; not needed for teleoperation)
    observation_space: int = 64
    action_space: int = 3         # [v_x, omega_z, fork_height]
    state_space: int = 0

    # Differential drive geometry
    wheel_radius: float = 0.3          # metres (matches URDF cylinder radius)
    wheel_base: float = 1.15           # distance between rear wheels (matches URDF y offsets ×2)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ForkliftEnv(DirectRLEnv):
    """Forklift pick-and-place environment with articulated joints."""

    cfg: ForkliftEnvCfg

    def __init__(self, cfg: ForkliftEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.actions = torch.zeros(self.num_envs, 3, device=self.device)

        # Cache joint indices after super().__init__ has set up the scene.
        # find_joints returns (indices_list, names_list); we want the first index.
        all_joints = self.forklift.joint_names
        print(f"[INFO] Forklift joints: {all_joints}")
        self._left_wheel_idx:  int = self.forklift.find_joints("rear_left_wheel_joint")[0][0]
        self._right_wheel_idx: int = self.forklift.find_joints("rear_right_wheel_joint")[0][0]
        self._fork_idx:        int = self.forklift.find_joints("fork_lift_joint")[0][0]
        print(f"[INFO] left_wheel={self._left_wheel_idx}, right_wheel={self._right_wheel_idx}, fork={self._fork_idx}")

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        # Deferred import: pxr (and isaaclab_assets) require Isaac Sim running
        from isaaclab_assets.robots.forklift import FORKLIFT_CFG

        # Floor and lighting
        spawn_ground_plane(prim_path="/World/Ground", cfg=GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Articulated forklift — wheels + fork lift joint.
        # FORKLIFT_CFG already uses {ENV_REGEX_NS} as prim_path placeholder;
        # Isaac Lab expands it to the per-env path automatically.
        self.forklift = Articulation(FORKLIFT_CFG)

        # Red box — physics rigid body, randomized each episode
        self.box = RigidObject(RigidObjectCfg(
            prim_path="/World/envs/env_.*/Box",
            spawn=sim_utils.CuboidCfg(
                size=(0.4, 0.4, 0.4),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            ),
        ))

        # Green pallet — kinematic, randomized each episode
        self.pallet = RigidObject(RigidObjectCfg(
            prim_path="/World/envs/env_.*/Pallet",
            spawn=sim_utils.CuboidCfg(
                size=(0.8, 0.8, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
            ),
        ))

        # Overhead camera looking down at the scene
        self.camera = Camera(CameraCfg(
            prim_path="/World/envs/env_.*/Camera",
            update_period=1 / 30,
            height=224,
            width=224,
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 50.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 8.0),
                rot=(0.7071, 0.0, 0.0, 0.7071),
            ),
        ))

        # Register physics assets and sensors with the scene
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["forklift"] = self.forklift
        self.scene.rigid_objects["box"]      = self.box
        self.scene.rigid_objects["pallet"]   = self.pallet
        self.scene.sensors["camera"]         = self.camera

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # Reset forklift articulation to default pose, zero velocity
        default_root_state = self.forklift.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.forklift.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        self.forklift.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

        default_joint_pos = self.forklift.data.default_joint_pos[env_ids].clone()
        default_joint_vel = self.forklift.data.default_joint_vel[env_ids].clone()
        self.forklift.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

        # Randomize box position
        box_pos = torch.zeros(n, 7, device=self.device)
        box_pos[:, 0] = torch.FloatTensor(n).uniform_(-3.0, 3.0).to(self.device)
        box_pos[:, 1] = torch.FloatTensor(n).uniform_(-3.0, 3.0).to(self.device)
        box_pos[:, 2] = 0.2
        box_pos[:, 6] = 1.0
        box_pos[:, :3] += self.scene.env_origins[env_ids]
        self.box.write_root_pose_to_sim(box_pos, env_ids=env_ids)

        # Randomize pallet position
        pallet_pos = torch.zeros(n, 7, device=self.device)
        pallet_pos[:, 0] = torch.FloatTensor(n).uniform_(-3.0, 3.0).to(self.device)
        pallet_pos[:, 1] = torch.FloatTensor(n).uniform_(-3.0, 3.0).to(self.device)
        pallet_pos[:, 2] = 0.025
        pallet_pos[:, 6] = 1.0
        pallet_pos[:, :3] += self.scene.env_origins[env_ids]
        self.pallet.write_root_pose_to_sim(pallet_pos, env_ids=env_ids)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        v_x      = self.actions[:, 0]   # forward / backward (m/s in body frame)
        omega_z  = self.actions[:, 1]   # yaw rate (rad/s)
        fork_cmd = self.actions[:, 2]   # +1 raise, -1 lower, 0 hold

        # --- Drive the base directly via root velocity (body → world frame) ---
        # Wheel contact physics requires precise friction tuning; driving the root
        # is more robust for this simulation.
        heading = self.forklift.data.heading_w   # (N,) yaw in world frame

        vel = torch.zeros(self.num_envs, 6, device=self.device)
        vel[:, 0] = v_x * torch.cos(heading)   # world X
        vel[:, 1] = v_x * torch.sin(heading)   # world Y
        vel[:, 5] = omega_z                     # yaw
        self.forklift.write_root_velocity_to_sim(vel)

        # Also spin the visual wheel joints to match, so the wheels look like
        # they're rolling (cosmetic only — the root velocity drives motion).
        R = self.cfg.wheel_radius
        L = self.cfg.wheel_base
        omega_left  = (v_x - omega_z * L / 2.0) / R
        omega_right = (v_x + omega_z * L / 2.0) / R
        joint_vel_target = self.forklift.data.joint_vel_target.clone()
        joint_vel_target[:, self._left_wheel_idx]  = omega_left
        joint_vel_target[:, self._right_wheel_idx] = omega_right
        self.forklift.set_joint_velocity_target(joint_vel_target)

        # --- Fork lift: accumulate position target ---
        fork_pos = self.forklift.data.joint_pos[:, self._fork_idx].clone()
        fork_delta = fork_cmd * 0.05   # 0.05 m per step at 30 Hz → ~1.5 m/s
        new_fork_pos = torch.clamp(fork_pos + fork_delta, 0.0, 1.5)
        fork_pos_target = self.forklift.data.joint_pos_target.clone()
        fork_pos_target[:, self._fork_idx] = new_fork_pos
        self.forklift.set_joint_position_target(fork_pos_target)

    # ------------------------------------------------------------------
    # Observations, rewards, termination
    # ------------------------------------------------------------------

    def _get_observations(self):
        return {
            "rgb":        self.camera.data.output["rgb"],
            "depth":      self.camera.data.output["depth"],
            "box_pos":    self.box.data.root_pos_w,
            "pallet_pos": self.pallet.data.root_pos_w,
        }

    def _get_rewards(self) -> torch.Tensor:
        dist = torch.norm(
            self.box.data.root_pos_w[:, :2] - self.pallet.data.root_pos_w[:, :2], dim=1
        )
        return torch.exp(-dist)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        dist = torch.norm(
            self.box.data.root_pos_w[:, :2] - self.pallet.data.root_pos_w[:, :2], dim=1
        )
        return dist < 0.3, self.episode_length_buf >= self.max_episode_length


# ---------------------------------------------------------------------------
# Standalone entry point (scene inspection only)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Forklift environment — standalone inspection.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    env = ForkliftEnv(ForkliftEnvCfg())
    env.reset()
    print("[INFO] Forklift environment loaded. Press Ctrl+C to exit.")

    while simulation_app.is_running():
        env.step(torch.zeros(1, 3))

    env.close()
    simulation_app.close()
