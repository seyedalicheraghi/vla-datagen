"""
Forklift warehouse environment.

Task: Navigate the forklift through a cluttered warehouse to reach the unique
orange target box.  All other boxes are tan/cardboard-coloured obstacles.
The warehouse layout (box positions, target location) is re-randomised each
episode while guaranteeing a navigable path from the forklift start to the
target using BFS.

This file defines ForkliftEnv and ForkliftEnvCfg as importable classes.
AppLauncher must be initialised by the caller BEFORE importing this module.

Standalone usage (scene inspection):
    ./isaaclab.sh -p scripts/forklift/forklift_env.py
"""

# ---------------------------------------------------------------------------
# Isaac Lab imports — safe because caller has already run AppLauncher
# ---------------------------------------------------------------------------

import numpy as np
from collections import deque

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Warehouse layout constants
# ---------------------------------------------------------------------------

_WAREHOUSE_HALF = 8.5   # half-width of warehouse floor  → 17 m × 17 m
_WALL_T         = 0.3   # wall thickness (m)
_WALL_H         = 4.0   # wall height (m)

_GRID_N  = 11           # grid cells per axis (11 × 11)
_CELL    = 1.5          # metres per cell  → grid spans ±7.5 m inside the walls

_N_BOXES  = 48          # pre-spawned common boxes (upper bound per layout)
_BOX_SIZE = 0.55        # cube side length (m)
_PARK_Z   = -60.0       # underground parking depth for inactive boxes

_COMMON_COLOR = (0.70, 0.58, 0.38)   # cardboard / tan
_TARGET_COLOR = (0.95, 0.25, 0.05)   # bright orange-red


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@configclass
class ForkliftEnvCfg(DirectRLEnvCfg):
    """Settings for the warehouse forklift environment."""

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=4)
    decimation: int = 4
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=40.0)
    episode_length_s: float = 120.0

    observation_space: int = 64
    action_space: int = 3       # [v_x, omega_z, fork_cmd]
    state_space: int = 0

    wheel_radius: float = 0.3
    wheel_base: float = 1.15


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ForkliftEnv(DirectRLEnv):
    """Warehouse forklift environment with procedurally generated layouts."""

    cfg: ForkliftEnvCfg

    def __init__(self, cfg: ForkliftEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.actions = torch.zeros(self.num_envs, 3, device=self.device)

        all_joints = self.forklift.joint_names
        print(f"[INFO] Forklift joints: {all_joints}")
        self._left_wheel_idx  = self.forklift.find_joints("rear_left_wheel_joint")[0][0]
        self._right_wheel_idx = self.forklift.find_joints("rear_right_wheel_joint")[0][0]
        self._fork_idx        = self.forklift.find_joints("fork_lift_joint")[0][0]
        print(f"[INFO] left_wheel={self._left_wheel_idx}, "
              f"right_wheel={self._right_wheel_idx}, fork={self._fork_idx}")

        self._rng = np.random.default_rng()   # layout RNG (re-seeded per episode)

        # Kinematic box-grab state (one entry per env)
        self._box_grabbed: list[bool]  = [False] * self.num_envs
        self._grab_fwd:    list[float] = [0.0]   * self.num_envs  # body-frame fwd offset
        self._grab_lat:    list[float] = [0.0]   * self.num_envs  # body-frame lateral offset

    # ------------------------------------------------------------------
    # Scene setup (called once at init)
    # ------------------------------------------------------------------

    def _setup_scene(self):
        from isaaclab_assets.robots.forklift import FORKLIFT_CFG

        # Concrete-grey warehouse floor
        spawn_ground_plane(
            prim_path="/World/Ground",
            cfg=GroundPlaneCfg(color=(0.40, 0.40, 0.38)),
        )

        # Warm ambient lighting
        sim_utils.DomeLightCfg(
            intensity=3500.0,
            color=(0.95, 0.88, 0.75),
        ).func("/World/Light", sim_utils.DomeLightCfg(intensity=3500.0, color=(0.95, 0.88, 0.75)))

        # Four static warehouse walls
        self._spawn_walls()

        # Articulated forklift
        self.forklift = Articulation(FORKLIFT_CFG)

        # Common boxes: kinematic obstacles, parked underground at start
        self.common_boxes: list[RigidObject] = []
        for i in range(_N_BOXES):
            box = RigidObject(RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/CBox_{i:02d}",
                spawn=sim_utils.CuboidCfg(
                    size=(_BOX_SIZE,) * 3,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=40.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=_COMMON_COLOR,
                        roughness=0.8,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, _PARK_Z)),
            ))
            self.common_boxes.append(box)

        # Target box: physics-enabled so the forklift can push / lift it
        self.target_box = RigidObject(RigidObjectCfg(
            prim_path="/World/envs/env_.*/TargetBox",
            spawn=sim_utils.CuboidCfg(
                size=(_BOX_SIZE,) * 3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    linear_damping=0.5,
                    angular_damping=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=_TARGET_COLOR,
                    roughness=0.4,
                    metallic=0.1,
                ),
            ),
        ))

        # Overhead wide-angle camera centred on the warehouse
        self.camera = Camera(CameraCfg(
            prim_path="/World/envs/env_.*/Camera",
            update_period=1 / 30,
            height=512,
            width=512,
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=14.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 120.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 22.0),
                rot=(0.7071, 0.0, 0.0, 0.7071),   # looking straight down
            ),
        ))

        # Register everything with the scene
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["forklift"] = self.forklift
        for i, box in enumerate(self.common_boxes):
            self.scene.rigid_objects[f"cbox_{i:02d}"] = box
        self.scene.rigid_objects["target_box"] = self.target_box
        self.scene.sensors["camera"]           = self.camera

    # ------------------------------------------------------------------
    # Wall helpers
    # ------------------------------------------------------------------

    def _spawn_walls(self):
        """Spawn four kinematic walls enclosing the warehouse."""
        W = _WAREHOUSE_HALF
        T = _WALL_T
        H = _WALL_H

        # Shared visual / physics material
        def _wall_cfg(size):
            return sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1e6),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.78, 0.74, 0.68),
                    roughness=0.9,
                ),
            )

        walls = [
            ("Wall_N", (W * 2 + T * 2, T, H), ( 0.0,  W + T / 2, H / 2)),
            ("Wall_S", (W * 2 + T * 2, T, H), ( 0.0, -W - T / 2, H / 2)),
            ("Wall_E", (T, W * 2 + T * 2, H), ( W + T / 2,  0.0, H / 2)),
            ("Wall_W", (T, W * 2 + T * 2, H), (-W - T / 2,  0.0, H / 2)),
        ]
        for name, size, pos in walls:
            cfg = _wall_cfg(size)
            cfg.func(f"/World/{name}", cfg, translation=pos)

    # ------------------------------------------------------------------
    # Procedural layout generator
    # ------------------------------------------------------------------

    def _generate_layout(self) -> tuple[list[tuple[float, float]], tuple[float, float]]:
        """
        Build a random warehouse layout on an _GRID_N × _GRID_N grid and
        guarantee a navigable path from the forklift's starting cell to the
        target box using BFS.

        Returns
        -------
        box_xy   : list of (x, y) world-frame positions for active common boxes
        target_xy: (x, y) world-frame position for the unique target box
        """
        N = _GRID_N
        C = _CELL

        # Grid-cell centre → world (x, y)
        # Cell (row=0, col=0) is the bottom-left corner.
        # Forklift is at the centre cell (N//2, N//2) = world (0, 0).
        def g2w(row: int, col: int) -> tuple[float, float]:
            x = (col - (N - 1) / 2.0) * C
            y = (row - (N - 1) / 2.0) * C
            return float(x), float(y)

        start_r = N // 2   # = 5 for N=11
        start_c = N // 2

        # 3×3 cells always kept clear so the forklift has room to start
        CLEAR = {
            (start_r + dr, start_c + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
        }

        for _attempt in range(40):
            grid = np.zeros((N, N), dtype=np.int8)   # 0 = free, 1 = box

            # ── random box clusters ──────────────────────────────────────
            n_clusters = int(self._rng.integers(4, 8))   # 4–7 clusters
            for _ in range(n_clusters):
                ch = int(self._rng.integers(1, 3))        # 1–2 rows tall
                cw = int(self._rng.integers(2, 5))        # 2–4 cols wide
                r0 = int(self._rng.integers(0, N - ch + 1))
                c0 = int(self._rng.integers(0, N - cw + 1))
                for r in range(r0, min(r0 + ch, N)):
                    for c in range(c0, min(c0 + cw, N)):
                        if (r, c) not in CLEAR:
                            grid[r, c] = 1

            # ── BFS from forklift start ──────────────────────────────────
            visited: set[tuple[int, int]] = set()
            q: deque[tuple[int, int]] = deque([(start_r, start_c)])
            visited.add((start_r, start_c))
            while q:
                r, c = q.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < N and 0 <= nc < N
                            and grid[nr, nc] == 0
                            and (nr, nc) not in visited):
                        visited.add((nr, nc))
                        q.append((nr, nc))

            # ── candidate cells for the target box ───────────────────────
            # Must be reachable, outside the clear zone, and at least 4
            # Manhattan-distance steps away so it's not trivially close.
            candidates = [
                (r, c) for r, c in visited
                if (r, c) not in CLEAR
                and abs(r - start_r) + abs(c - start_c) >= 4
            ]
            if not candidates:
                continue   # layout too dense — retry

            # Pick target cell randomly
            idx = int(self._rng.integers(0, len(candidates)))
            t_r, t_c = candidates[idx]
            grid[t_r, t_c] = 0   # ensure target cell itself is free

            # ── collect box world positions ──────────────────────────────
            box_xy = [
                g2w(r, c)
                for r in range(N)
                for c in range(N)
                if grid[r, c] == 1
            ]
            target_xy = g2w(t_r, t_c)
            return box_xy, target_xy

        # Fallback: clear warehouse, target in far corner
        print("[WARN] Layout generator fallback — placing target at far corner.")
        return [], g2w(N - 1, N - 1)

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        origins = self.scene.env_origins[env_ids]

        # Reset forklift to its default pose at the centre of the warehouse
        default_root = self.forklift.data.default_root_state[env_ids].clone()
        default_root[:, :3] += origins
        self.forklift.write_root_pose_to_sim(default_root[:, :7], env_ids=env_ids)
        self.forklift.write_root_velocity_to_sim(default_root[:, 7:], env_ids=env_ids)
        dj_pos = self.forklift.data.default_joint_pos[env_ids].clone()
        dj_vel = self.forklift.data.default_joint_vel[env_ids].clone()
        self.forklift.write_joint_state_to_sim(dj_pos, dj_vel, env_ids=env_ids)

        # Clear grab state for every resetting env
        for env_id in env_ids.tolist():
            self._box_grabbed[env_id] = False

        # Per-env layout generation (runs once per reset; num_envs=1 typical)
        for i, env_id in enumerate(env_ids.tolist()):
            origin = origins[i].cpu().numpy()
            ox, oy = float(origin[0]), float(origin[1])

            box_xy, target_xy = self._generate_layout()

            # ── place active common boxes ────────────────────────────────
            env_id_t = torch.tensor([env_id], device=self.device)
            for j, (bx, by) in enumerate(box_xy[:_N_BOXES]):
                pose = torch.zeros(1, 7, device=self.device)
                pose[0, 0] = bx + ox
                pose[0, 1] = by + oy
                pose[0, 2] = _BOX_SIZE / 2   # sit on the floor
                pose[0, 6] = 1.0             # quaternion w
                self.common_boxes[j].write_root_pose_to_sim(pose, env_ids=env_id_t)

            # ── park unused common boxes underground ─────────────────────
            for j in range(len(box_xy), _N_BOXES):
                pose = torch.zeros(1, 7, device=self.device)
                pose[0, 0] = ox
                pose[0, 1] = oy
                pose[0, 2] = _PARK_Z
                pose[0, 6] = 1.0
                self.common_boxes[j].write_root_pose_to_sim(pose, env_ids=env_id_t)

            # ── place target box ─────────────────────────────────────────
            tpose = torch.zeros(1, 7, device=self.device)
            tpose[0, 0] = target_xy[0] + ox
            tpose[0, 1] = target_xy[1] + oy
            tpose[0, 2] = _BOX_SIZE / 2
            tpose[0, 6] = 1.0
            self.target_box.write_root_pose_to_sim(tpose, env_ids=env_id_t)
            self.target_box.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=self.device), env_ids=env_id_t
            )

            n_active = min(len(box_xy), _N_BOXES)
            print(
                f"[INFO] Env {env_id}: {n_active} obstacle boxes placed, "
                f"target at ({target_xy[0]:.1f}, {target_xy[1]:.1f}) world"
            )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        v_x      = self.actions[:, 0]
        omega_z  = self.actions[:, 1]
        fork_cmd = self.actions[:, 2]   # +1 raise, -1 lower, 0 hold

        # ── base: integrate heading directly, then drive root velocity ───────
        # Writing omega_z to velocity AND immediately overwriting the pose with
        # the old heading causes the rotation to be lost.  Instead, advance the
        # heading by omega_z * sim_dt ourselves and bake it into the pose write.
        heading = self.forklift.data.heading_w
        dt = self.cfg.sim.dt                     # one physics sub-step
        new_heading = heading + omega_z * dt

        vel = torch.zeros(self.num_envs, 6, device=self.device)
        vel[:, 0] = v_x * torch.cos(new_heading)
        vel[:, 1] = v_x * torch.sin(new_heading)
        # angular velocity left at 0 — heading is managed by the pose write below
        self.forklift.write_root_velocity_to_sim(vel)

        # ── constrain to ground plane with updated heading ─────────────────
        # chassis centre is 0.6 m above ground (wheel radius 0.3 + joint offset 0.3)
        pose = self.forklift.data.root_state_w[:, :7].clone()
        ground_z = self.scene.env_origins[:, 2] + 0.6   # chassis centre height
        pose[:, 2] = ground_z
        half_yaw = new_heading / 2.0
        pose[:, 3] = torch.cos(half_yaw)   # qw
        pose[:, 4] = 0.0                   # qx  (zero roll)
        pose[:, 5] = 0.0                   # qy  (zero pitch)
        pose[:, 6] = torch.sin(half_yaw)   # qz
        self.forklift.write_root_pose_to_sim(pose)

        # Cosmetic wheel spin
        R = self.cfg.wheel_radius
        L = self.cfg.wheel_base
        omega_left  = (v_x - omega_z * L / 2.0) / R
        omega_right = (v_x + omega_z * L / 2.0) / R
        jvt = self.forklift.data.joint_vel_target.clone()
        jvt[:, self._left_wheel_idx]  = omega_left
        jvt[:, self._right_wheel_idx] = omega_right
        self.forklift.set_joint_velocity_target(jvt)

        # ── fork: direct state write — bypasses actuator spring forces ───
        # Lower limit 0.0 m (tines at ground level); upper 1.5 m.
        fork_pos   = self.forklift.data.joint_pos[:, self._fork_idx].clone()
        fork_delta = fork_cmd * 0.04   # 0.04 m/step at 30 Hz → 1.2 m/s
        new_fork   = torch.clamp(fork_pos + fork_delta, -0.3, 1.5)

        # Teleport the fork joint directly (no actuator spring force on chassis)
        all_jpos = self.forklift.data.joint_pos.clone()
        all_jvel = self.forklift.data.joint_vel.clone()
        all_jpos[:, self._fork_idx] = new_fork
        all_jvel[:, self._fork_idx] = 0.0
        self.forklift.write_joint_state_to_sim(all_jpos, all_jvel)

        # Sync position target → actuator spring force stays at zero
        jpt = self.forklift.data.joint_pos_target.clone()
        jpt[:, self._fork_idx] = new_fork
        self.forklift.set_joint_position_target(jpt)

        # ── kinematic box grab ───────────────────────────────────────────
        self._update_box_grab(fork_cmd, new_fork)

    # ------------------------------------------------------------------
    # Kinematic grab
    # ------------------------------------------------------------------

    def _update_box_grab(self, fork_cmd: torch.Tensor, fork_joint: torch.Tensor):
        """
        Detect when the forklift's forks are positioned under the target box
        and kinematically attach / detach it.

        Grab triggers when ALL of:
          - Box is 0.5–2.2 m forward of chassis in the forklift's body frame
          - Box is within ±0.5 m laterally
          - fork_cmd > 0  (operator is pressing raise)
          - Box is within 0.5 m vertically of where it would sit on the tines

        Carry: each step the box position is set to follow the tine position.
        Release: when fork_joint drops below 0.03 m (forks near the floor).
        """
        fl_pos     = self.forklift.data.root_pos_w   # (N, 3)
        fl_heading = self.forklift.data.heading_w    # (N,)
        box_pos    = self.target_box.data.root_pos_w # (N, 3)

        cos_h = torch.cos(fl_heading)
        sin_h = torch.sin(fl_heading)

        for i in range(self.num_envs):
            j       = fork_joint[i].item()
            cmd     = fork_cmd[i].item()
            cos_i   = cos_h[i].item()
            sin_i   = sin_h[i].item()
            fl_x    = fl_pos[i, 0].item()
            fl_y    = fl_pos[i, 1].item()
            box_x   = box_pos[i, 0].item()
            box_y   = box_pos[i, 1].item()
            box_z   = box_pos[i, 2].item()

            # Box in forklift body frame
            dx  = box_x - fl_x
            dy  = box_y - fl_y
            fwd = dx * cos_i + dy * sin_i          # positive = in front
            lat = abs(-dx * sin_i + dy * cos_i)    # unsigned lateral

            # Tine centre height: chassis(0.6) + mast_joint(-0.3) + fork_lift_origin(0.15)
            #                     + fork_carriage_to_tine(-0.125) = 0.325 + j
            tine_z = j + 0.325

            env_t = torch.tensor([i], device=self.device)

            if not self._box_grabbed[i]:
                # ── try to grab ──────────────────────────────────────────
                in_zone    = 0.5 < fwd < 2.2 and lat < 0.5
                height_ok  = abs(box_z - (tine_z + _BOX_SIZE / 2)) < 0.5
                if in_zone and height_ok and cmd > 0.05:
                    self._box_grabbed[i] = True
                    self._grab_fwd[i] = fwd
                    self._grab_lat[i] = -dx * sin_i + dy * cos_i  # signed
                    print(f"[GRAB] env={i}  fork={j:.3f}m  "
                          f"fwd={fwd:.2f}m  lat={lat:.2f}m")
            else:
                # ── release when forks reach the floor ───────────────────
                if j < 0.03:
                    self._box_grabbed[i] = False
                    print(f"[DROP] env={i}  fork={j:.3f}m")
                else:
                    # ── carry: move box with the forks ───────────────────
                    fwd_i = self._grab_fwd[i]
                    lat_i = self._grab_lat[i]
                    new_bx = fl_x + fwd_i * cos_i - lat_i * sin_i
                    new_by = fl_y + fwd_i * sin_i + lat_i * cos_i
                    new_bz = tine_z + _BOX_SIZE / 2

                    pose = torch.zeros(1, 7, device=self.device)
                    pose[0, 0] = new_bx
                    pose[0, 1] = new_by
                    pose[0, 2] = new_bz
                    pose[0, 6] = 1.0
                    self.target_box.write_root_pose_to_sim(pose, env_ids=env_t)
                    self.target_box.write_root_velocity_to_sim(
                        torch.zeros(1, 6, device=self.device), env_ids=env_t
                    )

    # ------------------------------------------------------------------
    # Observations, rewards, termination
    # ------------------------------------------------------------------

    def _get_observations(self):
        return {
            "rgb":        self.camera.data.output["rgb"],
            "depth":      self.camera.data.output["depth"],
            "target_pos": self.target_box.data.root_pos_w,
            "fork_pos":   self.forklift.data.joint_pos[
                              :, self._fork_idx : self._fork_idx + 1
                          ],
        }

    def _get_rewards(self) -> torch.Tensor:
        forklift_xy = self.forklift.data.root_pos_w[:, :2]
        target_xy   = self.target_box.data.root_pos_w[:, :2]
        dist = torch.norm(forklift_xy - target_xy, dim=1)
        return torch.exp(-dist)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        forklift_xy = self.forklift.data.root_pos_w[:, :2]
        target_xy   = self.target_box.data.root_pos_w[:, :2]
        dist = torch.norm(forklift_xy - target_xy, dim=1)
        success = dist < 0.6
        timeout = self.episode_length_buf >= self.max_episode_length
        return success, timeout


# ---------------------------------------------------------------------------
# Standalone entry point (scene inspection only)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Forklift warehouse — standalone inspection.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    env = ForkliftEnv(ForkliftEnvCfg())
    env.reset()
    print("[INFO] Warehouse environment loaded. Press Ctrl+C to exit.")

    while simulation_app.is_running():
        env.step(torch.zeros(1, 3))

    env.close()
    simulation_app.close()
