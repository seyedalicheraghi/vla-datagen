"""Generic Isaac Lab environment for any registered heavy machine.

AppLauncher must be started by the caller BEFORE importing this module.

    from heavy_machines.machine_env import HeavyMachineEnv, make_env_cfg
    env = HeavyMachineEnv(make_env_cfg("wheel_loader"))
    obs, _ = env.reset()
    obs, rew, term, trunc, info = env.step(torch.zeros(1, env.machine_spec.action_dim))

Action  (all in [-1, 1]):  [drive, turn, tool_1, ..., tool_k]
    drive  +1 = full forward speed, -1 = full reverse speed
    turn   +1 = turn left (CCW from above), -1 = right
    tool_i +1 = tool joint at its maximum rate in its "positive" direction
            (see assets/heavy_machines/<name>/<name>.json for labels and units)

Observation dict:
    "state"      (N, 6 + k): [x, y, yaw, speed, yaw_rate, steer/articulation, tool positions...]
                 x, y are env-local (relative to the env origin)
    "joint_pos"  (N, J):     every articulation joint, in articulation order
    "base_pose_w"(N, 7):     world position + quaternion (w, x, y, z)
    "rgb_front", "rgb_left", "rgb_right" (N, H, W, 3/4) when cameras are enabled

How the machine moves
---------------------
Like the forklift environment, the base is moved kinematically: every physics
substep the environment integrates the drive model (skid steer, Ackermann or
articulated), writes the root pose/velocity, and writes every joint position
directly (tools, steering, articulation, cosmetic wheel spin). Gravity is
disabled on the machine so it never sags; the ground plane and props still
collide with it. This keeps motion exact and stable for data collection,
at the cost of no load/terrain dynamics.
"""

from __future__ import annotations

import math
import random

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from . import kinematics as K
from . import registry
from .spec import ROLLING_ROLES, MachineSpec
from .urdf_builder import write_assets


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@configclass
class HeavyMachineEnvCfg(DirectRLEnvCfg):
    """Use :func:`make_env_cfg` — it fills action/observation sizes from the machine."""

    machine: str = "hydraulic_excavator"

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=4)
    decimation: int = 4                       # 30 Hz control
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=90.0)
    episode_length_s: float = 300.0

    action_space: int = 6
    observation_space: int = 10
    state_space: int = 0

    enable_cameras: bool = True
    camera_resolution: tuple = (224, 224)     # (W, H)
    camera_update_hz: float = 15.0

    # Spawn: env-local (x, y, yaw_rad); used when randomize_spawn is False.
    spawn_pose: tuple = (0.0, 0.0, 0.0)
    randomize_spawn: bool = False
    spawn_region: tuple = (-8.0, -8.0, 8.0, 8.0)   # xmin, ymin, xmax, ymax
    spawn_seed: int | None = None

    # Construction-site props (static colliders) in an annulus around each env origin.
    n_props: int = 28
    props_seed: int = 0
    props_inner_radius: float = 14.0
    props_outer_radius: float = 34.0

    status_interval: int = 60                 # control steps between status lines; 0 = silent


def make_env_cfg(machine: str, **overrides) -> HeavyMachineEnvCfg:
    """Environment config for ``machine`` (registry name or abbreviation)."""
    spec = registry.get(machine)
    cfg = HeavyMachineEnvCfg()
    cfg.machine = spec.name
    cfg.action_space = spec.action_dim
    cfg.observation_space = len(spec.state_labels)
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"HeavyMachineEnvCfg has no field {key!r}")
        setattr(cfg, key, value)
    return cfg


def articulation_cfg(spec: MachineSpec, prim_path: str, fix_base: bool = False) -> ArticulationCfg:
    """ArticulationCfg that imports the machine's generated URDF."""
    urdf_path, _ = write_assets(spec)
    gains = UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=1.0e6, damping=1.0e4)
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UrdfFileCfg(
            asset_path=urdf_path,
            fix_base=fix_base,
            merge_fixed_joints=True,
            make_instanceable=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force", target_type="position", gains=gains),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, K.root_height(spec)),
            joint_pos={j.name: j.init for j in spec.movable_joints},
            joint_vel={".*": 0.0},
        ),
        actuators={
            # Positions are written directly every substep; the PD drive only
            # holds them between writes, so gains are stiff and uniform.
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=1.0e6,
                damping=1.0e4,
                effort_limit_sim=1.0e10,
                velocity_limit_sim=1.0e3,
            ),
        },
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class HeavyMachineEnv(DirectRLEnv):
    """Drive and operate any machine from :mod:`heavy_machines.registry`."""

    cfg: HeavyMachineEnvCfg

    def __init__(self, cfg: HeavyMachineEnvCfg, render_mode: str | None = None, **kwargs):
        self.machine_spec: MachineSpec = registry.get(cfg.machine)
        spec = self.machine_spec
        if cfg.action_space != spec.action_dim:
            raise ValueError(f"cfg.action_space={cfg.action_space} but {spec.name} needs "
                             f"{spec.action_dim}; build the config with make_env_cfg()")
        self._root_height = K.root_height(spec)
        self._spawn_rng = random.Random(cfg.spawn_seed)
        super().__init__(cfg, render_mode, **kwargs)

        self.action_labels = spec.action_labels
        self.state_labels = spec.state_labels
        self._build_joint_maps()
        self._build_buffers()
        self._substep = 0
        self._control_step = 0
        self.last_spawn: list[dict] = [{} for _ in range(self.num_envs)]
        print(f"[heavy_machines] {spec.display_name} ({spec.abbrev}) — {spec.reference}", flush=True)
        print(f"[heavy_machines] action = {self.action_labels}", flush=True)
        print(f"[heavy_machines] joints = {self.robot.joint_names}", flush=True)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def _setup_scene(self):
        spec = self.machine_spec
        spawn_ground_plane("/World/Ground",
                           GroundPlaneCfg(color=(0.43, 0.37, 0.29), size=(500.0, 500.0)))
        light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.98, 0.95, 0.90))
        light.func("/World/Light", light)

        self._spawn_site_props()
        self.robot = Articulation(articulation_cfg(spec, "/World/envs/env_.*/Machine"))

        self.cams: dict[str, Camera] = {}
        if self.cfg.enable_cameras:
            w, h = self.cfg.camera_resolution
            for cam in spec.cameras:
                self.cams[cam.name] = Camera(CameraCfg(
                    prim_path=f"/World/envs/env_.*/MachineCam_{cam.name}",
                    update_period=1.0 / self.cfg.camera_update_hz,
                    height=h, width=w,
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=10.0, horizontal_aperture=20.955,
                        clipping_range=(0.1, 150.0)),
                    offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0),
                                               convention="world"),
                ))

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["machine"] = self.robot
        for name, cam in self.cams.items():
            self.scene.sensors[f"cam_{name}"] = cam

    def _spawn_site_props(self):
        """Static construction-site clutter (dirt piles, boulders, barriers, cones, pipes)."""
        rng = random.Random(self.cfg.props_seed)
        r_in, r_out = self.cfg.props_inner_radius, self.cfg.props_outer_radius
        col = sim_utils.CollisionPropertiesCfg()

        def mat(rgb, rough=0.9):
            return sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=rough)

        kinds = ["dirt", "dirt", "gravel", "boulder", "boulder", "barrier", "barrier",
                 "cone", "cone", "cone", "pipe"]
        for i in range(self.cfg.n_props):
            kind = kinds[i % len(kinds)]
            r = math.sqrt(rng.uniform(r_in * r_in, r_out * r_out))
            a = rng.uniform(-math.pi, math.pi)
            x, y = r * math.cos(a), r * math.sin(a)
            yaw = rng.uniform(-math.pi, math.pi)
            quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
            path = f"/World/envs/env_0/Site/{kind}_{i}"
            if kind in ("dirt", "gravel"):
                rad, hgt = rng.uniform(2.0, 4.0), rng.uniform(1.0, 2.2)
                color = (0.40, 0.29, 0.18) if kind == "dirt" else (0.55, 0.53, 0.50)
                cfg = sim_utils.ConeCfg(radius=rad, height=hgt, collision_props=col,
                                        visual_material=mat(color))
                cfg.func(path, cfg, translation=(x, y, hgt / 2), orientation=quat)
            elif kind == "boulder":
                rad = rng.uniform(0.3, 0.9)
                cfg = sim_utils.SphereCfg(radius=rad, collision_props=col,
                                          visual_material=mat((0.45, 0.44, 0.42)))
                cfg.func(path, cfg, translation=(x, y, rad * 0.7))
            elif kind == "barrier":
                cfg = sim_utils.CuboidCfg(size=(2.0, 0.6, 0.8), collision_props=col,
                                          visual_material=mat((0.78, 0.78, 0.76)))
                cfg.func(path, cfg, translation=(x, y, 0.4), orientation=quat)
            elif kind == "cone":
                cfg = sim_utils.ConeCfg(radius=0.18, height=0.7, collision_props=col,
                                        visual_material=mat((0.95, 0.40, 0.05), 0.6))
                cfg.func(path, cfg, translation=(x, y, 0.35))
            else:  # pipe lying on the ground
                cfg = sim_utils.CylinderCfg(radius=0.45, height=4.0, axis="X", collision_props=col,
                                            visual_material=mat((0.30, 0.30, 0.32), 0.5))
                cfg.func(path, cfg, translation=(x, y, 0.45), orientation=quat)

    # ------------------------------------------------------------------
    # Joint bookkeeping
    # ------------------------------------------------------------------

    def _build_joint_maps(self):
        spec, dev = self.machine_spec, self.device
        names = list(self.robot.joint_names)
        self._jidx = {n: i for i, n in enumerate(names)}
        missing = [j.name for j in spec.movable_joints if j.name not in self._jidx]
        if missing:
            raise RuntimeError(f"joints missing after URDF import: {missing}; got {names}")
        self._num_joints = len(names)

        tools = [spec.joint(t.joint) for t in spec.tools]
        self._tool_ids = torch.tensor([self._jidx[j.name] for j in tools], dtype=torch.long, device=dev)
        self._tool_rate = torch.tensor([j.velocity for j in tools], device=dev)
        self._tool_lo = torch.tensor([j.lower if j.type != "continuous" else -math.inf for j in tools], device=dev)
        self._tool_hi = torch.tensor([j.upper if j.type != "continuous" else math.inf for j in tools], device=dev)
        self._tool_wrap = torch.tensor([j.type == "continuous" for j in tools], dtype=torch.bool, device=dev)

        self._roll: list[tuple[int, str, float]] = [
            (self._jidx[j.name], j.role, j.wheel_radius) for j in spec.joints_with_role(*ROLLING_ROLES)]

        jf = K.joint_frames(spec)
        steers = spec.joints_with_role("steer")
        self._steer_ids = torch.tensor([self._jidx[j.name] for j in steers], dtype=torch.long, device=dev)
        self._steer_y = torch.tensor([jf[j.name].t[1] for j in steers], device=dev)

        d = spec.drive
        self._artic_idx = -1
        self._artic_lo = self._artic_hi = 0.0
        self._x_ref = 0.0
        if d.type == "articulated":
            aj = spec.joint(d.articulation_joint)
            self._artic_idx = self._jidx[aj.name]
            self._artic_lo, self._artic_hi = aj.lower, aj.upper
            self._x_ref = jf[aj.name].t[0] + d.front_length       # front axle in root frame
        elif d.type == "ackermann":
            self._x_ref = d.fixed_axle_x

        self._followers = []
        for j in spec.joints_with_role("follower"):
            src, k, c = j.follows
            lo, hi = (j.lower, j.upper) if j.type != "continuous" else (-math.inf, math.inf)
            self._followers.append((self._jidx[j.name], self._jidx[src], k, c, lo, hi))

        q0 = [0.0] * self._num_joints
        for j in spec.movable_joints:
            q0[self._jidx[j.name]] = j.init
        self._q_init = torch.tensor(q0, device=dev)
        self._joint_names = names               # articulation joint order (for camera FK)

    def _build_buffers(self):
        n, dev = self.num_envs, self.device
        self.actions = torch.zeros(n, self.machine_spec.action_dim, device=dev)
        self._q = self._q_init.repeat(n, 1)
        self._qd = torch.zeros(n, self._num_joints, device=dev)
        self._xy = torch.zeros(n, 2, device=dev)
        self._yaw = torch.zeros(n, device=dev)
        self._speed = torch.zeros(n, device=dev)
        self._yaw_rate = torch.zeros(n, device=dev)
        self._steer = torch.zeros(n, device=dev)      # virtual steer angle or articulation angle

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.to(self.device).float().clamp(-1.0, 1.0)

    def _apply_action(self):
        spec, d = self.machine_spec, self.machine_spec.drive
        dt = self.cfg.sim.dt
        a = self.actions
        drive, turn = a[:, 0], a[:, 1]
        v = torch.where(drive >= 0, drive * d.max_speed, drive * d.reverse_speed)
        qd = torch.zeros_like(self._q)

        # --- base kinematics ------------------------------------------------
        if d.type == "skid":
            w = turn * d.max_yaw_rate
        elif d.type == "ackermann":
            s = 1.0 if d.steer_axle == "front" else -1.0
            target = s * turn * d.max_steer
            step = (target - self._steer).clamp(-d.steer_rate * dt, d.steer_rate * dt)
            self._steer = self._steer + step
            w = s * v * torch.tan(self._steer) / d.wheelbase
            # per-wheel Ackermann angles
            tan_d = torch.tan(self._steer)
            ok = tan_d.abs() > 1e-6
            r_icr = s * d.wheelbase / torch.where(ok, tan_d, torch.ones_like(tan_d))
            ang = s * torch.atan(d.wheelbase / (r_icr.unsqueeze(1) - self._steer_y.unsqueeze(0)))
            ang = torch.where(ok.unsqueeze(1), ang, torch.zeros_like(ang))
            self._q[:, self._steer_ids] = ang
            qd[:, self._steer_ids] = step.unsqueeze(1) / dt
        else:  # articulated
            g_max = min(abs(self._artic_lo), abs(self._artic_hi))
            target = (-turn * g_max).clamp(self._artic_lo, self._artic_hi)     # turn left → gamma < 0
            step = (target - self._steer).clamp(-d.articulation_rate * dt, d.articulation_rate * dt)
            self._steer = self._steer + step
            g_dot = step / dt
            w = -(v * torch.sin(self._steer) + d.rear_length * g_dot) / (
                d.front_length * torch.cos(self._steer) + d.rear_length)
            self._q[:, self._artic_idx] = self._steer
            qd[:, self._artic_idx] = g_dot

        yaw_mid = self._yaw + 0.5 * w * dt
        c, sn = torch.cos(yaw_mid), torch.sin(yaw_mid)
        # Root velocity = v·e + (−x_ref)·w·n  (the reference axle point rolls without slip)
        vx = v * c + self._x_ref * w * sn
        vy = v * sn - self._x_ref * w * c
        self._xy[:, 0] += vx * dt
        self._xy[:, 1] += vy * dt
        self._yaw = torch.atan2(torch.sin(self._yaw + w * dt), torch.cos(self._yaw + w * dt))
        self._speed, self._yaw_rate = v, w

        # --- cosmetic rolling elements --------------------------------------
        half_w = 0.5 * d.track_width
        for idx, role, radius in self._roll:
            if role == "roll_left":
                sp = v - w * half_w
            elif role == "roll_right":
                sp = v + w * half_w
            else:
                sp = v
            omega = sp / radius
            ang = self._q[:, idx] + omega * dt
            self._q[:, idx] = torch.atan2(torch.sin(ang), torch.cos(ang))
            qd[:, idx] = omega

        # --- tools --------------------------------------------------------------
        if len(spec.tools):
            rate = a[:, 2:] * self._tool_rate
            q = self._q[:, self._tool_ids] + rate * dt
            q = torch.where(self._tool_wrap, torch.atan2(torch.sin(q), torch.cos(q)),
                            torch.maximum(torch.minimum(q, self._tool_hi), self._tool_lo))
            hit = (q <= self._tool_lo) | (q >= self._tool_hi)
            self._q[:, self._tool_ids] = q
            qd[:, self._tool_ids] = torch.where(hit, torch.zeros_like(rate), rate)

        for idx, src, k, c0, lo, hi in self._followers:
            self._q[:, idx] = (k * self._q[:, src] + c0).clamp(lo, hi)
            qd[:, idx] = k * qd[:, src]
        self._qd = qd

        self._write_state()

        self._substep = (self._substep + 1) % self.cfg.decimation
        if self._substep == 0 and self.cams:
            self._update_cameras()

    def _root_pose_w(self, env_ids=None) -> torch.Tensor:
        ids = slice(None) if env_ids is None else env_ids
        origins = self.scene.env_origins[ids]
        yaw = self._yaw[ids]
        pose = torch.zeros(origins.shape[0], 7, device=self.device)
        pose[:, 0] = origins[:, 0] + self._xy[ids, 0]
        pose[:, 1] = origins[:, 1] + self._xy[ids, 1]
        pose[:, 2] = origins[:, 2] + self._root_height
        pose[:, 3] = torch.cos(yaw / 2)
        pose[:, 6] = torch.sin(yaw / 2)
        return pose

    def _write_state(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        pose = self._root_pose_w(env_ids)
        vel = torch.zeros(pose.shape[0], 6, device=self.device)
        if env_ids is None:
            c, s = torch.cos(self._yaw), torch.sin(self._yaw)
            vel[:, 0] = self._speed * c + self._x_ref * self._yaw_rate * s
            vel[:, 1] = self._speed * s - self._x_ref * self._yaw_rate * c
            vel[:, 5] = self._yaw_rate
        self.robot.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(vel, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(self._q[ids], self._qd[ids], env_ids=env_ids)
        self.robot.set_joint_position_target(self._q[ids], env_ids=env_ids)

    # ------------------------------------------------------------------
    # Cameras (pose = base pose ∘ forward kinematics ∘ mount)
    # ------------------------------------------------------------------

    def _update_cameras(self, env_ids=None):
        spec = self.machine_spec
        ids = list(range(self.num_envs)) if env_ids is None else [int(i) for i in env_ids]
        q_cpu = self._q[ids].detach().cpu().tolist()
        pose_cpu = self._root_pose_w(torch.tensor(ids, device=self.device)).detach().cpu().tolist()
        yaw_cpu = self._yaw[ids].detach().cpu().tolist()
        pos = {c.name: [] for c in spec.cameras}
        rot = {c.name: [] for c in spec.cameras}
        names = self._joint_names
        for row, (qrow, prow, yaw) in enumerate(zip(q_cpu, pose_cpu, yaw_cpu)):
            qdict = {names[i]: qrow[i] for i in range(len(names))}
            tfs = K.link_transforms(spec, qdict)
            base = K.Tf.planar(prow[0], prow[1], prow[2], yaw)
            for cam in spec.cameras:
                tf = base @ tfs[cam.link] @ K.Tf.from_xyz_rpy(cam.xyz, cam.rpy)
                pos[cam.name].append(list(tf.t))
                rot[cam.name].append(list(K.mat_to_quat_wxyz(tf.R)))
        for cam in spec.cameras:
            self.cams[cam.name].set_world_poses(
                torch.tensor(pos[cam.name], device=self.device),
                torch.tensor(rot[cam.name], device=self.device),
                env_ids=ids, convention="world")

    # ------------------------------------------------------------------
    # Observations / rewards / termination
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        state = torch.cat([
            self._xy,
            self._yaw.unsqueeze(1),
            self._speed.unsqueeze(1),
            self._yaw_rate.unsqueeze(1),
            self._steer.unsqueeze(1),
            self._q[:, self._tool_ids],
        ], dim=1)
        obs = {"policy": state, "state": state, "joint_pos": self._q.clone(),
               "base_pose_w": self._root_pose_w()}
        for name, cam in self.cams.items():
            obs[f"rgb_{name}"] = cam.data.output["rgb"]

        self._control_step += 1
        if self.cfg.status_interval and self._control_step % self.cfg.status_interval == 0:
            labels = [t.label for t in self.machine_spec.tools]
            tools = " ".join(f"{lb}={v:+.2f}" for lb, v in zip(labels, state[0, 6:].tolist()))
            print(f"[{self.machine_spec.abbrev} step={self._control_step}] "
                  f"x={state[0, 0]:.2f} y={state[0, 1]:.2f} yaw={math.degrees(state[0, 2]):.0f}° "
                  f"v={state[0, 3]:+.2f} w={state[0, 4]:+.2f} steer={state[0, 5]:+.2f} | {tools}",
                  flush=True)
        return obs

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(timeout), timeout

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        for e in env_ids.tolist():
            if self.cfg.randomize_spawn:
                x0, y0, x1, y1 = self.cfg.spawn_region
                x, y = self._spawn_rng.uniform(x0, x1), self._spawn_rng.uniform(y0, y1)
                yaw = self._spawn_rng.uniform(-math.pi, math.pi)
            else:
                x, y, yaw = self.cfg.spawn_pose
            self._xy[e, 0], self._xy[e, 1], self._yaw[e] = x, y, yaw
            self.last_spawn[e] = {"xy": (x, y), "yaw_rad": yaw}
        self._q[env_ids] = self._q_init
        self._qd[env_ids] = 0.0
        self._steer[env_ids] = 0.0
        self._speed[env_ids] = 0.0
        self._yaw_rate[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self._write_state(env_ids)
        if self.cams:
            self._update_cameras(env_ids)
