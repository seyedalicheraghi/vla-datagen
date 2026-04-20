"""
Forklift warehouse environment.

Physics-based warehouse with a GMA 48×40 compound pallet (real fork-pocket geometry)
and 4 individual cardboard boxes stacked on top.  The operator drives the forklift,
inserts the forks into the pallet's lower clearance gap, and lifts the entire load.

This file defines ForkliftEnv and ForkliftEnvCfg as importable classes.
AppLauncher must be initialised by the caller BEFORE importing this module.

Standalone usage:
    ./isaaclab.sh -p scripts/forklift/forklift_env.py
"""

# ---------------------------------------------------------------------------
# Isaac Lab imports — safe because caller has already run AppLauncher
# ---------------------------------------------------------------------------

import math
import os
import numpy as np
from collections import deque

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg, TiledCamera, TiledCameraCfg
from isaaclab.sensors import RayCaster, RayCasterCfg
from isaaclab.sensors.ray_caster import patterns as rc_patterns
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Warehouse layout constants
# ---------------------------------------------------------------------------

_WAREHOUSE_HALF = 16.0   # half-width of warehouse floor → 32 m × 32 m
_WALL_T         = 0.3
_WALL_H         = 4.0

_GRID_N = 15             # grid cells per axis (keeps placements inside walls)
_CELL   = 2.0            # metres per cell

# ---------------------------------------------------------------------------
# GMA 48×40 pallet geometry  (all dimensions in metres)
# ---------------------------------------------------------------------------
#
#  Side view (approach from +X):
#
#   ┌──────────────────────────────────┐  ← top deck  (DECK_H = 25 mm)
#   │                                  │
#   │  ┌──────┐  ┌──┐  ┌──────┐       │  ← stringers  (STG_H = 200 mm)
#   │  │      │  │  │  │      │       │    stringer notch (NOTCH_W = 200 mm)
#   │  └──────┘  └──┘  └──────┘       │    provides 4-way fork entry
#   └──────────────────────────────────┘  ← bottom blocks  (BOT_H = 60 mm)
#
#   Fork pocket height = STG_H — must be tall enough for ForkliftC tines
#   Forks must be at tine_z < STG_H + small_tolerance to enter
#

_PALLET_L       = 1.219   # 48 in — forklift approaches along this axis
_PALLET_W       = 1.500   # wider pallet for easier fork entry
_PALLET_DECK_H  = 0.025   # top deck thickness
_PALLET_STG_H   = 0.200   # stringer height — tall enough for ForkliftC fork tines
_PALLET_BOT_H   = 0.060   # bottom blocks — ground clearance for fork entry
_PALLET_H       = _PALLET_DECK_H + _PALLET_STG_H + _PALLET_BOT_H   # ~0.285 m
_PALLET_CL      = _PALLET_STG_H                                      # 0.200 m
_PALLET_STG_W   = 0.090   # stringer width  (~3.5 in)
_PALLET_NOTCH_W = 0.200   # 4-way fork-entry notch width in each stringer (~8 in)
_PALLET_MASS    = 25.0    # kg — empty GMA pallet
_PALLET_COLOR        = (0.62, 0.44, 0.22)   # weathered pine
_PALLET_COLOR_TARGET = (0.20, 0.45, 0.75)   # blue — target pallet stands out

# ---------------------------------------------------------------------------
# Cargo boxes  (individual rigid bodies stacked on the pallet)
# ---------------------------------------------------------------------------
#
#  Layout — 2 × 2 grid, single layer:
#
#   ┌───────┬───────┐
#   │  box  │  box  │
#   ├───────┼───────┤
#   │  box  │  box  │
#   └───────┴───────┘
#   ← 1.219 m pallet →

_BOX_L    = 0.35     # m — fits three along pallet length with margin
_BOX_W    = 0.40     # m — fits three along pallet width with margin
_BOX_H    = 0.60     # m
_BOX_MASS = 12.0     # kg each
_BOX_COLS = 3        # columns along pallet L  (X-axis)
_BOX_ROWS = 3        # rows    along pallet W  (Y-axis)
_N_BOXES  = _BOX_COLS * _BOX_ROWS    # = 9

# Pre-compute each box's LOCAL offset from pallet centre (z = 0 = pallet bottom)
_BOX_LOCAL_OFFSETS: list[tuple[float, float, float]] = []
for _col in range(_BOX_COLS):
    for _row in range(_BOX_ROWS):
        _x = (_col - (_BOX_COLS - 1) / 2.0) * (_BOX_L + 0.02)
        _y = (_row - (_BOX_ROWS - 1) / 2.0) * (_BOX_W + 0.02)
        _z = _PALLET_H + _BOX_H / 2.0
        _BOX_LOCAL_OFFSETS.append((_x, _y, _z))

_BOX_COLOR        = (0.80, 0.65, 0.45)   # cardboard brown
_BOX_COLOR_TARGET = (0.30, 0.55, 0.85)  # blue — matches target pallet
_PARK_Z           = -60.0               # underground parking for inactive units

# Interactable pallets (can be picked up, carried, stacked)
_N_INTERACTABLE   = 4                   # number of interactable pallet+cargo sets
_PALLET_COLORS    = [
    _PALLET_COLOR_TARGET,               # pallet 0: blue (primary pick target)
    _PALLET_COLOR,                      # pallet 1: brown
    (0.50, 0.35, 0.18),                 # pallet 2: darker brown
    (0.55, 0.42, 0.25),                 # pallet 3: medium brown
]
_CARGO_COLORS     = [
    _BOX_COLOR_TARGET,                  # pallet 0 cargo: blue
    _BOX_COLOR,                         # pallet 1 cargo: cardboard
    (0.72, 0.55, 0.35),                 # pallet 2 cargo: lighter cardboard
    (0.65, 0.45, 0.28),                 # pallet 3 cargo: dark cardboard
]
# Total cargo height per pallet+cargo unit (for stacking)
_UNIT_H = _PALLET_H + _BOX_H           # pallet + one layer of boxes

# ---------------------------------------------------------------------------
# Scattered warehouse boxes (decorative / obstacles)
# ---------------------------------------------------------------------------

_SCATTER_COLORS = [
    _BOX_COLOR,            # cardboard brown (same as cargo boxes)
    _BOX_COLOR_TARGET,     # gray (same as target box)
    (0.72, 0.55, 0.35),   # lighter cardboard
    (0.65, 0.45, 0.28),   # darker cardboard
]

_N_SCATTER_PALLETS = 50      # number of pallet+box sets around the warehouse

# ---------------------------------------------------------------------------
# Sensor constants
# ---------------------------------------------------------------------------

# Ouster OS1-64 LiDAR defaults
_LIDAR_CHANNELS     = 64
_LIDAR_VERT_FOV     = (-16.6, 16.6)    # degrees
_LIDAR_HORIZ_FOV    = (-180.0, 180.0)  # full 360°
_LIDAR_HORIZ_RES    = 360.0 / 1024     # ~0.3516° → 1024 horizontal samples
_LIDAR_MAX_RANGE    = 120.0            # metres
_LIDAR_UPDATE_HZ    = 10.0             # Hz
_LIDAR_MOUNT_FWD    = 1.0              # metres forward of forklift root
_LIDAR_MOUNT_UP     = 2.5              # metres above ground — sees top of stacked cargo

# Camera defaults
_CAM_W, _CAM_H      = 224, 224         # OpenPI-friendly resolution
_CAM_UPDATE_HZ       = 15.0            # Hz (10–20 typical for VLA training)
_DEBUG_SENSOR_DIR    = os.path.join(os.path.dirname(__file__), "debug_sensors")


# ---------------------------------------------------------------------------
# Compound pallet builder  (module-level — called during _setup_scene)
# ---------------------------------------------------------------------------

def _build_compound_pallet(stage, root_path: str, color=_PALLET_COLOR) -> None:
    """Create a GMA pallet as a compound USD rigid body at *root_path*.

    Structure (z = 0 is the pallet bottom surface):
    - 1  top deck  (full L × W footprint)
    - 3 stringers × 2 half-pieces = 6 pieces (4-way notch at x = 0)
    - 6 bottom blocks (one directly under each stringer piece)

    Total = 13 child Cube prims under a kinematic Xform root.

    Fork entry — long-side (X approach):
        The two open lanes between left/centre and centre/right stringers.
    Fork entry — short-side (Y approach):
        The 200 mm notch cut in every stringer at x = 0.
    """
    from pxr import UsdGeom, UsdPhysics, Gf

    L, W        = _PALLET_L, _PALLET_W
    top_h       = _PALLET_DECK_H
    stg_h       = _PALLET_STG_H
    bot_h       = _PALLET_BOT_H
    stg_w       = _PALLET_STG_W
    notch_w     = _PALLET_NOTCH_W
    stg_half_l  = (L - notch_w) / 2.0    # 0.510 m

    # ── root Xform ───────────────────────────────────────────────────────
    stage.DefinePrim(root_path, "Xform")
    root = stage.GetPrimAtPath(root_path)

    # Kinematic rigid body (we move it ourselves via write_root_pose_to_sim)
    rb_api = UsdPhysics.RigidBodyAPI.Apply(root)
    rb_api.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr().Set(_PALLET_MASS)

    col = Gf.Vec3f(*color)

    def _cube(name: str, size: tuple, pos: tuple,
             collide: bool = False) -> None:
        """Add a scaled unit-cube child prim.

        When *collide* is True a ``UsdPhysics.CollisionAPI`` is applied so
        PhysX treats this piece as a solid wall for dynamic bodies (the
        forklift articulation).  Stringers are left collision-free so the
        fork tines can enter the pocket between them.
        """
        p = f"{root_path}/{name}"
        cube = UsdGeom.Cube.Define(stage, p)
        cube.GetSizeAttr().Set(1.0)           # unit cube, scaled below
        xf = UsdGeom.XformCommonAPI(cube.GetPrim())
        xf.SetTranslate(Gf.Vec3d(*pos))
        xf.SetScale(Gf.Vec3f(*size))
        cube.GetDisplayColorAttr().Set([col])
        if collide:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    # ── Top deck (with collision) ────────────────────────────────────────
    _cube("top_deck",
          (L, W, top_h),
          (0.0, 0.0, bot_h + stg_h + top_h / 2),
          collide=True)

    # ── 3 stringers × 2 half-pieces  +  6 bottom blocks ─────────────────
    stg_ys = [-W / 3, 0.0, W / 3]   # left, centre, right stringer Y positions
    for si, sy in enumerate(stg_ys):
        for pi, sign in enumerate([-1, 1]):
            px = sign * (notch_w / 2 + stg_half_l / 2)

            # Stringer piece — NO collision (fork pocket must stay open)
            _cube(f"stg_{si}_{pi}",
                  (stg_half_l, stg_w, stg_h),
                  (px, sy, bot_h + stg_h / 2))

            # Bottom block (with collision)
            _cube(f"bot_{si}_{pi}",
                  (stg_half_l, stg_w, bot_h),
                  (px, sy, bot_h / 2),
                  collide=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@configclass
class ForkliftEnvCfg(DirectRLEnvCfg):
    """Settings for the warehouse forklift environment."""

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=4)
    decimation: int = 4
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=70.0)
    episode_length_s: float = 120.0

    observation_space: int = 64
    action_space: int = 3        # [v_x, omega_z, fork_cmd]
    state_space: int = 0

    # ForkliftC kinematics (Ackermann, rear-wheel-steered, front-wheel-driven)
    wheel_radius: float = 0.325
    wheel_base:   float = 1.65
    chassis_z_offset: float = 0.0   # ForkliftC USD root at ground level


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ForkliftEnv(DirectRLEnv):
    """Warehouse forklift with a compound pallet and stacked boxes."""

    cfg: ForkliftEnvCfg

    def __init__(self, cfg: ForkliftEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.actions = torch.zeros(self.num_envs, 3, device=self.device)

        all_joints = self.forklift.joint_names
        print(f"[INFO] ForkliftC joints: {all_joints}", flush=True)

        # Drive wheels (all 4 spun cosmetically)
        self._lf_wheel_idx = self.forklift.find_joints("left_front_wheel_joint")[0][0]
        self._rf_wheel_idx = self.forklift.find_joints("right_front_wheel_joint")[0][0]
        self._lb_wheel_idx = self.forklift.find_joints("left_back_wheel_joint")[0][0]
        self._rb_wheel_idx = self.forklift.find_joints("right_back_wheel_joint")[0][0]

        # Ackermann steering
        self._l_steer_idx = self.forklift.find_joints("left_rotator_joint")[0][0]
        self._r_steer_idx = self.forklift.find_joints("right_rotator_joint")[0][0]

        # Fork lift prismatic joint
        self._fork_idx = self.forklift.find_joints("lift_joint")[0][0]

        print(f"[INFO] lf={self._lf_wheel_idx} rf={self._rf_wheel_idx} "
              f"lb={self._lb_wheel_idx} rb={self._rb_wheel_idx} "
              f"l_steer={self._l_steer_idx} r_steer={self._r_steer_idx} "
              f"fork={self._fork_idx}", flush=True)

        self._rng = np.random.default_rng()

        # Print sensor banner after scene is set up
        self._print_sensor_banner()

        # Authoritative fork position — never read back from physics solver
        # (collision with carried pallet corrupts the solver's joint state)
        self._fork_pos = torch.zeros(self.num_envs, device=self.device)

        # Authoritative heading — solver heading gets corrupted by collisions
        # with kinematic pallet/boxes, causing drift even with zero input.
        self._heading = torch.zeros(self.num_envs, device=self.device)

        # Tracked X,Y position — the physics solver's position gets corrupted
        # by collisions with kinematic loads, so we integrate ourselves.
        self._carry_pos = torch.zeros(self.num_envs, 2, device=self.device)

        # Per-env grab state (multi-pallet)
        self._grabbed_idx:       list[int]   = [-1]    * self.num_envs  # which pallet is grabbed (-1=none)
        self._grab_fwd:          list[float] = [0.0]   * self.num_envs
        self._grab_lat:          list[float] = [0.0]   * self.num_envs
        self._grab_heading:      list[float] = [0.0]   * self.num_envs
        # Box offsets relative to pallet when grabbed (body-frame)
        # Indexed as [env_id][pallet_idx] → list of (x,y,z) offsets
        self._grab_box_offsets: list[list[list[tuple[float, float, float]]]] = \
            [[list(_BOX_LOCAL_OFFSETS) for _ in range(_N_INTERACTABLE)]
             for _ in range(self.num_envs)]
        # Track base-z for each pallet (ground level or stacked height)
        self._pallet_base_z: list[list[float]] = \
            [[0.0] * _N_INTERACTABLE for _ in range(self.num_envs)]

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        import omni.usd
        from isaaclab_assets.robots.forklift import FORKLIFT_CFG

        # ── Warehouse floor (flat gray) ───────────────────────────────
        spawn_ground_plane("/World/Ground",
                           cfg=GroundPlaneCfg(color=(0.45, 0.45, 0.45), size=(200.0, 200.0)))

        # ── Lighting ─────────────────────────────────────────────────
        sim_utils.DomeLightCfg(
            intensity=3500.0,
            color=(0.95, 0.88, 0.75),
        ).func("/World/Light",
               sim_utils.DomeLightCfg(intensity=3500.0, color=(0.95, 0.88, 0.75)))

        # ── Warehouse walls ───────────────────────────────────────────
        self._spawn_walls()

        # ── Scattered warehouse boxes (decoration / obstacles) ────────
        self._spawn_scatter_boxes()

        # ── Forklift articulation ─────────────────────────────────────
        self.forklift = Articulation(FORKLIFT_CFG)

        # ── Interactable pallets (can be picked up, carried, stacked) ──
        # Build at env_0 BEFORE cloning — the cloner copies to all envs.
        stage = omni.usd.get_context().get_stage()

        self.pallets: list[RigidObject] = []
        self.pallet_boxes: list[list[RigidObject]] = []

        for pi in range(_N_INTERACTABLE):
            pal_color = _PALLET_COLORS[pi % len(_PALLET_COLORS)]
            box_color = _CARGO_COLORS[pi % len(_CARGO_COLORS)]

            # Build compound pallet USD prim
            _build_compound_pallet(stage, f"/World/envs/env_0/Pallet_{pi}",
                                   color=pal_color)
            pallet = RigidObject(RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Pallet_{pi}",
                spawn=None,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, _PARK_Z),
                ),
            ))
            self.pallets.append(pallet)

            # Cargo boxes for this pallet
            boxes_for_pallet: list[RigidObject] = []
            for bi in range(_N_BOXES):
                box = RigidObject(RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/CargoBox_{pi}_{bi}",
                    spawn=sim_utils.CuboidCfg(
                        size=(_BOX_L, _BOX_W, _BOX_H),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=True,
                            linear_damping=0.5,
                            angular_damping=2.0,
                        ),
                        mass_props=sim_utils.MassPropertiesCfg(mass=_BOX_MASS),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=box_color,
                            roughness=0.85,
                            metallic=0.0,
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(0.0, 0.0, _PARK_Z)),
                ))
                boxes_for_pallet.append(box)
            self.pallet_boxes.append(boxes_for_pallet)

        # Keep backward compat aliases for the primary (blue) pallet
        self.pallet = self.pallets[0]
        self.boxes  = self.pallet_boxes[0]

        # Collision spawn report
        for pi in range(_N_INTERACTABLE):
            print(f"[spawn] pallet_{pi}  rigid=kinematic  collider=compound(13 cubes)  "
                  f"mass={_PALLET_MASS}kg", flush=True)
            for bi in range(_N_BOXES):
                print(f"[spawn] cargo_{pi}_{bi}  rigid=kinematic  collider=cuboid  "
                      f"mass={_BOX_MASS}kg  size=({_BOX_L},{_BOX_W},{_BOX_H})", flush=True)

        # ── Three RGB cameras (224×224, ~15 Hz, world-pose updated each step)
        _cam_cfg = dict(
            update_period=1 / _CAM_UPDATE_HZ,
            height=_CAM_H,
            width=_CAM_W,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 80.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="world",
            ),
        )
        self.cam_front = Camera(CameraCfg(
            prim_path="/World/envs/env_.*/CamFrontCabin", **_cam_cfg))
        self.cam_top_left = Camera(CameraCfg(
            prim_path="/World/envs/env_.*/CamTopLeft", **_cam_cfg))
        self.cam_top_right = Camera(CameraCfg(
            prim_path="/World/envs/env_.*/CamTopRight", **_cam_cfg))

        # ── Ouster OS1-64 LiDAR (RayCaster with LidarPatternCfg) ─────
        # prim_path must point to an existing physics body — the forklift
        self.lidar = RayCaster(RayCasterCfg(
            prim_path="/World/envs/env_.*/Forklift",
            mesh_prim_paths=["/World/Ground"],
            offset=RayCasterCfg.OffsetCfg(
                pos=(_LIDAR_MOUNT_FWD, 0.0, _LIDAR_MOUNT_UP),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            ray_alignment="base",
            pattern_cfg=rc_patterns.LidarPatternCfg(
                channels=_LIDAR_CHANNELS,
                vertical_fov_range=_LIDAR_VERT_FOV,
                horizontal_fov_range=_LIDAR_HORIZ_FOV,
                horizontal_res=_LIDAR_HORIZ_RES,
            ),
            max_distance=_LIDAR_MAX_RANGE,
            update_period=1 / _LIDAR_UPDATE_HZ,
            debug_vis=False,
        ))

        # ── Register with scene ───────────────────────────────────────
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["forklift"] = self.forklift
        for pi, pallet in enumerate(self.pallets):
            self.scene.rigid_objects[f"pallet_{pi}"] = pallet
            for bi, box in enumerate(self.pallet_boxes[pi]):
                self.scene.rigid_objects[f"box_{pi}_{bi}"] = box
        self.scene.sensors["cam_front"]     = self.cam_front
        self.scene.sensors["cam_top_left"]  = self.cam_top_left
        self.scene.sensors["cam_top_right"] = self.cam_top_right
        self.scene.sensors["lidar"]         = self.lidar

        # Observability state
        self._debug_sensors_saved = False
        self._step_count = 0           # global step counter (not reset per episode)
        self._status_interval = 20     # print status every N control steps
        self._lidar_status = "INIT"    # last known LiDAR health
        self._cam_status = {"front": "INIT", "top_l": "INIT", "top_r": "INIT"}
        self._last_lidar_pts = 0
        self._last_lidar_range = (0.0, 0.0)

    # ------------------------------------------------------------------
    # Observability — banner, per-step status, LiDAR sanity dump
    # ------------------------------------------------------------------

    def _print_sensor_banner(self):
        """Print a startup banner listing every sensor and its config."""
        sim_dt = self.cfg.sim.dt
        ctrl_dt = sim_dt * self.cfg.decimation
        render_dt = sim_dt * self.cfg.sim.render_interval

        lines = ["=" * 60, "  SIM SENSORS", "=" * 60]

        # LiDAR
        try:
            n_rays = self.lidar.num_rays if hasattr(self.lidar, 'num_rays') else "?"
            lines.append(
                f"  LiDAR  [ENABLED]   Ouster OS1-64  |  {_LIDAR_CHANNELS} beams  |  "
                f"FOV +{_LIDAR_VERT_FOV[1]}\u00b0/{_LIDAR_VERT_FOV[0]}\u00b0  |  "
                f"1024 h-samples  |  {_LIDAR_UPDATE_HZ} Hz  |  {_LIDAR_MAX_RANGE} m  |  "
                f"mount: Forklift ({_LIDAR_MOUNT_FWD}, 0.0, {_LIDAR_MOUNT_UP})")
        except Exception as e:
            lines.append(f"  LiDAR  [DISABLED: {e}]")

        # Cameras
        for name, cam in [("front_cabin", self.cam_front),
                          ("top_left", self.cam_top_left),
                          ("top_right", self.cam_top_right)]:
            try:
                lines.append(
                    f"  Camera [ENABLED]   {name:12s}  |  {_CAM_W}x{_CAM_H} RGB  |  "
                    f"{_CAM_UPDATE_HZ} Hz  |  "
                    f"prim: {cam.cfg.prim_path}")
            except Exception as e:
                lines.append(f"  Camera [DISABLED: {name} — {e}]")

        lines.append(
            f"  Physics dt=1/{int(1/sim_dt)}s  |  Control dt=1/{int(1/ctrl_dt)}s  |  "
            f"Render dt=1/{int(1/render_dt)}s")
        lines.append("=" * 60)

        for ln in lines:
            print(ln, flush=True)

    def _print_step_status(self, obs: dict):
        """Print one status line with LiDAR + camera health."""
        self._step_count += 1
        if self._step_count % self._status_interval != 0:
            return

        sim_time = self._step_count * self.cfg.sim.dt * self.cfg.decimation
        x = self._carry_pos[0, 0].item()
        y = self._carry_pos[0, 1].item()
        yaw = math.degrees(self._heading[0].item())
        vx = self.actions[0, 0].item()
        vy = 0.0  # no lateral in Ackermann
        fork_h = self._fork_pos[0].item()

        # LiDAR health
        lidar_data = obs.get("lidar")
        if lidar_data is not None:
            hits = lidar_data[0].cpu().numpy()
            sensor_pos = self.lidar.data.pos_w[0].cpu().numpy()
            dists = np.linalg.norm(hits - sensor_pos[np.newaxis, :], axis=-1)
            valid = dists < _LIDAR_MAX_RANGE * 0.99
            n_valid = int(valid.sum())
            if n_valid == 0:
                self._lidar_status = "NO_RETURNS"
                self._last_lidar_pts = 0
                self._last_lidar_range = (0.0, 0.0)
            else:
                valid_dists = dists[valid]
                self._lidar_status = "OK"
                self._last_lidar_pts = n_valid
                self._last_lidar_range = (float(valid_dists.min()), float(valid_dists.max()))
        else:
            self._lidar_status = "NO_DATA"
            self._last_lidar_pts = 0

        lidar_str = (f"LiDAR: {self._lidar_status}"
                     if self._lidar_status != "OK"
                     else f"LiDAR: pts={self._last_lidar_pts} "
                          f"range[{self._last_lidar_range[0]:.2f}-"
                          f"{self._last_lidar_range[1]:.2f}]m")

        # Camera health
        cam_strs = []
        for key, label in [("rgb_front", "front"), ("rgb_left", "top_l"), ("rgb_right", "top_r")]:
            img = obs.get(key)
            if img is None:
                self._cam_status[label] = "NONE"
            else:
                arr = img[0]
                if arr.max().item() == 0:
                    self._cam_status[label] = "BLACK"
                else:
                    self._cam_status[label] = "OK"
            cam_strs.append(f"{label}={self._cam_status[label]}")

        grabbed = self._grabbed_idx[0]
        act = self.actions[0].cpu().tolist()
        act_str = f"[{act[0]:.1f},{act[1]:.2f},{act[2]:.1f}]"

        print(
            f"[t={sim_time:.2f}s step={self._step_count}] "
            f"base=(x={x:.2f}, y={y:.2f}, yaw={yaw:.1f}\u00b0)  "
            f"v=({vx:.2f}, {vy:.2f}) fork_h={fork_h:.2f}m  |  "
            f"{lidar_str}  |  Cams: {' '.join(cam_strs)}  |  "
            f"action={act_str} attach={grabbed}",
            flush=True)

    def _save_lidar_sanity_dump(self, obs: dict):
        """Save LiDAR first-frame dump to debug_sensors/."""
        os.makedirs(_DEBUG_SENSOR_DIR, exist_ok=True)

        lidar_data = obs.get("lidar")
        if lidar_data is None:
            with open(os.path.join(_DEBUG_SENSOR_DIR, "lidar_stats.txt"), "w") as f:
                f.write("LiDAR returned None — sensor may not be initialized\n")
            print("[WARN] LiDAR sanity dump: NO DATA", flush=True)
            return

        hits = lidar_data[0].cpu().numpy()
        sensor_pos = self.lidar.data.pos_w[0].cpu().numpy()
        dists = np.linalg.norm(hits - sensor_pos[np.newaxis, :], axis=-1)
        valid_mask = dists < _LIDAR_MAX_RANGE * 0.99
        n_total = len(dists)
        n_valid = int(valid_mask.sum())

        # Save raw point cloud
        np.save(os.path.join(_DEBUG_SENSOR_DIR, "lidar_first_frame.npy"), hits)

        # Save top-down PNG
        try:
            from PIL import Image as PILImage
            IMG_SIZE = 512
            RANGE_M = 40.0
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            if n_valid > 0:
                pts = hits[valid_mask]
                vd = dists[valid_mask]
                px = ((pts[:, 0] - sensor_pos[0]) / RANGE_M * IMG_SIZE / 2
                      + IMG_SIZE / 2).astype(int)
                py = ((pts[:, 1] - sensor_pos[1]) / RANGE_M * IMG_SIZE / 2
                      + IMG_SIZE / 2).astype(int)
                mask = (px >= 0) & (px < IMG_SIZE) & (py >= 0) & (py < IMG_SIZE)
                # Color by height: low=blue, mid=green, high=red
                hz = pts[mask, 2]
                r = np.clip((hz * 80).astype(int), 0, 255).astype(np.uint8)
                g = np.clip((120 - abs(hz - 1.5) * 60).astype(int), 0, 255).astype(np.uint8)
                b = np.clip((255 - hz * 80).astype(int), 0, 255).astype(np.uint8)
                img[py[mask], px[mask]] = np.stack([r, g, b], axis=-1)
            pil_img = PILImage.fromarray(img)
            pil_img.save(os.path.join(_DEBUG_SENSOR_DIR, "lidar_first_frame_topdown.png"))
        except Exception as e:
            print(f"[WARN] Could not save LiDAR PNG: {e}", flush=True)

        # Save stats
        stats_lines = [
            f"point_count_total: {n_total}",
            f"point_count_valid: {n_valid} ({100*n_valid/max(n_total,1):.1f}%)",
            f"shape: {hits.shape}",
        ]
        if n_valid > 0:
            valid_dists = dists[valid_mask]
            stats_lines += [
                f"range_min: {valid_dists.min():.3f} m",
                f"range_max: {valid_dists.max():.3f} m",
                f"range_mean: {valid_dists.mean():.3f} m",
                f"range_std: {valid_dists.std():.3f} m",
            ]
            # Per-beam histogram (how many returns per vertical channel)
            n_h = max(n_total // _LIDAR_CHANNELS, 1)
            stats_lines.append(f"expected_rays_per_beam: {n_h}")
            for ch in range(_LIDAR_CHANNELS):
                ch_start = ch * n_h
                ch_end = min(ch_start + n_h, n_total)
                ch_valid = int(valid_mask[ch_start:ch_end].sum())
                stats_lines.append(f"  beam_{ch:02d}: {ch_valid}/{n_h} returns")
        else:
            stats_lines.append("WARNING: ZERO VALID RETURNS — DEAD SENSOR")

        stats_path = os.path.join(_DEBUG_SENSOR_DIR, "lidar_stats.txt")
        with open(stats_path, "w") as f:
            f.write("\n".join(stats_lines) + "\n")

        # Also save camera debug frames
        for key in ("rgb_front", "rgb_left", "rgb_right"):
            img_data = obs.get(key)
            if img_data is not None:
                frame = img_data[0].cpu().numpy()
                if frame.shape[-1] == 4:
                    frame = frame[:, :, :3]
                np.save(os.path.join(_DEBUG_SENSOR_DIR, f"{key}.npy"), frame)

        status = "OK" if n_valid > 0 else "DEAD"
        print(f"[INFO] LiDAR sanity dump → {_DEBUG_SENSOR_DIR}/  "
              f"({n_valid}/{n_total} valid returns — {status})", flush=True)

    # ------------------------------------------------------------------
    # Wall helpers
    # ------------------------------------------------------------------

    def _spawn_walls(self):
        W, T, H = _WAREHOUSE_HALF, _WALL_T, _WALL_H

        def _wall_cfg(size):
            return sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1e6),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.78, 0.74, 0.68), roughness=0.9),
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
    # Scattered warehouse boxes
    # ------------------------------------------------------------------

    def _spawn_scatter_boxes(self):
        """Place pallets with cargo boxes at random locations around the warehouse.

        Each scatter unit is identical in structure to the main pallet:
        a compound GMA pallet with a 3×3 grid of cardboard boxes on top.
        """
        import omni.usd

        rng = np.random.default_rng()  # unseeded — different layout every run
        stage = omni.usd.get_context().get_stage()

        # Generate random positions, keeping clear of the corridor between
        # the forklift (origin) and the target pallet (10m ahead along +X).
        CLEAR_RADIUS = 3.0          # clear zone around forklift start
        CORRIDOR_W   = 2.5          # half-width of the clear corridor (metres)
        CORRIDOR_END = 12.0         # corridor extends past the target pallet
        MARGIN       = 2.0          # stay away from walls
        lo = -_WAREHOUSE_HALF + MARGIN
        hi =  _WAREHOUSE_HALF - MARGIN

        def _in_corridor(x: float, y: float) -> bool:
            """True if (x,y) is inside the protected corridor from forklift to target."""
            return -CLEAR_RADIUS < x < CORRIDOR_END and abs(y) < CORRIDOR_W

        positions: list[tuple[float, float]] = []
        attempts = 0
        while len(positions) < _N_SCATTER_PALLETS and attempts < 5000:
            attempts += 1
            cx = float(rng.uniform(lo, hi))
            cy = float(rng.uniform(lo, hi))
            if math.hypot(cx, cy) < CLEAR_RADIUS:
                continue
            if _in_corridor(cx, cy):
                continue
            too_close = any(math.hypot(cx - ox, cy - oy) < 3.0
                           for ox, oy in positions)
            if too_close:
                continue
            positions.append((cx, cy))

        for pi, (px, py) in enumerate(positions):
            # Random yaw for each pallet
            yaw = float(rng.uniform(0, 2 * math.pi))
            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
            qw = math.cos(yaw / 2)
            qz = math.sin(yaw / 2)

            # ── Build compound pallet ─────────────────────────────────
            pallet_path = f"/World/ScatterPallet_{pi}"
            _build_compound_pallet(stage, pallet_path)

            # Position the pallet root via UsdGeom
            from pxr import UsdGeom, Gf
            pallet_prim = stage.GetPrimAtPath(pallet_path)
            xf = UsdGeom.XformCommonAPI(pallet_prim)
            xf.SetTranslate(Gf.Vec3d(px, py, _PALLET_H / 2))
            xf.SetRotate(Gf.Vec3f(0, 0, math.degrees(yaw)))

            # ── Place cargo boxes on top of this pallet ───────────────
            for bi, (lx, ly, lz) in enumerate(_BOX_LOCAL_OFFSETS):
                color = _SCATTER_COLORS[rng.integers(len(_SCATTER_COLORS))]

                # Rotate local offset by pallet yaw
                wx = px + lx * cos_y - ly * sin_y
                wy = py + lx * sin_y + ly * cos_y
                wz = lz  # z is already relative to ground

                cfg = sim_utils.CuboidCfg(
                    size=(_BOX_L, _BOX_W, _BOX_H),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=_BOX_MASS),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=color,
                        roughness=0.85,
                        metallic=0.0,
                    ),
                )
                cfg.func(
                    f"/World/ScatterBox_{pi}_{bi}",
                    cfg,
                    translation=(wx, wy, wz),
                    orientation=(qw, 0.0, 0.0, qz),
                )

        print(f"[INFO] Spawned {_N_SCATTER_PALLETS} pallets with "
              f"{_N_SCATTER_PALLETS * _N_BOXES} boxes around the warehouse", flush=True)

    # ------------------------------------------------------------------
    # Pallet placement helper
    # ------------------------------------------------------------------

    def _place_pallet_and_cargo(self, env_id: int, env_t: torch.Tensor,
                                pi: int, tx: float, ty: float,
                                base_z: float, yaw: float = 0.0):
        """Place interactable pallet *pi* and its cargo boxes at (tx, ty, base_z).

        base_z is the z-coordinate of the pallet's bottom surface:
          - 0.0 for ground level
          - _UNIT_H for stacked on one pallet+cargo layer
        """
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        qw = math.cos(yaw / 2)
        qz = math.sin(yaw / 2)

        pal_pose = torch.zeros(1, 7, device=self.device)
        pal_pose[0, 0] = tx
        pal_pose[0, 1] = ty
        pal_pose[0, 2] = base_z + _PALLET_H / 2
        pal_pose[0, 3] = qw
        pal_pose[0, 6] = qz
        self.pallets[pi].write_root_pose_to_sim(pal_pose, env_ids=env_t)
        self.pallets[pi].write_root_velocity_to_sim(
            torch.zeros(1, 6, device=self.device), env_ids=env_t)

        for bi, (lx, ly, lz) in enumerate(_BOX_LOCAL_OFFSETS):
            jx = float(self._rng.uniform(-0.01, 0.01))
            jy = float(self._rng.uniform(-0.01, 0.01))
            bx = tx + lx * cos_y - ly * sin_y + jx
            by = ty + lx * sin_y + ly * cos_y + jy
            bz = base_z + lz

            bp = torch.zeros(1, 7, device=self.device)
            bp[0, 0], bp[0, 1], bp[0, 2] = bx, by, bz
            bp[0, 3], bp[0, 6] = qw, qz
            self.pallet_boxes[pi][bi].write_root_pose_to_sim(bp, env_ids=env_t)
            self.pallet_boxes[pi][bi].write_root_velocity_to_sim(
                torch.zeros(1, 6, device=self.device), env_ids=env_t)

            self._grab_box_offsets[env_id][pi][bi] = (lx + jx, ly + jy, lz)

    # ------------------------------------------------------------------
    # Procedural layout (target placement)
    # ------------------------------------------------------------------

    def _generate_layout(self) -> tuple[list, tuple[float, float]]:
        N, C = _GRID_N, _CELL

        def g2w(r, c):
            return float((c - (N - 1) / 2.0) * C), float((r - (N - 1) / 2.0) * C)

        sr, sc = N // 2, N // 2
        CLEAR = {(sr + dr, sc + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}

        for _ in range(40):
            grid = np.zeros((N, N), dtype=np.int8)
            for _ in range(int(self._rng.integers(4, 8))):
                ch = int(self._rng.integers(1, 3))
                cw = int(self._rng.integers(2, 5))
                r0 = int(self._rng.integers(0, N - ch + 1))
                c0 = int(self._rng.integers(0, N - cw + 1))
                for r in range(r0, min(r0 + ch, N)):
                    for c in range(c0, min(c0 + cw, N)):
                        if (r, c) not in CLEAR:
                            grid[r, c] = 1

            visited: set = set()
            q: deque = deque([(sr, sc)])
            visited.add((sr, sc))
            while q:
                r, c = q.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < N and 0 <= nc < N and grid[nr, nc] == 0 \
                            and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc))

            candidates = [(r, c) for r, c in visited
                          if (r, c) not in CLEAR
                          and abs(r - sr) + abs(c - sc) >= 4]
            if not candidates:
                continue

            t_r, t_c = candidates[int(self._rng.integers(0, len(candidates)))]
            return [], g2w(t_r, t_c)

        return [], g2w(N - 1, N - 1)

    # ------------------------------------------------------------------
    # Episode reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        origins = self.scene.env_origins[env_ids]

        # Reset forklift
        default_root = self.forklift.data.default_root_state[env_ids].clone()
        default_root[:, :3] += origins
        self.forklift.write_root_pose_to_sim(default_root[:, :7], env_ids=env_ids)
        self.forklift.write_root_velocity_to_sim(default_root[:, 7:], env_ids=env_ids)
        dj_pos = self.forklift.data.default_joint_pos[env_ids].clone()
        dj_vel = self.forklift.data.default_joint_vel[env_ids].clone()
        self.forklift.write_joint_state_to_sim(dj_pos, dj_vel, env_ids=env_ids)

        # Clear grab state and reset authoritative position/heading/fork
        self._fork_pos[env_ids] = 0.0
        self._carry_pos[env_ids, 0] = default_root[:, 0]
        self._carry_pos[env_ids, 1] = default_root[:, 1]
        # Extract heading from reset quaternion (default is identity → heading=0)
        qw_r = default_root[:, 3]
        qz_r = default_root[:, 6]
        self._heading[env_ids] = 2.0 * torch.atan2(qz_r, qw_r)
        for env_id in env_ids.tolist():
            self._grabbed_idx[env_id] = -1
            self._grab_heading[env_id] = 0.0
            for pi in range(_N_INTERACTABLE):
                self._grab_box_offsets[env_id][pi] = list(_BOX_LOCAL_OFFSETS)
                self._pallet_base_z[env_id][pi] = 0.0

        # Per-env layout: place interactable pallets
        # Layout:
        #   Pallet 0 (blue): 10m ahead — primary pick target
        #   Pallet 1: 6m right, 8m ahead — on ground
        #   Pallet 2: 6m left, 8m ahead — already stacked on pallet 3
        #   Pallet 3: 6m left, 8m ahead — bottom of a stack (same XY as pallet 2)
        _POSITIONS = [
            (10.0,   0.0, 0.0),         # pallet 0: ground level
            ( 8.0,  -6.0, 0.0),         # pallet 1: ground level
            ( 8.0,   6.0, _UNIT_H),     # pallet 2: stacked on pallet 3
            ( 8.0,   6.0, 0.0),         # pallet 3: ground level (under pallet 2)
        ]

        for i, env_id in enumerate(env_ids.tolist()):
            origin = origins[i].cpu().numpy()
            ox, oy = float(origin[0]), float(origin[1])
            env_t = torch.tensor([env_id], device=self.device)

            for pi in range(_N_INTERACTABLE):
                dx, dy, base_z = _POSITIONS[pi]
                tx = ox + dx
                ty = oy + dy
                self._pallet_base_z[env_id][pi] = base_z

                self._place_pallet_and_cargo(
                    env_id, env_t, pi, tx, ty, base_z, yaw=0.0)

            print(f"[INFO] Env {env_id}: {_N_INTERACTABLE} pallets placed "
                  f"(pallet 2 stacked on pallet 3)", flush=True)

        # Snap driver camera immediately
        self._update_driver_cam()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        v_x      = self.actions[:, 0]
        omega_z  = self.actions[:, 1]
        fork_cmd = self.actions[:, 2]

        # ── Heading + root velocity ───────────────────────────────────
        # Use authoritative heading — never read from solver (collisions
        # with kinematic loads corrupt the solver's orientation).
        dt          = self.cfg.sim.dt
        self._heading += omega_z * dt
        new_heading = self._heading

        vel = torch.zeros(self.num_envs, 6, device=self.device)
        vel[:, 0] = v_x * torch.cos(new_heading)
        vel[:, 1] = v_x * torch.sin(new_heading)
        self.forklift.write_root_velocity_to_sim(vel)

        # ── Constrain to ground plane ─────────────────────────────────
        # Two modes:
        #  - FREE (not carrying): use solver position so PhysX collisions
        #    with cargo/pallets/walls work. The solver moves the forklift.
        #  - CARRY: integrate manually so kinematic cargo on the forks
        #    doesn't push the forklift via PhysX collision response.
        pose = self.forklift.data.root_state_w[:, :7].clone()
        grabbed_mask = torch.tensor(
            [g >= 0 for g in self._grabbed_idx],
            dtype=torch.bool, device=self.device)

        # Manual integration (carry mode)
        manual_x = self._carry_pos[:, 0] + vel[:, 0] * dt
        manual_y = self._carry_pos[:, 1] + vel[:, 1] * dt

        # Solver position (free mode — includes collision response)
        solver_x = pose[:, 0]
        solver_y = pose[:, 1]

        pose[:, 0] = torch.where(grabbed_mask, manual_x, solver_x)
        pose[:, 1] = torch.where(grabbed_mask, manual_y, solver_y)

        # Sync tracked position for next step and consistent reads
        self._carry_pos[:, 0] = pose[:, 0]
        self._carry_pos[:, 1] = pose[:, 1]

        ground_z = self.scene.env_origins[:, 2] + self.cfg.chassis_z_offset
        pose[:, 2] = ground_z
        half_yaw = new_heading / 2.0
        pose[:, 3] = torch.cos(half_yaw)
        pose[:, 4] = 0.0
        pose[:, 5] = 0.0
        pose[:, 6] = torch.sin(half_yaw)
        self.forklift.write_root_pose_to_sim(pose)

        # ── Cosmetic wheel spin + steering ───────────────────────────
        R = self.cfg.wheel_radius
        L = self.cfg.wheel_base
        omega_wheel = v_x / R
        jvt = self.forklift.data.joint_vel_target.clone()
        jvt[:, self._lf_wheel_idx] = omega_wheel
        jvt[:, self._rf_wheel_idx] = omega_wheel
        jvt[:, self._lb_wheel_idx] = omega_wheel
        jvt[:, self._rb_wheel_idx] = omega_wheel
        self.forklift.set_joint_velocity_target(jvt)

        speed = v_x.abs().clamp(min=0.05)
        steer_angle = torch.atan2(L * omega_z, speed).clamp(-0.785, 0.785)

        # Visual rear-wheel angle: steer_angle flips sign between forward
        # and backward because omega_z flips, but the physical wheel angle
        # should stay the same for a given steering input.  Multiply by
        # sign(v_x) so the visual is consistent regardless of direction.
        fwd_sign = torch.sign(v_x)
        fwd_sign[fwd_sign == 0] = 1.0          # default to forward when stopped
        visual_steer = -steer_angle * fwd_sign  # negate for rear-steer convention

        # ── Fork — direct state write ─────────────────────────────────
        # Use authoritative fork state (not physics solver, which gets
        # corrupted by collision with the carried kinematic pallet)
        fork_pos   = self._fork_pos.clone()
        fork_delta = fork_cmd * 0.04
        new_fork   = torch.clamp(fork_pos + fork_delta, -0.3, 1.5)
        self._fork_pos = new_fork

        all_jpos = self.forklift.data.joint_pos.clone()
        all_jvel = self.forklift.data.joint_vel.clone()
        all_jpos[:, self._fork_idx] = new_fork
        all_jvel[:, self._fork_idx] = 0.0
        # Always force-write steering and wheel state — collisions with
        # kinematic pallet/boxes corrupt the solver's joint values during
        # carry, and the corruption persists after dropping.
        all_jpos[:, self._l_steer_idx] = visual_steer
        all_jpos[:, self._r_steer_idx] = visual_steer
        all_jvel[:, self._l_steer_idx] = 0.0
        all_jvel[:, self._r_steer_idx] = 0.0
        all_jvel[:, self._lf_wheel_idx] = omega_wheel
        all_jvel[:, self._rf_wheel_idx] = omega_wheel
        all_jvel[:, self._lb_wheel_idx] = omega_wheel
        all_jvel[:, self._rb_wheel_idx] = omega_wheel
        self.forklift.write_joint_state_to_sim(all_jpos, all_jvel)

        jpt = self.forklift.data.joint_pos_target.clone()
        jpt[:, self._l_steer_idx] = visual_steer
        jpt[:, self._r_steer_idx] = visual_steer
        jpt[:, self._fork_idx]    = new_fork
        self.forklift.set_joint_position_target(jpt)

        # ── Pallet + box grab / carry ─────────────────────────────────
        # Use pre-raise fork position for grab detection so the raise
        # command doesn't push tine_z past the pocket threshold in one step.
        self._update_pallet_grab(fork_cmd, new_fork, fork_pos)

        # ── Driver camera ─────────────────────────────────────────────
        self._update_driver_cam()

    # ------------------------------------------------------------------
    # Pallet grab & carry
    # ------------------------------------------------------------------

    def _update_pallet_grab(self, fork_cmd: torch.Tensor, fork_joint: torch.Tensor,
                            fork_joint_prev: torch.Tensor):
        """Multi-pallet grab / carry / release with stacking support.

        Checks ALL interactable pallets and grabs the closest valid one.
        On release, detects if the pallet is above another and stacks it.
        """
        fl_heading = self._heading
        cos_h = torch.cos(fl_heading)
        sin_h = torch.sin(fl_heading)

        for i in range(self.num_envs):
            j       = fork_joint[i].item()
            j_prev  = fork_joint_prev[i].item()
            cmd     = fork_cmd[i].item()
            cos_i   = cos_h[i].item()
            sin_i   = sin_h[i].item()
            fl_x    = self._carry_pos[i, 0].item()
            fl_y    = self._carry_pos[i, 1].item()
            tine_z_prev = j_prev + 0.325
            env_t = torch.tensor([i], device=self.device)

            grabbed_pi = self._grabbed_idx[i]

            if grabbed_pi < 0:
                # ── Try to grab closest valid pallet ──────────────────
                if cmd <= 0.01:
                    continue
                forks_in = tine_z_prev < _PALLET_CL + 0.15
                if not forks_in:
                    continue

                best_pi, best_fwd, best_lat_raw = -1, 1e9, 0.0
                for pi in range(_N_INTERACTABLE):
                    pp = self.pallets[pi].data.root_pos_w
                    pl_x = pp[i, 0].item()
                    pl_y = pp[i, 1].item()
                    pl_z = pp[i, 2].item()

                    dx = pl_x - fl_x
                    dy = pl_y - fl_y
                    fwd = dx * cos_i + dy * sin_i
                    lat_raw = -dx * sin_i + dy * cos_i
                    lat = abs(lat_raw)

                    # Check if forks are at the right height for this pallet
                    pal_bottom = pl_z - _PALLET_H / 2
                    pocket_top = pal_bottom + _PALLET_BOT_H + _PALLET_STG_H
                    if tine_z_prev > pocket_top + 0.15:
                        continue

                    in_zone = -0.2 < fwd < (_PALLET_L + 1.0) \
                              and lat < (_PALLET_W / 2 + 0.3)
                    if in_zone and fwd < best_fwd:
                        best_pi = pi
                        best_fwd = fwd
                        best_lat_raw = lat_raw

                if best_pi >= 0:
                    self._grabbed_idx[i] = best_pi
                    self._grab_fwd[i] = best_fwd
                    self._grab_lat[i] = best_lat_raw
                    self._grab_heading[i] = fl_heading[i].item()
                    print(f"[GRAB] env={i} pallet={best_pi}  "
                          f"tine_z={tine_z_prev:.3f} m  "
                          f"fwd={best_fwd:.2f} m", flush=True)
            else:
                # ── Carry or release grabbed pallet ───────────────────
                tine_z = j + 0.325
                if j < -0.25:
                    # Release — check for stacking
                    drop_pos = self.pallets[grabbed_pi].data.root_pos_w[i]
                    drop_x = drop_pos[0].item()
                    drop_y = drop_pos[1].item()
                    drop_z = max(tine_z - _PALLET_CL, 0.0)

                    # Check if dropping onto another pallet
                    for other_pi in range(_N_INTERACTABLE):
                        if other_pi == grabbed_pi:
                            continue
                        op = self.pallets[other_pi].data.root_pos_w[i]
                        ox, oy = op[0].item(), op[1].item()
                        if abs(drop_x - ox) < _PALLET_L * 0.8 \
                                and abs(drop_y - oy) < _PALLET_W * 0.8:
                            # Stack: place on top of the other pallet's cargo
                            other_base = self._pallet_base_z[i][other_pi]
                            drop_z = other_base + _UNIT_H
                            break

                    self._pallet_base_z[i][grabbed_pi] = drop_z
                    self._place_pallet_and_cargo(
                        i, env_t, grabbed_pi, drop_x, drop_y, drop_z)
                    self._grabbed_idx[i] = -1
                    print(f"[DROP] env={i} pallet={grabbed_pi}  "
                          f"base_z={drop_z:.3f} m", flush=True)
                else:
                    # Carry — pallet follows forklift kinematically
                    fwd_i = self._grab_fwd[i]
                    lat_i = self._grab_lat[i]
                    new_px = fl_x + fwd_i * cos_i - lat_i * sin_i
                    new_py = fl_y + fwd_i * sin_i + lat_i * cos_i

                    pallet_bottom = max(tine_z - _PALLET_CL, 0.0)
                    new_pz = pallet_bottom + _PALLET_H / 2

                    delta_h = fl_heading[i].item() - self._grab_heading[i]
                    cos_d = math.cos(delta_h)
                    sin_d = math.sin(delta_h)
                    carried_yaw = delta_h
                    half_yaw = carried_yaw / 2.0
                    qw = math.cos(half_yaw)
                    qz_val = math.sin(half_yaw)

                    pal_pose = torch.zeros(1, 7, device=self.device)
                    pal_pose[0, :3] = torch.tensor([new_px, new_py, new_pz])
                    pal_pose[0, 3] = qw
                    pal_pose[0, 6] = qz_val
                    self.pallets[grabbed_pi].write_root_pose_to_sim(
                        pal_pose, env_ids=env_t)

                    # Carry boxes kinematically
                    for bi, (lx, ly, lz_local) in enumerate(
                            self._grab_box_offsets[i][grabbed_pi]):
                        bx = new_px + lx * cos_d - ly * sin_d
                        by = new_py + lx * sin_d + ly * cos_d
                        bz = pallet_bottom + lz_local

                        bp = torch.zeros(1, 7, device=self.device)
                        bp[0, :3] = torch.tensor([bx, by, bz])
                        bp[0, 3] = qw
                        bp[0, 6] = qz_val
                        self.pallet_boxes[grabbed_pi][bi].write_root_pose_to_sim(
                            bp, env_ids=env_t)
                        self.pallet_boxes[grabbed_pi][bi].write_root_velocity_to_sim(
                            torch.zeros(1, 6, device=self.device), env_ids=env_t)

    # ------------------------------------------------------------------
    # Driver camera
    # ------------------------------------------------------------------

    def _update_driver_cam(self):
        """Update all three cameras each substep.

        cam_front_cabin — inside cabin, forward-facing, pitched down so fork
                          tips are visible in the lower frame.  Primary VLA camera.
        cam_top_left    — roof left side, angled slightly down and forward-left.
        cam_top_right   — roof right side, angled slightly down and forward-right.
        """
        # Build authoritative position tensor
        ground_z = self.scene.env_origins[:, 2] + self.cfg.chassis_z_offset
        fl_pos = torch.stack([self._carry_pos[:, 0],
                              self._carry_pos[:, 1],
                              ground_z], dim=1)
        heading = self._heading

        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)

        def _yaw_pitch_quat(yaw: torch.Tensor, pitch: float):
            """Quaternion for yaw (tensor) + pitch (scalar) in ZYX order."""
            cP = math.cos(pitch / 2)
            sP = math.sin(pitch / 2)
            half = yaw / 2.0
            cH = torch.cos(half)
            sH = torch.sin(half)
            return torch.stack([cH * cP, sH * sP, -cH * sP, sH * cP], dim=1)

        # ── Front cabin camera ────────────────────────────────────────
        # Inside the cabin, forward-facing, pitched ~35° down so fork
        # tips are visible in the lower portion of the frame.
        CABIN_FWD   = 0.3     # behind the mast, inside cab
        CABIN_UP    = 1.8     # operator eye height
        CABIN_PITCH = -0.60   # ~34° down

        f_x = fl_pos[:, 0] + CABIN_FWD * cos_h
        f_y = fl_pos[:, 1] + CABIN_FWD * sin_h
        f_z = fl_pos[:, 2] + CABIN_UP
        self.cam_front.set_world_poses(
            torch.stack([f_x, f_y, f_z], dim=1),
            _yaw_pitch_quat(heading, CABIN_PITCH),
            convention="world",
        )

        # ── Top-left camera ──────────────────────────────────────────
        SIDE_FWD     = 0.2
        SIDE_LAT     = 0.6
        SIDE_UP      = 2.3
        SIDE_PITCH   = -0.35   # ~20° down
        SIDE_YAW_OFF = 0.4     # ~23° outward from forward

        tl_x = fl_pos[:, 0] + SIDE_FWD * cos_h - SIDE_LAT * sin_h
        tl_y = fl_pos[:, 1] + SIDE_FWD * sin_h + SIDE_LAT * cos_h
        tl_z = fl_pos[:, 2] + SIDE_UP
        self.cam_top_left.set_world_poses(
            torch.stack([tl_x, tl_y, tl_z], dim=1),
            _yaw_pitch_quat(heading + SIDE_YAW_OFF, SIDE_PITCH),
            convention="world",
        )

        # ── Top-right camera ─────────────────────────────────────────
        tr_x = fl_pos[:, 0] + SIDE_FWD * cos_h + SIDE_LAT * sin_h
        tr_y = fl_pos[:, 1] + SIDE_FWD * sin_h - SIDE_LAT * cos_h
        tr_z = fl_pos[:, 2] + SIDE_UP
        self.cam_top_right.set_world_poses(
            torch.stack([tr_x, tr_y, tr_z], dim=1),
            _yaw_pitch_quat(heading - SIDE_YAW_OFF, SIDE_PITCH),
            convention="world",
        )

    # ------------------------------------------------------------------
    # Observations, rewards, termination
    # ------------------------------------------------------------------

    def _get_observations(self):
        obs = {
            "rgb_front":  self.cam_front.data.output["rgb"],
            "rgb_left":   self.cam_top_left.data.output["rgb"],
            "rgb_right":  self.cam_top_right.data.output["rgb"],
            "lidar":      self.lidar.data.ray_hits_w,
            "pallet_pos": self.pallet.data.root_pos_w,
            "fork_pos":   self._fork_pos.unsqueeze(1),
            "grabbed":    torch.tensor(
                              [[float(self._grabbed_idx[i] >= 0)]
                               for i in range(self.num_envs)],
                              device=self.device),
            "state":      self._get_proprioception(),
        }
        # LiDAR sanity dump after first 10 steps
        if not self._debug_sensors_saved and self._step_count >= 10:
            self._save_lidar_sanity_dump(obs)
            self._debug_sensors_saved = True

        # Per-step status line
        self._print_step_status(obs)
        return obs

    def _get_proprioception(self) -> torch.Tensor:
        """Proprioception vector: [x, y, yaw, vx, vy, omega_z,
           fork_height, grabbed_idx] — shape (N, 8).
        grabbed_idx: -1 = nothing, 0..N_INTERACTABLE-1 = which pallet."""
        vx = self.actions[:, 0]
        omega = self.actions[:, 1]
        cos_h = torch.cos(self._heading)
        sin_h = torch.sin(self._heading)
        return torch.stack([
            self._carry_pos[:, 0],
            self._carry_pos[:, 1],
            self._heading,
            vx * cos_h,
            vx * sin_h,
            omega,
            self._fork_pos,
            torch.tensor([float(g) for g in self._grabbed_idx],
                         device=self.device),
        ], dim=1)

    def _get_rewards(self) -> torch.Tensor:
        forklift_xy = self._carry_pos                 # authoritative XY
        pallet_xy   = self.pallet.data.root_pos_w[:, :2]
        dist        = torch.norm(forklift_xy - pallet_xy, dim=1)
        return torch.exp(-dist)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        forklift_xy = self._carry_pos                 # authoritative XY
        pallet_xy   = self.pallet.data.root_pos_w[:, :2]
        dist        = torch.norm(forklift_xy - pallet_xy, dim=1)
        success     = dist < 0.6
        timeout     = self.episode_length_buf >= self.max_episode_length
        return success, timeout


# ---------------------------------------------------------------------------
# Standalone entry point
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
    print("[INFO] Warehouse loaded. Press Ctrl+C to exit.", flush=True)

    while simulation_app.is_running():
        env.step(torch.zeros(1, 3))

    env.close()
    simulation_app.close()
