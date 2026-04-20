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
        print(f"[INFO] ForkliftC joints: {all_joints}")

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
              f"fork={self._fork_idx}")

        self._rng = np.random.default_rng()

        # Authoritative fork position — never read back from physics solver
        # (collision with carried pallet corrupts the solver's joint state)
        self._fork_pos = torch.zeros(self.num_envs, device=self.device)

        # Tracked X,Y position — during carry the kinematic load pushes
        # the forklift via PhysX collision, so we integrate velocity
        # ourselves and ignore the solver's position.
        self._carry_pos = torch.zeros(self.num_envs, 2, device=self.device)

        # Per-env grab state
        self._pallet_grabbed:    list[bool]  = [False] * self.num_envs
        self._grab_fwd:          list[float] = [0.0]   * self.num_envs
        self._grab_lat:          list[float] = [0.0]   * self.num_envs
        self._grab_heading:      list[float] = [0.0]   * self.num_envs
        # Box offsets relative to pallet when grabbed (body-frame)
        self._grab_box_offsets: list[list[tuple[float, float, float]]] = \
            [list(_BOX_LOCAL_OFFSETS) for _ in range(self.num_envs)]

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

        # ── Compound pallet (pre-built in USD, then wrapped by RigidObject)
        # Build at env_0 BEFORE cloning — the cloner copies it to all envs.
        stage = omni.usd.get_context().get_stage()
        _build_compound_pallet(stage, "/World/envs/env_0/Pallet", color=_PALLET_COLOR_TARGET)

        self.pallet = RigidObject(RigidObjectCfg(
            prim_path="/World/envs/env_.*/Pallet",
            spawn=None,    # prim already created above
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, _PARK_Z),
            ),
        ))

        # ── Cargo boxes (individual physics rigid bodies) ─────────────
        self.boxes: list[RigidObject] = []
        for i in range(_N_BOXES):
            color = _BOX_COLOR_TARGET  # all target cargo boxes are blue
            box = RigidObject(RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/CargoBox_{i}",
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
                        diffuse_color=color,
                        roughness=0.85,
                        metallic=0.0,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, _PARK_Z)),
            ))
            self.boxes.append(box)

        # ── Driver cameras (world-level prims, poses updated every step) ─
        _cam_cfg = dict(
            update_period=1 / 30,
            height=480,
            width=640,
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
        self.camera = Camera(CameraCfg(prim_path="/World/envs/env_.*/DriverCam", **_cam_cfg))
        self.camera_left = Camera(CameraCfg(prim_path="/World/envs/env_.*/DriverCamLeft", **_cam_cfg))
        self.camera_right = Camera(CameraCfg(prim_path="/World/envs/env_.*/DriverCamRight", **_cam_cfg))

        # ── Register with scene ───────────────────────────────────────
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["forklift"] = self.forklift
        self.scene.rigid_objects["pallet"]   = self.pallet
        for i, box in enumerate(self.boxes):
            self.scene.rigid_objects[f"box_{i}"] = box
        self.scene.sensors["camera"] = self.camera
        self.scene.sensors["camera_left"] = self.camera_left
        self.scene.sensors["camera_right"] = self.camera_right

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
        import math
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
              f"{_N_SCATTER_PALLETS * _N_BOXES} boxes around the warehouse")

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

        # Clear grab state and reset authoritative fork position
        self._fork_pos[env_ids] = 0.0
        for env_id in env_ids.tolist():
            self._pallet_grabbed[env_id] = False
            self._grab_heading[env_id] = 0.0
            self._grab_box_offsets[env_id] = list(_BOX_LOCAL_OFFSETS)

        # Per-env layout + object placement
        PALLET_FRONT_DIST = 10.0  # metres in front of forklift

        for i, env_id in enumerate(env_ids.tolist()):
            origin = origins[i].cpu().numpy()
            ox, oy = float(origin[0]), float(origin[1])

            env_t = torch.tensor([env_id], device=self.device)

            # ── Place pallet directly in front of forklift ────────────
            tx = ox + PALLET_FRONT_DIST
            ty = oy
            pal_pose = torch.zeros(1, 7, device=self.device)
            pal_pose[0, 0] = tx
            pal_pose[0, 1] = ty
            pal_pose[0, 2] = _PALLET_H / 2     # pallet centre height (half its height)
            pal_pose[0, 6] = 1.0                # w=1 (no rotation)
            self.pallet.write_root_pose_to_sim(pal_pose, env_ids=env_t)

            # ── Place cargo boxes on pallet ────────────────────────────
            for bi, (lx, ly, lz) in enumerate(_BOX_LOCAL_OFFSETS):
                # Small random nudge for realism (±1 cm)
                jitter_x = float(self._rng.uniform(-0.01, 0.01))
                jitter_y = float(self._rng.uniform(-0.01, 0.01))
                bx = tx + lx + jitter_x
                by = ty + ly + jitter_y
                bz = lz   # z is relative to pallet bottom which is at z=0

                bp = torch.zeros(1, 7, device=self.device)
                bp[0, 0], bp[0, 1], bp[0, 2], bp[0, 6] = bx, by, bz, 1.0
                self.boxes[bi].write_root_pose_to_sim(bp, env_ids=env_t)
                self.boxes[bi].write_root_velocity_to_sim(
                    torch.zeros(1, 6, device=self.device), env_ids=env_t)

                # Record grab offsets (with jitter baked in)
                self._grab_box_offsets[env_id][bi] = (lx + jitter_x, ly + jitter_y, lz)

            print(f"[INFO] Env {env_id}: pallet at ({PALLET_FRONT_DIST:.1f}m ahead)")

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
        heading     = self.forklift.data.heading_w
        dt          = self.cfg.sim.dt
        new_heading = heading + omega_z * dt

        vel = torch.zeros(self.num_envs, 6, device=self.device)
        vel[:, 0] = v_x * torch.cos(new_heading)
        vel[:, 1] = v_x * torch.sin(new_heading)
        self.forklift.write_root_velocity_to_sim(vel)

        # ── Constrain to ground plane ─────────────────────────────────
        pose = self.forklift.data.root_state_w[:, :7].clone()

        # During carry the kinematic load's collision shapes push the
        # forklift via PhysX.  Ignore the solver's X,Y and integrate
        # velocity ourselves so the forklift only moves when commanded.
        grabbed_mask = torch.tensor(self._pallet_grabbed, device=self.device)
        if grabbed_mask.any():
            new_x = self._carry_pos[:, 0] + vel[:, 0] * dt
            new_y = self._carry_pos[:, 1] + vel[:, 1] * dt
            pose[:, 0] = torch.where(grabbed_mask, new_x, pose[:, 0])
            pose[:, 1] = torch.where(grabbed_mask, new_y, pose[:, 1])
        # Sync tracked position (solver value when free, tracked when carrying)
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
        # Force steering state ONLY during carry — collision with the
        # kinematic pallet/boxes corrupts the solver's steering values.
        # When not carrying, the actuator drives steering normally.
        for i in range(self.num_envs):
            if self._pallet_grabbed[i]:
                all_jpos[i, self._l_steer_idx] = visual_steer[i].item()
                all_jpos[i, self._r_steer_idx] = visual_steer[i].item()
                all_jvel[i, self._l_steer_idx] = 0.0
                all_jvel[i, self._r_steer_idx] = 0.0
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
        """Detect fork insertion into the pallet pocket and carry the load.

        Grab conditions (all must be satisfied):
          - Pallet is 0.3–(PALLET_L + 0.5) m forward of the forklift centre
          - Pallet is within ±(PALLET_W/2 + 0.15) m laterally
          - fork_cmd > 0  (operator pressing raise)
          - pre-raise tine_z < PALLET_CL + 0.15  (forks were inside the fork pocket)

        While grabbed:
          - Pallet pose follows the forklift kinematically
          - All boxes follow the pallet (kinematic carry)

        Release:
          - fork_joint drops below -0.25 (forks near floor)
        """
        fl_pos     = self.forklift.data.root_pos_w    # (N, 3)
        fl_heading = self.forklift.data.heading_w     # (N,)
        pal_pos    = self.pallet.data.root_pos_w      # (N, 3)

        cos_h = torch.cos(fl_heading)
        sin_h = torch.sin(fl_heading)

        for i in range(self.num_envs):
            j       = fork_joint[i].item()
            j_prev  = fork_joint_prev[i].item()
            cmd     = fork_cmd[i].item()
            cos_i   = cos_h[i].item()
            sin_i   = sin_h[i].item()
            fl_x    = fl_pos[i, 0].item()
            fl_y    = fl_pos[i, 1].item()
            pl_x    = pal_pos[i, 0].item()
            pl_y    = pal_pos[i, 1].item()

            # Pallet centre in forklift body frame
            dx  = pl_x - fl_x
            dy  = pl_y - fl_y
            fwd = dx * cos_i + dy * sin_i
            lat = abs(-dx * sin_i + dy * cos_i)

            # Use PRE-RAISE tine height for grab detection so pressing
            # raise doesn't jump past the pocket threshold in one step
            tine_z_prev = j_prev + 0.325

            env_t = torch.tensor([i], device=self.device)

            if not self._pallet_grabbed[i]:
                # ── Try to grab ───────────────────────────────────────
                in_zone  = -0.2 < fwd < (_PALLET_L + 1.0) \
                           and lat < (_PALLET_W / 2 + 0.3)
                forks_in = tine_z_prev < _PALLET_CL + 0.15   # forks were inside pocket
                if in_zone and forks_in and cmd > 0.01:
                    self._pallet_grabbed[i] = True
                    self._grab_fwd[i] = fwd
                    self._grab_lat[i] = -dx * sin_i + dy * cos_i
                    self._grab_heading[i] = fl_heading[i].item()
                    print(f"[GRAB] env={i}  tine_z={tine_z_prev:.3f} m  "
                          f"fwd={fwd:.2f} m  lat={lat:.2f} m")
            else:
                # ── Release ───────────────────────────────────────────
                tine_z = j + 0.325   # current tine height for carry positioning
                if j < -0.25:
                    self._pallet_grabbed[i] = False
                    print(f"[DROP] env={i}  tine_z={tine_z:.3f} m")
                else:
                    # ── Carry — pallet follows forklift ───────────────
                    fwd_i = self._grab_fwd[i]
                    lat_i = self._grab_lat[i]
                    new_px = fl_x + fwd_i * cos_i - lat_i * sin_i
                    new_py = fl_y + fwd_i * sin_i + lat_i * cos_i

                    # Pallet Z: tine_z lifts the bottom of the stringer;
                    # pallet bottom = tine_z - PALLET_CL (clamped to ≥ 0)
                    pallet_bottom = max(tine_z - _PALLET_CL, 0.0)
                    new_pz = pallet_bottom + _PALLET_H / 2   # pallet centre z

                    # Heading delta since grab — rotate pallet & boxes
                    # by exactly how much the forklift has turned
                    import math
                    delta_h = fl_heading[i].item() - self._grab_heading[i]
                    cos_d = math.cos(delta_h)
                    sin_d = math.sin(delta_h)

                    # Pallet orientation: preserve original reset yaw
                    # (qz=1 at reset) and add the heading delta on top.
                    # Original yaw = π, so carried yaw = π + delta_h.
                    carried_yaw = math.pi + delta_h
                    half_yaw = carried_yaw / 2.0
                    qw = math.cos(half_yaw)
                    qz_val = math.sin(half_yaw)

                    pal_pose = torch.zeros(1, 7, device=self.device)
                    pal_pose[0, 0] = new_px
                    pal_pose[0, 1] = new_py
                    pal_pose[0, 2] = new_pz
                    pal_pose[0, 3] = qw
                    pal_pose[0, 6] = qz_val
                    self.pallet.write_root_pose_to_sim(pal_pose, env_ids=env_t)

                    # ── Carry boxes kinematically with pallet ─────────
                    # Rotate local offsets by heading delta so boxes
                    # stay fixed on the forks during turns
                    for bi, (lx, ly, lz_local) in enumerate(
                            self._grab_box_offsets[i]):
                        bx = new_px + lx * cos_d - ly * sin_d
                        by = new_py + lx * sin_d + ly * cos_d
                        bz = pallet_bottom + lz_local

                        bp = torch.zeros(1, 7, device=self.device)
                        bp[0, 0] = bx
                        bp[0, 1] = by
                        bp[0, 2] = bz
                        bp[0, 3] = qw
                        bp[0, 6] = qz_val
                        self.boxes[bi].write_root_pose_to_sim(bp, env_ids=env_t)
                        self.boxes[bi].write_root_velocity_to_sim(
                            torch.zeros(1, 6, device=self.device), env_ids=env_t)

    # ------------------------------------------------------------------
    # Driver camera
    # ------------------------------------------------------------------

    def _update_driver_cam(self):
        """Update all three roof-mounted cameras: center, left-side, right-side.

        Center camera: front-center of the roof, looking forward and down.
        Left camera:   left side of the roof, looking 90° left.
        Right camera:  right side of the roof, looking 90° right.
        """
        import math

        fl_pos  = self.forklift.data.root_pos_w   # (N, 3)
        heading = self.forklift.data.heading_w     # (N,)

        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)

        # ── Center camera (forward-facing) ────────────────────────────
        CAM_FWD  = 0.5
        CAM_UP   = 2.2
        PITCH    = -0.55  # ~31° down

        ctr_x = fl_pos[:, 0] + CAM_FWD * cos_h
        ctr_y = fl_pos[:, 1] + CAM_FWD * sin_h
        ctr_z = fl_pos[:, 2] + CAM_UP
        ctr_pos = torch.stack([ctr_x, ctr_y, ctr_z], dim=1)

        cP = math.cos(PITCH / 2)
        sP = math.sin(PITCH / 2)
        half_yaw = heading / 2.0
        cH = torch.cos(half_yaw)
        sH = torch.sin(half_yaw)

        ctr_ori = torch.stack([
             cH * cP,
             sH * sP,
            -cH * sP,
             sH * cP,
        ], dim=1)

        self.camera.set_world_poses(ctr_pos, ctr_ori, convention="world")

        # ── Side cameras (left / right) ───────────────────────────────
        SIDE_LATERAL = 0.6   # metres to the side from centre
        SIDE_UP      = 2.2   # same roof height
        SIDE_PITCH   = -0.30 # ~17° down

        cPs = math.cos(SIDE_PITCH / 2)
        sPs = math.sin(SIDE_PITCH / 2)

        # Left camera — yaw + 90° (pi/2)
        left_yaw = heading + math.pi / 2
        half_ly = left_yaw / 2.0
        cHL = torch.cos(half_ly)
        sHL = torch.sin(half_ly)

        # Position: offset to the left (perpendicular to heading)
        left_x = fl_pos[:, 0] - SIDE_LATERAL * sin_h
        left_y = fl_pos[:, 1] + SIDE_LATERAL * cos_h
        left_z = fl_pos[:, 2] + SIDE_UP
        left_pos = torch.stack([left_x, left_y, left_z], dim=1)

        left_ori = torch.stack([
             cHL * cPs,
             sHL * sPs,
            -cHL * sPs,
             sHL * cPs,
        ], dim=1)

        self.camera_left.set_world_poses(left_pos, left_ori, convention="world")

        # Right camera — yaw - 90° (-pi/2)
        right_yaw = heading - math.pi / 2
        half_ry = right_yaw / 2.0
        cHR = torch.cos(half_ry)
        sHR = torch.sin(half_ry)

        right_x = fl_pos[:, 0] + SIDE_LATERAL * sin_h
        right_y = fl_pos[:, 1] - SIDE_LATERAL * cos_h
        right_z = fl_pos[:, 2] + SIDE_UP
        right_pos = torch.stack([right_x, right_y, right_z], dim=1)

        right_ori = torch.stack([
             cHR * cPs,
             sHR * sPs,
            -cHR * sPs,
             sHR * cPs,
        ], dim=1)

        self.camera_right.set_world_poses(right_pos, right_ori, convention="world")

    # ------------------------------------------------------------------
    # Observations, rewards, termination
    # ------------------------------------------------------------------

    def _get_observations(self):
        return {
            "rgb":        self.camera.data.output["rgb"],
            "rgb_left":   self.camera_left.data.output["rgb"],
            "rgb_right":  self.camera_right.data.output["rgb"],
            "pallet_pos": self.pallet.data.root_pos_w,
            "fork_pos":   self._fork_pos.unsqueeze(1),
            "grabbed":    torch.tensor(
                              [[float(self._pallet_grabbed[i])]
                               for i in range(self.num_envs)],
                              device=self.device),
        }

    def _get_rewards(self) -> torch.Tensor:
        forklift_xy = self.forklift.data.root_pos_w[:, :2]
        pallet_xy   = self.pallet.data.root_pos_w[:, :2]
        dist        = torch.norm(forklift_xy - pallet_xy, dim=1)
        return torch.exp(-dist)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        forklift_xy = self.forklift.data.root_pos_w[:, :2]
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
    print("[INFO] Warehouse loaded. Press Ctrl+C to exit.")

    while simulation_app.is_running():
        env.step(torch.zeros(1, 3))

    env.close()
    simulation_app.close()
