# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the articulated forklift.

The base_link visual now uses the 48-forklift 3-D mesh (assets/48-forklift/Forklift.obj).
Reconvert the URDF to USD any time the URDF or the mesh changes:

    ./isaaclab.sh -p scripts/tools/convert_urdf.py \
        assets/forklift/forklift.urdf \
        assets/forklift/forklift.usd

Note: do NOT use --fix-base (causes articulation crash).

Joint names (from URDF):
    Drive wheels (continuous → velocity-controlled):
        rear_left_wheel_joint, rear_right_wheel_joint
    Caster wheels (continuous → implicit, no actuator needed):
        front_left_caster_joint, front_right_caster_joint
    Fork lift (prismatic → position/effort-controlled):
        fork_lift_joint  (lower=0.0, upper=1.5 m)

Available configurations:
    * FORKLIFT_CFG: Articulated forklift with drive wheels and fork lift joint.
"""

import os

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils

##
# Path to the converted USD
##

_FORKLIFT_USD = os.path.join(
    os.path.dirname(__file__),
    "..",   # robots/ -> isaaclab_assets/
    "..",   # isaaclab_assets/ -> isaaclab_assets pkg root
    "..",   # isaaclab_assets pkg -> source/
    "..",   # source/ -> project root (IsaacLab/)
    "assets",
    "forklift",
    "forklift.usd",
)
FORKLIFT_USD_PATH = os.path.normpath(_FORKLIFT_USD)

##
# ArticulationCfg
##

FORKLIFT_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Forklift",
    spawn=sim_utils.UsdFileCfg(
        usd_path=FORKLIFT_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            linear_damping=5.0,
            angular_damping=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),   # wheel joints at z=-0.3 from chassis → wheel centre=0.3, bottom=0.0
        joint_pos={
            "rear_left_wheel_joint":   0.0,
            "rear_right_wheel_joint":  0.0,
            "front_left_caster_joint": 0.0,
            "front_right_caster_joint": 0.0,
            "fork_lift_joint":         0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Drive wheels — velocity-controlled implicit actuator
        "drive_wheels": ImplicitActuatorCfg(
            joint_names_expr=["rear_left_wheel_joint", "rear_right_wheel_joint"],
            effort_limit=500.0,
            velocity_limit=10.0,
            stiffness=0.0,    # pure velocity mode
            damping=500.0,
        ),
        # Caster wheels — free-rolling, high damping so they don't flop
        "caster_wheels": ImplicitActuatorCfg(
            joint_names_expr=["front_left_caster_joint", "front_right_caster_joint"],
            effort_limit=10.0,
            velocity_limit=10.0,
            stiffness=0.0,
            damping=50.0,
        ),
        # Fork lift — position-controlled with low stiffness.
        # _apply_action teleports via write_joint_state_to_sim and syncs the
        # position target so net spring force is always ~0 at runtime.
        "fork_lift": ImplicitActuatorCfg(
            joint_names_expr=["fork_lift_joint"],
            effort_limit=5000.0,
            velocity_limit=2.0,
            stiffness=150.0,
            damping=100.0,
        ),
    },
)
"""ArticulationCfg for the forklift with driven rear wheels and prismatic fork lift."""
