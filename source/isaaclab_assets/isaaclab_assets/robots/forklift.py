# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Isaac Sim ForkliftC asset.

Uses the NVIDIA ForkliftC USD from the Nucleus asset server:
    {ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/ForkliftC/forklift_c.usd

ForkliftC is a rear-wheel-steered, front-wheel-driven forklift (Ackermann kinematics).

Joint names:
    Drive wheels (velocity-controlled):
        left_front_wheel_joint, right_front_wheel_joint
        left_back_wheel_joint,  right_back_wheel_joint
    Steering (position-controlled, rear axle):
        left_rotator_joint, right_rotator_joint
    Fork lift (prismatic, position-controlled):
        lift_joint

Kinematics:
    wheel_base:          1.65 m
    front_wheel_radius:  0.325 m
    back_wheel_radius:   0.255 m

Available configurations:
    * FORKLIFT_CFG: ForkliftC with Ackermann drive wheels, steering, and fork lift joint.
"""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# ArticulationCfg
##

FORKLIFT_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Forklift",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/ForkliftC/forklift_c.usd",
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
        # ForkliftC USD origin is at ground level (NVIDIA convention)
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "left_front_wheel_joint":  0.0,
            "right_front_wheel_joint": 0.0,
            "left_back_wheel_joint":   0.0,
            "right_back_wheel_joint":  0.0,
            "left_rotator_joint":      0.0,
            "right_rotator_joint":     0.0,
            "lift_joint":         0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Front drive wheels — velocity-controlled (high damping resists idle gravity spin)
        "front_drive_wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_front_wheel_joint", "right_front_wheel_joint"],
            effort_limit=500.0,
            velocity_limit=10.0,
            stiffness=0.0,
            damping=2000.0,
        ),
        # Rear wheels — driven but also carry steering torque; same high damping
        "rear_drive_wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_back_wheel_joint", "right_back_wheel_joint"],
            effort_limit=500.0,
            velocity_limit=10.0,
            stiffness=0.0,
            damping=2000.0,
        ),
        # Rear steering rotators — position-controlled
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["left_rotator_joint", "right_rotator_joint"],
            effort_limit=200.0,
            velocity_limit=2.0,
            stiffness=800.0,
            damping=80.0,
        ),
        # Fork lift — position-controlled.
        # _apply_action teleports via write_joint_state_to_sim and syncs the
        # position target so net spring force is always ~0 at runtime.
        "fork_lift": ImplicitActuatorCfg(
            joint_names_expr=["lift_joint"],
            effort_limit=5000.0,
            velocity_limit=2.0,
            stiffness=150.0,
            damping=100.0,
        ),
    },
)
"""ArticulationCfg for the NVIDIA Isaac Sim ForkliftC (Nucleus asset)."""
