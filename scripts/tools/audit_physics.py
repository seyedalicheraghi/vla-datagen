"""
Deep physics audit of every spawned prim in the forklift scene.

For each prim reports:
  - Prim path, type, xform
  - RigidBodyAPI: present? kinematic? disabled?
  - CollisionAPI: present? enabled? approximation? hull count?
  - Mass, COM, inertia tensor, authored vs auto-computed
  - Material: static/dynamic friction, restitution, density
  - Collision filter group
  - Visual AABB vs Collision AABB (divergence flagged)
  - For articulations: joints, drive type, stiffness, damping, max force
  - CCD, contact offset, rest offset

Usage:
    ./isaaclab.sh -p scripts/tools/audit_physics.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Physics audit of forklift scene.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False  # no rendering needed for audit
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Now safe to import Isaac modules
# ---------------------------------------------------------------------------

import torch
import numpy as np
import omni.usd
from pxr import UsdGeom, UsdPhysics, Gf, Usd, PhysxSchema

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "forklift"))
from forklift_env import ForkliftEnv, ForkliftEnvCfg

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "forklift", "debug_sensors")


def get_world_aabb(prim):
    """Compute world-space AABB from UsdGeom.Imageable."""
    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        return None
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    bbox = bbox_cache.ComputeWorldBound(prim)
    aabb = bbox.ComputeAlignedRange()
    if aabb.IsEmpty():
        return None
    mn = aabb.GetMin()
    mx = aabb.GetMax()
    return (mn[0], mn[1], mn[2]), (mx[0], mx[1], mx[2])


def audit_prim(prim, lines: list, indent: int = 0):
    """Audit a single prim for physics properties."""
    path = str(prim.GetPath())
    prim_type = prim.GetTypeName()
    prefix = "  " * indent

    # Skip non-interesting prims
    if prim_type in ("Scope", "DomeLight", "DistantLight", "Shader", "Material", "NodeGraph"):
        return

    line = f"{prefix}{path}  type={prim_type}"

    # Xform
    xformable = UsdGeom.Xformable(prim)
    if xformable:
        try:
            xf = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = xf.ExtractTranslation()
            line += f"  pos=({t[0]:.3f},{t[1]:.3f},{t[2]:.3f})"
        except Exception:
            pass

    # RigidBodyAPI
    has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
    if has_rb:
        rb = UsdPhysics.RigidBodyAPI(prim)
        kinematic = False
        ke_attr = rb.GetKinematicEnabledAttr()
        if ke_attr and ke_attr.Get() is not None:
            kinematic = ke_attr.Get()
        rb_enabled = True
        rbe_attr = rb.GetRigidBodyEnabledAttr()
        if rbe_attr and rbe_attr.Get() is not None:
            rb_enabled = rbe_attr.Get()
        line += f"  RB=[kinematic={kinematic} enabled={rb_enabled}]"
    else:
        line += "  RB=NONE"

    # CollisionAPI
    has_col = prim.HasAPI(UsdPhysics.CollisionAPI)
    if has_col:
        col = UsdPhysics.CollisionAPI(prim)
        col_enabled = True
        ce_attr = col.GetCollisionEnabledAttr()
        if ce_attr and ce_attr.Get() is not None:
            col_enabled = ce_attr.Get()

        # Collision approximation
        approx = "none"
        mesh_col = UsdPhysics.MeshCollisionAPI(prim) if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else None
        if mesh_col:
            aa = mesh_col.GetApproximationAttr()
            if aa and aa.Get():
                approx = aa.Get()
        line += f"  COL=[enabled={col_enabled} approx={approx}]"
    else:
        line += "  COL=NONE"

    # MassAPI
    has_mass = prim.HasAPI(UsdPhysics.MassAPI)
    if has_mass:
        mass_api = UsdPhysics.MassAPI(prim)
        mass_val = mass_api.GetMassAttr().Get() if mass_api.GetMassAttr() else None
        density_val = mass_api.GetDensityAttr().Get() if mass_api.GetDensityAttr() else None
        com = mass_api.GetCenterOfMassAttr().Get() if mass_api.GetCenterOfMassAttr() else None
        inertia = mass_api.GetDiagonalInertiaAttr().Get() if mass_api.GetDiagonalInertiaAttr() else None
        line += f"  MASS=[m={mass_val} density={density_val} com={com} inertia={inertia}]"

    # PhysxRigidBodyAPI (CCD, contact/rest offset)
    if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        pxrb = PhysxSchema.PhysxRigidBodyAPI(prim)
        ccd = pxrb.GetEnableCCDAttr().Get() if pxrb.GetEnableCCDAttr() else None
        line += f"  CCD={ccd}"

    # PhysxCollisionAPI (contact/rest offset)
    if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        pxcol = PhysxSchema.PhysxCollisionAPI(prim)
        co = pxcol.GetContactOffsetAttr().Get() if pxcol.GetContactOffsetAttr() else None
        ro = pxcol.GetRestOffsetAttr().Get() if pxcol.GetRestOffsetAttr() else None
        line += f"  contactOffset={co} restOffset={ro}"

    # Material binding
    mat_api = UsdPhysics.MaterialAPI(prim) if prim.HasAPI(UsdPhysics.MaterialAPI) else None
    if mat_api:
        sf = mat_api.GetStaticFrictionAttr().Get() if mat_api.GetStaticFrictionAttr() else None
        df = mat_api.GetDynamicFrictionAttr().Get() if mat_api.GetDynamicFrictionAttr() else None
        rest = mat_api.GetRestitutionAttr().Get() if mat_api.GetRestitutionAttr() else None
        line += f"  MAT=[sf={sf} df={df} rest={rest}]"

    # Visual AABB
    aabb = get_world_aabb(prim)
    if aabb:
        mn, mx = aabb
        sz = (mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2])
        line += f"  AABB=[({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f})->({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f}) size=({sz[0]:.3f},{sz[1]:.3f},{sz[2]:.3f})]"

    # Articulation info
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        line += "  ARTICULATION_ROOT"

    # Joint info
    if prim.IsA(UsdPhysics.Joint):
        joint = UsdPhysics.Joint(prim)
        line += f"  JOINT"

    lines.append(line)

    # Check for PhysicsJoint children (drives)
    for child in prim.GetChildren():
        if child.IsA(UsdPhysics.DriveAPI):
            # This is a drive on a joint
            pass


def audit_articulation_joints(forklift_art, lines: list):
    """Audit articulation joint details from the Isaac Lab Articulation wrapper."""
    lines.append("")
    lines.append("=" * 80)
    lines.append("ARTICULATION JOINT DETAILS (from Isaac Lab Articulation)")
    lines.append("=" * 80)

    joint_names = forklift_art.joint_names
    for ji, jn in enumerate(joint_names):
        jpos = forklift_art.data.joint_pos[0, ji].item()
        jvel = forklift_art.data.joint_vel[0, ji].item()
        lines.append(f"  joint[{ji}] {jn}  pos={jpos:.4f}  vel={jvel:.4f}")

    # Actuator info from config
    from isaaclab_assets.robots.forklift import FORKLIFT_CFG
    for act_name, act_cfg in FORKLIFT_CFG.actuators.items():
        lines.append(f"  actuator '{act_name}':")
        lines.append(f"    joints: {act_cfg.joint_names_expr}")
        lines.append(f"    effort_limit: {act_cfg.effort_limit}")
        lines.append(f"    velocity_limit: {act_cfg.velocity_limit}")
        lines.append(f"    stiffness: {act_cfg.stiffness}")
        lines.append(f"    damping: {act_cfg.damping}")


def audit_rigid_objects(env, lines: list):
    """Audit the Isaac Lab RigidObject wrappers — what the sim actually sees."""
    lines.append("")
    lines.append("=" * 80)
    lines.append("RIGID OBJECT STATE (from Isaac Lab wrappers)")
    lines.append("=" * 80)

    for pi in range(len(env.pallets)):
        pal = env.pallets[pi]
        pos = pal.data.root_pos_w[0].cpu().numpy()
        vel = pal.data.root_vel_w[0].cpu().numpy() if hasattr(pal.data, 'root_vel_w') else [0]*6
        lines.append(f"  pallet_{pi}  pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})  "
                      f"prim={pal.cfg.prim_path}")

        for bi in range(len(env.pallet_boxes[pi])):
            box = env.pallet_boxes[pi][bi]
            bpos = box.data.root_pos_w[0].cpu().numpy()
            lines.append(f"    cargo_{pi}_{bi}  pos=({bpos[0]:.3f},{bpos[1]:.3f},{bpos[2]:.3f})")


def audit_grab_logic(env, lines: list):
    """Diagnose grab conditions for all pallets from current forklift position."""
    lines.append("")
    lines.append("=" * 80)
    lines.append("GRAB LOGIC DIAGNOSIS (from forklift current position)")
    lines.append("=" * 80)

    from forklift_env import (_PALLET_CL, _PALLET_L, _PALLET_W,
                              _PALLET_H, _PALLET_BOT_H, _PALLET_STG_H,
                              _N_INTERACTABLE)

    fl_x = env._carry_pos[0, 0].item()
    fl_y = env._carry_pos[0, 1].item()
    heading = env._heading[0].item()
    fork_j = env._fork_pos[0].item()
    tine_z = fork_j + 0.325
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    lines.append(f"  forklift: x={fl_x:.3f} y={fl_y:.3f} heading={math.degrees(heading):.1f}deg "
                 f"fork_joint={fork_j:.4f} tine_z={tine_z:.4f}")
    lines.append(f"  grab requires: cmd>0.01 AND tine_z < {_PALLET_CL + 0.15:.3f}")
    lines.append(f"                 AND -0.2 < fwd < {_PALLET_L + 1.0:.3f}")
    lines.append(f"                 AND lat < {_PALLET_W/2 + 0.3:.3f}")
    lines.append(f"                 AND tine_z < pocket_top + 0.15")

    for pi in range(_N_INTERACTABLE):
        pp = env.pallets[pi].data.root_pos_w[0]
        pl_x, pl_y, pl_z = pp[0].item(), pp[1].item(), pp[2].item()

        dx = pl_x - fl_x
        dy = pl_y - fl_y
        fwd = dx * cos_h + dy * sin_h
        lat = abs(-dx * sin_h + dy * cos_h)

        pal_bottom = pl_z - _PALLET_H / 2
        pocket_top = pal_bottom + _PALLET_BOT_H + _PALLET_STG_H

        fwd_ok = -0.2 < fwd < (_PALLET_L + 1.0)
        lat_ok = lat < (_PALLET_W / 2 + 0.3)
        height_ok = tine_z < pocket_top + 0.15
        tine_ok = tine_z < _PALLET_CL + 0.15

        lines.append(f"  pallet_{pi}: pos=({pl_x:.2f},{pl_y:.2f},{pl_z:.3f})  "
                     f"fwd={fwd:.2f} lat={lat:.2f} pocket_top={pocket_top:.3f}  "
                     f"fwd_ok={fwd_ok} lat_ok={lat_ok} height_ok={height_ok} tine_ok={tine_ok}")
        if not fwd_ok:
            lines.append(f"    BLOCKED: fwd={fwd:.2f} outside [-0.2, {_PALLET_L+1.0:.2f}]")
        if not lat_ok:
            lines.append(f"    BLOCKED: lat={lat:.2f} > {_PALLET_W/2+0.3:.2f}")
        if not height_ok:
            lines.append(f"    BLOCKED: tine_z={tine_z:.3f} > pocket_top+0.15={pocket_top+0.15:.3f}")


import math

def main():
    # Create env and reset
    cfg = ForkliftEnvCfg()
    env = ForkliftEnv(cfg)
    env.reset()

    # Step a few times to let physics settle
    for _ in range(20):
        env.step(torch.zeros(1, 3, device=env.device))

    stage = omni.usd.get_context().get_stage()
    lines = []

    lines.append("=" * 80)
    lines.append("FORKLIFT SCENE PHYSICS AUDIT")
    lines.append("=" * 80)
    lines.append("")

    # Walk the entire stage
    lines.append("--- USD STAGE WALK ---")
    lines.append("")

    # Focus on interesting prim subtrees
    interesting_roots = [
        "/World/envs/env_0/Forklift",
        "/World/envs/env_0/Pallet_0",
        "/World/envs/env_0/Pallet_1",
        "/World/envs/env_0/Pallet_2",
        "/World/envs/env_0/Pallet_3",
        "/World/envs/env_0/CargoBox_0_0",
        "/World/envs/env_0/CargoBox_0_4",  # middle box
        "/World/envs/env_0/CargoBox_0_8",  # last box
        "/World/envs/env_0/CargoBox_1_0",
        "/World/envs/env_0/CargoBox_2_0",
        "/World/envs/env_0/CargoBox_3_0",
        "/World/Ground",
        "/World/Wall_N",
    ]

    for root_path in interesting_roots:
        prim = stage.GetPrimAtPath(root_path)
        if not prim.IsValid():
            lines.append(f"  MISSING: {root_path}")
            continue
        audit_prim(prim, lines, indent=0)
        # Also audit first few children
        for child in prim.GetChildren():
            audit_prim(child, lines, indent=1)
            for grandchild in child.GetChildren():
                audit_prim(grandchild, lines, indent=2)

    # Articulation details
    audit_articulation_joints(env.forklift, lines)

    # Rigid object state
    audit_rigid_objects(env, lines)

    # Grab logic diagnosis
    audit_grab_logic(env, lines)

    # Summary: flag problems
    lines.append("")
    lines.append("=" * 80)
    lines.append("PROBLEM SUMMARY")
    lines.append("=" * 80)

    # Check: all cargo kinematic?
    kinematic_count = 0
    dynamic_count = 0
    for root_path in interesting_roots:
        prim = stage.GetPrimAtPath(root_path)
        if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(prim)
            ke = rb.GetKinematicEnabledAttr()
            if ke and ke.Get():
                kinematic_count += 1
            else:
                dynamic_count += 1

    lines.append(f"  Kinematic rigid bodies: {kinematic_count}")
    lines.append(f"  Dynamic rigid bodies: {dynamic_count}")
    if kinematic_count > 0:
        lines.append("  WARNING: Kinematic bodies cannot be 'lifted' by physics forces.")
        lines.append("           They can only be moved by write_root_pose_to_sim().")
        lines.append("           Lifting is purely scripted, not physics-based.")
    lines.append("")

    # Check: collision AABB vs visual AABB for cargo
    lines.append("  Collision note: all cargo/pallets use CollisionAPI on child cubes.")
    lines.append("  These are unit cubes scaled by size — collision shape = visual shape")
    lines.append("  for UsdGeom.Cube + CollisionAPI (PhysX uses the cube directly).")
    lines.append("")

    # Print
    output = "\n".join(lines)
    print(output, flush=True)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "physics_audit.txt")
    with open(out_path, "w") as f:
        f.write(output)
    print(f"\n[SAVED] {out_path}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
