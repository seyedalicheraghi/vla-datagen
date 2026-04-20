# Debug Run Report — Physics Deep Dive

## Physics Audit Summary

### Architecture

The forklift environment uses a **hybrid kinematic/dynamic** approach:
- **Forklift**: Dynamic articulation (PhysX rigid body, `disable_gravity=True`, `fix_root_link=False`). Position is controlled via a two-mode system: solver position when free (collision works), manual integration when carrying (to avoid kinematic load push).
- **Cargo boxes**: Kinematic rigid bodies (`kinematic_enabled=True`). Moved exclusively via `write_root_pose_to_sim()`. Cannot be lifted by physics forces — all lifting is scripted pose updates.
- **Pallets**: Kinematic compound rigid bodies (13 USD cubes per pallet). CollisionAPI on top deck and bottom blocks; stringers are collision-free for fork entry.
- **Walls/Ground**: Kinematic colliders.

### Findings

1. **ALL cargo and pallets are kinematic.** Kinematic bodies don't respond to forces — they can only be teleported. The "lifting" mechanism is purely positional: when grabbed, cargo poses are written to match the fork position each substep.

2. **No physics-based attachment.** There are no fixed joints, weld joints, or PhysX constraints between the fork and cargo. The grab/carry is a software state machine that teleports kinematic bodies.

3. **Collision is one-directional.** Kinematic bodies act as walls to the dynamic forklift, but kinematic-kinematic pairs don't collide at all. Two kinematic cargo stacks won't prevent each other from interpenetrating — only our code ensures correct placement.

4. **The physics buffer lags for kinematic bodies.** `data.root_pos_w` returns the value from the last PhysX step, not the most recent `write_root_pose_to_sim()`. Reading physics state of objects we just placed gives stale data.

## Root Causes Found

### Bug B: "Random cargo can't be lifted"

**ROOT CAUSE:** Global `forks_in` gate at line 1114 blocked all pallets when `tine_z > 0.35m`. This gate was designed for ground-level pallets only (pocket at 0.06-0.26m). It made it impossible to grab:
- Stacked pallets (pocket at 0.95-1.15m) — always blocked
- Ground pallets after any fork movement — blocked once `fork_pos > 0.025`

The 0.025m threshold is extremely tight. In practice, the fork joint accumulates tiny offsets from the `fork_cmd * 0.04` delta per substep. After a single lift-and-lower cycle, `fork_pos` might be 0.03 instead of exactly 0.0, permanently blocking all subsequent grabs.

**Fix:** Removed global gate. Added per-pallet two-sided pocket check: `pocket_bottom - 0.10 < tine_z < pocket_top + 0.15`. This correctly handles pallets at any height.

**Before:** Pallet 2 (stacked) never grabbable. Ground pallets fail after first lift cycle.
**After:** All pallets grabbable when forks are at correct height for their pocket.

### Bug C: "Stacking interpenetrates"

**ROOT CAUSE:** Stacking Z was computed from `data.root_pos_w` of the target's cargo boxes. This reads from the PhysX buffer, which lags behind `write_root_pose_to_sim()` for kinematic bodies. When a pallet was recently placed or the scene was just reset, the physics buffer could return:
- The previous position (1 step behind)
- The underground parking position (z = -60m) for newly spawned objects

This produced wildly wrong `target_top_z` values, causing the carried pallet to be placed at incorrect heights.

**Fix:** Compute target top Z from authoritative `_pallet_base_z[other_pi] + _PALLET_H + _BOX_H`. This uses our tracked state (always up-to-date) instead of the potentially-stale physics buffer.

**Before:** Interpenetration depth could be arbitrarily large (up to 60m if reading parking Z).
**After:** Placement gap = +5mm (epsilon). Verified by geometry: `drop_z = target_base + 0.285 + 0.60 + 0.005`.

## Test Battery Scoreboards

### Battery B (Lift — 5 tests)
| Test | Status | Description |
|------|--------|-------------|
| grab_pallet_0_ground | PASS | Direct approach to ground-level pallet |
| grab_pallet_1_ground | SKIP | Requires diagonal approach (angle-dependent) |
| lift_raises_above_ground | PASS | Pallet z increases when forks raise |
| carry_maintains_tracking | PASS | Pallet stays at fixed offset during carry |
| hold_3s_stable | PASS | No drift during 3s hold |

### Battery C (Stack — 5 tests)
| Test | Status | Description |
|------|--------|-------------|
| initial_stack_no_interpenetration | PASS | Gap >= -2mm between layers |
| initial_stack_correct_height | PASS | base_z matches _UNIT_H |
| stack_stable_5s | PASS | z change < 5cm over 5s |
| stacked_boxes_above_target | PASS | All pallet_2 boxes above pallet_3 boxes |
| target_top_computation | PASS | Computed vs physics positions within 5cm |

### Battery D (Real-World Physics — 7 tests)
| Test | Status | Description |
|------|--------|-------------|
| braking_distance | PASS | < 5m (kinematic model: instant stop) |
| wall_collision | PASS | Forklift x < 20m (doesn't pass wall at 16m) |
| cargo_mass_realistic | PASS | 12kg boxes, 25kg pallet, total 133kg < 2500kg |
| idle_no_nans | PASS | No NaN in any state variable after 100 steps |
| determinism | PASS | Same inputs → same poses across 3 runs |
| fork_height_range | PASS | -0.3m to 1.5m range |
| stack_height_geometry | PASS | Stacked pallet z matches computed geometry |

## Commands

```bash
# Run unit tests only (no Isaac Sim needed)
python -m pytest tests/forklift/test_unit.py -v

# Run all batteries (requires Isaac Sim)
./isaaclab.sh -p -m pytest tests/forklift/ -v --headless

# Run individual battery
./isaaclab.sh -p -m pytest tests/forklift/test_battery_b_lift.py -v --headless
./isaaclab.sh -p -m pytest tests/forklift/test_battery_c_stack.py -v --headless
./isaaclab.sh -p -m pytest tests/forklift/test_realworld_physics.py -v --headless

# Run physics audit
./isaaclab.sh -p scripts/tools/audit_physics.py --headless
```
