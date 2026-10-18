# Notes for Claude Code (read before doing anything)

## Start here

- Read `WORK_LOG.md` (repo root). It covers:
  - what has been built: 18 heavy machines, the drone backyard scene, two LUCID cameras on the drone, and a 2 km flight
    over a real map;
  - what ran where;
  - what is not verified;
  - how to run everything.
- Details are in `scripts/drone_scene/README.md` and `scripts/heavy_machines/README.md`.

## Where things stand

- `main` of `seyedalicheraghi/IsaacSim` (private) holds all of it, up to commit `ce68ec9` and the commit that added this file.
- **Nothing has been run in Isaac Sim yet.** Everything so far was checked only by unit tests and by dry runs with Isaac Sim
  replaced by stand-ins. How things look and run inside Isaac Sim is unverified.
- **Next step:** the first real run on this PC.
  1. Do a quick check first:
     ```bash
     ./isaaclab.sh -p scripts/drone_scene/quadcopter_pbr.py --route straight --headless --record 60 --cameras front back --resolution_scale 0.5
     ```
  2. Read `drone_scene_videos/drone_scene.log`:
     - does `HdrColor` work, or is there a "no HdrColor" warning?
     - the "mean scene radiance = ... x an 18 % grey card" line;
     - the camera mount lines;
     - the `[rtx_look] WARNING` lines (unregistered settings);
     - any traceback.
  3. Fix what fails, then do the full flight (`--record -1 --cameras front back chase`) and look at the videos.

## Rules from the owner

- **Run and render only on this PC, with its GPU.** Never on a remote or cloud machine, and never make preview renders
  elsewhere.
- **Helper agents (subagents)** use cheaper models unless the task really needs the strongest one.
- **Repo:** `seyedalicheraghi/IsaacSim` (private), branch `main`. **Ask before pushing.** Don't push to other repos. The old public
  fork `vla-datagen` is being deleted.
- **Explanations:** be plain and exact. Give full paths, say which machine and folder you mean, and give commands that
  were actually checked.
- **Setup:** Ubuntu PC with Isaac Sim 5.1 and Isaac Lab 2.3.2 in the IsaacLab folder's `.venv`, run with
  `./isaaclab.sh -p ...`. The Windows/WSL route was abandoned.

## Code map

| Path | What |
|---|---|
| `scripts/heavy_machines/` | 18 construction machines (spec files -> URDF/JSON), Isaac Lab env, teleop, ROS 2 / LeRobot collector, quarry scenario renderer |
| `scripts/drone_scene/quadcopter_pbr.py` | the Isaac script: backyard scene, `--route garden|straight`, `--vehicle lmf2|crazyflie`, LUCID cameras, recording |
| `scripts/drone_scene/flight.py` | `Flight` (20 s garden tour), `StraightFlight` (take-off, cruise with drag tilt, landing) |
| `scripts/drone_scene/lucid.py`, `sensor_sim.py` | LUCID camera/lens models; GPU sensor + image pipeline fed by Replicator `HdrColor` |
| `scripts/drone_scene/vehicles.py` | LMF2 / Crazyflie, LMF2 PBR materials, camera mounts (kept out of their own view) |
| `scripts/drone_scene/world/` | real-world map builder (terrain, land cover, buildings, roads, trees, site choice, USD); runs before Isaac Sim via `runner.py` |
| `scripts/drone_scene/scene.py`, `pavers.py`, `drone_look.py`, `rtx_look.py`, `prepare.py`, `hdr.py`, `assets.py`, `nucleus.py` | backyard scene, assets, sky calibration, RTX settings |
| `scripts/drone_scene/test/`, `scripts/heavy_machines/test/` | plain `unittest` files, no Isaac Sim needed: `python scripts/drone_scene/test/test_world.py` etc. |

## Things learned the hard way

- **Python 3.11** (Isaac Sim 5.1): no f-strings that reuse their own quote type inside, which needs 3.12. Check with
  `python3.11 -m py_compile`.
- **0-joint drone:** the LMF2 drone has no movable joints. Its `ArticulationCfg` needs `init_state.joint_pos={}`,
  `joint_vel={}` and `actuators={}`, or Isaac Lab raises "Not all regular expressions are matched".
- **Map builder dependencies** (rasterio, pyproj, pyarrow, shapely, usd-core, apsw) go into `~/.cache/drone_scene/world_libs`
  with `pip install --target`. The builder runs as a separate process. Never pip-install them into the Isaac Sim
  environment, because they could upgrade numpy under Isaac.
- **Map cache:** maps live in `~/.cache/drone_scene/worlds/<name>_<key>/`.
  - Bump `BUILD_VERSION` in `world/build.py` whenever the map output changes.
  - `--force` rebuilds.
  - `--when` and `--cruise_agl` don't need a rebuild.
  - Generated textures are versioned by `TEXTURE_VERSION` in `world/materials.py`.
- **https references:** usd-core (outside Isaac) cannot open the map's `https://` tree and grass references. That is expected;
  Isaac Sim's resolver loads them.
- **Unverified values:**
  - The RTX tonemapper enum and several RTX setting names in `rtx_look.py`; it warns about unregistered keys.
  - The lens distortion, vignetting and lateral-colour values in `lucid.py`, which are typical, not calibrated.
