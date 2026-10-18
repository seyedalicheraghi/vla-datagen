# Work log: heavy machines and drone scene

This is a record of the work done in the Claude Code session on branch `claude/review-last-request-I6mXJ`:
- what was built;
- where the code is;
- what was run, and where;
- what has not been verified;
- how to run it on your PC.

## Where the code is

| Location | State |
|---|---|
| `seyedalicheraghi/IsaacSim` (your private repo), branch `main` | **All the work below.** Same commit on its `claude/review-last-request-I6mXJ` branch. |
| `seyedalicheraghi/vla-datagen` (old public fork), branch `claude/review-last-request-I6mXJ` | The same commit. Safe to delete. |

`main` is your earlier `main` (Oct 6 forklift work) plus the commits listed at the end: 175+ files, about 58,000 added
lines. Only two of your existing files changed: `README.md` (2 lines added) and `.gitignore` (9 lines added).

To update your PC, in your IsaacLab folder whose `origin` is IsaacSim:

```bash
git checkout main
git pull origin main
```

Move your local untracked `CLAUDE.md` aside first: the repo now has one (notes for Claude Code sessions), and git will
not overwrite an untracked file. If git complains about other local changes, run `git stash` first.

## Where things ran: please read

- **This session ran on Anthropic's cloud machine** (4 CPU cores, no GPU), not on your PC. It has no connection to
  your PC, so nothing ran on your PC and your Isaac Sim install was not touched.
- **Run on the cloud machine:**
  - the unit tests;
  - downloads of public data (NVIDIA asset server, USGS terrain, Overture Maps, ESA WorldCover, Sentinel-2, Meta
    canopy height);
  - building the real-world map file;
  - dry runs of the Isaac script with Isaac Sim replaced by stand-ins (Python logic only, no rendering);
  - the multi-agent code review.
- **Earlier in the session, rendered on the cloud machine:**
  - the drone-scene preview images (`drone_previews/`);
  - the two 20 s flight-test videos (`drone_videos/`), path-traced with Mitsuba 3 on the CPU;
  - the heavy-machine preview images.

  You said rendering must happen only on your PC. Rendering on the cloud machine went against that.
- **Never run anywhere: Isaac Sim.** Nothing in this work has been rendered or tested in Isaac Sim. How the scenes
  look in Isaac Sim, and whether every Isaac API call works on your version, is unverified until you run it.
- **To have everything run on your PC from now on:** run Claude Code on the PC itself. In a terminal in your
  IsaacLab folder, run `claude remote-control`, or use the Claude Desktop app or Claude Code in VS Code.

## 1. Heavy machines (`scripts/heavy_machines/`, `assets/heavy_machines/`)

- **Machines.** 18 construction machines, each defined in one Python file and generated into URDF + JSON. Each
  file holds the machine's datasheet dimensions and its drive type (skid, Ackermann or articulated).
  - Abbreviations used in the scripts: HEX = excavator, ADT = articulated dump truck, CTL = compact track loader,
    WL = wheel loader.
  - The others: mini and wheeled excavator, backhoe loader, dozer, motor grader, scraper, rigid haul truck,
    concrete mixer truck, rough-terrain crane, telehandler, boom lift, skid steer loader, compactor, asphalt paver.
- **Environment and tools:** one Isaac Lab environment for all machines (`machine_env.py`), keyboard teleop, a
  scripted demo, a showcase, and a language-conditioned data collector (ROS 2 mcap bags + LeRobot).
- **Checks:** `check_machine.py` validates the kinematics, limits, sizes and masses, and that each tool moves as
  labelled.
- **Quarry scenario** (`scenarios/`, "ADT approaches HEX"): a procedural quarry, a scripted truck and excavator,
  camera rigs, baked PBR textures and rocks, and a render script.
- **Installer:** `setup_linux.sh`, a one-paste installer with no git needed. It writes `~/heavy_machines/render.sh`
  and `drone.sh`.
- **Windows:** the Windows installers and launchers never worked reliably on your PC and were removed. The
  Windows text-encoding fixes stay in.

## 2. Drone backyard scene (`scripts/drone_scene/`)

The Isaac Lab quadcopter demo in a photorealistic backyard:
- **Ground:** NVIDIA photo-scanned PBR materials, real 3D herringbone paver bricks, 3D lawn grass, a mulch bed,
  shrubs and a maple, fallen leaves.
- **Sky:** an HDR sky with its sun cut out and replaced by a calibrated sun light.
- **Renderer:** RTX settings tuned for a physical camera look (`rtx_look.py`).
- **Crazyflie details:** PCB, battery, motor and LED detail, and propeller-blur discs.
- **Offline preview:** a Mitsuba path-traced preview tool, used on the cloud machine; see above.
- **Garden flight:** a 20 s flight with front and back onboard cameras (`flight.py`).
- **Comparison:** two OpenArt-generated comparison videos, made from start and end frames of the cloud renders.

## 3. LUCID cameras on the drone (`lucid.py`, `sensor_sim.py`, `vehicles.py`)

- **Camera models** (datasheet specs):
  - LUCID Triton TRI032S-CC (default): Sony IMX265, 2048 × 1536, global shutter, 67 g;
  - Triton TRI050S-CC: 5 MP;
  - Phoenix PHX016S-CS: 1.6 MP;
  - each paired with a generic C-mount or M12 lens. The default 4 mm f/2.8 lens gives an 87° × 69° view.
- **Camera simulation**, run on your GPU with torch, starting from Isaac's linear HDR render:
  - lens distortion, corner darkening and colour fringing;
  - auto exposure;
  - photon and read noise, a 12-bit Bayer raw frame, demosaicing, white balance and gamma.
- **Calibration file:** each video gets a `camera_info.json` with the intrinsics, distortion, mount pose and the
  exposure of every frame.
- **Drone:** the default is now the 1.2 kg NTNU LMF2 quadrotor from NVIDIA's assets. The 27 g Crazyflie cannot
  carry two of these cameras; `--vehicle crazyflie` keeps the old drone.
  - Half of the LMF2 model was bound to a pure-black material; it was given proper metal, plastic and glass
    materials.
  - The cameras are mounted from the drone's own mesh and drawn on it, placed so the drone covers no more than
    1 % of either view.

## 4. 2 km flight over a large real-world map (`world/`, `--route straight`)

- **The map:** a real place, by default west Ames, Iowa, built on your PC before Isaac Sim starts.
  - **Terrain:** USGS 3DEP lidar (1 m), or Copernicus 30 m outside the US.
  - **Buildings, roads, water, land use:** Overture Maps (OpenStreetMap + Microsoft footprints).
  - **Land cover:** ESA WorldCover.
  - **Trees:** placed from the Meta/WRI 1 m canopy-height map.
  - **Colour:** a clear summer Sentinel-2 image.
- **Contents:**
  - about 3 km of full detail around the route;
  - 13 km of coarser terrain whose tree lines stand on the horizon, then 25 km of horizon terrain;
  - about 19,000 buildings with roofs and windows and about 680 km of roads;
  - about 50,000 instanced NVIDIA trees;
  - crop fields with real row direction.
- **The route:** chosen automatically. It takes off from a real backyard, where the hand-built patio is placed,
  and lands on open grass 2 km away. You can also choose it with `--region`, `--takeoff` and `--landing`.
- **The flight:** take-off, climb to 40 m (kept 15 m above anything on the line), 2 km at 10 m/s with a realistic
  forward lean, descent and touchdown, about 4.5 minutes in all.
  - The sun is placed for the real date and time.
  - A chase camera follows the drone.
  - Video frames are piped straight into ffmpeg.
- **Builder dependencies:** they install once into `~/.cache/drone_scene/world_libs`, not into Isaac Sim's own
  Python packages. The built map is cached under `~/.cache/drone_scene/worlds/`.

## 5. Code review

- **How:** reviewer agents checked five areas: Isaac integration, flight, map geometry, geodata and camera
  simulation.
- **Result:** they reported 29 issues. All were real and all were fixed in commits `ea62ab2` to `ee32bb6`. A
  second set of agents re-checked each one against the final code and found none still present.
- **Examples:**
  - `--record -1` did not record;
  - the back camera saw the drone's body;
  - roads sank into coarse terrain tiles;
  - roofs had holes;
  - lawns came out grey;
  - crop rows formed a checkerboard;
  - negative coordinates failed;
  - GUI flights ran 10× slower than real time.

## How to run it on your PC

From your clone, on `main`:

```bash
# the 2 km flight: front and back LUCID cameras plus the chase camera, whole flight (~4.5 min of video)
./isaaclab.sh -p scripts/drone_scene/quadcopter_pbr.py --route straight --headless --record -1 --cameras front back chase

# quick test: first 60 s, half-resolution cameras
./isaaclab.sh -p scripts/drone_scene/quadcopter_pbr.py --route straight --headless --record 60 --cameras front back --resolution_scale 0.5

# GUI, following the drone
./isaaclab.sh -p scripts/drone_scene/quadcopter_pbr.py --route straight --view chase

# the backyard garden flight (20 s)
./isaaclab.sh -p scripts/drone_scene/quadcopter_pbr.py --fly --headless --record 20 --cameras front back

# heavy-machine quarry video
./isaaclab.sh -p scripts/heavy_machines/scenarios/render_scenario.py --scenario adt_approach_hex --headless
```

- **First straight-route run:** installs the map tools (about 250 MB), downloads about 150 MB of map data, and
  builds the map in a few minutes on the CPU.
- **Output:** in `drone_scene_videos/`: `drone_front.mp4`, `drone_back.mp4`, `drone_chase.mp4`, the camera_info
  JSON files and `drone_scene.log`.
- **If something fails:** the cause is in `drone_scene.log`.

Useful flags:
- `--region=LAT,LON`, `--takeoff=LAT,LON`, `--landing=LAT,LON`: choose the place.
- `--when=2025-07-15T15:30:00Z`: set the sun.
- `--cruise_agl`, `--speed`: cruise height and speed.
- `--lucid`, `--lucid_lens`, `--camera_pitch`: camera model, lens and tilt.
- `--vehicle crazyflie`: the old drone.
- `--pathtracing`: the path tracer instead of real-time RTX.

Full details are in `scripts/drone_scene/README.md` and `scripts/heavy_machines/README.md`.

## Not verified, or known limits

- **Isaac Sim itself.** Nothing has been run in Isaac Sim, so the following are untested:
  - the RTX look and the exposure;
  - whether Isaac Sim 5.1 provides the `HdrColor` image the camera simulation needs (if not, it falls back to
    Isaac's 8-bit image and says so in the log);
  - load time and frame rate with the large map;
  - how the LMF2 drone behaves with 0 joints.
- **Lens values.** The lenses' distortion, corner darkening and colour fringing are typical values, not a
  calibration of a real lens.
- **Exposure time.** The reported exposure time assumes the calibrated sun and sky stand for 100,000 lux.
- **Map quality.** It depends on the public data for the place: building heights exist for about 60 % of Ames
  buildings, and roof shapes are mostly inferred. Outside the US the terrain is a surface model with the canopy
  subtracted, so expect 1-3 m errors.
- **Distant haze.** There is no atmospheric haze, so far terrain looks crisper than in reality.
- **Tests.** The unit tests (`scripts/drone_scene/test/`, `scripts/heavy_machines/test/`) check the Python logic and
  geometry only.

## Commits on the branch (newest first)

```
ee32bb6 Camera review fixes: bright overflowed pixels, exact lateral colour
189da11 Geodata review fixes: true colours, real fields, creeks, robust places
d1ad8dd Isaac review fixes: GUI flights in real time, chase camera at its real distance
951bc95 Drone map geometry fixes from the review
f715611 Drone map: coarse terrain out to 25 km so the land reaches the horizon
ea62ab2 Review fixes: --record -1, self-free camera mounts, chase camera, garden heading
52b1f79 Drone map: backyard sits on the map's lawn; take-offs clear of footpaths
5859132 Drone scene: 2 km straight flight over a real map built from open geodata
bfd380d Drone scene: two LUCID cameras with a simulated lens, sensor and ISP; LMF2 quadrotor
6026ba3 Remove the Windows installers (Linux workstation only)
96f1970 Add the rendered drone-scene preview images for download
182d820 Add the two 20 s drone camera videos (front and back) for download
d1fbaea Drone scene: physically based RTX camera pipeline (rtx_look)
57397d8 Drone scene: propeller blur discs a little denser (daylight shutter, gain 1.8)
a8cef0c Drone scene v4: real 3D herringbone paving and drone close-up detail
a156fd6 Drone scene: flying drone with front/back onboard cameras
d8bbcb1 Drone scene: asset fetcher, HDR sky tools, Mitsuba preview; first look-dev
e76b61f Drone scene: review fixes and asset placement
0546e8b Drone scene: PBR backyard for the Crazyflie demo (work in progress)
3d0abd6 setup_linux.sh: install imageio-ffmpeg when no ffmpeg is available
0962124 Linux one-paste setup without git
bf5ef85 Windows setup without git: download the heavy-machine folders from GitHub
a848c0d Windows: one-line setup script that checks every step
7cf9aa1 render_scenario: log every start-up failure and keep it visible
9b22747 Windows launcher: fixes from the cmd.exe / PowerShell 5.1 review
1ba43c1 Windows one-command launcher and a render log
e59095b Make heavy_machines run on Windows (PowerShell, pip Isaac Sim)
09218d7 Bake scenario textures in a cached pre-step before Isaac Sim starts
e2e5d93 scenario terrain: irregular pit walls (uneven bench heights/widths, wandering crests, slumps)
c667107 photo_preview: ray march reaches the far terrain (it stopped at ~30 m, so pit walls fell through to sky)
47b119d scenario terrain: closer, darker stratified pit walls with dusty ledges; lighter baked haze
23c6927 heavy_machines scenarios: photoreal pipeline — machine render twins, PBR textures, rocks, camera look, path tracing
0832b2f scenario: review fixes — seamless split terrain mesh, realistic berms, clearer event order, full arm in frame
bbcb1fd README: scenarios section
fc9088e scenario terrain: irregular blast-cut bench face, windrows, lumpy berms, lane crown, patchy ground
bed7c50 heavy_machines: scenario framework + 'ADT approaches HEX' quarry scene for Isaac Sim
c393b45 heavy_machines: finish fleet polish — every tool visible, no motion clashes, detailed reference excavator
12cae28 heavy_machines: rear-view cameras so the dump body, screed and ripper are visible
c5be646 heavy_machines: importable ROS 2 bag writer with an offline round-trip test
5dfd20a README: heavy machines section; check_machine.py --readme regenerates the fleet table
45c537a heavy_machines: approximate camera-view renders; visibility check samples the whole tool range
601b4cd heavy_machines: fix env issues from the Isaac Lab API review; final batch fixes
9f01cc5 heavy_machines: 9 more machines (loaders, haulers, dozer, backhoe, crane, telehandler, scraper)
676a60b heavy_machines: 8 more machines; review-driven fixes to checks, env, tasks
8793e8b heavy_machines: language-conditioned collector (ROS 2 bags + LeRobot), README
b0c5e94 Add heavy_machines: spec-driven construction machines for Isaac Lab
```
