# Drone scene: two LUCID cameras on a quadrotor in a PBR backyard

Isaac Lab's quadcopter demo in a photorealistic backyard. It uses photo-scanned PBR ground (3D herringbone
pavers, lawn with 3D grass, a mulch bed, a curb), NVIDIA shrubs and a maple, fallen leaves, and an HDR sky
whose sun is replaced by a calibrated DistantLight. The drone carries two LUCID Vision Labs machine-vision
cameras: `front` looks forward and `back` looks backward, both tilted 10° down. Their videos are what those
cameras would deliver, not Isaac's tone-mapped viewport image.

Everything renders on your own GPU, inside Isaac Sim 5.1 / Isaac Lab 2.3.

```bash
# the 20 s garden flight, both LUCID cameras -> drone_scene_videos/drone_front.mp4, drone_back.mp4
./isaaclab.sh -p scripts/drone_scene/quadcopter_pbr.py --fly --headless --record 20 --cameras front back

# quick test at half resolution, about 4x faster
./isaaclab.sh -p scripts/drone_scene/quadcopter_pbr.py --fly --headless --record 4 --cameras front back --resolution_scale 0.5

# path-traced instead of real-time RTX (much slower, best light)
./isaaclab.sh -p scripts/drone_scene/quadcopter_pbr.py --fly --headless --record 20 --cameras front back --pathtracing

# GUI: watch the flight through the front camera (undistorted pinhole view, no sensor simulation)
./isaaclab.sh -p scripts/drone_scene/quadcopter_pbr.py --fly --view front
```

Without git, `scripts/heavy_machines/setup_linux.sh --drone <the same arguments>` downloads this folder and
runs it with the Python that has Isaac Sim (it writes `~/heavy_machines/drone.sh` for the next time).

Output goes to `drone_scene_videos/` (`--out`):

| File | What it is |
|---|---|
| `drone_front.mp4`, `drone_back.mp4` | the camera videos (H.264, full sensor resolution, 25 fps) |
| `front/frame_*.png`, `back/frame_*.png` | the same frames as 8-bit PNGs |
| `front/raw_*.png` (with `--save_raw`) | the 12-bit RGGB Bayer raw frames (16-bit PNG, LUCID BayerRG12) |
| `drone_front_camera_info.json` | `K`, `D` (OpenCV plumb_bob), FOV, the mount pose on the drone, sensor/lens data, and the exposure time, gain and clipped fraction of every frame |
| `drone_scene.log` | everything the run printed; send this file if a run fails |

## The cameras (`lucid.py`)

| `--lucid` | Camera | Sensor | Resolution | Default lens (`--lucid_lens`) | FOV (h × v) |
|---|---|---|---|---|---|
| `TRI032S-CC` (default) | Triton 3.2 MP, 67 g | Sony IMX265 1/1.8", global shutter | 2048 × 1536 | `C4`: 4 mm f/2.8 C-mount | 87° × 69° |
| `TRI050S-CC` | Triton 5.0 MP, 67 g | Sony IMX264 2/3", global shutter | 2448 × 2048 | `C6`: 6 mm f/2.4 C-mount | 72° × 62° |
| `PHX016S-CS` | Phoenix 1.6 MP, 30 g | Sony IMX273 1/2.9", global shutter | 1440 × 1080 | `S3.6`: 3.6 mm f/2.0 M12 | 74° × 57° |

The camera specs come from the datasheets. The lenses are generic: their distortion, vignetting and lateral
colour are typical values for such lenses, not a calibration of a specific product. If you have a
calibration of your own camera (`K`, `D`), put the numbers into `lucid.LENSES` so the simulated images match
it. The run warns when the recording frame rate is above what the real camera can do (e.g. the 5 MP Triton
tops out at 22 fps), and when a lens does not cover the sensor.

## How an image is made (`sensor_sim.py`, on the GPU)

1. Isaac Sim renders a pinhole image in linear HDR radiance (Replicator `HdrColor`, before tone mapping).
   It is about 12 % wider than the default lens sees, because barrel distortion pulls more of the scene
   into the corners.
2. **Lens.** Barrel distortion (the rig's `D`), red and blue magnified slightly differently (lateral
   colour), corner fall-off (relative illumination) and a 0.5 px blur.
3. **Auto exposure.** Like the camera's `ExposureAuto = Continuous`: it meters the clipped raw signal and
   moves 35 % of the way to its target each frame. A bright sky therefore clips, as it does on a real
   camera. `--exposure fixed --exposure_us 400 --gain_db 0` sets the exposure by hand instead.
4. **Sensor.** The RGGB colour filter (the sensor's raw red and blue are weaker than green in daylight),
   Poisson photon noise, 0.6 % pixel gain non-uniformity, a 10.5 ke⁻ full well, 2.8 e⁻ read noise and a
   12-bit ADC.
5. **Image pipeline.** Malvar-He-Cutler demosaicing, daylight white balance and sRGB gamma, giving RGB8
   (`--gamma 1` gives linear output, like the camera's raw look).

The global shutter means no rolling-shutter skew. At daylight exposures (a few hundred µs) motion blur is
below a pixel, so none is added. The reported exposure times assume the calibrated sun and sky stand for
a clear day of 100 klx. The first frame's log line compares the render's brightness with that calibration.

## The drones (`vehicles.py`)

- **`--vehicle lmf2`** (default) is the LMF2 quadrotor from NTNU's Autonomous Robots Lab, the asset behind
  Isaac Lab's `ARL_ROBOT_1_CFG`. It weighs 1.2 kg and measures 0.52 m across its propeller guards. Two
  Tritons with lenses and brackets (~250 g) are well inside its payload. In NVIDIA's asset, half of the
  mesh (the aluminium frame, the blue anodised parts and the glass) is bound to a pure-black material.
  `vehicles.apply_lmf2_materials` replaces it with physically based aluminium, anodised aluminium, steel,
  brass, plastics and glass, chosen by the CAD appearance names.
- **`--vehicle crazyflie`** is the original 27 g demo drone, with its PCB, battery and motor textures,
  LEDs and propeller blur discs. It can carry about 15 g, so a real one could not lift these cameras. The
  log says so.

The cameras are mounted using the drone's own mesh. Each sits on the centre line, with the housing at least 16 mm clear of the body (room for the M12 GigE connector). The mount height and bracket length are chosen so the drone covers at most 1% of the camera's view: its triangles are drawn through the real lens to check. The cameras appear on the drone as housing, lens, connectors and an aluminium bracket. The third-person views (`hero`, `garden`, `demo`) stand back 3× for the larger drone.

## Long flight over a real map (`--route straight`)

The drone takes off from a real backyard and flies 2 km straight (`--distance`) to land on open grass. The
take-off is from the hand-built backyard patio, placed in a real yard. Both cameras record the whole flight:

```bash
# about 4.5 minutes of flight: take-off, climb to ~40 m, 2 km at 10 m/s, descent, landing on grass
./isaaclab.sh -p scripts/drone_scene/quadcopter_pbr.py --route straight --headless --record -1 --cameras front back chase

# quick look: first 60 s, half-resolution cameras
./isaaclab.sh -p scripts/drone_scene/quadcopter_pbr.py --route straight --headless --record 60 --cameras front back --resolution_scale 0.5

# GUI, following the drone
./isaaclab.sh -p scripts/drone_scene/quadcopter_pbr.py --route straight --view chase
```

The map is a real place, built from open geodata by `world/` on your PC before Isaac Sim starts:

| What | Source | Licence |
|---|---|---|
| terrain (bare earth) | USGS 3DEP lidar 1 m (US), else 3DEP 10 m / Copernicus DEM 30 m | public domain / Copernicus |
| buildings, roads, water, land use, single trees | Overture Maps (OpenStreetMap, Microsoft and Esri footprints) | ODbL 1.0, CC BY 4.0 |
| ground type | ESA WorldCover 2021, 10 m | CC BY 4.0 |
| individual trees (position, height) | Meta / WRI 1 m canopy height map | CC BY 4.0 |
| ground colour seen from far away, texture calibration | Sentinel-2 L2A (clearest summer scene) | Copernicus, free |
| trees, grass, ground textures | NVIDIA Omniverse assets | NVIDIA |

What gets built:

- **Terrain.** The detail zone (the route ±600 m and 400 m beyond each end) has 1-16 m cells, finest along
  the route. Out to ~6.5 km, coarser terrain is raised by the real canopy, so tree lines stand on the
  horizon, and coloured with the satellite image. Beyond that, 250 m terrain runs out to 25 km in the area's
  average colour, so the land reaches the true horizon (about 22 km away from 40 m up).
- **Ground.** NVIDIA photo-scanned materials per land class: lawn, meadow, forest floor, soil, concrete,
  sand. Crop fields get rows on a 0.76 m spacing along the survey grid. Satellite reflectance calibrates
  each patch's colour, so fields and lawns have their real colour from the air.
- **Buildings.** Every Overture footprint becomes walls with windows, plus gable or hip roofs on houses
  (cross-gables on L-shaped ones) and flat roofs on large buildings. Heights come from the data, or from
  the building type and size.
- **Roads.** Ribbons at the class width with lane markings: asphalt, concrete or gravel.
- **Trees.** One NVIDIA tree per detected crown, scaled to its measured height. The species mix is
  mostly oaks, maples, ash and honey locust, with some spruce.
- **Landing site.** 3D meadow grass around the landing spot.

What the run does:

- **Map.** The first run installs the map tools into `~/.cache/drone_scene/world_libs`, once and outside
  Isaac Sim's packages (about 250 MB). It downloads about 150 MB of map data and builds the map in a few
  minutes on the CPU. The map is cached under `~/.cache/drone_scene/worlds/`; later runs start at once.
- **Sun and sky.** The sun is placed for the real date and time (`--when`, default 15 Jul 2025, 10:30
  local). The sky is a pure-sky HDR whose glow is turned to the sun's direction.
- **Cruise height.** It is `--cruise_agl` (40 m) above the take-off, raised if needed to stay 15 m above
  every tree and building near the line.
- **Flight attitude.** At 10 m/s the drone leans about 12° into the air stream, like a real quadrotor. The
  LUCID cameras are tilted 20° down (`--camera_pitch`), so the front one looks about 32° down in cruise.
- **Video output.** Frames go straight into ffmpeg, so a 4-minute 3 MP flight doesn't need tens of GB of
  PNGs (`--save_frames` keeps them anyway).

Choosing the place: `--region LAT,LON` searches for a route around that point (default: west Ames, Iowa). The
search looks for a backyard behind a house, then open grass `--distance` metres away at least 35 m from
buildings, trees and roads. It prefers the route that crosses the most varied ground. To fly a specific path,
give `--takeoff LAT,LON --landing LAT,LON`. Outside the US the terrain falls back to Copernicus 30 m, a
surface model that includes trees and buildings.

## Tests (no Isaac Sim needed)

```bash
python scripts/drone_scene/test/test_lucid.py      # optics, distortion, sensor noise, demosaicing, exposure
python scripts/drone_scene/test/test_vehicles.py   # mounts, LMF2 materials, framing
python scripts/drone_scene/test/test_flight.py     # the garden flight, the straight flight and camera poses
python scripts/drone_scene/test/test_world.py      # map builder: terrain, land cover, buildings, roads, trees, site choice
```

What these tests cannot check, because they cannot run Isaac Sim: whether Isaac Sim 5.1 returns `HdrColor`
for these cameras, and how the frames look. If `HdrColor` is missing, the script says so in the log. It then
feeds the sensor simulation Isaac's tone-mapped image, which still runs but with Isaac's highlight
roll-off.
