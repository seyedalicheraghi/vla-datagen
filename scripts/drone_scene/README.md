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

The cameras are mounted from the drone's own mesh: on the centre line, at the mount height, with the
housing 16 mm clear of the body (room for the M12 GigE connector). They are drawn on the drone as housing,
lens, connectors and an aluminium bracket. The third-person views (`hero`, `garden`, `demo`) stand back 3×
for the larger drone.

## Tests (no Isaac Sim needed)

```bash
python scripts/drone_scene/test/test_lucid.py      # optics, distortion, sensor noise, demosaicing, exposure
python scripts/drone_scene/test/test_vehicles.py   # mounts, LMF2 materials, framing
python scripts/drone_scene/test/test_flight.py     # the flight path and camera poses
```

What these tests cannot check, because they cannot run Isaac Sim: whether Isaac Sim 5.1 returns `HdrColor`
for these cameras, and how the frames look. If `HdrColor` is missing, the script says so in the log. It then
feeds the sensor simulation Isaac's tone-mapped image, which still runs but with Isaac's highlight
roll-off.
