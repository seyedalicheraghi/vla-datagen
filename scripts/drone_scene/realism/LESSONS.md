# Lessons for the realism loop

Short, general lessons the agents read before each round and add to after it (fleet-scribe: small edits, merge
duplicates, keep under 150 lines, tag each new lesson with its round id).

## Calibration

Noise between two renders of the same code, measured by /first-run step 8. Treat a smaller change than this as no
change. Not measured yet.

- metrics: (per statistic, the largest A/A difference)
- judges: (how often a judge preferred one of two identical renders consistently in both orders)

## Before round r001 (from building the scene)

- Isaac Sim 5.1 runs Python 3.11: an f-string must not reuse its own quote type inside.
- The LMF2 drone has no movable joints: its ArticulationCfg needs `joint_pos={}`, `joint_vel={}`, `actuators={}`.
- The map is cached by `BUILD_VERSION` (world/build.py), generated textures by `TEXTURE_VERSION`
  (world/materials.py), the sky calibration by `PREPARE_VERSION` (prepare.py). Change the output, change the version,
  or the old cache is silently reused.
- USD `diffuseColor` and texture tints are linear, not sRGB; satellite reflectance is not albedo. Convert explicitly.
- Tree models: measure their height from their points, not their bounding box (one NVIDIA spruce has an empty bbox).
- Expected, not yet seen in a render: large flat-coloured areas read as fake from 40 m, so colour variation per patch
  may matter more than texture detail (ground_materials already varies patch luminance 0.75-1.3).
- The take-off pad is flattened to the backyard's lawn height; the map's terrain is the lawn beyond the patio.
- usd-core outside Isaac cannot open the map's https tree and grass references; that is expected.
- Unverified until the first run: several RTX setting names in rtx_look.py (it warns about unregistered keys), and the
  lens distortion / vignetting / lateral colour values in lucid.py (typical, not calibrated).
