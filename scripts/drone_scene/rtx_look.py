"""RTX renderer settings for a physically based, cinematic camera pipeline (Isaac Sim 5.1 / Kit 107).

The target is the look of an Unreal Engine 5 cinematic.  UE5 feature -> Omniverse RTX equivalent, or what
this scene does instead:

- Lumen GI and reflections  -> RTX Real-Time indirect diffuse + reflections (``/rtx/indirectDiffuse/*``), or
                               RTX Real-Time 2.0's cached path tracing (``/rtx/rtpt/*``); the Path Tracer is
                               the ground-truth reference
- Nanite micro-geometry     -> no switch: RTX traces the full meshes.  The fine detail here is photo-scanned
                               PBR maps (normal + ORM) and real 3D grass and shrubs
- virtual shadow maps       -> ray-traced shadows from a 0.53 deg DistantLight sun: physical penumbra, no
                               shadow-map resolution to tune
- Megascans                 -> NVIDIA "Base" photo-scanned PBR materials and vegetation (``assets.py``)
- physical camera exposure  -> ``/rtx/post/tonemap/{filmIso,cameraShutter,fNumber}`` with auto exposure off
                               (UE5's "Manual" metering: exposure = 1 / 2^(EV100 + compensation))
- ACES filmic tonemapper    -> ``/rtx/post/tonemap/op`` = Aces
- bloom, vignette, chromatic aberration, grain
                            -> ``/rtx/post/lensFlares`` (FFT bloom), ``/rtx/post/tvNoise`` (vignetting and
                               film grain), ``/rtx/post/chromaticAberration``; opt-in, ``OPTIONAL["lens"]``

Exposure: ``prepare.calibrate`` sizes the dome (~900) and the sun (~3800) so the horizontal illuminance
equals Isaac Lab's reference (dome 750 with the kloofendal sky, ``assets.REFERENCE_INTENSITY``).  Isaac's
default film ISO / shutter / f-number are tuned for exactly that illuminance, so no exposure key is set;
``apply(exposure_ev=...)`` scales film ISO for a deliberate offset.

``/rtx/sceneDb/ambientLightIntensity`` is 0 in every preset.  Isaac Lab's rendering presets set it to 1.0:
that is a uniform fill (``ambientLightColor``, default 0.1 grey, times the intensity) added everywhere,
whatever the geometry, i.e. not physically based light; its unit is undocumented, so how strong it is next
to a dome of 900 is unmeasured.  RTX Real-Time 2.0 has no such setting at all (its docs say to use a
DomeLight), and the default is 0.0.  The scene already has a DomeLight.

Path-tracer samples: Isaac Sim 5.1's own ``SimulationApp`` sets ``/rtx/pathtracing/spp``, ``totalSpp`` and
``clampSpp`` to one number (default 64), and Isaac Lab's rendering presets raise ``maxSamplesPerLaunch`` to
1000000, so the preset does the same: one launch per frame, no 32-sample cap assumed.  The path-tracer docs UI
says "Samples per Pixel per Frame (1 to 32)"; whether that is a hard limit with ``clampSpp`` raised is not
confirmed, so ``apply`` logs the limits it finds before it sets them.  Splitting one frame over several launches
is opt-in: ``path_tracing_samples(total, max_per_launch=32)`` (what ``heavy_machines`` does with ``--subframes``).

``PROVENANCE`` says where each key comes from.  docs.omniverse.nvidia.com could not be fetched from the build
sandbox, so "docs" means a Kit 107-era page excerpt returned by web search, not a page read in full.  "source"
means NVIDIA code read in a clone: Isaac Sim 5.1's ``SimulationApp``, its standalone examples (develop branch)
and kit-app-template.  Names were also cross-checked against the render settings NVIDIA saved in its Isaac Sim 5.1
asset scenes (rtx:rendermode, rtx:pathtracing:maxBounces / totalSpp, rtx:sceneDb:ambientLightColor /
ambientLightIntensity, rtx:post:lensFlares:*, rtx:post:tonemap:whitepoint).  Keys tagged "unverified" are
spelled from USD attribute or UI names and should be checked with ``get_setting`` (``apply`` warns about every key
the running Kit does not register).

The "realtime" preset assumes Isaac Lab's ``rendering_mode="quality"`` (what ``quadcopter_pbr.py`` uses): it
does not set the indirect-diffuse, reflection or ambient-occlusion denoisers, which that mode enables.  Apply it
after ``SimulationContext`` is created, because the mode's ``quality.kit`` sets ``ambientLightIntensity`` to 1.0.

Sources:
- https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html (path tracer)
- https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_rt.html (Real-Time 2.0)
- https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_rt_legacy.html (Real-Time)
- https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_common.html (dome light)
- https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx_post-processing.html (post stack)
- https://docs.omniverse.nvidia.com/workflows/latest/rtx_rt-dh-setup.html (lens effect values, raw USD)
- https://docs.omniverse.nvidia.com/guide_rtx-best-practices/latest/render-depth.html (8 spp, vignette)
- https://docs.isaacsim.omniverse.nvidia.com/6.1.0/action_and_event_data_generation/ext_replicator-agent/
  ext_isaacsim_replicator_agent_visual_noise_example.html (tvNoise paths)
- https://github.com/NVIDIA-Omniverse/ovrtx/tree/main/docs (Kit 110 setting names, light units)
- https://github.com/isaac-sim/IsaacSim: source/extensions/isaacsim.simulation_app/.../simulation_app.py (v5.1.0:
  spp / totalSpp / clampSpp), source/standalone_examples/api/isaacsim.replicator.examples/motion_blur_*.py
- https://github.com/NVIDIA-Omniverse/kit-app-template (``histogram.enabled``, ``ambientLightIntensity``)
- https://dev.epicgames.com/documentation/unreal-engine/lumen-global-illumination-and-reflections-in-unreal-engine
- https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-virtualized-geometry-in-unreal-engine
- https://dev.epicgames.com/documentation/unreal-engine/virtual-shadow-maps-in-unreal-engine
- https://dev.epicgames.com/documentation/en-us/unreal-engine/auto-exposure-in-unreal-engine
- https://dev.epicgames.com/documentation/unreal-engine/color-grading-and-the-filmic-tonemapper-in-unreal-engine
- this repo: apps/rendering_modes/quality.kit, source/isaaclab/isaaclab/sim/simulation_cfg.py (dome strategy),
  source/isaaclab/test/sim/test_simulation_render_config.py (``aa/op``), scripts/drone_scene/quadcopter_pbr.py

Pure Python: nothing here imports Isaac Sim.  Typical use, after ``SimulationContext`` exists::

    pt, launches = rtx_look.path_tracing_samples(args_cli.spp)       # launches == 1
    applied = rtx_look.apply(sim.set_setting, sim.get_setting, "pathtracing", log=say, overrides=pt)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

Value = bool | int | float | str

FILM_ISO = "/rtx/post/tonemap/filmIso"
# the docs' UI range "Samples per Pixel per Frame (1 to 32)".  Not a confirmed hard limit (see the module
# docstring), so it is only used when a caller asks to split a frame: path_tracing_samples(n, max_per_launch=...)
PT_MAX_SPP_PER_LAUNCH = 32
# the limits apply() logs before it sets the path tracer's sample counts; maxSamplesPerLaunch comes from the
# rendering-mode .kit that SimulationContext applied
PT_SAMPLE_KEYS = ("/rtx/pathtracing/spp", "/rtx/pathtracing/totalSpp", "/rtx/pathtracing/clampSpp",
                  "/rtx/pathtracing/maxSamplesPerLaunch")

# --- settings every preset shares: exposure and tone curve live in the post stack, shared by both renderers

_LOOK: dict[str, Value] = {
    # 0 = no uniform fill.  Real-Time legacy only; RT 2.0 ignores it, harmless there
    "/rtx/sceneDb/ambientLightIntensity": 0.0,
    # fixed physical exposure like a UE5 camera in Manual metering; auto exposure would pump as the drone
    # turns from lawn to sky and would not match the calibrated exposure of the offline preview
    "/rtx/post/histogram/enabled": False,
}

# Aces = UE5's filmic curve (both follow the Academy transform: soft shoulder, slight toe), and the same curve
# family as the Mitsuba preview (Narkowicz fit).  The op is an enum in the order of the docs' list: Clamp 0,
# Linear 1, Reinhard 2, Modified Reinhard 3, HejiHableAlu 4, HableUc2 5, Aces 6, Iray 7.  Linear = 1 agrees with
# NVIDIA's digital-human guide ("Linear (Off)", raw USD op = 1); 6 is that order plus a project that inverts the
# op-6 ACES curve to 1 DN.  Isaac Sim's own skill file says 4, which is HejiHableAlu in this order, so the enum is
# not settled: opt-in only (OPTIONAL["tonemap_aces"]); without it Kit keeps its own default tonemapper.
_TONEMAP_ACES: dict[str, Value] = {"/rtx/post/tonemap/op": 6}

# --- RTX Real-Time (RaytracedLighting): the mode quadcopter_pbr.py runs without --pathtracing

_REALTIME: dict[str, Value] = {
    "/rtx/rendermode": "RaytracedLighting",                 # the casing Isaac Lab's SimulationContext enforces
    # our sky has no sun disc (the sun is a separate DistantLight): exactly the case "Approximated IBL"
    # (4) is documented for, fast and free of the IBL sampling artifacts of 0.  Real-Time only
    "/rtx/domeLight/upperLowerStrategy": 4,
    # strategy 4 needs the direct-lighting denoiser; the DL denoiser (ray reconstruction) is it
    "/rtx-transient/dldenoiser/enabled": True,
    "/rtx/directLighting/sampledLighting/enabled": True,
    # NVIDIA's best-practice guide asks for 8 direct-lighting samples for clean sun and sky shadows; the
    # Isaac Lab default is 1.  Cheap here: one dome, one sun, offline frame rate
    "/rtx/directLighting/sampledLighting/samplesPerPixel": 8,
    "/rtx/indirectDiffuse/enabled": True,
    # default 2; a third bounce lights the hedge interiors and shaded undersides the sun never reaches.
    # RT 2.0's default is 3
    "/rtx/indirectDiffuse/maxBounces": 3,
    "/rtx/translucency/enabled": True,                      # the prop-blur discs are translucent (drone_look.py)
    "/rtx/reflections/enabled": True,
    "/rtx/shadows/enabled": True,
}

# --- RTX Real-Time 2.0: cached path tracing.  Experimental: the token is in Isaac Sim 6.x (SimulationApp, the
# --- motion-blur examples), not in 5.1's SimulationApp, so it is untested on Kit 107.  Thin: mode plus _LOOK

_REALTIME2: dict[str, Value] = {
    "/rtx/rendermode": "RealTimePathTracing",
}

# --- RTX Interactive (Path Tracing): offline quality

_PATHTRACING: dict[str, Value] = {
    "/rtx/rendermode": "PathTracing",
    # all three to one number, as Isaac Sim 5.1's SimulationApp does (its default is 64, the --spp default of
    # quadcopter_pbr.py).  A moving drone resets the accumulation every frame, so totalSpp = spp: one launch
    # per frame.  clampSpp is part of what makes more than 32 samples take effect; if the docs' 32 is a hard
    # cap, Kit trims it and the render is the same as with 32.  path_tracing_samples() builds these three
    "/rtx/pathtracing/spp": 64,
    "/rtx/pathtracing/totalSpp": 64,
    "/rtx/pathtracing/clampSpp": 64,
    # default 4; every extra diffuse bounce adds light to shaded interiors (hedges, under the drone).  8 is what
    # quadcopter_pbr.py already uses and the Mitsuba preview's max_depth (NVIDIA's own "Photo" preset, saved in
    # its LookDev templates, uses 12 and totalSpp 600).  maxSpecularAndTransmissionBounces stays at its
    # default of 6: there is no glass
    "/rtx/pathtracing/maxBounces": 8,
    "/rtx/pathtracing/optixDenoiser/enabled": True,         # default; makes a few dozen spp usable
    # on by default.  The clamps keep their default 3200.  Kit 110 calls them "unexposed intensity per sample";
    # how that relates to the sun's intensity (3859) is unverified, so if the sunlit lawn or patio renders
    # dull or flat, raise "/rtx/pathtracing/fireflyFilter/maxIntensityPerSample" (and ...Diffuse) via overrides
    "/rtx/pathtracing/fireflyFilter/enabled": True,
}

SETTINGS: dict[str, dict[str, Value]] = {
    "realtime": {**_REALTIME, **_LOOK},
    "realtime2": {**_REALTIME2, **_LOOK},
    "pathtracing": {**_PATHTRACING, **_LOOK},
}

# --- opt-in groups (apply(optional=...)): camera artefacts and effects that are not physically required

OPTIONAL: dict[str, dict[str, Value]] = {
    "tonemap_aces": _TONEMAP_ACES,
    # anti-aliasing is left to Isaac Lab (RenderCfg(antialiasing_mode="DLAA") sets it through Replicator), so a
    # raw /rtx/post/aa/op here could only disagree with it
    # the photographic look render_flight.py gives the Mitsuba frames (bloom, vignette, colour fringing,
    # grain), at the subtle magnitudes NVIDIA uses for its digital-human setup; the units differ from
    # camera_look.py's, so compare a frame against the preview and adjust
    "lens": {
        "/rtx/post/lensFlares/enabled": True,               # FFT bloom
        "/rtx/post/lensFlares/flareScale": 0.1,
        "/rtx/post/lensFlares/haloFlareWeight": 0.01,
        # parent switch of vignetting and film grain.  It also gates the retro-TV effects (scanlines, scroll bug,
        # splotches); their defaults are unverified, so print get_setting for them on the first Kit run and
        # look for a CRT texture in the first frame
        "/rtx/post/tvNoise/enabled": True,
        "/rtx/post/tvNoise/enableVignetting": True,
        "/rtx/post/tvNoise/vignettingStrength": 0.06,       # best-practice guide 0.06; digital-human guide 0.02
        "/rtx/post/tvNoise/enableFilmGrain": True,
        "/rtx/post/tvNoise/grainAmount": 0.02,              # digital-human guide 0.04 (off there)
        "/rtx/post/chromaticAberration/enabled": True,
        "/rtx/post/chromaticAberration/strengthR": -0.01,   # red and blue fringe in opposite directions
        "/rtx/post/chromaticAberration/strengthG": 0.0,
        "/rtx/post/chromaticAberration/strengthB": 0.01,
    },
    # The post effect of the two Real-Time modes, from motion vectors (reported broken by Replicator
    # rt_subframes > 0: IsaacSim #722).  The path tracer has no such effect: it blurs by rendering sub-steps of
    # the physics, i.e. ``/omni/replicator/captureMotionBlur`` = True and
    # ``/omni/replicator/pathTracedMotionBlurSubSamples`` = N with the physics rate scaled up N times
    # (Isaac Sim's motion_blur_pathtracing.py), which needs Replicator's capture loop, so it is not a group here.
    # exposureFraction is "exposure time fraction in frames (1.0 = one frame duration)": 0.5 is the film
    # convention of a 180 degree shutter (NVIDIA's example uses 1.0).  The same example sets aa/op = 2 (FXAA)
    # with motion blur; the "realtime" preset's DLAA (4) with it is untested
    "motion_blur": {
        "/rtx/post/motionblur/enabled": True,
        "/rtx/post/motionblur/exposureFraction": 0.5,
        "/rtx/post/motionblur/numSamples": 8,                # NVIDIA example value; more is smoother, slower
        "/rtx/post/motionblur/maxBlurDiameterFraction": 0.02,   # of the larger image side; NVIDIA example value
    },
    # Real-Time only: the path tracer is expected to use the USD camera's fStop and focusDistance natively (the
    # hero and garden cameras set them, the onboard cameras do not); this switches on the post effect to match
    "dof": {
        "/rtx/post/dof/enabled": True,
    },
}

# source      set by NVIDIA code read in a clone: Isaac Sim 5.1's SimulationApp, its standalone examples
#             (develop branch) or kit-app-template
# repo        already used by this repo's Isaac Lab presets, tests or scripts, so registered in Isaac Sim 5.x
# docs        path and default in a Kit 107-era NVIDIA docs excerpt (search result, page not read in full)
# example     path named in an NVIDIA guide / forum snippet only
# unverified  spelled from the USD attribute or UI name; not seen as a carb path
PROVENANCE: dict[str, str] = {
    "/rtx/sceneDb/ambientLightIntensity": "repo",
    "/rtx/post/histogram/enabled": "source",             # kit-app-template usd_explorer .kit
    "/rtx/post/tonemap/op": "example",
    "/rtx/domeLight/upperLowerStrategy": "repo",
    "/rtx-transient/dldenoiser/enabled": "repo",
    "/rtx/post/aa/op": "repo",
    "/rtx/directLighting/sampledLighting/enabled": "repo",
    "/rtx/directLighting/sampledLighting/samplesPerPixel": "repo",
    "/rtx/indirectDiffuse/enabled": "repo",
    "/rtx/indirectDiffuse/maxBounces": "docs",
    "/rtx/translucency/enabled": "repo",
    "/rtx/reflections/enabled": "repo",
    "/rtx/shadows/enabled": "repo",
    "/rtx/rendermode": "repo",                              # RealTimePathTracing: see UNVERIFIED_VALUES
    "/rtx/pathtracing/spp": "source",                       # Isaac Sim 5.1.0 SimulationApp._set_render_settings
    "/rtx/pathtracing/totalSpp": "source",
    "/rtx/pathtracing/clampSpp": "source",
    "/rtx/pathtracing/maxBounces": "source",
    "/rtx/pathtracing/optixDenoiser/enabled": "source",
    "/rtx/pathtracing/fireflyFilter/enabled": "docs",
    "/rtx/post/lensFlares/enabled": "example",
    "/rtx/post/lensFlares/flareScale": "example",
    "/rtx/post/lensFlares/haloFlareWeight": "example",
    "/rtx/post/tvNoise/enabled": "example",
    "/rtx/post/tvNoise/enableVignetting": "unverified",
    "/rtx/post/tvNoise/vignettingStrength": "unverified",
    "/rtx/post/tvNoise/enableFilmGrain": "example",
    "/rtx/post/tvNoise/grainAmount": "example",
    "/rtx/post/chromaticAberration/enabled": "unverified",
    "/rtx/post/chromaticAberration/strengthR": "unverified",
    "/rtx/post/chromaticAberration/strengthG": "unverified",
    "/rtx/post/chromaticAberration/strengthB": "unverified",
    "/rtx/post/motionblur/enabled": "example",
    "/rtx/post/motionblur/exposureFraction": "source",      # Isaac Sim motion_blur_raytracing.py
    "/rtx/post/motionblur/numSamples": "source",
    "/rtx/post/motionblur/maxBlurDiameterFraction": "source",
    "/rtx/post/dof/enabled": "unverified",
    FILM_ISO: "example",
}


# values (not paths) that no source confirms for Kit 107
UNVERIFIED_VALUES: dict[str, str] = {
    "/rtx/post/tonemap/op": "6 = Aces: enum order taken from the docs' operator list",
    "/rtx/rendermode": "RealTimePathTracing (realtime2): token from Isaac Sim 6.x (SimulationApp, examples), "
                       "not in 5.1's SimulationApp",
    "/rtx/pathtracing/clampSpp": "semantics undocumented: set to the sample count like SimulationApp; "
                                 "check what Kit does above 32",
}


def unverified_keys() -> list[str]:
    """Keys no source confirmed as a carb path (``PROVENANCE`` tag "unverified")."""
    return sorted(k for k, tag in PROVENANCE.items() if tag == "unverified")


def path_tracing_samples(total_spp: int, max_per_launch: int | None = None) -> tuple[dict[str, int], int]:
    """``(settings, launches)`` for ``total_spp`` samples per output frame.

    By default one launch renders the whole frame: ``spp = totalSpp = clampSpp = total_spp``, ``launches`` is 1,
    like Isaac Sim's ``SimulationApp``.  Pass ``max_per_launch`` (e.g. ``PT_MAX_SPP_PER_LAUNCH``) to split the
    frame instead: ``spp`` is then the samples of one launch and ``totalSpp`` lets the accumulation run
    ``launches`` times, so the caller must render ``launches`` times per frame without changing the scene in
    between (Replicator ``rt_subframes = launches - 1``).  The total rounds up by less than one sample per launch.
    ``clampSpp`` is always the total.  Pass ``settings`` as ``apply(overrides=...)``.
    """
    if total_spp < 1:
        raise ValueError(f"total_spp must be >= 1, got {total_spp}")
    if max_per_launch is not None and max_per_launch < 1:
        raise ValueError(f"max_per_launch must be >= 1, got {max_per_launch}")
    launches = 1 if max_per_launch is None else math.ceil(total_spp / max_per_launch)
    spp = math.ceil(total_spp / launches)
    total = spp * launches
    return {"/rtx/pathtracing/spp": spp, "/rtx/pathtracing/totalSpp": total,
            "/rtx/pathtracing/clampSpp": total}, launches


def apply(set_setting: Callable[[str, Value], object], get_setting: Callable[[str], object], preset: str,
          log: Callable[[str], object] = print, *, optional: Iterable[str] = (),
          overrides: dict[str, Value] | None = None, exposure_ev: float = 0.0) -> dict[str, Value]:
    """Apply ``SETTINGS[preset]``, then the ``OPTIONAL`` groups, then ``overrides``; return {key: value} set.

    ``set_setting`` / ``get_setting`` are ``sim.set_setting`` / ``sim.get_setting``.  A key ``get_setting``
    reports as None is not registered in this Kit: it is warned about and still set (a renderer that loads
    later may pick it up).  For the "pathtracing" preset the sample limits found in Kit (``PT_SAMPLE_KEYS``) are
    logged first, so the first real run shows what ``clampSpp`` and ``maxSamplesPerLaunch`` were.
    ``exposure_ev`` multiplies the current film ISO by 2^ev (one stop per unit) and is relative to what
    ``get_setting`` returns, so apply it once.
    """
    if preset not in SETTINGS:
        raise ValueError(f"unknown preset {preset!r}, choose from {sorted(SETTINGS)}")
    wanted = dict(SETTINGS[preset])
    for group in (optional,) if isinstance(optional, str) else optional:
        if group not in OPTIONAL:
            raise ValueError(f"unknown optional group {group!r}, choose from {sorted(OPTIONAL)}")
        wanted.update(OPTIONAL[group])
    wanted.update(overrides or {})
    if preset == "pathtracing":
        log("[rtx_look] path tracer sample limits before: "
            + ", ".join(f"{k.rsplit('/', 1)[-1]}={get_setting(k)!r}" for k in PT_SAMPLE_KEYS))
    applied: dict[str, Value] = {}
    for key, value in wanted.items():
        if get_setting(key) is None:
            log(f"[rtx_look] WARNING: {key} is not a registered renderer setting in this Kit "
                f"({PROVENANCE.get(key, 'no provenance')}); it may have no effect")
        set_setting(key, value)
        applied[key] = value
    if exposure_ev:
        iso = get_setting(FILM_ISO)
        if isinstance(iso, (int, float)) and not isinstance(iso, bool) and iso > 0:
            applied[FILM_ISO] = float(iso) * 2.0 ** exposure_ev
            set_setting(FILM_ISO, applied[FILM_ISO])
        else:
            log(f"[rtx_look] WARNING: {FILM_ISO} reads {iso!r}; exposure_ev={exposure_ev:g} not applied")
    log(f"[rtx_look] {preset}: {len(applied)} settings applied")
    return applied
