"""Scenario description: terrain + machines with time-varying poses + camera rigs (pure Python)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from .. import kinematics as K
from .. import registry
from .terrain import QuarryTerrain


@dataclass
class ActorState:
    x: float
    y: float
    yaw: float
    ground_z: float                     # terrain height under the root's ground-contact point
    joints: dict[str, float] = field(default_factory=dict)


@dataclass
class Actor:
    name: str                           # prim name in the stage
    machine: str                        # registry name
    state: Callable[[float], ActorState]
    static: bool = False                # pose never changes (spawned with a fixed base)

    @property
    def spec(self):
        return registry.get(self.machine)

    def root_tf(self, t: float) -> tuple[K.Tf, dict[str, float]]:
        """World transform of the root link and the full (follower-resolved) joint map."""
        s = self.state(t)
        spec = self.spec
        q = K.resolve_followers(spec, {**K.init_q(spec), **s.joints})
        return K.Tf.planar(s.x, s.y, s.ground_z + K.root_height(spec), s.yaw), q


@dataclass
class CameraRig:
    name: str
    actor: str | None                   # attached to this actor's link, or None = fixed in the world
    link: str = ""
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)   # looks along +x; pitch > 0 looks down
    width: int = 1280
    height: int = 720
    focal_length: float = 10.0          # mm, on a 20.955 mm horizontal aperture (~93° HFOV)
    horizontal_aperture: float = 20.955


@dataclass
class Lighting:
    sun_azimuth_deg: float = 135.0      # direction the sunlight COMES FROM, CCW from +x
    sun_elevation_deg: float = 62.0
    sun_intensity: float = 3000.0
    sun_color: tuple[float, float, float] = (1.0, 0.97, 0.92)
    sky_intensity: float = 1000.0
    sky_hdr: str = "Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"   # under ISAAC_NUCLEUS_DIR/Materials/Textures
    sky_color: tuple[float, float, float] = (0.78, 0.84, 0.92)               # fallback when no HDR


@dataclass
class Scenario:
    name: str
    description: str
    duration: float
    fps: int
    terrain: QuarryTerrain
    actors: list[Actor]
    cameras: list[CameraRig]
    lighting: Lighting = field(default_factory=Lighting)
    local_rect: tuple[float, float, float, float] = (-45.0, 85.0, -30.0, 35.0)   # detailed terrain
    near_rect: tuple[float, float, float, float] | None = None   # densest textures (along the camera path)
    far_rect: tuple[float, float, float, float] = (-900.0, 900.0, -900.0, 900.0)
    rocks: int = 160
    keep_clear: list[tuple[float, float, float, float]] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return int(round(self.duration * self.fps))

    def actor(self, name: str) -> Actor:
        return next(a for a in self.actors if a.name == name)

    def camera_tf(self, cam: CameraRig, t: float) -> K.Tf:
        mount = K.Tf.from_xyz_rpy(cam.xyz, cam.rpy)
        if cam.actor is None:
            return mount
        base, q = self.actor(cam.actor).root_tf(t)
        link = K.link_transforms(self.actor(cam.actor).spec, q)[cam.link]
        return base @ link @ mount


def sun_direction(lighting: Lighting) -> tuple[float, float, float]:
    """Unit vector the sunlight travels along (from the sun toward the ground)."""
    az, el = math.radians(lighting.sun_azimuth_deg), math.radians(lighting.sun_elevation_deg)
    return (-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), -math.sin(el))


# ---------------------------------------------------------------------------
# Motion helpers
# ---------------------------------------------------------------------------

@dataclass
class CruiseThenStop:
    """1-D motion: constant ``v0`` until ``t_brake``, smooth (cosine) braking over
    ``brake_time``, then standing still.  Optional standstill ``t_start`` at the beginning."""
    v0: float
    t_brake: float
    brake_time: float
    t_start: float = 0.0

    def speed(self, t: float) -> float:
        if t < self.t_start:
            return 0.0
        if t < self.t_brake:
            return self.v0
        tau = t - self.t_brake
        if tau < self.brake_time:
            return self.v0 * 0.5 * (1.0 + math.cos(math.pi * tau / self.brake_time))
        return 0.0

    def distance(self, t: float) -> float:
        if t <= self.t_start:
            return 0.0
        cruise_end = min(t, self.t_brake)
        s = self.v0 * (cruise_end - self.t_start)
        if t > self.t_brake:
            tau = min(t - self.t_brake, self.brake_time)
            s += 0.5 * self.v0 * (tau + self.brake_time / math.pi * math.sin(math.pi * tau / self.brake_time))
        return s

    @property
    def total_distance(self) -> float:
        return self.v0 * (self.t_brake - self.t_start) + 0.5 * self.v0 * self.brake_time
