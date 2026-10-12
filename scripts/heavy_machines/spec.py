"""Machine description format (pure Python, no Isaac imports).

Frames and conventions
----------------------
* Every frame is right-handed: +x forward, +y left, +z up (URDF/ROS).
* The root link's origin can sit anywhere on the machine. The environment
  computes how high it must be spawned so the lowest ground-contact shape
  touches z = 0 (see :func:`heavy_machines.kinematics.root_height`).
* Revolute joints rotate the child about ``axis`` (right-hand rule). Rotating
  about +y by a positive angle tips the child's +x DOWN, so a joint whose
  positive direction should RAISE something pointing forward uses
  ``axis=(0, -1, 0)``.
* Joint ``init`` is the position used on reset and for the "transport pose"
  that ``envelope`` and the ground-contact checks refer to.

Roles
-----
Each non-fixed joint has a role that tells the environment how to drive it:

``tool``         Operator-controlled; appears in the action vector via ``tools``.
``wheel``        Spun cosmetically at v / wheel_radius.
``roll_left``    Left side of a skid-steered / tracked machine (spun at v_left).
``roll_right``   Right side of a skid-steered / tracked machine (spun at v_right).
``steer``        Ackermann steering joint (revolute about z).
``articulation`` Centre hinge of an articulated machine (revolute about z).
``follower``     Set every step to ``k * q_source + c`` (e.g. a crane hook that
                 must hang vertically while the boom luffs).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal, Union

Vec3 = tuple[float, float, float]
RGBA = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Box:
    size: Vec3


@dataclass(frozen=True)
class Cylinder:
    radius: float
    length: float  # along the shape's local z axis (URDF convention)


@dataclass(frozen=True)
class Sphere:
    radius: float


Geometry = Union[Box, Cylinder, Sphere]


@dataclass
class Shape:
    geom: Geometry
    xyz: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)
    material: str = "body"
    collision: bool = True
    visual: bool = True


@dataclass
class Link:
    name: str
    mass: float
    shapes: list[Shape] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Joints
# ---------------------------------------------------------------------------

JointType = Literal["revolute", "continuous", "prismatic", "fixed"]
JointRole = Literal["fixed", "tool", "wheel", "roll_left", "roll_right",
                    "steer", "articulation", "follower"]

ROLLING_ROLES = ("wheel", "roll_left", "roll_right")


@dataclass
class Joint:
    name: str
    type: JointType
    parent: str
    child: str
    xyz: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)
    axis: Vec3 = (0.0, 0.0, 1.0)
    lower: float = 0.0
    upper: float = 0.0
    velocity: float = 1.0            # max speed (rad/s or m/s); tool command scale
    effort: float = 1.0e7
    role: JointRole = "fixed"
    init: float = 0.0
    wheel_radius: float = 0.0        # rolling radius for wheel / roll_* roles
    follows: tuple[str, float, float] | None = None  # (source joint, k, c)


# ---------------------------------------------------------------------------
# Drive, tools, cameras
# ---------------------------------------------------------------------------

@dataclass
class Drive:
    """How the base moves.  action[0] = drive in [-1, 1], action[1] = turn in [-1, 1].

    turn > 0 always means "turn left" (counter-clockwise seen from above).
    """
    type: Literal["skid", "ackermann", "articulated"]
    max_speed: float                      # m/s forward
    max_reverse_speed: float = 0.0        # m/s; 0 → same as max_speed
    # skid / tracked
    max_yaw_rate: float = 0.5             # rad/s
    track_width: float = 0.0              # m, centre-to-centre of left/right rolling elements
    # ackermann
    wheelbase: float = 0.0                # m, fixed axle → steered axle
    max_steer: float = 0.6                # rad, virtual centre-wheel angle
    steer_rate: float = 0.8               # rad/s
    fixed_axle_x: float = 0.0             # x of the non-steered axle (tandem centre) in root frame
    steer_axle: Literal["front", "rear"] = "front"
    # articulated
    articulation_joint: str = ""
    front_length: float = 0.0             # m, hinge → front axle
    rear_length: float = 0.0              # m, hinge → rear axle (tandem centre)
    articulation_rate: float = 0.5        # rad/s
    # wheeled machines: steering / articulation is limited at speed so the
    # lateral acceleration never exceeds this (real machines roll over first)
    max_lat_accel: float = 3.0            # m/s²

    @property
    def reverse_speed(self) -> float:
        return self.max_reverse_speed or self.max_speed


@dataclass
class Tool:
    """One operator-controlled degree of freedom (one action channel)."""
    joint: str
    label: str
    positive: str = ""   # what +1 does, e.g. "raise", "extend", "left", "curl"
    negative: str = ""


@dataclass
class Camera:
    """Camera mounted on a link.  Looks along +x of (link frame ∘ xyz/rpy).

    pitch > 0 looks down.  Every machine must define cameras named
    ``front``, ``left`` and ``right`` (same layout as the forklift dataset).
    """
    name: Literal["front", "left", "right"]
    link: str
    xyz: Vec3
    rpy: Vec3 = (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Machine
# ---------------------------------------------------------------------------

DEFAULT_MATERIALS: dict[str, RGBA] = {
    "body":     (0.96, 0.72, 0.05, 1.0),   # construction yellow
    "accent":   (0.12, 0.12, 0.12, 1.0),   # black trim
    "dark":     (0.20, 0.20, 0.20, 1.0),
    "steel":    (0.50, 0.50, 0.52, 1.0),
    "rubber":   (0.06, 0.06, 0.06, 1.0),
    "glass":    (0.25, 0.38, 0.48, 1.0),
    "hub":      (0.75, 0.62, 0.10, 1.0),
    "red":      (0.75, 0.10, 0.08, 1.0),
    "orange":   (0.95, 0.45, 0.05, 1.0),
    "white":    (0.90, 0.90, 0.88, 1.0),
    "rust":     (0.45, 0.25, 0.12, 1.0),
}

CATEGORIES = ("excavation", "loading", "hauling", "lifting", "grading",
              "compaction", "paving", "dozing")


@dataclass
class MachineSpec:
    name: str                    # registry key, snake_case
    abbrev: str                  # industry abbreviation, e.g. "HEX"
    display_name: str
    category: str
    reference: str               # real machine class it is modelled on
    description: str
    root: str
    links: list[Link]
    joints: list[Joint]
    drive: Drive
    tools: list[Tool]
    cameras: list[Camera]
    envelope: Vec3               # expected L, W, H (m) in the init pose
    operating_mass: float        # kg
    materials: dict[str, RGBA] = field(default_factory=lambda: dict(DEFAULT_MATERIALS))
    ground_contact: list[str] = field(default_factory=list)  # extra links that touch the ground
    allowed_overlaps: list[tuple[str, str]] = field(default_factory=list)
    notes: str = ""

    # --- lookups ---------------------------------------------------------
    def link(self, name: str) -> Link:
        for lk in self.links:
            if lk.name == name:
                return lk
        raise KeyError(f"{self.name}: no link {name!r}")

    def joint(self, name: str) -> Joint:
        for jt in self.joints:
            if jt.name == name:
                return jt
        raise KeyError(f"{self.name}: no joint {name!r}")

    def joints_with_role(self, *roles: str) -> list[Joint]:
        return [j for j in self.joints if j.role in roles]

    @property
    def movable_joints(self) -> list[Joint]:
        return [j for j in self.joints if j.type != "fixed"]

    @property
    def action_dim(self) -> int:
        return 2 + len(self.tools)

    @property
    def action_labels(self) -> list[str]:
        return ["drive", "turn"] + [t.label for t in self.tools]

    @property
    def state_labels(self) -> list[str]:
        steer = {"skid": "unused", "ackermann": "steer_angle",
                 "articulated": "articulation_angle"}[self.drive.type]
        return (["x", "y", "yaw", "speed", "yaw_rate", steer]
                + [f"{t.label}_pos" for t in self.tools])

    @property
    def total_mass(self) -> float:
        return sum(lk.mass for lk in self.links)


# ---------------------------------------------------------------------------
# Shape / link / joint helpers (keep machine files short and readable)
# ---------------------------------------------------------------------------

_AXIS_RPY = {"x": (0.0, math.pi / 2, 0.0), "y": (math.pi / 2, 0.0, 0.0),
             "z": (0.0, 0.0, 0.0)}


def box(sx: float, sy: float, sz: float, at: Vec3 = (0.0, 0.0, 0.0),
        rpy: Vec3 = (0.0, 0.0, 0.0), mat: str = "body",
        collision: bool = True, visual: bool = True) -> Shape:
    return Shape(Box((sx, sy, sz)), tuple(at), tuple(rpy), mat, collision, visual)


def cyl(radius: float, length: float, at: Vec3 = (0.0, 0.0, 0.0), axis: str = "z",
        rpy: Vec3 | None = None, mat: str = "body",
        collision: bool = True, visual: bool = True) -> Shape:
    """Cylinder whose length runs along ``axis`` ('x', 'y' or 'z') unless ``rpy`` is given."""
    return Shape(Cylinder(radius, length), tuple(at),
                 tuple(rpy) if rpy is not None else _AXIS_RPY[axis],
                 mat, collision, visual)


def sphere(radius: float, at: Vec3 = (0.0, 0.0, 0.0), mat: str = "body",
           collision: bool = True, visual: bool = True) -> Shape:
    return Shape(Sphere(radius), tuple(at), (0.0, 0.0, 0.0), mat, collision, visual)


def beam(p0: tuple[float, float], p1: tuple[float, float], width: float, depth: float,
         y: float = 0.0, mat: str = "body", collision: bool = True) -> Shape:
    """Box spanning p0 → p1 in the link's x-z plane (side view), centred at ``y``.

    ``width`` is the size along y, ``depth`` the thickness perpendicular to the
    beam in the x-z plane.  Handy for booms, sticks and lift arms.
    """
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dz)
    pitch = -math.atan2(dz, dx)          # +x tilted up by atan2(dz, dx)
    centre = ((p0[0] + p1[0]) / 2, y, (p0[1] + p1[1]) / 2)
    return Shape(Box((length, width, depth)), centre, (0.0, pitch, 0.0), mat, collision, True)


def wheel_link(name: str, radius: float, width: float, mass: float,
               tire: str = "rubber", hub: str = "hub") -> Link:
    """Tyre + rim + a spoke bar so cosmetic rotation is visible. Spins about y."""
    return Link(name, mass, [
        cyl(radius, width, axis="y", mat=tire),
        cyl(radius * 0.55, width * 1.04, axis="y", mat=hub, collision=False),
        box(radius * 1.0, width * 1.08, radius * 0.16, mat="accent", collision=False),
    ])


def wheel_joint(name: str, parent: str, child: str, xyz: Vec3, radius: float,
                role: str = "wheel") -> Joint:
    return Joint(name, "continuous", parent, child, xyz=tuple(xyz), axis=(0.0, 1.0, 0.0),
                 velocity=60.0, role=role, wheel_radius=radius)


def track_shapes(length: float, height: float, width: float, y: float, z: float,
                 mat: str = "rubber", frame_mat: str = "dark") -> list[Shape]:
    """A crawler track (belt + rounded ends + inner frame) centred at (0, y, z).

    ``length`` is the overall length including the rounded ends.  All shapes
    collide; the belt bottom sits at ``z - height / 2``.
    """
    r = height / 2.0
    straight = max(length - height, 0.01)
    return [
        box(straight, width, height, at=(0.0, y, z), mat=mat),
        cyl(r, width, at=(straight / 2, y, z), axis="y", mat=mat),
        cyl(r, width, at=(-straight / 2, y, z), axis="y", mat=mat),
        box(straight * 0.9, width * 0.6, height * 0.55, at=(0.0, y, z),
            mat=frame_mat, collision=False),
    ]


def sprocket_link(name: str, radius: float, width: float, mass: float = 150.0) -> Link:
    """Visual drive sprocket for a track (does not collide; the track does)."""
    return Link(name, mass, [
        cyl(radius, width, axis="y", mat="dark", collision=False),
        box(radius * 1.9, width * 1.05, radius * 0.25, mat="steel", collision=False),
        box(radius * 0.25, width * 1.05, radius * 1.9, mat="steel", collision=False),
    ])


_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def valid_name(name: str) -> bool:
    return bool(_NAME_RE.match(name))
