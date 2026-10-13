"""ADT approaches a parked HEX at the loading spot (recreates the OpenArt clip
"approaches the HEX at the loading spot", Wan 3.0, 1280x720, 30 fps, 10 s).

Clip description it is built from:
    forward camera ~4 m high on the front of an articulated haul truck in an
    open-pit quarry at clear midday · the truck drives slowly and perfectly
    straight along a loading lane · passes through a gap between two low earth
    berms · then a raised earth bench wall runs along its left side and the
    lane narrows · ahead, slightly left, a yellow tracked excavator stands
    parked on the bench at the loading position, arm raised, growing larger ·
    the truck slows and stops smoothly at the loading spot beside it, at a
    safe distance · light sandy ground, pale sky, steady camera.

World frame: +x = driving direction, +y = left, the ADT's front axle stops at
the origin.  All numbers are in ``LAYOUT`` so the framing can be matched to
the real clip once its frames are available.
"""

from __future__ import annotations

import math

from .. import registry
from .scenario import Actor, ActorState, CameraRig, CruiseThenStop, Lighting, Scenario
from .terrain import QuarryTerrain

LAYOUT = {
    # truck motion: 3 m/s (11 km/h, slow for a haul truck), brake from t=5 s, stopped at 8.5 s
    "truck_speed": 3.0,
    "truck_brake_at": 5.0,
    "truck_brake_time": 3.5,
    # excavator parked on the loading bench, ahead-left of the stop point
    "hex_xy": (13.0, 9.0),
    "hex_yaw": 0.0,                  # tracks parallel to the lane
    "hex_joints": {"swing_joint": -2.0, "boom_joint": 0.75, "stick_joint": -0.40, "bucket_joint": 1.20},
    # forward video camera on the ADT: front of the cab roof, 4.0 m above the ground
    "cam_height_agl": 4.0,
    "cam_x": -0.20,
    "cam_pitch": 0.20,               # 11.5° down
}

DURATION, FPS = 10.0, 30


def build() -> Scenario:
    L = LAYOUT
    adt = registry.get("articulated_dump_truck")
    motion = CruiseThenStop(L["truck_speed"], L["truck_brake_at"], L["truck_brake_time"])
    start_x = -motion.total_distance
    wheels = [(j.name, 1.0 if j.axis[1] > 0 else -1.0, j.wheel_radius)
              for j in adt.joints if j.role == "wheel"]

    def truck_state(t: float) -> ActorState:
        s = motion.distance(t)
        joints = {name: sign * ((s / r + math.pi) % (2 * math.pi) - math.pi) for name, sign, r in wheels}
        return ActorState(start_x + s, 0.0, 0.0, 0.0, joints)

    hx, hy = L["hex_xy"]
    terrain = QuarryTerrain(flat_pads=[(hx - 4.0, hx + 4.0, hy - 2.5, hy + 2.5)])
    bench_z = terrain.bench_top()

    def hex_state(t: float) -> ActorState:
        return ActorState(hx, hy, L["hex_yaw"], bench_z, dict(L["hex_joints"]))

    cam_z = L["cam_height_agl"] - 0.805          # ADT root (front axle) is 0.805 m above the ground
    return Scenario(
        name="adt_approach_hex",
        description=__doc__.strip().splitlines()[0],
        duration=DURATION,
        fps=FPS,
        terrain=terrain,
        actors=[
            Actor("ADT", "articulated_dump_truck", truck_state),
            Actor("HEX", "hydraulic_excavator", hex_state, static=True),
        ],
        cameras=[
            # the clip's camera
            CameraRig("front", "ADT", "front_frame", (L["cam_x"], 0.0, cam_z), (0.0, L["cam_pitch"], 0.0)),
            # third-person chase view (behind-right of the truck, looking at it and the excavator)
            CameraRig("chase", "ADT", "front_frame", (-17.0, -10.0, 6.5 - 0.805),
                      (0.0, 0.27, math.atan2(10.0, 13.0))),
        ],
        # midday sun from behind-right: the bench wall facing the lane is lit, shadows fall ahead-left
        lighting=Lighting(sun_azimuth_deg=-120.0, sun_elevation_deg=64.0),
        keep_clear=[(hx - 5.0, hx + 9.0, hy - 4.0, hy + 4.0)],
    )
