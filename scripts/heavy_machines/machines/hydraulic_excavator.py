"""Hydraulic excavator (HEX), 20 t class — modelled on a CAT 320 / Komatsu PC210.

Reference numbers (typical 20 t crawler excavator datasheet):
    operating mass 20.5 t · track length 4.45 m · gauge 2.38 m · 600 mm shoes
    (width 2.98 m) · ground clearance 0.47 m · tail-swing radius 2.83 m ·
    cab height 2.96 m · counterweight clearance 1.05 m · boom 5.7 m ·
    stick 2.9 m · 1.0 m³ bucket · swing 11 rpm · travel 5.4 km/h ·
    shipping length ≈ 9.5 m.

Init pose = shipping/transport pose: boom lowered, stick folded under the
boom, bucket curled clear of the ground.

Root frame: centre of the undercarriage at track mid-height.
"""

from __future__ import annotations

from ..spec import (Camera, Drive, Joint, Link, MachineSpec, Tool, beam, box, cyl,
                    sprocket_link, track_shapes)

# Undercarriage
TRACK_L, TRACK_H, SHOE_W, GAUGE = 4.45, 0.95, 0.60, 2.38
SWING_Z = 0.745                 # top of the swing bearing above the root origin
# Boom geometry in the boom frame (origin = foot pin, x forward when q = 0)
BOOM_FOOT = (1.10, -0.09, 0.75)  # in the upper-structure frame
BOOM_KNEE = (2.75, 1.55)
BOOM_TIP = (5.60, 1.05)          # stick pin; 5.70 m from the foot pin
STICK_LEN = 2.90
STICK_PITCH = 1.37               # stick hangs 78.5° below the boom frame x at q = 0


def build() -> MachineSpec:
    y_t = GAUGE / 2
    undercarriage = Link("undercarriage", 8000.0, [
        *track_shapes(TRACK_L, TRACK_H, SHOE_W, +y_t, 0.0),
        *track_shapes(TRACK_L, TRACK_H, SHOE_W, -y_t, 0.0),
        box(2.2, GAUGE - SHOE_W, 0.50, at=(0.0, 0.0, 0.245), mat="dark"),      # car body
        cyl(0.90, 0.25, at=(0.0, 0.0, 0.62), mat="steel"),                       # swing bearing
    ])

    # Upper structure: z = 0 is the top of the swing bearing.
    house = Link("upper_structure", 9000.0, [
        box(3.50, 2.54, 0.35, at=(-0.45, 0.0, 0.175)),                               # deck
        box(1.80, 2.54, 1.00, at=(-1.40, 0.0, 0.85)),                                # engine hood
        box(0.63, 2.54, 1.37, at=(-2.515, 0.0, 0.515), mat="dark"),                 # counterweight
        box(1.50, 0.82, 0.80, at=(0.25, -0.86, 0.75)),                               # tank / toolbox
        box(1.70, 1.00, 1.39, at=(0.40, 0.77, 1.045)),                               # cab
        box(0.04, 0.90, 0.95, at=(1.26, 0.77, 1.17), mat="glass", collision=False),  # windscreen
        box(1.30, 0.04, 0.80, at=(0.45, 1.28, 1.20), mat="glass", collision=False),  # side window
        box(0.70, 0.60, 0.60, at=(0.90, -0.09, 0.45), mat="dark"),                   # boom foot bracket
        cyl(0.08, 0.5, at=(-1.9, -0.9, 1.45), mat="accent", collision=False),       # exhaust stack
    ])

    boom = Link("boom", 1800.0, [
        beam((0.0, 0.0), BOOM_KNEE, 0.55, 0.70),
        beam(BOOM_KNEE, BOOM_TIP, 0.55, 0.60),
        cyl(0.35, 0.56, at=(BOOM_KNEE[0], 0.0, BOOM_KNEE[1]), axis="y"),           # knee
        cyl(0.22, 0.70, axis="y", mat="steel", collision=False),                     # foot pin boss
    ])

    stick = Link("stick", 900.0, [
        beam((-0.45, 0.0), (STICK_LEN, 0.0), 0.45, 0.55),
        cyl(0.18, 0.62, axis="y", mat="steel", collision=False),                    # pin boss
        cyl(0.14, 0.50, at=(STICK_LEN, 0.0, 0.0), axis="y", mat="steel", collision=False),
    ])

    # Bucket frame: origin at the bucket pin, +x toward the cutting edge,
    # +z out of the opening (forward when the stick hangs straight down).
    bucket = Link("bucket", 750.0, [
        box(1.20, 1.20, 0.06, at=(0.60, 0.0, -0.25), mat="dark"),                    # back wall
        box(0.06, 1.20, 0.70, at=(1.20, 0.0, 0.10), mat="dark"),                     # floor / lip
        box(0.06, 1.20, 0.30, at=(0.05, 0.0, -0.10), mat="dark"),                    # top plate
        box(1.26, 0.05, 0.72, at=(0.62, 0.60, 0.08), mat="dark"),                    # left side
        box(1.26, 0.05, 0.72, at=(0.62, -0.60, 0.08), mat="dark"),                   # right side
        *[box(0.06, 0.10, 0.16, at=(1.22, y, 0.52), mat="steel", collision=False)   # teeth
          for y in (-0.45, -0.15, 0.15, 0.45)],
    ])

    sprocket_x = -(TRACK_L - TRACK_H) / 2
    links = [undercarriage, house, boom, stick, bucket,
             sprocket_link("left_sprocket", 0.475, 0.64, 25.0),
             sprocket_link("right_sprocket", 0.475, 0.64, 25.0)]

    joints = [
        Joint("swing_joint", "continuous", "undercarriage", "upper_structure",
              xyz=(0.0, 0.0, SWING_Z), axis=(0, 0, 1), velocity=1.15, role="tool"),
        Joint("boom_joint", "revolute", "upper_structure", "boom",
              xyz=BOOM_FOOT, axis=(0, -1, 0), lower=-0.90, upper=1.00, init=-0.45,
              velocity=0.55, role="tool"),
        Joint("stick_joint", "revolute", "boom", "stick",
              xyz=(BOOM_TIP[0], 0.0, BOOM_TIP[1]), rpy=(0.0, STICK_PITCH, 0.0),
              axis=(0, -1, 0), lower=-1.35, upper=0.90, init=-1.35, velocity=0.65, role="tool"),
        Joint("bucket_joint", "revolute", "stick", "bucket",
              xyz=(STICK_LEN, 0.0, 0.0), axis=(0, -1, 0), lower=-1.20, upper=1.75, init=-0.80,
              velocity=1.0, role="tool"),
        Joint("left_sprocket_joint", "continuous", "undercarriage", "left_sprocket",
              xyz=(sprocket_x, y_t, 0.0), axis=(0, 1, 0), velocity=20.0, role="roll_left",
              wheel_radius=TRACK_H / 2),
        Joint("right_sprocket_joint", "continuous", "undercarriage", "right_sprocket",
              xyz=(sprocket_x, -y_t, 0.0), axis=(0, 1, 0), velocity=20.0, role="roll_right",
              wheel_radius=TRACK_H / 2),
    ]

    return MachineSpec(
        name="hydraulic_excavator",
        abbrev="HEX",
        display_name="Hydraulic Excavator",
        category="excavation",
        reference="20 t crawler excavator (CAT 320 / Komatsu PC210 class)",
        description="Tracked excavator with 360° swing, two-piece boom/stick arm and a 1 m³ bucket.",
        root="undercarriage",
        links=links,
        joints=joints,
        drive=Drive(type="skid", max_speed=1.5, max_yaw_rate=0.5, track_width=GAUGE),
        tools=[
            Tool("swing_joint", "swing", positive="left", negative="right"),
            Tool("boom_joint", "boom", positive="raise", negative="lower"),
            Tool("stick_joint", "stick", positive="out", negative="in"),
            Tool("bucket_joint", "bucket", positive="curl", negative="dump"),
        ],
        cameras=[
            Camera("front", "upper_structure", (1.15, 0.77, 1.45), (0.0, 0.35, 0.0)),
            Camera("left", "upper_structure", (0.2, 1.25, 1.80), (0.0, 0.40, 0.55)),
            Camera("right", "upper_structure", (0.2, -1.25, 1.30), (0.0, 0.40, -0.55)),
        ],
        envelope=(9.53, 2.98, 2.96),
        operating_mass=20500.0,
        ground_contact=["undercarriage"],
        # The stick folds tight under the boom in the shipping pose and the
        # bucket tucks in front of the tracks; real machines do the same.
        allowed_overlaps=[("boom", "bucket")],
        notes="Bucket curl is not auto-verified (no up/out/left semantics); see render.",
    )
