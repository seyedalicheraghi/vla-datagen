"""Photorealistic (PBR) scenery for the Crazyflie quadcopter demo.

``quadcopter_pbr.py`` is ``scripts/demos/quadcopter.py`` (a Crazyflie hovering in place) dressed in a
realistic setting: photo-scanned PBR ground materials and props from NVIDIA's asset server, an HDR sky
whose sun is replaced by a matching directional light, real PBR materials on the drone itself, and path
tracing.  Everything that defines the scene is pure Python + pxr so it can be checked offline, and
``preview.py`` path-traces the same USD stage with Mitsuba 3 (no Isaac Sim needed).

Nothing in this package imports Isaac Sim except ``quadcopter_pbr.py``; nothing on the Isaac side
imports Mitsuba.
"""
