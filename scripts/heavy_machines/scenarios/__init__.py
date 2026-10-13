"""Scripted multi-machine scenarios (terrain + machines + cameras) rendered in Isaac Sim.

Everything that defines a scenario (terrain heights, baked textures, machine
poses over time, camera rigs) is pure Python/numpy so it can be previewed and
tested offline; ``render_scenario.py`` only assembles it in Isaac Sim and
records the cameras.
"""
