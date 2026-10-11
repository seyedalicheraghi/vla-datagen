"""Procedurally described heavy construction machines for Isaac Lab.

Every machine is a pure-Python :class:`~heavy_machines.spec.MachineSpec`
(links built from boxes / cylinders / spheres plus joints). From one spec we
generate the URDF that Isaac Sim imports, a JSON sidecar describing the action
and state layout, offline validation, and preview renders. The Isaac Lab
environment in ``machine_env.py`` drives any registered machine.

Nothing in this package imports Isaac Sim except ``machine_env.py``,
``teleop_machine.py``, ``scripted_demo.py`` and ``showcase.py``.
"""
