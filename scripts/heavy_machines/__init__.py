"""Procedurally described heavy construction machines for Isaac Lab.

Every machine is a pure-Python :class:`~heavy_machines.spec.MachineSpec`
(links built from boxes / cylinders / spheres plus joints). From one spec we
generate the URDF that Isaac Sim imports, a JSON sidecar describing the action
and state layout, offline validation, and preview renders. The Isaac Lab
environment in ``machine_env.py`` drives any registered machine.

Nothing in this package imports Isaac Sim except ``machine_env.py``,
``teleop_machine.py``, ``scripted_demo.py`` and ``showcase.py``.
"""


def safe_console() -> None:
    """Never crash on a character the console cannot encode.

    On Windows a redirected stdout (``> log.txt``, ``| Tee-Object``) uses the ANSI code page, which
    cannot encode characters such as an arrow or a machine name with an umlaut.  Unencodable characters
    print as ``?`` instead of raising; UTF-8 consoles (Linux) are unaffected.
    """
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):        # replaced or detached stream
            pass
