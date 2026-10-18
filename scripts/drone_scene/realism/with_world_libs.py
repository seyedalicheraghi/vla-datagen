"""Run a Python script with the map tools first on the import path, the way the world builder runs (runner.py):
the packages in ~/.cache/drone_scene/world_libs, installed by the first --route straight run.

    ./isaaclab.sh -p scripts/drone_scene/realism/with_world_libs.py scripts/drone_scene/test/test_world.py
"""

import os
import runpy
import sys

libs = os.environ.get("DRONE_WORLD_LIBS") or os.path.join(os.path.expanduser("~"), ".cache", "drone_scene", "world_libs")
if not os.path.isdir(libs):
    sys.exit(f"no map tools in {libs}: run the straight route once (it installs them), or set DRONE_WORLD_LIBS")
sys.path.insert(0, libs)
if len(sys.argv) < 2:
    sys.exit(__doc__)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
