#!/usr/bin/env bash
# One-paste setup + render on native Linux with Isaac Lab - no git needed.
#
#   curl -fsSL https://raw.githubusercontent.com/seyedalicheraghi/vla-datagen/<commit>/scripts/heavy_machines/setup_linux.sh -o /tmp/hm_setup.sh && bash /tmp/hm_setup.sh
#
# Downloads scripts/heavy_machines, scripts/drone_scene and assets/heavy_machines from the branch into
# ~/heavy_machines (your Isaac Lab clone is not touched), finds the Python that has Isaac Sim + Isaac Lab,
# and renders.  Afterwards:
#   ~/heavy_machines/render.sh   the truck/excavator video (args go to render_scenario.py)
#   ~/heavy_machines/drone.sh    the Crazyflie demo in a PBR backyard (args go to quadcopter_pbr.py)
# Both update themselves the same way first.
#
#   ISAACLAB=/path/to/IsaacLab   your Isaac Lab clone (default ~/Projects/IsaacLab; used to find its Python)
#   HM_DEST=/path                 install folder (default ~/heavy_machines)
#   --no-render                   update only
#   --drone                       run the drone demo instead of the truck/excavator render

set -uo pipefail
ISAACLAB="${ISAACLAB:-$HOME/Projects/IsaacLab}"
DEST="${HM_DEST:-$HOME/heavy_machines}"
BRANCH="claude/review-last-request-I6mXJ"
say() { printf '\033[36m[setup] %s\033[0m\n' "$*"; }
stop() { say "STOPPED: $*"; say "Nothing was deleted."; exit 1; }

NO_RENDER=0
DRONE=0
ARGS=()
for a in "$@"; do
    case "$a" in
        --no-render) NO_RENDER=1 ;;
        --drone) DRONE=1 ;;
        *) ARGS+=("$a") ;;
    esac
done

command -v python3 >/dev/null || stop "python3 not found"
mkdir -p "$DEST" || stop "cannot create $DEST"

# 1. latest heavy-machine code from GitHub (skipped when already current)
python3 - "$DEST" "$BRANCH" <<'PY' || stop "download failed (see the error above)"
import io, json, os, sys, urllib.request, zipfile
dest, branch = sys.argv[1], sys.argv[2]
repo = "seyedalicheraghi/vla-datagen"
stamp = os.path.join(dest, "scripts", "heavy_machines", ".installed_commit")
hdr = {"User-Agent": "heavy-machines-setup"}
sha = None
try:
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/commits/{branch}", headers=hdr)
    sha = json.load(urllib.request.urlopen(req, timeout=30))["sha"]
except Exception as exc:  # noqa: BLE001
    print(f"[setup] could not ask GitHub for the latest version ({exc}) - downloading anyway")
current = open(stamp).read().strip() if os.path.exists(stamp) else None
if sha and sha == current:
    print(f"[setup] already up to date ({sha[:7]})")
    sys.exit(0)
print("[setup] downloading the latest version from GitHub (about 40 MB) ...", flush=True)
req = urllib.request.Request(f"https://codeload.github.com/{repo}/zip/refs/heads/{branch}", headers=hdr)
data = urllib.request.urlopen(req, timeout=900).read()
n = 0
with zipfile.ZipFile(io.BytesIO(data)) as z:
    for info in z.infolist():
        rel = info.filename.split("/", 1)[1] if "/" in info.filename else ""
        if info.is_dir() or not rel.startswith(("scripts/heavy_machines/", "scripts/drone_scene/",
                                                 "assets/heavy_machines/")):
            continue
        path = os.path.join(dest, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # new file + rename: bash may be executing the old copy of this very script
        with open(path + ".part", "wb") as f:
            f.write(z.read(info))
        os.replace(path + ".part", path)
        n += 1
if n < 50:
    sys.exit(f"[setup] the download looks incomplete ({n} files)")
if sha:
    with open(stamp, "w") as f:
        f.write(sha + "\n")
print(f"[setup] installed {n} files ({sha[:7] if sha else 'unknown version'}) into {dest}")
PY
chmod +x "$DEST/scripts/heavy_machines/setup_linux.sh" 2>/dev/null
printf '#!/usr/bin/env bash\n# update (no git) and render; extra arguments go to render_scenario.py\nISAACLAB="${ISAACLAB:-%s}" HM_DEST="%s" exec bash "%s/scripts/heavy_machines/setup_linux.sh" "$@"\n' \
    "$ISAACLAB" "$DEST" "$DEST" > "$DEST/render.sh" && chmod +x "$DEST/render.sh"
printf '#!/usr/bin/env bash\n# update (no git) and run the Crazyflie PBR demo; extra arguments go to quadcopter_pbr.py\nISAACLAB="${ISAACLAB:-%s}" HM_DEST="%s" exec bash "%s/scripts/heavy_machines/setup_linux.sh" --drone "$@"\n' \
    "$ISAACLAB" "$DEST" "$DEST" > "$DEST/drone.sh" && chmod +x "$DEST/drone.sh"
[ "$NO_RENDER" = 1 ] && exit 0

# 2. the Python that has Isaac Sim and Isaac Lab (what ./isaaclab.sh -p would run, but checked)
has_isaac() {   # $1 = python; true if it can import Isaac Lab and knows Isaac Sim
    [ -x "$1" ] || return 1
    "$1" -c "import importlib.util as u, importlib.metadata as m, sys
ok = u.find_spec('isaaclab') is not None
try: m.version('isaacsim')
except Exception: ok = ok and u.find_spec('isaacsim') is not None
sys.exit(0 if ok else 1)" >/dev/null 2>&1
}
PY=""
for cand in "${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}" "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" \
            "$ISAACLAB/_isaac_sim/python.sh" "$ISAACLAB/.venv/bin/python" "$ISAACLAB/env_isaaclab/bin/python" \
            "$HOME/.venv/bin/python" "$HOME"/*/bin/python "$ISAACLAB"/*/bin/python \
            "$HOME"/miniconda3/envs/*/bin/python "$HOME"/anaconda3/envs/*/bin/python "$HOME"/miniforge3/envs/*/bin/python \
            "$(command -v python 2>/dev/null)" "$(command -v python3 2>/dev/null)"; do
    [ -n "$cand" ] && has_isaac "$cand" && { PY="$cand"; break; }
done
[ -n "$PY" ] || stop "no Python with Isaac Sim + Isaac Lab found. Activate the environment you use for ./isaaclab.sh (e.g. conda activate <env>) and run this again, or set ISAACLAB=/path/to/IsaacLab"
say "Isaac Sim Python: $PY"

# ffmpeg for the mp4 (frames are written either way)
if ! "$PY" -c "import shutil, sys, importlib.util as u; sys.exit(0 if shutil.which('ffmpeg') or u.find_spec('imageio_ffmpeg') else 1)"; then
    say "no ffmpeg found - installing imageio-ffmpeg into that Python for the mp4 encoding ..."
    if "$PY" -m pip --version >/dev/null 2>&1; then
        "$PY" -m pip install --quiet imageio-ffmpeg
    elif command -v uv >/dev/null; then
        uv pip install --quiet --python "$PY" imageio-ffmpeg
    else
        say "neither pip nor uv available - the render will write PNG frames only (sudo apt install ffmpeg fixes that)"
    fi
fi

cd "$DEST" || stop "cannot enter $DEST"
if [ "$DRONE" = 1 ]; then
    say "Crazyflie PBR demo - the first start downloads ~300 MB of NVIDIA assets (cached afterwards)"
    "$PY" "$DEST/scripts/drone_scene/quadcopter_pbr.py" ${ARGS[@]+"${ARGS[@]}"}
    code=$?
    [ "$code" = 0 ] || { say "FAILED (exit code $code). Send Claude this file:"; echo "    $DEST/drone_scene_videos/drone_scene.log"; }
    exit "$code"
fi

# 3. render (outputs in $DEST/heavy_machine_videos/<scenario>/, log: render.log there)
say "rendering - Isaac Sim may take a few minutes to start; progress is printed every second of video"
"$PY" "$DEST/scripts/heavy_machines/scenarios/render_scenario.py" --headless ${ARGS[@]+"${ARGS[@]}"}
code=$?
if [ "$code" = 0 ]; then
    say "done. Videos:"; ls -1 "$DEST"/heavy_machine_videos/*/*.mp4 2>/dev/null | sed 's/^/    /'
    say "next time just run: $DEST/render.sh"
else
    say "FAILED (exit code $code). Send Claude this file:"
    ls -1t "$DEST"/heavy_machine_videos/*/render.log 2>/dev/null | head -1 | sed 's/^/    /'
fi
exit "$code"
