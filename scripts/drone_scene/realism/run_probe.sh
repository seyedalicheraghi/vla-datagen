#!/usr/bin/env bash
# Isaac Sim renders for the realism loop, on this PC's GPU.  Renders never overlap: each takes a lock file and the
# others queue (this holds across separate Claude Code sessions too).  A render can outlast an agent's 10-minute
# command limit, so it runs in the background and you wait for it in steps.
#
#   run_probe.sh start  OUT_DIR [CODE_DIR] [-- ARGS...]  probe stills (default below); returns at once
#   run_probe.sh record OUT_DIR [CODE_DIR] [-- ARGS...]  the whole flight as video (front back chase)
#   run_probe.sh wait   OUT_DIR [MAX_SECONDS]            wait up to MAX_SECONDS (default 540, stay under 600):
#                                                        exit 0 = done, 1 = failed, 3 = still queued/running
#   run_probe.sh status OUT_DIR
#   run_probe.sh stop   OUT_DIR
#   run_probe.sh compare BASE_DIR CAND_DIR [SEED]        image statistics B - A and blind A/B pairs of the stills
#                                                        both probes have -> CAND_DIR/ab/ (key.json: which is A)
#   run_probe.sh naip   OUT_DIR [REF_DIR]                real NAIP orthophotos of the nadir stills' squares
#                                                        (default REF_DIR <main checkout>/.realism/reference/naip)
#                                                        and their colour/structure difference from the stills
#
# OUT_DIR   gets drone_scene.log, probe/<camera>_<moment>.png, probe/probe.json, probe/metrics.json, contact.jpg
# CODE_DIR  the checkout whose scripts/drone_scene is rendered, e.g. a candidate's worktree (default: this one)
# ARGS      appended to the defaults; a repeated flag replaces the default (argparse keeps the last one), e.g.
#           -- --probe cruise_mid final --cameras front nadir
#
# Probe default:  --route straight --headless --probe --cameras front back chase nadir --resolution_scale 0.5
# Record default: --route straight --headless --record -1 --cameras front back chase
#
# A render that runs longer than DRONE_PROBE_TIMEOUT (default 7200 s) or DRONE_RECORD_TIMEOUT (21600 s), counted
# from when it gets the GPU, is killed with everything it started, so a hung Isaac Sim cannot hold the GPU forever.
#
# Settings: environment, or <main checkout>/.realism/config.env (written by /first-run):
#   ISAACLAB_DIR    folder whose ./isaaclab.sh runs Isaac Sim's Python (default: the main checkout of this repo)
#   DRONE_GPU_LOCK  the lock file (default /tmp/drone_scene_gpu.lock)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || (cd "$HERE/../../.." && pwd))"
MAIN="$(dirname "$(git -C "$HERE" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || echo "$REPO/.git")")"
CONFIG="${DRONE_REALISM_HOME:-$MAIN/.realism}/config.env"
# shellcheck disable=SC1090
[[ -f "$CONFIG" ]] && source "$CONFIG"
ISAACLAB_DIR="${ISAACLAB_DIR:-$MAIN}"
LOCK="${DRONE_GPU_LOCK:-/tmp/drone_scene_gpu.lock}"
PROBE_ARGS=(--route straight --headless --probe --cameras front back chase nadir --resolution_scale 0.5)
RECORD_ARGS=(--route straight --headless --record -1 --cameras front back chase)

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; exit 2; }
status_of() { cat "$1/run.status" 2>/dev/null || echo "missing"; }
need_isaac() {
  if [[ ! -x "$ISAACLAB_DIR/isaaclab.sh" ]]; then
    echo "no executable $ISAACLAB_DIR/isaaclab.sh: set ISAACLAB_DIR in $CONFIG to your Isaac Lab folder" >&2; exit 2
  fi
}
py() { (cd "$ISAACLAB_DIR" && ./isaaclab.sh -p "$@"); }
# true when PID is alive and its command line contains TEXT (guards against a recycled PID)
ours() { [[ -n "$1" ]] && kill -0 "$1" 2>/dev/null && tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -q -- "$2"; }
# true while PID runs (an exited child that is not reaped yet is a zombie and still answers kill -0)
alive() { local s; s="$(ps -o stat= -p "$1" 2>/dev/null)" || return 1; [[ -n "$s" && "$s" != Z* ]]; }

cmd="${1:-}"
[[ $# -gt 0 ]] && shift
case "$cmd" in
  start|record)
    [[ $# -ge 1 ]] || usage
    need_isaac
    out="$(realpath -m "$1")"; shift
    code="$REPO"
    if [[ $# -gt 0 && "$1" != "--" ]]; then code="$(realpath "$1")"; shift; fi
    [[ "${1:-}" == "--" ]] && shift
    if [[ ! -f "$code/scripts/drone_scene/quadcopter_pbr.py" ]]; then
      echo "no scripts/drone_scene/quadcopter_pbr.py under $code" >&2; exit 2
    fi
    mkdir -p "$out"
    case "$(status_of "$out")" in queued*|running*)
      if ours "$(cat "$out/run.pid" 2>/dev/null)" run_probe.sh; then
        echo "already queued or running: $out"; exit 2
      fi;;
    esac
    # a re-run must not show the previous run's log if it fails before writing its own
    rm -f "$out/drone_scene.log" "$out/run.pid" "$out/render.pgid" "$out/code_commit"
    echo "queued $(date -Is)" > "$out/run.status"
    setsid nohup bash "${BASH_SOURCE[0]}" _run "$cmd" "$out" "$code" "$@" > "$out/run.log" 2>&1 < /dev/null &
    echo $! > "$out/run.pid"
    echo "started ($cmd): $out - code from $code, Isaac Sim from $ISAACLAB_DIR, log $out/drone_scene.log"
    ;;
  _run)
    mode="$1"; out="$2"; code="$3"; shift 3
    if [[ "$mode" == record ]]; then defaults=("${RECORD_ARGS[@]}"); limit="${DRONE_RECORD_TIMEOUT:-21600}"
    else defaults=("${PROBE_ARGS[@]}"); limit="${DRONE_PROBE_TIMEOUT:-7200}"; fi
    exec 9>"$LOCK"
    echo "waiting for the GPU (lock $LOCK) ..."
    flock 9
    echo "running $(date -Is)" > "$out/run.status"
    git -C "$code" rev-parse --short HEAD > "$out/code_commit" 2>/dev/null
    t0=$(date +%s)
    # the render gets its own process group, so a time-out (or stop) can kill Isaac Sim and everything it started
    setsid bash -c 'cd "$1" && shift && exec ./isaaclab.sh -p "$@"' _ "$ISAACLAB_DIR" \
      "$code/scripts/drone_scene/quadcopter_pbr.py" "${defaults[@]}" --out "$out" "$@" &
    rpid=$!
    echo "$rpid" > "$out/render.pgid"
    timed_out=0
    while alive "$rpid"; do
      if (( $(date +%s) - t0 > limit )); then
        timed_out=1
        kill -TERM -- "-$rpid" 2>/dev/null; sleep 30; kill -KILL -- "-$rpid" 2>/dev/null
        break
      fi
      sleep 5
    done
    wait "$rpid"; rc=$?
    t1=$(date +%s)
    flock -u 9
    if (( timed_out )); then
      echo "failed (timed out after ${limit} s; raise DRONE_PROBE_TIMEOUT / DRONE_RECORD_TIMEOUT if it was only slow)" \
        > "$out/run.status"
    elif [[ $rc -eq 0 && ( "$mode" == record || -f "$out/probe/probe.json" ) ]]; then
      if [[ "$mode" != record ]]; then
        { py "$code/scripts/drone_scene/realism/metrics.py" "$out/probe" \
          && py "$code/scripts/drone_scene/realism/montage.py" contact "$out/probe" --out "$out/contact.jpg"; } \
          || echo "metrics or contact sheet failed (the stills are fine)"
      fi
      echo "done in $((t1 - t0)) s" > "$out/run.status"
    else
      echo "failed (exit code $rc) after $((t1 - t0)) s" > "$out/run.status"
    fi
    ;;
  wait)
    [[ $# -ge 1 ]] || usage
    out="$(realpath -m "$1")"; max="${2:-540}"
    end=$(( $(date +%s) + max ))
    while :; do
      st="$(status_of "$out")"
      case "$st" in
        done*) echo "$st"; tail -n 3 "$out/drone_scene.log" 2>/dev/null; exit 0;;
        failed*|missing)
          echo "$st"
          if [[ -s "$out/drone_scene.log" ]]; then tail -n 60 "$out/drone_scene.log"; else tail -n 60 "$out/run.log"; fi
          exit 1;;
      esac
      if [[ -f "$out/run.pid" ]] && ! kill -0 "$(cat "$out/run.pid")" 2>/dev/null; then
        sleep 1; st="$(status_of "$out")"
        case "$st" in done*|failed*) continue;; esac
        echo "the render process ended without finishing ($st)"; tail -n 60 "$out/run.log"; exit 1
      fi
      if (( $(date +%s) >= end )); then
        echo "still $st"; tail -n 2 "$out/drone_scene.log" 2>/dev/null; exit 3
      fi
      sleep 10
    done
    ;;
  status)
    [[ $# -ge 1 ]] || usage
    out="$(realpath -m "$1")"
    status_of "$out"; tail -n 5 "$out/drone_scene.log" 2>/dev/null
    ;;
  stop)
    [[ $# -ge 1 ]] || usage
    out="$(realpath -m "$1")"
    st="$(status_of "$out")"
    case "$st" in queued*|running*) ;; *) echo "not running (status: $st)"; exit 0;; esac
    pid="$(cat "$out/run.pid" 2>/dev/null)"
    rp="$(cat "$out/render.pgid" 2>/dev/null)"
    ours "$rp" quadcopter_pbr && kill -TERM -- "-$rp" 2>/dev/null
    ours "$pid" run_probe.sh && kill -TERM -- "-$pid" 2>/dev/null
    echo "failed (stopped) $(date -Is)" > "$out/run.status"
    echo "stopped $out"
    ;;
  compare)
    [[ $# -ge 2 ]] || usage
    need_isaac
    base="$(realpath -m "$1")"; cand="$(realpath -m "$2")"; seed="${3:-0}"
    naip="$MAIN/.realism/reference/naip"
    pairs=()
    [[ -d "$naip" ]] && pairs=(--pairs "$naip")
    py "$HERE/metrics.py" "$base/probe" "$cand/probe" "${pairs[@]}" --out "$cand/metrics_vs_base.json" || exit 1
    py "$HERE/montage.py" ab "$base/probe" "$cand/probe" --out "$cand/ab" --seed "$seed" || exit 1
    echo "pairs: $cand/ab/*.jpg; key (do not show to judges): $cand/ab/key.json; statistics: $cand/metrics_vs_base.json"
    ;;
  naip)
    [[ $# -ge 1 ]] || usage
    need_isaac
    run="$(realpath -m "$1")"; ref="$(realpath -m "${2:-$MAIN/.realism/reference/naip}")"
    py "$HERE/with_world_libs.py" "$HERE/reference.py" "$run/probe" "$ref" || exit 1
    py "$HERE/metrics.py" "$run/probe" --pairs "$ref" --out "$run/metrics_vs_naip.json"
    ;;
  *) usage;;
esac
