#!/usr/bin/env bash
# Every drone_scene unit test (no Isaac Sim app, no GPU), with Isaac Sim's Python; the map tests get the map tools.
#
#   run_tests.sh [CODE_DIR]      CODE_DIR: the checkout to test (default: this one); exit 0 when all suites pass
#
# test_mitsuba_preview is skipped: it needs usd-core and Mitsuba 3, which are not in Isaac Sim's Python (set
# RUN_MITSUBA_TESTS=1 to run it anyway). Each run writes its logs to its own temporary folder, so parallel runs for
# different worktrees do not mix.
#
# ISAACLAB_DIR (environment, or <main checkout>/.realism/config.env): the folder whose ./isaaclab.sh has Isaac
# Sim's Python.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || (cd "$HERE/../../.." && pwd))"
MAIN="$(dirname "$(git -C "$HERE" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || echo "$REPO/.git")")"
CONFIG="${DRONE_REALISM_HOME:-$MAIN/.realism}/config.env"
# shellcheck disable=SC1090
[[ -f "$CONFIG" ]] && source "$CONFIG"
ISAACLAB_DIR="${ISAACLAB_DIR:-$MAIN}"
code="$(realpath "${1:-$REPO}")"
logs="$(mktemp -d "${TMPDIR:-/tmp}/drone_scene_tests.XXXXXX")"

if [[ ! -x "$ISAACLAB_DIR/isaaclab.sh" ]]; then
  echo "no executable $ISAACLAB_DIR/isaaclab.sh: set ISAACLAB_DIR (in $CONFIG) to your Isaac Lab folder" >&2; exit 2
fi
failed=0
for t in "$code"/scripts/drone_scene/test/test_*.py; do
  name="$(basename "$t" .py)"
  if [[ "$name" == test_mitsuba_preview && -z "${RUN_MITSUBA_TESTS:-}" ]]; then
    echo "SKIP $name (needs usd-core and Mitsuba 3, which Isaac Sim's Python does not have)"
    continue
  fi
  if [[ "$name" == test_world ]]; then
    run=("$code/scripts/drone_scene/realism/with_world_libs.py" "$t")
  else
    run=("$t")
  fi
  if (cd "$ISAACLAB_DIR" && ./isaaclab.sh -p "${run[@]}" > "$logs/$name.log" 2>&1); then
    echo "PASS $name ($(grep -Eo 'Ran [0-9]+ tests?' "$logs/$name.log" | tail -n 1))"
  else
    echo "FAIL $name: $logs/$name.log"
    tail -n 15 "$logs/$name.log" | sed 's/^/    /'
    failed=1
  fi
done
if [[ $failed -eq 0 ]]; then rm -rf "$logs"; else echo "logs: $logs"; fi
exit $failed
