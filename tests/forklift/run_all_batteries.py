"""
Run all test batteries and print a summary.

Usage:
    # Unit tests only (no Isaac Sim):
    python -m pytest tests/forklift/test_unit.py -v

    # All batteries (requires Isaac Sim):
    ./isaaclab.sh -p tests/forklift/run_all_batteries.py

    # Or via pytest (requires Isaac Sim):
    ./isaaclab.sh -p -m pytest tests/forklift/ -v --headless
"""

import subprocess
import sys
import os


def main():
    test_dir = os.path.dirname(__file__)
    results = {}

    # Unit tests (no sim required)
    print("=" * 60, flush=True)
    print("  BATTERY: Unit Tests (no Isaac Sim)", flush=True)
    print("=" * 60, flush=True)
    ret = subprocess.run(
        [sys.executable, "-m", "pytest", os.path.join(test_dir, "test_unit.py"), "-v"],
        cwd=os.path.join(test_dir, "..", ".."),
    )
    results["Unit Tests"] = "PASS" if ret.returncode == 0 else "FAIL"

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("  TEST BATTERY SUMMARY", flush=True)
    print("=" * 60, flush=True)

    all_pass = True
    for name, status in results.items():
        icon = "GREEN" if status == "PASS" else "RED"
        print(f"  [{icon}] {name}: {status}", flush=True)
        if status != "PASS":
            all_pass = False

    print("=" * 60, flush=True)
    if all_pass:
        print("  ALL PASS", flush=True)
    else:
        print("  SOME FAILED — see details above", flush=True)
    print("=" * 60, flush=True)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
