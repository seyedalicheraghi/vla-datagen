"""Validate machines, regenerate their URDF/JSON, and optionally render previews.

Usage (no Isaac Sim needed):
    python scripts/heavy_machines/check_machine.py                    # all machines
    python scripts/heavy_machines/check_machine.py wheel_loader HEX   # by name or abbrev
    python scripts/heavy_machines/check_machine.py --render /tmp/prev # + PNG previews (matplotlib)
    python scripts/heavy_machines/check_machine.py --list

Exit code is 1 if any machine has errors.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heavy_machines import safe_console  # noqa: E402

safe_console()

from heavy_machines import registry  # noqa: E402
from heavy_machines.checks import validate  # noqa: E402
from heavy_machines.urdf_builder import write_assets  # noqa: E402


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path)
    except ValueError:                              # Windows: path and cwd on different drives
        return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("machines", nargs="*", help="names or abbreviations (default: all)")
    p.add_argument("--render", metavar="DIR", help="write <name>_poses.png / <name>_sweeps.png here")
    p.add_argument("--no-write", action="store_true", help="do not regenerate assets/heavy_machines")
    p.add_argument("--list", action="store_true", help="list registered machines and exit")
    p.add_argument("--readme", action="store_true",
                   help="regenerate the machine table in scripts/heavy_machines/README.md and exit")
    args = p.parse_args()

    if args.readme:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        head, rest = text.split("<!-- MACHINE_TABLE_START -->", 1)
        _, tail = rest.split("<!-- MACHINE_TABLE_END -->", 1)
        rows = ["| Machine | Abbrev | Category | Modelled on | Drive | Tools (action[2:]) | Mass |",
                "|---|---|---|---|---|---|---|"]
        for s in sorted(registry.all_specs().values(), key=lambda s: (s.category, s.name)):
            rows.append(f"| {s.display_name} (`{s.name}`) | {s.abbrev} | {s.category} | {s.reference} | "
                        f"{s.drive.type} | {', '.join(t.label for t in s.tools) or '—'} | "
                        f"{s.operating_mass / 1000:.1f} t |")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(head + "<!-- MACHINE_TABLE_START -->\n" + "\n".join(rows) + "\n<!-- MACHINE_TABLE_END -->" + tail)
        print(f"updated {path} ({len(rows) - 2} machines)")
        return 0

    if args.list:
        for s in registry.all_specs().values():
            print(f"{s.name:28s} {s.abbrev:5s} {s.category:11s} action={s.action_labels}  ({s.reference})")
        return 0

    failed = 0
    if args.machines:
        specs = []
        for n in args.machines:
            try:
                specs.append(registry.get(n))
            except (KeyError, RuntimeError) as exc:
                print(f"== {n}: FAIL\n  ERROR   {exc}")
                failed += 1
    else:
        specs = list(registry.all_specs().values())
        for mod, tb in registry.load_errors().items():
            print(f"== {mod}: FAIL (machines/{mod}.py did not load)\n{tb}")
            failed += 1
    for spec in specs:
        rep = validate(spec)
        print(rep.format())
        failed += not rep.ok
        if not args.no_write and rep.ok:
            urdf, meta = write_assets(spec)
            print(f"  wrote   {_rel(urdf)} , {_rel(meta)}")
        if args.render:
            from heavy_machines.render import render_camera_views, render_poses, render_tool_sweeps
            os.makedirs(args.render, exist_ok=True)
            print("  render  " + render_poses(spec, os.path.join(args.render, f"{spec.name}_poses.png")))
            sweep = render_tool_sweeps(spec, os.path.join(args.render, f"{spec.name}_sweeps.png"))
            if sweep:
                print("  render  " + sweep)
            print("  render  " + render_camera_views(spec, os.path.join(args.render, f"{spec.name}_cameras.png")))
    total = len(specs) + (0 if args.machines else len(registry.load_errors()))
    print(f"\n{total - failed}/{total} machines OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
