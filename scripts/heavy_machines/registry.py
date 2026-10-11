"""Discovers machines: every module in ``heavy_machines/machines/`` that defines
``build() -> MachineSpec`` is registered under ``spec.name``.

Adding a machine = adding one file. Nothing else needs editing.

A module that fails to import or build is skipped (so one broken file does not
take down every other machine) and its error is available from
:func:`load_errors`; asking for it by name re-raises the error.
"""

from __future__ import annotations

import importlib
import pkgutil
import traceback
from functools import lru_cache

from .spec import MachineSpec


@lru_cache(maxsize=None)
def _load() -> tuple[dict[str, MachineSpec], dict[str, str]]:
    from . import machines

    specs: dict[str, MachineSpec] = {}
    errors: dict[str, str] = {}
    for mod in sorted(pkgutil.iter_modules(machines.__path__), key=lambda m: m.name):
        if mod.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{machines.__name__}.{mod.name}")
            build = getattr(module, "build", None)
            if build is None:
                continue
            spec = build()
            if spec.name != mod.name:
                raise ValueError(f"machines/{mod.name}.py builds spec named {spec.name!r}; "
                                 f"the spec name must match the file name")
            specs[spec.name] = spec
        except Exception:  # noqa: BLE001 — report, keep loading the others
            errors[mod.name] = traceback.format_exc()
    return specs, errors


def all_specs() -> dict[str, MachineSpec]:
    return _load()[0]


def load_errors() -> dict[str, str]:
    """Module name → traceback for machine files that failed to load."""
    return _load()[1]


def names() -> list[str]:
    return list(all_specs())


def get(name: str) -> MachineSpec:
    specs, errors = _load()
    if name in specs:
        return specs[name]
    by_abbrev = {s.abbrev.lower(): s for s in specs.values()}
    if name.lower() in by_abbrev:
        return by_abbrev[name.lower()]
    if name in errors:
        raise RuntimeError(f"machines/{name}.py failed to load:\n{errors[name]}")
    raise KeyError(f"unknown machine {name!r}; available: {', '.join(sorted(specs))}")
