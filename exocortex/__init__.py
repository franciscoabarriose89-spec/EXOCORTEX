"""Compatibility shim that delegates to the real src/exocortex package."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

_pkg_path = pathlib.Path(__file__).resolve().parent.parent / "src" / "exocortex"
_spec = importlib.util.spec_from_file_location(
    __name__,
    _pkg_path / "__init__.py",
    submodule_search_locations=[str(_pkg_path)],
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load package from {_pkg_path}")

_module = importlib.util.module_from_spec(_spec)
sys.modules[__name__] = _module
_spec.loader.exec_module(_module)
