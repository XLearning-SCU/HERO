"""Shared import-path helpers for plugin loading."""

from __future__ import annotations

import importlib
from typing import TypeVar

T = TypeVar("T")


def import_subclass(
    import_path: str,
    base_type: type[T],
    *,
    kind: str = "import path",
) -> type[T]:
    """Resolve ``module:Class`` and verify it subclasses ``base_type``."""
    module_name, sep, attr_name = import_path.partition(":")
    if not sep or not module_name or not attr_name:
        raise ValueError(f"Invalid {kind}: {import_path!r}")

    module = importlib.import_module(module_name)
    obj = getattr(module, attr_name)
    if not isinstance(obj, type):
        raise TypeError(
            f"{import_path} resolved to {type(obj).__name__}, expected a class"
        )
    if not issubclass(obj, base_type):
        raise TypeError(f"{import_path} is not a subclass of {base_type.__name__}")
    return obj
