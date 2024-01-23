"""Utilities for collecting context key-value pairs as metadata in benchmark runs."""

import platform
from typing import Any, Callable

ContextProvider = Callable[[], dict[str, Any]]
"""A function providing a dictionary of context values."""


def system() -> dict[str, str]:
    return {"system": platform.system()}


def cpuarch() -> dict[str, str]:
    return {"cpuarch": platform.machine()}


def python_version() -> dict[str, str]:
    return {"python_version": platform.python_version()}
