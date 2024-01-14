"""Utilities for collecting context key-value pairs as metadata in benchmark runs."""

import platform
from typing import Any, Callable, Sequence

ContextTuple = tuple[str, Any]
ContextProvider = Callable[[], ContextTuple | Sequence[ContextTuple]]
"""A function providing a context value. Context tuple is structured as context key name and value."""


def system() -> tuple[str, str]:
    return "system", platform.system()


def cpuarch() -> tuple[str, str]:
    return "cpuarch", platform.machine()


def python_version() -> tuple[str, str]:
    return "python_version", platform.python_version()
