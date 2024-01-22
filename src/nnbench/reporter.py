"""
A lightweight interface for refining, displaying, and streaming benchmark results to various sinks.
"""
from __future__ import annotations

import importlib
import sys
import types
from typing import Any

from nnbench.types import BenchmarkResult


class BaseReporter:
    """
    The base interface for a benchmark reporter class.

    A benchmark reporter consumes benchmark results from a run, and subsequently
    reports them in the way specified by the respective implementation's `report()`
    method.

    For example, to write benchmark results to a database, you could save the credentials
    for authentication in the class constructor, and then stream the results directly to
    the database in `report()`, with preprocessing if necessary.

    Parameters
    ----------
    **kwargs: Any
        Additional keyword arguments, for compatibility with subclass interfaces.
    """

    def __init__(self, **kwargs: Any):
        pass

    def report(self, result: BenchmarkResult) -> None:
        raise NotImplementedError


class ConsoleReporter(BaseReporter):
    # TODO: Implement regex filters, context values, display options, ... (__init__)
    def report(self, result: BenchmarkResult) -> None:
        try:
            from tabulate import tabulate
        except ModuleNotFoundError:
            raise ValueError(
                f"{self.__class__.__name__} requires `tabulate` to be installed. "
                f"To install, run `{sys.executable} -m pip install --upgrade tabulate`."
            )

        benchmarks = result["benchmarks"]
        print(tabulate(benchmarks, headers="keys"))


# internal, mutable
_reporter_registry: dict[str, type[BaseReporter]] = {
    "console": ConsoleReporter,
}

# external, immutable
reporter_registry: types.MappingProxyType[str, type[BaseReporter]] = types.MappingProxyType(
    _reporter_registry
)


def register_reporter(key: str, cls_or_name: str | type[BaseReporter]) -> None:
    """
    Register a reporter class by its fully qualified module path.

    Parameters
    ----------
    key: str
        The key to register the reporter under. Subsequently, this key can be used in place
        of reporter classes in code.
    cls_or_name: str | type[BaseReporter]
        Name of or full module path to the reporter class. For example, when registering a class
        ``MyReporter`` located in ``my_module``, ``name`` should be ``my_module.MyReporter``.
    """

    if isinstance(cls_or_name, str):
        name = cls_or_name
        modname, clsname = name.rsplit(".", 1)
        mod = importlib.import_module(modname)
        cls = getattr(mod, clsname)
        _reporter_registry[key] = cls
    else:
        # name = cls_or_name.__module__ + "." + cls_or_name.__qualname__
        _reporter_registry[key] = cls_or_name
