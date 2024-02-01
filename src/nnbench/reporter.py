"""
A lightweight interface for refining, displaying, and streaming benchmark results to various sinks.
"""
from __future__ import annotations

import importlib
import sys
import types

from nnbench.types import BenchmarkRecord


# TODO: Add IO mixins for database, file, and HTTP IO
class BenchmarkReporter:
    """
    The base interface for a benchmark reporter class.

    A benchmark reporter consumes benchmark results from a previous run, and subsequently
    reports them in the way specified by the respective implementation's `report_result()`
    method.

    For example, to write benchmark results to a database, you could save the credentials
    for authentication on the class, and then stream the results directly to
    the database in `report_result()`, with preprocessing if necessary.
    """

    merge: bool = False
    """Whether to merge multiple BenchmarkRecords before reporting."""

    def report_result(self, record: BenchmarkRecord) -> None:
        raise NotImplementedError

    def report(self, *records: BenchmarkRecord) -> None:
        if self.merge:
            raise NotImplementedError
        for record in records:
            self.report_result(record)


class ConsoleReporter(BenchmarkReporter):
    # TODO: Implement regex filters, context values, display options, ... (__init__)
    def report_result(self, record: BenchmarkRecord) -> None:
        try:
            from tabulate import tabulate
        except ModuleNotFoundError:
            raise ValueError(
                f"{self.__class__.__name__} requires `tabulate` to be installed. "
                f"To install, run `{sys.executable} -m pip install --upgrade tabulate`."
            )

        benchmarks = record["benchmarks"]
        print(tabulate(benchmarks, headers="keys"))


# internal, mutable
_reporter_registry: dict[str, type[BenchmarkReporter]] = {
    "console": ConsoleReporter,
}

# external, immutable
reporter_registry: types.MappingProxyType[str, type[BenchmarkReporter]] = types.MappingProxyType(
    _reporter_registry
)


def register_reporter(key: str, cls_or_name: str | type[BenchmarkReporter]) -> None:
    """
    Register a reporter class by its fully qualified module path.

    Parameters
    ----------
    key: str
        The key to register the reporter under. Subsequently, this key can be used in place
        of reporter classes in code.
    cls_or_name: str | type[BenchmarkReporter]
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
