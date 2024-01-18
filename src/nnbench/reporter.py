"""
A lightweight interface for refining, displaying, and streaming benchmark results to various sinks.
"""

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
