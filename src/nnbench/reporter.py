"""
A lightweight interface for refining, displaying, and streaming benchmark results to various sinks.
"""
from __future__ import annotations

import collections
import importlib
import re
import sys
import types
from typing import Any

from nnbench.types import BenchmarkRecord


def nullcols(_benchmarks: list[dict[str, Any]]) -> tuple[str, ...]:
    """
    Extracts columns that only contain false-ish data from a list of benchmarks.

    Since this data is most often not interesting, the result of this
    can be used to filter out these columns from the benchmark dictionaries.

    Parameters
    ----------
    _benchmarks: list[dict[str, Any]]
        The benchmarks to filter.

    Returns
    -------
    tuple[str, ...]
        Tuple of the columns (key names) that only contain false-ish values
        across all benchmarks.
    """
    nulls: dict[str, bool] = collections.defaultdict(bool)
    for bm in _benchmarks:
        for k, v in bm.items():
            nulls[k] = nulls[k] or bool(v)
    return tuple(k for k, v in nulls.items() if not v)


def flatten(d: dict[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:
    """
    Turn a nested dictionary into a flattened dictionary.

    Parameters
    ----------
    d: dict[str, Any]
        (Possibly) nested dictionary to flatten.
    prefix: str
        Key prefix to apply at the top-level (nesting level 0).
    sep: str
        Separator on which to join keys, "." by default.

    Returns
    -------
    dict[str, Any]
        The flattened dictionary.
    """

    items: list[tuple[str, Any]] = []
    for key, value in d.items():
        new_key = prefix + sep + key if prefix else key
        if isinstance(value, dict):
            items.extend(flatten(value, new_key, sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


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
    def __init__(self, tablefmt: str = "simple"):
        self.tablefmt = tablefmt

    def report_result(
        self,
        record: BenchmarkRecord,
        benchmark_filter: str | None = None,
        include_context: tuple[str, ...] = (),
        exclude_empty: bool = True,
    ) -> None:
        try:
            from tabulate import tabulate
        except ModuleNotFoundError:
            raise ValueError(
                f"class {self.__class__.__name__}() requires `tabulate` to be installed. "
                f"To install, run `{sys.executable} -m pip install --upgrade tabulate`."
            )

        ctx, benchmarks = record["context"], record["benchmarks"]

        nulls = set() if not exclude_empty else nullcols(benchmarks)

        if benchmark_filter is not None:
            regex = re.compile(benchmark_filter, flags=re.IGNORECASE)
        else:
            regex = None

        filtered = []
        for bm in benchmarks:
            if regex is not None and regex.search(bm["name"]) is None:
                continue
            ctx = flatten(ctx)
            filteredctx = {
                k: v for k, v in ctx.items() if any(k.startswith(i) for i in include_context)
            }
            filteredbm = {k: v for k, v in bm.items() if k not in nulls}
            filteredbm.update(filteredctx)
            filtered.append(filteredbm)

        # TODO: Add support for custom formatters
        print(tabulate(filtered, headers="keys", tablefmt=self.tablefmt))


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
