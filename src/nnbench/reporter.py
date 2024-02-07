"""
A lightweight interface for refining, displaying, and streaming benchmark results to various sinks.
"""
from __future__ import annotations

import copy
import importlib
import re
import sys
import types
from typing import Any

from nnbench.types import BenchmarkRecord


def nullcols(_benchmarks: list[dict[str, Any]]) -> set[str]:
    nulls = set()
    for i, bm in enumerate(_benchmarks):
        bm_nulls = set(k for k, v in bm.items() if not v)

        if i == 0:
            # NB: This breaks if the schema is unstable.
            nulls = bm_nulls
        else:
            # intersection to drop nulls that are populated in later benchmarks.
            nulls &= bm_nulls
    return nulls


def nested_getitem(obj: dict[str, Any], item: str) -> Any:
    items = item.split(".")
    for n, it in enumerate(items):
        try:
            obj = obj[it]
        except KeyError:
            if n == 0:
                msg = f"context has no member {item!r}"
            else:
                val = ".".join(items[: n + 1])
                msg = f"nested context value {val!r} has no member {it!r}"
            raise KeyError(msg) from None
    return obj


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
            bm_new = copy.copy(bm)
            filteredctx = {k: nested_getitem(ctx, k) for k in include_context}
            bm_new.update(filteredctx)
            for nc in nulls:
                bm_new.pop(nc)
            filtered.append(bm_new)

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
