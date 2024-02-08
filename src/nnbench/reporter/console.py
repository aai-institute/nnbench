from __future__ import annotations

import re
import sys
from typing import Any, Callable

from nnbench.reporter.base import BenchmarkReporter
from nnbench.reporter.util import flatten, nullcols
from nnbench.types import BenchmarkRecord


class ConsoleReporter(BenchmarkReporter):
    def __init__(
        self,
        tablefmt: str = "simple",
        custom_formatters: dict[str, Callable[[Any], Any]] | None = None,
    ):
        self.tablefmt = tablefmt
        self.custom_formatters: dict[str, Callable[[Any], Any]] = custom_formatters or {}

    def write(
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
            filteredctx = {
                k: v
                for k, v in flatten(ctx).items()
                if any(k.startswith(i) for i in include_context)
            }
            filteredbm = {k: v for k, v in bm.items() if k not in nulls}
            filteredbm.update(filteredctx)
            # only apply custom formatters after context merge
            #  to allow custom formatting of context values.
            filteredbm = {
                k: self.custom_formatters.get(k, lambda x: x)(v) for k, v in filteredbm.items()
            }
            filtered.append(filteredbm)

        print(tabulate(filtered, headers="keys", tablefmt=self.tablefmt))
