import json
import os
from collections.abc import Iterable
from typing import Any

from rich.console import Console
from rich.table import Table

from nnbench.types import BenchmarkReporter, BenchmarkResult

_MISSING = "-----"


def get_value_by_name(result: dict[str, Any]) -> str:
    if result.get("error_occurred", False):
        errmsg = result.get("error_message", "<unknown>")
        return "[red]ERROR: [/red]" + errmsg
    return str(result.get("value", _MISSING))


class ConsoleReporter(BenchmarkReporter):
    """
    The base interface for a console reporter class.

    Wraps a ``rich.Console()`` to display values in a rich-text table.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a console reporter.

        Parameters
        ----------
        *args: Any
            Positional arguments, unused.
        **kwargs: Any
            Keyword arguments, forwarded directly to ``rich.Console()``.
        """
        super().__init__(*args, **kwargs)
        # TODO: Add context manager to register live console prints
        self.console = Console(**kwargs)

    def read(self, fp: str | os.PathLike[str], **kwargs: Any) -> list[BenchmarkResult]:
        raise NotImplementedError

    def write(
        self,
        results: Iterable[BenchmarkResult],
        outfile: str | os.PathLike[str] = None,
        **options: Any,
    ) -> None:
        """
        Display a benchmark result in the console as a rich-text table.

        Gives a summary of all present context values directly above the table,
        as a pretty-printed JSON result.

        By default, displays only the benchmark name, value, execution wall time,
        and parameters.

        Parameters
        ----------
        results: Iterable[BenchmarkResult]
            The benchmark result to display.
        outfile: str | os.PathLike[str]
            For compatibility with the `BenchmarkFileIO` interface, unused.
        options: Any
            Display options used to format the resulting table.
        """
        del outfile
        t = Table()

        rows: list[list[str]] = []
        columns: list[str] = ["Benchmark", "Value", "Wall time (ns)", "Parameters"]

        for res in results:
            # print context values
            print("Context values:")
            print(json.dumps(res.context, indent=4))

            for bm in res.benchmarks:
                row = [bm["name"], get_value_by_name(bm), str(bm["time_ns"]), str(bm["parameters"])]
                rows.append(row)

            for column in columns:
                t.add_column(column)
            for row in rows:
                t.add_row(*row)

            self.console.print(t, overflow="ellipsis")
