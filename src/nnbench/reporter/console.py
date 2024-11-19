from typing import Any

from rich.console import Console
from rich.table import Table

from nnbench.reporter.base import BenchmarkReporter
from nnbench.types import BenchmarkRecord

_MISSING = "-----"


def get_value_by_name(result: dict[str, Any]) -> str:
    if result.get("error_occurred", False):
        errmsg = result.get("error_message", "<unknown>")
        return "[red]ERROR: [/red]" + errmsg
    return str(result.get("value", _MISSING))


class ConsoleReporter(BenchmarkReporter):
    """
    The base interface for a console reporter class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Add context manager to register live console prints
        self.console = Console(**kwargs)

    def display(self, record: BenchmarkRecord) -> None:
        """
        Display a benchmark record in the console.

        Benchmarks and context values will be filtered before display
        if any filtering is applied.

        Columns that do not contain any useful information are omitted by default.

        Parameters
        ----------
        record: BenchmarkRecord
            The benchmark record to display.
        """
        t = Table()

        rows: list[list[str]] = []
        columns: list[str] = ["Benchmark", "Value", "Wall time (ns)", "Parameters"]

        # print context values
        for k, v in record.context.items():
            print(f"{k}: {v}")

        for bm in record.benchmarks:
            row = [bm["name"], get_value_by_name(bm), str(bm["time_ns"]), str(bm["parameters"])]
            rows.append(row)

        for column in columns:
            t.add_column(column)
        for row in rows:
            t.add_row(*row)

        self.console.print(t, overflow="ellipsis")
