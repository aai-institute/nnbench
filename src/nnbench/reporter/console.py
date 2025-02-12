import json
from typing import Any

from rich.console import Console
from rich.table import Table

from nnbench.types import BenchmarkRecord

_MISSING = "-----"


def get_value_by_name(result: dict[str, Any]) -> str:
    if result.get("error_occurred", False):
        errmsg = result.get("error_message", "<unknown>")
        return "[red]ERROR: [/red]" + errmsg
    return str(result.get("value", _MISSING))


class ConsoleReporter:
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

    def display(self, record: BenchmarkRecord) -> None:
        """
        Display a benchmark record in the console as a rich-text table.

        Gives a summary of all present context values directly above the table,
        as a pretty-printed JSON record.

        By default, displays only the benchmark name, value, execution wall time,
        and parameters.

        Parameters
        ----------
        record: BenchmarkRecord
            The benchmark record to display.
        """
        t = Table()

        rows: list[list[str]] = []
        columns: list[str] = ["Benchmark", "Value", "Wall time (ns)", "Parameters"]

        # print context values
        print("Context values:")
        print(json.dumps(record.context, indent=4))

        for bm in record.benchmarks:
            row = [bm["name"], get_value_by_name(bm), str(bm["time_ns"]), str(bm["parameters"])]
            rows.append(row)

        for column in columns:
            t.add_column(column)
        for row in rows:
            t.add_row(*row)

        self.console.print(t, overflow="ellipsis")
