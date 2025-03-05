"""Contains machinery to compare multiple benchmark records side by side."""

import operator
from collections.abc import Sequence
from typing import Any, Protocol

from rich.console import Console
from rich.table import Table

from nnbench.types import BenchmarkRecord

_MISSING = "N/A"


class Comparator(Protocol):
    def __call__(self, val1: Any, val2: Any) -> bool: ...


# TODO: Add vectorized comparators


class gt(Comparator):
    def __call__(self, val1: Any, val2: Any) -> bool:
        return operator.gt(val1, val2)


def get_value_by_name(record: BenchmarkRecord, name: str, missing: str) -> Any:
    """
    Get the value of a metric by name from a benchmark record, or a placeholder
    if the metric name is not present in the record.

    If the name is found, but the benchmark did not complete successfully
    (i.e. the ``error_occurred`` value is set to ``True``), the returned value
    will be set to the value of the ``error_message`` field.

    Parameters
    ----------
    record: BenchmarkRecord
        The benchmark record to extract a metric value from.
    name: str
        The name of the target metric.
    missing: str
        A placeholder string to return in the event of a missing metric.

    Returns
    -------
    str
        A string containing the metric value (or error message) formatted
        as rich text.

    """
    metric_names = [b["name"] for b in record.benchmarks]
    if name not in metric_names:
        return missing

    res = record.benchmarks[metric_names.index(name)]
    if res.get("error_occurred", False):
        errmsg = res.get("error_message", "<unknown>")
        return f"[red]ERROR: {errmsg} [/red]"
    return res.get("value", missing)


class Comparison:
    def __init__(self, placeholder: str = _MISSING):
        """
        Initialize a comparison class.

        Parameters
        ----------
        placeholder: str
            A placeholder string to show in the event of a missing metric.
        """
        self.placeholder = placeholder
        self.comparisons: list[dict[str, Any]] = []

    def compare2(self, metric_name: str, val1: Any, val2: Any) -> bool:
        """
        Compare two values of a metric across runs.

        A comparison is a function taking two values and returning a boolean
        indicating whether val2 compares favorably (the ``True`` case)
        or unfavorably to val1 (the ``False`` case).

        This method should generally be overwritten by child classes.

        Parameters
        ----------
        metric_name: str
            Name of the metric to compare.
        val1: Any
            Value of the metric in the first benchmark record.
        val2: Any
            Value of the metric in the second benchmark record.

        Returns
        -------
        bool
            The comparison between the two values.
        """
        # TODO: Make this abstract for the library user to implement
        if any(v == self.placeholder for v in (val1, val2)):
            return False
        _name = metric_name.lower()
        if "accuracy" in _name:
            return operator.le(val1, val2)
        else:
            return operator.abs(val1 - val2) <= 0.01

    def get_comparison(self, metric_name: str) -> str:
        _name = metric_name.lower()
        if "accuracy" in _name:
            return "val1 ≤ val2"
        else:
            return "|val1 - val2| ≤ 0.01"

    def render(self, records: Sequence[BenchmarkRecord]) -> None:
        """
        Compare a series of benchmark records, displaying their results in a table
        side by side.

        Parameters
        ----------
        records: Sequence[BenchmarkRecord]
            The benchmark records to compare.
        """
        if len(records) < 2:
            raise ValueError("must give at least two records to compare")

        t = Table()

        runs = [rec.run for rec in records]
        rows: list[list[str]] = []
        columns: list[str] = ["Metric Name"] + runs + ["Comparison", "Result"]

        metrics = [(bm["name"], bm["value"]) for bm in records[0].benchmarks]
        # Main loop, extracts values from the individual records,
        # or a placeholder if there are any.
        for metric in metrics:
            name, val = metric
            status = ""
            comparison = self.get_comparison(name)
            row = [name, str(val)]
            for record in records[1:]:
                compval = get_value_by_name(record, name, self.placeholder)
                success = self.compare2(name, val, compval)
                status += ":white_check_mark:" if success else ":x:"
                row += [str(compval)]
            row += [comparison, status]
            rows.append(row)

        for column in columns:
            t.add_column(column)
        for row in rows:
            t.add_row(*row)

        c = Console()
        c.print(t)

    @property
    def success(self) -> bool:
        """Indicates if the current comparison was successful."""
        return True
