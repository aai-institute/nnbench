"""Contains machinery to compare multiple benchmark records side by side."""

import operator
from collections.abc import Iterable
from typing import Any, Protocol

from rich.console import Console
from rich.table import Table

from nnbench.types import BenchmarkRecord

_MISSING = "N/A"
_STATUS_KEY = "Status"


class Comparator(Protocol):
    def __call__(self, val1: Any, val2: Any) -> bool: ...


class GreaterEqual(Comparator):
    def __call__(self, val1: Any, val2: Any) -> bool:
        return operator.ge(val1, val2)

    def __str__(self):
        return "x ≥ y"


class LessEqual(Comparator):
    def __call__(self, val1: Any, val2: Any) -> bool:
        return operator.le(val1, val2)

    def __str__(self):
        return "x ≤ y"


class AbsDiffLessEqual(Comparator):
    def __init__(self, thresh: float):
        self.thresh = thresh

    def __call__(self, val1: float, val2: float) -> bool:
        return operator.le(operator.abs(val1 - val2), self.thresh)

    def __str__(self):
        return f"|x - y| <= {self.thresh:.2%}"


def make_row(record: BenchmarkRecord) -> dict[str, Any]:
    d = dict()
    d["run"] = record.run
    for bm in record.benchmarks:
        # TODO: Guard against key errors from database queries
        name, value = bm["function"], bm["value"]
        d[name] = value
        # TODO: Check for errors
    return d


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
    metric_names = [b["function"] for b in record.benchmarks]
    if name not in metric_names:
        return missing

    res = record.benchmarks[metric_names.index(name)]
    if res.get("error_occurred", False):
        errmsg = res.get("error_message", "<unknown>")
        return f"[red]ERROR: {errmsg} [/red]"
    return res.get("value", missing)


class AbstractComparison:
    def compare2(self, name: str, val1: Any, val2: Any) -> bool:
        """A subroutine to compare two values of a metric ``name`` against each other."""
        raise NotImplementedError

    def render(self) -> None:
        """A method to render a previously computed comparison to a stream."""
        raise NotImplementedError

    @property
    def success(self) -> bool:
        """
        Indicates whether a comparison has been succesful based on the criteria
        expressed by the chosen set of comparators.
        """
        raise NotImplementedError


class TabularComparison(AbstractComparison):
    def __init__(
        self,
        records: Iterable[BenchmarkRecord],
        comparators: dict[str, Comparator] | None = None,
        placeholder: str = _MISSING,
    ):
        """
        Initialize a tabular comparison class, rendering the result to a rich table.

        Parameters
        ----------
        records: Sequence[BenchmarkRecord]
            The benchmark records to compare.
        comparators: dict[str, Comparator]
            A mapping from benchmark functions to comparators, i.e. a function
            comparing two results and returning a boolean indicating a favourable or
            unfavourable comparison.
        placeholder: str
            A placeholder string to show in the event of a missing metric.
        """
        self.placeholder = placeholder
        self.comparators = comparators or {}

        self.records: list[BenchmarkRecord] = list(records)
        self.data: list[dict[str, Any]] = [make_row(rec) for rec in self.records]
        self.metrics: list[str] = []
        self._success: bool = False

        self.display_names: dict[str, str] = {}
        for rec in self.records:
            for bm in rec.benchmarks:
                name, func = bm["name"], bm["function"]
                if func not in self.display_names:
                    self.display_names[func] = name
                if func not in self.metrics:
                    self.metrics.append(func)

        if len(self.data) < 2:
            raise ValueError("must give at least two records to compare")

    def compare2(self, name: str, val1: Any, val2: Any) -> bool:
        """
        Compare two values of a metric across runs.

        A comparison is a function taking two values and returning a boolean
        indicating whether val2 compares favorably (the ``True`` case)
        or unfavorably to val1 (the ``False`` case).

        This method should generally be overwritten by child classes.

        Parameters
        ----------
        name: str
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
        if any(v == self.placeholder for v in (val1, val2)):
            return False
        return self.comparators[name](val1, val2)

    def format_value(self, name: str, val: Any) -> str:
        if val == self.placeholder:
            return self.placeholder
        if name == "accuracy":
            return f"{val:.2%}"
        else:
            return f"{val:.2f}"

    def render(self) -> None:
        """
        Compare a series of benchmark records, displaying their results in a table
        side by side.
        """
        c = Console()
        t = Table()

        if self.comparators:
            c.print("Comparison strategy: All vs. first")
            c.print("Comparisons:")
            for k, v in self.comparators.items():
                c.print(f"    {k}: {v}")

        rows: list[list[str]] = []
        columns: list[str] = ["Run Name"] + list(self.display_names.values())
        if set(self.metrics) | self.comparators.keys():
            columns += ["Status"]

        for i, d in enumerate(self.data):
            row = [d["run"]]
            status = ""
            for metric in self.metrics:
                val = d.get(metric, self.placeholder)
                sval = self.format_value(metric, val)
                comparator = self.comparators.get(metric, None)
                if i and comparator is not None:
                    cd = self.data[0]
                    compval = cd.get(metric, self.placeholder)
                    success = self.compare2(metric, val, compval)
                    self._success |= success
                    status += ":white_check_mark:" if success else ":x:"
                    sval += " (vs. " + self.format_value(metric, compval) + ")"
                row += [sval]
            if "Status" in columns:
                row += [status]
            rows.append(row)

        for column in columns:
            t.add_column(column)
        for row in rows:
            t.add_row(*row)

        c.print(t)

    @property
    def success(self) -> bool:
        """Indicates if the current comparison was successful."""
        return self._success
