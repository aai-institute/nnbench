"""Contains machinery to compare multiple benchmark records side by side."""

import copy
from collections.abc import Sequence

from rich.console import Console
from rich.table import Table

from nnbench.types import BenchmarkRecord
from nnbench.util import flatten

_MISSING = "-----"


def get_value_by_name(record: BenchmarkRecord, name: str, missing: str) -> str:
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
        return "[red]ERROR: [/red]" + errmsg
    return str(res.get("value", missing))


def compare(
    records: Sequence[BenchmarkRecord],
    parameters: Sequence[str] | None = None,
    contextvals: Sequence[str] | None = None,
    missing: str = _MISSING,
) -> None:
    """
    Compare a series of benchmark records, displaying their results in a table
    side by side.

    Parameters
    ----------
    records: Sequence[BenchmarkRecord]
        The benchmark records to compare.
    parameters: Sequence[str] | None
        Names of parameters to display as extra columns.
    contextvals: Sequence[str] | None
        Names of context values to display as extra columns. Supports nested access
        via dotted syntax.
    missing: str
        A placeholder string to show in the event of a missing metric.
    """
    t = Table()

    rows: list[list[str]] = []
    columns: list[str] = ["Benchmark run"]

    # Add metric names first, without duplicates.
    for record in records:
        names = [b["name"] for b in record.benchmarks]
        for name in names:
            if name not in set(columns):
                columns.append(name)

    names = copy.deepcopy(columns[1:])

    # Then parameters, if any
    if parameters is not None:
        columns += [f"Params->{p}" for p in parameters]

    if contextvals is not None:
        columns += contextvals

    # Main loop, extracts values from the individual records,
    # or a placeholder if there are any.
    for record in records:
        # flatten facilitates dotted access to nested context values, e.g. git.branch
        ctx = flatten(record.context)
        row = [record.run]
        row += [get_value_by_name(record, name, _MISSING) for name in names]
        # hacky, extra cols is likely now broken
        b = record.benchmarks[0]
        # TODO: Add record-level parameters struct as the union of all benchmark inputs
        if parameters is not None:
            params = b.get("parameters", {})
            row += [str(params.get(p)) for p in parameters]
        if contextvals is not None:
            row += [str(ctx.get(cval, missing)) for cval in contextvals]
        rows.append(row)

    for column in columns:
        t.add_column(column)
    for row in rows:
        t.add_row(*row)

    c = Console()
    c.print(t)
