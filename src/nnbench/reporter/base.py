import re
from typing import Any, Callable, Sequence

from tabulate import tabulate

from nnbench.reporter.util import flatten, nullcols
from nnbench.types import BenchmarkRecord


def default_merge(records: Sequence[BenchmarkRecord]) -> BenchmarkRecord:
    """
    Merges a number of benchmark records into one.

    The resulting record has an empty top-level context, since the context
    values might be different in all respective records.

    TODO: Think about merging contexts here to preserve the record model,
     padding missing values with a placeholder if not present.
     -> Might be easier with an OOP Context class.

    Parameters
    ----------
    records: Sequence[BenchmarkRecord]
        The records to merge.

    Returns
    -------
    BenchmarkRecord
        The merged record, with all benchmark contexts inlined into their
        respective benchmarks.

    """
    merged = BenchmarkRecord(context=dict(), benchmarks=[])
    for record in records:
        ctx, benchmarks = record["context"], record["benchmarks"]
        for bm in benchmarks:
            bm["context"] = ctx
        merged["benchmarks"].extend(benchmarks)
    return merged


# TODO: Add IO mixins for database, file, and HTTP IO
class BenchmarkReporter:
    """
    The base interface for a benchmark reporter class.

    A benchmark reporter consumes benchmark results from a previous run, and subsequently
    reports them in the way specified by the respective implementation's ``report_result()``
    method.

    For example, to write benchmark results to a database, you could save the credentials
    for authentication on the class, and then stream the results directly to
    the database in ``report_result()``, with preprocessing if necessary.
    """

    merge: bool = False
    """Whether to merge multiple BenchmarkRecords before reporting."""

    def __init__(
        self,
        tablefmt: str = "simple",
        custom_formatters: dict[str, Callable[[Any], Any]] | None = None,
    ):
        self.tablefmt = tablefmt
        self.custom_formatters: dict[str, Callable[[Any], Any]] = custom_formatters or {}

    def initialize(self):
        """
        Initialize the reporter's state.

        This is the place where to create a result directory, a database connection,
        or a HTTP client.
        """
        pass

    def finalize(self):
        """
        Finalize the reporter's state.

        This is the place to destroy / release resources that were previously
        acquired in ``initialize()``.
        """
        pass

    merge_records = staticmethod(default_merge)

    def display(
        self,
        record: BenchmarkRecord,
        benchmark_filter: str | None = None,
        include_context: tuple[str, ...] = (),
        exclude_empty: bool = True,
    ) -> None:
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

    def read(self) -> BenchmarkRecord:
        raise NotImplementedError

    def read_batched(self) -> list[BenchmarkRecord]:
        raise NotImplementedError

    def write(self, record: BenchmarkRecord) -> None:
        raise NotImplementedError

    def write_batched(self, records: Sequence[BenchmarkRecord]) -> None:
        # By default, just loop over the records and write() everything.
        for record in records:
            self.write(record)
