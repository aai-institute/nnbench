from typing import Sequence

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

    def read(self) -> BenchmarkRecord:
        raise NotImplementedError

    def read_batched(self) -> list[BenchmarkRecord]:
        raise NotImplementedError

    def write(self, record: BenchmarkRecord) -> None:
        raise NotImplementedError

    def write_batched(self, records: Sequence[BenchmarkRecord]) -> None:
        # by default, merge first and then write.
        if self.merge:
            merged = self.merge_records(records)
            self.write(merged)
        else:
            # write everything in a loop.
            for record in records:
                self.write(record)
