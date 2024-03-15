"""Metaclasses for defining transforms acting on benchmark records."""

from typing import Sequence

from nnbench.types import BenchmarkRecord


class Transform:
    """The basic transform which every transform has to inherit from."""

    pass


class OneToOneTransform(Transform):
    invertible: bool = True
    """
    Whether this transform is invertible,
    i.e. records can be converted back and forth with no changes or data loss.
    """

    def apply(self, record: BenchmarkRecord) -> BenchmarkRecord:
        """
        Apply this transform to a benchmark record.

        Parameters
        ----------
        record: BenchmarkRecord
            Benchmark record to apply the transform on.

        Returns
        -------
        BenchmarkRecord
            The transformed benchmark record.
        """

    def iapply(self, record: BenchmarkRecord) -> BenchmarkRecord:
        """
        Apply the inverse of this transform.

        In general, applying the inverse on a record not previously transformed
        may yield unexpected results.

        Parameters
        ----------
        record: BenchmarkRecord
            Benchmark record to apply the inverse transform on.

        Returns
        -------
        BenchmarkRecord
            The inversely transformed benchmark record.
        """


class ManyToOneTransform(Transform):
    """
    A many-to-one transform reducing a collection of records to a single record.

    This is useful for computing statistics on a collection of runs.
    """

    invertible: bool = True
    """
    Whether this transform is invertible,
    i.e. records can be converted back and forth with no changes or data loss.
    """

    def apply(self, record: Sequence[BenchmarkRecord]) -> BenchmarkRecord:
        """
        Apply this transform to a benchmark record.

        Parameters
        ----------
        record: Sequence[BenchmarkRecord]
            A sequence of benchmark record to apply the transform on,
            yielding a single resulting record.

        Returns
        -------
        BenchmarkRecord
            The transformed (reduced) benchmark record.
        """

    def iapply(self, record: BenchmarkRecord) -> Sequence[BenchmarkRecord]:
        """
        Apply the inverse of this transform.

        In general, applying the inverse on a record not previously transformed
        may yield unexpected results.

        Parameters
        ----------
        record: BenchmarkRecord
            Benchmark record to apply the inverse transform on.

        Returns
        -------
        Sequence[BenchmarkRecord]
            The inversely transformed benchmark record sequence.
        """
        # TODO: Does this even make sense? Can't hurt to allow it on paper, though.


class ManyToManyTransform(Transform):
    """
    A many-to-many transform mapping an input record collection to an output collection.

    Use this to programmatically wrangle metadata or types in records, or to
    convert parameters into database-ready representations.
    """

    invertible: bool = True
    """
    Whether this transform is invertible,
    i.e. records can be converted back and forth with no changes or data loss.
    """
    length_invariant: bool = True
    """
    Whether this transform preserves the number of records, i.e. no records are dropped.
    """

    def apply(self, record: Sequence[BenchmarkRecord]) -> Sequence[BenchmarkRecord]:
        """
        Apply this transform to a benchmark record.

        Parameters
        ----------
        record: Sequence[BenchmarkRecord]
            A sequence of benchmark record to apply the transform on.

        Returns
        -------
        Sequence[BenchmarkRecord]
            The transformed benchmark record sequence.
        """

    def iapply(self, record: Sequence[BenchmarkRecord]) -> Sequence[BenchmarkRecord]:
        """
        Apply the inverse of this transform.

        In general, applying the inverse on a record not previously transformed
        may yield unexpected results.

        Parameters
        ----------
        record: Sequence[BenchmarkRecord]
            A sequence of benchmark record to apply the transform on.

        Returns
        -------
        Sequence[BenchmarkRecord]
            The inversely transformed benchmark record sequence.
        """
