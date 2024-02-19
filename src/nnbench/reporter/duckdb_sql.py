from __future__ import annotations

import os
import shutil
import sys
import tempfile
import weakref
from pathlib import Path

from nnbench.context import Context

try:
    import duckdb

    DUCKDB_INSTALLED = True
except ImportError:
    DUCKDB_INSTALLED = False

from nnbench.reporter.file import FileReporter
from nnbench.types import BenchmarkRecord


class DuckDBReporter(FileReporter):
    """
    A reporter for streaming benchmark results to duckdb.

    Serializes records into flat files in a temporary directory, and
    reads them in again afterwards.

    Parameters
    ----------
    dbname: str
        Name of the database to connect to. The default value ``"memory"``
        uses an in-memory duckdb database.
    read_only: bool
        Connect to a database in read-only mode.
    directory: str | os.PathLike[str] | None
        Destination directory to write records to. Must point to an existing directory.
        If omitted, a temporary directory will be used.
    delete: bool
        Delete the directory containing the written records after use. If the destination
        directory is a temporary directory, delete is implicitly true always.

    Raises
    ------
    ModuleNotFoundError
        If ``duckdb`` is not installed.
    """

    def __init__(
        self,
        dbname: str = ":memory:",
        read_only: bool = False,
        directory: str | os.PathLike[str] | None = None,
        delete: bool = False,
    ):
        if not DUCKDB_INSTALLED:
            raise ModuleNotFoundError(
                f"class {self.__class__.__name__!r} needs `duckdb` to be installed. "
                f"To install, run `{sys.executable} -m pip install --upgrade duckdb`."
            )

        super().__init__()
        self.dbname = dbname
        self.read_only = read_only

        # A place to store intermediate JSON records.
        if not directory:
            self._directory = Path(tempfile.mkdtemp())
            self.delete = True
        else:
            self._directory = Path(directory)
            self.delete = delete

        weakref.finalize(self, self.finalize)

        self.conn: duckdb.DuckDBPyConnection | None = None

    @property
    def directory(self) -> os.PathLike[str]:
        return self._directory

    def initialize(self):
        self.conn = duckdb.connect(self.dbname, read_only=self.read_only)
        self._initialized = True

    def finalize(self):
        if self.conn:
            self.conn.close()

        if self.delete:
            shutil.rmtree(self._directory, ignore_errors=True)

    def read_sql(
        self,
        file: str | os.PathLike[str],
        driver: str | None = None,
        include: tuple[str, ...] | None = None,
        alias: dict[str, str] | None = None,
        limit: int | None = None,
    ) -> BenchmarkRecord:
        if not self._initialized:
            self.initialize()

        driver = driver or Path(file).suffix.removeprefix(".")
        if driver not in ["json", "csv", "parquet"]:
            raise NotImplementedError("duckdb only supports reading JSON, CSV or parquet files")

        alias = alias or {}
        limit = limit or 0
        if limit < 0:
            raise ValueError("'limit' must be non-negative")

        if include is None:
            cols = "*"
        else:
            cols = ", ".join(i if i not in alias else f"{i} AS {alias[i]}" for i in include)

        # TODO: Query support for WHERE
        query = f"SELECT {cols} FROM read_json_auto('{str(file)}')"  # nosec B608

        rel = self.conn.sql(query)
        columns = rel.columns
        results = rel.fetchall()

        benchmarks = [dict(zip(columns, r)) for r in results]
        context = Context()
        for bm in benchmarks:
            context.update(bm.pop("context", {}))

        return BenchmarkRecord(context=context, benchmarks=benchmarks)

    def raw_sql(self, query: str) -> duckdb.DuckDBPyRelation:
        rel = self.conn.sql(query=query)
        return rel
