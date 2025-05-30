import os
import sqlite3
from typing import Any

from nnbench.types import BenchmarkReporter, BenchmarkResult

# TODO: Add tablename state (f-string)
_DEFAULT_COLS = ("run", "benchmark", "context", "timestamp")
_DEFAULT_CREATION_QUERY = "CREATE TABLE IF NOT EXISTS nnbench(" + ", ".join(_DEFAULT_COLS) + ")"
_DEFAULT_INSERT_QUERY = "INSERT INTO nnbench VALUES(:run, :benchmark, :context, :timestamp)"
_DEFAULT_READ_QUERY = """SELECT * FROM nnbench"""


class SQLiteReporter(BenchmarkReporter):
    @staticmethod
    def strip_protocol(uri: str | os.PathLike[str]) -> str:
        s = str(uri)
        if s.startswith("sqlite://"):
            return s[9:]
        return s

    def read(
        self,
        uri: str | os.PathLike[str],
        query: str = _DEFAULT_READ_QUERY,
        **kwargs: Any,
    ) -> list[BenchmarkResult]:
        uri = self.strip_protocol(uri)
        # query: str | None = options.pop("query", _DEFAULT_READ_QUERY)
        if query is None:
            raise ValueError(f"need a query to read from SQLite database {uri!r}")

        db = f"file:{uri}?mode=ro"  # open DB in read-only mode
        conn = sqlite3.connect(db, uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query)
        records = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return BenchmarkResult.from_records(records)

    def write(
        self,
        result: BenchmarkResult,
        uri: str | os.PathLike[str],
        query: str = _DEFAULT_INSERT_QUERY,
        **kwargs: Any,
    ) -> None:
        uri = self.strip_protocol(uri)

        conn = sqlite3.connect(uri)
        cursor = conn.cursor()

        # TODO: Guard by exists_ok state
        cursor.execute(_DEFAULT_CREATION_QUERY)
        cursor.executemany(query, result.to_records())
        conn.commit()
