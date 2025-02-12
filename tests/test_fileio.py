from pathlib import Path

import pytest

from nnbench.reporter.file import FileReporter
from nnbench.types import BenchmarkRecord


@pytest.mark.parametrize(
    "ext",
    ["yaml", "json", "ndjson", "csv", "parquet"],
)
def test_fileio_writes_no_compression_inline(tmp_path: Path, ext: str) -> None:
    """Tests data integrity for file IO roundtrips with both context modes."""
    f = FileReporter()

    rec = BenchmarkRecord(
        run="my-run",
        context={"a": "b", "s": 1, "b.c": 1.0},
        benchmarks=[{"name": "foo", "value": 1}, {"name": "bar", "value": 2}],
    )
    file = tmp_path / f"record.{ext}"
    f.write(rec, file)
    rec2 = f.read(file)
    # Python stdlib csv coerces everything to string.
    if ext == "csv":
        for bm1, bm2 in zip(rec.benchmarks, rec2.benchmarks):
            assert bm1.keys() == bm2.keys()
            assert set(map(str, bm1.values())) == set(bm2.values())
    else:
        assert rec2 == rec
