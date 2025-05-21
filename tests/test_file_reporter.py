from pathlib import Path

import pytest

from nnbench.reporter.file import FileReporter
from nnbench.types import BenchmarkResult


@pytest.mark.parametrize(
    "ext",
    ("yaml", "json", "ndjson", "parquet"),
)
def test_file_reporter_roundtrip(tmp_path: Path, ext: str) -> None:
    """Tests data integrity for file reporter roundtrips."""

    res = BenchmarkResult(
        run="my-run",
        context={"a": "b", "s": 1, "b.c": 1.0},
        benchmarks=[{"name": "foo", "value": 1}, {"name": "bar", "value": 2}],
        timestamp=0,
    )
    file = tmp_path / f"result.{ext}"
    f = FileReporter()
    f.write([res], file)
    (res2,) = f.read(file)

    if ext == "csv":
        for bm1, bm2 in zip(res.benchmarks, res2.benchmarks):
            assert bm1.keys() == bm2.keys()
            assert set(bm1.values()) == set(bm2.values())
    else:
        assert res2 == res
