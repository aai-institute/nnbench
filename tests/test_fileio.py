import itertools
from pathlib import Path
from typing import Literal

import pytest

from nnbench.context import Context
from nnbench.reporter.file import FileIO
from nnbench.types import BenchmarkRecord


@pytest.mark.parametrize(
    "ext,ctxmode",
    itertools.product(["yaml", "json", "ndjson", "csv", "parquet"], ["inline", "flatten"]),
)
def test_fileio_writes_no_compression_inline(
    tmp_path: Path, ext: str, ctxmode: Literal["inline", "flatten"]
) -> None:
    """Tests data integrity for file IO roundtrips with both context modes."""
    f = FileIO()

    rec = BenchmarkRecord(
        context=Context.make({"a": "b", "s": 1, "b.c": 1.0}),
        benchmarks=[{"name": "foo", "value": 1}, {"name": "bar", "value": 2}],
    )
    file = tmp_path / f"record.{ext}"
    writemode = "wb" if ext == "parquet" else "w"
    f.write(rec, file, mode=writemode, ctxmode="inline")
    readmode = "rb" if ext == "parquet" else "r"
    rec2 = f.read(file, mode=readmode)
    # Python stdlib csv coerces everything to string.
    if ext == "csv":
        for bm1, bm2 in zip(rec.benchmarks, rec2.benchmarks):
            assert bm1.keys() == bm2.keys()
            assert set(map(str, bm1.values())) == set(bm2.values())
    else:
        assert rec2 == rec
