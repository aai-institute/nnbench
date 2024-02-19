from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Callable, Literal, Sequence

from nnbench.reporter.base import BenchmarkReporter
from nnbench.types import BenchmarkRecord


@dataclass(frozen=True)
class FileDriverOptions:
    options: dict[str, Any] = field(default_factory=dict)
    """Options to pass to the underlying serializer library call, e.g. ``json.dump``."""
    ctxmode: Literal["flatten", "inline", "omit"] = "inline"
    """How to handle the context struct."""


_Options = dict[str, Any]
SerDe = tuple[
    Callable[[Sequence[BenchmarkRecord], IO, FileDriverOptions], None],
    Callable[[IO, FileDriverOptions], list[BenchmarkRecord]],
]


_file_drivers: dict[str, SerDe] = {}
_compressions: dict[str, Callable] = {}
_file_driver_lock = threading.Lock()
_compression_lock = threading.Lock()


def yaml_save(records: Sequence[BenchmarkRecord], fp: IO, fdoptions: FileDriverOptions) -> None:
    try:
        import yaml
    except ImportError:
        raise ModuleNotFoundError("`pyyaml` is not installed")

    bms = []
    for r in records:
        bms += r.compact(mode=fdoptions.ctxmode)
    yaml.safe_dump(bms, fp, **fdoptions.options)


def yaml_load(fp: IO, fdoptions: FileDriverOptions) -> list[BenchmarkRecord]:
    try:
        import yaml
    except ImportError:
        raise ModuleNotFoundError("`pyyaml` is not installed")

    # TODO: Use expandmany()
    bms = yaml.safe_load(fp)
    return [BenchmarkRecord.expand(bms)]


def json_save(records: Sequence[BenchmarkRecord], fp: IO, fdoptions: FileDriverOptions) -> None:
    import json

    newline: bool = fdoptions.options.pop("newline", False)
    bm = []
    for r in records:
        bm += r.compact(mode=fdoptions.ctxmode)
    if newline:
        fp.write("\n".join([json.dumps(b) for b in bm]))
    else:
        json.dump(bm, fp, **fdoptions.options)


def json_load(fp: IO, fdoptions: FileDriverOptions) -> list[BenchmarkRecord]:
    import json

    newline: bool = fdoptions.options.pop("newline", False)
    benchmarks: list[dict[str, Any]]
    if newline:
        benchmarks = [json.loads(line, **fdoptions.options) for line in fp]
    else:
        benchmarks = json.load(fp, **fdoptions.options)
    return [BenchmarkRecord.expand(benchmarks)]


def csv_save(records: Sequence[BenchmarkRecord], fp: IO, fdoptions: FileDriverOptions) -> None:
    import csv

    bm = []
    for r in records:
        bm += r.compact(mode=fdoptions.ctxmode)
    writer = csv.DictWriter(fp, fieldnames=bm[0].keys(), **fdoptions.options)

    for b in bm:
        writer.writerow(b)


def csv_load(fp: IO, fdoptions: FileDriverOptions) -> list[BenchmarkRecord]:
    import csv

    reader = csv.DictReader(fp, **fdoptions.options)

    benchmarks: list[dict[str, Any]] = []
    # apparently csv.DictReader has no appropriate type hint for __next__,
    # so we supply one ourselves.
    bm: dict[str, Any]
    for bm in reader:
        benchmarks.append(bm)

    return [BenchmarkRecord.expand(benchmarks)]


with _file_driver_lock:
    _file_drivers["yaml"] = (yaml_save, yaml_load)
    _file_drivers["json"] = (json_save, json_load)
    _file_drivers["csv"] = (csv_save, csv_load)
    # TODO: Add parquet support


class FileIO:
    def __init__(
        self,
        drivers: dict[str, SerDe] | None = None,
        compressions: dict[str, Callable] | None = None,
    ):
        super().__init__()
        self.drivers = drivers or _file_drivers
        self.compressions = compressions or _compressions

    def read(
        self,
        file: str | os.PathLike[str],
        driver: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> BenchmarkRecord:
        """
        Greedy version of ``FileIO.read_batched()``, returning the first read record.
        When reading a multi-record file, this uses as much memory as the batched version.
        """
        records = self.read_batched(file=file, driver=driver, options=options)
        return records[0]

    def read_batched(
        self,
        file: str | os.PathLike[str],
        driver: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[BenchmarkRecord]:
        """
        Reads a set of benchmark records from the given file path.

        The file driver is chosen based on the extension found on the ``file`` path.

        Parameters
        ----------
        file: str | os.PathLike[str]
            The file name to read from.
        driver: str | None
            File driver implementation to use. If None, the file driver inferred from the
            given file path's extension will be used.
        options: dict[str, Any] | None
            Options to pass to the respective file driver implementation.

        Returns
        -------
        list[BenchmarkRecord]
            The benchmark records contained in the file.

        Raises
        ------
        KeyError
            If the given file does not have a driver implementation available.
        """
        driver = driver or Path(file).suffix.removeprefix(".")

        try:
            _, de = self.drivers[driver]
        except KeyError:
            raise KeyError(f"unsupported file format {driver!r}") from None

        # dummy value, since the context mode is unused in read ops.
        fdoptions = FileDriverOptions(ctxmode="omit", options=options or {})

        with open(file, "r") as fp:
            return de(fp, fdoptions)

    def write(
        self,
        record: BenchmarkRecord,
        file: str | os.PathLike[str],
        driver: str | None = None,
        ctxmode: Literal["flatten", "inline", "omit"] = "inline",
        options: dict[str, Any] | None = None,
    ) -> None:
        """Greedy version of ``FileIO.write_batched()``"""
        self.write_batched([record], file=file, driver=driver, ctxmode=ctxmode, options=options)

    def write_batched(
        self,
        records: Sequence[BenchmarkRecord],
        file: str | os.PathLike[str],
        driver: str | None = None,
        ctxmode: Literal["flatten", "inline", "omit"] = "inline",
        options: dict[str, Any] | None = None,
    ) -> None:
        """
        Writes a benchmark record to the given file path.

        The file driver is chosen based on the extension found on the ``file`` path.

        Parameters
        ----------
        records: Sequence[BenchmarkRecord]
            The record to write to the database.
        file: str | os.PathLike[str]
            The file name to write to.
        driver: str | None
            File driver implementation to use. If None, the file driver inferred from the
            given file path's extension will be used.
        ctxmode: Literal["flatten", "inline", "omit"]
            How to handle the benchmark context when writing the record data.
        options: dict[str, Any] | None
            Options to pass to the respective file driver implementation.

        Raises
        ------
        KeyError
            If the given file does not have a driver implementation available.
        """
        driver = driver or Path(file).suffix.removeprefix(".")

        try:
            ser, _ = self.drivers[driver]
        except KeyError:
            raise KeyError(f"unsupported file format {driver!r}") from None

        fdoptions = FileDriverOptions(ctxmode=ctxmode, options=options or {})
        with open(file, "w") as fp:
            ser(records, fp, fdoptions)


class FileReporter(FileIO, BenchmarkReporter):
    pass
