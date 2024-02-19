from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Callable, Literal, Sequence

from nnbench.types import BenchmarkRecord


@dataclass(frozen=True)
class FileDriverOptions:
    options: dict[str, Any] = field(default_factory=dict)
    """Options to pass to the underlying serializer library call, e.g. ``json.dump``."""
    ctxmode: Literal["flatten", "inline", "omit"] = "inline"
    """How to handle the context struct."""


_Options = dict[str, Any]
SerDe = tuple[
    Callable[[BenchmarkRecord, IO, FileDriverOptions], None],
    Callable[[IO, FileDriverOptions], BenchmarkRecord],
]


_file_drivers: dict[str, SerDe] = {}
_compressions: dict[str, Callable] = {}
_file_driver_lock = threading.Lock()
_compression_lock = threading.Lock()


def yaml_save(record: BenchmarkRecord, fp: IO, fdoptions: FileDriverOptions) -> None:
    try:
        import yaml
    except ImportError:
        raise ModuleNotFoundError("`pyyaml` is not installed")

    bms = record.compact(mode=fdoptions.ctxmode)
    yaml.safe_dump(bms, fp, **fdoptions.options)


def yaml_load(fp: IO, fdoptions: FileDriverOptions) -> BenchmarkRecord:
    try:
        import yaml
    except ImportError:
        raise ModuleNotFoundError("`pyyaml` is not installed")

    bms = yaml.safe_load(fp)
    return BenchmarkRecord.expand(bms)


def json_save(record: BenchmarkRecord, fp: IO, fdoptions: FileDriverOptions) -> None:
    import json

    benchmarks = record.compact(mode=fdoptions.ctxmode)
    json.dump(benchmarks, fp, **fdoptions.options)


def json_load(fp: IO, fdoptions: FileDriverOptions) -> BenchmarkRecord:
    import json

    benchmarks: list[dict[str, Any]] = json.load(fp, **fdoptions.options)
    return BenchmarkRecord.expand(benchmarks)


def csv_save(record: BenchmarkRecord, fp: IO, fdoptions: FileDriverOptions) -> None:
    import csv

    benchmarks = record.compact(mode=fdoptions.ctxmode)
    writer = csv.DictWriter(fp, fieldnames=benchmarks[0].keys(), **fdoptions.options)

    for bm in benchmarks:
        writer.writerow(bm)


def csv_load(fp: IO, fdoptions: FileDriverOptions) -> BenchmarkRecord:
    import csv

    reader = csv.DictReader(fp, **fdoptions.options)

    benchmarks: list[dict[str, Any]] = []
    # apparently csv.DictReader has no appropriate type hint for __next__,
    # so we supply one ourselves.
    bm: dict[str, Any]
    for bm in reader:
        benchmarks.append(bm)

    return BenchmarkRecord.expand(benchmarks)


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
        self.drivers = drivers or _file_drivers
        self.compressions = compressions or _compressions

    def read(
        self,
        file: str | os.PathLike[str],
        driver: str | None = None,
        ctxmode: Literal["flatten", "inline", "omit"] = "inline",
        options: dict[str, Any] | None = None,
    ) -> BenchmarkRecord:
        """
        Writes a benchmark record to the given file path.

        The file driver is chosen based on the extension found on the ``file`` path.

        Parameters
        ----------
        file: str | os.PathLike[str]
            The file name to write to.
        driver: str | None
            File driver implementation to use. If None, the file driver inferred from the
            given file path's extension will be used.
        ctxmode: Literal["flatten", "inline", "omit"]
            How to handle the benchmark context when writing the record data.
        options: dict[str, Any] | None
            Options to pass to the respective file driver implementation.

        Returns
        -------
        BenchmarkRecord
            The benchmark record contained in the file.

        Raises
        ------
        KeyError
            If the given file does not have a driver implementation available.
        """
        driver = driver or Path(file).suffix.removeprefix(".")

        try:
            _, de = self.drivers[driver]
        except KeyError:
            raise ValueError(f"unsupported file format {driver!r}")

        fdoptions = FileDriverOptions(ctxmode=ctxmode, options=options or {})

        with open(file, "w") as fp:
            return de(fp, fdoptions)

    def read_batched(
        self,
        records: Sequence[BenchmarkRecord],
        file: str | os.PathLike[str],
        driver: str | None = None,
        ctxmode: Literal["flatten", "inline", "omit"] = "inline",
        options: dict[str, Any] | None = None,
    ) -> None:
        """A batched version of ``FileIO.read()``."""
        pass

    def write(
        self,
        record: BenchmarkRecord,
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
        record: BenchmarkRecord
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
            ser(record, fp, fdoptions)

    def write_batched(
        self,
        records: Sequence[BenchmarkRecord],
        file: str | os.PathLike[str],
        driver: str | None = None,
        ctxmode: Literal["flatten", "inline", "omit"] = "inline",
        options: dict[str, Any] | None = None,
    ) -> None:
        """A batched version of ``FileIO.write()``."""
        driver = driver or Path(file).suffix.removeprefix(".")

        try:
            ser, _ = self.drivers[driver]
        except KeyError:
            raise KeyError(f"unsupported file format {driver!r}") from None

        fdoptions = FileDriverOptions(ctxmode=ctxmode, options=options or {})
        with open(file, "a+") as fp:
            for record in records:
                ser(record, fp, fdoptions)
