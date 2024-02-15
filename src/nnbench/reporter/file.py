import os
import threading
from pathlib import Path
from typing import IO, Any, Callable

from nnbench.context import Context

from nnbench.reporter.base import BenchmarkReporter
from nnbench.types import BenchmarkRecord

_Options = dict[str, Any]
SerDe = tuple[
    Callable[[BenchmarkRecord, IO, _Options], None], Callable[[IO, _Options], BenchmarkRecord]
]


_file_drivers: dict[str, SerDe] = {}
_file_driver_lock = threading.Lock()


def yaml_save(record: BenchmarkRecord, fp: IO, options: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError:
        raise ModuleNotFoundError("`pyyaml` is not installed")

    yaml.safe_dump(record, fp, **options)


def yaml_load(fp: IO, options: dict[str, Any]) -> BenchmarkRecord:
    try:
        import yaml
    except ImportError:
        raise ModuleNotFoundError("`pyyaml` is not installed")

    # takes no options, but the slot is useful for passing options to file loaders.
    obj = yaml.safe_load(fp)
    return BenchmarkRecord(context=obj["context"], benchmarks=obj["benchmarks"])


def json_save(record: BenchmarkRecord, fp: IO, options: dict[str, Any]) -> None:
    import json

    context, benchmarks = record["context"], record["benchmarks"]
    for bm in benchmarks:
        bm["context"] = context
    json.dump(benchmarks, fp, **options)


def json_load(fp: IO, options: dict[str, Any]) -> BenchmarkRecord:
    import json

    benchmarks: list[dict[str, Any]] = json.load(fp, **options)
    context = Context()
    for bm in benchmarks:
        context.update(bm.pop("context", {}))

    return BenchmarkRecord(context=context, benchmarks=benchmarks)


def csv_save(record: BenchmarkRecord, fp: IO, options: dict[str, Any]) -> None:
    import csv

    fieldnames = set(record["benchmarks"][0].keys()) | {"context"}
    writer = csv.DictWriter(fp, fieldnames=fieldnames, **options)

    context = record["context"]
    for bm in record["benchmarks"]:
        bm["context"] = context
        writer.writerow(bm)


def csv_load(fp: IO, options: dict[str, Any]) -> BenchmarkRecord:
    import csv

    reader = csv.DictReader(fp, **options)

    context = Context()
    benchmarks: list[dict[str, Any]] = []

    # apparently csv.DictReader has no appropriate type hint for __next__,
    # so we supply one ourselves.
    bm: dict[str, Any]
    for bm in reader:
        context.update(bm.pop("context", {}))
        benchmarks.append(bm)

    return BenchmarkRecord(context=context, benchmarks=benchmarks)


with _file_driver_lock:
    _file_drivers["yaml"] = (yaml_save, yaml_load)
    _file_drivers["json"] = (json_save, json_load)
    _file_drivers["csv"] = (csv_save, csv_load)
    # TODO: Add parquet support


class FileReporter(BenchmarkReporter):
    def __init__(self):
        super().__init__()

    def read(
        self,
        file: str | os.PathLike[str],
        driver: str | None = None,
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
            _, de = _file_drivers[driver]
        except KeyError:
            raise ValueError(f"unsupported file format {driver!r}")

        with open(file, "w") as fp:
            return de(fp, options or {})

    def write(
        self,
        record: BenchmarkRecord,
        file: str | os.PathLike[str],
        driver: str | None = None,
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
        options: dict[str, Any] | None
            Options to pass to the respective file driver implementation.

        Raises
        ------
        KeyError
            If the given file does not have a driver implementation available.
        """
        driver = driver or Path(file).suffix.removeprefix(".")

        try:
            ser, _ = _file_drivers[driver]
        except KeyError:
            raise KeyError(f"unsupported file format {driver!r}") from None

        with open(file, "w") as fp:
            ser(record, fp, options or {})
