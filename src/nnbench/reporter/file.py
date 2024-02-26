from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Callable, Literal, Sequence, cast

from nnbench.reporter.base import BenchmarkReporter
from nnbench.types import BenchmarkRecord


@dataclass(frozen=True)
class FileDriverOptions:
    options: dict[str, Any] = field(default_factory=dict)
    """Options to pass to the underlying serialization API call, e.g. ``json.dump``."""
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

    bm = []
    for r in records:
        bm += r.compact(mode=fdoptions.ctxmode)
    json.dump(bm, fp, **fdoptions.options)


def json_load(fp: IO, fdoptions: FileDriverOptions) -> list[BenchmarkRecord]:
    import json

    benchmarks: list[dict[str, Any]] = json.load(fp, **fdoptions.options)
    return [BenchmarkRecord.expand(benchmarks)]


def ndjson_save(records: Sequence[BenchmarkRecord], fp: IO, fdoptions: FileDriverOptions) -> None:
    import json

    bm = []
    for r in records:
        bm += r.compact(mode=fdoptions.ctxmode)
    fp.write("\n".join([json.dumps(b) for b in bm]))


def ndjson_load(fp: IO, fdoptions: FileDriverOptions) -> list[BenchmarkRecord]:
    import json

    benchmarks: list[dict[str, Any]]
    benchmarks = [json.loads(line, **fdoptions.options) for line in fp]
    return [BenchmarkRecord.expand(benchmarks)]


def csv_save(records: Sequence[BenchmarkRecord], fp: IO, fdoptions: FileDriverOptions) -> None:
    import csv

    bm = []
    for r in records:
        bm += r.compact(mode=fdoptions.ctxmode)
    writer = csv.DictWriter(fp, fieldnames=bm[0].keys(), **fdoptions.options)
    writer.writeheader()

    for b in bm:
        writer.writerow(b)


def csv_load(fp: IO, fdoptions: FileDriverOptions) -> list[BenchmarkRecord]:
    import csv
    import json

    reader = csv.DictReader(fp, **fdoptions.options)

    benchmarks: list[dict[str, Any]] = []
    # apparently csv.DictReader has no appropriate type hint for __next__,
    # so we supply one ourselves.
    bm: dict[str, Any]
    for bm in reader:
        benchmarks.append(bm)
        # it can happen that the context is inlined as a stringified JSON object
        # (e.g. in CSV), so we optionally JSON-load the context.
        for key in ("context", "_contextkeys"):
            if key in bm:
                strctx: str = bm[key]
                # TODO: This does not play nicely with doublequote, maybe re.sub?
                strctx = strctx.replace("'", '"')
                bm[key] = json.loads(strctx)
    return [BenchmarkRecord.expand(benchmarks)]


def parquet_save(records: Sequence[BenchmarkRecord], fp: IO, fdoptions: FileDriverOptions) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    bm = []
    for r in records:
        bm += r.compact(mode=fdoptions.ctxmode)

    table = pa.Table.from_pylist(bm)
    pq.write_table(table, fp, **fdoptions.options)


def parquet_load(fp: IO, fdoptions: FileDriverOptions) -> list[BenchmarkRecord]:
    import pyarrow.parquet as pq

    table = pq.read_table(fp, **fdoptions.options)
    benchmarks: list[dict[str, Any]] = table.to_pylist()
    return [BenchmarkRecord.expand(benchmarks)]


def get_driver_implementation(name: str) -> SerDe:
    try:
        return _file_drivers[name]
    except KeyError:
        raise KeyError(f"unsupported file format {name!r}") from None


def register_driver_implementation(name: str, impl: SerDe, clobber: bool = False) -> None:
    if name in _file_drivers and not clobber:
        raise RuntimeError(
            f"driver {name!r} is already registered (to force registration, "
            f"rerun with clobber=True)"
        )

    with _file_driver_lock:
        _file_drivers[name] = impl


def deregister_driver_implementation(name: str) -> SerDe | None:
    with _file_driver_lock:
        return _file_drivers.pop(name, None)


def gzip_compression(filename: str | os.PathLike[str], mode: Literal["r", "w"] = "r") -> IO:
    import gzip

    # gzip.GzipFile does not inherit from IO[bytes],
    # but it has all required methods, so we allow it.
    return cast(IO[bytes], gzip.GzipFile(filename=filename, mode=mode))


def bz2_compression(filename: str | os.PathLike[str], mode: Literal["r", "w"] = "r") -> IO:
    import bz2

    return bz2.BZ2File(filename=filename, mode=mode)


def get_compression_algorithm(name: str) -> Callable:
    try:
        return _compressions[name]
    except KeyError:
        raise KeyError(f"unsupported compression algorithm {name!r}") from None


def register_compression(name: str, impl: Callable, clobber: bool = False) -> None:
    if name in _compressions and not clobber:
        raise RuntimeError(
            f"compression {name!r} is already registered (to force registration, "
            f"rerun with clobber=True)"
        )

    with _compression_lock:
        _compressions[name] = impl


def deregister_compression(name: str) -> Callable | None:
    with _compression_lock:
        return _compressions.pop(name, None)


register_driver_implementation("yaml", (yaml_save, yaml_load))
register_driver_implementation("json", (json_save, json_load))
register_driver_implementation("ndjson", (ndjson_save, ndjson_load))
register_driver_implementation("csv", (csv_save, csv_load))
register_driver_implementation("parquet", (parquet_save, parquet_load))
register_compression("gz", gzip_compression)
register_compression("bz2", bz2_compression)


class FileIO:
    def read(
        self,
        file: str | os.PathLike[str],
        mode: str = "r",
        driver: str | None = None,
        compression: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> BenchmarkRecord:
        """
        Greedy version of ``FileIO.read_batched()``, returning the first read record.
        When reading a multi-record file, this uses as much memory as the batched version.
        """
        records = self.read_batched(
            file=file, mode=mode, driver=driver, compression=compression, options=options
        )
        return records[0]

    def read_batched(
        self,
        file: str | os.PathLike[str],
        mode: str = "r",
        driver: str | None = None,
        compression: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[BenchmarkRecord]:
        """
        Reads a set of benchmark records from the given file path.

        The file driver is chosen based on the extension found on the ``file`` path.

        Parameters
        ----------
        file: str | os.PathLike[str]
            The file name to read from.
        mode: str
            File mode to use. Can be any of the modes used in builtin ``open()``.
        driver: str | None
            File driver implementation to use. If None, the file driver inferred from the
            given file path's extension will be used.
        compression: str | None
            Compression engine to use. If None, the compression inferred from the given
            file path's extension will be used.
        options: dict[str, Any] | None
            Options to pass to the respective file driver implementation.

        Returns
        -------
        list[BenchmarkRecord]
            The benchmark records contained in the file.

        Raises
        ------
        ValueError
            If no registered file driver matches the file extension and no other driver
            was explicitly specified.
        """
        fileext = Path(file).suffix.removeprefix(".")
        # if the extension looks like FORMAT.COMPRESSION, we split.
        if fileext.count(".") == 1:
            # TODO: Are there file extensions with more than one meaningful part?
            ext_driver, ext_compression = fileext.rsplit(".", 1)
        else:
            ext_driver, ext_compression = fileext, None

        driver = driver or ext_driver
        compression = compression or ext_compression

        if driver is None:
            raise ValueError(
                f"could not infer a file driver to write file {str(file)!r}, "
                f"and no file driver was specified (available drivers: "
                f"{', '.join(repr(d) for d in _file_drivers)})"
            )
        _, de = get_driver_implementation(driver)

        if compression is not None:
            fd = get_compression_algorithm(compression)(file, mode)
        else:
            fd = open(file, mode)

        # dummy value, since the context mode is unused in read ops.
        fdoptions = FileDriverOptions(ctxmode="omit", options=options or {})

        with fd as fp:
            return de(fp, fdoptions)

    def write(
        self,
        record: BenchmarkRecord,
        file: str | os.PathLike[str],
        mode: str = "w",
        driver: str | None = None,
        compression: str | None = None,
        ctxmode: Literal["flatten", "inline", "omit"] = "inline",
        options: dict[str, Any] | None = None,
    ) -> None:
        """Greedy version of ``FileIO.write_batched()``"""
        self.write_batched(
            [record],
            file=file,
            mode=mode,
            driver=driver,
            compression=compression,
            ctxmode=ctxmode,
            options=options,
        )

    def write_batched(
        self,
        records: Sequence[BenchmarkRecord],
        file: str | os.PathLike[str],
        mode: str = "w",
        driver: str | None = None,
        compression: str | None = None,
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
        mode: str
            File mode to use. Can be any of the modes used in builtin ``open()``.
        driver: str | None
            File driver implementation to use. If None, the file driver inferred from the
            given file path's extension will be used.
        compression: str | None
            Compression engine to use. If None, the compression inferred from the given
            file path's extension will be used.
        ctxmode: Literal["flatten", "inline", "omit"]
            How to handle the benchmark context when writing the record data.
        options: dict[str, Any] | None
            Options to pass to the respective file driver implementation.

        Raises
        ------
        ValueError
            If no registered file driver matches the file extension and no other driver
            was explicitly specified.
        """
        fileext = Path(file).suffix.removeprefix(".")
        # if the extension looks like FORMAT.COMPRESSION, we split.
        if fileext.count(".") == 1:
            ext_driver, ext_compression = fileext.rsplit(".", 1)
        else:
            ext_driver, ext_compression = fileext, None

        driver = driver or ext_driver
        compression = compression or ext_compression

        if driver is None:
            raise ValueError(
                f"could not infer a file driver to write file {str(file)!r}, "
                f"and no file driver was specified (available drivers: "
                f"{', '.join(repr(d) for d in _file_drivers)})"
            )
        ser, _ = get_driver_implementation(driver)

        if compression is not None:
            fd = get_compression_algorithm(compression)(file, mode)
        else:
            fd = open(file, mode)

        fdoptions = FileDriverOptions(ctxmode=ctxmode, options=options or {})
        with fd as fp:
            ser(records, fp, fdoptions)


class FileReporter(FileIO, BenchmarkReporter):
    pass
