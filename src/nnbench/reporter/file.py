import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any, Literal, cast

from nnbench.reporter.base import BenchmarkReporter
from nnbench.types import BenchmarkRecord

_Options = dict[str, Any]
SerDe = tuple[
    Callable[[BenchmarkRecord, IO, dict[str, Any]], None],
    Callable[[IO, dict[str, Any]], BenchmarkRecord],
]


_file_drivers: dict[str, SerDe] = {}
_compressions: dict[str, Callable] = {}
_file_driver_lock = threading.Lock()
_compression_lock = threading.Lock()


def yaml_save(record: BenchmarkRecord, fp: IO, options: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError:
        raise ModuleNotFoundError("`pyyaml` is not installed")

    yaml.safe_dump(record.to_json(), fp, **options)


def yaml_load(fp: IO, options: dict[str, Any]) -> BenchmarkRecord:
    try:
        import yaml
    except ImportError:
        raise ModuleNotFoundError("`pyyaml` is not installed")

    bms = yaml.safe_load(fp)
    return BenchmarkRecord.expand(bms)


def json_save(record: BenchmarkRecord, fp: IO, options: dict[str, Any]) -> None:
    import json

    json.dump(record.to_json(), fp, **options)


def json_load(fp: IO, options: dict[str, Any]) -> BenchmarkRecord:
    import json

    benchmarks = json.load(fp, **options)
    return BenchmarkRecord.expand(benchmarks)


def ndjson_save(record: BenchmarkRecord, fp: IO, options: dict[str, Any]) -> None:
    # mode is unused, since NDJSON requires every individual benchmark to be one line.
    import json

    bms = record.to_list()
    fp.write("\n".join([json.dumps(b, **options) for b in bms]))


def ndjson_load(fp: IO, options: dict[str, Any]) -> BenchmarkRecord:
    import json

    benchmarks: list[dict[str, Any]]
    benchmarks = [json.loads(line, **options) for line in fp]
    return BenchmarkRecord.expand(benchmarks)


def csv_save(record: BenchmarkRecord, fp: IO, options: dict[str, Any]) -> None:
    # mode is unused, since NDJSON requires every individual benchmark to be one line.
    import csv

    bm = record.to_list()
    writer = csv.DictWriter(fp, fieldnames=bm[0].keys(), **options)
    writer.writeheader()

    for b in bm:
        writer.writerow(b)


def csv_load(fp: IO, options: dict[str, Any]) -> BenchmarkRecord:
    import csv
    import json

    reader = csv.DictReader(fp, **options)

    benchmarks: list[dict[str, Any]] = []
    # apparently csv.DictReader has no appropriate type hint for __next__,
    # so we supply one ourselves.
    bm: dict[str, Any]
    for bm in reader:
        benchmarks.append(bm)
        # it can happen that the context is inlined as a stringified JSON object
        # (e.g. in CSV), so we optionally JSON-load the context.
        if "context" in bm:
            strctx: str = bm["context"]
            # TODO: This does not play nicely with doublequote, maybe re.sub?
            strctx = strctx.replace("'", '"')
            bm["context"] = json.loads(strctx)
    return BenchmarkRecord.expand(benchmarks)


def parquet_save(record: BenchmarkRecord, fp: IO, options: dict[str, Any]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(record.to_list())
    pq.write_table(table, fp, **options)


def parquet_load(fp: IO, options: dict[str, Any]) -> BenchmarkRecord:
    import pyarrow.parquet as pq

    table = pq.read_table(fp, **options)
    benchmarks: list[dict[str, Any]] = table.to_pylist()
    return BenchmarkRecord.expand(benchmarks)


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


def gzip_compression(
    filename: str | os.PathLike[str], mode: Literal["rb", "wb"] = "rb", compresslevel: int = 9
) -> IO:
    import gzip

    # gzip.GzipFile does not inherit from IO[bytes],
    # but it has all required methods, so we allow it.
    return cast(IO[bytes], gzip.GzipFile(filename=filename, mode=mode))


def bz2_compression(
    filename: str | os.PathLike[str], mode: Literal["rb", "wb"] = "rb", compresslevel: int = 9
) -> IO:
    import bz2

    return bz2.BZ2File(filename, mode, compresslevel=compresslevel)


def lzma_compression(
    filename: str | os.PathLike[str], mode: Literal["rb", "wb"] = "rb", compresslevel: int = 9
) -> IO:
    import lzma

    # not available for LZMA.
    del compresslevel
    return lzma.LZMAFile(filename, mode)


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


register_driver_implementation(".yaml", (yaml_save, yaml_load))
register_driver_implementation(".yml", (yaml_save, yaml_load))
register_driver_implementation(".json", (json_save, json_load))
register_driver_implementation(".ndjson", (ndjson_save, ndjson_load))
register_driver_implementation(".csv", (csv_save, csv_load))
register_driver_implementation(".parquet", (parquet_save, parquet_load))
register_compression(".gz", gzip_compression)
register_compression(".bz2", bz2_compression)
register_compression(".xz", lzma_compression)


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
        Reads a benchmark record from the given file path.

        The file driver is chosen based on the extension in the ``file`` pathname.

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
        BenchmarkRecord
            The benchmark record contained in the file.

        Raises
        ------
        ValueError
            If no registered file driver matches the file extension and no other driver
            was explicitly specified.
        """
        suffixes = Path(file).suffixes
        if len(suffixes) == 1:
            ext_driver, ext_compression = suffixes[0], None
        else:
            ext_driver = "".join(suffixes[:-1])
            ext_compression = suffixes[-1]

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

        with fd as fp:
            return de(fp, options or {})

    def write(
        self,
        record: BenchmarkRecord,
        file: str | os.PathLike[str],
        mode: str = "w",
        driver: str | None = None,
        compression: str | None = None,
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

        Raises
        ------
        ValueError
            If no registered file driver matches the file extension and no other driver
            was explicitly specified.
        """
        suffixes = Path(file).suffixes
        if len(suffixes) == 1:
            ext_driver, ext_compression = suffixes[0], None
        else:
            ext_driver = "".join(suffixes[:-1])
            ext_compression = suffixes[-1]

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

        with fd as fp:
            ser(record, fp, options or {})


class FileReporter(FileIO, BenchmarkReporter):
    pass
