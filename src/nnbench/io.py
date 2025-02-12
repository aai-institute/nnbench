import os
import re
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, AnyStr, Protocol, cast

from nnbench.reporter.base import BenchmarkReporter
from nnbench.types import BenchmarkRecord

if TYPE_CHECKING:
    from _typeshed import OpenBinaryMode, OpenTextMode, SupportsRead, SupportsWrite


class NamedReader(SupportsRead[AnyStr]):
    @property
    def name(self) -> str: ...

    def __iter__(self) -> AnyStr: ...


class NamedWriter(SupportsWrite[AnyStr]):
    @property
    def name(self) -> str: ...


# TODO: Make common base class.
class BenchmarkFileIO(Protocol):
    def read(self, fp: NamedReader, options: dict[str, Any]) -> BenchmarkRecord: ...

    def write(self, record: BenchmarkRecord, fp: NamedWriter, options: dict[str, Any]) -> None: ...


class YAMLFileIO(BenchmarkFileIO):
    extensions = (".yaml", ".yml")

    def __init__(self):
        try:
            import yaml

            self.yaml = yaml
        except ImportError:
            raise ModuleNotFoundError("`pyyaml` is not installed")

    def read(self, fp: NamedReader, options: dict[str, Any]) -> BenchmarkRecord:
        del options
        bms = self.yaml.safe_load(fp)
        return BenchmarkRecord.expand(bms)

    def write(self, record: BenchmarkRecord, fp: NamedWriter, options: dict[str, Any]) -> None:
        self.yaml.safe_dump(record.to_json(), fp, **options)


class JSONFileIO(BenchmarkFileIO):
    extensions = (".json", ".ndjson")

    def __init__(self):
        import json

        self.json = json

    def read(self, fp: NamedReader, options: dict[str, Any]) -> BenchmarkRecord:
        newline_delimited = fp.name.endswith(".ndjson")
        benchmarks: list[dict[str, Any]]
        if newline_delimited:
            benchmarks = [self.json.loads(line, **options) for line in fp]
        else:
            benchmarks = self.json.load(fp, **options)
        return BenchmarkRecord.expand(benchmarks)

    def write(self, record: BenchmarkRecord, fp: NamedWriter, options: dict[str, Any]) -> None:
        newline_delimited = fp.name.endswith(".ndjson")
        if newline_delimited:
            bms = record.to_list()
            fp.write("\n".join([self.json.dumps(b, **options) for b in bms]))
        else:
            self.json.dump(record.to_json(), fp, **options)


class CSVFileIO(BenchmarkFileIO):
    extensions = (".csv",)

    def __init__(self):
        import csv

        self.csv = csv

    def read(self, fp: NamedReader, options: dict[str, Any]) -> BenchmarkRecord:
        import json

        reader = self.csv.DictReader(fp, **options)
        benchmarks: list[dict[str, Any]] = []
        # csv.DictReader has no appropriate type hint for __next__,
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

    def write(self, record: BenchmarkRecord, fp: NamedWriter, options: dict[str, Any]) -> None:
        bm = record.to_list()
        writer = self.csv.DictWriter(fp, fieldnames=bm[0].keys(), **options)
        writer.writeheader()

        for b in bm:
            writer.writerow(b)


class ParquetFileIO(BenchmarkFileIO):
    extensions = (".parquet", ".pq")

    def __init__(self):
        import pyarrow.parquet as pq

        self.pyarrow_parquet = pq

    def read(self, fp: NamedReader, options: dict[str, Any]) -> BenchmarkRecord:
        table = self.pyarrow_parquet.read_table(fp, **options)
        benchmarks: list[dict[str, Any]] = table.to_pylist()
        return BenchmarkRecord.expand(benchmarks)

    def write(self, record: BenchmarkRecord, fp: NamedWriter, options: dict[str, Any]) -> None:
        from pyarrow import Table

        table = Table.from_pylist(record.to_list())
        self.pyarrow_parquet.write_table(table, fp, **options)


def get_extension(f: str | os.PathLike[str] | IO) -> str:
    """
    Given a path or file-like object, returns file extension
    (can be the empty string, if the file has no extension).
    """
    if isinstance(f, str | os.PathLike):
        return Path(f).suffix
    else:
        return Path(f.name).suffix


def get_protocol(url: str | os.PathLike[str]) -> str:
    url = str(url)
    parts = re.split(r"(::|://)", url, maxsplit=1)
    if len(parts) > 1:
        return parts[0]
    return "file"


file_io_mapping: dict[str, type[BenchmarkFileIO]] = {
    ".yaml": YAMLFileIO,
    ".yml": YAMLFileIO,
    ".json": JSONFileIO,
    ".ndjson": JSONFileIO,
    ".csv": CSVFileIO,
    ".parquet": ParquetFileIO,
    ".pq": ParquetFileIO,
}


class FileReporter(BenchmarkReporter):
    def _make_file(
        self, file_like: str | os.PathLike[str] | IO, mode: OpenTextMode | OpenBinaryMode
    ) -> IO:
        if hasattr(file_like, "read") or hasattr(file_like, "write"):
            return cast(IO, file_like)
        elif isinstance(file_like, str | os.PathLike):
            protocol = get_protocol(file_like)
            fd: IO
            if protocol == "file":
                fd = open(file_like, mode)
            else:
                try:
                    import fsspec
                except ImportError:
                    raise RuntimeError("non-local URIs require the fsspec package")
                fs = fsspec.filesystem(protocol)
                # NB(njunge): I sure hope this is standardized by fsspec
                fd = fs.open(file_like, mode)
            return fd
        raise TypeError("filename must be a str, bytes, file or PathLike object")

    def read(
        self,
        file: str | os.PathLike[str] | IO[str],
        mode: OpenBinaryMode | OpenTextMode = "r",
        options: dict[str, Any] | None = None,
    ) -> BenchmarkRecord:
        """
        Reads a benchmark record from the given file path.

        The file IO is chosen based on the extension in the ``file`` pathname.

        Parameters
        ----------
        file: str | os.PathLike[str] | IO[str]
            The file name, or object, to read from.
        mode: str
            Mode to use when opening a new file from a path.
            Can be any of the read modes supported by built-in ``open()``.
        options: dict[str, Any] | None
            Options to pass to the respective file IO implementation.

        Returns
        -------
        BenchmarkRecord
            The benchmark record contained in the file.

        Raises
        ------
        ValueError
            If the extension of the given file is not supported.
        """

        fd = self._make_file(file, mode=mode)

        ext = get_extension(fd)
        try:
            file_io = file_io_mapping[ext]()
        except KeyError:
            raise ValueError(f"unimplemented benchmark file format {ext!r}") from None
        with fd as fp:
            return file_io.read(fp, options or {})  # type: ignore

    def write(
        self,
        record: BenchmarkRecord,
        file: str | os.PathLike[str] | IO[str],
        mode: OpenBinaryMode | OpenTextMode = "w",
        options: dict[str, Any] | None = None,
    ) -> None:
        """
        Writes a benchmark record to the given file path.

        The file IO is chosen based on the extension found on the ``file`` path.

        Parameters
        ----------
        record: BenchmarkRecord
            The record to write to the database.
        file: str | os.PathLike[str]
            The file name, or object, to write to.
        mode: str
            Mode to use when opening a new file from a path.
            Can be any of the write modes supported by built-in ``open()``.
        options: dict[str, Any] | None
            Options to pass to the respective file IO implementation.

        Raises
        ------
        ValueError
            If the extension of the given file is not supported.
        """
        fd = self._make_file(file, mode=mode)
        ext = get_extension(fd)
        try:
            file_io = file_io_mapping[ext]()
        except KeyError:
            raise ValueError(f"unimplemented benchmark file format {ext!r}") from None
        with fd as fp:
            file_io.write(record, fp, options or {})  # type: ignore
