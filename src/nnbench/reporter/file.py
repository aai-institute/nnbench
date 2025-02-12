import os
import re
from pathlib import Path
from typing import IO, Any, Literal, Protocol, cast

from nnbench.types import BenchmarkRecord


def make_file_descriptor(
    file: str | os.PathLike[str] | IO,
    mode: Literal["r", "w", "rb", "wb"],  # TODO: Extend this to append/x modes
    **open_kwargs: Any,
) -> IO:
    if hasattr(file, "read") or hasattr(file, "write"):
        return cast(IO, file)
    elif isinstance(file, str | os.PathLike):
        protocol = get_protocol(file)
        fd: IO
        if protocol == "file":
            fd = open(file, mode, **open_kwargs)
        else:
            try:
                import fsspec
            except ImportError:
                raise RuntimeError("non-local URIs require the fsspec package")
            fs = fsspec.filesystem(protocol)  # fs: AbstractFileSystem
            # NB(njunge): I sure hope this is standardized by fsspec
            fd = fs.open(file, mode, **open_kwargs)
        return fd
    raise TypeError("filename must be a str, bytes, file or PathLike object")


class BenchmarkFileIO(Protocol):
    def read(self, fp: str | os.PathLike[str], options: dict[str, Any]) -> BenchmarkRecord: ...

    def write(
        self, record: BenchmarkRecord, fp: str | os.PathLike[str], options: dict[str, Any]
    ) -> None: ...


class YAMLFileIO(BenchmarkFileIO):
    extensions = (".yaml", ".yml")

    def __init__(self):
        try:
            import yaml

            self.yaml = yaml
        except ImportError:
            raise ModuleNotFoundError("`pyyaml` is not installed")

    def read(self, fp: str | os.PathLike[str], options: dict[str, Any]) -> BenchmarkRecord:
        del options
        with make_file_descriptor(fp, mode="r") as fd:
            bms = self.yaml.safe_load(fd)
        return BenchmarkRecord.expand(bms)

    def write(
        self, record: BenchmarkRecord, fp: str | os.PathLike[str], options: dict[str, Any]
    ) -> None:
        with make_file_descriptor(fp, mode="w") as fd:
            self.yaml.safe_dump(record.to_json(), fd, **options)


class JSONFileIO(BenchmarkFileIO):
    extensions = (".json", ".ndjson")

    def __init__(self):
        import json

        self.json = json

    def read(self, fp: str | os.PathLike[str], options: dict[str, Any]) -> BenchmarkRecord:
        newline_delimited = Path(fp).suffix == ".ndjson"
        benchmarks: list[dict[str, Any]]
        with make_file_descriptor(fp, mode="r") as fd:
            if newline_delimited:
                benchmarks = [self.json.loads(line, **options) for line in fd]
            else:
                benchmarks = self.json.load(fd, **options)
            return BenchmarkRecord.expand(benchmarks)

    def write(
        self, record: BenchmarkRecord, fp: str | os.PathLike[str], options: dict[str, Any]
    ) -> None:
        newline_delimited = Path(fp).suffix == ".ndjson"
        with make_file_descriptor(fp, mode="w") as fd:
            if newline_delimited:
                bms = record.to_list()
                fd.write("\n".join([self.json.dumps(b, **options) for b in bms]))
            else:
                self.json.dump(record.to_json(), fd, **options)


class CSVFileIO(BenchmarkFileIO):
    extensions = (".csv",)

    def __init__(self):
        import csv

        self.csv = csv

    def read(self, fp: str | os.PathLike[str], options: dict[str, Any]) -> BenchmarkRecord:
        import json

        with make_file_descriptor(fp, mode="r") as fd:
            reader = self.csv.DictReader(fd, **options)
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

    def write(
        self, record: BenchmarkRecord, fp: str | os.PathLike[str], options: dict[str, Any]
    ) -> None:
        bm = record.to_list()
        with make_file_descriptor(fp, mode="w") as fd:
            writer = self.csv.DictWriter(fd, fieldnames=bm[0].keys(), **options)
            writer.writeheader()

            for b in bm:
                writer.writerow(b)


class ParquetFileIO(BenchmarkFileIO):
    extensions = (".parquet", ".pq")

    def __init__(self):
        import pyarrow.parquet as pq

        self.pyarrow_parquet = pq

    def read(self, fp: str | os.PathLike[str], options: dict[str, Any]) -> BenchmarkRecord:
        table = self.pyarrow_parquet.read_table(str(fp), **options)
        benchmarks: list[dict[str, Any]] = table.to_pylist()
        return BenchmarkRecord.expand(benchmarks)

    def write(
        self, record: BenchmarkRecord, fp: str | os.PathLike[str], options: dict[str, Any]
    ) -> None:
        from pyarrow import Table

        table = Table.from_pylist(record.to_list())
        self.pyarrow_parquet.write_table(table, str(fp), **options)


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


_file_io_mapping: dict[str, type[BenchmarkFileIO]] = {
    ".yaml": YAMLFileIO,
    ".yml": YAMLFileIO,
    ".json": JSONFileIO,
    ".ndjson": JSONFileIO,
    ".csv": CSVFileIO,
    ".parquet": ParquetFileIO,
    ".pq": ParquetFileIO,
}


def get_file_io_class(file: str | os.PathLike[str]) -> BenchmarkFileIO:
    ext = get_extension(file)
    try:
        return _file_io_mapping[ext]()
    except KeyError:
        raise ValueError(f"unsupported benchmark file format {ext!r}") from None


def register_file_io_class(name: str, klass: type[BenchmarkFileIO], clobber: bool = False) -> None:
    if name in _file_io_mapping and not clobber:
        raise RuntimeError(
            f"driver {name!r} is already registered "
            f"(to force registration, rerun with clobber=True)"
        )
    _file_io_mapping[name] = klass


class FileReporter:
    def read(
        self,
        file: str | os.PathLike[str],
        options: dict[str, Any] | None = None,
    ) -> BenchmarkRecord:
        """
        Reads a benchmark record from the given file path.

        The file IO is chosen based on the extension in the ``file`` pathname.

        Parameters
        ----------
        file: str | os.PathLike[str]
            The file name, or object, to read from.
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

        ext = get_extension(file)
        file_io = get_file_io_class(ext)
        return file_io.read(file, options or {})

    def write(
        self,
        record: BenchmarkRecord,
        file: str | os.PathLike[str],
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
        options: dict[str, Any] | None
            Options to pass to the respective file IO implementation.

        Raises
        ------
        ValueError
            If the extension of the given file is not supported.
        """
        ext = get_extension(file)
        file_io = get_file_io_class(ext)
        file_io.write(record, file, options or {})
