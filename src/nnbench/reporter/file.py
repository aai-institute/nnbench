import os
from pathlib import Path
from typing import IO, Any, Literal, cast

from nnbench.reporter import BenchmarkReporter
from nnbench.reporter.util import get_extension, get_protocol
from nnbench.types import BenchmarkResult


def make_file_descriptor(
    file: str | os.PathLike[str] | IO,
    mode: Literal["r", "w", "a", "x", "rb", "wb", "ab", "xb"],
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
            fs = fsspec.filesystem(protocol)
            fd = fs.open(file, mode, **open_kwargs)
        return fd
    raise TypeError("filename must be a str, bytes, file or PathLike object")


class YAMLFileIO(BenchmarkReporter):
    extensions = (".yaml", ".yml")

    def read(self, fp: str | os.PathLike[str], **kwargs: Any) -> list[BenchmarkResult]:
        import yaml

        del kwargs
        with make_file_descriptor(fp, mode="r") as fd:
            bms = yaml.safe_load(fd)
        return BenchmarkResult.from_records(bms)

    def write(self, result: BenchmarkResult, fp: str | os.PathLike[str], **kwargs: Any) -> None:
        import yaml

        with make_file_descriptor(fp, mode="w") as fd:
            yaml.safe_dump(result.to_records(), fd, **kwargs)


class JSONFileIO(BenchmarkReporter):
    extensions = (".json", ".ndjson")

    def read(self, fp: str | os.PathLike[str], **kwargs: Any) -> BenchmarkResult:
        import json

        newline_delimited = Path(fp).suffix == ".ndjson"
        benchmarks: list[dict[str, Any]]
        with make_file_descriptor(fp, mode="r") as fd:
            if newline_delimited:
                benchmarks = [json.loads(line, **kwargs) for line in fd]
                return BenchmarkResult.from_records(benchmarks)
            else:
                benchmarks = json.load(fd, **kwargs)
                return BenchmarkResult.from_json(benchmarks)

    def write(self, result: BenchmarkResult, fp: str | os.PathLike[str], **kwargs: Any) -> None:
        import json

        newline_delimited = Path(fp).suffix == ".ndjson"
        with make_file_descriptor(fp, mode="w") as fd:
            if newline_delimited:
                bms = result.to_records()
                fd.write("\n".join([json.dumps(b, **kwargs) for b in bms]))
            else:
                json.dump(result.to_json(), fd, **kwargs)


class CSVFileIO(BenchmarkReporter):
    extensions = (".csv",)

    def read(self, fp: str | os.PathLike[str], **kwargs: Any) -> BenchmarkResult:
        import csv

        with make_file_descriptor(fp, mode="r") as fd:
            reader = csv.DictReader(fd, **kwargs)
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
                    bm["context"] = strctx
            return BenchmarkResult.from_records(benchmarks)

    def write(self, result: BenchmarkResult, fp: str | os.PathLike[str], **kwargs: Any) -> None:
        import csv

        bm = result.to_records()
        with make_file_descriptor(fp, mode="w") as fd:
            writer = csv.DictWriter(fd, fieldnames=bm[0].keys(), **kwargs)
            writer.writeheader()

            for b in bm:
                writer.writerow(b)


class ParquetFileIO(BenchmarkReporter):
    extensions = (".parquet", ".pq")

    def read(self, fp: str | os.PathLike[str], **kwargs: Any) -> list[BenchmarkResult]:
        import pyarrow.parquet as pq

        table = pq.read_table(str(fp), **kwargs)
        benchmarks: list[dict[str, Any]] = table.to_pylist()
        return BenchmarkResult.from_records(benchmarks)

    def write(self, result: BenchmarkResult, fp: str | os.PathLike[str], **kwargs: Any) -> None:
        import pyarrow.parquet as pq
        from pyarrow import Table

        table = Table.from_pylist(result.to_records())
        pq.write_table(table, str(fp), **kwargs)


_file_io_mapping: dict[str, type[BenchmarkReporter]] = {
    ".yaml": YAMLFileIO,
    ".yml": YAMLFileIO,
    ".json": JSONFileIO,
    ".ndjson": JSONFileIO,
    ".csv": CSVFileIO,
    ".parquet": ParquetFileIO,
    ".pq": ParquetFileIO,
}


def get_file_io_class(file: str | os.PathLike[str]) -> BenchmarkReporter:
    ext = get_extension(file)
    try:
        return _file_io_mapping[ext]()
    except KeyError:
        raise ValueError(f"unsupported benchmark file format {ext!r}") from None


def register_file_io_class(
    name: str, klass: type[BenchmarkReporter], clobber: bool = False
) -> None:
    if name in _file_io_mapping and not clobber:
        raise RuntimeError(
            f"driver {name!r} is already registered "
            f"(to force registration, rerun with clobber=True)"
        )
    _file_io_mapping[name] = klass


class FileReporter:
    @staticmethod
    def filter_open_kwds(kwargs: dict[str, Any]) -> dict[str, Any]:
        _OPEN_KWDS = ("buffering", "encoding", "errors", "newline", "closefd", "opener")
        return {k: v for k, v in kwargs.items() if k in _OPEN_KWDS}

    def from_json(self, file: str | os.PathLike[str], **kwargs: Any) -> list[BenchmarkResult]:
        import json

        newline_delimited = Path(file).suffix == ".ndjson"
        benchmarks: list[dict[str, Any]]
        with make_file_descriptor(file, mode="r") as fd:
            if newline_delimited:
                benchmarks = [json.loads(line, **kwargs) for line in fd]
                return BenchmarkResult.from_records(benchmarks)
            else:
                benchmarks = json.load(fd, **kwargs)
                return BenchmarkResult.from_json(benchmarks)

    def to_json(self, result: BenchmarkResult, file: str | os.PathLike[str], **kwargs: Any) -> None:
        import json

        newline_delimited = Path(file).suffix == ".ndjson"
        with make_file_descriptor(file, mode="w") as fd:
            if newline_delimited:
                bms = result.to_records()
                fd.write("\n".join([json.dumps(b, **kwargs) for b in bms]))
            else:
                json.dump(result.to_json(), fd, **kwargs)

    def from_yaml(self, fp: str | os.PathLike[str], **kwargs: Any) -> list[BenchmarkResult]:
        import yaml

        del kwargs
        with make_file_descriptor(fp, mode="r") as fd:
            bms = yaml.safe_load(fd)
        return BenchmarkResult.from_records(bms)

    def to_yaml(self, result: BenchmarkResult, fp: str | os.PathLike[str], **kwargs: Any) -> None:
        import yaml

        with make_file_descriptor(fp, mode="w") as fd:
            yaml.safe_dump(result.to_records(), fd, **kwargs)

    def from_parquet(self, file: str | os.PathLike[str], **kwargs: Any) -> list[BenchmarkResult]:
        import pyarrow.parquet as pq

        table = pq.read_table(str(file), **kwargs)
        benchmarks: list[dict[str, Any]] = table.to_pylist()
        return BenchmarkResult.from_records(benchmarks)

    def to_parquet(
        self, result: BenchmarkResult, file: str | os.PathLike[str], **kwargs: Any
    ) -> None:
        import pyarrow.parquet as pq
        from pyarrow import Table

        table = Table.from_pylist(result.to_records())
        pq.write_table(table, str(file), **kwargs)

    def read(self, file: str | os.PathLike[str], **kwargs: Any) -> BenchmarkResult:
        """
        Reads a benchmark record from the given file path.

        The reading implementation is chosen based on the extension in the ``file`` path.

        Extensions ``json``, ``ndjson``, ``yaml``, and ``parquet`` are supported,
        as well as abbreviations ``.yml`` and ``.pq``.

        Parameters
        ----------
        file: str | os.PathLike[str]
            The file name, or object, to read from.
        **kwargs: Any | None
            Options to pass to the respective file IO implementation.

        Returns
        -------
        BenchmarkResult
            The benchmark record contained in the file.

        Raises
        ------
        ValueError
            If the extension of the given file is not supported.
        """

        ext = get_extension(file)

        # TODO: Filter open keywords (in methods?)
        if ext in (".json", ".ndjson"):
            return self.from_json(file, **kwargs)
        elif ext in (".yml", ".yaml"):
            return self.from_yaml(file, **kwargs)
        elif ext in (".parquet", ".pq"):
            return self.from_parquet(file, **kwargs)
        else:
            raise ValueError(f"unsupported benchmark file format {ext!r}")

    def write(
        self,
        result: BenchmarkResult,
        file: str | os.PathLike[str],
        **kwargs: Any,
    ) -> None:
        """
        Writes a benchmark record to the given file path.

        The writing is chosen based on the extension found on the ``file`` path.

        Extensions ``json``, ``ndjson``, ``yaml``, and ``parquet`` are supported,
        as well as abbreviations ``.yml`` and ``.pq``.

        Parameters
        ----------
        result: BenchmarkResult
            The record to write to the database.
        file: str | os.PathLike[str]
            The file name, or object, to write to.
        **kwargs: Any | None
            Options to pass to the respective file writer implementation.

        Raises
        ------
        ValueError
            If the extension of the given file is not supported.
        """
        ext = get_extension(file)

        # TODO: Filter open keywords (in methods?)
        if ext in (".json", ".ndjson"):
            return self.to_json(result, file, **kwargs)
        elif ext in (".yml", ".yaml"):
            return self.to_yaml(result, file, **kwargs)
        elif ext in (".parquet", ".pq"):
            return self.to_parquet(result, file, **kwargs)
        else:
            raise ValueError(f"unsupported benchmark file format {ext!r}")
