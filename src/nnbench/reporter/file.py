import os
from pathlib import Path
from typing import IO, Any, Literal, cast

from nnbench.reporter.util import get_extension, get_protocol
from nnbench.types import BenchmarkReporter, BenchmarkResult


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


class FileReporter(BenchmarkReporter):
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
                return [BenchmarkResult.from_json(benchmarks)]

    def to_json(
        self,
        result: BenchmarkResult,
        file: str | os.PathLike[str],
        **kwargs: Any,
    ) -> None:
        import json

        newline_delimited = Path(file).suffix == ".ndjson"
        with make_file_descriptor(file, mode="w") as fd:
            if newline_delimited:
                fd.write("\n".join([json.dumps(r, **kwargs) for r in result.to_records()]))
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

    def read(self, file: str | os.PathLike[str], **kwargs: Any) -> list[BenchmarkResult]:
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
        list[BenchmarkResult]
            The benchmark results contained in the file.

        Raises
        ------
        ValueError
            If the extension of the given filename is not supported.
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

    def write(self, result: BenchmarkResult, file: str | os.PathLike[str], **kwargs: Any) -> None:
        """
        Writes multiple benchmark results to the given file path.

        The writing is chosen based on the extension found on the ``file`` path.

        Extensions ``json``, ``ndjson``, ``yaml``, and ``parquet`` are supported,
        as well as abbreviations ``.yml`` and ``.pq``.

        Parameters
        ----------
        result: BenchmarkResult
            The benchmark result to write to a file.
        file: str | os.PathLike[str]
            The file name, or object, to write to.
        **kwargs: Any | None
            Options to pass to the respective file writer implementation.

        Raises
        ------
        ValueError
            If the extension of the given filename is not supported.
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
