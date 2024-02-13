from __future__ import annotations

import json
import os
from typing import IO, Any, Callable, List

from nnbench.reporter.base import BenchmarkReporter
from nnbench.types import BenchmarkRecord

ser = Callable[[IO, List[BenchmarkRecord], Any], None]
de = Callable[[IO, dict[str, Any]], List[BenchmarkRecord]]

# A registry of supported file loaders
_file_loaders: dict[str, tuple[ser, de]] = {}


# Register file loaders
def register_file_io(serializer: Callable, deserializer: Callable, file_type: str) -> None:
    """
    Registers a serializer and deserializer for a file type.

    Args:
    -----
        `serializer (Callable):` Defines how records are written to a file.
        `deserializer (Callable):` Defines how file contents are converted to `BenchmarkRecord`.
        `file_type (str):` File type extension (e.g., ".json", ".yaml").
    """
    _file_loaders[file_type] = (serializer, deserializer)


def _get_file_loader(file_type: str) -> tuple[ser, de]:
    """Helps retrieve registered file loaders of the given file_type with error handling"""
    file_loaders = _file_loaders.get(file_type)
    if not file_loaders:
        raise ValueError(f"File loaders for `{file_type}` files does not exist")
    return file_loaders


# json file loader:
def json_load(fp: IO, options: Any = None) -> List[BenchmarkRecord] | None:
    file_content = fp.read()
    if file_content:
        objs = [
            BenchmarkRecord(context=obj["context"], benchmarks=obj["benchmarks"])
            for obj in json.loads(file_content)
        ]
        return objs
    return None


def json_save(fp: IO, records: List[BenchmarkRecord], options: Any = None) -> None:
    fp.write(json.dumps(records))


# yaml file loader:
def yaml_load(fp: IO, options: Any = None) -> List[BenchmarkRecord] | None:
    try:
        import yaml
    except ImportError:
        raise ModuleNotFoundError("`pyyaml` is not installed")

    file_content = fp.read()
    if file_content:
        objs = [
            BenchmarkRecord(context=obj["context"], benchmarks=obj["benchmarks"])
            for obj in yaml.safe_load(file_content)
        ]
        return objs
    return None


def yaml_save(fp: IO, records: List[BenchmarkRecord], options: dict[str, Any] = None) -> None:
    try:
        import yaml
    except ImportError:
        raise ModuleNotFoundError("`pyyaml` is not installed")

    # To avoid `yaml.safe_dump()` error when trying to write numpy array
    for element in records[-1]["benchmarks"]:
        element["value"] = float(element["value"])
    yaml.safe_dump(records, fp, **(options or {}))


# Register json and yaml file loaders
register_file_io(json_save, json_load, file_type="json")
register_file_io(yaml_save, yaml_load, file_type="yaml")


class FileReporter(BenchmarkReporter):
    """
    Reports benchmark results to files in a given directory.

    This class implements a `BenchmarkReporter` subclass that persists benchmark
    records to files within a specified directory. It supports both reading and
    writing records, using file extensions to automatically determine the appropriate
    serialization format.

    Args:
    -----
        directory (str): The directory where benchmark files will be stored.

    Raises:
    -------
        BaseException: If the directory is not initialized.
    """

    def __init__(self, directory: str):
        self.directory = directory
        if not os.path.exists(directory):
            self.initialize()

    def initialize(self) -> None:
        os.makedirs(self.directory, exist_ok=True)

    def read(self, **kwargs: Any) -> List[BenchmarkRecord]:
        if not self.directory:
            raise BaseException("No directory is initialized")
        file_name = str(kwargs["file_name"])
        file_path = os.path.join(self.directory, file_name)
        file_type = file_name.split(".")[1]
        with open(file_path) as file:
            return _get_file_loader(file_type)[1](file, {})

    def write(self, record: BenchmarkRecord, **kwargs: dict[str, Any]) -> None:
        if not self.directory:
            raise BaseException("No directory is initialized")
        file_name = str(kwargs["file_name"])
        file_path = os.path.join(self.directory, file_name)
        # Create the file, if not already existing
        if not os.path.exists(file_path):
            with open(file_path, "w") as file:
                file.write("")
        prev_records = self.read(file_name=file_name)
        prev_records = prev_records if prev_records else []
        prev_records.append(record)  #
        file_type = file_name.split(".")[1]
        with open(file_path, "w") as file:
            _get_file_loader(file_type)[0](file, prev_records, {})
