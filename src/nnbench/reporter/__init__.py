"""
An interface for displaying, writing, or streaming benchmark results to
files, databases, or web services.
"""

import os

from nnbench.types import BenchmarkReporter

from .console import ConsoleReporter
from .file import FileReporter
from .mlflow import MLFlowReporter
from .sqlite import SQLiteReporter

_known_reporters: dict[str, type[BenchmarkReporter]] = {
    "stdout": ConsoleReporter,
    "s3": FileReporter,
    "gs": FileReporter,
    "gcs": FileReporter,
    "az": FileReporter,
    "lakefs": FileReporter,
    "file": FileReporter,
    "mlflow": MLFlowReporter,
    "sqlite": SQLiteReporter,
}


def get_reporter_implementation(uri: str | os.PathLike[str]) -> BenchmarkReporter:
    import sys

    if uri is sys.stdout:
        proto = "stdout"
    else:
        from .util import get_protocol

        proto = get_protocol(uri)
    try:
        return _known_reporters[proto]()
    except KeyError:
        raise ValueError(f"no benchmark reporter registered for format {proto!r}") from None


def register_reporter_implementation(
    name: str, klass: type[BenchmarkReporter], clobber: bool = False
) -> None:
    if name in _known_reporters and not clobber:
        raise RuntimeError(
            f"benchmark reporter {name!r} is already registered "
            f"(to force registration, rerun with clobber=True)"
        )
    _known_reporters[name] = klass


del os
