"""
An interface for displaying, writing, or streaming benchmark results to
files, databases, or web services.
"""

import os

from .console import ConsoleReporter
from .file import BenchmarkFileIO, FileReporter
from .service import BenchmarkServiceIO, MLFlowIO

_known_reporters = {
    "stdout": ConsoleReporter,
    "s3": FileReporter,
    "gs": FileReporter,
    "gcs": FileReporter,
    "az": FileReporter,
    "lakefs": FileReporter,
    "file": FileReporter,
    "mlflow": MLFlowIO,
}


def get_reporter_implementation(
    uri: str | os.PathLike[str],
) -> BenchmarkFileIO | BenchmarkServiceIO:
    import sys

    if uri is sys.stdout:
        proto = "stdout"
    else:
        from .util import get_protocol

        proto = get_protocol(uri)
    try:
        return _known_reporters[proto]()
    except KeyError:
        raise ValueError(f"unsupported benchmark IO protocol {proto!r}") from None


def register_io_implementation(name: str, klass: type, clobber: bool = False) -> None:
    if name in _known_reporters and not clobber:
        raise RuntimeError(
            f"benchmark IO {name!r} is already registered "
            f"(to force registration, rerun with clobber=True)"
        )
    _known_reporters[name] = klass


del os
