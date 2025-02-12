"""
An interface for displaying, writing, or streaming benchmark results to
files, databases, or web services.
"""

import os
from enum import Enum
from typing import TextIO

from .console import ConsoleReporter
from .file import FileReporter


class IOType(str, Enum):
    STDOUT = "stdout"
    FILE = "file"
    DATABASE = "database"
    SERVICE = "service"  # TODO: Pick a better name
    UNKNOWN = "unknown"


def get_io_type(uri: str | os.PathLike[str] | TextIO) -> IOType:
    import sys

    if uri is sys.stdout:
        return IOType.STDOUT

    from nnbench.util import get_protocol

    proto = get_protocol(uri)
    if proto in ["az", "file", "gcs", "gs", "lakefs", "s3"]:
        return IOType.FILE
    elif proto in ["sqlite"]:  # TODO: Not yet supported
        return IOType.DATABASE
    elif proto in ["mlflow"]:
        return IOType.SERVICE
    else:
        return IOType.UNKNOWN


del Enum, os, TextIO
