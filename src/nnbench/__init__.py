"""A framework for organizing and running benchmark workloads on machine learning models."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nnbench")
except PackageNotFoundError:
    # package is not installed
    pass

# TODO: This naming is unfortunate
from .core import benchmark, parametrize
from .reporter import BaseReporter
from .types import Benchmark, Params
