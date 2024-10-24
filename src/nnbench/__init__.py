"""A framework for organizing and running benchmark workloads on machine learning models."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nnbench")
except PackageNotFoundError:
    # package is not installed
    pass

from .core import benchmark, parametrize, product
from .reporter import BenchmarkReporter
from .runner import BenchmarkRunner
from .types import Benchmark, BenchmarkRecord, Memo, Parameters


# TODO: This isn't great, make it functional instead?
def default_runner() -> BenchmarkRunner:
    return BenchmarkRunner()


def default_reporter() -> BenchmarkReporter:
    return BenchmarkReporter()
