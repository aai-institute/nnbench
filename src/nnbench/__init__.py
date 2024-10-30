"""A framework for organizing and running benchmark workloads on machine learning models."""

from .core import benchmark, parametrize, product
from .reporter import BenchmarkReporter
from .runner import BenchmarkRunner
from .types import Benchmark, BenchmarkRecord, Memo, Parameters

__version__ = "0.3.0"


# TODO: This isn't great, make it functional instead?
def default_runner() -> BenchmarkRunner:
    return BenchmarkRunner()


def default_reporter() -> BenchmarkReporter:
    return BenchmarkReporter()
