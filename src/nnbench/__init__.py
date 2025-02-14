"""A framework for organizing and running benchmark workloads on machine learning models."""

from .core import benchmark, parametrize, product
from .runner import collect, run
from .types import Benchmark, BenchmarkFamily, BenchmarkRecord, Parameters

__version__ = "0.4.0"
