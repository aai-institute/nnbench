"""A framework for organizing and running benchmark workloads on machine learning models."""

from importlib.metadata import PackageNotFoundError, entry_points, version

try:
    __version__ = version("nnbench")
except PackageNotFoundError:
    # package is not installed
    pass

# TODO: This naming is unfortunate
from .core import benchmark, parametrize
from .reporter import BaseReporter, register_reporter
from .types import Benchmark, Params


def add_reporters():
    eps = entry_points()

    if hasattr(eps, "select"):  # Python 3.10+ / importlib.metadata >= 3.9.0
        reporters = eps.select(group="nnbench.reporters")
    else:
        reporters = eps.get("nnbench.reporters", [])  # type: ignore

    for rep in reporters:
        register_reporter(rep.name)


add_reporters()
