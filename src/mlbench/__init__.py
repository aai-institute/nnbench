"""A framework for organizing and running benchmark workloads on machine learning models."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mlbench")
except PackageNotFoundError:
    # package is not installed
    pass
