"""The abstract benchmark runner interface, which can be overridden for custom benchmark workloads."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from mlbench.core import Benchmark
from mlbench.util import import_file_as_module, ismodule

BenchmarkResult = dict[str, Any]

logger = logging.getLogger(__name__)


def iscontainer(s: Any) -> bool:
    return isinstance(s, (tuple, list))


def isdunder(s: str) -> bool:
    return s.startswith("__") and s.endswith("__")


class AbstractBenchmarkRunner:
    """An abstract benchmark runner class."""

    benchmark_type = Benchmark

    def __init__(self):
        self.benchmarks: list[Benchmark] = list()

    def clear(self) -> None:
        """Clear all registered benchmarks."""
        self.benchmarks.clear()

    def collect(
        self, path_or_module: str | os.PathLike[str] = "__main__", tags: tuple[str, ...] = ()
    ) -> None:
        # TODO: functools.cache this guy
        """
        Discover benchmarks in a module and memoize them for later use.

        Parameters
        ----------
        path_or_module: str | os.PathLike[str]
            Name or path of the module to discover benchmarks in. Can also be a directory,
            in which case benchmarks are collected from the Python files therein.
        tags: tuple[str, ...]
            Tags to filter for when collecting benchmarks. Only benchmarks containing either of
            these tags are collected.
        """
        if ismodule(path_or_module):
            module = sys.modules[str(path_or_module)]
        else:
            ppath = Path(path_or_module)
            if ppath.is_dir():
                pythonpaths = (p for p in ppath.iterdir() if p.suffix == ".py")
                for py in pythonpaths:
                    logger.debug(f"Collecting benchmarks from submodule {py.name!r}.")
                    self.collect(py)
                return
            elif ppath.is_file():
                module = import_file_as_module(path_or_module)
            else:
                raise ValueError(
                    f"expected a module name, Python file, or directory, "
                    f"got {str(path_or_module)!r}"
                )

        # iterate through the module dict members to register
        for k, v in module.__dict__.items():
            if isdunder(k):
                continue
            elif isinstance(v, self.benchmark_type):
                self.benchmarks.append(v)
            elif iscontainer(v):
                for bm in v:
                    if isinstance(bm, self.benchmark_type):
                        self.benchmarks.append(bm)

        # and finally, filter by tags.
        self.benchmarks = [b for b in self.benchmarks if set(tags) <= set(b.tags)]

    def run(
        self,
        path_or_module: str | os.PathLike[str] = "__main__",
        tags: tuple[str, ...] = (),
        clear: bool = False,
    ) -> list[BenchmarkResult]:
        """
        Run a previously collected benchmark workload.

        Parameters
        ----------
        path_or_module: str | os.PathLike[str]
            Name or path of the module to discover benchmarks in. Can also be a directory,
            in which case benchmarks are collected from the Python files therein.
        tags: tuple[str, ...]
            Tags to filter for when collecting benchmarks. Only benchmarks containing either of
            these tags are collected.
        clear: bool
            Unregister all available benchmarks after the run.

        Returns
        -------
        list[BenchmarkResult]
            A list of JSON outputs representing the benchmark results.
        """
        if not self.benchmarks:
            self.collect(path_or_module, tags)

        # if we still have no benchmarks after collection, warn.
        if not self.benchmarks:
            logger.warning(f"No benchmarks found in path/module {str(path_or_module)!r}.")

        results: list[BenchmarkResult] = []
        for benchmark in self.benchmarks:
            res: BenchmarkResult = {}
            try:
                benchmark.setUp(**benchmark.params)
                res.update(benchmark.fn(**benchmark.params))
            except Exception as e:
                # TODO: This needs work
                res["error_occurred"] = True
                res["error_message"] = str(e)
            finally:
                benchmark.tearDown(**benchmark.params)
                results.append(res)

        if clear:
            # TODO: Once discovery is cached, the cache needs to be cleared here
            #  (for _all_ subpaths in case of a directory)
            self.benchmarks.clear()
        return results

    def report(self) -> None:
        """Report collected results from a previous run."""
        raise NotImplementedError
