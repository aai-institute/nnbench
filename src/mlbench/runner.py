"""The abstract benchmark runner interface, which can be overridden for custom benchmark workloads."""

import sys
from typing import Any

from mlbench.benchmark import Benchmark

BenchmarkResult = dict[str, Any]


def is_dunder(s: str) -> bool:
    return s.startswith("__") and s.endswith("__")


class AbstractBenchmarkRunner:
    """An abstract benchmark runner class."""

    benchmark_type = Benchmark

    def __init__(self):
        self.benchmarks: list[Benchmark] = list()

    # TODO (n.junge): Add regex/tag filters (and corresponding slots on the Benchmark)
    def discover(self, module: str = "__main__") -> None:
        # TODO: functools.cache this guy
        """
        Discover benchmarks in a module and memoize them for later use.

        Parameters
        ----------
        module: str
            Name of the module to discover benchmarks in. Currently, only "__main__" (i.e. the caller module) is supported.
        """
        if module != "__main__":
            raise NotImplementedError("module discovery not implemented")

        # __main__ is always in sys.modules.
        for k, v in sys.modules[module].__dict__.items():
            if is_dunder(k):
                continue
            # memoize benchmarks.
            elif isinstance(v, self.benchmark_type):
                self.benchmarks.append(v)
            elif isinstance(v, list) and all(isinstance(b, self.benchmark_type) for b in v):
                self.benchmarks.extend(v)

        return None

    def run(self) -> list[BenchmarkResult]:
        """
        Run a previously collected benchmark workload.

        Returns
        -------
        list[BenchmarkResult]
            A list of JSON outputs representing the benchmark results.
        """
        if not self.benchmarks:
            self.discover()

        results: list[BenchmarkResult] = []
        for benchmark in self.benchmarks:
            # TODO (n.junge): Wrap this in execution context
            pkwargs = benchmark.setUp(**benchmark.params)
            res: BenchmarkResult = benchmark.fn(**pkwargs)
            benchmark.tearDown(**benchmark.params)
            results.append(res)

        # TODO: Once discovery is cached, the cache needs to be cleared here
        self.benchmarks.clear()
        return results

    def report(self) -> None:
        """Report collected results from a previous run."""
        raise NotImplementedError
