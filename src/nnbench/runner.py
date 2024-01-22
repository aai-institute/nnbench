"""The abstract benchmark runner interface, which can be overridden for custom benchmark workloads."""
from __future__ import annotations

import inspect
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from nnbench.context import ContextProvider
from nnbench.reporter import BaseReporter, reporter_registry
from nnbench.types import Benchmark, BenchmarkResult, Params
from nnbench.util import import_file_as_module, ismodule

logger = logging.getLogger(__name__)


def _check(params: dict[str, Any], benchmarks: list[Benchmark]) -> None:
    param_types = {k: type(v) for k, v in params.items()}
    benchmark_interface: dict[str, inspect.Parameter] = {}
    for bn in benchmarks:
        for name, param in inspect.signature(bn.fn).parameters.items():
            param_type = param.annotation
            if name in benchmark_interface and benchmark_interface[name].annotation != param_type:
                orig_type = benchmark_interface[name]
                raise TypeError(
                    f"got non-unique types {orig_type}, {
                        param_type} for parameter {name!r}"
                )
            benchmark_interface[name] = param
    for name, param in benchmark_interface.items():
        if name not in param_types and param.default == inspect.Parameter.empty:
            raise ValueError(f"missing value for parameter {name!r}")
        if not issubclass(param_types[name], param.annotation):
            raise TypeError(
                f"got wrong type for parameter {
                    name!r} (expected {param_types[name]!r})"
            )


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

        Raises
        ------
        ValueError
            If the given path is not a Python file, directory, or module name.
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
        path_or_module: str | os.PathLike[str],
        params: dict[str, Any] | Params,
        tags: tuple[str, ...] = (),
        context: Sequence[ContextProvider] = (),
    ) -> BenchmarkResult | None:
        """
        Run a previously collected benchmark workload.

        Parameters
        ----------
        path_or_module: str | os.PathLike[str]
            Name or path of the module to discover benchmarks in. Can also be a directory,
            in which case benchmarks are collected from the Python files therein.
        params: dict[str, Any] | Params
            Parameters to use for the benchmark run. Names have to match positional and keyword
            argument names of the benchmark functions.
        tags: tuple[str, ...]
            Tags to filter for when collecting benchmarks. Only benchmarks containing either of
            these tags are collected.
        context: Sequence[ContextProvider]
            Additional context to log with the benchmark in the output JSON record. Useful for
            obtaining environment information and configuration, like CPU/GPU hardware info,
            ML model metadata, and more.

        Returns
        -------
        BenchmarkResult | None
            A JSON output representing the benchmark results. Has two top-level keys, "context"
            holding the context information, and "benchmarks", holding an array with the
            benchmark results.
        """
        if not self.benchmarks:
            self.collect(path_or_module, tags)

        # if we still have no benchmarks after collection, warn.
        if not self.benchmarks:
            logger.warning(f"No benchmarks found in path/module {str(path_or_module)!r}.")
            return None

        if isinstance(params, Params):
            params_dict = asdict(params)
        else:
            params_dict = params  # type:ignore

        _check(params_dict, self.benchmarks)

        dcontext: dict[str, Any] = dict()

        for provider in context:
            ctxval = provider()
            if isinstance(ctxval, tuple):
                key, val = ctxval
                dcontext[key] = val
            else:
                # multi-value context information.
                for v in ctxval:
                    key, val = v
                    dcontext[key] = val

        results: list[dict[str, Any]] = []
        for benchmark in self.benchmarks:
            res: dict[str, Any] = {}
            # TODO: Validate against interface and pass only the kwargs relevant to the benchmark
            # params |= benchmark.params
            try:
                benchmark.setUp(**params_dict)
                # Todo: check params
                res["name"] = benchmark.fn.__name__
                res["value"] = benchmark.fn(**params_dict)
            except Exception as e:
                # TODO: This needs work
                res["error_occurred"] = True
                res["error_message"] = str(e)
            finally:
                benchmark.tearDown(**params_dict)
                results.append(res)

        return BenchmarkResult(
            context=dcontext,
            benchmarks=results,
        )

    def report(
        self, to: str | BaseReporter | Sequence[str | BaseReporter], result: BenchmarkResult
    ) -> None:
        """
        Report collected results from a previous run.

        Parameters
        ----------
        to: str | BaseReporter | Sequence[str | BaseReporter]
            The reporter to use when reporting / streaming results. Can be either a string
            (which prompts a lookup of all nnbench native reporters), a reporter instance,
            or a sequence thereof, which enables streaming result data to multiple sinks.
        result: BenchmarkResult
            The benchmark result to report.
        """

        def load_reporter(r: str | BaseReporter) -> BaseReporter:
            if isinstance(r, str):
                try:
                    return reporter_registry[r]()
                except KeyError:
                    # TODO: Add a note on nnbench reporter entry point once supported
                    raise KeyError(f"unknown reporter class {r!r}")
            else:
                return r

        dests: tuple[BaseReporter, ...] = ()

        if isinstance(to, (str, BaseReporter)):
            dests += (load_reporter(to),)
        else:
            dests += tuple(load_reporter(t) for t in to)

        for reporter in dests:
            reporter.report(result)
