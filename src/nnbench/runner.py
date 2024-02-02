"""The abstract benchmark runner interface, which can be overridden for custom benchmark workloads."""
from __future__ import annotations

import contextlib
import inspect
import logging
import os
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Sequence, get_origin

from nnbench.context import ContextProvider
from nnbench.types import Benchmark, BenchmarkRecord, Parameters
from nnbench.util import import_file_as_module, ismodule

logger = logging.getLogger(__name__)


def iscontainer(s: Any) -> bool:
    return isinstance(s, (tuple, list))


def isdunder(s: str) -> bool:
    return s.startswith("__") and s.endswith("__")


@contextlib.contextmanager
def timer(bm: dict[str, Any]) -> Generator[None, None, None]:
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        end = time.perf_counter_ns()
        bm["time_ns"] = end - start


class BenchmarkRunner:
    """An abstract benchmark runner class."""

    benchmark_type = Benchmark

    def __init__(self):
        self.benchmarks: list[Benchmark] = list()

    def _check(self, params: dict[str, Any]) -> None:
        param_types = {k: type(v) for k, v in params.items()}
        allvars: dict[str, tuple[type, Any]] = {}
        empty = inspect.Parameter.empty

        def _issubtype(t1: type, t2: type) -> bool:
            """Small helper to make typechecks work on generics."""

            def _canonicalize(t: type) -> type:
                t_origin = get_origin(t)
                if t_origin is not None:
                    return t_origin
                return t

            if t1 == t2:
                return True

            t1 = _canonicalize(t1)
            t2 = _canonicalize(t2)
            if not inspect.isclass(t1):
                return False
            # TODO: Extend typing checks to args.
            return issubclass(t1, t2)

        for bm in self.benchmarks:
            for var in bm.interface.variables:
                name, typ, default = var
                if name in params and default != empty:
                    logger.debug(
                        f"using given value {params[name]} over default value {default} "
                        f"for parameter {name!r} in benchmark {bm.fn.__name__}()"
                    )

                if typ == empty:
                    logger.debug(f"parameter {name!r} untyped in benchmark {bm.fn.__name__}().")

                if name in allvars:
                    currvar = allvars[name]
                    orig_type, orig_val = new_type, new_val = currvar
                    # If a benchmark has a variable without a default value,
                    # that variable is taken into the combined interface as no-default.
                    if default == empty:
                        new_val = default
                    # These types need not be exact matches, just compatible.
                    # Two types are compatible iff either is a subtype of the other.
                    # We only log the narrowest type for each varname in the final interface,
                    # since that determines whether an input value is admissible.
                    if _issubtype(orig_type, typ):
                        pass
                    elif _issubtype(typ, orig_type):
                        new_type = typ
                    else:
                        raise TypeError(
                            f"got incompatible types {orig_type}, {typ} for parameter {name!r}"
                        )
                    newvar = (new_type, new_val)
                    if newvar != currvar:
                        allvars[name] = newvar
                else:
                    allvars[name] = (typ, default)

        for name, (typ, default) in allvars.items():
            # check if a no-default variable has no parameter.
            if name not in param_types and default == empty:
                raise ValueError(f"missing value for required parameter {name!r}")

            # skip the subsequent type check if the variable is untyped.
            if typ == empty:
                continue
            # type-check parameter value against the narrowest hinted type.
            if name in param_types and not _issubtype(param_types[name], typ):
                raise TypeError(
                    f"expected type {typ} for parameter {name!r}, got {param_types[name]}"
                )

    def clear(self) -> None:
        """Clear all registered benchmarks."""
        self.benchmarks.clear()

    def collect(self, path_or_module: str | os.PathLike[str], tags: tuple[str, ...] = ()) -> None:
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
                    self.collect(py, tags)
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
                if not tags or set(tags) & set(v.tags):
                    self.benchmarks.append(v)
            elif iscontainer(v):
                for bm in v:
                    if isinstance(bm, self.benchmark_type):
                        if not tags or set(tags) & set(bm.tags):
                            self.benchmarks.append(bm)

    def run(
        self,
        path_or_module: str | os.PathLike[str],
        params: dict[str, Any] | Parameters | None = None,
        tags: tuple[str, ...] = (),
        context: Sequence[ContextProvider] = (),
    ) -> BenchmarkRecord:
        """
        Run a previously collected benchmark workload.

        Parameters
        ----------
        path_or_module: str | os.PathLike[str]
            Name or path of the module to discover benchmarks in. Can also be a directory,
            in which case benchmarks are collected from the Python files therein.
        params: dict[str, Any] | Parameters | None
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
        BenchmarkRecord
            A JSON output representing the benchmark results. Has two top-level keys, "context"
            holding the context information, and "benchmarks", holding an array with the
            benchmark results.

        Raises
        ------
        ValueError
            If any context value is provided more than once.
        """
        if not self.benchmarks:
            self.collect(path_or_module, tags)

        # if we still have no benchmarks after collection, warn and return an empty record.
        if not self.benchmarks:
            warnings.warn(f"No benchmarks found in path/module {str(path_or_module)!r}.")
            return BenchmarkRecord(context={}, benchmarks=[])

        params = params or {}
        if isinstance(params, Parameters):
            dparams = asdict(params)
        else:
            dparams = params

        self._check(dparams)

        ctx: dict[str, Any] = dict()
        ctxkeys = set(ctx.keys())

        for provider in context:
            ctxval = provider()
            valkeys = set(ctxval.keys())
            # we do not allow multiple values for a context key.
            duplicates = ctxkeys & valkeys
            if duplicates:
                dupe, *_ = duplicates
                raise ValueError(f"got multiple values for context key {dupe!r}")
            ctx |= ctxval
            ctxkeys |= valkeys

        results: list[dict[str, Any]] = []
        for benchmark in self.benchmarks:
            bmparams = {k: v for k, v in dparams.items() if k in benchmark.interface.names}
            # TODO: Wrap this into an execution context
            res: dict[str, Any] = {
                "name": benchmark.name,
                "function": f"{benchmark.fn.__qualname__}.{benchmark.fn.__name__}",
                "description": benchmark.fn.__doc__,
                "date": datetime.now().isoformat(timespec="seconds"),
                "error_occurred": False,
                "error_message": "",
            }
            try:
                benchmark.setUp(**bmparams)
                with timer(res):
                    res["value"] = benchmark.fn(**bmparams)
            except Exception as e:
                res["error_occurred"] = True
                res["error_message"] = str(e)
            finally:
                benchmark.tearDown(**bmparams)
                results.append(res)

        return BenchmarkRecord(
            context=ctx,
            benchmarks=results,
        )
