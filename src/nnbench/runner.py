"""The abstract benchmark runner interface, which can be overridden for custom benchmark workloads."""

from __future__ import annotations

import collections
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
from typing import Any, Callable, Generator, Sequence, get_origin

from nnbench.context import Context, ContextProvider
from nnbench.types import Benchmark, BenchmarkRecord, Parameters, State
from nnbench.types.memo import is_memo, is_memo_type
from nnbench.util import import_file_as_module, ismodule

logger = logging.getLogger(__name__)


def iscontainer(s: Any) -> bool:
    return isinstance(s, (tuple, list))


def isdunder(s: str) -> bool:
    return s.startswith("__") and s.endswith("__")


def qualname(fn: Callable) -> str:
    if fn.__name__ == fn.__qualname__:
        return fn.__name__
    return f"{fn.__qualname__}.{fn.__name__}"


@contextlib.contextmanager
def timer(bm: dict[str, Any]) -> Generator[None, None, None]:
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        end = time.perf_counter_ns()
        bm["time_ns"] = end - start


class BenchmarkRunner:
    """
    An abstract benchmark runner class.

    Collects benchmarks from a module or file using the collect() method.
    Runs a previously collected benchmark workload with parameters in the run() method,
    outputting the results to a JSON-like document.

    Optionally checks input parameters against the benchmark function's interfaces,
    raising an error if the input types do not match the expected types.

    Parameters
    ----------
    typecheck: bool
        Whether to check parameter types against the expected benchmark input types.
        Type mismatches will result in an error before the workload is run.
    params_repr_hooks: dict[type, Callable]
        A mapping of data types to functions computing a compressed representation
        of benchmark parameters of said types. Used in self.params_repr().
    """

    benchmark_type = Benchmark

    def __init__(self, typecheck: bool = True, params_repr_hooks: dict[type, Callable] = None):
        self.benchmarks: list[Benchmark] = list()
        self.typecheck = typecheck
        self._params_repr_hooks = params_repr_hooks or {}

    def register_repr_hook(self, typ: type, fn: Callable) -> None:
        self._params_repr_hooks[typ] = fn

    def deregister_repr_hook(self, typ: type) -> Callable | None:
        return self._params_repr_hooks.pop(typ, None)

    def _check(self, params: dict[str, Any]) -> None:
        if not self.typecheck:
            return

        allvars: dict[str, tuple[type, Any]] = {}
        required: set[str] = set()
        empty = inspect.Parameter.empty

        def _issubtype(t1: type, t2: type) -> bool:
            """Small helper to make typechecks work on generics."""

            if t1 == t2:
                return True

            t1 = get_origin(t1) or t1
            t2 = get_origin(t2) or t2
            if not inspect.isclass(t1):
                return False
            # TODO: Extend typing checks to args.
            return issubclass(t1, t2)

        # stitch together the union interface comprised of all benchmarks.
        for bm in self.benchmarks:
            for var in bm.interface.variables:
                name, typ, default = var
                if default == empty:
                    required.add(name)
                if name in params and default != empty:
                    logger.debug(
                        f"using given value {params[name]} over default value {default} "
                        f"for parameter {name!r} in benchmark {bm.name}()"
                    )

                if typ == empty:
                    logger.debug(f"parameter {name!r} untyped in benchmark {bm.name}().")

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

        # check if any required variable has no parameter.
        missing = required - params.keys()
        if missing:
            msng, *_ = missing
            raise ValueError(f"missing value for required parameter {msng!r}")

        for k, v in params.items():
            if k not in allvars:
                warnings.warn(
                    f"ignoring parameter {k!r} since it is not part of any benchmark interface."
                )
                continue

            typ, default = allvars[k]
            # skip the subsequent type check if the variable is untyped.
            if typ == empty:
                continue

            vtype = type(v)
            if is_memo(v) and not is_memo_type(typ):
                # in case of a thunk, check the result type of __call__() instead.
                vtype = inspect.signature(v).return_annotation

            # type-check parameter value against the narrowest hinted type.
            if not _issubtype(vtype, typ):
                raise TypeError(f"expected type {typ} for parameter {k!r}, got {vtype}")

    def params_repr(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Compute a compressed representation of benchmark parameters.

        This is necessary to break reference cycles from the parameters to the records,
        which prevent garbage collection of memory-intensive values.

        Parameters
        ----------
        params: dict[str, Any]
            Benchmark parameters to compute a compressed representation of.

        Returns
        -------
        dict[str, Any]
            A compressed representation of the benchmark input parameters.
        """
        containers = (tuple, list, set, frozenset)
        natives = (float, int, str, bool, bytes, complex)
        compressed: dict[str, Any] = {}

        def _compress_impl(val):
            vtype = type(val)
            if vtype in self._params_repr_hooks:
                return self._params_repr_hooks[vtype](val)
            if isinstance(val, natives):
                # save native types without modification...
                return val
            else:
                # ... or return the string repr.
                return repr(val)

        for k, v in params.items():
            if isinstance(v, containers):
                container_type = type(v)
                compressed[k] = container_type(_compress_impl(vv) for vv in v)
            elif isinstance(v, dict):
                compressed[k] = self.params_repr(v)
            else:
                compressed[k] = _compress_impl(v)
        return compressed

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
        ppath = Path(path_or_module)
        if ppath.is_dir():
            pythonpaths = (p for p in ppath.iterdir() if p.suffix == ".py")
            for py in pythonpaths:
                logger.debug(f"Collecting benchmarks from submodule {py.name!r}.")
                self.collect(py, tags)
            return
        elif ppath.is_file():
            module = import_file_as_module(path_or_module)
        elif ismodule(path_or_module):
            module = sys.modules[str(path_or_module)]
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
        context: Sequence[ContextProvider] | Context = (),
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
        context: Sequence[ContextProvider] | Context
            Additional context to log with the benchmark in the output JSON record. Useful for
            obtaining environment information and configuration, like CPU/GPU hardware info,
            ML model metadata, and more.

        Returns
        -------
        BenchmarkRecord
            A JSON output representing the benchmark results. Has two top-level keys, "context"
            holding the context information, and "benchmarks", holding an array with the
            benchmark results.
        """
        if not self.benchmarks:
            self.collect(path_or_module, tags)

        family_sizes: dict[str, Any] = collections.defaultdict(int)
        family_indices: dict[str, Any] = collections.defaultdict(int)

        if isinstance(context, Context):
            ctx = context
        else:
            ctx = Context()
            for provider in context:
                ctx.add(provider)

        # if we didn't find any benchmarks, warn and return an empty record.
        if not self.benchmarks:
            warnings.warn(f"No benchmarks found in path/module {str(path_or_module)!r}.")
            return BenchmarkRecord(context=ctx, benchmarks=[])

        for bm in self.benchmarks:
            family_sizes[bm.fn.__name__] += 1

        if isinstance(params, Parameters):
            dparams = asdict(params)
        else:
            dparams = params or {}

        self._check(dparams)
        results: list[dict[str, Any]] = []

        def _maybe_dememo(v, expected_type):
            """Compute and memoize a value if a memo is given for a variable."""
            if is_memo(v) and not is_memo_type(expected_type):
                return v()
            return v

        for benchmark in self.benchmarks:
            bm_family = benchmark.fn.__name__
            state = State(
                name=benchmark.name,
                family=bm_family,
                family_size=family_sizes[bm_family],
                family_index=family_indices[bm_family],
            )
            family_indices[bm_family] += 1
            bmtypes = dict(zip(benchmark.interface.names, benchmark.interface.types))
            bmparams = dict(zip(benchmark.interface.names, benchmark.interface.defaults))
            # TODO: Does this need a copy.deepcopy()?
            bmparams |= {k: v for k, v in dparams.items() if k in bmparams}
            bmparams = {k: _maybe_dememo(v, bmtypes[k]) for k, v in bmparams.items()}

            # TODO: Wrap this into an execution context
            res: dict[str, Any] = {
                "name": benchmark.name,
                "function": qualname(benchmark.fn),
                "description": benchmark.fn.__doc__ or "",
                "date": datetime.now().isoformat(timespec="seconds"),
                "error_occurred": False,
                "error_message": "",
                "parameters": self.params_repr(bmparams),
            }
            try:
                benchmark.setUp(state, bmparams)
                with timer(res):
                    res["value"] = benchmark.fn(**bmparams)
            except Exception as e:
                res["error_occurred"] = True
                res["error_message"] = str(e)
            finally:
                benchmark.tearDown(state, bmparams)
                results.append(res)

        return BenchmarkRecord(
            context=ctx,
            benchmarks=results,
        )
