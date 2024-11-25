"""The abstract benchmark runner interface, which can be overridden for custom benchmark workloads."""

import collections
import contextlib
import inspect
import logging
import os
import platform
import sys
import time
import uuid
import warnings
from collections.abc import Callable, Generator, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, get_origin

from nnbench.context import ContextProvider
from nnbench.fixtures import FixtureManager
from nnbench.types import Benchmark, BenchmarkRecord, Parameters, State
from nnbench.types.memo import is_memo, is_memo_type
from nnbench.util import import_file_as_module, ismodule

logger = logging.getLogger(__name__)


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
    """

    benchmark_type = Benchmark

    def __init__(self, typecheck: bool = True):
        self.benchmarks: list[Benchmark] = list()
        self.typecheck = typecheck

    def _check(self, params: dict[str, Any]) -> None:
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
                if default is empty:
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
                    if default is empty:
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

    def jsonify_params(
        self, params: dict[str, Any], repr_hooks: dict[type, Callable] | None = None
    ) -> dict[str, Any]:
        """
        Construct a JSON representation of benchmark parameters.

        This is necessary to break reference cycles from the parameters to the records,
        which prevent garbage collection of memory-intensive values.

        Parameters
        ----------
        params: dict[str, Any]
            Benchmark parameters to compute a JSON representation of.
        repr_hooks: dict[type, Callable] | None
            A dictionary mapping parameter types to functions returning a JSON representation
            of an instance of the type. Allows fine-grained control to achieve lossless,
            reproducible serialization of input parameter information.

        Returns
        -------
        dict[str, Any]
            A JSON-serializable representation of the benchmark input parameters.
        """
        repr_hooks = repr_hooks or {}
        natives = (float, int, str, bool, bytes, complex)
        json_params: dict[str, Any] = {}

        def _jsonify(val):
            vtype = type(val)
            if vtype in repr_hooks:
                return repr_hooks[vtype](val)
            if isinstance(val, natives):
                return val
            elif hasattr(val, "to_json"):
                return val.to_json()
            else:
                return str(val)

        for k, v in params.items():
            if isinstance(v, tuple | list | set | frozenset):
                container_type = type(v)
                json_params[k] = container_type(map(_jsonify, v))
            elif isinstance(v, dict):
                json_params[k] = self.jsonify_params(v)
            else:
                json_params[k] = _jsonify(v)
        return json_params

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
            if k.startswith("__") and k.endswith("__"):
                # dunder names are ignored.
                continue
            elif isinstance(v, self.benchmark_type):
                if not tags or set(tags) & set(v.tags):
                    self.benchmarks.append(v)
            elif isinstance(v, list | tuple | set | frozenset):
                for bm in v:
                    if isinstance(bm, self.benchmark_type):
                        if not tags or set(tags) & set(bm.tags):
                            self.benchmarks.append(bm)

    def run(
        self,
        path_or_module: str | os.PathLike[str],
        name: str | None = None,
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
        name: str | None
            A name for the currently started run. If None, a name will be automatically generated.
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
            A JSON output representing the benchmark results. Has three top-level keys,
            "name" giving the benchmark run name, "context" holding the context information,
            and "benchmarks", holding an array with the benchmark results.
        """
        run = name or "nnbench-" + platform.node() + "-" + uuid.uuid1().hex[:8]

        if not self.benchmarks:
            self.collect(path_or_module, tags)

        family_sizes: dict[str, Any] = collections.defaultdict(int)
        family_indices: dict[str, Any] = collections.defaultdict(int)

        ctx: dict[str, Any] = {}
        for provider in context:
            val = provider()
            duplicates = set(ctx.keys()) & set(val.keys())
            if duplicates:
                dupe, *_ = duplicates
                raise ValueError(f"got multiple values for context key {dupe!r}")
            ctx.update(val)

        # if we didn't find any benchmarks, warn and return an empty record.
        if not self.benchmarks:
            warnings.warn(f"No benchmarks found in path/module {str(path_or_module)!r}.")
            return BenchmarkRecord(run=run, context=ctx, benchmarks=[])

        for bm in self.benchmarks:
            family_sizes[bm.interface.funcname] += 1

        if isinstance(params, Parameters):
            dparams = asdict(params)
        else:
            dparams = params or {}

        if dparams and self.typecheck:
            self._check(dparams)

        results: list[dict[str, Any]] = []

        def _maybe_dememo(v, expected_type):
            """Compute and memoize a value if a memo is given for a variable."""
            if is_memo(v) and not is_memo_type(expected_type):
                return v()
            return v

        for benchmark in self.benchmarks:
            bm_family = benchmark.interface.funcname
            state = State(
                name=benchmark.name,
                family=bm_family,
                family_size=family_sizes[bm_family],
                family_index=family_indices[bm_family],
            )
            family_indices[bm_family] += 1

            # Assembling benchmark parameters: First grab all defaults from the interface...
            bmparams = {
                name: _maybe_dememo(val, typ)
                for name, typ, val in benchmark.interface.variables
                if val != inspect.Parameter.empty
            }
            # ... then hydrate with the input parameters.
            bmparams |= dparams
            # If any arguments are still unresolved, go look them up as fixtures.
            if set(bmparams) < set(benchmark.interface.names):
                # TODO: This breaks for a module name (like __main__).
                # Since that only means that we cannot resolve fixtures when benchmarking
                # a module name (which makes sense), and we can always pass extra
                # parameters in the module case, fixing this is not as urgent.
                p = Path(path_or_module)
                if p.is_file():
                    root = p.parent
                elif p.is_dir():
                    root = p
                else:
                    raise ValueError(f"expected a file or directory, got {path_or_module!r}")
                fm = FixtureManager(root)
                bmparams |= fm.resolve(benchmark)

            # TODO: Wrap this into an execution context
            res: dict[str, Any] = {
                "name": benchmark.name,
                "function": qualname(benchmark.fn),
                "description": benchmark.fn.__doc__ or "",
                "date": datetime.now().isoformat(timespec="seconds"),
                "error_occurred": False,
                "error_message": "",
                "parameters": self.jsonify_params(bmparams),
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
            run=run,
            context=ctx,
            benchmarks=results,
        )
