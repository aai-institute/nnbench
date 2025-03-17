"""The abstract benchmark runner interface, which can be overridden for custom benchmark workloads."""

import collections
import inspect
import logging
import os
import platform
import sys
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from nnbench.context import Context, ContextProvider
from nnbench.fixtures import FixtureManager
from nnbench.types import Benchmark, BenchmarkFamily, BenchmarkResult, Parameters, State
from nnbench.util import (
    all_python_files,
    collapse,
    exists_module,
    import_file_as_module,
    qualname,
    timer,
)

Benchmarkable = Benchmark | BenchmarkFamily

logger = logging.getLogger("nnbench.runner")


def jsonify(
    params: dict[str, Any], repr_hooks: dict[type, Callable] | None = None
) -> dict[str, Any]:
    """
    Construct a JSON representation of benchmark parameters.

    This is necessary to break reference cycles from the parameters to the results,
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
            try:
                return val.to_json()
            except TypeError:
                # if to_json() needs arguments, we're SOL.
                pass

        return repr(val)

    for k, v in params.items():
        if isinstance(v, tuple | list | set | frozenset):
            container_type = type(v)
            json_params[k] = container_type(map(_jsonify, v))
        elif isinstance(v, dict):
            json_params[k] = jsonify(v)
        else:
            json_params[k] = _jsonify(v)
    return json_params


def collect(
    path_or_module: str | os.PathLike[str], tags: tuple[str, ...] = ()
) -> list[Benchmark | BenchmarkFamily]:
    # TODO: functools.cache this guy
    """
    Discover benchmarks in a module or source file.

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
    benchmarks: list[Benchmark] = []
    ppath = Path(path_or_module)
    if ppath.is_dir():
        pythonpaths = all_python_files(ppath)
        for py in pythonpaths:
            benchmarks.extend(collect(py, tags))
        return benchmarks
    elif ppath.is_file():
        logger.debug(f"Collecting benchmarks from file {ppath}.")
        module = import_file_as_module(path_or_module)
    elif exists_module(path_or_module):
        module = sys.modules[str(path_or_module)]
    else:
        raise ValueError(
            f"expected a module name, Python file, or directory, got {str(path_or_module)!r}"
        )

    # iterate through the module dict members to register
    for k, v in module.__dict__.items():
        if k.startswith("__") and k.endswith("__"):
            # dunder names are ignored.
            continue
        elif isinstance(v, Benchmarkable):
            if not tags or set(tags) & set(v.tags):
                benchmarks.append(v)
        elif isinstance(v, list | tuple | set | frozenset):
            for bm in v:
                if isinstance(bm, Benchmarkable):
                    if not tags or set(tags) & set(bm.tags):
                        benchmarks.append(bm)
    return benchmarks


def run(
    benchmarks: Benchmark | BenchmarkFamily | Iterable[Benchmark | BenchmarkFamily],
    name: str | None = None,
    params: dict[str, Any] | Parameters | None = None,
    context: Context | Iterable[ContextProvider] = (),
    jsonifier: Callable[[dict[str, Any]], dict[str, Any]] = jsonify,
) -> BenchmarkResult:
    """
    Run a previously collected benchmark workload.

    Parameters
    ----------
    benchmarks: Benchmark | BenchmarkFamily | Iterable[Benchmark | BenchmarkFamily]
        A benchmark, family of benchmarks, or collection of discovered benchmarks to run.
    name: str | None
        A name for the currently started run. If None, a name will be automatically generated.
    params: dict[str, Any] | Parameters | None
        Parameters to use for the benchmark run. Names have to match positional and keyword
        argument names of the benchmark functions.
    context: Iterable[ContextProvider]
        Additional context to log with the benchmarks in the output JSON result. Useful for
        obtaining environment information and configuration, like CPU/GPU hardware info,
        ML model metadata, and more.
    jsonifier: Callable[[dict[str, Any], dict[str, Any]]]
        A function constructing a string representation from the input parameters.
        Defaults to ``nnbench.runner.jsonify_params()``. Must produce a dictionary containing
        only JSON-serializable values.

    Returns
    -------
    BenchmarkResult
        A JSON output representing the benchmark results. Has three top-level keys,
        "name" giving the benchmark run name, "context" holding the context information,
        and "benchmarks", holding an array with the benchmark results.
    """

    _run = name or "nnbench-" + platform.node() + "-" + uuid.uuid1().hex[:8]

    family_sizes: dict[str, Any] = collections.defaultdict(int)
    family_indices: dict[str, Any] = collections.defaultdict(int)

    if isinstance(context, dict):
        ctx = context
    else:
        ctx = dict()
        for provider in context:
            val = provider()
            duplicates = set(ctx.keys()) & set(val.keys())
            if duplicates:
                dupe, *_ = duplicates
                raise ValueError(f"got multiple values for context key {dupe!r}")
            ctx.update(val)

    if isinstance(benchmarks, Benchmarkable):
        benchmarks = [benchmarks]

    if isinstance(params, Parameters):
        dparams = asdict(params)
    else:
        dparams = params or {}

    results: list[dict[str, Any]] = []
    timestamp = int(time.time())

    for benchmark in collapse(benchmarks):
        bm_family = benchmark.interface.funcname
        state = State(
            name=benchmark.name,
            family=bm_family,
            family_size=family_sizes[bm_family],
            family_index=family_indices[bm_family],
        )
        family_indices[bm_family] += 1

        # Assemble benchmark parameters. First grab all defaults from the interface,
        bmparams = {
            name: val
            for name, _, val in benchmark.interface.variables
            if val is not inspect.Parameter.empty
        }
        # ... then hydrate with the appropriate subset of input parameters.
        bmparams |= {k: v for k, v in dparams.items() if k in benchmark.interface.names}
        # If any arguments are still unresolved, go look them up as fixtures.
        if set(bmparams) < set(benchmark.interface.names):
            # TODO: This breaks for a module name (like __main__).
            # Since that only means that we cannot resolve fixtures when benchmarking
            # a module name (which makes sense), and we can always pass extra
            # parameters in the module case, fixing this is not as urgent.
            mod = benchmark.__module__
            file = sys.modules[mod].__file__
            p = Path(file).parent
            fm = FixtureManager(p)
            bmparams |= fm.resolve(benchmark)

        res: dict[str, Any] = {
            "name": benchmark.name,
            "function": qualname(benchmark.fn),
            "description": benchmark.fn.__doc__ or "",
            "error_occurred": False,
            "error_message": "",
            "parameters": jsonifier(bmparams),
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

    return BenchmarkResult(
        run=_run,
        context=ctx,
        benchmarks=results,
        timestamp=timestamp,
    )
