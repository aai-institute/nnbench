"""Data model, registration, and parametrization facilities for defining benchmarks."""

from __future__ import annotations

import inspect
import itertools
from functools import partial, update_wrapper
from typing import Any, Callable, Iterable, overload

from nnbench.types import Benchmark


def _check_against_interface(params: dict[str, Any], fun: Callable) -> None:
    sig = inspect.signature(fun)
    fvarnames = set(sig.parameters.keys())
    varnames = set(params.keys())
    if not varnames <= fvarnames:
        # never empty due to the if condition.
        diffvar, *_ = varnames - fvarnames
        raise TypeError(
            f"benchmark {fun.__name__}() got an unexpected keyword argument {diffvar!r}"
        )


def NoOp(**kwargs: Any) -> None:
    pass


# Overloads for the ``benchmark`` decorator.
# Case #1: Bare application without parentheses
# @nnbench.benchmark
# def foo() -> int:
#     return 0
@overload
def benchmark(
    func: None = None,
    name: str | None = None,
    setUp: Callable[..., None] = NoOp,
    tearDown: Callable[..., None] = NoOp,
    tags: tuple[str, ...] = (),
) -> Callable[[Callable], Benchmark]:
    ...


# Case #2: Application with arguments
# @nnbench.benchmark(name="My foo experiment", tags=("hello", "world"))
# def foo() -> int:
#     return 0
@overload
def benchmark(
    func: Callable[..., Any],
    name: str | None = None,
    setUp: Callable[..., None] = NoOp,
    tearDown: Callable[..., None] = NoOp,
    tags: tuple[str, ...] = (),
) -> Benchmark:
    ...


def benchmark(
    func: Callable[..., Any] | None = None,
    name: str | None = None,
    setUp: Callable[..., None] = NoOp,
    tearDown: Callable[..., None] = NoOp,
    tags: tuple[str, ...] = (),
) -> Benchmark | Callable[[Callable], Benchmark]:
    """
    Define a benchmark from a function.

    The resulting benchmark can either be completely (i.e., the resulting function takes no
    more arguments) or incompletely parametrized. In the latter case, the remaining free
    parameters need to be passed in the calls to `BenchmarkRunner.run()`.

    Parameters
    ----------
    func: Callable[..., Any] | None
        The function to benchmark. This slot only exists to allow application of the decorator
        without parentheses, you should never fill it explicitly.
    name: str | None
        A display name to give to the benchmark. Useful in summaries and reports.
    setUp: Callable[..., None]
        A setup hook to run before the benchmark.
    tearDown: Callable[..., None]
        A teardown hook to run after the benchmark.
    tags: tuple[str, ...]
        Additional tags to attach for bookkeeping and selective filtering during runs.

    Returns
    -------
    Benchmark | Callable[[Callable], Benchmark]
        The resulting benchmark (if no arguments were given), or a parametrized decorator
        returning the benchmark.
    """

    def decorator(fun: Callable) -> Benchmark:
        return Benchmark(fun, name=name, setUp=setUp, tearDown=tearDown, tags=tags)

    if func is not None:
        return decorator(func)
    else:
        return decorator


def parametrize(
    parameters: Iterable[dict[str, Any]],
    setUp: Callable[..., None] = NoOp,
    tearDown: Callable[..., None] = NoOp,
    tags: tuple[str, ...] = (),
) -> Callable[[Callable], list[Benchmark]]:
    """
    Define a family of benchmarks over a function with varying parameters.

    The resulting benchmarks can either be completely (i.e., the resulting function takes no
    more arguments) or incompletely parametrized. In the latter case, the remaining free
    parameters need to be passed in the call to `BenchmarkRunner.run()`.

    Parameters
    ----------
    parameters: Iterable[dict[str, Any]]
        The different sets of parameters defining the benchmark family.
    setUp: Callable[..., None]
        A setup hook to run before each of the benchmarks.
    tearDown: Callable[..., None]
        A teardown hook to run after each of the benchmarks.
    tags: tuple[str, ...]
        Additional tags to attach for bookkeeping and selective filtering during runs.

    Returns
    -------
    Callable[[Callable], list[Benchmark]]
        A parametrized decorator returning the benchmark family.
    """

    def decorator(fn: Callable) -> list[Benchmark]:
        benchmarks = []
        for params in parameters:
            _check_against_interface(params, fn)
            name = fn.__name__ + "_" + "_".join(f"{k}={v}" for k, v in params.items())
            wrapper = update_wrapper(partial(fn, **params), fn)
            bm = Benchmark(wrapper, name=name, setUp=setUp, tearDown=tearDown, tags=tags)
            benchmarks.append(bm)
        return benchmarks

    return decorator


def product(
    setUp: Callable[..., None] = NoOp,
    tearDown: Callable[..., None] = NoOp,
    tags: tuple[str, ...] = (),
    **iterables: Iterable,
) -> Callable[[Callable], list[Benchmark]]:
    """
    Define a family of benchmarks over a cartesian product of one or more iterables.

    The resulting benchmarks can either be completely (i.e., the resulting function takes no
    more arguments) or incompletely parametrized. In the latter case, the remaining free
    parameters need to be passed in the call to `BenchmarkRunner.run()`.

    Parameters
    ----------
    setUp: Callable[..., None]
        A setup hook to run before each of the benchmarks.
    tearDown: Callable[..., None]
        A teardown hook to run after each of the benchmarks.
    tags: tuple[str, ...]
        Additional tags to attach for bookkeeping and selective filtering during runs.
    **iterables: Iterable
        The iterables parametrizing the benchmarks.

    Returns
    -------
    Callable[[Callable], list[Benchmark]]
        A parametrized decorator returning the benchmark family.
    """

    def decorator(fn: Callable) -> list[Benchmark]:
        benchmarks = []
        varnames = iterables.keys()
        for values in itertools.product(*iterables.values()):
            d = dict(zip(varnames, values))
            _check_against_interface(d, fn)
            name = fn.__name__ + "_" + "_".join(f"{k}={v}" for k, v in d.items())
            wrapper = update_wrapper(partial(fn, **d), fn)
            bm = Benchmark(wrapper, name=name, setUp=setUp, tearDown=tearDown, tags=tags)
            benchmarks.append(bm)
        return benchmarks

    return decorator
