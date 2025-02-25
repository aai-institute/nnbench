"""Data model, registration, and parametrization facilities for defining benchmarks."""

import itertools
from collections.abc import Callable, Iterable
from typing import Any, overload

from nnbench.types import Benchmark, BenchmarkFamily, NoOp


def _default_namegen(fn: Callable, **kwargs: Any) -> str:
    return fn.__name__ + "_" + "_".join(f"{k}={v}" for k, v in kwargs.items())


# Overloads for the ``benchmark`` decorator.
# Case #1: Bare application without parentheses
# @nnbench.benchmark
# def foo() -> int:
#     return 0
@overload
def benchmark(
    func: None = None,
    name: str = "",
    setUp: Callable[..., None] = NoOp,
    tearDown: Callable[..., None] = NoOp,
    tags: tuple[str, ...] = (),
) -> Callable[[Callable], Benchmark]: ...


# Case #2: Application with arguments
# @nnbench.benchmark(name="My foo experiment", tags=("hello", "world"))
# def foo() -> int:
#     return 0
@overload
def benchmark(
    func: Callable[..., Any],
    name: str = "",
    setUp: Callable[..., None] = NoOp,
    tearDown: Callable[..., None] = NoOp,
    tags: tuple[str, ...] = (),
) -> Benchmark: ...


def benchmark(
    func: Callable[..., Any] | None = None,
    name: str = "",
    setUp: Callable[..., None] = NoOp,
    tearDown: Callable[..., None] = NoOp,
    tags: tuple[str, ...] = (),
) -> Benchmark | Callable[[Callable], Benchmark]:
    """
    Define a benchmark from a function.

    The resulting benchmark can either be completely (i.e., the resulting function takes no
    more arguments) or incompletely parametrized. In the latter case, the remaining free
    parameters need to be passed in the calls to `nnbench.run()`.

    Parameters
    ----------
    func: Callable[..., Any] | None
        The function to benchmark. This slot only exists to allow application of the decorator
        without parentheses, you should never fill it explicitly.
    name: str
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
    namegen: Callable[..., str] = _default_namegen,
    tags: tuple[str, ...] = (),
) -> Callable[[Callable], BenchmarkFamily]:
    """
    Define a family of benchmarks over a function with varying parameters.

    The resulting benchmarks can either be completely (i.e., the resulting function takes no
    more arguments) or incompletely parametrized. In the latter case, the remaining free
    parameters need to be passed in the call to `nnbench.run()`.

    Parameters
    ----------
    parameters: Iterable[dict[str, Any]]
        The different sets of parameters defining the benchmark family.
    setUp: Callable[..., None]
        A setup hook to run before each of the benchmarks.
    tearDown: Callable[..., None]
        A teardown hook to run after each of the benchmarks.
    namegen: Callable[..., str]
        A function taking the benchmark function and given parameters that generates a unique
        custom name for the benchmark. The default name generated is the benchmark function's name
        followed by the keyword arguments in ``key=value`` format separated by underscores.
    tags: tuple[str, ...]
        Additional tags to attach for bookkeeping and selective filtering during runs.

    Returns
    -------
    Callable[[Callable], BenchmarkFamily]
        A parametrized decorator returning the benchmark family.
    """

    def decorator(fn: Callable) -> BenchmarkFamily:
        return BenchmarkFamily(
            fn,
            parameters,
            name=namegen,
            setUp=setUp,
            tearDown=tearDown,
            tags=tags,
        )

    return decorator


def product(
    setUp: Callable[..., None] = NoOp,
    tearDown: Callable[..., None] = NoOp,
    namegen: Callable[..., str] = _default_namegen,
    tags: tuple[str, ...] = (),
    **iterables: Iterable,
) -> Callable[[Callable], BenchmarkFamily]:
    """
    Define a family of benchmarks over a cartesian product of one or more iterables.

    The resulting benchmarks can either be completely (i.e., the resulting function takes no
    more arguments) or incompletely parametrized. In the latter case, the remaining free
    parameters need to be passed in the call to `nnbench.run()`.

    Parameters
    ----------
    setUp: Callable[..., None]
        A setup hook to run before each of the benchmarks.
    tearDown: Callable[..., None]
        A teardown hook to run after each of the benchmarks.
    namegen: Callable[..., str]
        A function taking the benchmark function and given parameters that generates a unique
        custom name for the benchmark. The default name generated is the benchmark function's name
        followed by the keyword arguments in ``key=value`` format separated by underscores.
    tags: tuple[str, ...]
        Additional tags to attach for bookkeeping and selective filtering during runs.
    **iterables: Iterable
        The iterables parametrizing the benchmarks.

    Returns
    -------
    Callable[[Callable], BenchmarkFamily]
        A parametrized decorator returning the benchmark family.
    """

    def decorator(fn: Callable) -> BenchmarkFamily:
        names, values = iterables.keys(), iterables.values()

        # NB: This line forces the exhaustion of all input iterables by nature of the
        # cartesian product (values need to be persisted to be accessed multiple times).
        parameters = (dict(zip(names, vals)) for vals in itertools.product(*values))

        return BenchmarkFamily(
            fn,
            parameters,
            name=namegen,
            setUp=setUp,
            tearDown=tearDown,
            tags=tags,
        )

    return decorator
