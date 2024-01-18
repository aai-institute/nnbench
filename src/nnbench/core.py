"""Data model, registration, and parametrization facilities for defining benchmarks."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from nnbench.types import Benchmark


def NoOp(**kwargs: Any) -> None:
    pass


def benchmark(
    func: Callable[..., Any] | None = None,
    params: dict[str, Any] | None = None,
    setUp: Callable[..., None] = NoOp,
    tearDown: Callable[..., None] = NoOp,
    tags: tuple[str, ...] = (),
) -> Callable:
    """
    Define a benchmark from a function.

    The resulting benchmark can either be completely (i.e., the resulting function takes no
    more arguments) or incompletely parametrized. In the latter case, the remaining free
    parameters need to be passed in the calls to `AbstractBenchmarkRunner.run()`.

    Parameters
    ----------
    func: Callable[..., Any] | None
        The function to benchmark.
    params: dict[str, Any] | None
        The parameters (or a subset thereof) defining the benchmark.
    setUp: Callable[..., None]
        A setup hook to run before each of the benchmarks.
    tearDown: Callable[..., None]
        A teardown hook to run after each of the benchmarks.
    tags: tuple[str, ...]
        Additional tags to attach for bookkeeping and selective filtering during runs.

    Returns
    -------
    Callable
        A decorated callable yielding the benchmark.
    """

    # TODO: The above return typing is incorrect
    #  (needs a func is None vs. func is not None overload)
    def inner(fn: Callable) -> Benchmark:
        name = fn.__name__
        if params:
            name += "_" + "_".join(f"{k}={v}" for k, v in params.items())
        return Benchmark(fn, name=name, params=params, setUp=setUp, tearDown=tearDown, tags=tags)

    if func:
        return inner(func)  # type: ignore
    else:
        return inner


def parametrize(
    func: Callable[..., Any] | None = None,
    parameters: Iterable[dict] | None = None,
    setUp: Callable[..., None] = NoOp,
    tearDown: Callable[..., None] = NoOp,
    tags: tuple[str, ...] = (),
) -> Callable:
    """
    Define a family of benchmarks over a function with varying parameters.

    The resulting benchmark can either be completely (i.e., the resulting function takes no
    more arguments) or incompletely parametrized. In the latter case, the remaining free
    parameters need to be passed in the calls to `AbstractBenchmarkRunner.run()`.

    Parameters
    ----------
    func: Callable[..., Any] | None
        The function to benchmark.
    parameters: Iterable[dict] | None
        The different sets of parameters defining the benchmark family.
    setUp: Callable[..., None]
        A setup hook to run before each of the benchmarks.
    tearDown: Callable[..., None]
        A teardown hook to run after each of the benchmarks.
    tags: tuple[str, ...]
        Additional tags to attach for bookkeeping and selective filtering during runs.

    Returns
    -------
    Callable
        A decorated callable yielding the benchmark family.
    """

    # TODO: The above return typing is incorrect
    #  (needs a func is None vs. func is not None overload)
    def inner(fn: Callable) -> list[Benchmark]:
        benchmarks = []
        for params in parameters:
            name = fn.__name__
            if params:
                name += "_" + "_".join(f"{k}={v}" for k, v in params.items())
            bm = Benchmark(fn, name=name, params=params, setUp=setUp, tearDown=tearDown, tags=tags)
            benchmarks.append(bm)
        return benchmarks

    if func:
        return inner(func)  # type: ignore
    else:
        return inner
