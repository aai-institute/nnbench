"""Data model, registration, and parametrization facilities for defining benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


def NoOp(**kwargs: Any) -> None:
    pass


# TODO: Should this be frozen (since the setUp and tearDown hooks are empty returns)?
@dataclass(init=False)
class Params:
    """
    A dataclass designed to hold benchmark parameters. This class is not functional
    on its own, and needs to be subclassed according to your benchmarking workloads.

    The main advantage over passing parameters as a dictionary is, of course,
    static analysis and type safety for your benchmarking code.
    """

    pass


@dataclass(frozen=True)
class Benchmark:
    """
    Data model representing a benchmark. Subclass this to define your own custom benchmark.

    Parameters
    ----------
    fn: Callable[..., Any]
        The function defining the benchmark.
    name: str | None
        A name to display for the given benchmark. If not given, will be constructed from the
        function name and given parameters.
    params: dict[str, Any]
        Fixed parameters to pass to the benchmark.
    setUp: Callable[..., None]
        A setup hook run before the benchmark. Must take all members of `params` as inputs.
    tearDown: Callable[..., None]
        A teardown hook run after the benchmark. Must take all members of `params` as inputs.
    tags: tuple[str, ...]
        Additional tags to attach for bookkeeping and selective filtering during runs.
    """

    fn: Callable[..., Any]
    name: str | None = field(default=None)
    params: dict[str, Any] = field(repr=False, default_factory=dict)
    setUp: Callable[..., None] = field(repr=False, default=NoOp)
    tearDown: Callable[..., None] = field(repr=False, default=NoOp)
    tags: tuple[str, ...] = field(repr=False, default=())

    def __post_init__(self):
        if not self.name:
            name = self.fn.__name__
            if self.params:
                name += "_" + "_".join(f"{k}={v}" for k, v in self.params.items())

            super().__setattr__("name", name)
        # TODO: Parse interface using `inspect`, attach to the class


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
    params: dict[str, Any]
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
    setUp: Callable[..., None] = None,
    tearDown: Callable[..., None] = None,
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
    parameters: Iterable[dict]
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
