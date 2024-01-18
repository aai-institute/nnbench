from dataclasses import dataclass, field
from typing import Any, Callable, TypedDict


class BenchmarkResult(TypedDict):
    context: dict[str, Any]
    benchmarks: list[dict[str, Any]]


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
