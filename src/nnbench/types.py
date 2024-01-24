"""Useful type interfaces to override/subclass in benchmarking workflows."""
from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypedDict, TypeVar

T = TypeVar("T")


class BenchmarkResult(TypedDict):
    context: dict[str, Any]
    benchmarks: list[dict[str, Any]]


def NoOp(**kwargs: Any) -> None:
    pass


class Artifact(Generic[T]):
    """
    A base artifact class for loading (materializing) artifacts from disk or from remote storage.

    This is a helper to convey which kind of type gets loaded for a benchmark in a type-safe way.
    It is most useful when running models on already saved data or models, e.g. when
    comparing a newly trained model against a baseline in storage.

    Subclasses need to implement the `Artifact.materialize()` API, telling nnbench how to
    load the desired artifact from a path.

    Parameters
    ----------
    path: str | os.PathLike[str]
        Path to the artifact files.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        # Save the path for later just-in-time materialization.
        self.path = path
        self._value: T | None = None

    @classmethod
    def materialize(cls) -> "Artifact":
        """Load the artifact from storage."""
        raise NotImplementedError

    def value(self) -> T:
        if self._value is None:
            raise ValueError(
                f"artifact has not been instantiated yet, "
                f"perhaps you forgot to call {self.__class__.__name__}.materialize()?"
            )
        return self._value


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
    setUp: Callable[..., None]
        A setup hook run before the benchmark. Must take all members of `params` as inputs.
    tearDown: Callable[..., None]
        A teardown hook run after the benchmark. Must take all members of `params` as inputs.
    tags: tuple[str, ...]
        Additional tags to attach for bookkeeping and selective filtering during runs.
    """

    fn: Callable[..., Any]
    name: str | None = field(default=None)
    setUp: Callable[..., None] = field(repr=False, default=NoOp)
    tearDown: Callable[..., None] = field(repr=False, default=NoOp)
    tags: tuple[str, ...] = field(repr=False, default=())

    @dataclass(frozen=True)
    class Interface:
        fn: Callable[..., Any]

        def __post_init__(self) -> None:
            sig = inspect.signature(self.fn)
            super().__setattr__("varnames", tuple(sig.parameters.keys()))
            super().__setattr__("vartypes", tuple(sig.parameters.values()))
            super().__setattr__("varitems", tuple(sig.parameters.values()))
            super().__setattr__("defaults", {n: p.default for n, p in sig.parameters.items()})

    def __post_init__(self):
        if not self.name:
            name = self.fn.__name__

            super().__setattr__("name", name)
            super().__setattr__("interface", self.Interface(self.fn))
