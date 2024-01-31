"""Useful type interfaces to override/subclass in benchmarking workflows."""
from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypedDict, TypeVar

T = TypeVar("T")
Variable = tuple[str, type, Any]


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


@dataclass(init=False, frozen=True)
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
    interface: Interface
        Interface of the benchmark function
    """

    fn: Callable[..., Any]
    name: str | None = field(default=None)
    setUp: Callable[..., None] = field(repr=False, default=NoOp)
    tearDown: Callable[..., None] = field(repr=False, default=NoOp)
    tags: tuple[str, ...] = field(repr=False, default=())
    interface: Interface = field(init=False, repr=False)

    def __post_init__(self):
        if not self.name:
            super().__setattr__("name", self.fn.__name__)
        super().__setattr__("interface", Interface.from_callable(self.fn))


@dataclass(frozen=True)
class Interface:
    """
    Data model representing a function's interface. An instance of this class
    is created using the `from_callable` class method.

    Parameters:
    ----------
    names : tuple[str, ...]
        Names of the function parameters.
    types : tuple[type, ...]
        Types of the function parameters.
    defaults : tuple
        A tuple of the function parameters' default values.
    variables : tuple[Variable, ...]
        A tuple of tuples, where each inner tuple contains the parameter name and type.
    returntype: type
        The function's return type annotation, or NoneType if left untyped.
    """

    names: tuple[str, ...]
    types: tuple[type, ...]
    defaults: tuple
    variables: tuple[Variable, ...]
    returntype: type

    @classmethod
    def from_callable(cls, fn: Callable) -> Interface:
        """
        Creates an interface instance from the given callable.
        """
        # Set follow_wrapped=False to get the partially filled interfaces.
        # Otherwise we get missing value errors for parameters supplied in benchmark decorators.
        sig = inspect.signature(fn, follow_wrapped=False)
        ret = sig.return_annotation
        return cls(
            tuple(sig.parameters.keys()),
            tuple(p.annotation for p in sig.parameters.values()),
            tuple(p.default for p in sig.parameters.values()),
            tuple((k, v.annotation, v.default) for k, v in sig.parameters.items()),
            type(ret) if ret is None else ret,
        )
