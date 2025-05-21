"""Types for benchmarks and records holding results of a run."""

import copy
import inspect
import json
import os
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, TypeVar

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


@dataclass(frozen=True)
class State:
    """
    A dataclass holding some basic information about a benchmark and its hierarchy
    inside its *family* (i.e. a series of the same benchmark for different parameters).

    For benchmarks registered with ``@nnbench.benchmark``, meaning no parametrization,
    each benchmark constitutes its own family, and ``family_size == 1`` holds true.
    """

    name: str
    family: str
    family_size: int
    family_index: int


def NoOp(state: State, params: Mapping[str, Any] = MappingProxyType({})) -> None:
    """A no-op setup/teardown callback that does nothing."""
    pass


@dataclass(frozen=True)
class BenchmarkResult:
    """
    A dataclass representing the result of a benchmark run, i.e. the return value
    of a call to ``nnbench.run()``.
    """

    run: str
    """A name describing the run."""
    context: dict[str, Any]
    """A map of key-value pairs describing context information around the benchmark run."""
    benchmarks: list[dict[str, Any]]
    """The list of benchmark results, each given as a Python dictionary."""
    timestamp: int
    """A Unix timestamp indicating when the run was started."""

    def to_json(self) -> dict[str, Any]:
        """
        Export a benchmark record to JSON.

        Returns
        -------
        dict[str, Any]
            A JSON representation of the benchmark record.
        """
        return asdict(self)

    def to_records(self) -> list[dict[str, Any]]:
        """
        Export a benchmark record to a list of individual results,
        each with the benchmark run, context, and timestamp inlined.
        """
        records = []
        data = {"run": self.run, "context": json.dumps(self.context), "timestamp": self.timestamp}
        for bm in self.benchmarks:
            d = copy.deepcopy(data)
            d["benchmark"] = json.dumps(bm)
            records.append(d)
        return records

    @classmethod
    def from_json(cls, struct: dict[str, Any]) -> Self:
        """
        Load a benchmark result from its JSON representation.

        Parameters
        ----------
        struct: dict[str, Any]
            The JSON object containing the benchmark data.

        Returns
        -------
        Self
            A BenchmarkResult instance containing the run information.
        """
        benchmarks = struct.get("benchmarks", [])
        context = struct.get("context", {})
        run = struct.get("run", "")
        timestamp = struct.get("timestamp", 0)
        return cls(run=run, benchmarks=benchmarks, context=context, timestamp=timestamp)

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> list[Self]:
        """
        Expand a list of deserialized JSON-like objects into a benchmark record.
        This is equivalent to extracting the context given by the method it was
        serialized with, and then returning the rest of the data as is.

        Parameters
        ----------
        records: list[dict[str, Any]]
            The deserialized benchmark records (i.e. database rows) to form into a
            benchmark result.

        Returns
        -------
        BenchmarkResult
            The result representation, with the context and run name extracted.
        """
        ret: list[Self] = []
        benchmark_map = defaultdict(list)
        run_map: dict[str, Any] = {}
        for r in records:
            res = {}
            run = r.pop("run", "")
            # The following 3 values don't change across a run
            res["run"] = run
            res["context"] = json.loads(r.pop("context", "{}"))
            res["timestamp"] = r.pop("timestamp", 0)
            run_map[run] = res
            benchmark_map[run].append(json.loads(r.pop("benchmark", "{}")))

        for run, res in run_map.items():
            res["benchmarks"] = benchmark_map[run]
            ret.append(cls(**res))
        return ret


class BenchmarkReporter(Protocol):
    def read(self, fp: str | os.PathLike[str], **kwargs: Any) -> list[BenchmarkResult]: ...

    def write(
        self, results: Iterable[BenchmarkResult], fp: str | os.PathLike[str], **kwargs: Any
    ) -> None: ...


@dataclass(init=False, frozen=True)
class Parameters:
    """
    A dataclass designed to hold benchmark parameters.

    This class is not functional on its own, and needs to be subclassed
    according to your benchmarking workloads.

    The main advantage over passing parameters as a dictionary are static analysis
    and type safety for your benchmarking code.
    """


T = TypeVar("T")
Variable = tuple[str, type, Any]


@dataclass(frozen=True)
class Interface:
    """
    Data model representing a function's interface.

    An instance of this class is created using the ``Interface.from_callable()``
    class method.
    """

    funcname: str
    """Name of the function."""
    names: tuple[str, ...]
    """Names of the function parameters."""
    types: tuple[type, ...]
    """Type hints of the function parameters."""
    defaults: tuple
    """The function parameters' default values, or inspect.Parameter.empty if a parameter has no default."""
    variables: tuple[Variable, ...]
    """A tuple of tuples, where each inner tuple contains the parameter name, type, and default value."""
    returntype: type
    """The function's return type annotation, or NoneType if left untyped."""

    @classmethod
    def from_callable(cls, fn: Callable, defaults: dict[str, Any]) -> Self:
        """
        Creates an interface instance from the given callable.

        Wraps the information given by ``inspect.signature()``, with the option to
        supply a ``defaults`` map and overwrite any default set in the function's
        signature.
        """
        # Set `follow_wrapped=False` to get the partially filled interfaces.
        # Otherwise we get missing value errors for parameters supplied in benchmark decorators.
        sig = inspect.signature(fn, follow_wrapped=False)
        ret = sig.return_annotation
        _defaults = {k: defaults.get(k, v.default) for k, v in sig.parameters.items()}
        # defaults are the signature parameters, then the partial parametrization.
        return cls(
            fn.__name__,
            tuple(sig.parameters.keys()),
            tuple(p.annotation for p in sig.parameters.values()),
            tuple(_defaults.values()),
            tuple((k, v.annotation, _defaults[k]) for k, v in sig.parameters.items()),
            type(ret) if ret is None else ret,
        )


@dataclass(frozen=True)
class Benchmark:
    """
    Data model representing a benchmark. Subclass this to define your own custom benchmark.
    """

    fn: Callable[..., Any]
    """The function defining the benchmark."""
    name: str = ""
    """A name to display for the given benchmark. If not given, a name will be constructed from the function name and given parameters."""
    params: dict[str, Any] = field(default_factory=dict)
    """A partial parametrization to apply to the benchmark function. Internal only, you should not need to set this yourself."""
    setUp: Callable[[State, Mapping[str, Any]], None] = field(repr=False, default=NoOp)
    """A setup hook run before the benchmark. Must take all members of ``params`` as inputs."""
    tearDown: Callable[[State, Mapping[str, Any]], None] = field(repr=False, default=NoOp)
    """A teardown hook run after the benchmark. Must take all members of ``params`` as inputs."""
    tags: tuple[str, ...] = field(repr=False, default=())
    """Additional tags to attach for bookkeeping and selective filtering during runs."""
    interface: Interface = field(init=False, repr=False)
    """Benchmark interface, constructed from the given function. Implementation detail."""

    def __post_init__(self):
        if not self.name:
            super().__setattr__("name", self.fn.__name__)
        super().__setattr__("interface", Interface.from_callable(self.fn, self.params))


class BenchmarkFamily(Iterable[Benchmark]):
    def __init__(
        self,
        fn: Callable[..., Any],
        params: Iterable[dict[str, Any]],
        name: str | Callable[..., str],
        setUp: Callable[..., None] = NoOp,
        tearDown: Callable[..., None] = NoOp,
        tags: tuple[str, ...] = (),
    ):
        self.fn = fn
        self.params = params
        if isinstance(name, str):
            # if name is a str, we assume it's an f-string that should be
            # interpolated with the necessary parameters.
            self.name = lambda _fn, **kwargs: name.format(**kwargs)
        else:
            self.name = name
        self.setUp = setUp
        self.tearDown = tearDown
        self.tags = tags

    def __iter__(self):
        """
        Dispatch benchmarks lazily, creating a name from the arguments as dictated
        by ``self.name``.
        """
        for p in self.params:
            yield Benchmark(
                fn=self.fn,
                params=p,
                name=self.name(self.fn, **p),
                setUp=self.setUp,
                tearDown=self.tearDown,
                tags=self.tags,
            )
