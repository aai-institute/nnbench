"""Type interfaces for benchmarks and benchmark collections."""

import copy
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from nnbench.types.interface import Interface


@dataclass(frozen=True)
class State:
    name: str
    family: str
    family_size: int
    family_index: int


def NoOp(state: State, params: Mapping[str, Any] = MappingProxyType({})) -> None:
    pass


@dataclass(frozen=True)
class BenchmarkRecord:
    context: dict[str, Any]
    benchmarks: list[dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        """
        Export a benchmark record to JSON.

        Returns
        -------
        dict[str, Any]
            A JSON representation of the benchmark record.
        """
        return asdict(self)

    def to_list(self) -> list[dict[str, Any]]:
        """
        Export a benchmark record to a list of individual results,
        each with the benchmark context inlined.
        """
        results = []
        for b in self.benchmarks:
            bc = copy.deepcopy(b)
            # TODO: Give an option to elide (or process) the context?
            bc["context"] = self.context
            results.append(bc)
        return results

    @classmethod
    def expand(cls, bms: dict[str, Any] | list[dict[str, Any]]) -> Self:
        """
        Expand a list of deserialized JSON-like objects into a benchmark record.
        This is equivalent to extracting the context given by the method it was
        serialized with, and then returning the rest of the data as is.

        Parameters
        ----------
        bms: dict[str, Any] | list[dict[str, Any]]
            The deserialized benchmark record or list of records to expand into a record.

        Returns
        -------
        BenchmarkRecord
            The resulting record with the context extracted.

        """
        context: dict[str, Any]
        if isinstance(bms, dict):
            if "benchmarks" not in bms.keys():
                raise ValueError(f"no benchmark data found in struct {bms}")

            benchmarks = bms["benchmarks"]
            context = bms.get("context", {})
        else:
            context = {}
            benchmarks = bms
            for b in benchmarks:
                # Safeguard if the context is not in the record,
                # for example if it came from a DB query.
                if "context" in b:
                    # TODO: Log context key/value disagreements
                    context |= b.pop("context", {})
        return cls(benchmarks=benchmarks, context=context)


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
        A partial parametrization to apply to the benchmark function. Internal only,
        you should not need to set this yourself.
    setUp: Callable[..., None]
        A setup hook run before the benchmark. Must take all members of `params` as inputs.
    tearDown: Callable[..., None]
        A teardown hook run after the benchmark. Must take all members of `params` as inputs.
    tags: tuple[str, ...]
        Additional tags to attach for bookkeeping and selective filtering during runs.
    """

    fn: Callable[..., Any]
    name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    setUp: Callable[[State, Mapping[str, Any]], None] = field(repr=False, default=NoOp)
    tearDown: Callable[[State, Mapping[str, Any]], None] = field(repr=False, default=NoOp)
    tags: tuple[str, ...] = field(repr=False, default=())
    interface: Interface = field(init=False, repr=False)

    def __post_init__(self):
        if not self.name:
            super().__setattr__("name", self.fn.__name__)
        super().__setattr__("interface", Interface.from_callable(self.fn, self.params))


@dataclass(init=False, frozen=True)
class Parameters:
    """
    A dataclass designed to hold benchmark parameters. This class is not functional
    on its own, and needs to be subclassed according to your benchmarking workloads.

    The main advantage over passing parameters as a dictionary is, of course,
    static analysis and type safety for your benchmarking code.
    """
