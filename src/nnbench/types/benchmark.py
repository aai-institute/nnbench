"""Type interfaces for benchmarks and benchmark collections."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping

from nnbench.context import Context
from nnbench.types.interface import Interface


def NoOp(state: State, params: Mapping[str, Any] = MappingProxyType({})) -> None:
    pass


@dataclass(frozen=True)
class State:
    name: str
    family: str
    family_size: int
    family_index: int


@dataclass(frozen=True)
class BenchmarkRecord:
    context: Context
    benchmarks: list[dict[str, Any]]

    def compact(
        self,
        mode: Literal["flatten", "inline", "omit"] = "inline",
        sep: str = ".",
    ) -> list[dict[str, Any]]:
        """
        Prepare the benchmark results, optionally inlining the context either as a
        nested dictionary or in flattened form.

        Parameters
        ----------
        mode: Literal["flatten", "inline", "omit"]
            How to handle the context. ``"omit"`` leaves out the context entirely, ``"inline"``
            inserts it into the benchmark dictionary as a single entry named ``"context"``, and
            ``"flatten"`` inserts the flattened context values into the dictionary.
        sep: str
            The separator to use when flattening the context, i.e. when ``mode = "flatten"``.

        Returns
        -------
        list[dict[str, Any]]
            The updated list of benchmark records.
        """
        if mode == "omit":
            return self.benchmarks

        result = []

        for b in self.benchmarks:
            bc = copy.deepcopy(b)
            if mode == "inline":
                bc["context"] = self.context.data
            elif mode == "flatten":
                flat = self.context.flatten(sep=sep)
                bc.update(flat)
                bc["_contextkeys"] = list(self.context.keys())
            result.append(bc)
        return result

    @classmethod
    def expand(cls, bms: list[dict[str, Any]]) -> BenchmarkRecord:
        """
        Expand a list of deserialized JSON-like objects into a benchmark record.
        This is equivalent to extracting the context given by the method it was
        serialized with, and then returning the rest of the data as is.

        Parameters
        ----------
        bms: list[dict[str, Any]]
            The list of benchmark dicts to expand into a record.

        Returns
        -------
        BenchmarkRecord
            The resulting record with the context extracted.

        """
        dctx: dict[str, Any] = {}
        for b in bms:
            if "context" in b:
                dctx = b.pop("context")
            elif "_contextkeys" in b:
                ctxkeys = b.pop("_contextkeys")
                for k in ctxkeys:
                    # This should never throw, save for data corruption.
                    dctx[k] = b.pop(k)
        return cls(context=Context.make(dctx), benchmarks=bms)

    # TODO: Add an expandmany() API for returning a sequence of records for heterogeneous
    #  context data.


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

    pass
