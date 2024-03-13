"""Useful type interfaces to override/subclass in benchmarking workflows."""

from __future__ import annotations

import copy
import inspect
import os
import shutil
import weakref
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import mkdtemp
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    TypeVar,
)

from nnbench.context import Context

try:
    import fsspec

    HAS_FSSPEC = True
except ImportError:
    HAS_FSSPEC = False

T = TypeVar("T")
Variable = tuple[str, type, Any]


def NoOp(**kwargs: Any) -> None:
    pass


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


class ArtifactLoader:
    @abstractmethod
    def load(self) -> os.PathLike[str]:
        """Load the artifact"""


class LocalArtifactLoader(ArtifactLoader):
    """
    ArtifactLoader for loading artifacts from a local file system.

    Parameters
    ----------
    path : str | os.PathLike[str]
        The file system path to the artifact.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = path

    def load(self) -> Path:
        """
        Returns the path to the artifact on the local file system.
        """
        return Path(self._path).resolve()


class FilePathArtifactLoader(ArtifactLoader):
    """
    ArtifactLoader for loading artifacts using fsspec-supported file systems.

    This allows for loading from various file systems like local, S3, GCS, etc.,
    by using a unified API provided by fsspec.

    Parameters
    ----------
    path : str | os.PathLike[str]
        The path to the artifact, which can include a protocol specifier (like 's3://') for remote access.
    destination : str | os.PathLike[str] | None
        The local directory to which remote artifacts will be downloaded. If provided, the model data will be persisted. Otherwise, local artifacts are cleaned.
    storage_options : dict[str, Any] | None
        Storage options for remote storage.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        destination: str | os.PathLike[str] | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        self.source_path = str(path)
        if destination:
            target_path = str(Path(destination).resolve())
            delete = False
        else:
            target_path = str(Path(mkdtemp()).resolve())
            delete = True
        self._finalizer = weakref.finalize(
            self, lambda d, t: shutil.rmtree(t) if d else None, d=delete, t=target_path
        )
        self.target_path = target_path
        self.storage_options = storage_options or {}

    def load(self) -> Path:
        """
        Loads the artifact and returns the local path.

        Returns
        -------
        Path
            The path to the artifact on the local filesystem.

        Raises
        ------
        ImportError
            When fsspec is not installed.
        """
        if not HAS_FSSPEC:
            raise ImportError(
                "class {self.__class__.__name__} requires `fsspec` to be installed. You can install it by running `python -m pip install --upgrade fsspec`"
            )
        fs = fsspec.filesystem(fsspec.utils.get_protocol(self.source_path))
        fs.get(self.source_path, self.target_path, recursive=True)
        return Path(self.target_path).resolve()


class Artifact(Generic[T], metaclass=ABCMeta):
    """
    A base artifact class for loading (materializing) artifacts from disk or from remote storage.

    This is a helper to convey which kind of type gets loaded for a benchmark in a type-safe way.
    It is most useful when running models on already saved data or models, e.g. when
    comparing a newly trained model against a baseline in storage.

    You need to supply an ArtifactLoader with a load() method to load the Artifact into the
    local system storage.

    Subclasses need to implement the `Artifact.deserialize()` API, telling nnbench to
    load the desired artifact from their path.

    Parameters
    ----------
    loader: ArtifactLoader
        Loader to get the artifact.
    """

    def __init__(self, loader: ArtifactLoader) -> None:
        # Save the path for later just-in-time deserialization.
        self.path = loader.load()  # fetch the artifact from wherever it resides
        self._value: T | None = None

    @abstractmethod
    def deserialize(self) -> None:
        """Deserialize the artifact."""

    def is_deserialized(self) -> bool:
        """Checks if the artifact is already deserialized."""
        return self._value is not None

    def __str__(self) -> str:
        return f"Artifact(path={self.path!r}, is_deserialized={self.is_deserialized()})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path!r}, is_deserialized={self.is_deserialized()})"

    @property
    def value(self) -> T:
        """
        Returns the deserialized artifact value.

        Returns
        -------
        T
            The deserialized value of the artifact.
        """
        if self._value is None:
            self.deserialize()
        return self._value


@dataclass(init=False, frozen=True)
class Parameters:
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
