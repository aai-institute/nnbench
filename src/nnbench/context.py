"""Utilities for collecting context key-value pairs as metadata in benchmark runs."""

from __future__ import annotations

import itertools
import platform
import sys
from typing import Any, Callable, Iterator, Literal

ContextProvider = Callable[[], dict[str, Any]]
"""A function providing a dictionary of context values."""


def system() -> dict[str, str]:
    return {"system": platform.system()}


def cpuarch() -> dict[str, str]:
    return {"cpuarch": platform.machine()}


def python_version() -> dict[str, str]:
    return {"python_version": platform.python_version()}


class PythonInfo:
    """
    A context helper returning version info for requested installed packages.

    If a requested package is not installed, an empty string is returned instead.

    Parameters
    ----------
    *packages: str
        Names of the requested packages under which they exist in the current environment.
        For packages installed through ``pip``, this equals the PyPI package name.
    """

    key = "python"

    def __init__(self, *packages: str):
        self.packages = packages

    def __call__(self) -> dict[str, Any]:
        from importlib.metadata import PackageNotFoundError, version

        result: dict[str, Any] = dict()

        result["version"] = platform.python_version()
        result["implementation"] = platform.python_implementation()
        buildno, buildtime = platform.python_build()
        result["buildno"] = buildno
        result["buildtime"] = buildtime

        dependencies: dict[str, str] = {}
        for pkg in self.packages:
            try:
                dependencies[pkg] = version(pkg)
            except PackageNotFoundError:
                dependencies[pkg] = ""

        result["dependencies"] = dependencies
        return {self.key: result}


class GitEnvironmentInfo:
    """
    A context helper providing the current git commit, latest tag, and upstream repository name.

    Parameters
    ----------
    remote: str
        Remote name for which to provide info, by default ``"origin"``.
    """

    key = "git"

    def __init__(self, remote: str = "origin"):
        self.remote = remote

    def __call__(self) -> dict[str, dict[str, Any]]:
        import subprocess

        def git_subprocess(args: list[str]) -> subprocess.CompletedProcess:
            if platform.system() == "Windows":
                git = "git.exe"
            else:
                git = "git"

            return subprocess.run(  # nosec: B603
                [git, *args],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
            )

        result: dict[str, Any] = {
            "commit": "",
            "provider": "",
            "repository": "",
            "tag": "",
            "dirty": None,
        }

        # first, check if inside a repo.
        p = git_subprocess(["rev-parse", "--is-inside-work-tree"])
        # if not, return empty info.
        if p.returncode:
            return {"git": result}

        # secondly: get the current commit.
        p = git_subprocess(["rev-parse", "HEAD"])
        if not p.returncode:
            result["commit"] = p.stdout.strip()

        # thirdly, get the latest tag, without a short commit SHA attached.
        p = git_subprocess(["describe", "--tags", "--abbrev=0"])
        if not p.returncode:
            result["tag"] = p.stdout.strip()

        # and finally, get the remote repo name pointed to by the given remote.
        p = git_subprocess(["remote", "get-url", self.remote])
        if not p.returncode:
            remotename: str = p.stdout.strip()
            # it's an SSH remote.
            if "@" in remotename:
                prefix, sep = "git@", ":"
            else:
                # it is HTTPS.
                prefix, sep = "https://", "/"

            remotename = remotename.removeprefix(prefix)
            provider, reponame = remotename.split(sep, 1)

            result["provider"] = provider
            result["repository"] = reponame.removesuffix(".git")

        p = git_subprocess(["status", "--porcelain"])
        if not p.returncode:
            result["dirty"] = bool(p.stdout.strip())

        return {"git": result}


class CPUInfo:
    key = "cpu"

    def __init__(
        self,
        memunit: Literal["kB", "MB", "GB"] = "MB",
        frequnit: Literal["kHz", "MHz", "GHz"] = "MHz",
    ):
        self.memunit = memunit
        self.frequnit = frequnit
        self.conversion_table: dict[str, float] = {"k": 1e3, "M": 1e6, "G": 1e9}

    def __call__(self) -> dict[str, Any]:
        try:
            import psutil
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"context provider {self.__class__.__name__}() needs `psutil` installed. "
                f"To install, run `{sys.executable} -m pip install --upgrade psutil`."
            )

        result: dict[str, Any] = dict()

        # first, the platform info.
        result["architecture"] = platform.machine()
        result["bitness"] = platform.architecture()[0]
        result["processor"] = platform.processor()
        result["system"] = platform.system()
        result["system-version"] = platform.release()

        freq_struct = psutil.cpu_freq()
        freq_conversion = self.conversion_table[self.frequnit[0]]
        # result is in MHz, so we convert to Hz and apply the conversion factor.
        result["frequency"] = freq_struct.current * 1e6 / freq_conversion
        result["frequency_unit"] = self.frequnit
        result["min_frequency"] = freq_struct.min
        result["max_frequency"] = freq_struct.max
        result["num_cpus"] = psutil.cpu_count(logical=False)
        result["num_logical_cpus"] = psutil.cpu_count()

        mem_struct = psutil.virtual_memory()
        mem_conversion = self.conversion_table[self.memunit[0]]
        # result is in bytes, so no need for base conversion.
        result["total_memory"] = mem_struct.total / mem_conversion
        result["memory_unit"] = self.memunit
        # TODO: Lacks CPU cache info, which requires a solution other than psutil.
        return {self.key: result}


class Context:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = data or {}

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    def __eq__(self, other):
        if not isinstance(other, Context):
            raise NotImplementedError(
                f"cannot compare {type(self)} for equality with type {type(other)}"
            )
        return self._data.__eq__(other._data)

    @property
    def data(self):
        return self._data

    @staticmethod
    def _ctx_items(d: dict[str, Any], prefix: str, sep: str) -> Iterator[tuple[str, Any]]:
        """
        Iterate over nested dictionary items. Keys are formatted to indicate their nested path.

        Parameters
        ----------
        d : dict[str, Any]
            Dictionary to iterate over.
        prefix : str
            Current prefix to prepend to keys, used for recursion to build the full key path.
        sep : str
            The separator to use between levels of nesting in the key path.

        Yields
        ------
        tuple[str, Any]
            Iterator over key-value tuples.
        """
        for k, v in d.items():
            new_key = prefix + sep + k if prefix else k
            if isinstance(v, dict):
                yield from Context._ctx_items(d=v, prefix=new_key, sep=sep)
            else:
                yield new_key, v

    def keys(self, sep: str = ".") -> Iterator[str]:
        """
        Keys of the context dictionary, with an optional separator for nested keys.

        Parameters
        ----------
        sep : str, optional
            Separator to use for nested keys.

        Yields
        ------
        str
            Iterator over the context dictionary keys.
        """
        for k, _ in self._ctx_items(d=self._data, prefix="", sep=sep):
            yield k

    def values(self) -> Iterator[Any]:
        """
        Values of the context dictionary, including values from nested dictionaries.

        Yields
        ------
        Any
            Iterator over all values in the context dictionary.
        """
        for _, v in self._ctx_items(d=self._data, prefix="", sep=""):
            yield v

    def items(self, sep: str = ".") -> Iterator[tuple[str, Any]]:
        """
        Items (key-value pairs) of the context dictionary, with an separator for nested keys.

        Parameters
        ----------
        sep : str, optional
            Separator to use for nested dictionary keys.

        Yields
        ------
        tuple[str, Any]
            Iterator over the items of the context dictionary.
        """
        yield from self._ctx_items(d=self._data, prefix="", sep=sep)

    def add(self, provider: ContextProvider, replace: bool = False) -> None:
        """
        Adds data from a provider to the context.

        Parameters
        ----------
        provider : ContextProvider
            The provider to inject into this context.
        replace : bool
            Whether to replace existing context values upon key collision. Raises ValueError otherwise.
        """
        self.update(Context.make(provider()), replace=replace)

    def update(self, other: "Context", replace: bool = False) -> None:
        """
        Updates the context.

        Parameters
        ----------
        other : Context
            The other context to update this context with.
        replace : bool
            Whether to replace existing context values upon key collision. Raises ValueError otherwise.

        Raises
        ------
        ValueError
            If ``other contains top-level keys already present in the context and ``replace=False``.
        """
        duplicates = set(self.keys()) & set(other.keys())
        if not replace and duplicates:
            dupe, *_ = duplicates
            raise ValueError(f"got multiple values for context key {dupe!r}")
        self._data.update(other._data)

    @staticmethod
    def _flatten_dict(d: dict[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:
        """
        Turn a nested dictionary into a flattened dictionary.

        Parameters
        ----------
        d : dict[str, Any]
            (Possibly) nested dictionary to flatten.
        prefix : str
            Key prefix to apply at the top-level (nesting level 0).
        sep : str
            Separator on which to join keys, "." by default.

        Returns
        -------
        dict[str, Any]
            The flattened dictionary.
        """

        items: list[tuple[str, Any]] = []
        for key, value in d.items():
            new_key = prefix + sep + key if prefix else key
            if isinstance(value, dict):
                items.extend(Context._flatten_dict(d=value, prefix=new_key, sep=sep).items())
            else:
                items.append((new_key, value))
        return dict(items)

    def flatten(self, sep: str = ".") -> dict[str, Any]:
        """
        Flatten the context's dictionary, converting nested dictionaries into a single dictionary with keys separated by `sep`.

        Parameters
        ----------
        sep : str, optional
            The separator used to join nested keys.

        Returns
        -------
        dict[str, Any]
            The flattened context values as a Python dictionary.
        """

        return self._flatten_dict(self._data, prefix="", sep=sep)

    @staticmethod
    def unflatten(d: dict[str, Any], sep: str = ".") -> dict[str, Any]:
        """
        Recursively unflatten a dictionary by expanding keys seperated by `sep` into nested dictionaries.

        Parameters
        ----------
        d : dict[str, Any]
            The dictionary to unflatten.
        sep : str, optional
            The separator used in the flattened keys.

        Returns
        -------
        dict[str, Any]
            The unflattened dictionary.
        """
        sorted_keys = sorted(d.keys())
        unflattened = {}
        for prefix, keys in itertools.groupby(sorted_keys, key=lambda key: key.split(sep, 1)[0]):
            key_group = list(keys)
            if len(key_group) == 1 and sep not in key_group[0]:
                unflattened[prefix] = d[prefix]
            else:
                nested_dict = {key.split(sep, 1)[1]: d[key] for key in key_group}
                unflattened[prefix] = Context.unflatten(d=nested_dict, sep=sep)
        return unflattened

    @classmethod
    def make(cls, d: dict[str, Any]) -> "Context":
        """
        Create a new Context instance from a given dictionary.

        Parameters
        ----------
        d : dict[str, Any]
            The initialization dictionary.

        Returns
        -------
        Context
            The new Context instance.
        """
        return cls(data=cls.unflatten(d))
