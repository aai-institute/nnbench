"""Utilities for collecting context key-value pairs as metadata in benchmark runs."""
from __future__ import annotations

import itertools
import platform
import sys
from typing import Any, Callable, ItemsView, KeysView, Literal, ValuesView

ContextProvider = Callable[[], dict[str, Any]]
"""A function providing a dictionary of context values."""

ContextElement = dict[str, Any] | ContextProvider


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
                [git, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
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
    def __init__(self, context: dict[str, Any] | ContextProvider | "Context" | None = None) -> None:
        self._ctx_dict: dict[str, Any] = {}
        if context:
            self.merge(context, inplace=True)

    def keys(self) -> KeysView[str]:
        return self._ctx_dict.keys()

    def values(self) -> ValuesView[Any]:
        return self._ctx_dict.values()

    def items(self) -> ItemsView[str, Any]:
        return self._ctx_dict.items()

    def merge(
        self,
        other: "Context" | dict[str, Any] | ContextProvider,
        this_ctx: dict[str, Any] | None = None,
        inplace: bool = True,
    ) -> "Context":
        """
        Merge another dictionary, Context, or ContextProvider into this Context.

        Parameters
        ----------
        other : Context | dict[str, Any] | ContextProvider
            The other context to merge into this one.
        this_ctx : dict[str, Any] | None, optional
            The target dictionary to merge into. Takes this classes context if None.
        inplace : bool, optional
            If True, modifies the current context in place. Otherwise, returns a new merged Context.

        Raises
        ------
        TypeError
            If 'other' is not a Context, dict, or callable returning a dict, or if 'this_ctx'
            is specified but not a dict.
        ValueError
            If 'other' is a callable that does not return a dictionary.

        Returns
        -------
        Context
            The current context after merging, if inplace is True. Otherwise, a new Context instance.
        """
        if isinstance(other, Context):
            other = other._ctx_dict
        elif callable(other):
            result = other()
            if not isinstance(result, dict):
                raise ValueError("Provider did not return a dictionary.")
            other = result
        elif isinstance(other, dict):
            pass
        else:
            raise TypeError(f"Unknown type for source, got {type(other)}")

        if this_ctx is None:
            this = self._ctx_dict
        elif isinstance(this_ctx, dict):
            this = this_ctx
        else:
            raise TypeError(f"Unkown type for target, got {type(this_ctx)}")

        merged = {**other, **this}

        intersection_keys = set(this.keys()) & (set(other.keys()))
        for key in intersection_keys:
            if this[key] == other[key]:
                merged[key] = this[key]
            elif isinstance(this[key], dict) and isinstance(other[key], dict):
                merged[key] = self.merge(other=other[key], this_ctx=this[key], inplace=False)
            else:
                raise ValueError(f"Key collision upon merge, {key}")

        if inplace:
            self._ctx_dict = merged
            return self
        else:
            return Context(context=merged)

    @staticmethod
    def flatten_dict(d: dict[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:
        """
        Turn a nested dictionary into a flattened dictionary.

        Parameters
        ----------
        d: dict[str, Any]
            (Possibly) nested dictionary to flatten.
        prefix: str
            Key prefix to apply at the top-level (nesting level 0).
        sep: str
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
                items.extend(Context.flatten_dict(value, new_key, sep).items())
            else:
                items.append((new_key, value))
        return dict(items)

    def flatten(self, prefix: str = "", sep: str = ".", inplace: bool = True) -> "Context":
        """
        Flatten the context's dictionary, converting nested dictionaries into a single dictionary with keys separated by `sep`.

        Parameters
        ----------
        prefix : str, optional
            A prefix to prepend to each key in the flattened dictionary.
        sep : str, optional
            The separator used to join nested keys.
        inplace : bool, optional
            If True, the current context's dictionary is modified in-place. Otherwise, return a new Context instance.

        Returns
        -------
        Context
            The current context instance if `inplace` is True; otherwise, a new, flattened Context instance.
        """

        if inplace:
            self._ctx_dict = self.flatten_dict(d=self._ctx_dict, prefix=prefix, sep=sep)
            return self
        else:
            return Context(context=self.flatten_dict(self._ctx_dict, prefix=prefix, sep=sep))

    @staticmethod
    def unflatten_dict(d: dict[str, Any], sep: str = ".") -> dict[str, Any]:
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
                unflattened[prefix] = Context.unflatten_dict(d=nested_dict, sep=sep)
        return unflattened

    def unflatten(self, sep: str = ".", inplace: bool = True) -> "Context":
        """
        Unflatten the context's dictionary, expanding keys into nested dictionaries on the `sep` value.

        Parameters
        ----------
        sep : str, optional
            The separator used in the flattened keys.
        inplace : bool, optional
            If True, the current context's dictionary is modified in-place. Otherwise, return a new Context instance.

        Returns
        -------
        Context
            The current context instance if `inplace` is True; otherwise, a new unflattened Context instance.
        """
        if inplace:
            self._ctx_dict = self.unflatten_dict(self._ctx_dict, sep=sep)
            return self
        else:
            return Context(context=self.unflatten_dict(self._ctx_dict, sep=sep))

    @staticmethod
    def filter_dict(d: dict[str, Any], predicate: Callable[[str, Any], bool]) -> dict[str, Any]:
        """
        Recursively filter a dictionary based on a predicate.

        Parameters
        ----------
        d : dict[str, Any]
            The dictionary to filter.
        predicate : Callable[[str, Any], bool]
            The function to determine if a key-value pair should be included in the filtered dictionary.

        Returns
        -------
        dict[str, Any]
            The filtered dictionary, including applicable nested dictionaries.
        """
        filtered: dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                nested_filtered = Context.filter_dict(d=v, predicate=predicate)
                if nested_filtered:
                    filtered[k] = nested_filtered
            elif predicate(k, v):
                filtered[k] = v
        return filtered

    def filter(self, predicate: Callable[[str, Any], bool], inplace: bool = False) -> "Context":
        """
        Filter the context's dictionary based on a given predicate, including nested dictionaries.

        Parameters
        ----------
        predicate : Callable[[str, Any], bool]
            A function that takes a key and a value as arguments and returns True if the element should be included in the filtered result.
        inplace : bool, optional
            If True, modifies the current context in place. Otherwise, returns a new Context instance with the filtered result.

        Returns
        -------
        Context
            The current context after filtering, if inplace is True. Otherwise, a new filtered Context instance.
        """
        if inplace:
            self._ctx_dict = self.filter_dict(self._ctx_dict, predicate=predicate)
            return self
        else:
            return Context(context=self.filter_dict(self._ctx_dict, predicate=predicate))

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
        return cls(context=d)
