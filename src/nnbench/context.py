"""Utilities for collecting context key-value pairs as metadata in benchmark runs."""
from __future__ import annotations

import platform
import sys
from typing import Any, Callable, Iterable, Literal

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


class BenchmarkContext:
    def __init__(
        self,
        context: ContextElement | Iterable[dict | ContextProvider] | None = None,
        name: str | None = None,
    ) -> None:
        self.context: dict[str, Any] = {}
        if context:
            self.insert(context)
        self.name = name or f"BenchmarkContext_{id(self)}"

    @property
    def context_keys(self) -> set:
        return set(self.context)

    @staticmethod
    def _duplicate_keys(this: dict[str, Any], other: dict[str, Any]) -> set:
        this_keys = set(this.keys())
        other_keys = set(other.keys())
        return this_keys & other_keys

    def insert_dict(self, context: dict) -> BenchmarkContext:
        for key in self._duplicate_keys(self.context, context):
            if self.context[key] != context[key]:
                raise ValueError(f"got multiple values for context key {key!r}")
        self.context |= context
        return self

    def insert_provider(self, provider: ContextProvider) -> BenchmarkContext:
        context = provider()
        if not isinstance(context, dict):
            raise ValueError(f"Provider {provider} did not return a context dict. Got {context}")
        return self.insert_dict(context)

    def insert_single(self, context: ContextElement) -> BenchmarkContext:
        if callable(context):
            self.insert_provider(context)
        elif isinstance(context, dict):
            self.insert_dict(context)
        else:
            raise ValueError(f"Unknown context type {context}.")
        return self

    def insert(self, contexts: ContextElement | Iterable[ContextElement]) -> BenchmarkContext:
        if isinstance(contexts, Iterable) and not isinstance(contexts, dict):
            for context in contexts:
                self.insert_single(context)
        else:
            self.insert_single(contexts)
        return self

    def __getitem__(self, key: str) -> Any:
        return self.context[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.context[key] = value

    def merge(
        self, other: BenchmarkContext, inplace: bool = False, placeholder: str = "<MISSING>"
    ) -> BenchmarkContext:
        duplicate_keys = self._duplicate_keys(self.context, other.context)
        left_only_keys = self.context_keys ^ other.context_keys
        right_only_keys = other.context_keys ^ self.context_keys
        union_keys = duplicate_keys | left_only_keys | right_only_keys
        merged = {}
        for key in union_keys:
            if key in duplicate_keys and self.context[key] == other.context[key]:
                merged[key] = self.context[key]
            elif key in duplicate_keys:
                if self.name == other.name:
                    raise ValueError(
                        "Cannot merge BenchmarkContext with same name but deviating values for keys of same name."
                    )
                merged[f"{self.name}_{key}"] = self.context[key]
                merged[f"{other.name}_{key}"] = other.context[key]
            elif key in left_only_keys:
                merged[f"{self.name}_{key}"] = self.context[key]
                merged[f"{other.name}_{key}"] = placeholder
            elif key in right_only_keys:
                merged[f"{self.name}_{key}"] = placeholder
                merged[f"{other.name}_{key}"] = other.context[key]
        merged_name = f"merged_{self.name}_{other.name}"
        if inplace:
            self.context = merged
            self.name = merged_name
            return self
        else:
            return BenchmarkContext(context=merged, name=merged_name)

    def copy(self) -> BenchmarkContext:
        new = BenchmarkContext(context=self.context, name=self.name)
        return new

    def __add__(
        self, other: BenchmarkContext | ContextElement | Iterable[ContextElement]
    ) -> BenchmarkContext:
        new_instance = self.copy()
        if isinstance(other, BenchmarkContext):
            return new_instance.merge(other, inplace=False)
        else:
            return self.copy().insert(other)
