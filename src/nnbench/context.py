"""Utilities for collecting context key-value pairs as metadata in benchmark runs."""

import platform
import sys
from collections.abc import Callable, Sequence
from typing import Any, Literal

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
    packages: Sequence[str]
        Names of the requested packages under which they exist in the current environment.
        For packages installed through ``pip``, this equals the PyPI package name.
    """

    key = "python"

    def __init__(self, packages: Sequence[str] = ()):
        self.packages = tuple(packages)

    def __call__(self) -> dict[str, Any]:
        from importlib.metadata import PackageNotFoundError, version

        result: dict[str, Any] = dict()

        result["version"] = platform.python_version()
        result["implementation"] = platform.python_implementation()
        buildno, buildtime = platform.python_build()
        result["buildno"] = buildno
        result["buildtime"] = buildtime

        packages: dict[str, str] = {}
        for pkg in self.packages:
            try:
                packages[pkg] = version(pkg)
            except PackageNotFoundError:
                packages[pkg] = ""

        result["packages"] = packages
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
                capture_output=True,
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
            if "@" in remotename:
                # it's an SSH remote.
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
    """
    A context helper providing information about the host machine's CPU
    capabilities, operating system, and amount of memory.

    Parameters
    ----------
    memunit: Literal["kB", "MB", "GB"]
        The unit to display memory size in (either "kB" for kilobytes,
        "MB" for Megabytes, or "GB" for Gigabytes).
    frequnit: Literal["kHz", "MHz", "GHz"]
        The unit to display CPU clock speeds in (either "kHz" for kilohertz,
        "MHz" for Megahertz, or "GHz" for Gigahertz).
    """

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

        try:
            # The CPU frequency is not available on some ARM devices
            freq_struct = psutil.cpu_freq()
            result["min_frequency"] = float(freq_struct.min)
            result["max_frequency"] = float(freq_struct.max)
            freq_conversion = self.conversion_table[self.frequnit[0]]
            # result is in MHz, so we convert to Hz and apply the conversion factor.
            result["frequency"] = freq_struct.current * 1e6 / freq_conversion
        except RuntimeError:
            result["frequency"] = 0.0
            result["min_frequency"] = 0.0
            result["max_frequency"] = 0.0

        result["frequency_unit"] = self.frequnit
        result["num_cpus"] = psutil.cpu_count(logical=False)
        result["num_logical_cpus"] = psutil.cpu_count()

        mem_struct = psutil.virtual_memory()
        mem_conversion = self.conversion_table[self.memunit[0]]
        # result is in bytes, so no need for base conversion.
        result["total_memory"] = mem_struct.total / mem_conversion
        result["memory_unit"] = self.memunit
        return {self.key: result}


builtin_providers: dict[str, ContextProvider] = {
    "cpu": CPUInfo(),
    "git": GitEnvironmentInfo(),
    "python": PythonInfo(),
}
