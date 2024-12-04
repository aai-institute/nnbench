"""Utilities for parsing a ``[tool.nnbench]`` config block out of a pyproject.toml file."""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    from typing import Self

    import tomllib
else:
    import tomli as tomllib
    from typing_extensions import Self

logger = logging.getLogger("nnbench.config")


@dataclass
class ContextProviderDef:
    """
    A POD struct representing a custom context provider definition in a
    pyproject.toml table.
    """

    name: str
    """Name under which the provider should be registered by nnbench."""
    classpath: str
    """Full path to the class or callable returning the context dict."""
    arguments: dict[str, Any] = field(default_factory=dict)
    """
    Arguments needed to instantiate the context provider class,
    given as key-value pairs in the table.
    If the class path points to a function, no arguments may be given."""


@dataclass(frozen=True)
class NNBenchConfig:
    log_level: str
    """Log level to use for the ``nnbench`` module root logger."""
    context: list[ContextProviderDef]
    """A list of context provider definitions found in pyproject.toml."""

    @classmethod
    def from_toml(cls, d: dict[str, Any]) -> Self:
        """
        Returns an nnbench CLI config by processing fields obtained from
        parsing a [tool.nnbench] block in a pyproject.toml file.

        Parameters
        ----------
        d: dict[str, Any]
            Mapping containing the [tool.nnbench] table contents,
            as obtained by ``tomllib.load()``.

        Returns
        -------
        Self
            An nnbench config instance with the values from pyproject.toml,
            and defaults for values that were not set explicitly.
        """
        log_level = d.get("log-level", "NOTSET")
        provider_map = d.get("context", {})
        context = [ContextProviderDef(**cpd) for cpd in provider_map.values()]
        return cls(log_level=log_level, context=context)


def locate_pyproject(stop: os.PathLike[str] = Path.home()) -> os.PathLike[str] | None:
    """
    Locate a pyproject.toml file by walking up from the current directory,
    and checking for file existence, stopping at ``stop`` (by default, the
    current user home directory).

    If no pyproject.toml file can be found at any level, returns None.

    Returns
    -------
    os.PathLike[str] | None
        The path to pyproject.toml.
    """
    cwd = Path.cwd()
    for p in (cwd, *cwd.parents):
        if (pyproject_cand := (p / "pyproject.toml")).exists():
            return pyproject_cand
        if p == stop:
            break
    logger.debug(f"could not locate pyproject.toml in directory {cwd}")
    return None


def parse_nnbench_config(pyproject_path: str | os.PathLike[str] | None = None) -> NNBenchConfig:
    """
    Load an nnbench config from a given pyproject.toml file.

    If no path to the pyproject.toml file is given, an attempt at autodiscovery
    will be made. If that is unsuccessful, an empty config is returned.

    Parameters
    ----------
    pyproject_path: str | os.PathLike[str] | None
        Path to the current project's pyproject.toml file, optional.

    Returns
    -------
    NNBenchConfig
        The loaded config if found, or a default config.
    """
    pyproject_path = pyproject_path or locate_pyproject()
    if pyproject_path is None:
        # pyproject.toml could not be found, so return an empty config.
        return NNBenchConfig.from_toml({})

    with open(pyproject_path, "rb") as fp:
        pyproject = tomllib.load(fp)
        return NNBenchConfig.from_toml(pyproject.get("tool", {}).get("nnbench", {}))
