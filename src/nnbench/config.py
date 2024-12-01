"""Utilities for parsing an nnbench config block out of a pyproject.toml file."""

import logging
import os
import sys
from dataclasses import dataclass
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
    arguments: dict[str, Any]
    """Arguments needed to instantiate the context provider class,
    given as key-value pairs in the table."""


@dataclass(frozen=True)
class nnbenchConfig:
    log_level: str
    """Log level to use for the ``nnbench`` module root logger."""
    context: list[ContextProviderDef]
    """A list of context provider definitions found in pyproject.toml."""

    @classmethod
    def empty(cls) -> Self:
        """An empty default config, returned if no pyproject.toml is found."""
        return cls(log_level="NOTSET", context=[])

    @classmethod
    def from_toml(cls, d: dict[str, Any]) -> Self:
        """
        Returns an nnbench CLI config by parsing a [tool.nnbench] block from a
        pyproject.toml file.

        Parameters
        ----------
        d: dict[str, Any]
            Mapping containing the [tool.nnbench] block as obtained by
            ``tomllib.load``.

        Returns
        -------
        Self
            An nnbench config instance with the values from pyproject.toml,
            with defaults for values that were not set explicitly.
        """
        provider_map = d.get("context", {})
        context = [ContextProviderDef(**cpd) for cpd in provider_map.values()]
        log_level = d.get("log-level", "NOTSET")
        return cls(log_level=log_level, context=context)


def locate_pyproject() -> os.PathLike[str]:
    """
    Locate a pyproject.toml file by walking up from the current directory,
    and checking for file existence, stopping at the current user home
    directory.

    If no pyproject.toml file can be found, a RuntimeError is raised.

    Returns
    -------
    os.PathLike[str]
        The path to pyproject.toml.

    """
    cwd = Path.cwd()
    for p in (cwd, *cwd.parents):
        if (pyproject_cand := (p / "pyproject.toml")).exists():
            return pyproject_cand
        if p == Path.home():
            break
    raise RuntimeError("could not locate pyproject.toml")


def parse_nnbench_config(pyproject_path: str | os.PathLike[str] | None = None) -> nnbenchConfig:
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
    nnbenchConfig
        The loaded config if found, or a default config.

    """
    if pyproject_path is None:
        try:
            pyproject_path = locate_pyproject()
        except RuntimeError:
            # pyproject.toml cannot be found, so return an empty config.
            return nnbenchConfig.empty()

    with open(pyproject_path, "rb") as fp:
        pyproject_cfg = tomllib.load(fp)
    nnbench_cfg = pyproject_cfg.get("tool", {}).get("nnbench", {})
    return nnbenchConfig.from_toml(nnbench_cfg)
