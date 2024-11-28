"""Utilities for parsing an nnbench config block out of a pyproject.toml."""

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
    name: str
    classpath: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class nnbenchConfig:
    log_level: str
    context: list[ContextProviderDef]

    @classmethod
    def empty(cls) -> Self:
        return cls(log_level="NOTSET", context=[])

    @classmethod
    def from_toml(cls, d: dict[str, Any]) -> Self:
        provider_map = d.get("context", {})
        context = [ContextProviderDef(**cpd) for cpd in provider_map.values()]
        log_level = d.get("log-level", "NOTSET")
        return cls(log_level=log_level, context=context)


def locate_pyproject() -> os.PathLike[str]:
    cwd = Path.cwd()
    for p in (cwd, *cwd.parents):
        if (pyproject_cand := (p / "pyproject.toml")).exists():
            return pyproject_cand
        if p == Path.home():
            break
    raise RuntimeError("could not locate pyproject.toml")


def parse_nnbench_config(pyproject_path: str | os.PathLike[str] | None = None) -> nnbenchConfig:
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
