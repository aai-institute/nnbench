"""Utilities for parsing an nnbench config block out of a pyproject.toml."""

import logging
import os
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = logging.getLogger("nnbench.config")


def locate_pyproject() -> os.PathLike[str]:
    cwd = Path.cwd()
    for p in (cwd, *cwd.parents):
        if (pyproject_cand := (p / "pyproject.toml")).exists():
            return pyproject_cand
        if p == Path.home():
            break
    raise RuntimeError("could not locate pyproject.toml")


# TODO: It's not just Any in the return, but a few well-defined context keys -
# put some effort into parsing them
def parse_nnbench_config(pyproject_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    if pyproject_path is None:
        try:
            pyproject_path = locate_pyproject()
        except RuntimeError:
            # TODO: This should hold sensible default values for all top-level keys,
            # preferably as a dataclass.
            # pyproject.toml cannot be found, so return an empty config.
            return {"log-level": "NOTSET"}

    with open(pyproject_path, "rb") as fp:
        config = tomllib.load(fp)
    return config.get("tool", {}).get("nnbench", {})
