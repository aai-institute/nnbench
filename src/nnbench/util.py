"""Various utilities related to benchmark collection, filtering, and more."""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType


def ismodule(name: str | os.PathLike[str]) -> bool:
    """Checks if the current interpreter has an available Python module named `name`."""
    name = str(name)
    if name in sys.modules:
        return True

    root, *parts = name.split(".")

    for part in parts:
        spec = importlib.util.find_spec(root)
        if spec is None:
            return False
        root += f".{part}"

    return importlib.util.find_spec(name) is not None


def modulename(file: str | os.PathLike[str]) -> str:
    """Convert a file name to its corresponding Python module name."""
    fpath = Path(file)
    if len(fpath.parts) == 1:
        return str(fpath)

    filename = fpath.with_suffix("").as_posix()
    return filename.replace("/", ".")


def import_file_as_module(file: str | os.PathLike[str]) -> ModuleType:
    fpath = Path(file)
    if not fpath.is_file() or fpath.suffix != ".py":
        raise ValueError(f"path {str(file)!r} is not a Python file")

    # TODO: For absolute paths, the resulting module name will be horrifying
    #  -> find a sensible cutoff point (project root)
    modname = modulename(fpath)
    if modname in sys.modules:
        # return already loaded module
        return sys.modules[modname]

    spec: ModuleSpec | None = importlib.util.spec_from_file_location(modname, fpath)
    if spec is None:
        raise RuntimeError(f"could not import module {fpath}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module
