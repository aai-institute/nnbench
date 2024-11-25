"""Various utilities related to benchmark collection, filtering, and more."""

import importlib
import importlib.util
import itertools
import os
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType


def flatten(d: dict, sep: str = ".", prefix: str = "") -> dict:
    d_flat = {}
    for k, v in d.items():
        new_key = prefix + sep + k if prefix else k
        if isinstance(v, dict):
            d_flat.update(flatten(v, sep=sep, prefix=new_key))
        else:
            d_flat[k] = v
    return d_flat


def unflatten(d: dict, sep: str = ".") -> dict:
    sorted_keys = sorted(d.keys())
    unflattened = {}
    for prefix, keys in itertools.groupby(sorted_keys, key=lambda key: key.split(sep, 1)[0]):
        key_group = list(keys)
        if len(key_group) == 1 and sep not in key_group[0]:
            unflattened[prefix] = d[prefix]
        else:
            nested_dict = {key.split(sep, 1)[1]: d[key] for key in key_group}
            unflattened[prefix] = unflatten(nested_dict, sep=sep)
    return unflattened


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
    fpath = Path(file).with_suffix("")
    if len(fpath.parts) == 1:
        return str(fpath)

    filename = fpath.as_posix()
    return filename.replace("/", ".")


def import_file_as_module(file: str | os.PathLike[str]) -> ModuleType:
    fpath = Path(file)
    if not fpath.is_file() or fpath.suffix != ".py":
        raise ValueError(f"path {str(file)!r} is not a Python file")

    # TODO: Recomputing this map in a loop can be expensive if many modules are loaded.
    modmap = {m.__file__: m for m in sys.modules.values() if getattr(m, "__file__", None)}
    spath = str(fpath)
    if spath in modmap:
        # if the module under "file" has already been loaded, return it,
        # otherwise we get nasty type errors in collection.
        return modmap[spath]

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
