"""Various utilities related to benchmark collection, filtering, and more."""

import contextlib
import importlib.util
import itertools
import os
import sys
import time
from collections.abc import Callable, Generator, Iterable
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar

T = TypeVar("T")


def collate(_its: Iterable[T | Iterable[T]]) -> Generator[T, None, None]:
    for _it in _its:
        if isinstance(_it, Iterable):
            yield from _it
        else:
            yield _it


def flatten(d: dict[str, Any], sep: str = ".", prefix: str = "") -> dict:
    """
    Given a nested dictionary and a separator, returns another dictionary
    of depth 1, containing values under nested keys joined by the separator.

    Parameters
    ----------
    d: dict[str, Any]
        A dictionary to be flattened. All nested dictionaries must contain
        string keys only.
    sep: str
        The separator string to join keys on.
    prefix: str
        A prefix to apply to keys when calling ``flatten()`` recursively.
        You shouldn't need to pass this yourself.

    Returns
    -------
    dict[str, Any]
        The flattened dictionary.

    Examples
    --------
    >>> flatten({"a": 1, "b": {"c": 2}})
    {"a": 1, "b.c": 2}
    """
    d_flat = {}
    for k, v in d.items():
        new_key = prefix + sep + k if prefix else k
        if isinstance(v, dict):
            d_flat.update(flatten(v, sep=sep, prefix=new_key))
        else:
            d_flat[new_key] = v
    return d_flat


def unflatten(d: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    """
    Unflatten a previously flattened dictionary.

    Any key that does not contain the separator is passed through unchanged.

    This is, as the name suggests, the inverse operation to ``nnbench.util.flatten()``.

    Parameters
    ----------
    d: dict[str, Any]
        The dictionary to unflatten.
    sep: str
        The separator to split keys on, introducing dictionary nesting.

    Returns
    -------
    dict[str, Any]

    Examples
    --------
    >>> unflatten({"a": 1, "b.c": 2})
    {"a": 1, "b": {"c": 2}}

    >>> d = {"a": 1, "b": {"c": 2}}
    >>> unflatten(flatten(d)) == d
    True
    """
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


def exists_module(name: str | os.PathLike[str]) -> bool:
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
    """
    Convert a file name to its corresponding Python module name.

    Examples
    --------
    >>> modulename("path/to/my/file.py")
    "path.to.my.module"
    """
    fpath = Path(file).with_suffix("")
    if len(fpath.parts) == 1:
        return str(fpath)

    filename = fpath.as_posix()
    return filename.replace("/", ".")


def import_file_as_module(file: str | os.PathLike[str]) -> ModuleType:
    """
    Import a Python file as a module using importlib.

    Raises an error if the given path is not a Python file, or if the
    module spec could not be constructed.

    Parameters
    ----------
    file: str | os.PathLike[str]
        The file to import as a Python module.

    Returns
    -------
    ModuleType
        The imported module, with its file location set as ``__file__``.

    """
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


def all_python_files(_dir: str | os.PathLike[str]) -> Generator[Path, None, None]:
    if sys.version_info >= (3, 12):
        pathgen = Path(_dir).walk(top_down=True)
    else:
        pathgen = os.walk(_dir, topdown=True)

    for root, dirs, files in pathgen:
        proot = Path(root)
        # do not descend into potentially large __pycache__ dirs.
        if "__pycache__" in dirs:
            dirs.remove("__pycache__")
        for file in files:
            fp = proot / file
            if fp.suffix == ".py":
                yield fp


def qualname(fn: Callable) -> str:
    if fn.__name__ == fn.__qualname__:
        return fn.__name__
    return f"{fn.__qualname__}.{fn.__name__}"


@contextlib.contextmanager
def timer(bm: dict[str, Any]) -> Generator[None, None, None]:
    start = time.perf_counter_ns()
    try:
        yield
    finally:
        end = time.perf_counter_ns()
        bm["time_ns"] = end - start
