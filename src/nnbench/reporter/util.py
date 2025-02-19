import collections
import os
import re
from pathlib import Path
from typing import IO, Any


def nullcols(_benchmarks: list[dict[str, Any]]) -> set[str]:
    """
    Extracts columns that only contain false-ish data from a list of benchmarks.

    Since this data is most often not interesting, the result of this
    can be used to filter out these columns from the benchmark dictionaries.

    Parameters
    ----------
    _benchmarks: list[dict[str, Any]]
        The benchmarks to filter.

    Returns
    -------
    set[str]
        Set of the columns (key names) that only contain false-ish values
        across all benchmarks.
    """
    nulls: dict[str, bool] = collections.defaultdict(bool)
    for bm in _benchmarks:
        for k, v in bm.items():
            nulls[k] = nulls[k] or bool(v)
    return set(k for k, v in nulls.items() if not v)


def get_protocol(url: str | os.PathLike[str]) -> str:
    url = str(url)
    parts = re.split(r"(::|://)", url, maxsplit=1)
    if len(parts) > 1:
        return parts[0]
    return "file"


def get_extension(f: str | os.PathLike[str] | IO) -> str:
    """
    Given a path or file-like object, returns file extension
    (can be the empty string, if the file has no extension).
    """
    if isinstance(f, str | os.PathLike):
        return Path(f).suffix
    else:
        return Path(f.name).suffix
