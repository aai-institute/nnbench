import collections
from typing import Any


def nullcols(_benchmarks: list[dict[str, Any]]) -> tuple[str, ...]:
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
    tuple[str, ...]
        Tuple of the columns (key names) that only contain false-ish values
        across all benchmarks.
    """
    nulls: dict[str, bool] = collections.defaultdict(bool)
    for bm in _benchmarks:
        for k, v in bm.items():
            nulls[k] = nulls[k] or bool(v)
    return tuple(k for k, v in nulls.items() if not v)


def flatten(d: dict[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:
    """
    Turn a nested dictionary into a flattened dictionary.

    Parameters
    ----------
    d: dict[str, Any]
        (Possibly) nested dictionary to flatten.
    prefix: str
        Key prefix to apply at the top-level (nesting level 0).
    sep: str
        Separator on which to join keys, "." by default.

    Returns
    -------
    dict[str, Any]
        The flattened dictionary.
    """

    items: list[tuple[str, Any]] = []
    for key, value in d.items():
        new_key = prefix + sep + key if prefix else key
        if isinstance(value, dict):
            items.extend(flatten(value, new_key, sep).items())
        else:
            items.append((new_key, value))
    return dict(items)
