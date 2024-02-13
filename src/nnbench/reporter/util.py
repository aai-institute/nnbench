import collections
from typing import Any


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
