import collections
import inspect
from typing import Any, get_args, get_origin


def is_memo(v: Any) -> bool:
    return callable(v) and len(inspect.signature(v).parameters) == 0


def is_memo_type(t: type) -> bool:
    return get_origin(t) is collections.abc.Callable and get_args(t)[0] == []
