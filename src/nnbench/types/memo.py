from __future__ import annotations

import collections
import functools
import inspect
import logging
import threading
from typing import Any, Callable, Generic, TypeVar, get_args, get_origin

T = TypeVar("T")
Variable = tuple[str, type, Any]

_memo_cache: dict[int, Any] = {}
_cache_lock = threading.Lock()

logger = logging.getLogger(__name__)


def is_memo(v: Any) -> bool:
    return callable(v) and len(inspect.signature(v).parameters) == 0


def is_memo_type(t: type) -> bool:
    return get_origin(t) is collections.abc.Callable and get_args(t)[0] == []


def memo_cache_size() -> int:
    """
    Get the current size of the memo cache.

    Returns
    -------
    int
        The number of items currently stored in the memo cache.
    """
    return len(_memo_cache)


def clear_memo_cache() -> None:
    """
    Clear all items from memo cache in a thread_safe manner.
    """
    with _cache_lock:
        _memo_cache.clear()


def evict_memo(_id: int) -> Any:
    """
    Pop cached item with key `_id` from the memo cache.

    Parameters
    ----------
    _id : int
        The unique identifier (usually the id assigned by the Python interpreter) of the item to be evicted.

    Returns
    -------
    Any
        The value that was associated with the removed cache entry. If no item is found with the given `_id`, a KeyError is raised.
    """
    with _cache_lock:
        return _memo_cache.pop(_id)


def get_memo_by_value(val: Any) -> int | None:
    for k, v in _memo_cache.items():
        if v is val:
            return k
    return None


def cached_memo(fn: Callable) -> Callable:
    """
    Decorator that caches the result of a method call based on the instance ID.

    Parameters
    ----------
    fn: Callable
        The method to memoize.

    Returns
    -------
    Callable
        A wrapped version of the method that caches its result.
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        _tid = id(self)
        with _cache_lock:
            if _tid in _memo_cache:
                logger.debug(f"Returning memoized value from cache with ID {_tid}")
                return _memo_cache[_tid]
        logger.debug(f"Computing value on memo with ID {_tid} (cache miss)")
        value = fn(self, *args, **kwargs)
        with _cache_lock:
            _memo_cache[_tid] = value
        return value

    return wrapper


class Memo(Generic[T]):
    """Abstract base class for memoized values in benchmark runs."""

    # TODO: Make this better than the decorator application
    #  -> _Cached metaclass like in fsspec's AbstractFileSystem (maybe vendor with license)

    @cached_memo
    def __call__(self) -> T:
        """Placeholder to override when subclassing. The call should return the to be cached object."""
        raise NotImplementedError

    def __del__(self) -> None:
        """Delete the cached object and clear it from the cache."""
        with _cache_lock:
            sid = id(self)
            if sid in _memo_cache:
                logger.debug(f"Deleting cached value for memo with ID {sid}")
                del _memo_cache[sid]
