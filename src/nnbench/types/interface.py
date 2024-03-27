"""Type interface for the function interface"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

T = TypeVar("T")
Variable = tuple[str, type, Any]


@dataclass(frozen=True)
class Interface:
    """
    Data model representing a function's interface. An instance of this class
    is created using the `from_callable` class method.

    Parameters:
    ----------
    names : tuple[str, ...]
        Names of the function parameters.
    types : tuple[type, ...]
        Types of the function parameters.
    defaults : tuple
        A tuple of the function parameters' default values.
    variables : tuple[Variable, ...]
        A tuple of tuples, where each inner tuple contains the parameter name and type.
    returntype: type
        The function's return type annotation, or NoneType if left untyped.
    """

    names: tuple[str, ...]
    types: tuple[type, ...]
    defaults: tuple
    variables: tuple[Variable, ...]
    returntype: type

    @classmethod
    def from_callable(cls, fn: Callable, defaults: dict[str, Any]) -> Interface:
        """
        Creates an interface instance from the given callable.
        """
        # Set `follow_wrapped=False` to get the partially filled interfaces.
        # Otherwise we get missing value errors for parameters supplied in benchmark decorators.
        sig = inspect.signature(fn, follow_wrapped=False)
        ret = sig.return_annotation
        _defaults = {k: defaults.get(k, v.default) for k, v in sig.parameters.items()}
        # defaults are the signature parameters, then the partial parametrization.
        return cls(
            tuple(sig.parameters.keys()),
            tuple(p.annotation for p in sig.parameters.values()),
            tuple(_defaults.values()),
            tuple((k, v.annotation, _defaults[k]) for k, v in sig.parameters.items()),
            type(ret) if ret is None else ret,
        )
