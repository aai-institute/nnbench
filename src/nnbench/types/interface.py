"""Type interface for the function interface"""

import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

T = TypeVar("T")
Variable = tuple[str, type, Any]


@dataclass(frozen=True)
class Interface:
    """
    Data model representing a function's interface.

    An instance of this class is created using the ``Interface.from_callable()``
    class method.
    """

    funcname: str
    """Name of the function."""
    names: tuple[str, ...]
    """Names of the function parameters."""
    types: tuple[type, ...]
    """Type hints of the function parameters."""
    defaults: tuple
    """The function parameters' default values, or inspect.Parameter.empty if a parameter has no default."""
    variables: tuple[Variable, ...]
    """A tuple of tuples, where each inner tuple contains the parameter name, type, and default value."""
    returntype: type
    """The function's return type annotation, or NoneType if left untyped."""

    @classmethod
    def from_callable(cls, fn: Callable, defaults: dict[str, Any]) -> Self:
        """
        Creates an interface instance from the given callable.

        Wraps the information given by ``inspect.signature()``, with the option to
        supply a ``defaults`` map and overwrite any default set in the function's
        signature.
        """
        # Set `follow_wrapped=False` to get the partially filled interfaces.
        # Otherwise we get missing value errors for parameters supplied in benchmark decorators.
        sig = inspect.signature(fn, follow_wrapped=False)
        ret = sig.return_annotation
        _defaults = {k: defaults.get(k, v.default) for k, v in sig.parameters.items()}
        # defaults are the signature parameters, then the partial parametrization.
        return cls(
            fn.__name__,
            tuple(sig.parameters.keys()),
            tuple(p.annotation for p in sig.parameters.values()),
            tuple(_defaults.values()),
            tuple((k, v.annotation, _defaults[k]) for k, v in sig.parameters.items()),
            type(ret) if ret is None else ret,
        )
