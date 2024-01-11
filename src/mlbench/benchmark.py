"""The definition of mlbench's benchmark data model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Benchmark:
    """
    Data model representing a benchmark function. Subclass this to define your own custom benchmark class.
    """

    fn: Callable
    name: str
    params: dict[str, Any]
    setUp: Callable | None = None
    tearDown: Callable | None = None
