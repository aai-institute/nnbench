"""The definition of mlbench's benchmark data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Benchmark:
    """
    Data model representing a benchmark function. Subclass this to define your own custom benchmark class.
    """

    fn: Callable
    name: str | None = field(default=None)
    params: dict[str, Any] = field(repr=False, default_factory=dict)
    """Default parameters to attach to a benchmark. Must match the benchmark's interface."""
    setUp: Callable | None = field(repr=False, default=None)
    """A setup hook run before the benchmark. Must take all members of `params` as inputs."""
    tearDown: Callable | None = field(repr=False, default=None)
    """A teardown hook run after the benchmark. Must take all members of `params` as inputs."""
    tags: tuple[str, ...] = field(repr=False, default=())
    """Additional tags to attach for bookkeeping and selective filtering during runs."""

    def __post_init__(self):
        if not self.name:
            name = self.fn.__name__
            if self.params:
                name += "_" + "_".join(f"{k}={v}" for k, v in self.params.items())

            super().__setattr__("name", name)
        # TODO: Parse interface using `inspect`, attach to the class
