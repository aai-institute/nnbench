import os

import pytest

import nnbench
from nnbench.context import cpuarch, python_version, system


def test_runner_collection(testfolder: str) -> None:
    benchmarks = nnbench.collect(os.path.join(testfolder, "standard.py"), tags=("runner-collect",))
    assert len(benchmarks) == 1

    benchmarks = nnbench.collect(testfolder, tags=("non-existing-tag",))
    assert len(benchmarks) == 0

    benchmarks = nnbench.collect(testfolder, tags=("runner-collect",))
    assert len(benchmarks) == 1


def test_tag_selection(testfolder: str) -> None:
    PATH = os.path.join(testfolder, "tags.py")

    assert len(nnbench.collect(PATH)) == 3
    assert len(nnbench.collect(PATH, tags=("tag1",))) == 2
    assert len(nnbench.collect(PATH, tags=("tag2",))) == 1


def test_context_assembly(testfolder: str) -> None:
    context_providers = [system, cpuarch, python_version]
    benchmarks = nnbench.collect(testfolder, tags=("standard",))
    result = nnbench.run(
        benchmarks,
        params={"x": 1, "y": 1},
        context=context_providers,
    )

    ctx = result.context
    assert "system" in ctx
    assert "cpuarch" in ctx
    assert "python_version" in ctx


def test_error_on_duplicate_context_keys_in_runner(testfolder: str) -> None:
    def duplicate_context_provider() -> dict[str, str]:
        return {"system": "DuplicateSystem"}

    context_providers = [system, duplicate_context_provider]

    benchmarks = nnbench.collect(testfolder, tags=("standard",))
    with pytest.raises(ValueError, match="got multiple values for context key 'system'"):
        nnbench.run(
            benchmarks,
            params={"x": 1, "y": 1},
            context=context_providers,
        )


def test_filter_benchmarks_on_params(testfolder: str) -> None:
    @nnbench.benchmark
    def prod(a: int, b: int = 1) -> int:
        return a * b

    benchmarks = [prod]
    rec1 = nnbench.run(benchmarks, params={"a": 1, "b": 2})
    assert rec1.benchmarks[0]["parameters"] == {"a": 1, "b": 2}
    # Assert that the defaults are also present if not overridden.
    rec2 = nnbench.run(benchmarks, params={"a": 1})
    assert rec2.benchmarks[0]["parameters"] == {"a": 1, "b": 1}
