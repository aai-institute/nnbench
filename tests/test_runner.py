import os

import pytest

import nnbench
from nnbench.context import cpuarch, python_version, system


def test_runner_collection(testfolder: str) -> None:
    r = nnbench.BenchmarkRunner()

    r.collect(os.path.join(testfolder, "standard.py"), tags=("runner-collect",))
    assert len(r.benchmarks) == 1
    r.clear()

    r.collect(testfolder, tags=("non-existing-tag",))
    assert len(r.benchmarks) == 0
    r.clear()

    r.collect(testfolder, tags=("runner-collect",))
    assert len(r.benchmarks) == 1


def test_tag_selection(testfolder: str) -> None:
    PATH = os.path.join(testfolder, "tags.py")

    r = nnbench.BenchmarkRunner()

    r.collect(PATH)
    assert len(r.benchmarks) == 3
    r.clear()

    r.collect(PATH, tags=("tag1",))
    assert len(r.benchmarks) == 2
    r.clear()

    r.collect(PATH, tags=("tag2",))
    assert len(r.benchmarks) == 1
    r.clear()


def test_context_assembly(testfolder: str) -> None:
    r = nnbench.BenchmarkRunner()

    context_providers = [system, cpuarch, python_version]
    result = r.run(
        testfolder,
        tags=("standard",),
        params={"x": 1, "y": 1},
        context=context_providers,
    )

    ctx = result.context
    assert "system" in ctx
    assert "cpuarch" in ctx
    assert "python_version" in ctx


def test_error_on_duplicate_context_keys_in_runner(testfolder: str) -> None:
    r = nnbench.BenchmarkRunner()

    def duplicate_context_provider() -> dict[str, str]:
        return {"system": "DuplicateSystem"}

    context_providers = [system, duplicate_context_provider]

    with pytest.raises(ValueError, match="got multiple values for context key 'system'"):
        r.run(
            testfolder,
            tags=("standard",),
            params={"x": 1, "y": 1},
            context=context_providers,
        )


def test_filter_benchmarks_on_params(testfolder: str) -> None:
    @nnbench.benchmark
    def prod(a: int, b: int = 1) -> int:
        return a * b

    r = nnbench.BenchmarkRunner()
    r.benchmarks.append(prod)
    # TODO (nicholasjng): This is hacky
    rec1 = r.run("", params={"a": 1, "b": 2})
    assert rec1.benchmarks[0]["parameters"] == {"a": 1, "b": 2}
    # Assert that the defaults are also present if not overridden.
    rec2 = r.run("", params={"a": 1})
    assert rec2.benchmarks[0]["parameters"] == {"a": 1, "b": 1}
