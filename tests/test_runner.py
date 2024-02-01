import os

import pytest

from nnbench.context import cpuarch, python_version, system
from nnbench.runner import BenchmarkRunner


def test_runner_discovery(testfolder: str, another_testfolder: str) -> None:
    r = BenchmarkRunner()

    r.collect(os.path.join(testfolder, "standard_benchmarks.py"), tags=("runner-collect",))
    assert len(r.benchmarks) == 1
    r.clear()

    r.collect(testfolder, tags=("runner-collect",))
    assert len(r.benchmarks) == 1
    r.clear()

    r.collect(testfolder, tags=("non-existing-tag",))
    assert len(r.benchmarks) == 0
    r.clear()

    r.collect(os.path.join(testfolder), tags=("runner-collect",))
    r.collect(os.path.join(another_testfolder), tags=("runner-collect",))
    assert len(r.benchmarks) == 2


def test_tag_selection(testfolder: str) -> None:
    PATH = os.path.join(testfolder, "tag_selection_benchmark.py")

    r = BenchmarkRunner()

    r.collect(PATH, tags=())
    assert len(r.benchmarks) == 3
    r.clear()

    r.collect(PATH, tags=("tag1",))
    assert len(r.benchmarks) == 2
    r.clear()

    r.collect(PATH, tags=("tag2",))
    assert len(r.benchmarks) == 1
    r.clear()


def test_context_collection_in_runner(testfolder: str) -> None:
    r = BenchmarkRunner()

    context_providers = [system, cpuarch, python_version]
    result = r.run(
        os.path.join(testfolder),
        tags=("standard",),
        params={"x": 1, "y": 1},
        context=context_providers,
    )

    print(result)
    assert "system" in result["context"]
    assert "cpuarch" in result["context"]
    assert "python_version" in result["context"]


def test_error_on_duplicate_context_keys_in_runner(testfolder: str) -> None:
    r = BenchmarkRunner()

    def duplicate_context_provider() -> dict[str, str]:
        return {"system": "DuplicateSystem"}

    context_providers = [system, duplicate_context_provider]

    with pytest.raises(ValueError) as e:
        r.run(
            os.path.join(testfolder),
            tags=("standard",),
            params={"x": 1, "y": 1},
            context=context_providers,
        )

    assert "got multiple values for context key 'system'" in str(e.value)
