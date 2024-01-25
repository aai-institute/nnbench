import os

from nnbench.runner import AbstractBenchmarkRunner


def test_runner_discovery(testfolder: str) -> None:
    r = AbstractBenchmarkRunner()
    path = os.path.join(
        testfolder, "contains_single_file_for_collection_test", "simple_benchmark.py")

    r.collect(path)
    assert len(r.benchmarks) == 1

    r.clear()

    r.collect(path)
    assert len(r.benchmarks) == 1
